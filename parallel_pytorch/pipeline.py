"""
A simple pipeline scheduler that plays nicely with PyTorch.
"""

from typing import List
import torch.nn as nn
import torch

from parallel_pytorch.topology import Topology
from parallel_pytorch.utils import prep_tensor_for_mpi_op, split_list


def _pass_forward(topo: Topology, sendbuf, recvbuf, left_stage_idx, right_stage_idx):
    assert left_stage_idx < right_stage_idx
    if left_stage_idx + 1 == right_stage_idx:
        return recvbuf
    stage_idx = topo.get_pipeline_stage_idx()
    num_stages = topo.get_num_pipeline_stages()
    assert 0 <= left_stage_idx <= num_stages
    assert 0 <= right_stage_idx <= num_stages
    if left_stage_idx < stage_idx < right_stage_idx - 1:
        next_rank = topo.get_pipeline_rank_of_next_stage()
        prev_rank = topo.get_pipeline_rank_of_prev_stage()
        sendbuf = prep_tensor_for_mpi_op(sendbuf)
        recvbuf = prep_tensor_for_mpi_op(recvbuf)
        topo.pipeline_comm.Sendrecv(sendbuf=sendbuf, dest=next_rank, recvbuf=recvbuf, source=prev_rank)
        ret = recvbuf
    elif stage_idx == left_stage_idx:
        next_rank = topo.get_pipeline_rank_of_next_stage()
        sendbuf = prep_tensor_for_mpi_op(sendbuf)
        topo.pipeline_comm.Send(buf=sendbuf, dest=next_rank)
        ret = recvbuf
    elif stage_idx == right_stage_idx - 1:
        prev_rank = topo.get_pipeline_rank_of_next_stage()
        recvbuf = prep_tensor_for_mpi_op(recvbuf)
        topo.pipeline_comm.Recv(buf=recvbuf, source=prev_rank)
        ret = recvbuf
    else:
        ret = None
    return ret


def _pass_backward(topo: Topology, sendbuf, recvbuf, left_stage_idx, right_stage_idx):
    assert left_stage_idx < right_stage_idx
    if left_stage_idx + 1 == right_stage_idx:
        return recvbuf
    stage_idx = topo.get_pipeline_stage_idx()
    num_stages = topo.get_num_pipeline_stages()
    assert 0 <= left_stage_idx <= num_stages
    assert 0 <= right_stage_idx <= num_stages
    if left_stage_idx < stage_idx < right_stage_idx - 1:
        next_rank = topo.get_pipeline_rank_of_next_stage()
        prev_rank = topo.get_pipeline_rank_of_prev_stage()
        sendbuf = prep_tensor_for_mpi_op(sendbuf)
        recvbuf = prep_tensor_for_mpi_op(recvbuf)
        topo.pipeline_comm.Sendrecv(sendbuf=sendbuf, dest=prev_rank, recvbuf=recvbuf, source=next_rank)
        ret = recvbuf
    elif stage_idx == left_stage_idx:
        next_rank = topo.get_pipeline_rank_of_next_stage()
        recvbuf = prep_tensor_for_mpi_op(recvbuf)
        topo.pipeline_comm.Recv(buf=recvbuf, dest=next_rank)
        ret = recvbuf
    elif stage_idx == right_stage_idx - 1:
        prev_rank = topo.get_pipeline_rank_of_next_stage()
        sendbuf = prep_tensor_for_mpi_op(sendbuf)
        topo.pipeline_comm.Send(buf=sendbuf, source=prev_rank)
        ret = recvbuf
    else:
        ret = None
    return ret


def _pass_forward_pickle(topo: Topology, obj, left_stage_idx, right_stage_idx):
    assert left_stage_idx < right_stage_idx
    if left_stage_idx + 1 == right_stage_idx:
        return None
    stage_idx = topo.get_pipeline_stage_idx()
    num_stages = topo.get_num_pipeline_stages()
    assert 0 <= left_stage_idx <= num_stages
    assert 0 <= right_stage_idx <= num_stages
    if left_stage_idx < stage_idx < right_stage_idx - 1:
        next_rank = topo.get_pipeline_rank_of_next_stage()
        prev_rank = topo.get_pipeline_rank_of_prev_stage()
        left_neighbor_obj = topo.pipeline_comm.sendrecv(obj=obj, dest=next_rank, source=prev_rank)
    elif stage_idx == left_stage_idx:
        next_rank = topo.get_pipeline_rank_of_next_stage()
        topo.pipeline_comm.send(obj=obj, dest=next_rank)
        left_neighbor_obj = None
    elif stage_idx == right_stage_idx - 1:
        prev_rank = topo.get_pipeline_rank_of_prev_stage()
        left_neighbor_obj = topo.pipeline_comm.recv(source=prev_rank)
    else:
        left_neighbor_obj = None
    return left_neighbor_obj


def _pass_backward_pickle(topo: Topology, obj):
    stage_idx = topo.get_pipeline_stage_idx()
    num_stages = topo.get_num_pipeline_stages()
    if 0 < stage_idx < num_stages - 1:
        next_rank = topo.get_pipeline_rank_of_next_stage()
        prev_rank = topo.get_pipeline_rank_of_prev_stage()
        right_neighbor_obj = topo.pipeline_comm.sendrecv(obj=obj, dest=prev_rank, source=next_rank)
    elif stage_idx == 0:
        prev_rank = topo.get_pipeline_rank_of_prev_stage()
        topo.pipeline_comm.send(obj=obj, dest=prev_rank)
        right_neighbor_obj = None
    elif stage_idx == num_stages - 1:
        next_rank = topo.get_pipeline_rank_of_prev_stage()
        right_neighbor_obj = topo.pipeline_comm.recv(source=next_rank)
    else:
        assert False
    return right_neighbor_obj


class _Pipeline(nn.Module):

    @staticmethod
    def forward(ctx, micro_batches: List[torch.Tensor], topo: Topology, stage: nn.Module):
        inputs = []
        outputs = []
        buffer = None
        num_stages = topo.get_num_pipeline_stages()
        stage_idx = topo.get_pipeline_stage_idx()
        assert num_stages == len(micro_batches)
        for it in range(2 * num_stages - 1):
            if stage_idx <= it < stage_idx + num_stages:
                # input to this stage comes from the previous stage (i.e., the buffer) or the input
                if stage_idx == 0:
                    input = micro_batches[it].clone()
                else:
                    assert buffer is not None
                    input = buffer.clone()
            else:
                input = None
            if input is not None:
                # compute forward pass
                output = stage(input)
                # save the input/output for the backwards pass
                inputs.append(input)
                outputs.append(output)
                output_shape = output.shape
            else:
                output = None
                output_shape = None

            # compute the range of stages that need to participate in the communication
            left_stage_idx = max(0, it - num_stages + 1)
            right_stage_idx = min(num_stages, it + 2)

            # make necessary output buffers
            # Note: we can't make these all ahead of time because we won't know the output
            #       shape of a stage until the stage is called.
            recv_buffer_shape = _pass_forward_pickle(
                topo=topo,
                obj=output_shape,
                left_stage_idx=left_stage_idx,
                right_stage_idx=right_stage_idx,
            )
            if buffer is None and recv_buffer_shape is not None:
                buffer = torch.empty(recv_buffer_shape)

            buffer = _pass_forward(
                topo=topo,
                sendbuf=output,
                recvbuf=buffer,
                left_stage_idx=left_stage_idx,
                right_stage_idx=right_stage_idx
            )
            topo.pipeline_comm.Barrier()

        # broadcast the outputs in the last pipeline stage to all stages
        root = topo.get_pipeline_rank_of_last_stage()
        out = torch.stack(outputs)
        out = prep_tensor_for_mpi_op(out)
        topo.pipeline_comm.Bcast(buf=out, root=root)

        # save variables for backwards pass
        ctx.topo = topo
        ctx.inputs = inputs
        ctx.outputs = outputs
        ctx.buffer = buffer

        return out

    @staticmethod
    def backward(ctx, grads):
        micro_batches = ctx.micro_batches
        topo = ctx.topo
        inputs = ctx.inputs
        outputs = ctx.outputs
        buffer = ctx.buffer
        num_stages = topo.get_num_pipeline_stages()
        stage_idx = topo.get_pipeline_stage_idx()
        grad_outputs = []
        assert num_stages == len(micro_batches)
        for it in range(2 * num_stages - 1):
            if stage_idx == num_stages - 1:
                if it < num_stages:
                    grad = grads[it].clone().requires_grad_(True)
                else:
                    grad = None
            else:
                if num_stages - stage_idx - 1 <= it < 2 * num_stages - stage_idx - 1:
                    grad = buffer.clone().requires_grad_(True)
                else:
                    grad = None
            input = inputs[it]
            output = outputs[it]
            if grad:
                assert input is not None
                assert output is not None
                output.backward(grad)
                grad_output = input.grad
                buffer.copy_(grad_output)
            else:
                grad_output = None
            buffer = _pass_backward(topo=topo, buf=buffer)
            # save grad outputs
            grad_outputs.append(grad_output)
        root = topo.get_pipeline_rank_of_first_stage()
        out = torch.stack(grad_outputs[-num_stages:])
        out = prep_tensor_for_mpi_op(out)
        topo.pipeline_comm.Bcast(buf=out, root=root)
        return out


class Pipeline(nn.Module):
    """
    A simple pipeline parallelism implementation with the standard GPipe schedule.
    """

    def __init__(
        self,
        topo: Topology,
        layers: List[nn.Module],
    ):
        self.topo = topo
        stages = split_list(layers, topo.get_num_pipeline_stages())
        stages = [nn.Sequential(*stage) for stage in stages]
        assert len(stages) == topo.get_num_pipeline_stages()
        self.stage = stages[topo.get_pipeline_stage_idx()]

    def forward(self, x):
        shape = list(x.size())
        batch_size = shape[0]
        num_stages = self.topo.get_num_pipeline_stages()
        assert batch_size % num_stages == 0, \
            "Batch size must be divisible by the number of pipeline stages."
        x = x.view([num_stages, batch_size // num_stages] + shape[1:])
        return _Pipeline.apply(x, self.topo, self.stage)
