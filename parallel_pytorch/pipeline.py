"""
A simple pipeline scheduler that plays nicely with PyTorch.
"""

from typing import List
import torch.nn as nn
import torch

from parallel_pytorch.topology import Topology
from parallel_pytorch.utils import global_rank, prep_tensor_for_mpi_op, split_list


class Pipeline(nn.Module):
    """
    A pipeline module which implements the standard GPipe pipeline schedule.
    """

    def __init__(
        self,
        *,
        topo: Topology,
        layers: List[nn.Module],
    ):
        self.topo = topo
        self.inputs = []
        self.outputs = []
        self.buffer = None
        stages = split_list(layers, topo.get_num_pipeline_stages())
        stages = [nn.Sequential(*stage) for stage in stages]
        assert len(stages) == topo.get_num_pipeline_stages()
        self.stage = stages[topo.get_pipeline_stage_idx()]

    def __call__(self, batches):
        return self.forward(batches)

    def forward(self, batches: torch.Tensor):
        self.inputs = []
        self.outputs = []
        num_stages = self.topo.get_num_pipeline_stages()
        stage_idx = self.topo.get_pipeline_stage_idx()
        if self.topo.is_first_pipeline_stage():
            shape = list(batches.size())
            batch_size = shape[0]
            assert batch_size % num_stages == 0, \
                "Batch size must be divisible by the number of pipeline stages."
            micro_batches = batches.view([num_stages, batch_size // num_stages] + shape[1:])
            assert num_stages == len(micro_batches)
        for it in range(2 * num_stages - 1):
            if stage_idx <= it < stage_idx + num_stages:
                # input to this stage comes from the previous stage (i.e., the buffer) or the input
                if stage_idx == 0:
                    input = micro_batches[it].clone()
                else:
                    assert self.buffer is not None
                    input = self.buffer.clone()
            else:
                input = None
            if input is not None:
                # compute forward pass
                output = self.stage(input)
                # save the input/output for the backwards pass
                self.inputs.append(input)
                self.outputs.append(output)
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
            recv_buffer_shape = self._pass_forward_pickle(
                obj=output_shape,
                left_stage_idx=left_stage_idx,
                right_stage_idx=right_stage_idx,
            )
            if self.buffer is None and recv_buffer_shape is not None:
                self.buffer = torch.empty(recv_buffer_shape)
            self.buffer = self._pass_forward(
                sendbuf=output,
                recvbuf=self.buffer,
                left_stage_idx=left_stage_idx,
                right_stage_idx=right_stage_idx
            )
            self.topo.pipeline_comm.Barrier()
        # make outputs
        if self.topo.is_last_pipeline_stage():
            out = torch.cat(self.outputs, 0).requires_grad_(True)
        else:
            out = torch.zeros(1, requires_grad=True)
        out.register_hook(self._backward)
        return out

    def _backward(self, batches: torch.Tensor):
        num_stages = self.topo.get_num_pipeline_stages()
        stage_idx = self.topo.get_pipeline_stage_idx()
        grad_outputs = []
        if self.topo.is_last_pipeline_stage():
            shape = list(batches.size())
            batch_size = shape[0]
            micro_batches = batches.view([num_stages, batch_size // num_stages] + shape[1:])
            assert num_stages == len(micro_batches)
        for it in range(2 * num_stages - 1):
            if num_stages - stage_idx - 1 <= it < 2 * num_stages - stage_idx - 1:
                # this grad should come from either the input to backward or the buffer
                if stage_idx == num_stages - 1:
                    grad = micro_batches[it].clone()
                else:
                    grad = self.buffer.clone()
            else:
                grad = None
            if grad:
                input = self.inputs.pop(-1)
                output = self.outputs.pop(-1)
                output.backward(grad)
                grad_output = input.grad
                grad_outputs.append(grad_output)
                grad_output_shape = grad_output.shape
            else:
                grad_output = None
                grad_output_shape = None
            # compute the range of stages that need to participate in the communication
            left_stage_idx = max(num_stages - it - 2, 0)
            right_stage_idx = min(num_stages, 2 * num_stages - it - 1)
            # make necessary output buffers
            recv_buffer_shape = self._pass_backward_pickle(
                obj=grad_output_shape,
                left_stage_idx=left_stage_idx,
                right_stage_idx=right_stage_idx,
            )
            if buffer is None and recv_buffer_shape is not None:
                buffer = torch.empty(recv_buffer_shape)
            buffer = self._pass_backward(
                sendbuf=grad_output,
                recvbuf=buffer,
                left_stage_idx=left_stage_idx,
                right_stage_idx=right_stage_idx,
            )
        # make outputs
        if self.topo.is_first_pipeline_stage():
            out = torch.cat(grad_outputs, 0)
        else:
            out = torch.zeros(1)
        return out, None, None

    def _pass_forward(self, sendbuf, recvbuf, left_stage_idx, right_stage_idx):
        assert left_stage_idx < right_stage_idx
        if left_stage_idx + 1 == right_stage_idx:
            return recvbuf
        stage_idx = self.topo.get_pipeline_stage_idx()
        num_stages = self.topo.get_num_pipeline_stages()
        assert 0 <= left_stage_idx < num_stages
        assert 0 <= right_stage_idx <= num_stages
        if left_stage_idx < stage_idx < right_stage_idx - 1:
            next_rank = self.topo.get_pipeline_rank_of_next_stage()
            prev_rank = self.topo.get_pipeline_rank_of_prev_stage()
            sendbuf = prep_tensor_for_mpi_op(sendbuf)
            recvbuf = prep_tensor_for_mpi_op(recvbuf)
            self.topo.pipeline_comm.Sendrecv(sendbuf=sendbuf, dest=next_rank, recvbuf=recvbuf, source=prev_rank)
            ret = recvbuf
        elif stage_idx == left_stage_idx:
            next_rank = self.topo.get_pipeline_rank_of_next_stage()
            sendbuf = prep_tensor_for_mpi_op(sendbuf)
            self.topo.pipeline_comm.Send(buf=sendbuf, dest=next_rank)
            ret = recvbuf
        elif stage_idx == right_stage_idx - 1:
            prev_rank = self.topo.get_pipeline_rank_of_prev_stage()
            recvbuf = prep_tensor_for_mpi_op(recvbuf)
            self.topo.pipeline_comm.Recv(buf=recvbuf, source=prev_rank)
            ret = recvbuf
        else:
            ret = None
        return ret

    def _pass_backward(self, sendbuf, recvbuf, left_stage_idx, right_stage_idx):
        assert left_stage_idx < right_stage_idx
        if left_stage_idx + 1 == right_stage_idx:
            return recvbuf
        stage_idx = self.topo.get_pipeline_stage_idx()
        num_stages = self.topo.get_num_pipeline_stages()
        assert 0 <= left_stage_idx < num_stages
        assert 0 <= right_stage_idx <= num_stages
        if left_stage_idx < stage_idx < right_stage_idx - 1:
            next_rank = self.topo.get_pipeline_rank_of_next_stage()
            prev_rank = self.topo.get_pipeline_rank_of_prev_stage()
            sendbuf = prep_tensor_for_mpi_op(sendbuf)
            recvbuf = prep_tensor_for_mpi_op(recvbuf)
            self.topo.pipeline_comm.Sendrecv(sendbuf=sendbuf, dest=prev_rank, recvbuf=recvbuf, source=next_rank)
            ret = recvbuf
        elif stage_idx == left_stage_idx:
            next_rank = self.topo.get_pipeline_rank_of_next_stage()
            recvbuf = prep_tensor_for_mpi_op(recvbuf)
            self.topo.pipeline_comm.Recv(buf=recvbuf, dest=next_rank)
            ret = recvbuf
        elif stage_idx == right_stage_idx - 1:
            prev_rank = self.topo.get_pipeline_rank_of_next_stage()
            sendbuf = prep_tensor_for_mpi_op(sendbuf)
            self.topo.pipeline_comm.Send(buf=sendbuf, source=prev_rank)
            ret = recvbuf
        else:
            ret = None
        return ret

    def _pass_forward_pickle(self, obj, left_stage_idx, right_stage_idx):
        assert left_stage_idx < right_stage_idx
        if left_stage_idx + 1 == right_stage_idx:
            return None
        stage_idx = self.topo.get_pipeline_stage_idx()
        num_stages = self.topo.get_num_pipeline_stages()
        assert 0 <= left_stage_idx < num_stages
        assert 0 <= right_stage_idx <= num_stages
        if left_stage_idx < stage_idx < right_stage_idx - 1:
            next_rank = self.topo.get_pipeline_rank_of_next_stage()
            prev_rank = self.topo.get_pipeline_rank_of_prev_stage()
            left_neighbor_obj = self.topo.pipeline_comm.sendrecv(sendobj=obj, dest=next_rank, source=prev_rank)
        elif stage_idx == left_stage_idx:
            next_rank = self.topo.get_pipeline_rank_of_next_stage()
            self.topo.pipeline_comm.send(obj=obj, dest=next_rank)
            left_neighbor_obj = None
        elif stage_idx == right_stage_idx - 1:
            prev_rank = self.topo.get_pipeline_rank_of_prev_stage()
            left_neighbor_obj = self.topo.pipeline_comm.recv(source=prev_rank)
        else:
            left_neighbor_obj = None
        return left_neighbor_obj

    def _pass_backward_pickle(self, obj, left_stage_idx, right_stage_idx):
        assert left_stage_idx < right_stage_idx
        if left_stage_idx + 1 == right_stage_idx:
            return None
        stage_idx = self.topo.get_pipeline_stage_idx()
        num_stages = self.topo.get_num_pipeline_stages()
        assert 0 <= left_stage_idx < num_stages
        assert 0 <= right_stage_idx <= num_stages
        if left_stage_idx < stage_idx < num_stages - 1:
            next_rank = self.topo.get_pipeline_rank_of_next_stage()
            prev_rank = self.topo.get_pipeline_rank_of_prev_stage()
            right_neighbor_obj = self.topo.pipeline_comm.sendrecv(obj=obj, dest=prev_rank, source=next_rank)
        elif stage_idx == left_stage_idx:
            next_rank = self.topo.get_pipeline_rank_of_next_stage()
            right_neighbor_obj = self.topo.pipeline_comm.recv(source=next_rank)
        elif stage_idx == right_stage_idx - 1:
            prev_rank = self.topo.get_pipeline_rank_of_prev_stage()
            self.topo.pipeline_comm.send(obj=obj, dest=prev_rank)
            right_neighbor_obj = None
        else:
            right_neighbor_obj = None
        return right_neighbor_obj
