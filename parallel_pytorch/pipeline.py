"""
A simple pipeline scheduler that plays nicely with PyTorch.
"""

from collections import deque
from typing import List
import torch.nn as nn
import torch

from parallel_pytorch.utils import Topology, prep_tensor_for_mpi_op


def _pass_forward(topo: Topology, sendbuf, recvbuf):
    sendbuf = prep_tensor_for_mpi_op(sendbuf)
    recvbuf = prep_tensor_for_mpi_op(recvbuf)
    stage_idx = topo.get_pipeline_stage_idx()
    num_stages = topo.get_num_pipeline_stages()
    if 0 < stage_idx < num_stages - 1:
        next_rank = topo.get_pipeline_rank_of_next_stage()
        prev_rank = topo.get_pipeline_rank_of_prev_stage()
        topo.pipeline_comm.Sendrecv(sendbuf=sendbuf, dest=next_rank, recvbuf=recvbuf, source=prev_rank)
    elif stage_idx == 0:
        next_rank = topo.get_pipeline_rank_of_next_stage()
        topo.pipeline_comm.Send(buf=sendbuf, dest=next_rank)
    elif stage_idx == num_stages - 1:
        prev_rank = topo.get_pipeline_rank_of_next_stage()
        topo.pipeline_comm.Recv(buf=recvbuf, source=prev_rank)
    else:
        assert False
    return recvbuf


def _pass_backward(topo: Topology, sendbuf, recvbuf):
    sendbuf = prep_tensor_for_mpi_op(sendbuf)
    recvbuf = prep_tensor_for_mpi_op(recvbuf)
    stage_idx = topo.get_pipeline_stage_idx()
    num_stages = topo.get_num_pipeline_stages()
    if 0 < stage_idx < num_stages - 1:
        next_rank = topo.get_pipeline_rank_of_next_stage()
        prev_rank = topo.get_pipeline_rank_of_prev_stage()
        topo.pipeline_comm.Sendrecv(sendbuf=sendbuf, dest=prev_rank, recvbuf=recvbuf, source=next_rank)
    elif stage_idx == 0:
        next_rank = topo.get_pipeline_rank_of_next_stage()
        topo.pipeline_comm.Recv(buf=recvbuf, dest=next_rank)
    elif stage_idx == num_stages - 1:
        prev_rank = topo.get_pipeline_rank_of_next_stage()
        topo.pipeline_comm.Send(buf=sendbuf, source=prev_rank)
    else:
        assert False
    return recvbuf


class _Pipeline(nn.Module):
    """
    A simple pipeline parallelism implementation with the standard GPipe schedule.

    For now, this assumes that the number of stages is equal to the number of micro batches,
    and each rank receives a List[Tensor] where inputs are copied on all ranks.
    """

    @staticmethod
    def forward(ctx, micro_batches: List[torch.Tensor], topo: Topology, stage: nn.Module):
        # We need to allocate space for the inputs and outputs, so we can use them in the backwards pass.
        # Later, we can use gradient checkpointing to reduce this memory footprint.
        inputs = []
        outputs = []
        # Also allocate space for passing around the outputs of each layer.
        buffer = micro_batches[0].clone()
        ctx.topo = topo
        ctx.inputs = inputs
        ctx.outputs = outputs
        ctx.buffer = buffer
        num_stages = topo.get_num_pipeline_stages()
        stage_idx = topo.get_pipeline_stage_idx()
        assert num_stages == len(micro_batches)
        bi = 0  # micro batch index (just for the first stage)
        for _ in range(2 * num_stages - 1):
            if stage_idx == 0:
                # input to this stage comes from the original micro batch
                input = micro_batches[bi].clone()
                bi += 1
            else:
                # input to this stage comes from the previous stage (i.e., the buffer)
                input = buffer.clone()
            # compute forward pass even for idle workers (the results will be discarded anyways)
            output = stage(input)
            buffer = _pass_forward(topo=topo, sendbuf=output, recvbuf=buffer)
            # save the input/output for the backwards pass
            inputs.append(input)
            outputs.append(output)

    @staticmethod
    def backward(ctx, grads):
        micro_batches = ctx.micro_batches
        topo = ctx.topo
        inputs = ctx.inputs
        outputs = ctx.outputs
        buffer = ctx.buffer
        num_stages = topo.get_num_pipeline_stages()
        stage_idx = topo.get_pipeline_stage_idx()
        assert num_stages == len(micro_batches)
        # TODO
        # bi = 0
        # for it in range(2 * num_stages - 1):
        #     if stage_idx == num_stages - 1:
        #         grad = grads[bi].clone()
        #         bi += 1
        #     else:
        #         grad = buffer.clone().requires_grad_(True)
        #     output.backward(grad)
        #     input.grad
        #     micro_batches[bi] = stage(micro_batches[bi])
        #     micro_batches[bi] = _pass_backward(
        #         sendbuf=micro_batches[bi],
        #         recvbuf=
        #         topo)
        #     bi = max(bi + 1, num_stages)


class Pipeline(nn.Module):

    def __init__(self, topo: Topology, stages):
        self.topo = topo
        assert len(stages) == topo.get_num_pipeline_stages()
        self.stage = stages[topo.get_pipeline_stage_idx()]

    def forward(self, x):
        return _Pipeline.apply(x, self.topo, self.stage)
