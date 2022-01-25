"""
A simple pipeline scheduler that plays nicely with PyTorch.
"""

from collections import deque
from typing import List
import torch.nn as nn
from mpi4py import MPI

from parallel_pytorch.utils import Topology, prep_tensor_for_mpi_op


def _pass_forward(buf, topo: Topology):
    buf = prep_tensor_for_mpi_op(buf)
    stage_idx = topo.get_pipeline_stage_idx()
    num_stages = topo.get_num_pipeline_stages()
    if 0 < stage_idx < num_stages - 1:
        next_rank = topo.get_pipeline_rank_of_next_stage()
        prev_rank = topo.get_pipeline_rank_of_prev_stage()
        topo.pipeline_comm.Sendrecv_replace(buf=buf, dest=next_rank, source=prev_rank)
    elif stage_idx == 0:
        next_rank = topo.get_pipeline_rank_of_next_stage()
        topo.pipeline_comm.Send(buf=buf, dest=next_rank)
    elif stage_idx == num_stages - 1:
        prev_rank = topo.get_pipeline_rank_of_next_stage()
        topo.pipeline_comm.Recv(buf=buf, source=prev_rank)
    else:
        assert False
    return buf


def _pass_backward(buf, topo: Topology):
    buf = prep_tensor_for_mpi_op(buf)
    stage_idx = topo.get_pipeline_stage_idx()
    num_stages = topo.get_num_pipeline_stages()
    if 0 < stage_idx < num_stages - 1:
        next_rank = topo.get_pipeline_rank_of_next_stage()
        prev_rank = topo.get_pipeline_rank_of_prev_stage()
        topo.pipeline_comm.Sendrecv_replace(buf=buf, dest=prev_rank, source=next_rank)
    elif stage_idx == 0:
        next_rank = topo.get_pipeline_rank_of_next_stage()
        topo.pipeline_comm.Recv(buf=buf, dest=next_rank)
    elif stage_idx == num_stages - 1:
        prev_rank = topo.get_pipeline_rank_of_next_stage()
        topo.pipeline_comm.Send(buf=buf, source=prev_rank)
    else:
        assert False
    return buf


class _Pipeline(nn.Module):
    """
    A simple pipeline parallelism implementation with the standard GPipe schedule.

    For now, this assumes that the number of stages is equal to the number of micro batches.
    """

    @staticmethod
    def forward(ctx, micro_batches, topo: Topology, stage):
        ctx.micro_batches = micro_batches
        ctx.topo = topo
        ctx.stage = stage
        num_stages = topo.get_num_pipeline_stages()
        stage_idx = topo.get_pipeline_stage_idx()
        assert num_stages == len(micro_batches)
        bi = stage_idx  # micro batch index
        for it in range(2 * num_stages - 1):
            micro_batches[bi] = stage(micro_batches[bi])
            micro_batches[bi] = _pass_forward(micro_batches[bi], topo)
            bi = max(bi + 1, num_stages)

    @staticmethod
    def backward(ctx, stage):
        pass

class Pipeline(nn.Module):

    def __init__(self, topo: Topology, stages):
        self.topo = topo
        assert len(stages) == topo.get_num_pipeline_stages()
        self.stage = stages[topo.get_pipeline_stage_idx()]

    def forward(self, x):
        return _Pipeline.apply(x, self.topo, self.stage)
