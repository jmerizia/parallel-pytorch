"""
A simple pipeline scheduler that plays nicely with PyTorch.
"""

from collections import deque
import torch.nn as nn

from parallel_pytorch.utils import Topology, global_rank


class _Pipeline(nn.Module):

    @staticmethod
    def forward(ctx, micro_batches, topo: Topology, stage):
        ctx.topo = micro_batches
        num_stages = topo.get_num_pipeline_stages()
        assert num_stages == len(micro_batches)
        ctx.pasts = dict()
        Q = deque()
        stage_idx = topo.get_pipeline_stage_idx()
        if stage_idx == 0:
            for i in range(len(micro_batches)):
                Q.append(i)
        for it in range(2 * num_stages + 1):
            to_send = []
            if len(Q) > 0:
                v = Q.popleft()
                micro_batches[v] = stage(micro_batches[v])
                if v < num_stages - 1:
                    to_send.append((v, v + 1))
                ctx.pasts[v] = micro_batches[v]
            to_send = topo.pipeline_comm.allgather(to_send)

    @staticmethod
    def backward(ctx, topo: Topology, stage):
        pass

class Pipeline(nn.Module):

    def __init__(self, topo: Topology, stages):
        self.topo = topo
        assert len(stages) == topo.get_num_pipeline_stages()
        self.stage = stages[topo.get_pipeline_stage_idx()]

    def forward(self, x):
        return _Pipeline.apply(x, self.topo, self.stage)
