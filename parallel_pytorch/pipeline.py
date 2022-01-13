"""
A simple pipeline scheduler that plays nicely with PyTorch.
"""

import torch
import torch.nn as nn
from mpi4py import MPI


class Pipeline(nn.Module):

    def __init__(self, stages, comm):
        self.comm = comm
        assert len(stages) == self.comm.Get_size()

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x
