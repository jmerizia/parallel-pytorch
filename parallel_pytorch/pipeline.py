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
        rank = self.comm.Get_rank()
        self.stage = stages[rank]

    def forward(self, x):

        return x
