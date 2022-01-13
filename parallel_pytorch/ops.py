"""
Some simple wrappers around MPI communication functions
so that they play nicely with PyTorch.
"""

import torch
import torch.nn as nn
from mpi4py import MPI


class _Broadcast(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, comm):
        ctx.comm = comm
        x = x.detach()
        ctx.comm.Bcast(x)
        return x

    @staticmethod
    def backward(ctx, grad):
        ctx.comm.Allreduce(sendbuf=MPI.IN_PLACE, recvbuf=grad, op=MPI.SUM)
        return grad, None


class Broadcast(nn.Module):
    """ Broadcast a tensor to all workers """

    def __init__(self, comm):
        super().__init__()
        self.comm = comm

    def forward(self, x):
        return _Broadcast.apply(x, self.comm)


class _SumReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, comm):
        ctx.comm = comm
        x = x.detach()
        ctx.comm.Allreduce(sendbuf=MPI.IN_PLACE, recvbuf=x, op=MPI.SUM)
        return x

    @staticmethod
    def backward(ctx, grad):
        ctx.comm.Bcast(grad)
        return grad, None, None, None


class SumReduce(nn.Module):
    """ Sum-reduce all tensors down to the root worker """

    def __init__(self, comm):
        super().__init__()
        self.comm = comm

    def forward(self, x):
        return _SumReduce.apply(x, self.comm)


class _Scatter(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, comm):
        ctx.comm = comm
        x = x.detach()
        ctx.comm.Scatter(sendbuf=x, recvbuf=MPI.IN_PLACE, root=0)
        return x

    @staticmethod
    def backward(ctx, grad):
        ctx.comm.Gather(sendbuf=MPI.IN_PLACE, recvbuf=grad, root=0)
        return grad, None


class Scatter(nn.Module):
    """ Scatter a tensor of shape (workers, ...) """

    def __init__(self, comm):
        super().__init__()
        self.comm = comm

    def forward(self, x):
        assert x.shape[0] == self.comm.Get_size()
        return _Scatter.apply(x, self.comm)


class _Gather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, comm):
        ctx.comm = comm
        x = x.detach()
        ctx.comm.Gather(sendbuf=MPI.IN_PLACE, recvbuf=x, root=0)
        return x

    @staticmethod
    def backward(ctx, grad):
        ctx.comm.Scatter(sendbuf=grad, recvbuf=MPI.IN_PLACE, root=0)
        return grad, None


class Gather(nn.Module):
    """ Gather all tensors (of same shape) to shape (workers, ...)  on root """

    def __init__(self, comm):
        super().__init__()
        self.comm = comm

    def forward(self, x):
        return _Gather.apply(x, self.comm)
