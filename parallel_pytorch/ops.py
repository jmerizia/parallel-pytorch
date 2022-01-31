"""
Some simple wrappers around MPI communication functions
so that they play nicely with PyTorch.
"""

import torch
import torch.nn as nn
from mpi4py import MPI

from parallel_pytorch.utils import global_rank, prep_tensor_for_mpi_op


class BroadcastFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, comm):
        ctx.comm = comm
        x = prep_tensor_for_mpi_op(x)
        ctx.comm.Bcast(x)
        return x

    @staticmethod
    def backward(ctx, grad):
        grad = prep_tensor_for_mpi_op(grad)
        ctx.comm.Allreduce(sendbuf=MPI.IN_PLACE, recvbuf=grad, op=MPI.SUM)
        return grad, None


class Broadcast(nn.Module):
    """ Broadcast a tensor to all workers """

    def __init__(self, comm):
        super().__init__()
        self.comm = comm

    def forward(self, x):
        return BroadcastFunc.apply(x, self.comm)


class AllReduceFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, comm):
        ctx.comm = comm
        x = prep_tensor_for_mpi_op(x)
        ctx.comm.Allreduce(sendbuf=MPI.IN_PLACE, recvbuf=x, op=MPI.SUM)
        return x

    @staticmethod
    def backward(ctx, grad):
        grad = prep_tensor_for_mpi_op(grad)
        ctx.comm.Bcast(grad)
        return grad, None, None, None


class AllReduce(nn.Module):
    """ Sum-reduce all tensors down to the root worker """

    def __init__(self, comm):
        super().__init__()
        self.comm = comm

    def forward(self, x):
        return AllReduceFunc.apply(x, self.comm)


class ScatterFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, comm):
        ctx.comm = comm
        x = prep_tensor_for_mpi_op(x)
        # TODO: use an allocator
        recvbuf = torch.empty(x.size()[1:], dtype=x.dtype)
        if comm.Get_rank() == 0:
            sendbuf = x
        else:
            sendbuf = None
        ctx.comm.Scatter(sendbuf=sendbuf, recvbuf=recvbuf, root=0)
        return recvbuf

    @staticmethod
    def backward(ctx, grad):
        grad = prep_tensor_for_mpi_op(grad)
        ctx.comm.Gather(sendbuf=MPI.IN_PLACE, recvbuf=grad, root=0)
        return grad, None


class Scatter(nn.Module):
    """ Scatter a tensor of shape (workers, ...) """

    def __init__(self, comm):
        super().__init__()
        self.comm = comm

    def forward(self, x):
        assert x.shape[0] == self.comm.Get_size()
        return ScatterFunc.apply(x, self.comm)


class GatherFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, comm):
        ctx.comm = comm
        x = prep_tensor_for_mpi_op(x)
        ctx.comm.Gather(sendbuf=MPI.IN_PLACE, recvbuf=x, root=0)
        return x

    @staticmethod
    def backward(ctx, grad):
        grad = prep_tensor_for_mpi_op(grad)
        ctx.comm.Scatter(sendbuf=grad, recvbuf=MPI.IN_PLACE, root=0)
        return grad, None


class Gather(nn.Module):
    """ Gather all tensors (of same shape) to shape (workers, ...) on root """

    def __init__(self, comm):
        super().__init__()
        self.comm = comm

    def forward(self, x):
        return GatherFunc.apply(x, self.comm)
