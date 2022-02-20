"""
Some simple wrappers around MPI communication functions
so that they play nicely with PyTorch.
"""

import torch
import torch.nn as nn
from mpi4py import MPI
import numpy as np

from parallel_pytorch.utils import iter_cart_coords, prep_tensor_for_mpi_op


# functions for interoperability with torch tensors

def tensor_Bcast(x, comm):
    """
    A wrapper around MPI broadcast that assumes that all participating ranks
    have the same shape torch tensor.
    """

    x = prep_tensor_for_mpi_op(x)
    comm.Bcast(x)
    return x


def tensor_AllSumReduce(x, comm):
    """
    A wrapper round MPI all sum redice that assumes that all participating ranks
    have the same shape torch tensor.
    """

    x = prep_tensor_for_mpi_op(x)
    comm.Allreduce(sendbuf=MPI.IN_PLACE, recvbuf=x, op=MPI.SUM)
    return x


def tensor_Scatter(x, comm, recvbuf=None):
    """
    A wrapper around MPI scatter that assumes that the 0th rank has a tensor
    of shape [comm.Get_size(), ...].
    """

    x = prep_tensor_for_mpi_op(x)
    shape = list(x.size()[1:])
    if recvbuf is not None:
        assert list(recvbuf.size()) == shape
    else:
        recvbuf = torch.empty(shape, dtype=x.dtype, device=x.device)
    if comm.Get_rank() == 0:
        sendbuf = x
    else:
        sendbuf = None
    comm.Scatter(sendbuf=sendbuf, recvbuf=recvbuf, root=0)
    return recvbuf


def tensor_Gather(x, comm, recvbuf=None):
    """
    A wrapper around MPI gather that assumes that all participating ranks have
    the same shape tensor.
    """

    x = prep_tensor_for_mpi_op(x)
    size = comm.Get_size()
    shape = [size] + list(x.size())
    if comm.Get_rank() == 0:
        if recvbuf is not None:
            assert list(recvbuf.size()) == shape
        else:
            recvbuf = torch.empty(shape, dtype=x.dtype, device=x.device)
    comm.Gather(sendbuf=x, recvbuf=recvbuf, root=0)
    return recvbuf


class BroadcastFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, comm):
        ctx.comm = comm
        return tensor_Bcast(x, comm)

    @staticmethod
    def backward(ctx, grad):
        return tensor_AllSumReduce(grad, ctx.comm), None


class Broadcast(nn.Module):
    """ Broadcast a tensor to all workers """

    def __init__(self, comm):
        super().__init__()
        self.comm = comm

    def forward(self, x):
        return BroadcastFunc.apply(x, self.comm)


class AllSumReduceFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, comm):
        ctx.comm = comm
        return tensor_AllSumReduce(x, comm)

    @staticmethod
    def backward(ctx, grad):
        return tensor_Bcast(grad, ctx.comm), None


class AllSumReduce(nn.Module):
    """ Sum-reduce all tensors down to the root worker """

    def __init__(self, comm):
        super().__init__()
        self.comm = comm

    def forward(self, x):
        return AllSumReduceFunc.apply(x, self.comm)


class ScatterFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, comm):
        ctx.comm = comm
        return tensor_Scatter(x, comm)

    @staticmethod
    def backward(ctx, grad):
        return tensor_Gather(grad, ctx.comm), None


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
        return tensor_Gather(x, comm)

    @staticmethod
    def backward(ctx, grad):
        return tensor_Scatter(grad, ctx.comm), None


class Gather(nn.Module):
    """ Gather all tensors (of same shape) to shape (workers, ...) on root """

    def __init__(self, comm):
        super().__init__()
        self.comm = comm

    def forward(self, x):
        return GatherFunc.apply(x, self.comm)
