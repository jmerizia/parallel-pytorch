from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from mpi4py import MPI


def empty_like_on_all_ranks(x, comm, root):
    if comm.Get_rank() == root:
        shape = list(x.shape)
        dtype = x.dtype
        device = x.device
    else:
        shape = None
        dtype = None
        device = None
    shape = comm.bcast(shape, root=root)
    dtype = comm.bcast(dtype, root=root)
    device = comm.bcast(device, root=root)
    return torch.empty(shape, dtype=dtype, device=device)


class _Broadcast(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, comm, root, x_buffer):
        ctx.comm = comm
        ctx.root = root
        ctx.x_buffer = x_buffer
        if comm.Get_rank() != ctx.root:
            x = x_buffer
        x = x.detach()
        ctx.comm.Bcast(x, root=ctx.root)
        return x

    @staticmethod
    def backward(ctx, grad):
        grad = grad.detach()
        ctx.comm.Reduce(sendbuf=grad, recvbuf=ctx.x_buffer, op=MPI.SUM, root=ctx.root)
        if ctx.comm.Get_rank() != ctx.root:
            out = torch.empty(0)
        else:
            out = ctx.x_buffer
        return out, None, None, None


class Broadcast(nn.Module):
    """ Broadcast a tensor to all workers """

    def __init__(self, comm, root):
        super().__init__()
        self.comm = comm
        self.root = root
        self.x_buffer = None

    def forward(self, x):
        # we need a tensor of shape x on all ranks for the forward pass
        if not self.x_buffer:
            self.x_buffer = empty_like_on_all_ranks(x, self.comm, self.root)
        x = _Broadcast.apply(x, self.comm, self.root, self.x_buffer)
        return x


class _SumReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, comm, root, x_buffer):
        ctx.comm = comm
        ctx.root = root
        ctx.x_buffer = x_buffer
        ctx.comm.Reduce(sendbuf=x.detach(), recvbuf=ctx.x_buffer, op=MPI.SUM, root=ctx.root)
        if ctx.comm.Get_rank() == ctx.root:
            out = ctx.x_buffer
        else:
            out = torch.empty(0)
        return out

    @staticmethod
    def backward(ctx, grad):
        if ctx.comm.Get_rank() != ctx.root:
            grad = ctx.x_buffer
        grad = grad.detach()
        ctx.comm.Bcast(grad, root=ctx.root)
        return grad, None, None, None


class SumReduce(nn.Module):
    """ Sum-reduce all tensors down to the root worker """

    def __init__(self, comm, root):
        super().__init__()
        self.comm = comm
        self.root = root
        self.x_buffer = None

    def forward(self, x):
        # we need replica buffers of the input for the backwards pass
        if not self.x_buffer:
            self.x_buffer = torch.empty_like(x)
        x = _SumReduce.apply(x, self.comm, self.root, self.x_buffer)
        return x


# class Scatter(nn.Module):
#     """ Scatter a tensor of shape (workers, ...) """

#     def __init__(self, comm, root):
#         super().__init__()
#         self.comm = comm
#         self.root = root
#         self.x_buffer = None
#         self.mega_buffer = None

#     def _setup_buffers(self, xs):
#         # we receive a mega-tensor on rank 0, and need to create regular tensors on all ranks
#         if not self.x_buffer:
#             x = xs[0] if self.comm.Get_rank() == self.root else None
#             self.x_buffer = empty_like_on_all_ranks(x, self.comm, self.root)
#         # for the backwards pass, we'll need a mega-tensor on the root
#         if self.comm.Get_rank() == self.root and not self.mega_buffer:
#             self.mega_buffer = torch.empty_like(xs)

#     def forward(self, xs):
#         self._setup_buffers(xs)
#         assert self.x_buffer is not None
#         self.comm.Scatter(sendbuf=xs, recvbuf=self.x_buffer, root=self.root)
#         return self.x_buffer.requires_grad_(True)

#     def backward(self, grads):
#         assert self.mega_buffer is not None
#         self.comm.Gather(sendbuf=grads, recvbuf=self.mega_buffer, root=self.root)
#         if self.comm.Get_rank() == self.root:
#             return self.mega_buffer.requires_grad_(True)
#         else:
#             return None


# class Gather(nn.Module):
#     """ Gather all tensors (of same shape) to shape (workers, ...) on the root rank """

#     def __init__(self, comm, root):
#         super().__init__()
#         self.comm = comm
#         self.root = root
#         self.x_buffer = None
#         self.mega_buffer = None

#     def _setup_buffers(self, x):
#         size = self.comm.Get_size()
#         # for forwards, we need a regular tensor on all ranks to return
#         if not self.x_buffer:
#             self.x_buffer = x.detach().clone()
#         # for backwards, we'll need a mega-tensor on the root
#         if self.comm.Get_rank() == self.root and not self.mega_buffer:
#                 self.mega_buffer = torch.zeros([size] + list(x.shape), dtype=x.dtype)

#     def forward(self, x):
#         self._setup_buffers(x)
#         assert self.mega_buffer is not None
#         self.comm.Gather(sendbuf=x, recvbuf=self.mega_buffer, root=self.root)
#         return self.mega_buffer.requires_grad_(True)

#     def backwards(self, x):
#         assert self.x_buffer is not None
#         self.comm.Scatter(sendbuf=x, recvbuf=self.x_buffer, root=self.root)
#         return self.x_buffer.requires_grad_(True)
