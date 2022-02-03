from collections import OrderedDict
import torch
import torch.nn as nn
from parallel_pytorch.module import ParallelModule
from parallel_pytorch.topology import Topology

from parallel_pytorch.utils import cumsum, split_number
from parallel_pytorch.ops import AllSumReduce, tensor_merge, tensor_split


class ParallelSequential(nn.Sequential, ParallelModule):
    """
    A replacement to torch.nn.Sequential when sub-modules will be ParallelModules.
    """

    pass


class LinearDistributedOutput(ParallelModule):
    """
    This is a distributed linear layer which takes a non-sharded input
    and produces a sharded output.
    """

    def __init__(
        self,
        topo: Topology,
        in_size: int,
        out_size: int,
        bias=True,
        device=None,
    ):
        super().__init__()
        self.topo = topo
        size = topo.model_comm.Get_size()
        assert out_size % size == 0, \
            "Expected output size to be divisible by the number of model parallel workers."
        self.ff = nn.Linear(in_size, out_size // size, bias=bias, device=device)
        self.bias = bias

    def forward(self, x):
        return self.ff(x)

    def parallel_state_dict(self, prefix=''):
        d = OrderedDict()
        comm = self.topo.model_comm
        size = comm.Get_size()
        d[prefix + 'weight'] = tensor_merge(x=self.ff.weight.data, comm=comm, worker_shape=[size, 1])
        if self.bias:
            d[prefix + 'bias'] = tensor_merge(x=self.ff.bias.data, comm=comm, worker_shape=[size])
        return d

    def parallel_load_state_dict(self, state_dict, prefix=''):
        comm = self.topo.model_comm
        size = comm.Get_size()
        self.ff.weight.data = tensor_split(x=state_dict[prefix + 'weight'], comm=comm, worker_shape=[size, 1])
        if self.bias:
            self.ff.bias.data = tensor_split(x=state_dict[prefix + 'bias'], comm=comm, worker_shape=[size])


class LinearDistributedInput(ParallelModule):
    """
    This is a distributed linear layer which takes a sharded input
    and produces a non-sharded output.
    """

    def __init__(
        self,
        topo: Topology,
        in_size: int,
        out_size: int,
        bias=True,
        device=None,
    ):
        super().__init__()
        self.topo = topo
        size = topo.model_comm.Get_size()
        assert in_size % size == 0, \
            "Expected input size to be divisible by the number of model parallel workers."
        self.ff = nn.Linear(in_size // size, out_size, bias=bias, device=device)
        self.bias = bias

    def forward(self, x):
        return self.ff(x)

    def parallel_state_dict(self, prefix=''):
        d = OrderedDict()
        comm = self.topo.model_comm
        size = comm.Get_size()
        d[prefix + 'weight'] = tensor_merge(x=self.ff.weight.data, comm=comm, worker_shape=[1, size])
        if self.bias:
            d[prefix + 'bias'] = self.ff.bias.data
        return d

    def parallel_load_state_dict(self, state_dict, prefix=''):
        comm = self.topo.model_comm
        size = comm.Get_size()
        self.ff.weight.data = tensor_split(x=state_dict[prefix + 'weight'], comm=comm, worker_shape=[1, size])
        if self.bias:
            self.ff.bias.data = state_dict[prefix + 'bias']


class DistributedEmbedding(ParallelModule):
    """
    PyTorch's Embedding layer is a bit tricky to parallelize, so this does that.
    It uses the Megatron-LM style parallelization method by dividing over vocab space.
    """

    def __init__(
        self,
        topo: Topology,
        block_size: int,
        vocab_size: int,
        n_embd: int,
        device=None,
    ):
        super().__init__()
        self.topo = topo
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        size = topo.model_comm.Get_size()
        rank = topo.model_comm.Get_rank()
        parts = split_number(vocab_size, size)
        self.local_vocab_size = parts[rank]
        self.offset = ([0] + cumsum(parts))[rank]
        self.tok_emb = nn.Embedding(self.local_vocab_size + 1, n_embd, 0, device=device)
        self.sr = AllSumReduce(topo.model_comm)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd, device=device))

    def forward(self, idx):
        b, t = idx.size()
        # ensure all indices are within range (we lose the PyTorch error message)
        assert torch.all(0 <= idx) and torch.all(idx < self.vocab_size), "Token index is out of bounds"
        # translate the token indices
        idx -= self.offset
        # if the token is out of bounds of the current rank's embedding matrix, ignore it
        oob = torch.logical_or(idx < 1, idx > self.local_vocab_size)
        idx = torch.where(oob, torch.zeros_like(idx), idx)
        # apply the token embedding
        x = self.tok_emb(idx)
        x = self.sr(x)
        x += self.pos_emb[:, :t, :]
        self.state_dict()
        return x

    def parallel_state_dict(self, prefix=''):
        d = OrderedDict()
        comm = self.topo.model_comm
        size = comm.Get_size()
        d[prefix + 'tok_emb.weight'] = tensor_merge(x=self.tok_emb.weight.data, comm=comm, worker_shape=[1, size])
        d[prefix + 'pos_emb'] = self.pos_emb.data
        return d

    def parallel_load_state_dict(self, state_dict, prefix=''):
        comm = self.topo.model_comm
        size = comm.Get_size()
        self.tok_emb.weight.data = tensor_split(x=state_dict[prefix + 'tok_emb.weight'], comm=comm, worker_shape=[1, size])
        self.pos_emb.data = state_dict[prefix + 'pos_emb']
