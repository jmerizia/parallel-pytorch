import torch
import torch.nn as nn

from parallel_pytorch.utils import cumsum, divide_optimally
from parallel_pytorch.ops import SumReduce


class DistributedEmbedding(nn.Module):
    """
    PyTorch's Embedding layer is a bit tricky to parallelize, so this does that.
    It uses the Megatron-LM style parallelization method by dividing over vocab space.
    """

    def __init__(self, comm, config):
        super().__init__()
        self.comm = comm
        self.config = config
        size = self.comm.Get_size()
        rank = self.comm.Get_rank()
        parts = divide_optimally(config.vocab_size, size)
        self.local_vocab_size = parts[rank]
        self.offset = ([0] + cumsum(parts))[rank]
        self.tok_emb = nn.Embedding(self.local_vocab_size + 1, config.n_embd, 0)
        self.sr = SumReduce(comm)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))

    def forward(self, idx):
        b, t = idx.size()
        # ensure all indices are within range (we lose the PyTorch error message)
        assert torch.all(0 <= idx) and torch.all(idx < self.config.vocab_size), "Token index is out of bounds"
        # translate the token indices
        idx -= self.offset
        # if the token is out of bounds of the current rank's embedding matrix, ignore it
        oob = torch.logical_or(idx < 1, idx > self.local_vocab_size)
        idx = torch.where(oob, torch.zeros_like(idx), idx)
        # apply the token embedding
        x = self.tok_emb(idx)
        x = self.sr(x)
        x += self.pos_emb[:, :t, :]
        return x
