from mpi4py import MPI
import distdl
import torch.functional as F
from distdl.nn import Broadcast, SumReduce
from distdl.utilities.torch import zero_volume_tensor
from distdl.utilities.slicing import compute_subshape
import torch
import numpy as np
import time

from dataset import CharDataset
from dist_mingpt.model import GPT, GPTConfig
from dist_mingpt.model import Block
from dist_mingpt.utils import make_partition

block_size = 128
text = open('input.txt', 'r').read()
train_dataset = CharDataset(text, block_size)

D = 1
M = 8

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

assert D * M == size

P_input      = make_partition(shape=[D, 1, 1])
P_attn       = make_partition(shape=[D, 1, M])
P_mlp_hidden = make_partition(shape=[D, 1, M])

mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=8, n_head=8, n_embd=512,
                  D=D, M=M, P_input=P_input, P_attn=P_attn, P_mlp_hidden=P_mlp_hidden)

b = Block(mconf)
x = torch.zeros([64, 128, 512]) if P_input.active else zero_volume_tensor()  # torch.zeros([0, 0, 0])

MPI.COMM_WORLD.Barrier()
st = time.time()
y = b(x)
MPI.COMM_WORLD.Barrier()
en = time.time()
if rank == 0:
    print(en-st)

