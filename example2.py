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
batch_size = 64

# set up partitions and devices
D = 1
M = 1
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
assert D * M == size
P_input      = make_partition(shape=[D, 1, 1])
P_attn       = make_partition(shape=[D, 1, M])
P_mlp_hidden = make_partition(shape=[D, 1, M])
device = torch.device('cpu')
#device = torch.device('cuda:0')

# set up the model
config = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                   n_layer=12, n_head=12, n_embd=768,
                   D=D, M=M, P_input=P_input, P_attn=P_attn, P_mlp_hidden=P_mlp_hidden)

model = GPT(config)
model = model.to(device)

assert batch_size % D == 0
if P_input.active:
    x = torch.zeros([batch_size // D, config.block_size, config.n_embd]) 
else:
    zero_volume_tensor()
x = x.to(device)

MPI.COMM_WORLD.Barrier()
st = time.time()
y = model(x)
torch.cuda.synchronize()
MPI.COMM_WORLD.Barrier()
en = time.time()
if rank == 0:
    print(f'done in {en-st:0.4f} seconds')

