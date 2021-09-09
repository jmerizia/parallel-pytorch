from mpi4py import MPI

import distdl
import torch.functional as F
from distdl.nn import Broadcast, SumReduce
from distdl.utilities.torch import zero_volume_tensor
from distdl.utilities.slicing import compute_subshape
import torch
import numpy as np

from dist_mingpt.model import GPT, GPTConfig
from dataset import CharDataset

#rank = MPI.COMM_WORLD.Get_rank()

block_size = 128
text = open('input.txt', 'r').read()
train_dataset = CharDataset(text, block_size)

#mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
#                  n_layer=8, n_head=8, n_embd=512)
#model = GPT(mconf)

# [batch, sequence, n_embed] -> [batch, sequence, n_head, n_embed // n_head]

# We might partition with
# [B, 1, N]
# for a total of B*N workers (B is data parallel)

from dist_mingpt.model import Block
from dist_mingpt.utils import make_partition

D = 2
M = 3

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

P_input  = make_partition(shape=[D, 1, 1])
P_attn       = make_partition(shape=[D, 1, M])
P_mlp_hidden = make_partition(shape=[D, 1, M])

mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=8, n_head=8, n_embd=512,
                  D=D, M=M, P_input=P_mlp_input, P_mlp_hidden=P_mlp_hidden)

b = Block(mconf)
x = torch.zeros([16, 13, 40]) if P_mlp_input.active else zero_volume_tensor()
y = b(x)
print()

quit()

def make_world_partition():
    return distdl.backends.mpi.Partition(MPI.COMM_WORLD)

def make_partition(P_world, shape, ranks=None):
    if not ranks:
        volume = np.prod(shape)
        ranks = np.arange(volume)
    return P_world \
        .create_partition_inclusive(ranks) \
        .create_cartesian_topology_partition(shape)

# get our rank and the total number of workers
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

# worker layout for data/model parallel dimensions
D = 2  # data parallel worker dimension
M = 3  # model parallel worker dimension

# helpful assertion about worker topology
assert D * M == size, f'This program expects {D*M} workers!'

# set up world and root partitions
P_world = make_world_partition()
P_input = make_partition(P_world, shape=[D, 1, 1])

class MLP(torch.nn.Module):

    def __init__(self, in_features):
        super().__init__()

        self.P_1 = make_partition(P_world, shape=[D, 1, 1])
        self.P_M = make_partition(P_world, shape=[D, 1, M])

        local_in_features = compute_subshape(M, self.P_M.index[-1], in_features)[0]
        local_hidden_features = compute_subshape(M, self.P_M.index[-1], in_features * 4)[0]

        self.bc = distdl.nn.Broadcast(self.P_1, self.P_M)
        self.sr = distdl.nn.SumReduce(self.P_M, self.P_1)

        self.ff1 = torch.nn.Linear(in_features, local_hidden_features)
        self.ff2 = torch.nn.Linear(local_hidden_features, in_features)
        self.gelu = torch.nn.GELU()

    def forward(self, input):
        x = self.bc(input)
        x = self.ff1(x)
        x = self.gelu(x)
        x = self.ff2(x)
        x = self.sr(x)
        return x


# set up the network
mlp = MLP(40)

# set up the input (each worker of the P_input partition should retrieve it's own data for efficiency)
x = torch.zeros([16, 13, 40]) if P_input.active else zero_volume_tensor()

y = mlp(x)
print(rank, y.shape)
quit()

# row parallel Ax
# [N, 1] x [1, N] -> [N, 1]

# column parallel Ax
# [1, N] x [N, 1] -> [1, 1]



ff1 = DistributedLinear(config.n_embd, 4 * config.n_embd)
gelu = nn.GELU()
ff2 = DistributedLinear(4 * config.n_embd, config.n_embd)
do = nn.Dropout(config.resid_pdrop)

