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
from dist_mingpt.utils import make_partition
import logging

vocab_size = 51200
block_size = 1024
batch_size = 1
n_head = 24
n_layer = 35
n_embd = 96 * n_head

# set up partitions and devices
D = 1
M = 4
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
assert D * M == size
P_world = distdl.backends.mpi.Partition(MPI.COMM_WORLD)
P_input_base = P_world.create_partition_inclusive(np.arange(D))
P_input = P_input_base.create_cartesian_topology_partition([D, 1, 1])
P_model_base = P_world.create_partition_inclusive(np.arange(D*M))
P_model = P_model_base.create_cartesian_topology_partition([D, 1, M])
P_input2d_base = P_world.create_partition_inclusive(np.arange(D))
P_input2d = P_input2d_base.create_cartesian_topology_partition([D, 1])
P_model2d_base = P_world.create_partition_inclusive(np.arange(D*M))
P_model2d = P_model2d_base.create_cartesian_topology_partition([D, M])

#P_input      = make_partition(shape=[D, 1, 1])
#P_model      = make_partition(shape=[D, 1, M])
#P_input2d    = make_partition(shape=[D, 1, 1])
#P_model2d    = make_partition(shape=[D, 1, M])
#device = torch.device('cpu')
device = torch.device('cuda:' + str(rank % 8))

# set up the model
config = GPTConfig(vocab_size, block_size,
                   n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                   D=D, M=M, P_input=P_input, P_model=P_model,
                   P_input2d=P_input2d, P_model2d=P_model2d)
model = GPT(config)
model = model.to(device)

# set up the fake input
assert batch_size % D == 0
if P_input.active:
    x = torch.zeros((batch_size // D, config.block_size), dtype=torch.long, device=device)
    #x = torch.zeros([batch_size // D, config.block_size, config.n_embd]) 
else:
    x = zero_volume_tensor(device=device)

# pass fake data through the network a few times
times = []
for idx in range(12):
    MPI.COMM_WORLD.Barrier()
    st = time.time()
    y = model(x)
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    en = time.time()
    times.append(en-st)
    if rank == 0:
        print(f'{idx+1}/{12}', flush=True)
if rank == 0:
    # scrap the first two to avoid the warmup cost
    times = times[2:]
    print(f'average {np.mean(times):0.4f} seconds', flush=True)
time.sleep(60)
