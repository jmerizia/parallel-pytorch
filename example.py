import random
from mpi4py import MPI
import torch.optim as optim
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from parallel_pytorch.models.minGPT import GPT, GPTConfig


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
torch.manual_seed(rank)
random.seed(rank)
np.random.seed(rank)

# a = torch.ones(4, 4)
# b = torch.ones(4, 4)
# print(torch.logical_or(a < 10, b > 10))
# quit()

config = GPTConfig(
    vocab_size=17,  # this are usually of weird length
    block_size=4,
    n_layer=2,
    n_head=4,
    n_embd=4,
)

class TrainConfig:
    batch_size = 64
    learning_rate = 0.1
    weight_decay = 0.1
    betas = (0.9, 0.95)

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

train_config = TrainConfig()

batch_size = 2
model = GPT(config, comm)
data = [
    (
        torch.randint(0, config.vocab_size, [batch_size, config.block_size], dtype=torch.long),
        torch.randint(0, config.vocab_size, [batch_size, config.block_size], dtype=torch.long),
    ) for _ in range(300)
]

criterion = nn.MSELoss()
optimizer = model.configure_optimizers(train_config)

running_loss = 0
for it, (x, y) in enumerate(data):
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    if it % 10 == 9 and rank == 0:
        print(f'iter {it} loss: {running_loss:.3f}')
        running_loss = 0.0
