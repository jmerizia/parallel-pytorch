import torch
import torch.nn as nn
from parallel_pytorch.models.minGPT import GPT
from parallel_pytorch.utils import Topology, global_rank, set_seed


topo = Topology(dp=1, pp=4, mp=1)
set_seed(topo.data_comm.Get_rank())

# configs
batch_size = 2
vocab_size = 17
block_size = 4
n_layer = 2
n_head = 4
n_embd = 4
embd_pdrop = 0.1

class TrainConfig:
    batch_size = 64
    learning_rate = 0.1
    weight_decay = 0.1
    betas = (0.9, 0.95)

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

train_config = TrainConfig()

model = GPT(
    topo=topo,
    block_size=block_size,
    vocab_size=vocab_size,
    n_embd=n_embd,
    embd_pdrop=embd_pdrop,
    n_layer=n_layer,
)
data = [
    (
        torch.randint(0, vocab_size, [batch_size, block_size], dtype=torch.long),
        torch.randint(0, vocab_size, [batch_size, block_size], dtype=torch.long),
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
    if it % 10 == 9 and global_rank() == 0:
        print(f'iter {it} loss: {running_loss:.3f}')
        running_loss = 0.0
