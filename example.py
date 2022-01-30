import torch
import torch.nn as nn
from torch.nn.utils import skip_init
import logging

from parallel_pytorch.models.minGPT import configure_optimizers, criterion, make_pipelined_GPT
from parallel_pytorch.ops import Broadcast
from parallel_pytorch.topology import Topology
from parallel_pytorch.utils import global_rank, set_seed

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG,
)

topo = Topology(dp=1, pp=2, mp=2)
set_seed(topo.data_comm.Get_rank())

# configs
batch_size = 8
vocab_size = 17
block_size = 4
n_layer = 2
n_head = 4
n_embd = 4
embd_pdrop = 0.1
attn_pdrop = 0.1
resid_pdrop = 0.1

# training configs
batch_size = 64
learning_rate = 0.1
weight_decay = 0.1
betas = (0.9, 0.95)

pipeline = make_pipelined_GPT(
    topo=topo,
    block_size=block_size,
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_layer=n_layer,
    n_head=n_head,
    embd_pdrop=embd_pdrop,
    attn_pdrop=attn_pdrop,
    resid_pdrop=resid_pdrop,
)
data = [
    (
        torch.randint(0, vocab_size, [batch_size, block_size], dtype=torch.long),
        torch.randint(0, vocab_size, [batch_size, block_size], dtype=torch.long),
    ) for _ in range(10)
]

optimizer = configure_optimizers(
    pipeline=pipeline,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    betas=betas,
)

running_loss = 0
for it, (x, y) in enumerate(data):
    with torch.set_grad_enabled(True):

        optimizer.zero_grad()

        # notice the forward semantics here are slightly different for pipelines
        logits = pipeline.forward(x)
        loss = criterion(topo, logits, y)
        loss.backward()

        # since Pipeline is just a normal class, we must call backwards manually
        pipeline.backward(logits.grad)
        optimizer.step()

        # we will only have the loss if we're the last pipeline stage
        if topo.is_last_pipeline_stage() and topo.model_comm.Get_rank() == 0:
            running_loss += loss.item()
            logger.info(f'batch {it} loss: {running_loss:.3f}')
            running_loss = 0.0
