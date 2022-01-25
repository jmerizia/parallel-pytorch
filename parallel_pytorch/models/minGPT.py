"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from parallel_pytorch.layers import DistributedEmbedding
from parallel_pytorch.ops import Broadcast, SumReduce

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    def __init__(
        self,
        topo,
        n_embd,
    ):
        super().__init__()
        self.topo = topo
        size = self.topo.comm_model.Get_size()
        D = n_embd
        assert (4 * D) % size == 0
        self.mlp = nn.Sequential(
            Broadcast(topo.comm_model),
            nn.Linear(D, 4 * D // size),
            nn.ReLU(),
            nn.Linear(4 * D // size, D),
            SumReduce(topo.comm_model),
        )

    def forward(self, x):
        return self.mlp(x)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(
        self,
        topo,
        block_size,
        n_embd,
        n_head,
        attn_pdrop,
        resid_pdrop,
    ):
        super().__init__()
        self.topo = topo
        size = topo.comm_model.Get_size()
        assert n_embd % size == 0
        assert n_head % size == 0
        assert (n_embd // size) % (n_head // size) == 0
        # key, query, value projections for all heads
        self.bc = Broadcast(topo.comm_model)
        self.key = nn.Linear(n_embd, n_embd // size)
        self.query = nn.Linear(n_embd, n_embd // size)
        self.value = nn.Linear(n_embd, n_embd // size)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd // size, n_embd)
        self.sr = SumReduce(topo.comm_model)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
        self.local_n_head = n_head // size

    def forward(self, x, layer_past=None):

        x = self.bc(x)

        B, T, C = x.size()

        C //= self.topo.comm_model.Get_size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.local_n_head, C // self.local_n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.local_n_head, C // self.local_n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.local_n_head, C // self.local_n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        y = self.sr(y)
        y = self.resid_drop(y)
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(
        self,
        topo,
        n_embd,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(topo)
        self.mlp = MLP(topo)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(
        self,
        topo,
        block_size,
        vocab_size,
        n_embd,
        embd_pdrop,
        n_layer,
    ):
        super().__init__()
        self.topo = topo

        # input embedding stem
        self.emb = DistributedEmbedding(topo, block_size=block_size)
        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(topo) for _ in range(n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        self.block_size = block_size
        self.apply(self._init_weights)

        local_param_count = sum(p.numel() for p in self.parameters())
        param_count = topo.comm_model.allreduce(local_param_count)
        if topo.comm_model.Get_rank() == 0:
            logger.info("number of parameters: %e", param_count)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('emb.pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        x = self.emb(idx)
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

