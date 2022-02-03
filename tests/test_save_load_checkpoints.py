# mpirun -np 8 python ./tests/test_save_load_checkpoints.py

import torch
import torch.nn as nn
import numpy as np
from mpi4py import MPI 

from parallel_pytorch.data import scatter_batch
from parallel_pytorch.layers import LinearDistributedInput, LinearDistributedOutput
from parallel_pytorch.ops import AllSumReduce, Broadcast
from parallel_pytorch.pipeline import Pipeline
from parallel_pytorch.topology import Topology
from parallel_pytorch.utils import global_rank, set_seed


def build_model(
    *,
    topo: Topology,
    n_input: int,
    n_hidden: int,
    n_blocks: int,
    device=None,
):
    size = topo.model_comm.Get_size()
    assert n_hidden % size == 0
    assert n_blocks == topo.get_num_pipeline_stages()
    stages = [
        nn.Sequential(
            Broadcast(topo.model_comm),
            LinearDistributedOutput(topo, n_input, n_hidden, device=device),
            nn.ReLU(),
            LinearDistributedInput(topo, n_hidden, n_input, device=device),
            AllSumReduce(topo.model_comm),
        )
        for _ in range(n_blocks)
    ]
    pipeline = Pipeline(topo=topo, stages=stages)
    return pipeline


def run_test():
    # all of these different parallel configurations should yield the same output
    parameterizations = [
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 1),
        (2, 1, 1),
        (2, 2, 2),
        (4, 2, 1),
        (1, 2, 4),
        (8, 1, 1),
        (1, 8, 1),
        (1, 1, 8),
    ]
    batch_size = 16
    vocab_size = 17
    n_input = 16
    n_hidden = 16
    block_size = 4
    seed = 42

    set_seed(seed * global_rank())
    first_output = None
    x = torch.randint(0, vocab_size, [batch_size, block_size], dtype=torch.long)
    y = torch.randint(0, vocab_size, [batch_size, block_size], dtype=torch.long)
    for dp, pp, mp in parameterizations:
        topo = Topology(dp=dp, pp=pp, mp=mp)
        if topo.active:
            pipeline = build_model(topo=topo, n_input=n_input, n_hidden=n_hidden, n_blocks=1)
            mb_x, mb_y = scatter_batch(topo=topo, inputs=x, labels=y)
            logits = pipeline(mb_x)
            if global_rank() == 0:
                if first_output is None:
                    first_output = logits.detach().cpu().numpy()
                else:
                    assert np.allclose(first_output, logits.detach().cpu().numpy())

        MPI.COMM_WORLD.Barrier()


if __name__ == '__main__':
    run_test()
