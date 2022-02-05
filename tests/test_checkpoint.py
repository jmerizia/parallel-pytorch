# mpirun -np 8 python ./tests/test_save_load_checkpoints.py

import torch
import torch.nn as nn
from mpi4py import MPI 
import tempfile
import os
from pathlib import Path
import shutil

from parallel_pytorch.layers import LinearDistributedInput, LinearDistributedOutput, ParallelSequential
from parallel_pytorch.ops import AllSumReduce, Broadcast
from parallel_pytorch.pipeline import Pipeline
from parallel_pytorch.topology import Topology
from parallel_pytorch.utils import abort_on_exception, global_rank, set_seed


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
    blocks = [
        ParallelSequential(
            Broadcast(topo.model_comm),
            LinearDistributedOutput(topo, n_input, n_hidden, device=device),
            nn.ReLU(),
            LinearDistributedInput(topo, n_hidden, n_input, device=device),
            AllSumReduce(topo.model_comm),
        )
        for _ in range(n_blocks)
    ]
    pipeline = Pipeline(topo=topo, layers=blocks)
    def _init_weights(module):
        with torch.no_grad():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                module.bias.data.zero_()
    pipeline.stage.apply(_init_weights)
    return pipeline


def assert_checkpoint_files_equal(f1: Path, f2: Path):
    state_dict1 = torch.load(f1)
    state_dict2 = torch.load(f2)
    for (name1, param1), (name2, param2) in zip(state_dict1.items(), state_dict2.items()):
        assert name1 == name2, f'{name1} != {name2}'
        assert param1.shape == param2.shape, f'{param1.shape} != {param2.shape}'
        assert param1.dtype == param2.dtype, f'{param1.dtype} != {param2.dtype}'
        tot = 0
        if not torch.allclose(param1, param2):
            for e1, e2 in zip(param1.flatten(), param2.flatten()):
                tot += abs(e1 - e2).item()
        # print(tot, flush=True)
        assert torch.allclose(param1, param2), f'values for {name1} do not match'


@abort_on_exception
def test_1():

    assert MPI.COMM_WORLD.Get_size() >= 8, \
        "Need more workers to run this test!"
    # all of these different parallel configurations should yield the same output
    parameterizations = [
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
    n_input = 16
    n_hidden = 16
    n_blocks = 10
    seed = 42

    set_seed(seed * global_rank())

    test_dir = Path('test_dir')
    checkpoint_dir_1 = test_dir / 'checkpoint_dir_1'
    checkpoint_dir_2 = test_dir / 'checkpoint_dir_2'

    try:

        if global_rank() == 0:
            shutil.rmtree(test_dir, ignore_errors=True)
            test_dir.mkdir(exist_ok=True, parents=True)

        MPI.COMM_WORLD.Barrier()

        topo = Topology(dp=1, pp=1, mp=1)
        if topo.active:
            reference_pipeline = build_model(
                topo=topo,
                n_input=n_input,
                n_hidden=n_hidden,
                n_blocks=n_blocks,
            )
            reference_pipeline.save_checkpoint(checkpoint_dir_1)

        MPI.COMM_WORLD.Barrier()

        # ensure all the files are present
        for layer_idx in range(n_blocks):
            fn = checkpoint_dir_1 / f'layer_{layer_idx}.pt'
            assert fn.exists(), f'Expected to find file {fn}'

        MPI.COMM_WORLD.Barrier()

        for dp, pp, mp in parameterizations:
            topo = Topology(dp=dp, pp=pp, mp=mp)
            if topo.active:
                # make new model with this topology
                pipeline = build_model(topo=topo, n_input=n_input, n_hidden=n_hidden, n_blocks=n_blocks)
                # load the checkpoint
                pipeline.load_checkpoint(checkpoint_dir_1)
                # save a new checkpoint
                pipeline.save_checkpoint(checkpoint_dir_2)

            MPI.COMM_WORLD.Barrier()

            # All topologies will contain the world rank 0
            if global_rank() == 0:
                for f1, f2 in zip(checkpoint_dir_1.glob('*'), checkpoint_dir_2.glob('*')):
                    assert_checkpoint_files_equal(f1, f2)

            MPI.COMM_WORLD.Barrier()

    finally:

        if global_rank() == 0:
            shutil.rmtree(test_dir, ignore_errors=True)


def run_all():
    test_1()


if __name__ == '__main__':
    run_all()
