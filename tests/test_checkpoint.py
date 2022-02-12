import torch
import torch.nn as nn
from mpi4py import MPI 
from pathlib import Path
import shutil
import os

from parallel_pytorch.ops import AllSumReduce, Broadcast, BroadcastFunc
from parallel_pytorch.pipeline import Pipeline
from parallel_pytorch.topology import Topology
from parallel_pytorch.utils import abort_on_exception, global_rank, prep_tensor_for_mpi_op, set_seed
from parallel_pytorch.checkpoint import save_checkpoint, load_checkpoint


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
        nn.Sequential(
            Broadcast(topo.model_comm),
            nn.Linear(n_input, n_hidden // size, device=device),
            nn.ReLU(),
            nn.Linear(n_hidden // size, n_input, bias=False, device=device),
            AllSumReduce(topo.model_comm),
        )
        for _ in range(n_blocks)
    ]
    param_worker_shapes = {
        **{f'{i}.1.weight' : i for i in range(n_blocks)},
        **{f'{i}.1.bias'   : i for i in range(n_blocks)},
        **{f'{i}.3.weight' : i for i in range(n_blocks)},
        **{f'{i}.3.bias'   : i for i in range(n_blocks)},
    }
    pipeline = Pipeline(topo=topo, layers=blocks, param_worker_shapes=param_worker_shapes)
    def _init_weights(module):
        with torch.no_grad():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
    pipeline.stage.apply(_init_weights)
    return pipeline


def assert_tensors_equal(f1: Path, f2: Path):
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
        # topo_A,    topo_B
        [(1, 1, 2), (1, 1, 2)],
        [(1, 2, 1), (1, 2, 1)],
        [(2, 1, 1), (2, 1, 1)],
        [(2, 2, 2), (2, 2, 2)],
        [(4, 2, 1), (4, 2, 1)],
        [(1, 2, 4), (1, 1, 1)],
        [(8, 1, 1), (2, 1, 1)],
        [(1, 8, 1), (4, 1, 2)],
        [(1, 1, 1), (8, 1, 1)],
    ]
    n_input = 16
    n_hidden = 16
    n_blocks = 10
    seed = 42

    set_seed(seed * global_rank())

    test_dir = Path('test_dir')
    checkpoint_dir = test_dir / 'checkpoint'

    def clear(directory):
        if global_rank() == 0:
            shutil.rmtree(directory, ignore_errors=True)
            Path(directory).mkdir(exist_ok=False, parents=True)
        MPI.COMM_WORLD.Barrier()

    try:

        for (dp_A, pp_A, mp_A), (dp_B, pp_B, mp_B) in parameterizations:

            # clear the directories
            clear(test_dir)

            # create the checkpoint for the model with topology A
            topo_A = Topology(dp=dp_A, pp=pp_A, mp=mp_A)
            if topo_A.active:
                pipeline_A = build_model(topo=topo_A, n_input=n_input, n_hidden=n_hidden, n_blocks=n_blocks)
                save_checkpoint(topo=topo_A, module=pipeline_A, directory=checkpoint_dir)
            MPI.COMM_WORLD.Barrier()

            # make sure the files are all there
            if global_rank() == 0:
                fn = checkpoint_dir / f'topology.json'
                assert fn.exists(), f'Expected to find file {fn}'
                for block_idx in range(n_blocks):
                    for shard_idx in range(topo_A.mp):
                        fn = checkpoint_dir / f'shard{shard_idx}' / f'{block_idx}_{1}_weight.pt'
                        assert fn.exists(), f'Expected to find file {fn}.'
                        fn = checkpoint_dir / f'shard{shard_idx}' / f'{block_idx}_{1}_bias.pt'
                        assert fn.exists(), f'Expected to find file {fn}.'
            MPI.COMM_WORLD.Barrier()

            # load checkpoint created in topology A to a new model with topology B
            topo_B = Topology(dp=dp_B, pp=pp_B, mp=mp_B)
            if topo_B.active:
                pipeline_B = build_model(topo=topo_B, n_input=n_input, n_hidden=n_hidden, n_blocks=n_blocks)
                load_checkpoint(topo=topo_B, module=pipeline_B, directory=checkpoint_dir)
            MPI.COMM_WORLD.Barrier()

            # try passing the same data into each pipeline. The outputs should be equivalent
            x = torch.ones([1, n_input])
            if topo_A.active:
                out_A = pipeline_A(x)
                # broadcast, so that we have it on global rank 0
                root = topo_A.get_pipeline_rank_of_last_stage()
                out_A = topo_A.pipeline_comm.bcast(out_A, root)
            MPI.COMM_WORLD.Barrier()
            if topo_B.active:
                out_B = pipeline_B(x)
                # broadcast, so that we have it on global rank 0
                root = topo_B.get_pipeline_rank_of_last_stage()
                out_B = topo_B.pipeline_comm.bcast(out_B, root)
            MPI.COMM_WORLD.Barrier()
            if global_rank() == 0:
                assert torch.allclose(out_A, out_B)

    finally:

        if global_rank() == 0:
            shutil.rmtree(test_dir, ignore_errors=True)


def run_all():
    test_1()


if __name__ == '__main__':
    run_all()
