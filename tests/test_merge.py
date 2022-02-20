import torch
import numpy as np
from mpi4py import MPI

from parallel_pytorch.utils import tensor_merge
from parallel_pytorch.utils import abort_on_exception


@abort_on_exception
def test_1():
    shape = [2, 2]

    world = MPI.COMM_WORLD
    if world.Get_rank() == 0:
        x = [
            torch.tensor([[0, 1], [4, 5]]),
            torch.tensor([[2, 3], [6, 7]]),
            torch.tensor([[8, 9], [12, 13]]),
            torch.tensor([[10, 11], [14, 15]]),
        ]
        x = tensor_merge(tensors=x, shape=shape)
        e = torch.tensor([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
        ])
        assert torch.allclose(x, e), f'{x} != {e}'


@abort_on_exception
def test_2():
    shape = [1, 1]

    world = MPI.COMM_WORLD
    if world.Get_rank() == 0:
        x = [
            torch.tensor([[0, 1], [2, 3]])
        ]
        x = tensor_merge(tensors=x, shape=shape)
        e = torch.tensor([[0, 1], [2, 3]])
        assert torch.allclose(x, e), f'{x} != {e}'


def run_all():
    test_1()
    test_2()


if __name__ == '__main__':
    run_all()
