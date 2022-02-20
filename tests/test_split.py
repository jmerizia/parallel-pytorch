import torch
import numpy as np
from mpi4py import MPI

from parallel_pytorch.utils import tensor_split
from parallel_pytorch.utils import abort_on_exception


@abort_on_exception
def test_1():
    shape = [2, 2]

    world = MPI.COMM_WORLD
    if world.Get_rank() == 0:
        x = torch.tensor([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23],
        ])
        x = tensor_split(tensor=x, shape=shape)
        assert type(x) == list, type(x)
        e = [
            torch.tensor([[ 0,  1], [ 4,  5], [ 8,  9]]),
            torch.tensor([[ 2,  3], [ 6,  7], [10, 11]]),
            torch.tensor([[12, 13], [16, 17], [20, 21]]),
            torch.tensor([[14, 15], [18, 19], [22, 23]]),
        ]
        assert len(x) == len(e)
        for a, b in zip(x, e):
            assert torch.allclose(a, b), f'{a} != {b}'


@abort_on_exception
def test_2():
    shape = [1, 1]

    world = MPI.COMM_WORLD
    if world.Get_rank() == 0:
        x = torch.tensor([[0, 1], [2, 3]])
        x = tensor_split(tensor=x, shape=shape)
        e = [
            torch.tensor([[0, 1], [2, 3]]),
        ]
        assert len(e) == len(x)
        assert torch.allclose(x[0], e[0]), f'{x[0]} != {e[0]}'


def run_all():
    test_1()
    test_2()


if __name__ == '__main__':
    run_all()
