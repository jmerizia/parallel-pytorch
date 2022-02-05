import torch
import numpy as np
from mpi4py import MPI

from parallel_pytorch.ops import tensor_split
from parallel_pytorch.utils import abort_on_exception


@abort_on_exception
def test_1():
    x_shape = [4, 4]
    worker_shape = [2, 2]

    world = MPI.COMM_WORLD
    num_workers = np.array(worker_shape).prod()
    comm = MPI.COMM_WORLD.Split(color=0 if world.Get_rank() < num_workers else 1, key=world.Get_rank())
    if world.Get_rank() < num_workers:
        volume = np.array(x_shape).prod()
        x = torch.arange(volume).view(x_shape)
        x = tensor_split(x, comm=comm, worker_shape=worker_shape)
        if comm.Get_rank() == 0:
            e = torch.tensor([[0, 1], [4, 5]])
        elif comm.Get_rank() == 1:
            e = torch.tensor([[2, 3], [6, 7]])
        elif comm.Get_rank() == 2:
            e = torch.tensor([[8, 9], [12, 13]])
        elif comm.Get_rank() == 3:
            e = torch.tensor([[10, 11], [14, 15]])
        assert torch.allclose(x, e), f'{x} != {e}'


@abort_on_exception
def test_2():
    x_shape = [2, 2]
    worker_shape = [1, 1]

    world = MPI.COMM_WORLD
    num_workers = np.array(worker_shape).prod()
    comm = MPI.COMM_WORLD.Split(color=0 if world.Get_rank() < num_workers else 1, key=world.Get_rank())
    if world.Get_rank() < num_workers:
        volume = np.array(x_shape).prod()
        x = torch.arange(volume).view(x_shape)
        x = tensor_split(x, comm=comm, worker_shape=worker_shape)
        e = torch.tensor([[0, 1], [2, 3]])
        assert torch.allclose(x, e), f'{x} != {e}'


def run_all():
    test_1()
    test_2()


if __name__ == '__main__':
    run_all()
