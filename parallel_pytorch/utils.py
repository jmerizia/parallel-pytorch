from typing import Literal
from mpi4py import MPI


def divide_optimally(n, parts):
    return [n // parts + (1 if i < n % parts else 0) for i in range(parts)]


def global_rank():
    return MPI.COMM_WORLD.Get_rank()


def cumsum(l):
    return [sum(l[:i+1]) for i in range(len(l))]
