import distdl
import numpy as np
from mpi4py import MPI


def make_partition(shape, ranks=None):
    if not ranks:
        volume = np.prod(shape)
        ranks = np.arange(volume)
    return distdl.backends.mpi.Partition(MPI.COMM_WORLD) \
        .create_partition_inclusive(ranks) \
        .create_cartesian_topology_partition(shape)

