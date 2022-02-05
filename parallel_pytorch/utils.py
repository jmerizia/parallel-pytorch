from functools import wraps
import itertools
import torch
from typing import Any, List
from mpi4py import MPI
import random
import numpy as np
import traceback


def set_seed(seed):
    """ Set seeds in torch, numpy, and python random. """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def cumsum(l):
    return [sum(l[:i+1]) for i in range(len(l))]


def split_number(n: int, parts: int):
    """
    Returns a list of integers of length `parts` that sum to `n`
    where the positive difference of any two elements does not exceed 1.
    """
    return [n // parts + (1 if i < n % parts else 0) for i in range(parts)]


def split_list(v: List[Any], parts: int):
    """
    Divides a vector `v` into `parts` parts such that the positive
    difference of any two elements does not exceed 1.
    """
    sizes = split_number(len(v), parts)
    offsets = [0] + cumsum(sizes)
    return [v[o:o+s] for o, s in zip(offsets, sizes)]


def global_rank():
    return MPI.COMM_WORLD.Get_rank()


def compute_devices_per_node():
    comm = MPI.COMM_WORLD
    count = torch.cuda.device_count()
    counts = comm.allgather(count)
    assert len(set(counts)) == 1, "Some nodes have differing numbers of devices"
    return count


def prep_tensor_for_mpi_op(t):
    t = t.detach()
    t = t.contiguous()
    if t.is_cuda:
        torch.cuda.current_stream().synchronize()
    return t


def iter_cart_coords(shape, as_array=False):
    elems = itertools.product(*[range(shape[dim]) for dim in range(len(shape))])
    if not as_array:
        return list(elems)
    else:
        return [np.array(e) for e in elems]


def abort_on_exception(f):
    """
    If any MPI child process raises an error, this will abort all MPI processes,
    ensuring the program is killed, rather than deadlocks.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            print(traceback.format_exc(), flush=True)
            MPI.COMM_WORLD.Abort(1)
    return wrapper
