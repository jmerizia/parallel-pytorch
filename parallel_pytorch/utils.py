"Non-parallel utilities"

from functools import wraps
import itertools
import torch
from typing import Any, List, Optional
from mpi4py import MPI
import random
import numpy as np
import traceback
from torch import Tensor


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


def split_list_weighted(elems: List[Any], weights: List[int], parts: int):
    """
    Splits `elems` into `parts` subarrays such that the maximum associated positive weight is minimized.
    """

    assert len(elems) == len(weights)
    assert parts > 0
    assert len(elems) > 0
    for w in weights:
        assert w >= 0

    def solve(k):
        ans = [[elems[0]]]
        t = weights[0]
        for e, w in zip(elems[1:], weights[1:]):
            if t > k:
                return None
            if t + w <= k:
                t += w
                ans[-1].append(e)
            else:
                t = w
                ans.append([e])
        if len(ans) <= parts:
            for _ in range(parts - len(ans)):
                found = False
                for i, l in enumerate(ans):
                    if len(l) > 1:
                        found = True
                        l1 = l[:1]
                        l2 = l[1:]
                        ans[i] = l1
                        ans.insert(i+1, l2)
                        break
                if not found:
                    return None
            return ans
        else:
            return None

    # binary search
    l, r = 1, sum(weights)
    for _ in range(30):
        mid = (l + r) // 2
        ans = solve(mid)
        if solve(mid) is not None:
            r = mid
        else:
            l = mid

    ans = solve(r)
    assert ans
    return ans


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


def tensor_merge(tensors: List[Tensor], shape, buf=None) -> Tensor:
    """
    Merge the given tensors as defined by shape.
    The "shape" describes
    """

    assert len(tensors) > 0
    assert len(tensors[0].shape) == len(shape), \
        f"Dimensionality of tensor ({len(tensors[0].shape)}) differs from dimensionality of shape ({len(shape)})"
    first_shape = tensors[0].shape
    first_device = tensors[0].device
    first_dtype = tensors[0].dtype
    for t in tensors:
        assert list(first_shape) == list(t.shape), \
            "Expected shapes of tensors to all be the same."
        assert first_device == t.device, \
            "Expected devices of tensors to all be the same."
        assert first_dtype == t.dtype, \
            "Expected dtypes of tensors to all be the same."
    assert np.prod(shape) == len(tensors)
    in_shape = np.array(tensors[0].shape)
    out_shape = in_shape * np.array(shape)
    if buf is None:
        buf = torch.zeros(tuple(out_shape), device=tensors[0].device, dtype=tensors[0].dtype)
    else:
        assert list(buf.shape) == list(out_shape)
    for idx, coord in enumerate(iter_cart_coords(shape, as_array=True)):
        a = coord * in_shape
        b = (coord + 1) * in_shape
        s = [slice(i, j) for i, j in zip(a, b)]
        buf[s] = tensors[idx]
    return buf


def tensor_split(tensor: Tensor, shape, bufs: Optional[List[Tensor]] = None) -> List[Tensor]:
    shape = np.array(shape)
    assert len(tensor.shape) == len(shape)
    in_shape = np.array(tensor.shape)
    out_shape = in_shape // np.array(shape)
    for a, b in zip(in_shape, shape):
        assert a % b == 0, \
            "Tensor must be divisible by worker grid shape in all dimensions."
    if bufs is None:
        bufs = [
            torch.zeros(tuple(out_shape), device=tensor.device, dtype=tensor.dtype)
            for _ in range(np.prod(shape))
        ]
    for buf in bufs:
        assert list(buf.shape) == list(out_shape), \
            "Expected buffer shape to be equal to input tensor shape divided by cartesian shape."
    for idx, coord in enumerate(iter_cart_coords(shape, as_array=True)):
        a = coord * in_shape // shape
        b = (coord + 1) * out_shape
        s = [slice(i, j) for i, j in zip(a, b)]
        bufs[idx].copy_(tensor[s])
    return bufs
