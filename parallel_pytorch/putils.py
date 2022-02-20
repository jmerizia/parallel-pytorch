import torch
import numpy as np

from parallel_pytorch.ops import tensor_Gather, tensor_Scatter
from parallel_pytorch.utils import iter_cart_coords


def parallel_tensor_merge(x, comm, worker_shape, recvbuf=None, y_buf=None):
    """
    This takes a tensor x, which is sharded across the communicator with a
    certain worker grid shape, and merges it down to the root worker.
    All shards of x must have the same shape.
    Only rank 0 will return a torch tensor. All other ranks will return None.

    This is similar to `tensor_merge`, but works in parallel.
    """

    worker_shape = np.array(worker_shape)
    worker_volume = worker_shape.prod()
    assert worker_volume == comm.Get_size()
    x_shape = np.array(x.size())
    y_shape = x_shape * worker_shape
    x = tensor_Gather(x, comm, recvbuf=recvbuf)
    if comm.Get_rank() == 0:
        if y_buf is None:
            y = torch.empty(list(y_shape), dtype=x.dtype, device=x.device)
        else:
            assert list(y_buf.size()) == list(y_shape)
            y = y_buf
        # Now x is of shape [comm.Get_size(), ...], so we have to reshape it to shape y_shape
        for idx, coord in enumerate(iter_cart_coords(worker_shape, as_array=True)):
            a = coord * x_shape
            b = (coord + 1) * x_shape
            s = [slice(i, j) for i, j in zip(a, b)]
            y[s] = x[idx]
        return y
    else:
        return None


def parallel_tensor_split(x, comm, worker_shape, recvbuf=None, y_buf=None):
    """
    This takes a tensor, which lives just on the root worker, and shards it
    across all of the workers with the given worker grid shape.
    The tensor must be element-wise divisible by worker shape.

    This is similar to `tensor_split`, but works in parallel.
    """

    worker_shape = np.array(worker_shape)
    worker_volume = worker_shape.prod()
    assert worker_volume == comm.Get_size()
    x_shape = np.array(x.size())
    for a, b in zip(x_shape, worker_shape):
        assert a % b == 0, \
            "Tensor must be divisible by worker grid shape in all dimensions"
    y_shape = np.array([worker_volume] + list(x_shape // worker_shape))
    if y_buf is None:
        y = torch.empty(list(y_shape), dtype=x.dtype, device=x.device)
    else:
        assert list(y_buf.size()) == list(y_shape)
        y = y_buf
    for idx, coord in enumerate(iter_cart_coords(worker_shape, as_array=True)):
        a = coord * x_shape // worker_shape
        b = (coord + 1) * x_shape // worker_shape
        s = [slice(i, j) for i, j in zip(a, b)]
        y[idx] = x[s]
    y = tensor_Scatter(y, comm, recvbuf=recvbuf)
    return y
