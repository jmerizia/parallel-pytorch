from collections import defaultdict
import torch
from typing import Literal
from mpi4py import MPI


def divide_optimally(n, parts):
    return [n // parts + (1 if i < n % parts else 0) for i in range(parts)]


def global_rank():
    return MPI.COMM_WORLD.Get_rank()


def compute_devices_per_node():
    comm = MPI.COMM_WORLD
    count = torch.cuda.device_count()
    counts = comm.allgather(count)
    assert len(set(counts)) == 1, "Some nodes have differing numbers of devices"
    return count


class Topology:
    """
    A tiny class that stores all the MPI communicators and rank relationships.
    """

    def __init__(self, dp: int, pp: int, mp: int, device: Literal['cpu', 'cuda'] = 'cpu'):
        self.dp = dp
        self.pp = pp
        self.mp = mp
        self.device = device
        if device == 'cuda':
            assert self.mp == compute_devices_per_node(), \
                "Topology.mp must be equal to the number of devices on each node, or else deadlocks can occur."
        world = MPI.COMM_WORLD
        assert world.Get_size() == self.dp * self.pp * self.mp
        self.data_comm = world
        data_rank = self.data_comm.Get_rank()
        self.pipeline_comm = self.data_comm.Split(color=data_rank // (self.pp * self.mp), key=data_rank)
        pipeline_rank = self.pipeline_comm.Get_rank()
        self.model_comm = self.pipeline_comm.Split(color=pipeline_rank // self.mp, key=pipeline_rank)

    def get_pipeline_stage_idx(self):
        return self.pipeline_comm.Get_size() // self.mp

    def get_num_pipeline_stages(self):
        return self.pipeline_comm.Get_size() // self.model_comm.Get_size()

    def get_pipeline_rank_of_next_stage(self):
        assert self.get_pipeline_stage_idx() + 1 < self.get_num_pipeline_stages()
        return self.pipeline_comm.Get_rank() + self.model_comm.Get_size()

    def get_pipeline_rank_of_prev_stage(self):
        assert 0 < self.get_pipeline_stage_idx()
        return self.pipeline_comm.Get_rank() - self.model_comm.Get_size()
        

def cumsum(l):
    return [sum(l[:i+1]) for i in range(len(l))]


def prep_tensor_for_mpi_op(t):
    t = t.detach()
    t = t.contiguous()
    if t.is_cuda:
        torch.cuda.current_stream().synchronize()
    return t
