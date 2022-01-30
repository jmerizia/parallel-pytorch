from typing import Literal
from mpi4py import MPI

from parallel_pytorch.utils import compute_devices_per_node


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
        return self.pipeline_comm.Get_rank() // self.mp

    def get_num_pipeline_stages(self):
        return self.pipeline_comm.Get_size() // self.model_comm.Get_size()

    def get_pipeline_rank_of_next_stage(self):
        assert self.get_pipeline_stage_idx() + 1 < self.get_num_pipeline_stages()
        return self.pipeline_comm.Get_rank() + self.model_comm.Get_size()

    def get_pipeline_rank_of_prev_stage(self):
        assert 0 < self.get_pipeline_stage_idx()
        return self.pipeline_comm.Get_rank() - self.model_comm.Get_size()

    def get_pipeline_rank_of_last_stage(self):
        return self.pipeline_comm.Get_size() - 1

    def get_pipeline_rank_of_first_stage(self):
        return 0

    def is_first_pipeline_stage(self):
        return self.get_pipeline_stage_idx() == 0

    def is_last_pipeline_stage(self):
        return self.get_pipeline_stage_idx() == self.get_num_pipeline_stages() - 1
