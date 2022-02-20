"""
A simple pipeline scheduler that plays nicely with PyTorch.
"""

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, OrderedDict as OrderedDictType
import torch
from torch.nn import Module, Sequential
from torch import Tensor
import os

from parallel_pytorch.topology import Topology
from parallel_pytorch.utils import cumsum, prep_tensor_for_mpi_op, split_list, split_list_weighted


class Pipeline(object):
    """
    An implementation of the standard GPipe pipeline schedule.
    """

    def __init__(
        self,
        *,
        topo: Topology,
        layers: List[Module],
        param_worker_shapes: Dict[str, List[int]],
    ):
        super().__init__()
        self.topo = topo
        param_counts = [sum(p.numel() for p in layer.parameters()) for layer in layers]
        stages = split_list_weighted(layers, param_counts, topo.get_num_pipeline_stages())
        stage_idx = topo.get_pipeline_stage_idx()
        self.layer_offset = cumsum([0] + [len(stage) for stage in stages])[stage_idx]
        stages = [Sequential(*stage) for stage in stages]
        assert len(stages) == topo.get_num_pipeline_stages()
        self.stage = stages[topo.get_pipeline_stage_idx()]
        for idx, stage in enumerate(stages):
            if idx != topo.get_pipeline_stage_idx():
                del stage
        self.param_worker_shapes = param_worker_shapes
        self.apply = self.stage.apply
        self.named_parameters = self.stage.named_parameters
        self.named_children = self.stage.named_children

    def __call__(self, batches: Tensor):
        return self.forward(batches)

    def state_dict(self, prefix='') -> OrderedDictType[str, Tensor]:
        """
        Retrieves the state dict for this worker's stage of the pipeline.
        """

        state_dict = OrderedDict()
        for name, param in self.stage.state_dict().items():
            idx = int(name.split('.')[0])
            new_name = str(self.layer_offset + idx) + '.' + '.'.join(name.split('.')[1:])
            state_dict[new_name] = param
        return state_dict

    def forward(self, batches: Tensor):
        self.inputs = []
        self.outputs = []
        recv_buffer = None
        num_stages = self.topo.get_num_pipeline_stages()
        stage_idx = self.topo.get_pipeline_stage_idx()
        if self.topo.is_first_pipeline_stage():
            shape = list(batches.size())
            batch_size = shape[0]
            assert batch_size % num_stages == 0, \
                "Batch size must be divisible by the number of pipeline stages."
            micro_batches = batches.view([num_stages, batch_size // num_stages] + shape[1:])
            assert num_stages == len(micro_batches)
        for it in range(2 * num_stages - 1):
            if stage_idx <= it < stage_idx + num_stages:
                # input to this stage comes from the previous stage (i.e., the buffer) or the input
                if stage_idx == 0:
                    input = micro_batches[it].clone().detach()
                else:
                    assert recv_buffer is not None
                    input = recv_buffer.clone().detach().requires_grad_(True)
            else:
                input = None
            if input is not None:
                # compute forward pass
                output = self.stage(input)
                # save the input/output for the backwards pass
                self.inputs.append(input)
                self.outputs.append(output)
                output_shape = output.shape
            else:
                output = None
                output_shape = None
            # compute the range of stages that need to participate in the communication
            left_stage_idx = max(0, it - num_stages + 1)
            right_stage_idx = min(num_stages, it + 2)
            # make necessary output buffers
            # Note: we can't make these all ahead of time because we won't know the output
            #       shape of a stage until the stage is called.
            recv_buffer_shape = self._pass_forward_pickle(
                obj=output_shape,
                left_stage_idx=left_stage_idx,
                right_stage_idx=right_stage_idx,
            )
            if recv_buffer is None and recv_buffer_shape is not None:
                recv_buffer = torch.empty(recv_buffer_shape)
            recv_buffer = self._pass_forward(
                sendbuf=output,
                recvbuf=recv_buffer,
                left_stage_idx=left_stage_idx,
                right_stage_idx=right_stage_idx
            )
            self.topo.pipeline_comm.Barrier()
        # make outputs
        if self.topo.is_last_pipeline_stage():
            out = torch.cat(self.outputs, 0).requires_grad_(True)
        else:
            out = torch.zeros(1, requires_grad=True)
        out = out.detach().requires_grad_(True)
        return out

    def backward(self, batches: torch.Tensor):
        num_stages = self.topo.get_num_pipeline_stages()
        stage_idx = self.topo.get_pipeline_stage_idx()
        results = []
        recv_buffer = None
        if self.topo.is_last_pipeline_stage():
            shape = list(batches.size())
            batch_size = shape[0]
            micro_batches = batches.view([num_stages, batch_size // num_stages] + shape[1:])
            assert num_stages == len(micro_batches)
        for it in range(2 * num_stages - 1):
            if num_stages - stage_idx - 1 <= it < 2 * num_stages - stage_idx - 1:
                # this grad should come from either the input to backward or the buffer
                if stage_idx == num_stages - 1:
                    grad_output = micro_batches[it].clone()
                else:
                    grad_output = recv_buffer.clone()
            else:
                grad_output = None
            if grad_output is not None:
                input = self.inputs.pop(-1)
                output = self.outputs.pop(-1)
                output.backward(grad_output)
                result = input.grad
                if result is None:
                    # replace with a dummy value here until I implement arbitrary inputs/outputs for pipeline stages
                    result = torch.zeros(1)
                results.append(result)
                result_shape = result.shape
            else:
                result = None
                result_shape = None
            # compute the range of stages that need to participate in the communication
            left_stage_idx = max(num_stages - it - 2, 0)
            right_stage_idx = min(num_stages, 2 * num_stages - it - 1)
            # make necessary output buffers
            recv_buffer_shape = self._pass_backward_pickle(
                obj=result_shape,
                left_stage_idx=left_stage_idx,
                right_stage_idx=right_stage_idx,
            )
            if recv_buffer is None and recv_buffer_shape is not None:
                recv_buffer = torch.empty(recv_buffer_shape)
            recv_buffer = self._pass_backward(
                sendbuf=result,
                recvbuf=recv_buffer,
                left_stage_idx=left_stage_idx,
                right_stage_idx=right_stage_idx,
            )
            self.topo.pipeline_comm.Barrier()
        # make outputs
        if self.topo.is_first_pipeline_stage():
            out = torch.cat(results, 0)
        else:
            out = torch.zeros(1)
        return out, None, None

    def _pass_forward(self, sendbuf, recvbuf, left_stage_idx, right_stage_idx):
        assert left_stage_idx < right_stage_idx
        if left_stage_idx + 1 == right_stage_idx:
            return recvbuf
        stage_idx = self.topo.get_pipeline_stage_idx()
        num_stages = self.topo.get_num_pipeline_stages()
        assert 0 <= left_stage_idx < num_stages
        assert 0 <= right_stage_idx <= num_stages
        if left_stage_idx < stage_idx < right_stage_idx - 1:
            next_rank = self.topo.get_pipeline_rank_of_next_stage()
            prev_rank = self.topo.get_pipeline_rank_of_prev_stage()
            sendbuf = prep_tensor_for_mpi_op(sendbuf)
            recvbuf = prep_tensor_for_mpi_op(recvbuf)
            self.topo.pipeline_comm.Sendrecv(sendbuf=sendbuf, dest=next_rank, recvbuf=recvbuf, source=prev_rank)
            ret = recvbuf
        elif stage_idx == left_stage_idx:
            next_rank = self.topo.get_pipeline_rank_of_next_stage()
            sendbuf = prep_tensor_for_mpi_op(sendbuf)
            self.topo.pipeline_comm.Send(buf=sendbuf, dest=next_rank)
            ret = recvbuf
        elif stage_idx == right_stage_idx - 1:
            prev_rank = self.topo.get_pipeline_rank_of_prev_stage()
            recvbuf = prep_tensor_for_mpi_op(recvbuf)
            self.topo.pipeline_comm.Recv(buf=recvbuf, source=prev_rank)
            ret = recvbuf
        else:
            ret = None
        return ret

    def _pass_backward(self, sendbuf, recvbuf, left_stage_idx, right_stage_idx):
        assert left_stage_idx < right_stage_idx
        if left_stage_idx + 1 == right_stage_idx:
            return recvbuf
        stage_idx = self.topo.get_pipeline_stage_idx()
        num_stages = self.topo.get_num_pipeline_stages()
        assert 0 <= left_stage_idx < num_stages
        assert 0 <= right_stage_idx <= num_stages
        if left_stage_idx < stage_idx < right_stage_idx - 1:
            next_rank = self.topo.get_pipeline_rank_of_next_stage()
            prev_rank = self.topo.get_pipeline_rank_of_prev_stage()
            sendbuf = prep_tensor_for_mpi_op(sendbuf)
            recvbuf = prep_tensor_for_mpi_op(recvbuf)
            self.topo.pipeline_comm.Sendrecv(sendbuf=sendbuf, dest=prev_rank, recvbuf=recvbuf, source=next_rank)
            ret = recvbuf
        elif stage_idx == left_stage_idx:
            next_rank = self.topo.get_pipeline_rank_of_next_stage()
            recvbuf = prep_tensor_for_mpi_op(recvbuf)
            self.topo.pipeline_comm.Recv(buf=recvbuf, source=next_rank)
            ret = recvbuf
        elif stage_idx == right_stage_idx - 1:
            prev_rank = self.topo.get_pipeline_rank_of_prev_stage()
            sendbuf = prep_tensor_for_mpi_op(sendbuf)
            self.topo.pipeline_comm.Send(buf=sendbuf, dest=prev_rank)
            ret = recvbuf
        else:
            ret = None
        return ret

    def _pass_forward_pickle(self, obj, left_stage_idx, right_stage_idx):
        assert left_stage_idx < right_stage_idx
        if left_stage_idx + 1 == right_stage_idx:
            return None
        stage_idx = self.topo.get_pipeline_stage_idx()
        num_stages = self.topo.get_num_pipeline_stages()
        assert 0 <= left_stage_idx < num_stages
        assert 0 <= right_stage_idx <= num_stages
        if left_stage_idx < stage_idx < right_stage_idx - 1:
            next_rank = self.topo.get_pipeline_rank_of_next_stage()
            prev_rank = self.topo.get_pipeline_rank_of_prev_stage()
            left_neighbor_obj = self.topo.pipeline_comm.sendrecv(sendobj=obj, dest=next_rank, source=prev_rank)
        elif stage_idx == left_stage_idx:
            next_rank = self.topo.get_pipeline_rank_of_next_stage()
            self.topo.pipeline_comm.send(obj=obj, dest=next_rank)
            left_neighbor_obj = None
        elif stage_idx == right_stage_idx - 1:
            prev_rank = self.topo.get_pipeline_rank_of_prev_stage()
            left_neighbor_obj = self.topo.pipeline_comm.recv(source=prev_rank)
        else:
            left_neighbor_obj = None
        return left_neighbor_obj

    def _pass_backward_pickle(self, obj, left_stage_idx, right_stage_idx):
        assert left_stage_idx < right_stage_idx
        if left_stage_idx + 1 == right_stage_idx:
            return None
        stage_idx = self.topo.get_pipeline_stage_idx()
        num_stages = self.topo.get_num_pipeline_stages()
        assert 0 <= left_stage_idx < num_stages
        assert 0 <= right_stage_idx <= num_stages
        if left_stage_idx < stage_idx < right_stage_idx - 1:
            next_rank = self.topo.get_pipeline_rank_of_next_stage()
            prev_rank = self.topo.get_pipeline_rank_of_prev_stage()
            right_neighbor_obj = self.topo.pipeline_comm.sendrecv(sendobj=obj, dest=prev_rank, source=next_rank)
        elif stage_idx == left_stage_idx:
            next_rank = self.topo.get_pipeline_rank_of_next_stage()
            right_neighbor_obj = self.topo.pipeline_comm.recv(source=next_rank)
        elif stage_idx == right_stage_idx - 1:
            prev_rank = self.topo.get_pipeline_rank_of_prev_stage()
            self.topo.pipeline_comm.send(obj=obj, dest=prev_rank)
            right_neighbor_obj = None
        else:
            right_neighbor_obj = None
        return right_neighbor_obj
