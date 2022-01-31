import torch
import torch.nn as nn
from parallel_pytorch.ops import AllReduceFunc, ScatterFunc

from parallel_pytorch.topology import Topology


def aggregate_gradients(*, topo: Topology, model: nn.Module):
    """
    Aggregates the gradients of a model across all of its copies in the data parallel dimension.
    """

    for p in model.parameters():
        p.grad.data = AllReduceFunc.apply(p.grad.data, topo.per_stage_dp_comm)


def scatter_batch(*, topo: Topology, inputs: torch.Tensor, labels: torch.Tensor):
    """
    Scatters the input batch into mini-batches across all the processes.
    """

    inputs_shape = list(inputs.size())
    labels_shape = list(labels.size())
    assert inputs_shape[0] == labels_shape[0], \
        "Batch size of inputs doesn't match labels."
    batch_size = inputs_shape[0]
    assert batch_size % topo.get_num_data_parallel_copies() == 0, \
        "Batch size must be divisible by number of data parallel copies"
    dp = topo.get_num_data_parallel_copies()
    mini_inputs = inputs.view([dp, batch_size // dp] + inputs_shape[1:])
    mini_labels = labels.view([dp, batch_size // dp] + labels_shape[1:])
    mini_input = ScatterFunc.apply(mini_inputs, topo.per_stage_dp_comm)
    mini_label = ScatterFunc.apply(mini_labels, topo.per_stage_dp_comm)
    return mini_input, mini_label
