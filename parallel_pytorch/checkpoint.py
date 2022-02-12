"""
This file exposes functions `save_checkpoint` and `load_checkpoint` which works with any Module or Pipeline,
and is invariant to network topology.
It assumes that the filesystem is shared among all workers.
Only the root data parallel replica will participate in saving checkpoints.

The file format on disk of a checkpoint is a directory of files, where each file corresponds to a single parameter
in the network and 
There is also a "topology.json" file which describes the topology that was used when a checkpoint was created.

For example, a parallel MLP with pp = 2 and mp = 2 will produce the following checkpoint directory:

<checkpoint name>/
    shard_0/
        ff1_weight.pt
        ff1_bias.pt
        ff2_weight.pt
        ff2_bias.pt
    shard_1/
        ff1_weight.pt
        ff1_bias.pt
        ff2_weight.pt
        ff2_bias.pt
    topology.json

Each ".pt" file contains a serialized Tensor (as opposed to the typical serialized OrderedDict).
The "topology.json" file would contain the following JSON structure (for the above example):

{
    'dp': <num replicas>,
    'pp': 2,
    'mp': 2,
    'param_worker_shapes': {
        'ff1_weight': [1, 2],
        'ff1_bias': [2],
        'ff2_weight': [2, 1],
        'ff2_bias': None,
    }
}

Some notes:
- The reason we use many small files as opposed to a large file with a serialized DefaultDict (as torch normally does)
  is to allow for loading with less CPU memory (which may be important for certain large models).
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, OrderedDict as OrderedDictType
from collections import OrderedDict
import torch
from torch import Tensor
from torch.nn import Module
import os
from parallel_pytorch.ops import tensor_merge
import json
from parallel_pytorch.pipeline import Pipeline

from parallel_pytorch.topology import Topology


def _get_checkpoint_file_fn(directory: str, shard_idx: int, param_name: str):
    param_name = param_name.replace('.', '_')
    fn = os.path.join(directory, f'shard{shard_idx}', f'{param_name}.pt')
    return fn


def _get_topology_fn(directory: str):
    fn = os.path.join(directory, 'topology.json')
    return fn


def _aggregate_param_worker_shapes(module: Union[Module, Pipeline]) -> Dict[str, List[int]]:
    def _collect_recursive(mod, prefix=''):
        shapes = {}
        if hasattr(mod, 'param_worker_shapes'):
            for param_name, shape in mod.param_worker_shapes.items():
                n = prefix + param_name
                assert n not in shapes
                shapes[n] = shape
        for name, child in mod.named_children():
            child_shapes = _collect_recursive(child, prefix=prefix + name + '.')
            assert len(set(child_shapes.keys()) & set(shapes)) == 0
            shapes.update(child_shapes)
        return shapes
    shapes = _collect_recursive(module)
    # iterate over all parameters, and ensure we have a worker shape for them.
    for name, p in module.named_parameters():
        assert name in shapes, \
            f'Failed to find worker shape for parameter "{name}". ' + \
            f'It should be specified in `param_worker_shapes` in the respective module. ' + \
            f'If this parameter does not have a shape (i.e., it need not by merged), you must denote its shape as "None".'
    return shapes


def save_checkpoint(topo: Topology, module: Union[Module, Pipeline], directory: str):
    assert not os.path.exists(directory), \
        f'Checkpoint directory {directory} already exists.'
    if topo.get_data_parallel_idx() == 0:
        shard_idx = topo.model_comm.Get_rank()
        d = module.state_dict()
        for param_name, tensor in d.items():
            checkpoint_fn = _get_checkpoint_file_fn(directory, shard_idx, param_name)
            Path(os.path.dirname(checkpoint_fn)).mkdir(exist_ok=True, parents=True)
            torch.save(tensor, checkpoint_fn)
    if topo.is_root():
        # create the topology file
        topology_fn = _get_topology_fn(directory)
        with open(topology_fn, 'w') as f:
            json.dump({
                'dp': topo.dp,
                'pp': topo.pp,
                'mp': topo.mp,
                'param_worker_shapes': _aggregate_param_worker_shapes(module),
            }, f)
    topo.data_comm.Barrier()


def _load_parameter(directory: str, mp: int, param_name: str, shape: Optional[List[int]]) -> Tensor:
    tensors = []
    for shard_idx in range(mp):
        fn = _get_checkpoint_file_fn(directory, shard_idx, param_name)
        tensor = torch.load(fn)
        assert isinstance(tensor, Tensor)
        tensors.append(tensors)
    if shape is not None:
        # TODO: merge the tensors based on shape
        pass
    else:
        # shape is None, so we just return the first tensor
        return tensors[0]


def load_checkpoint(topo: Topology, module: Union[Module, Pipeline], directory: str):
    with open(_get_topology_fn(directory), 'r') as f:
        old_topology = json.load(f)
    # load this module's parameters
    for name, p in module.named_parameters():
        old_mp = old_topology['mp']
        old_shape = old_topology['param_worker_shapes'][name]
        tensor = _load_parameter(directory, mp=old_mp, param_name=name, shape=old_shape)
        # TODO: split the tensor to what is needed in the current topology
        p.data = tensor
    # TODO: catch unused params

    topo.data_comm.Barrier()
