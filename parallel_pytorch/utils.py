from typing import Literal
from mpi4py import MPI


def divide_optimally(n, parts):
    return [n // parts + (1 if i < n % parts else 0) for i in range(parts)]


def global_rank():
    return MPI.COMM_WORLD.Get_rank()


def make_topology_map(
    nodes: int,
    devices_per_node: int,
    pipeline_stages: int,
    data_parallel_copies: int,
    device: Literal['cpu', 'cuda'],
):
    """
    Make an object describing the 3D-parallelism topology.
    Assumes that the model will be horizontally distributed (model parallel) across devices in a node.
    """

    assert nodes == pipeline_stages * data_parallel_copies, "Invalid number of nodes"

    topo = []
    for i in range(data_parallel_copies):
        pipeline_topo = []
        for j in range(pipeline_stages):
            model_topo = []
            for k in range(devices_per_node):
                rank = i * pipeline_stages * devices_per_node + j * devices_per_node + k
                model_topo.append({
                    'rank': rank,
                    'device': 'cpu' if device == 'cpu' else f'cuda:{k}',
                })
            pipeline_topo.append(model_topo)
        topo.append(pipeline_topo)
    return topo

def count_nodes_in_topology_map(topo):
    nodes = 0
    for p in topo:
        for m in p:
            nodes += len(m)
    return nodes


def cumsum(l):
    return [sum(l[:i+1]) for i in range(len(l))]
