import torch
from mpi4py import MPI
from dist_mingpt.utils import Broadcast, SumReduce


torch.seed(0)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class MLP(torch.nn.Module):
    def __init__(self, D, comm):
        super().__init__()
        self.comm = comm
        self.D = D
        size = self.comm.Get_size()
        assert (4 * D) % size == 0
        self.mlp = torch.nn.Sequential(
            Broadcast(comm, 0),
            torch.nn.Linear(D, 4 * D // size),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * D // size, D, with_bias=False),
            SumReduce(comm, 0),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.1)
                m.bias.data.zero_()

    def forward(self, x):
        return x

D = 2
assert (4 * D) % size == 0
net_dist = MLP()

if rank == 0:
    net_seq = torch.nn.Sequential(
        torch.nn.Linear(D, 4 * D),
        torch.nn.ReLU(),
        torch.nn.Linear(4 * D, D),
    )

# set the weight and bias
if rank == 0:
    w1 = torch.ones([D * 4, D])
    b1 = torch.ones([D * 4])
    w2 = torch.ones([D, D * 4])
    b2 = torch.ones([D])
else:
    w1, b1, w2, b2 = None, None, None, None
w1 = comm.bcast(w1, root=0)
b1 = comm.bcast(b1, root=0)
w2 = comm.bcast(w2, root=0)
b2 = comm.bcast(b2, root=0)
k = 4 * D // size
net_dist[1].weight.data = w1[rank * k : (rank + 1) * k, :]
net_dist[1].bias.data = b1[rank * k : (rank + 1) * k]
net_dist[2].weight.data = w2[:, rank * k : (rank + 1) * k]
net_dist[2].bias.data = b2

if rank == 0:
    net_seq[0].weight.data = w1
    net_seq[0].bias.data = b1
    net_seq[2].weight.data = w2
    net_seq[2].bias.data = b2
comm.Barrier()


if rank == 0:
    x = torch.ones(D)
    dy = torch.ones(D)
else:
    x = torch.empty(0)
    dy = torch.empty(0)
x.requires_grad = True

y1 = net_dist(x)
print(rank, y1)
if rank == 0:
    y2 = net_seq(x)
    print(rank, y2)
quit()

y.backward(dy)
print(rank, x.grad)
