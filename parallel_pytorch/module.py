from collections import OrderedDict
import torch
import torch.nn as nn

from parallel_pytorch.topology import Topology


class ParallelModule(nn.Module):
    """
    A replacement to torch.nn.Module that adds functionality for parallelization, such as checkpointing.
    """

    def parallel_state_dict(self, prefix=''):
        """
        Get the state dict of the module recursively.
        """

        d = OrderedDict()
        for name, child in self.named_children():
            if isinstance(child, ParallelModule):
                d.update(child.parallel_state_dict(prefix=prefix + name + '.'))
            else:
                d.update(child.state_dict(prefix=prefix + name + '.'))
        return d

    def parallel_load_state_dict(self, state_dict, prefix=''):
        """
        Load the state dict to this module recursively.
        """

        def filtered(d, pre):
            d = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith(pre):
                    k2 = k[len(pre):]
                    d[k2] = v
            return d

        for name, child in self.named_children():
            if isinstance(child, ParallelModule):
                child.parallel_load_state_dict(state_dict, prefix=prefix + name + '.')
            else:
                d = filtered(state_dict, prefix + name + '.')
                child.load_state_dict(d, strict=True)
