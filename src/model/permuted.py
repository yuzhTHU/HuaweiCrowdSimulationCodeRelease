import torch
import torch.nn as nn


class Permuted(nn.Module):
    """
    Dimension permutation layer.
    
    Wraps `Tensor.permute` so it can be used inside `nn.Sequential`.
    """
    def __init__(self, *dims):
        """
        Args:
            *dims (int): Target dimension ordering.
        """
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)
