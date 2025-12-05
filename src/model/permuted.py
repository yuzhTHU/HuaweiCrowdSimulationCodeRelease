import torch
import torch.nn as nn


class Permuted(nn.Module):
    """
    维度置换层。
    
    封装了 Tensor.permute 操作，使其可以被放入 nn.Sequential 中。
    """
    def __init__(self, *dims):
        """
        Args:
            *dims (int): 目标维度的顺序索引。
        """
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)
