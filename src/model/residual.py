import torch
import torch.nn as nn


class Residual(nn.Module):
    """
    Residual block with an optional projection shortcut.

    Implements `y = f(x) + x`. When the input and output dimensions differ, a
    linear projection is applied to map `x` to the output dimension.
    """
    def __init__(self, *layers, input_dim=None, output_dim=None):
        """
        Args:
            *layers (nn.Module): Sequence of layers on the main path.
            input_dim (int, optional): Input dimension. Needed only when the input and output dimensions differ.
            output_dim (int, optional): Output dimension.
        """
        super().__init__()
        self.net = nn.Sequential(*layers)
        self.need_proj = (
            input_dim is not None and output_dim is not None and input_dim != output_dim
        )
        if self.need_proj:
            self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if self.need_proj:
            return self.proj(x) + self.net(x)
        else:
            return x + self.net(x)
