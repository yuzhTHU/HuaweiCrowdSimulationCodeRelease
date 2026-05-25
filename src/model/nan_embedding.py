import torch
import torch.nn as nn


class NanEmbedding(nn.Module):
    """
    Linear embedding layer with explicit NaN handling.

    Instead of passing NaN inputs through the normal linear map, this layer
    replaces them with a dedicated learnable vector (`nan_embed`). This is
    useful for missing observations in trajectory data.
    """
    def __init__(self, input_dim, embed_dim, disable=False):
        """
        Args:
            input_dim (int): Input feature dimension.
            embed_dim (int): Output embedding dimension.
            disable (bool): Whether to disable the NaN embedding behavior.
        """
        super().__init__()
        self.embed = nn.Linear(input_dim, embed_dim)  # Mapping for regular values.
        self.nan_embed = nn.Parameter(torch.randn(embed_dim))  # Learnable vector for NaN inputs.
        self.disable = disable

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor, possibly containing NaN values.
                Shape: (..., input_dim)
        
        Returns:
            torch.Tensor: Embedded tensor with NaN positions replaced.
                Shape: (..., embed_dim)
        """
        nan_mask = torch.isnan(x).all(dim=-1)
        x = torch.nan_to_num(x, nan=0.0)
        out = self.embed(x)  # (N, L, D)
        if not self.disable:
            out[nan_mask, :] = self.nan_embed
        else:
            out[nan_mask, :] = 0.0
        return out

