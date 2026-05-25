import math
import torch
import torch.nn as nn


class FourierPositionalEncoding(nn.Module):
    """
    Fourier positional encoding (Fourier feature mapping).

    Maps low-dimensional continuous coordinates such as `(x, y)` into a
    high-dimensional frequency space with sine and cosine functions. This helps
    neural networks learn high-frequency details more effectively, as in NeRF.
    """
    def __init__(self, out_dim: int = 256, num_bands: int = 64, min_freq: float = 1e-3):
        """
        Args:
            out_dim (int): Final output embedding dimension.
            num_bands (int): Number of frequency bands used.
            min_freq (float): Minimum base frequency.
        """
        super().__init__()
        self.freqs = torch.linspace(min_freq, 0.5, num_bands)
        self.proj = nn.Linear(num_bands * 4, out_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input coordinates.
                Shape: `(..., 2)`, assuming the last dimension is `(x, y)`.
        
        Returns:
            torch.Tensor: Positional encoding features.
                Shape: `(..., out_dim)`.
        """
        freqs = self.freqs.to(x.device) * math.pi * 2
        x_proj = x[..., (0,)] * freqs  # (..., num_bands)
        y_proj = x[..., (1,)] * freqs  # (..., num_bands)
        fourier = torch.cat([
            torch.sin(x_proj), 
            torch.cos(x_proj),
            torch.sin(y_proj),
            torch.cos(y_proj),
        ], dim=-1)  # (..., num_bands*4)
        fourier = torch.nan_to_num(fourier, nan=0.0)
        pe = self.proj(fourier)
        return pe
