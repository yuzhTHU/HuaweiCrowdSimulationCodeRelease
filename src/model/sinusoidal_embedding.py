import math
import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal embedding generator.

    Commonly used in diffusion models to map discrete timesteps into
    high-dimensional continuous feature vectors. Includes a sinusoidal encoding
    stage followed by an MLP projection.
    """
    def __init__(self, embed_dim):
        """
        Args:
            embed_dim (int): Output embedding dimension.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.half_dim = self.embed_dim // 2
        self.freq = torch.exp(
            -torch.arange(self.half_dim).float()
            * (math.log(10000.0) / (self.half_dim - 1))
        )[None, :]

    def forward(self, t: torch.LongTensor):
        """
        Args:
            t (torch.LongTensor): Timestep indices.
                Shape: (batch_size, )
        
        Returns:
            torch.Tensor: Timestep embeddings.
                Shape: (batch_size, embed_dim)
        """
        emb = t[:, None].float() * self.freq.to(t.device)  # (batch, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (batch, embed_dim)
        return self.mlp(emb)  # (batch, embed_dim)
