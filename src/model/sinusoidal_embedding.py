import math
import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    """
    正弦位置编码生成器。
    
    通常用于扩散模型（Diffusion Models）中，将离散的时间步（timesteps）
    映射为高维连续特征向量。包含一个正弦编码层和一个 MLP 投影层。
    """
    def __init__(self, embed_dim):
        """
        Args:
            embed_dim (int): 输出的嵌入维度。
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
            t (torch.LongTensor): 时间步索引。
                Shape: (batch_size, )
        
        Returns:
            torch.Tensor: 时间步的嵌入向量。
                Shape: (batch_size, embed_dim)
        """
        emb = t[:, None].float() * self.freq.to(t.device)  # (batch, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (batch, embed_dim)
        return self.mlp(emb)  # (batch, embed_dim)
