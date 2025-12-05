import math
import torch
import torch.nn as nn


class FourierPositionalEncoding(nn.Module):
    """
    傅里叶位置编码 (Fourier Feature Mapping)。
    
    通过正弦和余弦函数将低维连续坐标（如 x, y）映射到高维频率空间。
    这有助于神经网络更好地学习数据中的高频细节（参见 NeRF 等相关工作）。
    """
    def __init__(self, out_dim: int = 256, num_bands: int = 64, min_freq: float = 1e-3):
        """
        Args:
            out_dim (int): 最终输出的编码维度。
            num_bands (int): 使用的频率频带数量。
            min_freq (float): 最小频率基数。
        """
        super().__init__()
        self.freqs = torch.linspace(min_freq, 0.5, num_bands)
        self.proj = nn.Linear(num_bands * 4, out_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): 输入坐标。
                Shape: (..., 2) 假设最后一维是 (x, y)
        
        Returns:
            torch.Tensor: 位置编码特征。
                Shape: (..., out_dim)
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
