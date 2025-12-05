import torch
import torch.nn as nn


class Residual(nn.Module):
    """
    残差连接模块 (Skip Connection)。
    
    实现 y = f(x) + x 的结构。如果输入输出维度不一致，
    会自动应用一个线性投影层将 x 映射到输出维度。
    """
    def __init__(self, *layers, input_dim=None, output_dim=None):
        """
        Args:
            *layers (nn.Module): 主路径上的网络层序列。
            input_dim (int, optional): 输入维度。仅当输入输出维度不同时需要。
            output_dim (int, optional): 输出维度。
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
