import torch
import torch.nn as nn


class NanEmbedding(nn.Module):
    """
    支持 NaN 值处理的线性嵌入层。
    
    对于输入数据中的 NaN 值，该层不使用常规的线性映射，
    而是使用一个独立的可学习向量（nan_embed）进行替换。
    这对于处理轨迹数据中的缺失观测点非常有用。
    """
    def __init__(self, input_dim, embed_dim, disable=False):
        """
        Args:
            input_dim (int): 输入特征的维度。
            embed_dim (int): 输出嵌入向量的维度。
            disable (bool): 是否禁用 NaN 嵌入功能。
        """
        super().__init__()
        self.embed = nn.Linear(input_dim, embed_dim)  # 正常数值的映射
        self.nan_embed = nn.Parameter(torch.randn(embed_dim))  # 用于 nan 的可学习向量
        self.disable = disable

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，可能包含 NaN。
                Shape: (..., input_dim)
        
        Returns:
            torch.Tensor: 嵌入后的张量，NaN 位置已被替换。
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

