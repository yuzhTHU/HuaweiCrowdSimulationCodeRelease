import torch.nn as nn


class MeanPoolingLSTM(nn.Module):
    """
    基于 LSTM 的时序特征提取器。
    
    处理序列数据，取 LSTM 所有时间步输出的平均值作为该序列的最终表示。
    常用于编码行人的历史轨迹或车辆轨迹。
    """
    def __init__(self, input_dim, embed_dim, layer_num):
        """
        Args:
            input_dim (int): LSTM 输入特征维度。
            embed_dim (int): LSTM 隐藏层维度（输出维度）。
            layer_num (int): LSTM 的层数。
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=embed_dim,
            num_layers=layer_num,
            batch_first=True,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入序列张量。
                Shape: (batch_size, num_agents, seq_len, input_dim)
                或者任意前导维度，只要最后两维是 (seq_len, input_dim)。
        
        Returns:
            torch.Tensor: 池化后的特征向量。
                Shape: (batch_size, num_agents, embed_dim) 
                保持除倒数第二维（seq_len）外的所有维度结构。
        """
        shape = x.shape
        x = x.view(-1, *shape[-2:])  # (batch_size * N, seq_len, input_dim)
        out, _ = self.lstm(x)  # (batch_size, seq_len, embed_dim)
        out = out.mean(dim=-2)  # (batch_size, embed_dim)
        out = out.view(*shape[:-2], shape[-1])  # (batch_size, N, embed_dim)
        return out
