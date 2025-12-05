import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleCNN(nn.Module):
    """
    多尺度卷积神经网络模块。
    
    使用不同核大小（3x3, 5x5）和膨胀率（Dilation）的卷积层并行提取特征，
    用于捕获地图中不同尺度的环境信息（如局部细节和宏观结构）。
    """
    def __init__(self, args):
        """
        Args:
            args (Namespace): 配置参数，需包含：
                - map_feature_dim (int): 输入特征图的通道数。
                - model_dim (int): 输出特征的通道数。
        """
        super().__init__()
        dim = args.map_feature_dim
        
        # 分支1: 感受野 3x3 (看细节)
        self.branch1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        
        # 分支2: 感受野 5x5 (看中等物体)
        self.branch2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2)
        
        # 分支3: 膨胀卷积，感受野大 (看整体结构)
        self.branch3 = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2)
        
        self.fusion = nn.Conv2d(dim * 3, args.model_dim, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入特征图。
                Shape: (batch_size, in_channels, height, width)
        
        Returns:
            torch.Tensor: 融合多尺度特征后的输出。
                Shape: (batch_size, out_channels, height, width)
        """
        x1 = F.relu(self.branch1(x))
        x2 = F.relu(self.branch2(x))
        x3 = F.relu(self.branch3(x))
        
        # 拼接特征
        out = torch.cat([x1, x2, x3], dim=1)
        return self.fusion(out)
