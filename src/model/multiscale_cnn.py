import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleCNN(nn.Module):
    """
    Multi-scale convolutional neural network module.

    Uses convolution layers with different kernel sizes (`3x3`, `5x5`) and
    dilation rates in parallel to capture map information at multiple scales,
    from local detail to larger structure.
    """
    def __init__(self, args):
        """
        Args:
            args (Namespace): Configuration parameters containing:
                - map_feature_dim (int): Number of input feature channels.
                - model_dim (int): Number of output feature channels.
        """
        super().__init__()
        dim = args.map_feature_dim

        # Branch 1: 3x3 receptive field for local detail.
        self.branch1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

        # Branch 2: 5x5 receptive field for medium-scale structure.
        self.branch2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2)

        # Branch 3: dilated convolution for larger context.
        self.branch3 = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2)
        
        self.fusion = nn.Conv2d(dim * 3, args.model_dim, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input feature map.
                Shape: (batch_size, in_channels, height, width)
        
        Returns:
            torch.Tensor: Output after fusing multi-scale features.
                Shape: (batch_size, out_channels, height, width)
        """
        x1 = F.relu(self.branch1(x))
        x2 = F.relu(self.branch2(x))
        x3 = F.relu(self.branch3(x))
        
        # Concatenate features from all branches.
        out = torch.cat([x1, x2, x3], dim=1)
        return self.fusion(out)
