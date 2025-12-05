import torch
from functools import lru_cache

@lru_cache(maxsize=32)
def get_force_map(r=10, A=3.0, B=0.6, device='cpu'):
    """
    生成用于社会力模型 (Social Force Model) 的局部排斥力场模板。
    
    计算一个以中心为原点的径向排斥力场，用于模拟障碍物或行人之间的排斥作用。
    使用 LRU Cache 缓存结果以避免重复计算。

    Args:
        r (int, optional): 力场的半径（半边长）。网格大小为 (2r+1)x(2r+1)。默认为 10。
        A (float, optional): 排斥力的强度系数。默认为 3.0。
        B (float, optional): 排斥力的衰减范围系数。默认为 0.6。
        device (str, optional): 张量所在的设备。默认为 'cpu'。

    Returns:
        torch.Tensor: 生成的力场向量图。
            Shape: (2r+1, 2r+1, 2) 最后一维是 (fx, fy)。
    """
    x = torch.arange(-r, r+1).to(device=device)
    y = torch.arange(-r, r+1).to(device=device)
    p = torch.stack(torch.meshgrid(x, y, indexing='ij'), dim=-1).float()  # (2r+1, 2r+1, 2)
    d = torch.norm(p, dim=-1, keepdim=True)  # (2r+1, 2r+1, 1)
    n = -p / d.clamp(min=1e-6)
    F_map = A * torch.exp(-d / B) * n  # (2r+1, 2r+1, 2)
    return F_map
