import torch
from functools import lru_cache

@lru_cache(maxsize=32)
def get_force_map(r=10, A=3.0, B=0.6, device='cpu'):
    """
    Generate a local repulsive force-field template for the Social Force Model.

    Compute a radial repulsive force field centered at the origin, used to
    model repulsion between obstacles or pedestrians. The result is cached
    with LRU Cache to avoid repeated computation.

    Args:
        r (int, optional): Radius (half side length) of the force field.
            The grid size is (2r+1)x(2r+1). Default is 10.
        A (float, optional): Strength coefficient of the repulsive force.
            Default is 3.0.
        B (float, optional): Decay range coefficient of the repulsive force.
            Default is 0.6.
        device (str, optional): Device on which the tensor is stored.
            Default is 'cpu'.

    Returns:
        torch.Tensor: Generated force-field vector map.
            Shape: (2r+1, 2r+1, 2), where the last dimension is (fx, fy).
    """
    x = torch.arange(-r, r+1).to(device=device)
    y = torch.arange(-r, r+1).to(device=device)
    p = torch.stack(torch.meshgrid(x, y, indexing='ij'), dim=-1).float()  # (2r+1, 2r+1, 2)
    d = torch.norm(p, dim=-1, keepdim=True)  # (2r+1, 2r+1, 1)
    n = -p / d.clamp(min=1e-6)
    F_map = A * torch.exp(-d / B) * n  # (2r+1, 2r+1, 2)
    return F_map
