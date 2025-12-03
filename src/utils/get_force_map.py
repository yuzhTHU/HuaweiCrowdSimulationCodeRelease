import torch
from functools import lru_cache

@lru_cache(maxsize=32)
def get_force_map(r=10, A=3.0, B=0.6, device='cpu'):
    x = torch.arange(-r, r+1).to(device=device)
    y = torch.arange(-r, r+1).to(device=device)
    p = torch.stack(torch.meshgrid(x, y, indexing='ij'), dim=-1).float()  # (2r+1, 2r+1, 2)
    d = torch.norm(p, dim=-1, keepdim=True)  # (2r+1, 2r+1, 1)
    n = -p / d.clamp(min=1e-6)
    F_map = A * torch.exp(-d / B) * n  # (2r+1, 2r+1, 2)
    return F_map
