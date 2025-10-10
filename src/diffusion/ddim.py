import torch
from argparse import Namespace
from .ddpm import DDPM

class DDIM(DDPM):
    def __init__(self, args: Namespace, eta=0.0):
        super().__init__(args)
        self.eta = eta
    
    def denoise(self, xt, denoise_t, x0=None, noise=None, stride=1):
        """ DDIM backward: 预测噪声并去噪 """
        if not ((x0 is None) ^ (noise is None)):
            raise ValueError("x0 和 noise 只能传入一个")
        if denoise_t == 0:
            raise ValueError("denoise_t 不能为 0")
        if denoise_t - stride < 0:
            raise ValueError("denoise_t - stride 不能小于 0")
        at = self.alpha_bar[denoise_t]
        at_next = self.alpha_bar[denoise_t - stride]
        var = self.eta**2 * (1 - at_next) / (1 - at) * (1 - at / at_next)
        if x0 is None:
            coef1 = torch.sqrt(at_next / at)
            coef2 = torch.sqrt(1 - at_next - var) - torch.sqrt(at_next * (1 - at) / at)
            mean = coef1 * xt + coef2 * noise
        else:
            coef1 = torch.sqrt(at_next) - torch.sqrt((1 - at_next - var) * at / (1 - at))
            coef2 = torch.sqrt((1 - at_next - var) / (1 - at))
            mean = coef1 * x0 + coef2 * xt
        if denoise_t - stride > 0:
            mean = mean + var.sqrt() * torch.randn_like(mean)
        return mean
