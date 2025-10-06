import torch
from argparse import Namespace
from .ddpm import DDPM

class DDIM(DDPM):
    def __init__(self, args: Namespace, eta=0.0):
        super().__init__(args)
        self.eta = eta
    
    def denoise(self, xt, denoise_t, x0_pred, stride=1):
        """ DDIM backward: 预测噪声并去噪 """
        if denoise_t == 0:
            raise ValueError("denoise_t 不能为 0")
        if denoise_t - stride < 0:
            raise ValueError("denoise_t - stride 不能小于 0")
        coef1 = torch.sqrt(self.alpha_bar[denoise_t-stride]) - torch.sqrt((1 - self.alpha_bar[denoise_t-stride]) * self.alpha_bar[denoise_t] / (1 - self.alpha_bar[denoise_t]))
        coef2 = torch.sqrt((1 - self.alpha_bar[denoise_t-stride]) / (1 - self.alpha_bar[denoise_t]))
        mean = coef1 * x0_pred + coef2 * xt
        return mean
