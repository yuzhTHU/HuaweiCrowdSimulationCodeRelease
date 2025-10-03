import torch
from argparse import Namespace
from .ddpm import DDPM

class DDIM(DDPM):
    def __init__(self, args: Namespace, eta=0.0):
        super().__init__(args)
        self.eta = eta
    
    def denoise(self, x_t, t, noise_pred):
        a_t = self.alphas_cumprod[t]
        x0_hat = (x_t - torch.sqrt(1 - a_t) * noise_pred) / torch.sqrt(a_t)
        if t > 0: # DDIM 更新公式
            a_prev = self.alphas_cumprod[t - 1]
            coef1 = torch.sqrt(a_prev / a_t)
            coef2 = torch.sqrt((1 - a_prev) / (1 - a_t))
            mean = coef1 * (x_t - coef2 * noise_pred)
            if self.eta == 0:
                x_t = mean
            else:
                noise = torch.randn_like(x_t)
                sigma = self.eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - self.alpha[t] / a_prev))
                x_t = mean + sigma * noise
        else: # t=0, 直接输出
            x_t = x0_hat
        return x_t