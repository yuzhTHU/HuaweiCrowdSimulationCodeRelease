import torch
import logging
import numpy as np
import torch.nn as nn
import torch.functional as F
from typing import List
from argparse import Namespace

_logger = logging.getLogger(__name__)


class DDPM:
    def __init__(self, args: Namespace):
        self.args = args
        if args.beta_schedule == "cosine":
            beta = self.cosine_beta_schedule(args.T).to(self.args.device)
        elif args.beta_schedule == "linear":
            beta = self.linear_beta_schedule(args.T).to(self.args.device)
        else:
            raise ValueError(f"Unknown beta_schedule: {args.beta_schedule}")
        self.beta = beta
        self.alpha = 1 - self.beta
        self.alpha_bar = self.alpha.cumprod(dim=0)

    def add_noise(self, x0, denoise_t=None):
        """ DDPM forward: 给未来轨迹加噪 """
        if denoise_t is not None:
            raise NotImplementedError("指定 denoise_t 的功能尚未实现")
        batch_size = x0.shape[0]
        denoise_t = torch.randint(
            0, self.args.T, (batch_size,), device=self.args.device
        ).long()  # (batch_size,)
        a_t = self.alpha_bar[denoise_t].view(batch_size, 1, 1, 1)
        noise = torch.randn_like(x0, device=self.args.device)
        xt = torch.sqrt(a_t) * x0 + torch.sqrt(1 - a_t) * noise
        return xt, noise, denoise_t

    def denoise(self, xt, t, x0_pred, flexibility=0.0):
        """ DDPM backward: 预测噪声并去噪 """
        # coef1 = 1 / torch.sqrt(self.alpha[t])
        coef1 = (1 - self.alpha[t]) * torch.sqrt(self.alpha_bar[t-1]) / (1 - self.alpha_bar[t])
        coef2 = (1 - self.alpha_bar[t-1]) * torch.sqrt(self.alpha[t]) / (1 - self.alpha_bar[t])
        mean = coef1 * x0_pred + coef2 * xt
        if t > 0:
            noise = torch.randn_like(xt)
            var1 = self.beta[t]
            var2 = (1 - self.alpha_bar[t - 1]) / (1 - self.alpha_bar[t]) * self.beta[t]
            var = (1 - flexibility) * var1 + flexibility * var2
            mean = mean + var.sqrt() * noise
        return mean


    @staticmethod
    def cosine_beta_schedule(T, s=0.008):
        # steps = T + 1
        # x = torch.linspace(0, T, steps)
        # alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * np.pi / 2) ** 2
        # alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        # betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        # return torch.clamp(betas, 0.0001, 0.9999)
        timesteps = torch.arange(T + 1) / T + s
        alpha_bar = timesteps / (1 + s) * np.pi / 2
        alpha_bar = torch.cos(alpha_bar).pow(2)
        alpha_bar = alpha_bar / alpha_bar[0]
        alpha = alpha_bar[1:] / alpha_bar[:-1]
        beta = 1 - alpha
        beta = beta.clamp(max=0.999)
        return beta
    
    @staticmethod
    def linear_beta_schedule(T, beta_start=0.0001, beta_end=0.05):
        return torch.linspace(beta_start, beta_end, T)
