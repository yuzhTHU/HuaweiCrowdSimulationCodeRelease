import torch
import logging
import numpy as np
import torch.nn as nn
import torch.functional as F
from typing import List
from argparse import Namespace

_logger = logging.getLogger(__name__)


class DDPM:
    def __init__(self, args: Namespace, flexibility=0.0):
        self.args = args
        if args.beta_schedule == "cosine":
            beta = self.cosine_beta_schedule(args.T)
        elif args.beta_schedule == "linear":
            beta = self.linear_beta_schedule(args.T)
        else:
            raise ValueError(f"Unknown beta_schedule: {args.beta_schedule}")
        self.beta = torch.concatenate([torch.tensor([0.0], device=args.device), beta.to(args.device)])  # (T+1,)
        self.alpha = 1 - self.beta
        self.alpha_bar = self.alpha.cumprod(dim=0)
        self.flexibility = flexibility

    def add_noise(self, x0, denoise_t=None):
        """ DDPM forward: 给未来轨迹加噪 """
        if denoise_t is not None:
            raise NotImplementedError("指定 denoise_t 的功能尚未实现")
            if (denoise_t == 0).any():
                raise ValueError("denoise_t 不能为 0")
        batch_size = x0.shape[0]
        if self.args.antithetic_sampling:
            denoise_t_half = torch.randint(1, self.args.T+1, (batch_size // 2,), device=self.args.device)
            denoise_t = torch.cat([denoise_t_half, self.args.T + 1 - denoise_t_half], dim=0)  # (batch_size,)
            if batch_size % 2 == 1:
                t = torch.randint(1, self.args.T+1, (1,), device=self.args.device)
                denoise_t = torch.cat([denoise_t, t], dim=0)
        else:
            denoise_t = torch.randint(1, self.args.T+1, (batch_size,), device=self.args.device)  # (batch_size,)
        denoise_t = denoise_t.long()
        at = self.alpha_bar[denoise_t].view(batch_size, 1, 1, 1)
        noise = torch.randn_like(x0, device=self.args.device)
        xt = torch.sqrt(at) * x0 + torch.sqrt(1 - at) * noise
        return xt, noise, denoise_t

    def denoise(self, xt, denoise_t, x0=None, noise=None, stride=1):
        """ DDPM backward: 预测噪声并去噪 """
        if not ((x0 is None) ^ (noise is None)):
            raise ValueError("x0 和 noise 只能传入一个")
        if denoise_t == 0:
            raise ValueError("denoise_t 不能为 0")
        if denoise_t - stride < 0:
            raise ValueError("denoise_t - stride 不能小于 0")
        at = self.alpha[denoise_t]
        at_next = self.alpha[denoise_t - stride]
        if x0 is None:
            coef1 = (at_next / at).sqrt()
            coef2 = - (1 - at / at_next) / ((1 - at) * at / at_next).sqrt()
            mean = coef1 * xt + coef2 * noise
        else:
            coef1 = (1 - at / at_next) * torch.sqrt(at_next) / (1 - at)
            coef2 = (1 - at_next) * torch.sqrt(at / at_next) / (1 - at)
            mean = coef1 * x0 + coef2 * xt
        if denoise_t - stride > 0:
            var_upper = (1 - at / at_next)
            var_lower = (1 - at / at_next) * (1 - at_next) / (1 - at)
            var = (1 - self.flexibility) * var_upper + self.flexibility * var_lower
            mean = mean + var.sqrt() * torch.randn_like(xt)
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
