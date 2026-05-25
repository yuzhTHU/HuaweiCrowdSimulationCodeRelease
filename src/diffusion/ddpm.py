import torch
import logging
import numpy as np
import torch.nn as nn
import torch.functional as F
from typing import List
from argparse import Namespace

_logger = logging.getLogger(__name__)


class DDPM:
    """
    Denoising Diffusion Probabilistic Model (DDPM).

    Implements the DDPM forward noising process and reverse denoising process.
    Supports both linear and cosine beta schedules.
    """

    def __init__(self, args: Namespace, flexibility=0.0):
        """
        Initialize the DDPM noise schedule.

        Args:
            args (Namespace): Configuration object containing:
                - beta_schedule (str): `'linear'` or `'cosine'`.
                - T (int): Total number of diffusion steps.
                - device (str): Compute device.
                - antithetic_sampling (bool): Whether to use antithetic
                  sampling in `add_noise` to reduce variance.
            flexibility (float, optional): Variance interpolation coefficient
                controlling stochasticity during generation. `0.0` corresponds
                to fixed variance and `1.0` to full DDPM variance.
        """
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

    def to(self, device):
        """
        Move the noise schedule tensors to the target device.

        Args:
            device (torch.device or str): Target device.

        Returns:
            self: Returns the instance itself for chaining.
        """
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        return self

    def add_noise(self, x0, denoise_t=None):
        """
        DDPM forward process: add noise to clean data `x0` to produce `x_t`.

        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

        Args:
            x0 (torch.FloatTensor): Original clean data (t=0).
                Shape: (batch_size, ...) with arbitrary dimensions.
            denoise_t (torch.LongTensor, optional): Specified timestep t.
                If None, t is sampled randomly according to the
                `args.antithetic_sampling` strategy.
                Shape: (batch_size, )。

        Returns:
            tuple:
                - xt (torch.FloatTensor): Noisy data. Same shape as x0.
                - noise (torch.FloatTensor): Added standard Gaussian noise epsilon. Same shape as x0.
                - denoise_t (torch.LongTensor): Actual timestep t used. Shape: (batch_size, ).
        """
        if denoise_t is not None:
            raise NotImplementedError("Specifying `denoise_t` explicitly is not implemented yet.")
            if (denoise_t == 0).any():
                raise ValueError("`denoise_t` cannot be 0.")
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
        """
        DDPM reverse process: sample x_{t-stride} from x_t using the predicted x0 or noise.

        p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 * I)

        Args:
            xt (torch.FloatTensor): Noisy data at the current timestep t.
                Shape: (batch_size, ...)
            denoise_t (torch.LongTensor): Index of the current timestep t.
                Shape: (batch_size, )
            x0 (torch.FloatTensor, optional): Model-predicted original data x0.
            noise (torch.FloatTensor, optional): Model-predicted noise epsilon.
                Note: exactly one of x0 and noise must be provided.
            stride (int, optional): Reverse denoising step size. Default is 1.
                Used as a skip-step strategy to accelerate sampling.

        Returns:
            torch.FloatTensor: Denoised data at the previous step x_{t-stride}.
                Same shape as xt.
        """
        if not ((x0 is None) ^ (noise is None)):
            raise ValueError("Exactly one of `x0` and `noise` must be provided.")
        if denoise_t == 0:
            raise ValueError("`denoise_t` cannot be 0.")
        if denoise_t - stride < 0:
            raise ValueError("`denoise_t - stride` cannot be negative.")
        at = self.alpha_bar[denoise_t]
        at_next = self.alpha_bar[denoise_t - stride]
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

    def noise_to_x0(self, xt, denoise_t, noise):
        """
        Infer the original data x0 from the current noisy data xt and predicted noise epsilon.

        x_0 = (x_t - sqrt(1 - alpha_bar_t) * epsilon) / sqrt(alpha_bar_t)

        Args:
            xt (torch.FloatTensor): Noisy data x_t.
            denoise_t (torch.LongTensor or int): Timestep t.
            noise (torch.FloatTensor): Predicted noise epsilon.

        Returns:
            torch.FloatTensor: Estimated original data x0.
        """
        if (
            (isinstance(denoise_t, int) and (denoise_t == 0)) or
            (isinstance(denoise_t, torch.Tensor) and (denoise_t == 0).any())
        ):
            raise ValueError("`denoise_t` cannot be 0.")
        at = self.alpha_bar[denoise_t]
        if isinstance(denoise_t, torch.Tensor) and denoise_t.numel() > 1:
            at = at.view(xt.shape[0], *[1] * (xt.ndim - 1))
        x0 = (xt - torch.sqrt(1 - at) * noise) / torch.sqrt(at)
        return x0

    def x0_to_noise(self, xt, denoise_t, x0):
        """
        Infer the latent noise epsilon from the current noisy data xt and estimated x0.

        epsilon = (x_t - sqrt(alpha_bar_t) * x_0) / sqrt(1 - alpha_bar_t)

        Args:
            xt (torch.FloatTensor): Noisy data x_t.
            denoise_t (torch.LongTensor or int): Timestep t.
            x0 (torch.FloatTensor): Estimated original data x0.

        Returns:
            torch.FloatTensor: Inferred latent noise epsilon.
        """
        if (
            (isinstance(denoise_t, int) and (denoise_t == 0)) or
            (isinstance(denoise_t, torch.Tensor) and (denoise_t == 0).any())
        ):
            raise ValueError("`denoise_t` cannot be 0.")
        at = self.alpha_bar[denoise_t]
        if isinstance(denoise_t, torch.Tensor) and denoise_t.numel() > 1:
            at = at.view(xt.shape[0], *[1] * (xt.ndim - 1))
        noise = (xt - torch.sqrt(at) * x0) / torch.sqrt(1 - at)
        return noise

    @staticmethod
    def cosine_beta_schedule(T, s=0.008):
        """
        Generate a cosine annealing beta schedule.

        Args:
            T (int): Total number of timesteps.
            s (float, optional): Offset used to prevent beta from becoming too
                small at t=0. Default is 0.008.

        Returns:
            torch.FloatTensor: Sequence of beta values. Shape: (T,)
        """
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
        """
        Generate a linearly increasing beta schedule.

        Args:
            T (int): Total number of timesteps.
            beta_start (float, optional): Initial beta value. Default is 1e-4.
            beta_end (float, optional): Final beta value. Default is 0.05.

        Returns:
            torch.FloatTensor: Sequence of beta values. Shape: (T,)
        """
        return torch.linspace(beta_start, beta_end, T)
