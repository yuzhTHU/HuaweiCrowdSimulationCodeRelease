import torch
from argparse import Namespace
from .ddpm import DDPM


class DDIM(DDPM):
    """
    Denoising Diffusion Implicit Model (DDIM).

    Inherits from `DDPM`. DDIM enables faster sampling with skipped steps while
    preserving generation quality through a non-Markovian reverse process.
    """

    def __init__(self, args: Namespace, eta=0.0):
        """
        Initialize the DDIM sampler.

        Args:
            args (Namespace): Configuration argument object.
            eta (float, optional): Hyperparameter controlling the stochasticity
                of the sampling process.
                - eta=0.0: deterministic sampling (Standard DDIM).
                - eta=1.0: DDPM-equivalent variance (Standard DDPM).
                Default is 0.0.
        """
        super().__init__(args)
        self.eta = eta
    
    def denoise(self, xt, denoise_t, x0=None, noise=None, stride=1):
        """
        DDIM reverse process: deterministically or semi-deterministically derive
        x_{t-stride} from x_t.

        This method overrides the parent `DDPM.denoise` implementation and uses
        the DDIM update rule.

        x_{t-1} = sqrt(alpha_bar_{t-1}) * "predicted x0" + 
                  sqrt(1 - alpha_bar_{t-1} - sigma_t^2) * "predicted noise" + 
                  sigma_t * epsilon_t

        Args:
            xt (torch.FloatTensor): Noisy data at the current timestep t.
            denoise_t (torch.LongTensor): Index of the current timestep t.
            x0 (torch.FloatTensor, optional): Model-predicted original data x0.
            noise (torch.FloatTensor, optional): Model-predicted noise epsilon.
                Note: exactly one of x0 and noise must be provided.
            stride (int, optional): Sampling step size used for acceleration.
                Default is 1.

        Returns:
            torch.FloatTensor: Denoised data at the previous step x_{t-stride}.
        """
        if not ((x0 is None) ^ (noise is None)):
            raise ValueError("Exactly one of `x0` and `noise` must be provided.")
        if denoise_t == 0:
            raise ValueError("`denoise_t` cannot be 0.")
        if denoise_t - stride < 0:
            raise ValueError("`denoise_t - stride` cannot be negative.")
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
