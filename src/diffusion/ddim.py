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
            args (Namespace): 配置参数对象。
            eta (float, optional): 控制采样过程随机性的超参数。
                - eta=0.0: 对应确定性采样 (Standard DDIM)。
                - eta=1.0: 对应 DDPM 的方差 (Standard DDPM)。
                默认为 0.0。
        """
        super().__init__(args)
        self.eta = eta
    
    def denoise(self, xt, denoise_t, x0=None, noise=None, stride=1):
        """
        DDIM 反向过程：确定性或半确定性地从 x_t 推导 x_{t-stride}。
        
        该方法重写了父类 DDPM 的 denoise 方法，使用 DDIM 的更新公式。
        
        x_{t-1} = sqrt(alpha_bar_{t-1}) * "predicted x0" + 
                  sqrt(1 - alpha_bar_{t-1} - sigma_t^2) * "predicted noise" + 
                  sigma_t * epsilon_t

        Args:
            xt (torch.FloatTensor): 当前时间步 t 的带噪数据。
            denoise_t (torch.LongTensor): 当前时间步 t 的索引。
            x0 (torch.FloatTensor, optional): 模型预测的原始数据 x0。
            noise (torch.FloatTensor, optional): 模型预测的噪声 epsilon。
                注意：x0 和 noise 必须且只能提供其中一个。
            stride (int, optional): 采样步长，用于加速。默认为 1。

        Returns:
            torch.FloatTensor: 去噪后的上一时刻数据 x_{t-stride}。
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
