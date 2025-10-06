import torch
from argparse import Namespace
from .ddpm import DDPM

class DDIM(DDPM):
    def __init__(self, args: Namespace, eta=0.0):
        super().__init__(args)
        self.eta = eta
    
    def denoise(self, xt, t, x0_pred, stride=1):
        """ DDIM backward: 预测噪声并去噪 """
        coef1 = torch.sqrt(self.alpha_bar[t-stride]) - torch.sqrt((1 - self.alpha_bar[t-stride]) * self.alpha_bar[t] / (1 - self.alpha_bar[t]))
        coef2 = torch.sqrt((1 - self.alpha_bar[t-stride]) / (1 - self.alpha_bar[t]))
        mean = coef1 * x0_pred + coef2 * xt
        return mean
