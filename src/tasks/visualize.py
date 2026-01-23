import re
import sys
import time
import json
import torch
import shlex
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as D
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from socket import gethostname
from argparse import ArgumentParser
from setproctitle import setproctitle
from src.dataset import ETHDataset, UCYDataset, SDDDataset, GCDataset, WayMoDataset, ORCADataset
from src.model import Model, RelativeModel, NewModel
from src.diffusion import DDPM, DDIM
from src.utils.logger import init_logger
from src.utils.seed import seed_all
from src.utils.timer import NamedTimer
from src.utils.plot import get_fig
from src.utils.auto_gpu import AutoGPU
from src.utils.fix_parser import add_negation_flags, add_minus_flags
from src.utils.tag2ansi import tag2ansi
from src.utils.use_npu import USE_NPU, npu_attention_fallback_context
from src.utils.calc_xy_error import calc_xy_error
from src.tasks import train_once, test_once

def visualize(args, pos, vel, hst, for_plot, mask, pos_true, pos_pred, save_path, pid) -> None:
    """
    可视化单个样本的扩散过程。

    Args:
        args: 全局参数。
        pos: 当前时刻位置张量，形状为 (B, #pedestrian, 2)。
        vel: 当前时刻速度张量，形状为 (B, #pedestrian, 2)。
        hst: 历史轨迹张量，形状为 (B, hist_step, 2)。
        for_plot: 扩散过程中间结果列表，每个元素形状为 (B, #pedestrian, roll_step*pred_step, 2)。
        mask: 有效行人掩码张量，形状为 (B, )。
        pos_true: 真实未来轨迹张量，形状为 (B, pred_step, 2)。
        pos_pred: 预测未来轨迹张量，形状为 (sample_num, B, pred_step, 2)。
        save_path: 保存可视化图像的路径。
        pid: 要可视化的行人索引。
    """
    S = args.sample_num
    N = args.denoise_step
    batch_size, ped_num = pos.shape[:2]
    for_plot_acc = torch.concat([torch.stack(i, dim=0) for i in for_plot], dim=-2)  # (N+1, S*B, #pedestrian, roll_step*pred_step, 2)
    for_plot_acc = for_plot_acc.view(N+1, S, batch_size, ped_num, args.roll_step*args.pred_step, 2)  # (N+1, S, B, #pedestrian, roll_step*pred_step, 2)
    for_plot_vel = vel.unsqueeze(-2) + for_plot_acc.cumsum(dim=-2) / args.fps  # (N+1, S, B, #pedestrian, roll_step*pred_step, 2)
    for_plot_pos = pos.unsqueeze(-2) + for_plot_vel.cumsum(dim=-2) / args.fps  # (N+1, S, B, #pedestrian, roll_step*pred_step, 2)
    fi, fig, axes = get_fig(3, 4, AW=6, AH=6, dpi=300)
    if N + 1 <= 11:
        loader = range(N+1)
    else:
        loader = np.linspace(0, N+1, 12, dtype=int)[:-1].tolist()
    for idx, n in enumerate(loader):
        ax = axes[idx]
        ax.plot(*hst[mask, :, :][pid].cpu().numpy().T, color='blue', lw=1.0) # 历史轨迹
        ax.scatter(*pos[mask, :][pid].cpu().numpy(), color='blue') # 当前位置
        ax.plot(*pos_true[mask, :, :][pid].cpu().numpy().T, 'r.:', markevery=args.pred_step, lw=1.0) # 未来轨迹
        ax.title.set_text(f"Step {N-n} / {N}")
        for line in for_plot_pos[n, :, mask, :, :][:, pid]: # 逐步的扩散结果
            ax.plot(*line.cpu().numpy().T)
    ax = axes[-1]
    ax.plot(*hst[mask, :, :][pid].cpu().numpy().T, color='blue', lw=1.0) # 历史轨迹
    ax.scatter(*pos[mask, :][pid].cpu().numpy(), color='blue') # 当前位置
    ax.plot(*pos_true[mask, :, :][pid].cpu().numpy().T, 'r.:', markevery=args.pred_step, lw=1.0) # 未来轨迹
    for line in pos_pred[:, mask, :, :][:, pid]: # 最终的采样结果
        ax.plot(*line.cpu().numpy().T)
    for ax in axes:
        ax.axis('equal')
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    del fi, fig, axes
