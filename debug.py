import os
import sys
import json
import torch
import random
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from socket import gethostname
from setproctitle import setproctitle
from argparse import ArgumentParser, Namespace
from src.model.model import Model
from src.utils.seed import seed_all
from src.diffusion import DDPM, DDIM
from src.utils.auto_gpu import AutoGPU
from src.utils.logger import init_logger
from src.dataset import ETHDataset, UCYDataset, SDDDataset, GCDataset, WayMoDataset
from src.utils.plot import get_fig, plt, sns
from src.utils.timer import NamedTimer
from src.utils.tag2ansi import tag2ansi
from app import init_simulation, simulate_one_step

_logger = logging.getLogger('src.sample')


def main(args):
    timer = NamedTimer(unit='it', mode='pace')

    ## Check Arguments
    checkpoint_path = Path(args.reload_checkpoint)
    with open(checkpoint_path.parent / "args.json", 'r') as f:
        saved_args = Namespace(**json.load(f))
    for name in [
        # 'T', 'scale_accelerate', 'beta_schedule',
        # 'hist_step', 'pred_step', 'skip_step', 'fps', 'dot_per_meter', 
        # 'model_dim', 'map_feature_dim', 'head_num', 'latent_token_num'
        *(vars(saved_args) | vars(args)).keys()
    ]:
        if (v1 := getattr(args, name, None)) != (v2 := getattr(saved_args, name, None)):
            _logger.warning(f"Checkpoint {name} {v2} != current {name} {v1}!")
    timer.add('Check Args')

    ## Load Model & Load Checkpoint
    model = Model(args).to(args.device)
    if args.sampling_method == "DDIM":
        diffusion = DDIM(args)
    elif args.sampling_method == "DDPM":
        diffusion = DDPM(args, flexibility=0.0)
    else:
        raise ValueError(f"Unknown sampling_method {args.sampling_method}!")
    checkpoint_path = Path(args.reload_checkpoint)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "best.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found!")
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
    start_epoch = checkpoint["epoch"] + 1
    model.load_state_dict(checkpoint["model"])
    model.eval()
    torch.set_grad_enabled(False)
    _logger.info(f"Checkpoint loaded from {checkpoint_path}, resume from epoch {start_epoch}.")
    timer.add('Load Model')

    ## Prepare data
    # dataset = UCYDataset.load_data(args, "./data/UCY/data/data_zara/crowds_zara01.vsp")
    # frame_idx = 10
    dataset = WayMoDataset.load_data(args, "./data/WayMo/Processed/00383_3_9945506511750504/data.csv.gz")
    frame_idx = 7
    now = init_simulation(args, dataset, frame_idx, model)
    df_simulate = []
    timer.clear()
    for frame in range(frame_idx, frame_idx+args.roll_step, args.pred_step):
        df_new, now = simulate_one_step(args, model, diffusion, *now)
        df_simulate.append(df_new)
        timer.add(f"Embed Vehicle")
    df_simulate = pd.concat(df_simulate, axis=0)
    _logger.note(tag2ansi(
        f"[pink]Simulation done, sampled {1} times for {args.roll_step * args.pred_step} steps[reset]. "
        f"[bold underline orange]FPS = {timer._count['Embed Vehicle'] / timer.time:.2f} Hz[reset]. "
        f"[#66CCFF]Time Usage[reset]: {timer}"
    ))
    
    # # Visualize rollout
    # fi, fig, axes = get_fig(1, 1, AW=6, AH=6, dpi=300)
    # ax = axes[0]
    # for i, (pid, group) in enumerate(df_simulate.groupby('id')):
    #     color = sns.color_palette("hsv", traj.shape[1])[i]
    #     for s in range(S):
    #         ax.plot(traj[s, i, :, 0], traj[s, i, :, 1], alpha=0.5, color=color)
    #     ax.scatter(pos[i, 0], pos[i, 1], marker='o', color=color, s=10, zorder=10)
    #     ax.scatter(des[i, 0], des[i, 1], marker='*', color=color, s=10, zorder=10)
    #     ax.plot(hst[i, :, 0], hst[i, :, 1], color=color, linewidth=1, zorder=10)
    #     future = df_ped.loc[pd.IndexSlice[frame_idx+1:, pid], :].values
    #     ax.plot(future[:, 0], future[:, 1], color=color, linestyle='--', linewidth=1, zorder=10)

    # ax.set_facecolor('#b2bec3')
    # ax.imshow(map_data.map, extent=(map_data.xmin, map_data.xmax, map_data.ymin, map_data.ymax), cmap='gray_r', zorder=0, origin='lower')
    # ax.set_title(f"Rollout for {args.roll_step * args.pred_step} steps")
    # ax.set_xlabel("X (m)")
    # ax.set_ylabel("Y (m)")
    # ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    # fig.savefig(f"{args.exp_name}_rollout.png", bbox_inches='tight')
    # plt.close()
    # _logger.info(f"Rollout figure saved to {args.exp_name}_rollout.png")

    # _logger.note(f"Sampling finished. Re-run: {args.command}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default="sample")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument('--sampling_method', type=str, default="DDIM", choices=['DDPM', 'DDIM'])
    parser.add_argument("--T", type=int, default=100, help="训练时的扩散步数")
    parser.add_argument('--sample_num', type=int, default=20, help="测试时每个轨迹采样 {sample_num} 次")
    parser.add_argument('--denoise_step', type=int, default=10, help="采样时进行 {denoise_step} 次去噪")
    parser.add_argument('--step_offset', type=int, default=1, help="最后一步去噪从 x_{step_offset} 到 x_0")
    parser.add_argument('--scale_accelerate', type=float, default=1.0, help="加速度的缩放比例")
    parser.add_argument("--hist_step", type=int, default=8)
    parser.add_argument("--pred_step", type=int, default=1)
    parser.add_argument("--skip_step", type=int, default=1)
    parser.add_argument("--roll_step", type=int, default=12)
    parser.add_argument("--fps", type=int, default=2.5)
    parser.add_argument("--dot_per_meter", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="./logs/sample")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('--model_dim', type=int, default=128)
    parser.add_argument('--map_feature_dim', type=int, default=64)
    parser.add_argument('--head_num', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--attention_layer_num', type=int, default=1)
    parser.add_argument('--lstm_layer_num', type=int, default=3)
    parser.add_argument('--latent_token_num', type=int, default=16)
    parser.add_argument('--beta_schedule', type=str, default='linear', choices=['linear', 'cosine'])
    parser.add_argument('--cache_dataset', action='store_true', default=True)
    parser.add_argument('--reload_checkpoint', type=str, default=None, help='/path/to/checkpoint.pth', required=True)
    parser.add_argument('--predict_noise', action='store_true', default=True)
    parser.add_argument('--no_destination', action='store_true', default=False, help="不使用目的地信息")
    parser.add_argument('--no_speed', action='store_true', default=False, help="不使用速度信息")
    args, unknown = parser.parse_known_args()

    ## Build Save Path
    if args.exp_name is None:
        now = datetime.now()
        date = now.strftime("%Y%m%d")
        time = now.strftime("%H%M%S")
        host = gethostname()
        args.exp_name = f'{date}_{args.name}_{time}_{host}'
        invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        for char in invalid_chars:
            args.name = args.name.replace(char, '_')
    save_path = Path(args.save_dir) / args.exp_name
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        _logger.warning(f"Save path {save_path} already exists.")
    args.save_path = str(save_path)

    ## Set Seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    seed_all(args.seed)

    ## Init Logger
    init_logger(
        "src",
        exp_name=args.exp_name,
        log_file=save_path / "info.log",
        info_level="debug" if args.debug else "info",
    )

    ## Save Command
    args.command = ' '.join([sys.executable, *sys.argv])

    ## Warm Unknown Args
    if unknown:
        _logger.warning(f"Unknown args: {unknown}")

    ## Select GPU
    if args.device == "auto":
        args.device = AutoGPU().choice_gpu(memory_MB=6000, interval=15)

    ## Save Args
    args_path = save_path / "args.json"
    if args_path.exists():
        i = 1
        while args_path.with_suffix(f".json.{i}").exists(): i += 1
        args_path.rename(args_path.with_suffix(f".json.{i}"))
        _logger.warning(f"args.json already exists, backup to args.json.{i}")
    _logger.note(f"Args: {args}")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)

    ## Start Sampling
    setproctitle(f"{args.exp_name}@ZihanYu")
    main(args)
