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


_logger = logging.getLogger('src.sample')


def main(args):
    # Load Dataset
    dataset = UCYDataset.load_data(args, "./data/UCY/data/data_zara/crowds_zara01.vsp")

    # Load Model
    model = Model(args).to(args.device)
    if args.sampling_method == "DDIM":
        diffusion = DDIM(args)
    elif args.sampling_method == "DDPM":
        diffusion = DDPM(args, flexibility=0.0)
    else:
        raise ValueError(f"Unknown sampling_method {args.sampling_method}!")

    # Check Arguments
    checkpoint_path = Path(args.reload_checkpoint)
    with open(checkpoint_path.parent / "args.json", 'r') as f:
        saved_args = Namespace(**json.load(f))
    for name in [
        'T', 'scale_accelerate', 'beta_schedule',
        'hist_step', 'pred_step', 'skip_step', 'fps', 'dot_per_meter', 
        'model_dim', 'map_feature_dim', 'head_num', 'latent_token_num'
    ]:
        if getattr(args, name) != getattr(saved_args, name):
            _logger.warning(f"Checkpoint {name} {getattr(saved_args, name)} != current {name} {getattr(args, name)}!")

    # Load Checkpoint
    checkpoint_path = Path(args.reload_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found!")
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
    start_epoch = checkpoint["epoch"] + 1
    model.load_state_dict(checkpoint["model"])
    model.eval()
    torch.set_grad_enabled(False)
    _logger.info(f"Checkpoint loaded from {checkpoint_path}, resume from epoch {start_epoch}.")

    # Prepare frame
    frame_idx = 10
    df_data = dataset.df_data.set_index(['f', 'id']).sort_index()
    df_ped = df_data.loc[df_data['type'] == 'pedestrian', ['x', 'y']]
    df_veh = df_data.loc[df_data['type'] == 'vehicle', ['x', 'y']]
    ped_list = df_ped.loc[frame_idx].index.tolist() if frame_idx in df_ped.index else []
    veh_list = df_veh.loc[frame_idx].index.tolist() if frame_idx in df_veh.index else []
    _logger.info(f"Frame {frame_idx}: {len(ped_list)} pedestrians, {len(veh_list)} vehicles.")
    pos = (
        df_ped
        .reindex(pd.MultiIndex.from_product([
            [frame_idx], 
            ped_list
        ], names=['f', 'id']))
        # .fillna(0.0)  # 不应该有 nan
        .values.reshape(len(ped_list), 2) # (#pedestrian, 2)
    )
    vel = (
        df_ped
        .reindex(pd.MultiIndex.from_product([
            [frame_idx-1, frame_idx], 
            ped_list
        ], names=['f', 'id']))
        .unstack()
        .diff().mul(args.fps).iloc[1]
        .unstack().T
        .fillna(0.0)
        .values # (#pedestrian, 2)
    )
    hst = (
        df_ped
        .reindex(pd.MultiIndex.from_product([
            range(frame_idx-args.hist_step, frame_idx), 
            ped_list
        ], names=['f', 'id']))
        .values.reshape(args.hist_step, len(ped_list), 2) # (hist_step, #pedestrian, 2)
        .transpose(1, 0, 2) # (#pedestrian, hist_step, 2)
    )
    veh = (
        df_veh
        .reindex(pd.MultiIndex.from_product([
            range(frame_idx-args.hist_step, frame_idx + 1), 
            veh_list
        ], names=['f', 'id']))
        .values.reshape(args.hist_step+1, len(veh_list), 2) # (#vehicle, hist_step + 1, 2)
        .transpose(1, 0, 2) # (#vehicle, hist_step + 1, 2)
    )
    des = (
        df_ped
        .loc[pd.IndexSlice[frame_idx + 1:, ped_list], :]
        .groupby(level=1, sort=False).tail(1)
        .swaplevel(axis=0).reindex(index=ped_list, level=0)
        .values # (#pedestrian, 2)
    )
    spd = (
        df_ped
        .reindex(pd.MultiIndex.from_product([
            range(frame_idx, frame_idx+int(5*args.fps) + 1),
            ped_list
        ], names=['f', 'id']))
        .unstack().ffill().bfill().diff().mul(args.fps).iloc[1:]
        .stack(future_stack=True).pow(2).sum(axis='columns').pow(0.5)
        .unstack().mean(axis='rows')
        .values[:, np.newaxis] # (#pedestrian, 1)
    )
    map_data = dataset.map_data

    if args.no_destination:
        des *= np.nan
    
    if args.no_speed:
        spd *= np.nan

    # Rollout for N steps
    S = args.sample_num  # 采样次数
    N = args.denoise_step  # 采样步数
    assert args.T % N == 0, f"试图使用 {N} 步采样，然而训练步数 {args.T} mod {N} 不等于 0!"
    assert 1 <= args.step_offset <= args.T // N, f"step_offset 应该取值于 {{1, ..., {args.T // N}}}!"
    pos_now = torch.from_numpy(pos).to(device=args.device, dtype=torch.float32)[None, ...].repeat(S, 1, 1)  # (S*1, #pedestrian, 2)
    vel_now = torch.from_numpy(vel).to(device=args.device, dtype=torch.float32)[None, ...].repeat(S, 1, 1)  # (S*1, #pedestrian, 2)
    hst_now = torch.from_numpy(hst).to(device=args.device, dtype=torch.float32)[None, ...].repeat(S, 1, 1, 1)  # (S*1, #pedestrian, hist_step, 2)
    des_now = torch.from_numpy(des).to(device=args.device, dtype=torch.float32)[None, ...].repeat(S, 1, 1)  # (S*1, #pedestrian, 2)
    spd_now = torch.from_numpy(spd).to(device=args.device, dtype=torch.float32)[None, ...].repeat(S, 1, 1)  # (S*1, #pedestrian, 1)
    veh_now = torch.from_numpy(veh).to(device=args.device, dtype=torch.float32)[None, ...].repeat(S, 1, 1, 1)  # (S*1, #vehicle, hist_step + 1, 2)
    ped_length_repeat = torch.full((S, ), len(ped_list), device=args.device, dtype=torch.long)  # (S,)
    veh_length_repeat = torch.full((S, ), len(veh_list), device=args.device, dtype=torch.long)  # (S,)

    model.set_map_embedding(
        map=torch.from_numpy(map_data.map).to(device=args.device, dtype=torch.float32),
        xmin=map_data.xmin,
        xmax=map_data.xmax,
        ymin=map_data.ymin,
        ymax=map_data.ymax,
    )
    traj = []
    for frame in range(frame_idx, frame_idx+args.roll_step, args.pred_step):
        model.set_veh_embedding(veh=veh_now)
        model.set_ped_embedding(pos=pos_now, vel=vel_now, hst=hst_now, des=des_now, spd=spd_now)
        model.set_sur_info()

        shape = [S, len(ped_list), args.pred_step, 2]  # (S*1, #pedestrian, pred_step, 2)
        xt = torch.randn(shape, device=args.device)  # 从噪声开始
        stride = args.T // N
        steps = reversed(range(args.step_offset, args.T+1, stride))
        for t in steps:
            noisy_acc = xt
            denoise_t = torch.full((xt.shape[0],), t, device=args.device, dtype=torch.long)
            output = model(
                noisy_acc=noisy_acc, 
                denoise_t=denoise_t,
                ped_length=ped_length_repeat, 
                veh_length=veh_length_repeat,
            )  # (S*B, #pedestrian, pred_step, 2)
            if args.predict_noise:
                xt = diffusion.denoise(xt, t, noise=output, stride=min(stride, t))
            else:
                xt = diffusion.denoise(xt, t, x0=output, stride=min(stride, t))
        
        frame_new = frame + args.pred_step
        acc_new = xt / args.scale_accelerate
        vel_new = vel_now.unsqueeze(-2) + acc_new.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
        pos_new = pos_now.unsqueeze(-2) + vel_new.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
        veh_new = torch.from_numpy(
            df_veh
            .loc[frame+1:frame+args.pred_step]  # pandas 中的切片是闭区间，因此实际上切出来了 pred_step 帧
            .unstack().swaplevel(axis='columns').sort_index(axis='columns')
            .reindex(columns=veh_list, level=0)
            .reindex(index=range(frame+1, frame+args.pred_step+1))
            .values.reshape(args.hist_step+1, veh.shape[0], 2) # (hist_step + 1, #vehicle, 2)
            .transpose(1, 0, 2) # (#vehicle, hist_step + 1, 2)
        ).to(device=args.device, dtype=torch.float32)
        des_new = des_now  # (S*B, #pedestrian, 2)
        spd_new = spd_now  # (S*B, #pedestrian, 1)
        # spd_new = torch.from_numpy(
        #     df_ped
        #     .reindex(pd.MultiIndex.from_product([
        #         range(frame_new, frame_new + int(5*args.fps) + 1),
        #         ped_list
        #     ], names=['f', 'id']))
        #     .unstack().ffill().bfill().diff().mul(args.fps).iloc[1:]
        #     .stack().pow(2).sum(axis='columns').pow(0.5)
        #     .unstack().mean(axis='rows')
        #     .values[:, np.newaxis] # (#pedestrian, 1)
        # ).to(device=args.device, dtype=torch.float32)

        hst_now = torch.cat([hst_now, pos_now.unsqueeze(-2), pos_new], dim=-2)[:, :, -args.hist_step-1:-1, :] # (S*B, #pedestrian, hist_step, 2)
        veh_now = torch.cat([veh_now, veh_new.unsqueeze(0).repeat(S, 1, 1, 1)], dim=-2)[:, :, -args.hist_step-1:, :] # (S*B, #vehicle, hist_step + 1, 2)
        pos_now = pos_new[:, :, -1, :] # (S*B, #pedestrian, 2)
        vel_now = vel_new[:, :, -1, :] # (S*B, #pedestrian, 2)
        spd_now = spd_new  # (S*B, #pedestrian, 1)
        des_now = des_new  # (S*B, #pedestrian, 2)
        
        traj.append(pos_new.cpu().numpy())  # list of (S*B, #pedestrian, pred_step, 2)

    traj = np.concatenate(traj, axis=-2)  # (S*B, #pedestrian, roll_step * pred_step, 2)
    traj = traj.reshape(S, len(ped_list), args.roll_step * args.pred_step, 2)  # (S, #pedestrian, roll_step * pred_step, 2)
    
    # Visualize rollout
    fi, fig, axes = get_fig(1, 1, AW=6, AH=6, dpi=300)
    ax = axes[0]
    for i in range(traj.shape[1]):
        color = sns.color_palette("hsv", traj.shape[1])[i]
        for s in range(S):
            ax.plot(traj[s, i, :, 0], traj[s, i, :, 1], alpha=0.5, color=color)
        ax.scatter(pos[i, 0], pos[i, 1], marker='o', color=color, s=10, zorder=10)
        ax.scatter(des[i, 0], des[i, 1], marker='*', color=color, s=10, zorder=10)
        ax.plot(hst[i, :, 0], hst[i, :, 1], color=color, linewidth=1, zorder=10)
        future = df_ped.loc[pd.IndexSlice[frame_idx+1:, i], :].values
        ax.plot(future[:, 0], future[:, 1], color=color, linestyle='--', linewidth=1, zorder=10)

    ax.set_facecolor('#b2bec3')
    ax.imshow(map_data.map, extent=(map_data.xmin, map_data.xmax, map_data.ymin, map_data.ymax), cmap='gray_r', zorder=0, origin='lower')
    ax.set_title(f"Rollout for {args.roll_step * args.pred_step} steps")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    fig.savefig(f"{args.exp_name}_rollout.png", bbox_inches='tight')
    plt.close()
    _logger.info(f"Rollout figure saved to {args.exp_name}_rollout.png")

    _logger.info("Done.")


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
    parser.add_argument('--scale_accelerate', type=float, default=10.0, help="加速度的缩放比例")
    parser.add_argument("--hist_step", type=int, default=8)
    parser.add_argument("--pred_step", type=int, default=1)
    parser.add_argument("--skip_step", type=int, default=1)
    parser.add_argument("--roll_step", type=int, default=12)
    parser.add_argument("--fps", type=int, default=2.5)
    parser.add_argument("--dot_per_meter", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="./logs/sample")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('--model_dim', type=int, default=128)
    parser.add_argument('--map_feature_dim', type=int, default=64)
    parser.add_argument('--head_num', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--latent_token_num', type=int, default=16)
    parser.add_argument('--beta_schedule', type=str, default='linear', choices=['linear', 'cosine'])
    parser.add_argument('--no_cache_dataset', dest='cache_dataset', action='store_false', default=True)
    parser.add_argument('--reload_checkpoint', type=str, default=None, help='/path/to/checkpoint.pth', required=True)
    parser.add_argument('--no_predict_noise', action='store_false', dest='predict_noise', default=True)
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
    args.command = ' '.join(sys.argv)

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
