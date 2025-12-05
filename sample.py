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
from src.model import Model, RelativeModel, NewModel
from src.utils.seed import seed_all
from src.diffusion import DDPM, DDIM
from src.utils.auto_gpu import AutoGPU
from src.utils.logger import init_logger
from src.dataset import ETHDataset, UCYDataset, SDDDataset, GCDataset, WayMoDataset
from src.utils.plot import get_fig, plt, sns
from src.utils.timer import NamedTimer
from src.utils.tag2ansi import tag2ansi


_logger = logging.getLogger('src.sample')


def main(args):
    timer = NamedTimer(unit='it', mode='pace')

    ## Check Arguments
    checkpoint_path = Path(args.reload_checkpoint)
    with open(checkpoint_path.parent / "args.json", 'r') as f:
        saved_args = Namespace(**json.load(f))
    for name in args.__dict__.keys():
        if name not in [
            "name", "exp_name", "save_dir", "save_path", "command",
            "seed", "sample_num", "dropout", "reload_checkpoint",
            "roll_step", "no_destination", "no_speed",
        ]:
            if (current := getattr(args, name)) != (saved := getattr(saved_args, name, None)):
                _logger.warning(f"Saved {name}={saved} != current {name}={current}!")
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
    dataset = UCYDataset.load_data(args, "./data/UCY/data/data_zara/crowds_zara01.vsp")
    frame_idx = 10
    df_data = dataset.df_data.set_index(['f', 'id']).sort_index()
    df_ped = df_data.loc[df_data['type'] == 'pedestrian', ['x', 'y']]
    df_veh = df_data.loc[df_data['type'] == 'vehicle', ['x', 'y']]
    ped_list = df_ped.loc[frame_idx].index.tolist() if frame_idx in df_ped.index else []
    veh_list = df_veh.loc[frame_idx].index.tolist() if frame_idx in df_veh.index else []
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
    _logger.info(f"Frame {frame_idx}: {len(ped_list)} pedestrians, {len(veh_list)} vehicles.")
    timer.add('Prepare Initial Step')

    ## Simulation
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
    timer.add('Array to Tensor')
    model.set_map_embedding(
        map=torch.from_numpy(map_data.map).to(device=args.device, dtype=torch.float32),
        xmin=map_data.xmin,
        xmax=map_data.xmax,
        ymin=map_data.ymin,
        ymax=map_data.ymax,
    )
    timer.add('Embed Map')
    _logger.info(tag2ansi(
        f"[pink]Begin to simulate for {args.roll_step} steps[reset]. "
        f"[#66CCFF]Time Usage[reset] so far: {timer}"
    ))
    torch.cuda.synchronize()
    timer.clear(reset=True)
    traj = []
    for frame in range(frame_idx, frame_idx+args.roll_step, args.pred_step):
        model.set_veh_embedding(veh=veh_now)
        torch.cuda.synchronize()
        timer.add('Embed Vehicle')
        model.set_ped_embedding(pos=pos_now, vel=vel_now, hst=hst_now, des=des_now, spd=spd_now)
        torch.cuda.synchronize()
        timer.add('Embed Pedestrian')
        model.set_sur_info()
        torch.cuda.synchronize()
        timer.add('Embed Surroundings')

        shape = [S, len(ped_list), args.pred_step, 2]  # (S*1, #pedestrian, pred_step, 2)
        xt = torch.randn(shape, device=args.device)  # 从噪声开始
        stride = args.T // N
        for t in reversed(range(args.step_offset, args.T+1, stride)):
            noisy_acc = xt
            denoise_t = torch.full((xt.shape[0],), t, device=args.device, dtype=torch.long)
            output = model(
                noisy_acc=noisy_acc, 
                denoise_t=denoise_t,
                ped_length=ped_length_repeat, 
                veh_length=veh_length_repeat,
                timer=timer,
            )  # (S*B, #pedestrian, pred_step, 2)
            if args.predict_noise:
                xt = diffusion.denoise(xt, t, noise=output, stride=min(stride, t))
            else:
                xt = diffusion.denoise(xt, t, x0=output, stride=min(stride, t))
            torch.cuda.synchronize()
            timer.add('Denoise')
        acc_new = xt / args.scale_accelerate
        frame_new = frame + args.pred_step
        vel_new = vel_now.unsqueeze(-2) + acc_new.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
        pos_new = pos_now.unsqueeze(-2) + vel_new.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
        hst_now = torch.cat([hst_now, pos_now.unsqueeze(-2), pos_new], dim=-2)[:, :, -args.hist_step-1:-1, :] # (S*B, #pedestrian, hist_step, 2)
        vel_now = vel_new[:, :, -1, :] # (S*B, #pedestrian, 2)
        pos_now = pos_new[:, :, -1, :] # (S*B, #pedestrian, 2)
        if veh_list:
            veh_new = torch.from_numpy(
                df_veh
                .loc[frame+1:frame+args.pred_step]  # pandas 中的切片是闭区间，因此实际上切出来了 pred_step 帧
                .unstack().swaplevel(axis='columns').sort_index(axis='columns')
                .reindex(columns=veh_list, level=0)
                .reindex(index=range(frame+1, frame+args.pred_step+1))
                .values.reshape(args.hist_step+1, veh.shape[0], 2) # (hist_step + 1, #vehicle, 2)
                .transpose(1, 0, 2) # (#vehicle, hist_step + 1, 2)
            ).to(device=args.device, dtype=torch.float32)
            veh_now = torch.cat([veh_now, veh_new.unsqueeze(0).repeat(S, 1, 1, 1)], dim=-2)[:, :, -args.hist_step-1:, :] # (S*B, #vehicle, hist_step + 1, 2)
        traj.append(pos_new.cpu().numpy())  # list of (S*B, #pedestrian, pred_step, 2)
        timer.add('Prepare Next Step')

    L = args.roll_step * args.pred_step
    traj = np.concatenate(traj, axis=-2)  # (S*B, #pedestrian, roll_step * pred_step, 2)
    traj = traj.reshape(S, len(ped_list), L, 2)  # (S, #pedestrian, roll_step * pred_step, 2)
    _logger.note(tag2ansi(
        f"[pink]Simulation done, sampled {S} times for {L} steps[reset]. "
        f"[bold underline orange]FPS = {timer._count['Prepare Next Step'] / timer.time:.2f} Hz[reset]. "
        f"[#66CCFF]Time Usage[reset]: {timer}"
    ))
    
    # Calculate Accuracy
    records = {}
    df_true = (
        df_ped
        .reindex(pd.MultiIndex.from_product([
            frame_idx + np.arange(L), 
            ped_list
        ], names=['f', 'id'])) # (roll_step * pred_step, #pedestrian, 2)
    )
    true = df_true.values.reshape(L, len(ped_list), 2).transpose(1, 0, 2) # (#pedestrian, roll_step * pred_step, 2)
    dis_err = np.linalg.norm(traj - true[np.newaxis, :, :, :], axis=-1)  # (S, #pedestrian, roll_step * pred_step)
    # 选择 ade 最佳的 sample
    sample_idx = np.nanmean(dis_err, axis=-1).argmin(axis=0)  # (#pedestrian,)
    dis_err = dis_err[sample_idx, np.arange(len(sample_idx)), :]  # (#pedestrian, roll_step * pred_step)
    # 计算 ade, fde
    records['ade'] = np.nanmean(dis_err, axis=-1)
    # records['fde'] = dis_err[..., -1]
    records['fde'] = [row[np.isfinite(row)][-1] for row in dis_err]
    # 计算轨迹长度
    records['trajlen'] = np.nansum(np.linalg.norm(np.diff(true, axis=-2), axis=-1), axis=-1)
    # 统计行人和车辆数量
    records['ped_num'] = len(ped_list)
    records['veh_num'] = len(veh_list)
    # 统计 Rollout 用时
    records['rollout_time'] = timer.time / timer._count['Prepare Next Step']
    records['FPS'] = timer._count['Prepare Next Step'] / timer.time
    # 输出结果
    foo = lambda l: ', '.join(f'{i:.2f}' for i in l)
    _logger.note(tag2ansi(
        f"[green]Rollout Results: [reset]\n"
        f"[#66CCFF]ADE={np.nanmean(records['ade']):.4f}m ({foo(records['ade'])}),\n"
        f"[#66CCFF]FDE={np.nanmean(records['fde']):.4f}m ({foo(records['fde'])}),\n"
        f"[#66CCFF]TrajLen={np.nanmean(records['trajlen']):.4f}m ({foo(records['trajlen'])}),\n"
        f"[bold orange]Accuracy={1 - np.nanmean(records['ade']) / np.nanmean(records['trajlen']):.2%},\n"
        f"[#66CCFF]PedNum={records['ped_num']},\n"
        f"[#66CCFF]VehNum={records['veh_num']},\n"
        f"[#66CCFF]Rollout Time={records['rollout_time']*1000:.2f}ms/step,\n"
        f"[bold orange]FPS={records['FPS']:.2f}Hz."
    ))
    
    # Visualize rollout
    save_file = save_path / f"{args.exp_name}_rollout.png"
    fi, fig, axes = get_fig(1, 1, AW=6, AH=6, dpi=300)
    ax = axes[0]
    for i, pid in enumerate(ped_list):
        color = sns.color_palette("hsv", traj.shape[1])[i]
        for s in range(S):
            ax.plot(traj[s, i, :, 0], traj[s, i, :, 1], alpha=0.5, color=color)
        ax.scatter(pos[i, 0], pos[i, 1], marker='o', color=color, s=10, zorder=10)
        ax.scatter(des[i, 0], des[i, 1], marker='*', color=color, s=10, zorder=10)
        ax.plot(hst[i, :, 0], hst[i, :, 1], color=color, linewidth=1, zorder=10)
        future = df_ped.loc[pd.IndexSlice[frame_idx+1:, pid], :].values
        ax.plot(future[:, 0], future[:, 1], color=color, linestyle='--', linewidth=1, zorder=10)

    ax.set_facecolor('#b2bec3')
    ax.imshow(map_data.map, extent=(map_data.xmin, map_data.xmax, map_data.ymin, map_data.ymax), cmap='gray_r', zorder=0, origin='lower')
    ax.set_title(f"Rollout for {args.roll_step * args.pred_step} steps")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    fig.savefig(save_file, bbox_inches='tight')
    plt.close()
    _logger.info(tag2ansi(f"Rollout figure saved to [green]{save_file}[reset]"))

    _logger.note(f"Sampling finished. Re-run: {args.command}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default="sample")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument('--sampling_method', type=str, default="DDIM", choices=['DDPM', 'DDIM'])
    parser.add_argument("--T", type=int, default=100, help="训练时的扩散步数")
    parser.add_argument('--sample_num', type=int, default=10, help="测试时每个轨迹采样 {sample_num} 次")
    parser.add_argument('--denoise_step', type=int, default=2, help="采样时进行 {denoise_step} 次去噪")
    parser.add_argument('--step_offset', type=int, default=10, help="最后一步去噪从 x_{step_offset} 到 x_0")
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
    parser.add_argument('--model_dim', type=int, default=64)
    parser.add_argument('--map_feature_dim', type=int, default=64)
    parser.add_argument('--head_num', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--attention_layer_num', type=int, default=1)
    parser.add_argument('--lstm_layer_num', type=int, default=1)
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
