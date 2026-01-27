import re
import sys
import json
import torch
import shlex
import random
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
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
from src.utils.fix_parser import add_negation_flags, add_minus_flags
from src.utils.use_npu import USE_NPU, npu_attention_fallback_context
from src.tasks import init_simulation, simulate_one_step

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
    ## Load Model
    if args.use_new_model:
        model = NewModel(args).to(args.device)
    elif args.use_relative_model:
        model = RelativeModel(args).to(args.device)
    else:
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
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    start_epoch = checkpoint["epoch"] + 1
    model.load_state_dict(checkpoint["model"])
    model.eval()
    torch.set_grad_enabled(False)
    _logger.info(f"Checkpoint loaded from {checkpoint_path}, resume from epoch {start_epoch}.")
    timer.add('Load Model')

    ## Prepare data
    dataset = UCYDataset.load_data(args, "./data/UCY/data/data_zara/crowds_zara01.vsp")
    frame_idx = 10
    state = init_simulation(args, dataset, frame_idx, model)
    initial_state = deepcopy(state)
    init = initial_state.df_ped.loc[pd.MultiIndex.from_product([[10], initial_state.ped_list])].reset_index()
    init = pd.concat([init.assign(sample=s) for s in range(args.sample_num)], axis=0)
    df_sim = [init]

    timer.clear(reset=True)
    for frame in range(frame_idx, frame_idx+args.roll_step, args.pred_step):
        df_new, state = simulate_one_step(args, model, diffusion, state)
        df_sim.append(df_new)
        timer.add('Prepare Next Step')

    S = args.sample_num
    L = args.roll_step * args.pred_step
    df_sim = pd.concat(df_sim, axis=0)  # (S*B, #pedestrian, roll_step * pred_step, 2)
    _logger.note(tag2ansi(
        f"[pink]Simulation done, sampled {S} times for {L} steps[reset]. "
        # f"[bold underline orange]FPS = {timer._count['Prepare Next Step'] / timer.time:.2f} Hz[reset]. "
        f"[#66CCFF]Time Usage[reset]: {timer}"
    ))
    
    # Calculate Accuracy
    records = {}
    df_true = (
        state.df_ped
        .reindex(pd.MultiIndex.from_product([
            frame_idx + np.arange(L), 
            state.ped_list
        ], names=['f', 'id'])) # (roll_step * pred_step, #pedestrian, 2)
    )
    true = df_true.values.reshape(L, len(state.ped_list), 2).transpose(1, 0, 2) # (#pedestrian, roll_step * pred_step, 2)
    df_pred = (
        df_sim
        .set_index(['sample', 'f', 'id'])
        [['x', 'y']]
        .reindex(pd.MultiIndex.from_product([
            np.arange(S),
            frame_idx + np.arange(L), 
            state.ped_list
        ], names=['sample', 'f', 'id'])) # (roll_step * pred_step, #pedestrian, 2)
    )
    pred = df_pred.values.reshape(S, L, len(state.ped_list), 2).transpose(0, 2, 1, 3)  # (S, #pedestrian, roll_step * pred_step, 2)

    dis_err = np.linalg.norm(pred - true[np.newaxis, :, :, :], axis=-1)  # (S, #pedestrian, roll_step * pred_step)
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
    records['ped_num'] = len(state.ped_list)
    records['veh_num'] = len(state.veh_list)
    # 统计 Rollout 用时
    records['rollout_time'] = timer.time / timer._count['Prepare Next Step']
    records['FPS'] = timer._count['Prepare Next Step'] / timer.time
    # 输出结果
    foo = lambda l: ', '.join(f'{i:.2f}' for i in l)
    _logger.note(tag2ansi(
        f"[green]Rollout Results: [reset]\n"
        f"[#66CCFF]ADE={np.nanmean(records['ade']):.4f}m,\n"# ({foo(records['ade'])}),\n"
        f"[#66CCFF]FDE={np.nanmean(records['fde']):.4f}m,\n"# ({foo(records['fde'])}),\n"
        f"[#66CCFF]TrajLen={np.nanmean(records['trajlen']):.4f}m,\n"# ({foo(records['trajlen'])}),\n"
        f"[bold orange]Accuracy={1 - np.nanmean(records['ade']) / np.nanmean(records['trajlen']):.2%},\n"
        f"[#66CCFF]PedNum={records['ped_num']},\n"
        f"[#66CCFF]VehNum={records['veh_num']},\n"
        f"[#66CCFF]Rollout Time={records['rollout_time']*1000:.2f}ms/step,\n"
        f"[bold orange]FPS={records['FPS']:.2f}Hz."
    ))

    # Visualize rollout
    if False:
        save_file = save_path / f"{args.exp_name}_rollout.png"
        fig = visualize_png(args, initial_state, state, pred, frame_idx)
        fig.savefig(save_file, bbox_inches='tight')
    else:
        save_file = save_path / f"{args.exp_name}_rollout.gif"
        fig, ani = visualize_gif(args, initial_state, state, pred, frame_idx)
        ani.save(save_file, writer='pillow', fps=10)
    _logger.info(tag2ansi(f"Rollout figure saved to [green]{save_file}[reset]"))

    _logger.note(f"Sampling finished. Re-run: {args.command}")


def visualize_png(args, initial_state, state, pred, frame_idx):
    fi, fig, axes = get_fig(1, 1, AW=6, AH=6, dpi=300)
    ax = axes[0]
    pos = initial_state.pos_now[0].cpu().numpy()
    des = initial_state.des_now[0].cpu().numpy()
    hst = initial_state.hst_now[0].cpu().numpy()
    for i, pid in enumerate(state.ped_list):
        color = sns.color_palette("hsv", pred.shape[1])[i]
        for s in range(args.sample_num):
            ax.plot(pred[s, i, :, 0], pred[s, i, :, 1], alpha=0.5, color=color)
        ax.scatter(pos[i, 0], pos[i, 1], marker='o', color=color, s=10, zorder=10)
        ax.scatter(des[i, 0], des[i, 1], marker='*', color=color, s=10, zorder=10)
        ax.plot(hst[i, :, 0], hst[i, :, 1], color=color, linewidth=1, zorder=10)
        future = state.df_ped.loc[pd.IndexSlice[frame_idx+1:, pid], :].values
        ax.plot(future[:, 0], future[:, 1], color=color, linestyle='--', linewidth=1, zorder=10)

    veh = initial_state.veh_now[0].cpu().numpy()
    for i, vid in enumerate(state.veh_list):
        ax.plot(veh[i, :, 0], veh[i, :, 1], color='gray', linewidth=1, zorder=10)
        future = state.df_veh.loc[pd.IndexSlice[frame_idx+1:, vid], :].values
        ax.plot(future[:, 0], future[:, 1], color='blue', linestyle='--', linewidth=1, zorder=10)

    ax.set_facecolor('#b2bec3')
    ax.imshow(state.map_data.map.T, extent=(state.map_data.xmin, state.map_data.xmax, state.map_data.ymin, state.map_data.ymax), cmap='gray_r', zorder=0, origin='lower')
    ax.set_title(f"Rollout for {args.roll_step * args.pred_step} steps")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    return fig


def visualize_gif(args, initial_state, state, pred, frame_idx):
    import matplotlib.animation as animation

    save_file = save_path / f"{args.exp_name}_rollout.gif"
    fi, fig, axes = get_fig(1, 1, AW=6, AH=6, dpi=300)
    ax = axes[0]

    # 设置背景颜色和地图
    ax.set_facecolor('#b2bec3')
    ax.imshow(state.map_data.map.T, extent=(state.map_data.xmin, state.map_data.xmax, state.map_data.ymin, state.map_data.ymax), cmap='gray_r', zorder=0, origin='lower')

    # 绘制车辆 (全静态)
    veh = initial_state.veh_now[0].cpu().numpy()
    for i, vid in enumerate(state.veh_list):
        # 车辆历史
        ax.plot(veh[i, :, 0], veh[i, :, 1], color='gray', linewidth=1, zorder=10)
        # 车辆真实未来
        future = state.df_veh.loc[pd.IndexSlice[frame_idx+1:, vid], :].values
        ax.plot(future[:, 0], future[:, 1], color='blue', linestyle='--', linewidth=1, zorder=10)

    # 绘制行人静态部分 (历史轨迹, 真实未来轨迹, 目的地)
    pos = initial_state.pos_now[0].cpu().numpy()
    des = initial_state.des_now[0].cpu().numpy()
    hst = initial_state.hst_now[0].cpu().numpy()

    # 存储动态对象的容器
    dynamic_lines = []  # 存储预测轨迹线对象
    dynamic_dots = []   # 存储当前位置点对象

    for i, pid in enumerate(state.ped_list):
        color = sns.color_palette("hsv", pred.shape[1])[i]
        
        # Static: 目的地 (X / Star)
        ax.scatter(des[i, 0], des[i, 1], marker='*', color=color, s=20, zorder=10, label='Dest' if i==0 else "")
        
        # Static: 历史轨迹
        ax.plot(hst[i, :, 0], hst[i, :, 1], color=color, linewidth=1, zorder=10)
        
        # Static: 真实未来轨迹 (GT)
        future = state.df_ped.loc[pd.IndexSlice[frame_idx+1:, pid], :].values
        ax.plot(future[:, 0], future[:, 1], color=color, linestyle='--', linewidth=1, zorder=10)

        # ------------------------------------------------------
        # 2. 初始化动态对象 (预测轨迹线 + 移动的点)
        # ------------------------------------------------------
        # 每个行人有 args.sample_num 条预测轨迹
        ped_lines = []
        ped_dots = []
        
        for s in range(args.sample_num):
            # 初始化线：一开始是空的或者只有起点
            line, = ax.plot([], [], alpha=0.5, color=color, linewidth=1.5)
            ped_lines.append(line)
            
            # 初始化点：表示轨迹的"头" (当前推进的位置)
            # 你的原始代码中 pos[i] 是 t=0 的位置。
            dot = ax.scatter([], [], marker='o', color=color, s=15, zorder=11)
            ped_dots.append(dot)
        
        dynamic_lines.append(ped_lines)
        dynamic_dots.append(ped_dots)

    # 设置轴标签和网格
    ax.set_title(f"Rollout Animation ({args.roll_step * args.pred_step} steps)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)

    # ------------------------------------------------------------------
    # 3. 动画更新函数
    # ------------------------------------------------------------------
    # pred shape 假设为: [sample_num, num_peds, time_steps, 2]
    total_steps = pred.shape[2]

    def update(frame):
        # frame 从 0 到 total_steps-1
        # 每一帧，我们画出从 0 到 frame 的轨迹，并将点移动到 frame 的位置
        
        for i in range(len(state.ped_list)): # 遍历行人
            for s in range(args.sample_num): # 遍历 Sample
                # 获取该 sample 该行人的完整轨迹数据
                # 数据切片：取直到当前 frame 的所有点
                current_x_path = pred[s, i, :frame+1, 0]
                current_y_path = pred[s, i, :frame+1, 1]
                
                # 更新轨迹线 (随着时间变长)
                dynamic_lines[i][s].set_data(current_x_path, current_y_path)
                
                # 更新点的位置 (位于轨迹的最前端)
                # scatter 的 set_offsets 需要一个 (N, 2) 的数组
                current_pos = np.c_[pred[s, i, frame, 0], pred[s, i, frame, 1]]
                dynamic_dots[i][s].set_offsets(current_pos)
                
        return [item for sublist in dynamic_lines for item in sublist] + \
            [item for sublist in dynamic_dots for item in sublist]

    # ------------------------------------------------------------------
    # 4. 生成并保存 GIF
    # ------------------------------------------------------------------
    # 数据是 2.5Hz (即每个点间隔 0.4秒)。
    # 为了加速效果，我们设置 fps=10 (即每秒播放10帧数据)，这意味着 4倍速 播放。
    # 也可以根据喜好调整 fps。
    ani = animation.FuncAnimation(fig, update, frames=total_steps, blit=True)
    return fig, ani


if __name__ == "__main__":
    parser = ArgumentParser()
    # 基础配置
    parser.add_argument("--name", type=str, default="sample", help="实验任务名称，用于生成实验ID")
    parser.add_argument("--exp_name", type=str, default=None, help="手动指定实验名称（若指定则覆盖自动生成的名称）")
    parser.add_argument("--device", type=str, default="auto", help="计算设备，可选 'cpu', 'cuda:0' 或 'auto'（自动选择显存充足的 GPU）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，固定以复现实验结果")
    parser.add_argument("--save_dir", type=str, default="./logs/sample", help="日志和模型权重的保存根目录")
    parser.add_argument("--debug", action="store_true", help="是否开启调试模式（输出更多日志，不保存部分文件）")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader 的工作线程数（0 表示主线程）")
    
    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=128, help="训练批次大小")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率 (Learning Rate)")
    parser.add_argument("--epochs", type=int, default=10000, help="最大训练轮数")
    parser.add_argument('--patience', type=int, default=20, help="Early Stopping 的耐心值（多少个 epoch 验证集指标不提升则停止）")
    parser.add_argument('--loss_type', type=str, default='noise', choices=['position', 'accelerate', 'noise'], help="损失函数计算的目标类型")
    parser.add_argument('--reload_checkpoint', type=str, required=True, help="用于测试的 checkpoint（.pth 文件）")
    parser.add_argument('--required_memory_MB', type=int, default=6000, help="自动选择 GPU 时要求的最小剩余显存 (MB)")

    # 扩散模型参数 (Diffusion)
    parser.add_argument('--sampling_method', type=str, default="DDIM", choices=['DDPM', 'DDIM'], help="采样/生成方法")
    parser.add_argument("--T", type=int, default=100, help="训练时的最大扩散步数 (Timesteps)")
    parser.add_argument('--sample_num', type=int, default=20, help="测试推理时，为每个轨迹生成的样本数量（用于评估多样性和准确性）")
    parser.add_argument('--denoise_step', type=int, default=2, help="DDIM 采样时的去噪步数（加速采样）")
    parser.add_argument('--step_offset', type=int, default=10, help="采样的起始时间步偏移量（从 T-offset 开始采样）")
    parser.add_argument('--antithetic_sampling', action='store_true', default=True, help="是否使用对偶采样以减少方差")
    parser.add_argument('--beta_schedule', type=str, default='linear', choices=['linear', 'cosine'], help="噪声调度表类型")
    parser.add_argument('--predict_noise', action='store_true', default=True, help="模型是否预测噪声（True预测epsilon, False预测x0）")
    parser.add_argument('--rollout_lambda', type=float, default=1.0, help="多帧 Rollout 损失的时间衰减系数")
    parser.add_argument('--multi_frame_rollout', type=int, default=1, help="训练时单次迭代预测未来的帧数（Rollout 步数）")
    parser.add_argument('--scale_accelerate', type=float, default=1.0, help="加速度数据的缩放因子（用于稳定训练）")

    # 数据增强与Dropout
    parser.add_argument('--p_drop_map', type=float, default=None, help="以一定概率丢弃地图信息（实现无地图引导的生成）")
    parser.add_argument('--p_drop_destination', type=float, default=None, help="以一定概率丢弃目的地条件（实现无目标引导的生成）")
    parser.add_argument('--p_drop_speed', type=float, default=None, help="以一定概率丢弃初始速度条件")
    parser.add_argument('--dropout', type=float, default=0.5, help="模型中的 Dropout 比率")

    # 数据集配置
    parser.add_argument('--datasets', type=str, default=["ETH"], nargs='*', choices=['ETH', 'UCY', 'GC', 'SDD', 'WayMo', 'ORCA', 'All', 'debug'], help="使用的训练数据集列表")
    parser.add_argument("--hist_step", type=int, default=8, help="输入的历史轨迹长度（帧数）")
    parser.add_argument("--pred_step", type=int, default=1, help="单步预测的未来轨迹长度（帧数，通常配合 Rollout 使用）")
    parser.add_argument("--skip_step", type=int, default=1, help="数据采样的滑动窗口步长")
    parser.add_argument("--roll_step", type=int, default=12, help="测试/验证时需要预测的总未来帧数")
    parser.add_argument("--fps", type=int, default=2.5, help="数据重采样后的目标帧率 (Hz)")
    parser.add_argument("--dot_per_meter", type=int, default=1, help="栅格化地图的分辨率（每米对应的像素点数）")
    parser.add_argument('--test_name', type=str, default=None, nargs='+', help="指定作为测试集的场景名称（substring匹配）")
    parser.add_argument('--test_ratio', type=float, default=None, help="自动划分测试集的比例 (0.0 ~ 1.0)")
    parser.add_argument('--split_by_scenario', action='store_true', help="是否按场景划分训练/测试集（否则按轨迹样本划分）")
    parser.add_argument('--cache_dataset', action='store_true', default=True, help="是否缓存预处理后的数据集以加速加载")
    parser.add_argument('--collision_threshold', type=float, default=0.6, help="碰撞检测的距离阈值（单位：米）")

    # 消融实验/条件控制
    parser.add_argument('--no_destination', action='store_true', default=False, help="[消融] 强制不使用目的地条件进行生成")
    parser.add_argument('--no_speed', action='store_true', default=False, help="[消融] 强制不使用初始速度条件进行生成")

    # 模型结构参数
    parser.add_argument('--model_dim', type=int, default=64, help="模型的隐藏层维度 (Hidden Dimension)")
    parser.add_argument('--map_feature_dim', type=int, default=64, help="地图特征提取网络的输出维度")
    parser.add_argument('--head_num', type=int, default=4, help="Transformer 注意力头的数量")
    parser.add_argument('--attention_layer_num', type=int, default=1, help="Transformer 层的堆叠数量")
    parser.add_argument('--lstm_layer_num', type=int, default=1, help="LSTM 层的堆叠数量")
    parser.add_argument('--latent_token_num', type=int, default=16, help="地图特征的 Latent Token 数量")
    parser.add_argument('--use_relative_model', action='store_true', default=True, help="是否使用相对坐标模型结构")
    parser.add_argument('--use_spatial_anchor', action='store_true', default=True, help="是否使用空间锚点增强位置编码")
    parser.add_argument('--use_new_model', action='store_true', default=False, help="是否使用改进版的新模型结构")

    # 条件引导参数
    parser.add_argument('--cfg_des', type=float, default=None, help="通过 Classifier-Free Guidance 引导控制目的地条件的影响强度")
    parser.add_argument('--cfg_map', type=float, default=None, help="通过 Classifier-Free Guidance 引导控制地图条件的影响强度")
    parser.add_argument('--cg_dir', type=float, default=None, help="通过 Classifier Guidance 引导控制向目的地前进的影响强度")
    parser.add_argument('--cg_dis', type=float, default=None, help="通过 Classifier Guidance 引导控制与目的地距离的影响强度")
    parser.add_argument('--cg_sfm_des', type=float, default=None, help="通过 Classifier Guidance 引导控制社会力目标引导条件的影响强度")
    parser.add_argument('--cg_sfm_obs', type=float, default=None, help="通过 Classifier Guidance 引导控制社会力地图排斥条件的影响强度")
    parser.add_argument('--cg_sfm_soc', type=float, default=None, help="通过 Classifier Guidance 引导控制社会力社交排斥条件的影响强度")
    parser.add_argument('--sfm_t_des', type=float, default=0.5, help="社会力中目标引导力的弛豫时间")
    parser.add_argument('--sfm_a_ped', type=float, default=25, help="社会力中行人排斥力的强度系数")
    parser.add_argument('--sfm_a_veh', type=float, default=30, help="社会力中车辆排斥力的强度系数")
    parser.add_argument('--sfm_a_map', type=float, default=30, help="社会力中地图排斥力的强度系数")
    parser.add_argument('--sfm_b_ped', type=float, default=0.08, help="社会力中行人排斥力的衰减系数")
    parser.add_argument('--sfm_b_veh', type=float, default=0.10, help="社会力中车辆排斥力的衰减系数")
    parser.add_argument('--sfm_b_map', type=float, default=0.10, help="社会力中地图排斥力的衰减系数")
    parser.add_argument('--sfm_r_map', type=int, default=10, help="社会力中地图排斥力距离阈值 (in pixel)")
    parser.add_argument('--sfm_a_damp', type=float, default=0.5, help="社会力中速度阻尼系数（用于计算引导力时的速度衰减）")
    parser.add_argument('--use_sfm', action='store_true', default=False, help="使用社会力模型代替神经网络计算引导力")

    parser = add_minus_flags(parser) ## --key_name -> --key-name
    parser = add_negation_flags(parser) ## --action-as-true -> --no-action-as-true
    args, unknown = parser.parse_known_args()

    ## Build Save Path
    if args.exp_name is None:
        now = datetime.now()
        date = now.strftime("%Y%m%d")
        curr = now.strftime("%H%M%S")
        host = gethostname()
        exp_name = f'{date}_{args.name}_{curr}_{host}'
        exp_name = re.compile(r'[ <>:"/\\|?*\x00-\x1f]').sub('_', exp_name.strip())
        exp_name = exp_name or 'unnamed'
        exp_name = exp_name[:255] # Max filename length on most filesystems
        args.exp_name = exp_name
    save_path = Path(args.save_dir) / args.exp_name
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        _logger.warning(f"Save path {save_path} already exists.")
    args.save_path = str(save_path)

    ## Init Logger
    init_logger(
        "src",
        exp_name=args.exp_name,
        log_file=save_path / "info.log",
        info_level="debug" if args.debug else "info",
    )

    ## Warm Unknown Args
    if unknown:
        _logger.warning(f"Unknown args: {unknown}")

    ## Set Seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    seed_all(args.seed)
    ## Set Command
    args.command = ' '.join(map(shlex.quote, [sys.executable, *sys.argv]))
    ## Select GPU
    if args.device == "auto":
        args.device = AutoGPU().choice_gpu(memory_MB=args.required_memory_MB, interval=15) if not USE_NPU else 'npu'

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
