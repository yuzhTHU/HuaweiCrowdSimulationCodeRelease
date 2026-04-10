import torch
import logging
import numpy as np
import pandas as pd
import torch.nn.functional as F
from typing import List, Tuple
from argparse import Namespace
from dataclasses import dataclass
from ..model.model import Model
from ..diffusion import DDPM
from ..dataset import BaseDataset
from ..utils.get_force_map import get_force_map
from ..utils.extract_patches import extract_patches_torch
from .guidance import guidance

_logger = logging.getLogger('src.simulate')

@dataclass
class SimulateState:
    df_ped: pd.DataFrame  # 行人数据 DataFrame
    df_veh: pd.DataFrame # 车辆数据 DataFrame
    map_data: object # 地图数据对象
    ped_list: List[int] # 行人 ID 列表 (#pedestrian,)
    veh_list: List[int] # 车辆 ID 列表 (#vehicle,)
    frame: int  # 当前帧
    pos_now: torch.Tensor # 当前行人位置 (Batch, #pedestrian, 2)
    vel_now: torch.Tensor # 当前行人速度 (Batch, #pedestrian, 2)
    hst_now: torch.Tensor # 当前行人历史位置 (Batch, #pedestrian, hist_step, 2)
    des_now: torch.Tensor  # 当前行人目的地 (Batch, #pedestrian, 2)
    spd_now: torch.Tensor  # 当前行人期望速度 (Batch, #pedestrian, 1)
    veh_now: torch.Tensor  # 当前车辆位置 (Batch, #vehicle, hist_step + 1, 2)


def init_simulation(
    args: Namespace, dataset: BaseDataset, frame_idx: int, model: Model
) -> SimulateState:
    """
    初始化模拟所需的数据

    Args:
        args: 配置参数
        dataset: 数据集对象
        frame_idx: 初始帧
        model: 模型对象

    Returns:
        SimulateState: 包含初始化状态的对象
    """
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
    assert pos.shape == (len(ped_list), 2)
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
    assert vel.shape == (len(ped_list), 2)
    hst = (
        df_ped
        .reindex(pd.MultiIndex.from_product([
            range(frame_idx-args.hist_step, frame_idx), 
            ped_list
        ], names=['f', 'id']))
        .values.reshape(args.hist_step, len(ped_list), 2) # (hist_step, #pedestrian, 2)
        .transpose(1, 0, 2) # (#pedestrian, hist_step, 2)
    )
    assert hst.shape == (len(ped_list), args.hist_step, 2)
    veh = (
        df_veh
        .reindex(pd.MultiIndex.from_product([
            range(frame_idx-args.hist_step, frame_idx + 1), 
            veh_list
        ], names=['f', 'id']))
        .values.reshape(args.hist_step+1, len(veh_list), 2) # (#vehicle, hist_step + 1, 2)
        .transpose(1, 0, 2) # (#vehicle, hist_step + 1, 2)
    )
    assert veh.shape == (len(veh_list), args.hist_step + 1, 2)
    des = (
        df_ped
        .loc[df_ped.index.get_level_values('id').isin(ped_list)]
        .groupby(level=1, sort=False).tail(1)
        .swaplevel(axis=0).reindex(index=ped_list, level=0)
        .values # (#pedestrian, 2)
    )
    assert des.shape == (len(ped_list), 2)

    # 使用用户自定义的目的地覆盖（如果有）
    # 注意：user_destinations 的键是字符串类型，需要将 ped_id 转换为字符串进行比较
    if hasattr(dataset, 'user_destinations') and dataset.user_destinations:
        for idx, ped_id in enumerate(ped_list):
            ped_id_str = str(ped_id)
            if ped_id_str in dataset.user_destinations:
                user_des = dataset.user_destinations[ped_id_str]
                des[idx, 0] = user_des['x']
                des[idx, 1] = user_des['y']
                _logger.info(f"Using user-defined destination for pedestrian {ped_id}: ({user_des['x']}, {user_des['y']})")
    # 计算期望速度 spd
    # 注意：模拟数据集可能被截断，没有足够的未来帧数据
    # 此时使用当前速度作为期望速度的估计
    spd_data_range = range(frame_idx, frame_idx + int(5 * args.fps) + 1)
    available_frames = df_ped.index.get_level_values('f').unique()
    has_future_data = any(f in available_frames for f in spd_data_range if f > frame_idx)

    if has_future_data:
        # 有足够的未来帧数据，正常计算
        spd = (
            df_ped
            .reindex(pd.MultiIndex.from_product([
                spd_data_range,
                ped_list
            ], names=['f', 'id']))
            .unstack().ffill().bfill().diff().mul(args.fps).iloc[1:]
            .stack(future_stack=True)
            .pow(2).sum(axis='columns', min_count=2).pow(0.5)
            .unstack().mean(axis='rows')
            .values[:, np.newaxis]
        )
    else:
        # 模拟数据集被截断，使用当前速度作为期望速度
        _logger.info(f"No future data for spd calculation, using current velocity as estimated desired speed")
        spd = np.linalg.norm(vel, axis=1, keepdims=True)
        # 设置合理的默认期望速度（如果当前速度太小）
        default_spd = 1.0
        spd = np.where(spd < default_spd, default_spd, spd)
    assert spd.shape == (len(ped_list), 1), "spd shape mismatch"
    map_data = dataset.map_data
    if args.no_destination: 
        des *= np.nan
    if args.no_speed: 
        spd *= np.nan

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
    model.set_map_embedding(
        map=torch.from_numpy(map_data.map).to(device=args.device, dtype=torch.float32),
        xmin=map_data.xmin,
        xmax=map_data.xmax,
        ymin=map_data.ymin,
        ymax=map_data.ymax,
    )
    frame = frame_idx
    return SimulateState(
        df_ped=df_ped,
        df_veh=df_veh,
        map_data=map_data,
        ped_list=ped_list,
        veh_list=veh_list,
        frame=frame,
        pos_now=pos_now,
        vel_now=vel_now,
        hst_now=hst_now,
        des_now=des_now,
        spd_now=spd_now,
        veh_now=veh_now,
    )


def simulate_one_step(
    args: Namespace, model: Model, diffusion: DDPM, state: SimulateState
) -> Tuple[pd.DataFrame, SimulateState]:
    model.eval()
    torch.set_grad_enabled(False)

    # _logger.info(f"Simulating from frame {frame} to frame {frame + args.pred_step}...")
    S = args.sample_num  # 采样次数
    N = args.denoise_step  # 采样步数
    ped_length_repeat = torch.full((S, ), len(state.ped_list), device=args.device, dtype=torch.long)  # (S,)
    veh_length_repeat = torch.full((S, ), len(state.veh_list), device=args.device, dtype=torch.long)  # (S,)

    # _logger.info(f"  Starting setting vehicle embeddings...")
    model.set_veh_embedding(veh=state.veh_now)
    # _logger.info(f"  Starting setting pedestrian embeddings...")
    model.set_ped_embedding(pos=state.pos_now, vel=state.vel_now, hst=state.hst_now, des=state.des_now, spd=state.spd_now)
    # _logger.info(f"  Starting setting surrounding info...")
    model.set_sur_info()

    shape = [S, len(state.ped_list), args.pred_step, 2]  # (S*1, #pedestrian, pred_step, 2)
    xt = torch.randn(shape, device=args.device)  # 从噪声开始
    stride = args.T // N
    for t in reversed(range(args.step_offset, args.T+1, stride)):
        # _logger.info(f"  Denoising step at t={t}...")
        noisy_acc = xt
        denoise_t = torch.full((xt.shape[0],), t, device=args.device, dtype=torch.long)
        output = model(
            noisy_acc=noisy_acc, 
            denoise_t=denoise_t,
            ped_length=ped_length_repeat, 
            veh_length=veh_length_repeat,
        )  # (S*B, #pedestrian, pred_step, 2)

        ## 获取估计的 x0
        x0 = diffusion.noise_to_x0(xt, denoise_t, output) if args.predict_noise else output

        ## 尝试各种引导，各种 guidance 应为 0~1 左右的系数
        x0 = x0 + guidance(args, x0, state, model, diffusion, noisy_acc, xt, denoise_t, ped_length_repeat, veh_length_repeat)

        ## 去噪
        xt = diffusion.denoise(xt, t, x0=x0, stride=min(stride, t))
    acc_new = xt / args.scale_accelerate
    
    if args.use_sfm:
        future_vel = state.vel_now.unsqueeze(-2)
        future_pos = state.pos_now.unsqueeze(-2)
        # 目的地导向力
        desire_vel = F.normalize(state.des_now.unsqueeze(-2) - future_pos, dim=-1) * state.spd_now.unsqueeze(-2)  # (S*B, #pedestrian, pred_step, 2)
        des_force = (desire_vel - future_vel).nan_to_num(0.0) / args.sfm_t_des  # (S*B, #pedestrian, pred_step, 2)
        # 场景障碍物排斥力
        F_map = get_force_map(r=args.sfm_r_map, A=args.sfm_a_map, B=args.sfm_b_map, device=args.device)  # (2r+1, 2r+1, 2)
        idx = future_pos[..., 0].sub(model.xmin).div(model.xmax - model.xmin).mul(model.map.shape[0]).round().long().clamp(0, model.map.shape[0] - 1)  # (S*B, #pedestrian, pred_step)
        jdx = future_pos[..., 1].sub(model.ymin).div(model.ymax - model.ymin).mul(model.map.shape[1]).round().long().clamp(0, model.map.shape[1] - 1)  # (S*B, #pedestrian, pred_step)
        patches = extract_patches_torch(model.map, idx.reshape(-1), jdx.reshape(-1), r=args.sfm_r_map).reshape(*idx.shape, 2*args.sfm_r_map+1, 2*args.sfm_r_map+1) # (S*B, #pedestrian, pred_step, 2r+1, 2r+1)
        map_force = (patches[..., None] * F_map).nan_to_num(0.0).flatten(-3, -2).sum(-2) # (S*B, #pedestrian, pred_step, 2)
        # 其他行人排斥力
        p = future_pos[:, None, :, :, :] - future_pos[:, :, None, :, :] # (S*B, #focal-pedestrian, #other-pedestrian, pred_step, 2)
        d = torch.norm(p, dim=-1, keepdim=True)
        n = -p / d.clamp(min=1e-6)
        F_ped = args.sfm_a_ped * torch.exp(-d / args.sfm_b_ped) * n
        ped_force = F_ped.nan_to_num(0.0).sum(dim=1) # (S*B, #pedestrian, pred_step, 2)
        # 其它车辆排斥力 (车辆用最后一帧位置)
        p = state.veh_now[:, :, None, -1:, :] - future_pos[:, None, :, :, :] # (S*B, #vehicle, #pedestrian, pred_step, 2)
        d = torch.norm(p, dim=-1, keepdim=True)
        n = -p / d.clamp(min=1e-6)
        F_veh = args.sfm_a_veh * torch.exp(-d / args.sfm_b_veh) * n
        veh_force = F_veh.nan_to_num(0.0).sum(dim=1) # (S*B, #pedestrian, pred_step, 2)
        # 阻尼力
        damp_force = -args.sfm_a_damp * future_vel  # (S*B, #pedestrian, pred_step, 2)
        # 合力
        acc_new = des_force # + map_force + ped_force + veh_force + damp_force

    vel_new = state.vel_now.unsqueeze(-2) + acc_new.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
    # 将到达目的地的行人速度设置为 0
    if args.threshold_of_arrive > 0:
        _pos_new = state.pos_now.unsqueeze(-2) + vel_new.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
        arrived = (_pos_new - state.des_now[:, :, None, :]).norm(dim=-1) < args.threshold_of_arrive  # (S*B, #pedestrian, pred_step)
        arrived = arrived.cumsum(dim=-1) > 0  # (S*B, #pedestrian, pred_step)
        vel_new[arrived, :] = 0.0
    pos_new = state.pos_now.unsqueeze(-2) + vel_new.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
    if state.veh_list:
        veh_new = torch.from_numpy(
            state.df_veh
            .loc[state.frame+1:state.frame+args.pred_step]  # pandas 中的切片是闭区间，因此实际上切出来了 pred_step 帧
            .unstack().swaplevel(axis='columns').sort_index(axis='columns')
            .reindex(
                index=range(state.frame+1, state.frame+args.pred_step+1),
                columns=pd.MultiIndex.from_product([state.veh_list, ['x', 'y']])
            )
            .values.reshape(args.pred_step, len(state.veh_list), 2) # (pred_step, #vehicle, 2)
            .transpose(1, 0, 2) # (#vehicle, pred_step, 2)
        ).to(device=args.device, dtype=torch.float32)
    des_new = state.des_now  # (S*B, #pedestrian, 2)
    spd_new = state.spd_now  # (S*B, #pedestrian, 1)
    # _logger.info(f"  Computed new positions and velocities.")

    state.hst_now = torch.cat([state.hst_now, state.pos_now.unsqueeze(-2), pos_new], dim=-2)[:, :, -args.hist_step-1:-1, :] # (S*B, #pedestrian, hist_step, 2)
    if state.veh_list:
        state.veh_now = torch.cat([state.veh_now, veh_new.unsqueeze(0).repeat(S, 1, 1, 1)], dim=-2)[:, :, -args.hist_step-1:, :] # (S*B, #vehicle, hist_step + 1, 2)
    state.pos_now = pos_new[:, :, -1, :] # (S*B, #pedestrian, 2)
    state.vel_now = vel_new[:, :, -1, :] # (S*B, #pedestrian, 2)
    state.spd_now = spd_new  # (S*B, #pedestrian, 1)
    state.des_now = des_new  # (S*B, #pedestrian, 2)
    # _logger.info(f"  Updated states for next step.")

    # _logger.info(f"  Converting new positions to CPU numpy. {pos_new.shape}, {type(pos_new)}")
    df_ped_new = pd.DataFrame([
        {
            'f': state.frame + 1 + f,
            'id': state.ped_list[i],
            'type': 'pedestrian',
            'x': float(pos_new[s, i, f, 0]),
            'y': float(pos_new[s, i, f, 1]),
            'sample': s,
        }
        for i in range(pos_new.shape[1])
        for f in range(pos_new.shape[2])
        for s in range(pos_new.shape[0])
    ])
    # _logger.info(f"  Converted new positions to CPU numpy. {pos_new.shape}, {type(pos_new)}")
    df_veh_orig = state.df_veh.loc[state.frame+1:state.frame+args.pred_step].reset_index().assign(type='vehicle')
    df_veh_new = pd.concat([
        df_veh_orig.assign(sample=s) for s in range(pos_new.shape[0])
    ], ignore_index=True)
    df_new = pd.concat([df_ped_new, df_veh_new], ignore_index=True)
    df_new['sample'] = df_new['sample'].astype(int)
    state.frame = state.frame + args.pred_step
    # _logger.info(f"Completed simulation up to frame {frame}.")

    return df_new, state