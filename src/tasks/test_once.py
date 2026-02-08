import time
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.utils.data as D
from tqdm import tqdm
from typing import List, Dict
from ..model import Model
from ..diffusion import DDPM
from ..utils.timer import NamedTimer
from ..utils.tag2ansi import tag2ansi
from ..utils.calc_xy_error import calc_xy_error
from .visualize import visualize
from .guidance import guidance
from .simulate import SimulateState

_logger = logging.getLogger(__name__)


def get_xy_error(args, pos, veh, mask, pos_true, pos_pred, sample_idx, valid_idx):
    traj_diff = (pos_pred - pos_true)[:, mask, :, :][sample_idx, valid_idx, :, :] # (valid{B*#pedestrian}, roll_step*pred_step, 2)
    ped_pos = pos[mask, :] # (valid{B*#pedestrian}, 2)
    batch_indices = torch.nonzero(mask)[:, 0]  # 有效行人所属的 batch index (valid{B*#pedestrian},)
    num_valid_ped = batch_indices.shape[0]
    if veh.shape[1] == 0: # 场景中完全没有车辆数据
        veh_pos = torch.full((num_valid_ped, 2), float('nan'), device=pos.device)
        veh_vel = torch.full((num_valid_ped, 2), float('nan'), device=pos.device)
    else: # 找到每个场景中距离各个行人最近的车辆
        all_veh_pos = veh[..., -1, :] # (batch_size, #vehicle, 2)
        all_veh_vel = (veh[..., -1, :] - veh[..., -2, :]) * args.fps  # (batch_size, #vehicle, 2)
        # 取出每个有效行人对应场景的车辆数据
        batch_veh_pos = all_veh_pos[batch_indices] # (valid{B*#pedestrian}, #vehicle, 2)
        batch_veh_vel = all_veh_vel[batch_indices] # (valid{B*#pedestrian}, #vehicle, 2)
        # 计算行人到同场景所有车辆的距离 (将无效车辆的距离设为无穷大)
        dist = (ped_pos.unsqueeze(1) - batch_veh_pos).norm(dim=-1).nan_to_num_(nan=float('inf')) # (valid{B*#pedestrian}, #vehicle)
        # 找到最近车辆的索引
        min_dist, nearest_idx = torch.min(dist, dim=1) # (valid{B*#pedestrian},)
        has_vehicle = min_dist != float('inf')
        # Gather 最近车辆的位置和速度
        gather_idx = nearest_idx.view(-1, 1, 1).expand(-1, 1, 2)
        veh_pos = torch.gather(batch_veh_pos, 1, gather_idx).squeeze(1) # (valid{B*#pedestrian}, 2)
        veh_vel = torch.gather(batch_veh_vel, 1, gather_idx).squeeze(1) # (valid{B*#pedestrian}, 2)
        # 如果该行人所在的场景没有任何车辆，设为 NaN
        veh_pos[~has_vehicle] = float('nan')
        veh_vel[~has_vehicle] = float('nan')
    norm_err, tan_err = calc_xy_error(traj_diff, ped_pos, veh_pos, veh_vel)
    return norm_err, tan_err


def get_collision_rate(args, map_data, future_veh, veh_length, mask, pos_pred):
    S, B, P, T, _ = pos_pred.shape
    flat_pos = pos_pred.permute(0, 1, 3, 2, 4).reshape(-1, P, 2)  ## 将 (S, B, P, T, 2) -> (S, B, T, P, 2) -> (S*B*T, P, 2)
    dist_matrix = torch.cdist(flat_pos, flat_pos, p=2)  # (S*B*T, P, P)
    eye_matrix = torch.eye(P, device=pos_pred.device, dtype=torch.bool).unsqueeze(0)
    # 只有当行人 i 和行人 j 都有效 (mask=True) 时才计入碰撞
    mask_expanded = mask.unsqueeze(0).unsqueeze(2).expand(S, -1, T, -1).reshape(-1, P) # (B, P) -> (1, B, 1, P) -> (S, B, T, P) -> (S*B*T, P)
    valid_pair_mask = mask_expanded.unsqueeze(2) & mask_expanded.unsqueeze(1)
    collision_matrix = (
        (dist_matrix < args.collision_threshold) &
        (~eye_matrix) &
        valid_pair_mask
    )
    collision_rate = collision_matrix.sum() / (S * mask.sum() * T)
    collision_ped = collision_rate.item()

    if future_veh.shape[1] == 0:
        collision_veh = float('nan')
    else:
        _, V, _, _ = future_veh.shape # (B, V, T, 2)
        flat_veh = future_veh.unsqueeze(0).expand(S, -1, -1, -1, -1).permute(0, 1, 3, 2, 4).reshape(-1, V, 2)  ## 将 (B, V, T, 2) -> (1, B, V, T, 2) -> (S, B, T, V, 2) -> (S*B*T, V, 2)
        dist_matrix = torch.cdist(flat_pos, flat_veh, p=2) # (S*B*T, P, V)
        # 只有当行人 i 和车辆 j 都有效时才计入碰撞
        veh_mask = torch.arange(V, device=args.device).expand(B, V) < veh_length.unsqueeze(-1)  # (B, V)
        veh_mask_expanded = veh_mask.unsqueeze(0).unsqueeze(2).expand(S, -1, T, -1).reshape(-1, V) # (B, V) -> (1, B, 1, V) -> (S, B, T, V) -> (S*B*T, V)
        valid_pair_mask = mask_expanded.unsqueeze(2) & veh_mask_expanded.unsqueeze(1) # (S*B*T, P, V)
        collision_matrix = (dist_matrix < args.collision_threshold) & valid_pair_mask
        collision_rate = collision_matrix.sum() / 2 / (S * mask.sum() * T) # 除以 2 因为碰撞双方只有一方是行人
        collision_veh = collision_rate.item()


    if not np.isfinite(map_data.map).any():
        collision_map = float('nan')
    else:
        idx = pos_pred[..., 0].sub(map_data.xmin).div(map_data.xmax-map_data.xmin).mul(map_data.map.shape[0]).round().long().clamp(0, map_data.map.shape[0] - 1)  # (batch_size, #pedestrian)
        jdx = pos_pred[..., 1].sub(map_data.ymin).div(map_data.ymax-map_data.ymin).mul(map_data.map.shape[1]).round().long().clamp(0, map_data.map.shape[1] - 1)  # (batch_size, #pedestrian)
        sur_info = map_data.map[idx.cpu().numpy(), jdx.cpu().numpy()] # (batch_size, #pedestrian)
        collision_rate = (sur_info > 0.9).mean()
        collision_map = collision_rate.item()

    return collision_ped, collision_veh, collision_map


def get_collision_rate2(args, map_data, future_veh, veh_length, mask, pos_pred, pos_true):
    S, B, P, T, _ = pos_true.unsqueeze(0).shape
    flat_pos_true = pos_true.unsqueeze(0).permute(0, 1, 3, 2, 4).reshape(-1, P, 2)  ## 将 (S, B, P, T, 2) -> (S, B, T, P, 2) -> (S*B*T, P, 2)
    dist_matrix = torch.cdist(flat_pos_true, flat_pos_true, p=2)  # (S*B*T, P, P)
    eye_matrix = torch.eye(P, device=pos_true.unsqueeze(0).device, dtype=torch.bool).unsqueeze(0)
    # 只有当行人 i 和行人 j 都有效 (mask=True) 时才计入碰撞
    mask_expanded_true = mask.unsqueeze(0).unsqueeze(2).expand(S, -1, T, -1).reshape(-1, P) # (B, P) -> (1, B, 1, P) -> (S, B, T, P) -> (S*B*T, P)
    valid_pair_mask = mask_expanded_true.unsqueeze(2) & mask_expanded_true.unsqueeze(1)
    collision_matrix_true = (
        (dist_matrix < args.collision_threshold) &
        (~eye_matrix) &
        valid_pair_mask
    )

    S, B, P, T, _ = pos_pred.shape
    flat_pos = pos_pred.permute(0, 1, 3, 2, 4).reshape(-1, P, 2)  ## 将 (S, B, P, T, 2) -> (S, B, T, P, 2) -> (S*B*T, P, 2)
    dist_matrix = torch.cdist(flat_pos, flat_pos, p=2)  # (S*B*T, P, P)
    eye_matrix = torch.eye(P, device=pos_pred.device, dtype=torch.bool).unsqueeze(0)
    # 只有当行人 i 和行人 j 都有效 (mask=True) 时才计入碰撞
    mask_expanded = mask.unsqueeze(0).unsqueeze(2).expand(S, -1, T, -1).reshape(-1, P) # (B, P) -> (1, B, 1, P) -> (S, B, T, P) -> (S*B*T, P)
    valid_pair_mask = mask_expanded.unsqueeze(2) & mask_expanded.unsqueeze(1)
    collision_matrix = (
        (dist_matrix < args.collision_threshold) &
        (~eye_matrix) &
        valid_pair_mask
    )
    collision_matrix = collision_matrix.reshape(S, B, T, P, P) & (~collision_matrix_true).reshape(1, B, T, P, P)  # 只统计预测中出现但真实中没有的碰撞
    collision_rate = collision_matrix.sum() / (S * mask.sum() * T)
    collision_ped = collision_rate.item()

    if future_veh.shape[1] == 0:
        collision_veh = float('nan')
    else:
        S, B, P, T, _ = pos_true.unsqueeze(0).shape
        _, V, _, _ = future_veh.shape # (B, V, T, 2)
        flat_veh = future_veh.unsqueeze(0).expand(S, -1, -1, -1, -1).permute(0, 1, 3, 2, 4).reshape(-1, V, 2)  ## 将 (B, V, T, 2) -> (1, B, V, T, 2) -> (S, B, T, V, 2) -> (S*B*T, V, 2)
        dist_matrix = torch.cdist(flat_pos_true, flat_veh, p=2) # (S*B*T, P, V)
        # 只有当行人 i 和车辆 j 都有效时才计入碰撞
        veh_mask = torch.arange(V, device=args.device).expand(B, V) < veh_length.unsqueeze(-1)  # (B, V)
        veh_mask_expanded = veh_mask.unsqueeze(0).unsqueeze(2).expand(S, -1, T, -1).reshape(-1, V) # (B, V) -> (1, B, 1, V) -> (S, B, T, V) -> (S*B*T, V)
        valid_pair_mask = mask_expanded_true.unsqueeze(2) & veh_mask_expanded.unsqueeze(1) # (S*B*T, P, V)
        collision_matrix_true = (dist_matrix < args.collision_threshold) & valid_pair_mask

        S, B, P, T, _ = pos_pred.shape
        _, V, _, _ = future_veh.shape # (B, V, T, 2)
        flat_veh = future_veh.unsqueeze(0).expand(S, -1, -1, -1, -1).permute(0, 1, 3, 2, 4).reshape(-1, V, 2)  ## 将 (B, V, T, 2) -> (1, B, V, T, 2) -> (S, B, T, V, 2) -> (S*B*T, V, 2)
        dist_matrix = torch.cdist(flat_pos, flat_veh, p=2) # (S*B*T, P, V)
        # 只有当行人 i 和车辆 j 都有效时才计入碰撞
        veh_mask = torch.arange(V, device=args.device).expand(B, V) < veh_length.unsqueeze(-1)  # (B, V)
        veh_mask_expanded = veh_mask.unsqueeze(0).unsqueeze(2).expand(S, -1, T, -1).reshape(-1, V) # (B, V) -> (1, B, 1, V) -> (S, B, T, V) -> (S*B*T, V)
        valid_pair_mask = mask_expanded.unsqueeze(2) & veh_mask_expanded.unsqueeze(1) # (S*B*T, P, V)
        collision_matrix = (dist_matrix < args.collision_threshold) & valid_pair_mask
        collision_matrix = collision_matrix.reshape(S, B, T, P, V) & (~collision_matrix_true).reshape(1, B, T, P, V)  # 只统计预测中出现但真实中没有的碰撞
        collision_rate = collision_matrix.sum() / 2 / (S * mask.sum() * T) # 除以 2 因为碰撞双方只有一方是行人
        collision_veh = collision_rate.item()

    if not np.isfinite(map_data.map).any():
        collision_map = float('nan')
    else:
        idx = pos_true[None, ..., 0].sub(map_data.xmin).div(map_data.xmax-map_data.xmin).mul(map_data.map.shape[0]).round().long().clamp(0, map_data.map.shape[0] - 1)  # (S, B, P, T)
        jdx = pos_true[None, ..., 1].sub(map_data.ymin).div(map_data.ymax-map_data.ymin).mul(map_data.map.shape[1]).round().long().clamp(0, map_data.map.shape[1] - 1)  # (S, B, P, T)
        sur_info = map_data.map[idx.cpu().numpy(), jdx.cpu().numpy()] # (S, B, P, T)
        collision_matrix_true = sur_info > 0.9

        idx = pos_pred[..., 0].sub(map_data.xmin).div(map_data.xmax-map_data.xmin).mul(map_data.map.shape[0]).round().long().clamp(0, map_data.map.shape[0] - 1)  # (S, B, P, T)
        jdx = pos_pred[..., 1].sub(map_data.ymin).div(map_data.ymax-map_data.ymin).mul(map_data.map.shape[1]).round().long().clamp(0, map_data.map.shape[1] - 1)  # (S, B, P, T)
        sur_info = map_data.map[idx.cpu().numpy(), jdx.cpu().numpy()] # (S, B, P, T)
        collision_matrix = (sur_info > 0.9).reshape(S, B, P, T) & (~collision_matrix_true).reshape(1, B, P, T)  # 只统计预测中出现但真实中没有的碰撞
        collision_rate = collision_matrix.sum() / (S * mask.sum() * T)
        collision_map = collision_rate.item()

    return collision_ped, collision_veh, collision_map


def test_once(
    args,
    test_loaders: List[D.DataLoader],
    model: Model,
    criterion: nn.Module,
    diffusion: DDPM,
    epoch: int,
) -> Dict:
    """
    进行单步测试

    Args:
        args: 全局参数
        test_loaders: 测试数据加载器列表
        model: 待测试模型
        criterion: 损失函数
        diffusion: 扩散模型
        epoch: 当前训练轮数

    Returns:
        all_records: 测试记录字典    
    """
    test_timer = NamedTimer(unit='it', mode='pace')
    records_list = []
    for loader in test_loaders:
        map_data = loader.dataset.map_data
        map = torch.from_numpy(map_data.map).to(args.device).float()
        test_timer.add('prepare data')
        model.set_map_embedding(
            map=map,
            xmin=map_data.xmin,
            xmax=map_data.xmax,
            ymin=map_data.ymin,
            ymax=map_data.ymax,
        )
        test_timer.add('embed map')
        records = dict(
            loss=[], ade=[], fde=[], trajlen=[], 
            ped_num=[], veh_num=[], rollout_time=[], 
            collision_ped=[], collision_veh=[], collision_map=[], 
            collision_ped_base=[], collision_veh_base=[], collision_map_base=[], 
            collision_ped2=[], collision_veh2=[], collision_map2=[],
            apd=[], norm_err=[], tan_err=[]
        )
        for batch_idx, batch in enumerate(tqdm(loader, disable=False, leave=False, dynamic_ncols=True)):
            pos = batch['pos'].to(args.device)  # (batch_size, #pedestrian, 2)
            vel = batch['vel'].to(args.device)  # (batch_size, #pedestrian, 2)
            hst = batch['hst'].to(args.device)  # (batch_size, #pedestrian, hist_step, 2)
            des = batch['des'].to(args.device)  # (batch_size, #pedestrian, 2)
            spd = batch['spd'].to(args.device)  # (batch_size, #pedestrian, 1)
            veh = batch['veh'].to(args.device)  # (batch_size, #vehicle, hist_step + 1, 2)
            future_acc = batch['future_acc'].to(args.device)  # (batch_size, #pedestrian, pred_step*roll_step, 2)
            future_pos = batch['future_pos'].to(args.device)  # (batch_size, #pedestrian, pred_step*roll_step, 2)
            future_veh = batch['future_veh'].to(args.device)  # (batch_size, #vehicle, pred_step*roll_step, 2)
            ped_length = batch['ped_length'].to(args.device)  # (batch_size,)
            veh_length = batch['veh_length'].to(args.device)  # (batch_size,)

            S = args.sample_num  # 采样次数
            N = args.denoise_step  # 采样步数
            assert args.T % N == 0, f"试图使用 {N} 步采样，然而训练步数 {args.T} mod {N} 不等于 0!"
            assert 1 <= args.step_offset <= args.T // N, f"step_offset 应该取值于 {{1, ..., {args.T // N}}}!"
            pos_now = pos.repeat(S, 1, 1)  # (S*B, #pedestrian, 2)
            vel_now = vel.repeat(S, 1, 1)  # (S*B, #pedestrian, 2)
            hst_now = hst.repeat(S, 1, 1, 1)  # (S*B, #pedestrian, hist_step, 2)
            des_now = des.repeat(S, 1, 1)  # (S*B, #pedestrian, 2)
            spd_now = spd.repeat(S, 1, 1)  # (S*B, #pedestrian, 1)
            veh_now = veh.repeat(S, 1, 1, 1)  # (S*B, #vehicle, hist_step + 1, 2)
            ped_length_repeat = ped_length.repeat(S)  # (S*B,)
            veh_length_repeat = veh_length.repeat(S)  # (S*B,)
            test_timer.add('prepare data', n=0)

            for_plot = []
            acc_pred = []
            start_time = time.time()
            for step in range(args.roll_step):
                model.set_veh_embedding(veh=veh_now)
                model.set_ped_embedding(pos=pos_now, vel=vel_now, hst=hst_now, des=des_now, spd=spd_now)
                model.set_sur_info()
                test_timer.add('embed data')

                shape = list(future_acc.shape)
                shape[0] *= S
                shape[2] = args.pred_step
                xt = torch.randn(shape, device=args.device)  # 从噪声开始
                for_plot.append([diffusion.noise_to_x0(xt=xt, denoise_t=args.T, noise=0) / args.scale_accelerate])
                stride = args.T // N
                steps = reversed(range(args.step_offset, args.T+1, stride))
                for t in tqdm(steps, disable=True, leave=False, dynamic_ncols=True):
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
                    state = SimulateState(
                        df_ped=None, df_veh=None, map_data=None, 
                        ped_list=None, veh_list=None, frame=None,
                        pos_now=pos_now, vel_now=vel_now, des_now=des_now, 
                        hst_now=hst_now, spd_now=spd_now, veh_now=veh_now,
                    )
                    x0 = x0 + guidance(args, x0, state, model, diffusion, noisy_acc, xt, denoise_t, ped_length_repeat, veh_length_repeat)
                    ## 去噪
                    xt = diffusion.denoise(xt, t, x0=x0, stride=min(stride, t))
                    for_plot[-1].append(x0 / args.scale_accelerate)

                acc_new = xt / args.scale_accelerate  # (S*B, #pedestrian, pred_step, 2)
                acc_pred.append(acc_new)
                test_timer.add('denoise')

                vel_new = vel_now.unsqueeze(-2) + acc_new.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
                pos_new = pos_now.unsqueeze(-2) + vel_new.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
                veh_new = future_veh[:, :, step*args.pred_step:(step+1)*args.pred_step, :].repeat(S, 1, 1, 1)  # (S*B, #vehicle, pred_step, 2)

                hst_now = torch.cat([hst_now, pos_now.unsqueeze(-2), pos_new], dim=-2)[:, :, -args.hist_step-1:-1, :] # (S*B, #pedestrian, hist_step, 2)
                veh_now = torch.cat([veh_now, veh_new], dim=-2)[:, :, -args.hist_step-1:, :]  # (S*B, #vehicle, hist_step + 1, 2)
                pos_now = pos_new[:, :, -1, :] # (S*B, #pedestrian, 2)
                vel_now = vel_new[:, :, -1, :] # (S*B, #pedestrian, 2)
                test_timer.add('rollout')
            rollout_time = (time.time() - start_time) / args.roll_step

            acc_pred = torch.concat(acc_pred, dim=-2)  # (S*B, #pedestrian, roll_step*pred_step, 2)
            batch_size, ped_num, _, _ = future_acc.shape
            acc_pred = acc_pred.view(S, batch_size, ped_num, args.roll_step*args.pred_step, 2)  # (S, B, #pedestrian, roll_step*pred_step, 2)
            # 获取有效的行人掩模
            mask = torch.arange(ped_num, device=args.device).expand(batch_size, ped_num) < ped_length.unsqueeze(-1)  # (B, #pedestrian)
            # 计算 pos_true 和 vel_true
            acc_true = future_acc # (B, #pedestrian, roll_step*pred_step, 2)
            vel_true = vel.unsqueeze(-2) + acc_true.cumsum(dim=-2) / args.fps # (B, #pedestrian, roll_step*pred_step, 2)
            pos_true = pos.unsqueeze(-2) + vel_true.cumsum(dim=-2) / args.fps # (B, #pedestrian, roll_step*pred_step, 2)
            max_err = np.nanmax((pos_true - future_pos).abs().cpu().numpy(), axis=(-2, -1))
            _logger.debug(
                f"pos_true 和 future 最大差距 > 1: {(max_err > 1).mean():.2%}, "
                f"pos_true 和 future 最大差距 > 1e-6: {(max_err > 1e-6).mean():.2%}"
            )
            # pos_true = future_pos # (B, #pedestrian, roll_step*pred_step, 2)
            # 计算 loss
            loss = criterion(acc_pred, acc_true.expand(acc_pred.shape)) # float
            records['loss'].extend([loss.item()] * future_acc.shape[0]) # List[float]
            # 计算 distance error
            vel_pred = vel.unsqueeze(-2) + acc_pred.cumsum(dim=-2) / args.fps  # (S, B, #pedestrian, roll_step*pred_step, 2)
            pos_pred = pos.unsqueeze(-2) + vel_pred.cumsum(dim=-2) / args.fps  # (S, B, #pedestrian, roll_step*pred_step, 2)
            dis_err = (pos_pred - pos_true).norm(dim=-1) # (S, B, #pedestrian, roll_step*pred_step)
            test_timer.add('evaluate')
            # 可视化
            if False and batch_idx == 0:
                pid = 0
                save_path = f"{args.save_path}/visualize/epoch{epoch}_{loader.dataset.name}_idx{batch_idx}_pid{pid}.png"
                visualize(args, pos, vel, hst, for_plot, mask, pos_true, pos_pred, save_path, pid)
                test_timer.add('visualize')
            # 移除 padding 的行人
            dis_err = dis_err[:, mask, :] # (S, valid{B*#pedestrian}, roll_step*pred_step)
            # 选择 ade 最佳的 sample
            sample_idx = dis_err.mean(dim=-1).argmin(dim=0)  # (valid{B*#pedestrian},)
            valid_idx = torch.arange(dis_err.shape[1], device=args.device)  # (valid{B*#pedestrian},)
            dis_err = dis_err[sample_idx, valid_idx, :]  # (valid{B*#pedestrian}, roll_step*pred_step)
            # 计算 ade, fde
            ade = dis_err.mean(dim=-1) # (valid{B*#pedestrian})
            fde = dis_err[..., -1] # (valid{B*#pedestrian})
            records['ade'].extend(ade.cpu().tolist()) # List[float]
            records['fde'].extend(fde.cpu().tolist()) # List[float]
            # 计算切向误差和法向误差
            norm_err, tan_err = get_xy_error(args, pos, veh, mask, pos_true, pos_pred, sample_idx, valid_idx)
            records['norm_err'].extend(norm_err.cpu().tolist()) # List[float]
            records['tan_err'].extend(tan_err.cpu().tolist()) # List[float]
            # 计算碰撞数
            collision_ped, collision_veh, collision_map = get_collision_rate(args, map_data, future_veh, veh_length, mask, pos_pred)
            records['collision_ped'].append(collision_ped)
            records['collision_veh'].append(collision_veh)
            records['collision_map'].append(collision_map)
            collision_ped_base, collision_veh_base, collision_map_base = get_collision_rate(args, map_data, future_veh, veh_length, mask, pos_true.unsqueeze(0))
            records['collision_ped_base'].append(collision_ped_base)
            records['collision_veh_base'].append(collision_veh_base)
            records['collision_map_base'].append(collision_map_base)
            collision_ped2, collision_veh2, collision_map2 = get_collision_rate2(args, map_data, future_veh, veh_length, mask, pos_pred, pos_true)
            records['collision_ped2'].append(collision_ped2)
            records['collision_veh2'].append(collision_veh2)
            records['collision_map2'].append(collision_map2)
            # 计算 APD 多样性指标
            tmp = pos_pred[:, mask, :, :] # (S, valid{B*#pedestrian}, roll_step*pred_step, 2)
            tmp = (tmp[None, :, ...] - tmp[:, None, ...]).norm(dim=-1).mean(dim=-1)  # (S, S, valid{B*#pedestrian})
            apd = tmp.flatten(0, 1).sum(dim=0) / (S * (S - 1)) # (valid{B*#pedestrian},)
            records['apd'].extend(apd.cpu().tolist()) # List[float]
            # 计算轨迹长度
            trajlen = pos_true.diff(dim=-2).norm(dim=-1).sum(dim=-1)[mask] # (valid{B*#pedestrian})
            records['trajlen'].extend(trajlen.cpu().tolist()) # List[float]
            # 统计行人和车辆数量
            records['ped_num'].extend(ped_length.cpu().tolist()) # List[int]
            records['veh_num'].extend(veh_length.cpu().tolist()) # List[int]
            # 统计 Rollout 用时
            records['rollout_time'].append(rollout_time)  # List[float]
            test_timer.add('evaluate', n=0)
        records_list.append(records)
        _logger.info(tag2ansi(
            f"[#66CCFF][Epoch {epoch}/{args.epochs}] Eval on {loader.dataset.name}: "
            f"[bold underline orange]Accuracy={1 - np.mean(records['ade']) / np.mean(records['trajlen']):.2%}[reset], "
            f"[#66CCFF]Loss={np.mean(records['loss']):.4f}, "
            f"[#66CCFF]ADE={np.mean(records['ade']):.4f}, "
            f"[#66CCFF]FDE={np.mean(records['fde']):.4f}, "
            f"[#66CCFF]X_ERROR (normal)={np.nanmean(records['norm_err']):.4f}, "
            f"[#66CCFF]Y_ERROR (tangential)={np.nanmean(records['tan_err']):.4f}, "
            f"[#66CCFF]Collision-Ped={np.mean(records['collision_ped']) - (base:= np.mean(records['collision_ped_base'])):.2%} (+{base:.2%}), "
            f"[#66CCFF]Collision-Veh={np.mean(records['collision_veh']) - (base:= np.mean(records['collision_veh_base'])):.2%} (+{base:.2%}), "
            f"[#66CCFF]Collision-Map={np.mean(records['collision_map']) - (base:= np.mean(records['collision_map_base'])):.2%} (+{base:.2%}), "
            f"[#66CCFF]Collision-Ped2={np.mean(records['collision_ped2']):.2%}, "
            f"[#66CCFF]Collision-Veh2={np.mean(records['collision_veh2']):.2%}, "
            f"[#66CCFF]Collision-Map2={np.mean(records['collision_map2']):.2%}, "
            f"[#66CCFF]APD={np.mean(records['apd']):.4f}, "
            f"[#66CCFF]AvgLen={np.mean(records['trajlen']):.4f}, "
            f"[#66CCFF]PedNum={np.mean(records['ped_num']):.1f}, "
            f"[#66CCFF]VehNum={np.mean(records['veh_num']):.1f}, "
            f"[#66CCFF]RolloutTime={np.mean(records['rollout_time'])*1000:.2f}ms "
            f"([bold underline orange]FPS={1/np.mean(records['rollout_time']):.2f} Hz[reset])"
        ))
    all_records = {
        'epoch': epoch,
        'dataset_class': [type(loader.dataset).__name__.removesuffix('Dataset') for loader in test_loaders],
        'dataset_names': [loader.dataset.name for loader in test_loaders],
        'sample_nums': [len(loader.dataset) for loader in test_loaders],
    }
    for records in records_list:
        for k, v in records.items():
            if k not in all_records:
                all_records[k] = []
            all_records[k].append(np.mean(v))
    w = np.array(all_records['sample_nums'], dtype=float)
    w /= w.sum()
    all_records['accuracy'] = 1 - np.sum(w * all_records['ade']) / np.sum(w * all_records['trajlen'])
    all_records['unweighted_accuracy'] = 1 - np.mean(all_records['ade']) / np.mean(all_records['trajlen'])
    _logger.note(tag2ansi(
        f"[#66CCFF][Epoch {epoch}/{args.epochs}] Overall: "
        f"[bold underline orange]Accuracy={all_records['accuracy']:.2%}[reset] (unweighted={all_records['unweighted_accuracy']:.2%}), "
        f"[#66CCFF]Loss={np.sum(w * all_records['loss']):.4f}, "
        f"[#66CCFF]ADE={np.sum(w * all_records['ade']):.4f}, "
        f"[#66CCFF]FDE={np.sum(w * all_records['fde']):.4f}, "
        f"[#66CCFF]X_ERROR (normal)={np.nansum(w * all_records['norm_err']) / np.sum(w * np.isfinite(all_records['norm_err'])):.4f}, "
        f"[#66CCFF]Y_ERROR (tangential)={np.nansum(w * all_records['tan_err']) / np.sum(w * np.isfinite(all_records['tan_err'])):.4f}, "
        f"[#66CCFF]Collision-Ped={np.sum(w * all_records['collision_ped']) - (base := np.sum(w * all_records['collision_ped_base'])):.2%} (+{base:.2%}), "
        f"[#66CCFF]Collision-Veh={np.sum(w * all_records['collision_veh']) - (base := np.sum(w * all_records['collision_veh_base'])):.2%} (+{base:.2%}), "
        f"[#66CCFF]Collision-Map={np.sum(w * all_records['collision_map']) - (base := np.sum(w * all_records['collision_map_base'])):.2%} (+{base:.2%}), "
        f"[#66CCFF]Collision-Ped2={np.sum(w * all_records['collision_ped2']):.2%}, "
        f"[#66CCFF]Collision-Veh2={np.sum(w * all_records['collision_veh2']):.2%}, "
        f"[#66CCFF]Collision-Map2={np.sum(w * all_records['collision_map2']):.2%}, "
        f"[#66CCFF]APD={np.sum(w * all_records['apd']):.4f}, "
        f"[#66CCFF]AvgLen={np.sum(w * all_records['trajlen']):.4f}, "
        f"[#66CCFF]PedNum={np.sum(w * all_records['ped_num']):.4f}, "
        f"[#66CCFF]VehNum={np.sum(w * all_records['veh_num']):.4f}, "
        f"[#66CCFF]RolloutTime={np.mean(all_records['rollout_time'])*1000:.2}ms "
        f"([bold underline orange]FPS={1/np.mean(all_records['rollout_time']):.2f} Hz[reset]), "
        f"[#66CCFF]Time={test_timer}"
    ))
    if len(set(all_records['dataset_class'])) > 1:
        for klass in sorted(list(set(all_records['dataset_class']))):
            idxs = [i for i, k in enumerate(all_records['dataset_class']) if k == klass]
            ade = np.array([all_records['ade'][i] for i in idxs])
            fde = np.array([all_records['fde'][i] for i in idxs])
            norm_err = np.array([all_records['norm_err'][i] for i in idxs])
            tan_err = np.array([all_records['tan_err'][i] for i in idxs])
            trajlen = np.array([all_records['trajlen'][i] for i in idxs])
            ped_num = np.array([all_records['ped_num'][i] for i in idxs])
            veh_num = np.array([all_records['veh_num'][i] for i in idxs])
            collision_ped = np.array([all_records['collision_ped'][i] for i in idxs])
            collision_veh = np.array([all_records['collision_veh'][i] for i in idxs])
            collision_map = np.array([all_records['collision_map'][i] for i in idxs])
            collision_ped_base = np.array([all_records['collision_ped_base'][i] for i in idxs])
            collision_veh_base = np.array([all_records['collision_veh_base'][i] for i in idxs])
            collision_map_base = np.array([all_records['collision_map_base'][i] for i in idxs])
            collision_ped2 = np.array([all_records['collision_ped2'][i] for i in idxs])
            collision_veh2 = np.array([all_records['collision_veh2'][i] for i in idxs])
            collision_map2 = np.array([all_records['collision_map2'][i] for i in idxs])
            rollout_time = np.array([all_records['rollout_time'][i] for i in idxs])
            apd = np.array([all_records['apd'][i] for i in idxs])
            w = np.array([all_records['sample_nums'][i] for i in idxs], dtype=float)
            w /= w.sum()
            acc = 1 - np.sum(w * ade) / np.sum(w * trajlen)
            _logger.note(tag2ansi(
                f"[#66CCFF][Epoch {epoch}/{args.epochs}] Overall on {klass} datasets: "
                f"[bold underline orange]Accuracy={acc:.2%}[reset], "
                f"[#66CCFF]ADE={np.sum(w * ade):.4f}, "
                f"[#66CCFF]FDE={np.sum(w * fde):.4f}, "
                f"[#66CCFF]X_ERROR (normal)={np.nansum(w * norm_err) / np.sum(w * np.isfinite(norm_err)):.4f}, "
                f"[#66CCFF]Y_ERROR (tangential)={np.nansum(w * tan_err) / np.sum(w * np.isfinite(tan_err)):.4f}, "
                f"[#66CCFF]Collision-Ped={np.sum(w * collision_ped) - (base := np.sum(w * collision_ped_base)):.2%} (+{base:.2%}), "
                f"[#66CCFF]Collision-Veh={np.sum(w * collision_veh) - (base := np.sum(w * collision_veh_base)):.2%} (+{base:.2%}), "
                f"[#66CCFF]Collision-Map={np.sum(w * collision_map) - (base := np.sum(w * collision_map_base)):.2%} (+{base:.2%}), "
                f"[#66CCFF]Collision-Ped2={np.sum(w * collision_ped2):.2%}, "
                f"[#66CCFF]Collision-Veh2={np.sum(w * collision_veh2):.2%}, "
                f"[#66CCFF]Collision-Map2={np.sum(w * collision_map2):.2%}, "
                f"[#66CCFF]APD={np.sum(w * apd):.4f}, "
                f"[#66CCFF]AvgLen={np.sum(w * trajlen):.4f}, "
                f"[#66CCFF]PedNum={np.sum(w * ped_num):.4f}, "
                f"[#66CCFF]VehNum={np.sum(w * veh_num):.4f}, "
                f"[#66CCFF]RolloutTime={np.mean(rollout_time)*1000:.2f}ms "
                f"([bold underline orange]FPS={1/np.mean(rollout_time):.2f} Hz[reset])"
            ))
    return all_records
