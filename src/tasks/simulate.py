import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from argparse import Namespace
from ..model.model import Model
from ..diffusion import DDPM
from ..dataset import BaseDataset
from ..utils.get_force_map import get_force_map
from ..utils.extract_patches import extract_patches_torch


def init_simulation(args: Namespace, dataset: BaseDataset, frame_idx: int, model: Model):
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
    spd = (
        df_ped
        .reindex(pd.MultiIndex.from_product([
            range(frame_idx, frame_idx+int(5*args.fps) + 1),
            ped_list
        ], names=['f', 'id']))
        .unstack().ffill().bfill().diff().mul(args.fps).iloc[1:]
        .stack(future_stack=True)
        .pow(2).sum(axis='columns', min_count=2).pow(0.5)
        .unstack().mean(axis='rows')
        .values[:, np.newaxis] # (#pedestrian, 1)
    )
    assert spd.shape == (len(ped_list), 1), "您可能需要将这里上方的 future_stack=True 改成 dropna=False 再试一试，或者用我们推荐的 pandas 版本 2.3.3"
    map_data = dataset.map_data

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
    return ped_list, veh_list, df_veh, frame, pos_now, vel_now, hst_now, des_now, spd_now, veh_now


def simulate_one_step(
        args: Namespace, model: Model, diffusion: DDPM,
        ped_list, veh_list, df_veh, frame, pos_now, vel_now, hst_now, des_now, spd_now, veh_now,
    ):
    model.eval()
    torch.set_grad_enabled(False)

    # _logger.info(f"Simulating from frame {frame} to frame {frame + args.pred_step}...")
    S = args.sample_num  # 采样次数
    N = args.denoise_step  # 采样步数
    ped_length_repeat = torch.full((S, ), len(ped_list), device=args.device, dtype=torch.long)  # (S,)
    veh_length_repeat = torch.full((S, ), len(veh_list), device=args.device, dtype=torch.long)  # (S,)

    # _logger.info(f"  Starting setting vehicle embeddings...")
    model.set_veh_embedding(veh=veh_now)
    # _logger.info(f"  Starting setting pedestrian embeddings...")
    model.set_ped_embedding(pos=pos_now, vel=vel_now, hst=hst_now, des=des_now, spd=spd_now)
    # _logger.info(f"  Starting setting surrounding info...")
    model.set_sur_info()

    shape = [S, len(ped_list), args.pred_step, 2]  # (S*1, #pedestrian, pred_step, 2)
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
        total_guidance = 0.0
        future_acc = x0 / args.scale_accelerate  # (S*B, #pedestrian, pred_step, 2)
        future_vel = vel_now.unsqueeze(-2) + future_acc.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
        future_pos = pos_now.unsqueeze(-2) + future_vel.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
        # 目的地 CFG 引导
        if args.des_cfg > 0:
            if 'model_wo_des' not in locals():
                dst_nan = torch.full_like(des_now, torch.nan)
                # model_wo_des = deepcopy(model)
                model_wo_des = Model(model.args).to(device=args.device)
                model_wo_des.load_state_dict(model.state_dict())
                model_wo_des.eval()
                model_wo_des.set_map_embedding(map=model.map, xmin=model.xmin, xmax=model.xmax, ymin=model.ymin, ymax=model.ymax)
                model_wo_des.set_veh_embedding(veh=veh_now)
                model_wo_des.set_ped_embedding(pos=pos_now, vel=vel_now, hst=hst_now, des=dst_nan, spd=spd_now)
                model_wo_des.set_sur_info()
            output_wo_des = model_wo_des(
                noisy_acc=noisy_acc, 
                denoise_t=denoise_t,
                ped_length=ped_length_repeat, 
                veh_length=veh_length_repeat,
            )  # (S*B, #pedestrian, pred_step, 2)
            x0_wo_des = diffusion.noise_to_x0(xt, denoise_t, output_wo_des) if args.predict_noise else output_wo_des
            total_guidance = total_guidance + args.des_cfg * (x0 - x0_wo_des)
        # 障碍物 CFG 引导
        if args.map_cfg > 0:
            if 'model_wo_map' not in locals():
                map_nan = torch.full_like(model.map, torch.nan)
                model_wo_map = Model(model.args).to(device=args.device)
                model_wo_map.load_state_dict(model.state_dict())
                model_wo_map.eval()
                model_wo_map.set_map_embedding(map=map_nan, xmin=model.xmin, xmax=model.xmax, ymin=model.ymin, ymax=model.ymax)
                model_wo_map.set_veh_embedding(veh=veh_now)
                model_wo_map.set_ped_embedding(pos=pos_now, vel=vel_now, hst=hst_now, des=des_now, spd=spd_now)
                model_wo_map.set_sur_info()
            output_wo_map = model_wo_map(
                noisy_acc=noisy_acc, 
                denoise_t=denoise_t,
                ped_length=ped_length_repeat, 
                veh_length=veh_length_repeat,
            )  # (S*B, #pedestrian, pred_step, 2)
            x0_wo_map = diffusion.noise_to_x0(xt, denoise_t, output_wo_map) if args.predict_noise else output_wo_map
            total_guidance = total_guidance + args.map_cfg * (x0 - x0_wo_map)
        # 目的地 CG 引导 (acc 方向应指向 pos_now 与 des_now 连线的方向)
        if args.direction_cg > 0:
            direction = F.normalize(des_now.unsqueeze(-2) - future_pos, dim=-1).nan_to_num(0.0)  # (S*B, #pedestrian, pred_step, 2)
            # loss = F.mse_loss(future_acc, direction.detach())
            # grad = torch.autograd.grad(loss, x0)[0]
            grad = 2 * (future_acc - direction) / args.scale_accelerate  # 可以直接手算
            total_guidance = total_guidance - args.direction_cg * grad
        # 目的地 CG 引导 (acc 应使得在目的地形成的势阱中能量更低)
        if args.energy_cg > 0:
            with torch.enable_grad():
                x0_grad = x0.detach().requires_grad_(True)
                future_acc_grad = x0_grad / args.scale_accelerate  # (S*B, #pedestrian, pred_step, 2)
                future_vel_grad = vel_now.unsqueeze(-2) + future_acc_grad.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
                future_pos_grad = pos_now.unsqueeze(-2) + future_vel_grad.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
                loss = (des_now.unsqueeze(-2) - future_pos_grad).nan_to_num(0.0).pow(2).sum()
                grad = torch.autograd.grad(loss, x0_grad)[0]
            total_guidance = total_guidance - args.energy_cg * grad
        # 社会力-目的地引导力 CG 引导 (acc 应类似于社会力中的目标导向力)
        if args.sfm_des_cg > 0:
            desire_vel = F.normalize(des_now.unsqueeze(-2) - future_pos, dim=-1) * spd_now.unsqueeze(-2)  # (S*B, #pedestrian, pred_step, 2)
            des_force = (desire_vel - future_vel).nan_to_num(0.0) / args.t_des_force  # (S*B, #pedestrian, pred_step, 2)
            # loss = F.mse_loss(future_acc, des_force.detach())
            # grad = torch.autograd.grad(loss, x0)[0]
            grad = 2 * (future_acc - des_force) / args.scale_accelerate  # 可以直接手算
            total_guidance = total_guidance - args.sfm_des_cg * grad
        # 基于社会力，引导 acc 方向远离障碍物
        if args.sfm_map_cg > 0:
            # 场景障碍物排斥力
            F_map = get_force_map(r=args.r, A=args.a_map_force, B=args.d_map_force, device=args.device)  # (2r+1, 2r+1, 2)
            idx = future_pos[..., 0].sub(model.xmin).div(model.xmax - model.xmin).mul(model.map.shape[0]).round().long().clamp(0, model.map.shape[0] - 1)  # (S*B, #pedestrian, pred_step)
            jdx = future_pos[..., 1].sub(model.ymin).div(model.ymax - model.ymin).mul(model.map.shape[1]).round().long().clamp(0, model.map.shape[1] - 1)  # (S*B, #pedestrian, pred_step)
            patches = extract_patches_torch(model.map, idx.reshape(-1), jdx.reshape(-1), r=10).reshape(*idx.shape, 2*args.r+1, 2*args.r+1) # (S*B, #pedestrian, pred_step, 2r+1, 2r+1)
            map_force = (patches[..., None] * F_map).nan_to_num(0.0).flatten(-3, -2).sum(-2) # (S*B, #pedestrian, pred_step, 2)
            # loss = F.mse_loss(future_acc, map_force.detach())
            # grad = torch.autograd.grad(loss, x0)[0]
            grad = 2 * (future_acc - map_force) / args.scale_accelerate  # 可以直接手算
            total_guidance = total_guidance - args.sfm_map_cg * grad
        # 基于社会力，引导 acc 方向远离其他行人和车辆
        if args.sfm_social_cg > 0:
            # 其他行人排斥力
            p = future_pos[:, None, :, :, :] - future_pos[:, :, None, :, :] # (S*B, #focal-pedestrian, #other-pedestrian, pred_step, 2)
            d = torch.norm(p, dim=-1, keepdim=True)
            n = -p / d.clamp(min=1e-6)
            F_ped = args.a_ped_force * torch.exp(-d / args.d_ped_force) * n
            ped_force = F_ped.nan_to_num(0.0).sum(dim=1) # (S*B, #pedestrian, pred_step, 2)
            # 其它车辆排斥力 (车辆用最后一帧位置)
            p = veh_now[:, :, None, -1:, :] - future_pos[:, None, :, :, :] # (S*B, #vehicle, #pedestrian, pred_step, 2)
            d = torch.norm(p, dim=-1, keepdim=True)
            n = -p / d.clamp(min=1e-6)
            F_veh = args.a_veh_force * torch.exp(-d / args.d_veh_force) * n
            veh_force = F_veh.nan_to_num(0.0).sum(dim=1) # (S*B, #pedestrian, pred_step, 2)
            # 合力
            social_force = ped_force + veh_force  # (S*B, #pedestrian, pred_step, 2)
            # loss = F.mse_loss(future_acc, social_force.detach())
            # grad = torch.autograd.grad(loss, x_in)[0]
            grad = 2 * (future_acc - social_force) / args.scale_accelerate  # 可以直接手算
            total_guidance = total_guidance - args.sfm_social_cg * grad
        x0 = x0 + total_guidance

        ## 去噪
        xt = diffusion.denoise(xt, t, x0=x0, stride=min(stride, t))
    acc_new = xt / args.scale_accelerate
    
    if args.use_sfm:
        future_vel = vel_now.unsqueeze(-2)
        future_pos = pos_now.unsqueeze(-2)
        # 目的地导向力
        desire_vel = F.normalize(des_now.unsqueeze(-2) - future_pos, dim=-1) * spd_now.unsqueeze(-2)  # (S*B, #pedestrian, pred_step, 2)
        des_force = (desire_vel - future_vel).nan_to_num(0.0) / args.t_des_force  # (S*B, #pedestrian, pred_step, 2)
        # 场景障碍物排斥力
        F_map = get_force_map(r=args.r, A=args.a_map_force, B=args.d_map_force, device=args.device)  # (2r+1, 2r+1, 2)
        idx = future_pos[..., 0].sub(model.xmin).div(model.xmax - model.xmin).mul(model.map.shape[0]).round().long().clamp(0, model.map.shape[0] - 1)  # (S*B, #pedestrian, pred_step)
        jdx = future_pos[..., 1].sub(model.ymin).div(model.ymax - model.ymin).mul(model.map.shape[1]).round().long().clamp(0, model.map.shape[1] - 1)  # (S*B, #pedestrian, pred_step)
        patches = extract_patches_torch(model.map, idx.reshape(-1), jdx.reshape(-1), r=10).reshape(*idx.shape, 2*args.r+1, 2*args.r+1) # (S*B, #pedestrian, pred_step, 2r+1, 2r+1)
        map_force = (patches[..., None] * F_map).nan_to_num(0.0).flatten(-3, -2).sum(-2) # (S*B, #pedestrian, pred_step, 2)
        # 其他行人排斥力
        p = future_pos[:, None, :, :, :] - future_pos[:, :, None, :, :] # (S*B, #focal-pedestrian, #other-pedestrian, pred_step, 2)
        d = torch.norm(p, dim=-1, keepdim=True)
        n = -p / d.clamp(min=1e-6)
        F_ped = args.a_ped_force * torch.exp(-d / args.d_ped_force) * n
        ped_force = F_ped.nan_to_num(0.0).sum(dim=1) # (S*B, #pedestrian, pred_step, 2)
        # 其它车辆排斥力 (车辆用最后一帧位置)
        p = veh_now[:, :, None, -1:, :] - future_pos[:, None, :, :, :] # (S*B, #vehicle, #pedestrian, pred_step, 2)
        d = torch.norm(p, dim=-1, keepdim=True)
        n = -p / d.clamp(min=1e-6)
        F_veh = args.a_veh_force * torch.exp(-d / args.d_veh_force) * n
        veh_force = F_veh.nan_to_num(0.0).sum(dim=1) # (S*B, #pedestrian, pred_step, 2)
        # 阻尼力
        vel_force = -args.vel_damping * future_vel  # (S*B, #pedestrian, pred_step, 2)
        # 合力
        acc_new = des_force + map_force + ped_force + veh_force + vel_force

    vel_new = vel_now.unsqueeze(-2) + acc_new.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
    pos_new = pos_now.unsqueeze(-2) + vel_new.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
    veh_new = torch.from_numpy(
        df_veh
        .loc[frame+1:frame+args.pred_step]  # pandas 中的切片是闭区间，因此实际上切出来了 pred_step 帧
        .unstack().swaplevel(axis='columns').sort_index(axis='columns')
        .reindex(
            index=range(frame+1, frame+args.pred_step+1),
            columns=pd.MultiIndex.from_product([veh_list, ['x', 'y']])
        )
        .values.reshape(args.pred_step, len(veh_list), 2) # (pred_step, #vehicle, 2)
        .transpose(1, 0, 2) # (#vehicle, pred_step, 2)
    ).to(device=args.device, dtype=torch.float32)
    des_new = des_now  # (S*B, #pedestrian, 2)
    spd_new = spd_now  # (S*B, #pedestrian, 1)
    # _logger.info(f"  Computed new positions and velocities.")

    hst_now = torch.cat([hst_now, pos_now.unsqueeze(-2), pos_new], dim=-2)[:, :, -args.hist_step-1:-1, :] # (S*B, #pedestrian, hist_step, 2)
    veh_now = torch.cat([veh_now, veh_new.unsqueeze(0).repeat(S, 1, 1, 1)], dim=-2)[:, :, -args.hist_step-1:, :] # (S*B, #vehicle, hist_step + 1, 2)
    pos_now = pos_new[:, :, -1, :] # (S*B, #pedestrian, 2)
    vel_now = vel_new[:, :, -1, :] # (S*B, #pedestrian, 2)
    spd_now = spd_new  # (S*B, #pedestrian, 1)
    des_now = des_new  # (S*B, #pedestrian, 2)
    # _logger.info(f"  Updated states for next step.")

    # _logger.info(f"  Converting new positions to CPU numpy. {pos_new.shape}, {type(pos_new)}")
    df_ped_new = pd.DataFrame([
        {
            'f': frame + 1 + f,
            'id': ped_list[i],
            'type': 'pedestrian',
            'x': float(pos_new[0, i, f, 0]),
            'y': float(pos_new[0, i, f, 1]),
        }
        for i in range(pos_new.shape[1])
        for f in range(pos_new.shape[2])
    ])
    # _logger.info(f"  Converted new positions to CPU numpy. {pos_new.shape}, {type(pos_new)}")
    df_veh_new = df_veh.loc[frame+1:frame+args.pred_step].reset_index().assign(type='vehicle')
    df_new = pd.concat([df_ped_new, df_veh_new], ignore_index=True)
    frame = frame + args.pred_step
    # _logger.info(f"Completed simulation up to frame {frame}.")

    return df_new, (ped_list, veh_list, df_veh, frame, pos_now, vel_now, hst_now, des_now, spd_now, veh_now)
