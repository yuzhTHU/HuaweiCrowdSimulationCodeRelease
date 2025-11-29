import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from argparse import Namespace
from src.model.model import Model
from src.diffusion import DDPM
from src.dataset import BaseDataset


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
        .loc[df_ped.index.get_level_values('id').isin(ped_list)]
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
        # 试试 CFG
        if (des_cfg := getattr(args, 'des_cfg', 0.0)) > 0:
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
            x0 = x0 + des_cfg * (x0 - x0_wo_des)
        # 试试 CFG
        if (map_cfg := getattr(args, 'map_cfg', 0.0)) > 0:
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
            x0 = x0 + map_cfg * (x0 - x0_wo_map)
        # 引导 acc 方向指向 pos_now 与 des_now 连线的方向
        if (des_guidance := getattr(args, 'des_guidance', 0.0)) > 0:
            direction = F.normalize(des_now - pos_now, dim=-1).nan_to_num(0.0).unsqueeze(-2) # (S*B, #pedestrian, 1, 2)
            x0 = x0 + des_guidance * direction
        # 基于能量，以 pos_now 与 des_now 的距离作为能量项对 acc 进行引导
        if (energy_guidance := getattr(args, 'energy_guidance', 0.0)) > 0:
            direction = (des_now - pos_now).nan_to_num(0.0).unsqueeze(-2)  # (S*B, #pedestrian, 1, 2)
            x0 = x0 + energy_guidance * direction
        # 基于能量，但是以 pos_now + vel_now * dt + 0.5 * acc * dt^2 与 des_now 的距离作为能量项对 acc 进行引导
        if (energy2_guidance := getattr(args, 'energy2_guidance', 0.0)) > 0:
            future_acc = x0 / args.scale_accelerate  # (S*B, #pedestrian, pred_step, 2)
            future_vel = vel_now.unsqueeze(-2) + future_acc.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
            future_pos = pos_now.unsqueeze(-2) + future_vel.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
            direction = (des_now.unsqueeze(-2) - future_pos).nan_to_num(0.0)  # (S, ped, pred_step, 2)
            x0 = x0 + energy2_guidance * direction
        # 基于社会力，以 (spd_now * normalize(des_now - pos_now) - vel_now) / 0.5 对 acc 进行引导
        if (sfm_guidance := getattr(args, 'sfm_guidance', 0.0)) > 0:
            future_acc = x0 / args.scale_accelerate  # (S*B, #pedestrian, pred_step, 2)
            future_vel = vel_now.unsqueeze(-2) + future_acc.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
            future_pos = pos_now.unsqueeze(-2) + future_vel.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
            desire_vel = F.normalize(des_now.unsqueeze(-2) - future_pos, dim=-1).nan_to_num(0.0) * spd_now.unsqueeze(-2)  # (S*B, #pedestrian, pred_step, 2)
            direction = (desire_vel - future_vel) / 0.5  # (S*B, #pedestrian, pred_step, 2)
            x0 = x0 + sfm_guidance * direction

        ## 去噪
        xt = diffusion.denoise(xt, t, x0=x0, stride=min(stride, t))
    # _logger.info(f"  Denoising completed.")
    
    acc_new = xt / args.scale_accelerate
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
