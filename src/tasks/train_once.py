import torch
import random
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

_logger = logging.getLogger(__name__)


def train_once(
    args,
    train_loaders: List[D.DataLoader],
    model: Model,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    diffusion: DDPM,
    epoch: int,
) -> Dict:
    """ 
    进行单步训练

    Args:
        args: 全局参数
        train_loaders: 训练数据加载器列表
        model: 待训练模型
        optimizer: 优化器
        criterion: 损失函数
        diffusion: 扩散模型
        epoch: 当前训练轮数
    
    Returns:
        all_records: 训练记录字典
    """
    train_timer = NamedTimer(unit='it', mode='pace')
    records_list = []
    for loader in train_loaders:
        map_data = loader.dataset.map_data
        map = torch.from_numpy(map_data.map).to(args.device).float()
        records = dict(loss=[], rollout_loss=[])
        for batch in tqdm(loader, total=len(loader), disable=False, leave=False, dynamic_ncols=True):
            optimizer.zero_grad()
            pos = batch['pos'].to(args.device)  # (batch_size, #pedestrian, 2)
            vel = batch['vel'].to(args.device)  # (batch_size, #pedestrian, 2)
            hst = batch['hst'].to(args.device)  # (batch_size, #pedestrian, hist_step, 2)
            des = batch['des'].to(args.device)  # (batch_size, #pedestrian, 2)
            spd = batch['spd'].to(args.device)  # (batch_size, #pedestrian)
            veh = batch['veh'].to(args.device)  # (batch_size, #vehicle, hist_step + 1, 2)
            future_acc = batch['future_acc'].to(args.device)  # (batch_size, #pedestrian, pred_step*roll_step, 2)
            future_pos = batch['future_pos'].to(args.device)  # (batch_size, #pedestrian, pred_step*roll_step, 2)
            future_veh = batch['future_veh'].to(args.device)  # (batch_size, #vehicle, pred_step*roll_step, 2)
            ped_length = batch['ped_length'].to(args.device)  # (batch_size,)
            veh_length = batch['veh_length'].to(args.device)  # (batch_size,)
            train_timer.add('prepare data')

            # Rollout
            pos_now = pos
            vel_now = vel
            hst_now = hst
            des_now = des
            spd_now = spd
            veh_now = veh
            rollout_loss = []
            for step in range(args.multi_frame_rollout):
                # DDPM forward
                # pos_true = future_pos[:, :, args.pred_step*step:args.pred_step*(step+1), :] # (B, #pedestrian, pred_step, 2)
                # vel_true = pos_true.diff(dim=-2, prepend=pos_now.unsqueeze(-2)) * args.fps  # (B, #pedestrian, roll_step*pred_step, 2)
                # acc_true = vel_true.diff(dim=-2, prepend=vel_now.unsqueeze(-2)) * args.fps  # (B, #pedestrian, roll_step*pred_step, 2)
                acc_true = future_acc[:, :, args.pred_step*step:args.pred_step*(step+1), :] # (B, #pedestrian, pred_step, 2)
                noisy_acc, noise_true, denoise_t = diffusion.add_noise(acc_true * args.scale_accelerate)
                train_timer.add('add noise')

                if args.p_drop_map and random.random() < args.p_drop_map:
                    map = torch.full_like(map, torch.nan, device=map.device)
                if args.p_drop_destination and random.random() < args.p_drop_destination:
                    des_now = torch.full_like(des_now, torch.nan, device=des_now.device)
                if args.p_drop_speed and random.random() < args.p_drop_speed:
                    spd_now = torch.full_like(spd_now, torch.nan, device=spd_now.device)

                # DDPM backward
                model.set_map_embedding(
                    map=map,
                    xmin=map_data.xmin,
                    xmax=map_data.xmax,
                    ymin=map_data.ymin,
                    ymax=map_data.ymax,
                )
                model.set_veh_embedding(veh=veh_now)
                model.set_ped_embedding(pos=pos_now, vel=vel_now, hst=hst_now, des=des_now, spd=spd_now)
                model.set_sur_info()
                output = model(
                    noisy_acc=noisy_acc, 
                    denoise_t=denoise_t,
                    ped_length=ped_length, 
                    veh_length=veh_length,
                )  # (B, #pedestrian, pred_step, 2)
                train_timer.add('forward')

                # Compute Loss
                if args.predict_noise:
                    noise_pred = output
                    acc_pred = diffusion.noise_to_x0(xt=noisy_acc, denoise_t=denoise_t, noise=noise_pred) / args.scale_accelerate
                else:
                    acc_pred = output / args.scale_accelerate

                if args.loss_type == 'accelerate':
                    loss = criterion(acc_pred, acc_true)
                elif args.loss_type == 'position':
                    # acc_true 是从 pos_true 算出来的，因此不用再返回去计算 pos_true 了
                    vel_true = vel_now.unsqueeze(-2) + acc_true.cumsum(dim=-2) / args.fps
                    pos_true = pos_now.unsqueeze(-2) + vel_true.cumsum(dim=-2) / args.fps
                    vel_pred = vel_now.unsqueeze(-2) + acc_pred.cumsum(dim=-2) / args.fps
                    pos_pred = pos_now.unsqueeze(-2) + vel_pred.cumsum(dim=-2) / args.fps
                    loss = criterion(pos_pred, pos_true)
                elif args.loss_type == 'noise':
                    if not args.predict_noise:
                        raise ValueError("When using noise prediction loss, the model must predict noise!")
                    loss = criterion(noise_pred, noise_true)
                else:
                    raise ValueError(f"Unknown loss type {args.loss_type}!")
                rollout_loss.append(loss.detach().cpu().tolist())
                train_timer.add('compute loss')
                (args.rollout_lambda ** (args.multi_frame_rollout - step) * loss).backward()

                acc_new = acc_pred.detach()  # (B, #pedestrian, pred_step, 2)
                vel_new = vel_now.unsqueeze(-2) + acc_new.cumsum(dim=-2) / args.fps  # (B, #pedestrian, pred_step, 2)
                pos_new = pos_now.unsqueeze(-2) + vel_new.cumsum(dim=-2) / args.fps  # (B, #pedestrian, pred_step, 2)
                veh_new = future_veh[:, :, step*args.pred_step:(step+1)*args.pred_step, :]  # (B, #vehicle, pred_step, 2)

                hst_now = torch.cat([hst_now, pos_now.unsqueeze(-2), pos_new], dim=-2)[:, :, -args.hist_step-1:-1, :] # (B, #pedestrian, hist_step, 2)
                veh_now = torch.cat([veh_now, veh_new], dim=-2)[:, :, -args.hist_step-1:, :]  # (B, #vehicle, hist_step + 1, 2)
                pos_now = pos_new[:, :, -1, :] # (B, #pedestrian, 2)
                vel_now = vel_new[:, :, -1, :] # (B, #pedestrian, 2)
                train_timer.add('rollout')

            ## Backpropagate
            optimizer.step()
            records['loss'].extend([loss.item()] * acc_true.shape[0])
            records['rollout_loss'].extend([rollout_loss] * acc_true.shape[0])
            train_timer.add('backpropagate')
        records_list.append(records)
        _logger.debug(
            f"[Epoch {epoch}/{args.epochs}] Train on {loader.dataset.name}: "
            f"Loss={np.mean(records['loss']):.4f}"
        )
    all_records = {
        'epoch': epoch,
        'dataset_names': [loader.dataset.name for loader in train_loaders],
        'sample_nums': [len(loader.dataset) for loader in train_loaders],
    }
    for records in records_list:
        for k, v in records.items():
            if k not in all_records:
                all_records[k] = []
            if isinstance(v[0], (int, float)):
                mean_v = np.mean(v)
            else: 
                mean_v = np.mean(v, axis=0).tolist()
            all_records[k].append(mean_v)
    _logger.info(tag2ansi(
        f"[#66CCFF][Epoch {epoch}/{args.epochs}] "
        f"[#66CCFF]Loss={np.mean(all_records['loss']):.4f} "
        f"[#66CCFF]Rollout Loss={np.mean(all_records['rollout_loss'], axis=0).round(4).tolist()} "
        f"[#66CCFF]Time={train_timer}"
    ))
    return all_records
