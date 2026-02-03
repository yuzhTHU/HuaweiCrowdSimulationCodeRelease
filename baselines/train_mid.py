import re
import sys
import json
import yaml
import time
import dill
import torch
import shlex
import random
import logging
import numpy as np
import torch.nn as nn
import torch.utils.data as D
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from socket import gethostname
from setproctitle import setproctitle
from argparse import ArgumentParser, Namespace
from scipy.spatial.distance import pdist, squareform
from src.dataset import ETHDataset, UCYDataset, SDDDataset, GCDataset, WayMoDataset, ORCADataset
from src.model import Model, RelativeModel, NewModel
from src.diffusion import DDPM, DDIM
from src.utils.logger import init_logger
from src.utils.seed import seed_all
from src.utils.timer import NamedTimer
from src.utils.auto_gpu import AutoGPU
from src.utils.fix_parser import add_negation_flags, add_minus_flags
from src.utils.tag2ansi import tag2ansi
from src.utils.calc_xy_error import calc_xy_error
from src.utils.use_npu import USE_NPU, npu_attention_fallback_context

from baselines.mid.models.trajectron import Trajectron
from baselines.mid.utils.model_registrar import ModelRegistrar
from baselines.mid.utils.trajectron_hypers import get_traj_hypers
from baselines.mid.environment.scene_graph import TemporalSceneGraph
from baselines.mid.environment.node_type import NodeType
from baselines.mid.models.autoencoder import AutoEncoder
from baselines.mid.models.encoders.mgcvae import MultimodalGenerativeCVAE
from baselines.mid.environment.environment import Environment
from baselines.mid.environment.scene import Scene

_logger = logging.getLogger("src.train_spdiff")


def train_once(
    args,
    train_loaders: List[D.DataLoader],
    model: Model,
    optimizer: torch.optim.Optimizer,
    scheduler,
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
        records = dict(loss=[])
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

            batch_size, ped_num, _ = pos.shape
            mask = torch.arange(ped_num, device=args.device).expand(batch_size, ped_num) < ped_length.unsqueeze(-1)  # (B, #pedestrian)
            history = torch.zeros(hst.shape[0], hst.shape[1], hst.shape[2]+1, 6, device=args.device)
            history[:, :, :-1, :2] = (hst_pos := hst)
            history[:, :, 1:-1, 2:4] = (hst_vel := hst_pos.diff(axis=-2) * args.fps)
            history[:, :, 2:-1, 4:6] = (hst_acc := hst_vel.diff(axis=-2) * args.fps)
            history[:, :, -1, :2] = pos
            history[:, :, -1, 2:4] = vel
            history[:, :, -1, 4:6] = (vel - (hst[:, :, -1, :] - hst[:, :, -2, :]) * args.fps) * args.fps
            history = history[:, :, 1:, :]
            history = history.nan_to_num(0.0)
            future = future_pos.nan_to_num(0.0)
            x_t_list = []
            x_st_t_list = []
            y_t_list = []
            y_st_t_list = []
            neighbors_data_st_list = []
            neighbors_edge_value_list = []
            for batch_idx in range(pos.shape[0]):
                history_t = history[batch_idx, mask[batch_idx], :, :]
                future_t = future[batch_idx, mask[batch_idx], :, :]
                dist_cube = np.stack([
                    squareform(pdist(history_t[:, -3, :].cpu().numpy(), metric='euclidean')),
                    squareform(pdist(history_t[:, -2, :].cpu().numpy(), metric='euclidean')),
                    squareform(pdist(history_t[:, -1, :].cpu().numpy(), metric='euclidean'))
                ], axis=0)
                adj_cube = (dist_cube < 3.0).astype(int)
                weight_cube = np.divide(1., dist_cube, out=np.zeros_like(dist_cube), where=(dist_cube>0))
                edge_scaling = TemporalSceneGraph.calculate_edge_scaling(adj_cube, edge_addition_filter=[0.25, 0.5, 0.75, 1.0], edge_removal_filter=[1.0, 0.0])
                connection_mask = edge_scaling > 1e-2
                weight_cube = torch.from_numpy(weight_cube).float().to(args.device)
                connection_mask = torch.from_numpy(connection_mask).float().to(args.device)
                x_t = history_t  # (num_pedestrian, hist_step, 6)
                x_st_t = x_t - x_t[:, (-1,), :]; x_st_t[:, :2] /= 3  # (num_pedestrian, hist_step, 6)
                y_t = future_t  # (num_pedestrian, pred_step, 2)
                y_st_t = y_t - y_t[:, (0,), :2]; y_st_t[:, :2] /= 2  # (num_pedestrian, pred_step, 2)
                neighbors_data_st = [[] for i in range(mask[batch_idx].sum())]
                neighbors_edge_value = [[] for i in range(mask[batch_idx].sum())]
                for i, j in connection_mask[0, :, :].nonzero():
                    neighbors_data_st[i].append(x_st_t[j])
                    neighbors_edge_value[i].append(weight_cube[-1, i, j])
                x_t_list.extend(list(x_t))
                x_st_t_list.extend(list(x_st_t))
                y_t_list.extend(list(y_t))
                y_st_t_list.extend(list(y_st_t))
                neighbors_data_st_list.extend(neighbors_data_st)
                neighbors_edge_value_list.extend([torch.stack(x) if len(x) else torch.tensor([]) for x in neighbors_edge_value])
            x_t = torch.stack(x_t_list, dim=0)  # (batch_size * num_pedestrian, hist_step, 6)
            x_st_t = torch.stack(x_st_t_list, dim=0)  # (batch_size * num_pedestrian, hist_step, 6)
            y_t = torch.stack(y_t_list, dim=0)  # (batch_size * num_pedestrian, pred_step, 2)
            y_st_t = torch.stack(y_st_t_list, dim=0)  # (batch_size * num_pedestrian, pred_step, 2)
            node_type = NodeType('PEDESTRIAN', 1)
            neighbors_data_st = {(node_type, node_type): neighbors_data_st_list}
            neighbors_edge_value = {(node_type, node_type): neighbors_edge_value_list}
            first_history_index = torch.zeros(x_t.shape[0], dtype=torch.long).to(args.device)
            robot_traj_st_t = None
            mid_map = None
            batch = (
                first_history_index,  # 0
                x_t,  # (batch_size, hist-step, 6)
                y_t,  # (batch_size, pred-step, 2)
                x_st_t,  # (batch_size, hist-step, 6) 
                y_st_t,  # (batch_size, pred-step, 2)
                neighbors_data_st, # (batch_size, hist-step, 6)
                neighbors_edge_value, #
                robot_traj_st_t, # None
                mid_map # None
            )
            feat_x_encoded = model.encode(batch, node_type) # B * 64
            loss = model.diffusion.get_loss(y_t.cuda(), feat_x_encoded)
            train_timer.add('forward')
            
            ## Backpropagate
            loss.backward()
            optimizer.step()
            records['loss'].extend([loss.item()] * future_acc.shape[0])
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
        f"[#66CCFF]Time={train_timer}"
    ))
    return all_records


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
        records = dict(loss=[], ade=[], fde=[], trajlen=[], ped_num=[], veh_num=[], rollout_time=[], collision_ped=[], collision_veh=[], collision_map=[])
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
            test_timer.add('prepare data', n=0)

            start_time = time.time()

            batch_size, ped_num, _ = pos.shape
            mask = torch.arange(ped_num, device=args.device).expand(batch_size, ped_num) < ped_length.unsqueeze(-1)  # (B, #pedestrian)
            history = torch.zeros(hst.shape[0], hst.shape[1], hst.shape[2]+1, 6, device=args.device)
            history[:, :, :-1, :2] = (hst_pos := hst)
            history[:, :, 1:-1, 2:4] = (hst_vel := hst_pos.diff(axis=-2) * args.fps)
            history[:, :, 2:-1, 4:6] = (hst_acc := hst_vel.diff(axis=-2) * args.fps)
            history[:, :, -1, :2] = pos
            history[:, :, -1, 2:4] = vel
            history[:, :, -1, 4:6] = (vel - (hst[:, :, -1, :] - hst[:, :, -2, :]) * args.fps) * args.fps
            history = history[:, :, 1:, :]
            history = history.nan_to_num(0.0)
            future = future_pos.nan_to_num(0.0)
            x_t_list = []
            x_st_t_list = []
            y_t_list = []
            y_st_t_list = []
            neighbors_data_st_list = []
            neighbors_edge_value_list = []
            for batch_idx in range(pos.shape[0]):
                history_t = history[batch_idx, mask[batch_idx], :, :]
                future_t = future[batch_idx, mask[batch_idx], :, :]
                dist_cube = np.stack([
                    squareform(pdist(history_t[:, -3, :].cpu().numpy(), metric='euclidean')),
                    squareform(pdist(history_t[:, -2, :].cpu().numpy(), metric='euclidean')),
                    squareform(pdist(history_t[:, -1, :].cpu().numpy(), metric='euclidean'))
                ], axis=0)
                adj_cube = (dist_cube < 3.0).astype(int)
                weight_cube = np.divide(1., dist_cube, out=np.zeros_like(dist_cube), where=(dist_cube>0))
                edge_scaling = TemporalSceneGraph.calculate_edge_scaling(adj_cube, edge_addition_filter=[0.25, 0.5, 0.75, 1.0], edge_removal_filter=[1.0, 0.0])
                connection_mask = edge_scaling > 1e-2
                weight_cube = torch.from_numpy(weight_cube).float().to(args.device)
                connection_mask = torch.from_numpy(connection_mask).float().to(args.device)
                x_t = history_t  # (num_pedestrian, hist_step, 6)
                x_st_t = x_t - x_t[:, (-1,), :]; x_st_t[:, :2] /= 3  # (num_pedestrian, hist_step, 6)
                y_t = future_t  # (num_pedestrian, pred_step, 2)
                y_st_t = y_t - y_t[:, (0,), :2]; y_st_t[:, :2] /= 2  # (num_pedestrian, pred_step, 2)
                neighbors_data_st = [[] for i in range(mask[batch_idx].sum())]
                neighbors_edge_value = [[] for i in range(mask[batch_idx].sum())]
                for i, j in connection_mask[0, :, :].nonzero():
                    neighbors_data_st[i].append(x_st_t[j])
                    neighbors_edge_value[i].append(weight_cube[-1, i, j])
                x_t_list.extend(list(x_t))
                x_st_t_list.extend(list(x_st_t))
                y_t_list.extend(list(y_t))
                y_st_t_list.extend(list(y_st_t))
                neighbors_data_st_list.extend(neighbors_data_st)
                neighbors_edge_value_list.extend([torch.stack(x) if len(x) else torch.tensor([]) for x in neighbors_edge_value])
            x_t = torch.stack(x_t_list, dim=0)  # (batch_size * num_pedestrian, hist_step, 6)
            x_st_t = torch.stack(x_st_t_list, dim=0)  # (batch_size * num_pedestrian, hist_step, 6)
            y_t = torch.stack(y_t_list, dim=0)  # (batch_size * num_pedestrian, pred_step, 2)
            y_st_t = torch.stack(y_st_t_list, dim=0)  # (batch_size * num_pedestrian, pred_step, 2)
            node_type = NodeType('PEDESTRIAN', 1)
            neighbors_data_st = {(node_type, node_type): neighbors_data_st_list}
            neighbors_edge_value = {(node_type, node_type): neighbors_edge_value_list}
            first_history_index = torch.zeros(x_t.shape[0], dtype=torch.long).to(args.device)
            robot_traj_st_t = None
            mid_map = None
            batch = (
                first_history_index,  # 0
                x_t,  # (batch_size, hist-step, 6)
                y_t,  # (batch_size, pred-step, 2)
                x_st_t,  # (batch_size, hist-step, 6) 
                y_st_t,  # (batch_size, pred-step, 2)
                neighbors_data_st, # (batch_size, hist-step, 6)
                neighbors_edge_value, #
                robot_traj_st_t, # None
                mid_map # None
            )
            traj_pred = model.generate(batch, node_type, num_points=12, sample=args.sample_num, bestof=True, sampling='ddim', step=100//5) # B * 20 * 12 * 2
            cnt = 0
            pos_pred = np.zeros((args.sample_num, batch_size, ped_num, args.roll_step*args.pred_step, 2))
            for i, m in enumerate(mask.cpu().numpy()):
                n = m.sum()
                pos_pred[:, i, m, :, :] = traj_pred[:, :n, :, :]
                cnt += n
            pos_pred = torch.from_numpy(pos_pred).to(args.device)
            rollout_time = (time.time() - start_time)

            batch_size, ped_num, _, _ = future_acc.shape
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
            records['loss'].extend([np.nan] * future_acc.shape[0]) # List[float]
            # 计算 distance error
            dis_err = (pos_pred - pos_true).norm(dim=-1) # (S, B, #pedestrian, roll_step*pred_step)
            test_timer.add('evaluate')
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
            records['norm_err'], records['tan_err'] = calc_xy_error(traj_diff, ped_pos, veh_pos, veh_vel)
            # 计算碰撞数
            S, B, P, T, _ = pos_pred.shape
            flat_pos = pos_pred.permute(0, 1, 3, 2, 4).reshape(-1, P, 2)  ## 将 (S, B, P, T, 2) -> (S, B, T, P, 2) -> (S*B*T, P, 2)
            dist_matrix = torch.cdist(flat_pos, flat_pos, p=2)  # (S*B*T, P, P)
            eye_matrix = torch.eye(P, device=pos_pred.device, dtype=torch.bool).unsqueeze(0)
            # 只有当行人 i 和行人 j 都有效 (mask=True) 时才计入碰撞
            mask_expanded = mask.unsqueeze(0).unsqueeze(2).expand(S, -1, T, -1).reshape(-1, P) # (B, P) -> (1, B, 1, P) -> (S, B, T, P) -> (S*B*T, P)
            valid_pair_mask = mask_expanded.unsqueeze(2) & mask_expanded.unsqueeze(1)
            # fiends = (dist_matrix.reshape(S, B, T, P, P) < args.collision_threshold).float().mean(axis=2)
            collision_matrix = (
                (dist_matrix < args.collision_threshold) &
                (~eye_matrix) &
                valid_pair_mask
            )
            collision_rate = collision_matrix.sum() / (S * mask.sum() * T)
            records['collision_ped'].append(collision_rate.item())

            if future_veh.shape[1] == 0:
                records['collision_veh'].append(float('nan'))
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
                records['collision_veh'].append(collision_rate.item())

            if not np.isfinite(map_data.map).any():
                records['collision_map'].append(float('nan'))
            else:
                idx = pos_pred[..., 0].sub(map_data.xmin).div(map_data.xmax-map_data.xmin).mul(map_data.map.shape[0]).round().long().clamp(0, map_data.map.shape[0] - 1)  # (batch_size, #pedestrian)
                jdx = pos_pred[..., 1].sub(map_data.ymin).div(map_data.ymax-map_data.ymin).mul(map_data.map.shape[1]).round().long().clamp(0, map_data.map.shape[1] - 1)  # (batch_size, #pedestrian)
                sur_info = map_data.map[idx.cpu().numpy(), jdx.cpu().numpy()] # (batch_size, #pedestrian)
                collision_rate = (sur_info > 0.9).mean()
                records['collision_map'].append(collision_rate.item())
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
            f"[#66CCFF]Collision-Ped={np.mean(records['collision_ped']):.2%}, "
            f"[#66CCFF]Collision-Veh={np.mean(records['collision_veh']):.2%}, "
            f"[#66CCFF]Collision-Map={np.mean(records['collision_map']):.2%}, "
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
        f"[#66CCFF]Collision-Ped={np.sum(w * all_records['collision_ped']):.2%}, "
        f"[#66CCFF]Collision-Veh={np.sum(w * all_records['collision_veh']):.2%}, "
        f"[#66CCFF]Collision-Map={np.sum(w * all_records['collision_map']):.2%}, "
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
            rollout_time = np.array([all_records['rollout_time'][i] for i in idxs])
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
                f"[#66CCFF]Collision-Ped={np.sum(w * all_records['collision_ped']):.2%}, "
                f"[#66CCFF]Collision-Veh={np.sum(w * all_records['collision_veh']):.2%}, "
                f"[#66CCFF]Collision-Map={np.sum(w * all_records['collision_map']):.2%}, "
                f"[#66CCFF]AvgLen={np.sum(w * trajlen):.4f}, "
                f"[#66CCFF]PedNum={np.sum(w * ped_num):.4f}, "
                f"[#66CCFF]VehNum={np.sum(w * veh_num):.4f}, "
                f"[#66CCFF]RolloutTime={np.mean(rollout_time)*1000:.2f}ms "
                f"([bold underline orange]FPS={1/np.mean(rollout_time):.2f} Hz[reset])"
            ))
    return all_records


def main(args):
    ## Load Dataset
    if args.test_datasets:
        train_dataset = []
        test_dataset = []
        for (datasets, dataset_names) in zip((train_dataset, test_dataset), (args.train_datasets, args.test_datasets)):
            for dataset in dataset_names:
                if dataset == 'eth':
                    datasets.append(ETHDataset.load_data(args, "./data/ETH/seq_eth/obsmat.txt"))
                elif dataset == 'hotel':
                    datasets.append(ETHDataset.load_data(args, "./data/ETH/seq_hotel/obsmat.txt"))
                elif dataset == 'zara01':
                    datasets.append(UCYDataset.load_data(args, './data/UCY/data/data_zara/crowds_zara01.vsp'))
                elif dataset == 'zara02':
                    datasets.append(UCYDataset.load_data(args, './data/UCY/data/data_zara/crowds_zara02.vsp'))
                elif dataset == 'univ':
                    datasets.append(UCYDataset.load_data(args, './data/UCY/data/data_university_students/students003.vsp'))
                elif dataset == 'GC_train':
                    _gc_dataset = GCDataset.load_data(args, "./data/GC/Annotation") if '_gc_dataset' not in locals() else _gc_dataset
                    d = deepcopy(_gc_dataset)
                    test_ratio = 0.2
                    train_num = int(len(_gc_dataset) * (1-test_ratio))
                    d.name = d.name + f"_{100-test_ratio*100:.0f}train"
                    d.samples = d.samples[:train_num]
                    datasets.append(d)
                elif dataset == 'GC_test':
                    _gc_dataset = GCDataset.load_data(args, "./data/GC/Annotation") if '_gc_dataset' not in locals() else _gc_dataset
                    d = deepcopy(_gc_dataset)
                    test_ratio = 0.2
                    train_num = int(len(_gc_dataset) * (1-test_ratio))
                    d.name = d.name + f"_{test_ratio*100:.0f}test"
                    d.samples = d.samples[train_num:]
                    datasets.append(d)
                elif dataset == 'SDD_train':
                    for path in ['bookstore/video0','bookstore/video1','bookstore/video2','bookstore/video3','coupa/video0','coupa/video1','coupa/video2','deathCircle/video0','deathCircle/video1','deathCircle/video2','deathCircle/video3','gates/video0','gates/video1','gates/video2','gates/video3','gates/video4','gates/video5','gates/video6','gates/video7','hyang/video0','hyang/video1','hyang/video2','hyang/video3','hyang/video4','hyang/video5','hyang/video6','hyang/video7','hyang/video8','hyang/video9','hyang/video10','hyang/video11','hyang/video12','hyang/video13','little/video0','little/video1','little/video2','nexus/video0','nexus/video1','nexus/video2','nexus/video3','nexus/video4','nexus/video5','nexus/video6','nexus/video7','nexus/video8','quad/video0','quad/video1','quad/video2']:
                        datasets.append(SDDDataset.load_data(args, f"./data/SDD/annotations/{path}/annotations.txt"))
                elif dataset == 'SDD_test':
                    for path in ['bookstore/video4','bookstore/video5','bookstore/video6','coupa/video3','deathCircle/video4','gates/video8','hyang/video14','little/video3','nexus/video9','nexus/video10','nexus/video11','quad/video3']:
                        datasets.append(SDDDataset.load_data(args, f"./data/SDD/annotations/{path}/annotations.txt"))
                elif dataset == 'WayMo_train':
                    _waymo_datasets = WayMoDataset.load_data_batch(args, "./data/WayMo/Processed/", total=500) if '_waymo_datasets' not in locals() else _waymo_datasets
                    for d in _waymo_datasets:
                        d = deepcopy(d)
                        test_ratio = 0.2
                        train_num = int(len(d) * (1-test_ratio))
                        d.name = d.name + f"_{100-test_ratio*100:.0f}train"
                        d.samples = d.samples[:train_num]
                        datasets.append(d)
                elif dataset == 'WayMo_test':
                    _waymo_datasets = WayMoDataset.load_data_batch(args, "./data/WayMo/Processed/", total=500) if '_waymo_datasets' not in locals() else _waymo_datasets
                    for d in _waymo_datasets:
                        d = deepcopy(d)
                        test_ratio = 0.2
                        train_num = int(len(d) * (1-test_ratio))
                        d.name = d.name + f"_{test_ratio*100:.0f}test"
                        d.samples = d.samples[train_num:]
                        datasets.append(d)
                else:
                    raise ValueError(f"Unknown dataset {dataset}!")
    else:
        # 加载数据集
        dataset_list = []
        if 'All' in args.datasets:
            args.datasets.remove('All')
            args.datasets += ['ETH', 'UCY', 'GC', 'SDD', 'WayMo']
        if "ETH" in args.datasets:
            dataset_list += ETHDataset.load_data_batch(args, "./data/ETH/")
            args.datasets.remove("ETH")
        if "UCY" in args.datasets:
            dataset_list += UCYDataset.load_data_batch(args, "./data/UCY/data/")
            args.datasets.remove("UCY")
        if "GC" in args.datasets:
            dataset_list += [GCDataset.load_data(args, "./data/GC/Annotation")]
            args.datasets.remove("GC")
        if "SDD" in args.datasets:
            dataset_list += SDDDataset.load_data_batch(args, "./data/SDD/annotations/")
            args.datasets.remove("SDD")
        if 'WayMo' in args.datasets:
            dataset_list += WayMoDataset.load_data_batch(args, "./data/WayMo/Processed/", total=500)
            args.datasets.remove("WayMo")
        if 'ORCA' in args.datasets:
            dataset_list += ORCADataset.load_data_batch(args, "./data/ORCA/")
            args.datasets.remove("ORCA")
        if 'debug' in args.datasets:
            # dataset_list += [UCYDataset.load_data(args, './data/UCY/data/data_university_students/students003.vsp')]
            # dataset_list += [SDDDataset.load_data(args, "./data/SDD/annotations/hyang/video0/annotations.txt")]
            dataset_list += [WayMoDataset.load_data(args, './data/WayMo/Processed/00000_1_2aa43fad083efbf3/data.csv.gz')]
            # dataset_list[0].samples = dataset_list[0].samples[int(len(dataset_list[0].samples) * 0.8):]
            args.datasets.remove('debug')
        if len(args.datasets) > 0:
            raise ValueError(f"Unknown datase: {args.datasets}!")
        # 检查地图
        for dataset in dataset_list:
            map_data = dataset.map_data
            delta_x = map_data.xmax - map_data.xmin
            delta_y = map_data.ymax - map_data.ymin
            w, h = map_data.map.shape
            if not (0.8 < (ratio := (delta_x / w) / (delta_y / h)) < 1.2):
                _logger.warning(
                    f"Map aspect ratio of {dataset.name} mismatch: "
                    f"data ratio={ratio:.4f} (xrange={delta_x:.4f}, yrange={delta_y:.4f}, "
                    f"map shape={map_data.map.shape}), may cause distortion."
                )
                exit(1)
        # 划分训练集和测试集
        if args.test_name is not None:
            # 将名称中包含指定字符串的场景划分到测试集
            train_dataset = []
            test_dataset = []
            for d in dataset_list:
                if any(test_name in d.name for test_name in args.test_name):
                    test_dataset.append(d)
                else:
                    train_dataset.append(d)
        elif args.test_ratio is not None and args.split_by_scenario:
            # 将特定比例的场景划分到测试集
            random.shuffle(dataset_list)
            test_size = max(1, int(len(dataset_list) * args.test_ratio))
            train_dataset = dataset_list[:-test_size]
            test_dataset = dataset_list[-test_size:]
        elif args.test_ratio is not None and not args.split_by_scenario:
            # 将每个场景的特定比例样本划分到测试集
            train_dataset = []
            test_dataset = []
            for d in dataset_list:
                d1 = d
                d2 = deepcopy(d)
                train_num = int(len(d) * (1-args.test_ratio))
                d1.name = d1.name + f"_{100-args.test_ratio*100:.0f}train"
                d2.name = d2.name + f"_{args.test_ratio*100:.0f}test"
                d1.samples = d1.samples[:train_num]
                d2.samples = d2.samples[train_num:]
                train_dataset.append(d1)
                test_dataset.append(d2)
        else:
            # 训练集和测试集相同
            train_dataset = dataset_list
            test_dataset = dataset_list
    # 创建数据加载器
    train_loaders = []
    test_loaders = []
    for dataset in train_dataset:
        if len(dataset) == 0:
            _logger.warning(f"Dataset {dataset.name} has no training samples!")
            continue
        train_loaders.append(D.DataLoader(
            dataset,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=dataset.collate_fn,
        ))
    for dataset in test_dataset:
        if len(dataset) == 0:
            _logger.warning(f"Dataset {dataset.name} has no testing samples!")
            continue
        test_loaders.append(D.DataLoader(
            dataset,
            shuffle=False,
            batch_size=args.batch_size // args.sample_num,  # 在实际测试时 batch_size 会乘上 sample_num，可能会很大导致 OOM
            num_workers=args.num_workers,
            collate_fn=dataset.collate_fn,
        ))
    _logger.note(
        "Datasets:\n"
        f"Train on {[d.name for d in train_dataset]} datasets ({sum([len(d) for d in train_dataset]):,} samples in total)\n"
        f"Test on {[d.name for d in test_dataset]} datasets ({sum([len(d) for d in test_dataset]):,} samples in total)"
    )

    ## Load Model
    config = yaml.safe_load(open(args.config))
    config["exp_name"] = args.config.split("/")[-1].split(".")[0]
    # config["dataset"] = args.dataset
    config = Namespace(**config)
    config.device = args.device

    hyperparams = get_traj_hypers()
    hyperparams['enc_rnn_dim_edge'] = config.encoder_dim//2
    hyperparams['enc_rnn_dim_edge_influence'] = config.encoder_dim//2
    hyperparams['enc_rnn_dim_history'] = config.encoder_dim//2
    hyperparams['enc_rnn_dim_future'] = config.encoder_dim//2
    registrar = ModelRegistrar(args.save_dir, args.device)
    encoder = Trajectron(registrar, hyperparams, args.device)
    model = AutoEncoder(config, encoder = encoder).to(args.device)
    optimizer = torch.optim.Adam([
        {'params': registrar.get_all_but_name_match('map_encoder').parameters()},
        {'params': model.parameters()}
    ], lr=args.lr)
    node_type = NodeType('PEDESTRIAN', 1)
    scene = Scene(0, dt=1/args.fps)
    env = Environment(
        ['PEDESTRIAN'], 
        {'PEDESTRIAN': {'position': {'x': {'mean': 0, 'std': 1}, 'y': {'mean': 0, 'std': 1}}, 'velocity': {'x': {'mean': 0, 'std': 2}, 'y': {'mean': 0, 'std': 2}}, 'acceleration': {'x': {'mean': 0, 'std': 1}, 'y': {'mean': 0, 'std': 1}}}},
        [scene],
        attention_radius={(node_type, node_type): 3.0},
        robot_type=None
    )
    encoder.node_models_dict[node_type] = MultimodalGenerativeCVAE(
        env,
        node_type,
        registrar,
        hyperparams,
        args.device,
        [(node_type, node_type)]
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.98)
    criterion = torch.nn.MSELoss()
    if args.sampling_method == "DDIM":
        diffusion = DDIM(args)
    elif args.sampling_method == "DDPM":
        diffusion = DDPM(args, flexibility=0.0)
    else:
        raise ValueError(f"Unknown sampling_method {args.sampling_method}!")
    _logger.note(
        "Model Parameters:\n"
        f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n"
        f"Total: {sum(p.numel() for p in model.parameters()):,}"
    )

    ## Reload Checkpoint
    if args.reload_checkpoint is not None:
        # 如果指定了 checkpoint 路径，则从该路径加载
        checkpoint_path = Path(args.reload_checkpoint)
    elif (Path(args.save_path) / "checkpoint.pth").exists():
        # 如果当前保存路径下存在 checkpoint，则从该路径加载
        checkpoint_path = Path(args.save_path) / "checkpoint.pth"
    else:
        checkpoint_path = None
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found!")
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        start_epoch = checkpoint["epoch"] + 1
        if 'args' in checkpoint:
            saved_args = checkpoint['args']
            for key in sorted(set(saved_args.keys()) | set(vars(args).keys())):
                val1 = saved_args.get(key, None)
                val2 = getattr(args, key, None)
                if val1 != val2:
                    _logger.warning(
                        f"Argument '{key}' differs from the saved checkpoint: "
                        f"saved_args={val1} vs. current_args={val2}"
                    )
        model.load_state_dict(checkpoint["model"])
        _logger.note(tag2ansi(f"Checkpoint loaded from [underline green]{checkpoint_path}[reset], resume from epoch [underline green]{start_epoch}[reset]."))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            _logger.warning("Optimizer state not found in checkpoint, optimizer re-initialized.")
    else:
        start_epoch = 0

    ## Train
    timer = NamedTimer()
    for epoch in range(start_epoch, args.epochs+1):
        # 训练一个 epoch
        if epoch > 0:
            torch.set_grad_enabled(True)
            model.train()
            train_records = train_once(args, train_loaders, model, optimizer, scheduler, criterion, diffusion, epoch)
            timer.add('train')
        else:
            train_records = None

        # 测试一个 epoch
        if (epoch > 0 and not epoch % args.test_per_epoch) or (epoch == 0 and args.test_before_train):
            torch.set_grad_enabled(False)
            model.eval()
            with npu_attention_fallback_context(model, enable=USE_NPU):
                test_records = test_once(args, test_loaders, model, criterion, diffusion, epoch)
            timer.add('test')
        else:
            test_records = None

        # 保存日志
        with open(f"{args.save_path}/records.jsonl", "a") as f:
            if train_records is not None:
                f.write(json.dumps(train_records) + "\n")
            if test_records is not None:
                f.write(json.dumps(test_records) + "\n")

        # 保存加载点
        if '_last_checkpoint_time' not in locals() or (datetime.now() - _last_checkpoint_time).seconds  > 300:
            # 只在间隔超过 5 min 时保存
            _last_checkpoint_time = datetime.now()
            save_path = f"{args.save_path}/checkpoint.pth"
            torch.save({
                "epoch": epoch,
                "args": vars(args),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, save_path)
            _logger.info(tag2ansi(f"Checkpoint saved to [underline green]{save_path}[reset]."))
            timer.add('save_checkpoint')
        
        # 定期保存
        if set(str(epoch)[1:]) == {'0'}:
            # 只在 epoch=10,20,...,100,...,1000,... 时保存
            save_path = Path(args.save_path) / "checkpoints" / f"epoch{epoch}.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "args": vars(args),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, save_path)
            _logger.note(tag2ansi(f"Model saved to [underline green]{save_path}[reset]"))
            timer.add('save_periodly')

        # 保存最佳模型
        if test_records is not None:
            if (
                'best_records' not in locals() or 
                np.mean(test_records['ade']) < np.mean(best_records['ade'])
            ):
                patience = args.patience
                best_records = test_records
                save_path = f"{args.save_path}/best.pth"
                torch.save({
                    "epoch": epoch,
                    "args": vars(args),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, save_path)
                _logger.note(tag2ansi(f"Best model saved to [underline green]{save_path}[reset]"))
            else:
                patience -= 1
                _logger.info(tag2ansi(
                    f"Patience left: [brightred]{patience}/{args.patience}[reset] ("
                    f"[bold underline orange]best Accuracy={best_records['accuracy']:.2%}[reset] "
                    f"at epoch [#66CCFF]{best_records['epoch']}[reset]. "
                    f"[#66CCFF]ADE={np.mean(best_records['ade']):.4f}, "
                    f"[#66CCFF]FDE={np.mean(best_records['fde']):.4f}, "
                    f"[#66CCFF]X_ERROR (normal)={np.nanmean(best_records['norm_err']):.4f}, "
                    f"[#66CCFF]Y_ERROR (tangential)={np.nanmean(best_records['tan_err']):.4f}, "
                    f"[#66CCFF]Collision-Ped={np.mean(best_records['collision_ped']):.2%}, "
                    f"[#66CCFF]Collision-Veh={np.mean(best_records['collision_veh']):.2%}, "
                    f"[#66CCFF]Collision-Map={np.mean(best_records['collision_map']):.2%}, "
                    f"[#66CCFF]AvgLen={np.mean(best_records['trajlen']):.4f}, "
                    f"[#66CCFF]Loss={np.mean(best_records['loss']):.4f}, "
                    f"[#66CCFF]PedNum={np.mean(best_records['ped_num']):.1f}, "
                    f"[#66CCFF]VehNum={np.mean(best_records['veh_num']):.1f})"
                ))
            timer.add('save_best')

        # 打印用时
        allocated = torch.cuda.memory_allocated(args.device) / 1024 / 1024 / 1024
        reserved = torch.cuda.memory_reserved(args.device) / 1024 / 1024 / 1024
        peak = torch.cuda.max_memory_allocated(args.device) / 1024 / 1024 / 1024
        _logger.info(tag2ansi(
            f"[pink][Epoch {epoch}/{args.epochs}] finished. "
            f"Time Usage={timer}, "
            f"CUDA ({args.device}) usage: allocated={allocated:.1f}GiB, peak={peak:.1f}GiB, reserved={reserved:.1f}GiB"
            # f"adjust reserved memory from {reserved_raw/1024:.1f}GiB to {reserved_new/1024:.1f}GiB"
            "[reset]"
        ))

        # 释放额外的显存
        if args.minimize_gpu and train_records is not None:
            peak = torch.cuda.max_memory_allocated(args.device) / 1024 / 1024
            reserved_raw = torch.cuda.memory_reserved(args.device) / 1024 / 1024
            torch.cuda.empty_cache() # 释放 reserved 但是未被 allocated 的 block
            reserved_new = torch.cuda.memory_reserved(args.device) / 1024 / 1024
            if reserved_new < peak: # 释放了过多的显存，之后可能会 OOM
                allocated = torch.cuda.memory_allocated(args.device) / 1024 / 1024
                if (keep_MB := int(np.ceil(peak - allocated))) > 0: # 把需要的显存再占回来
                    tmp = AutoGPU.allocate_gpu(device=args.device, memory_MB=keep_MB, block_MB=None)
                    del tmp
                reserved_new = torch.cuda.memory_reserved(args.device) / 1024 / 1024
            _logger.info(tag2ansi(
                f"[brown]Adjust reserved memory from {reserved_raw/1024:.1f}GiB to {reserved_new/1024:.1f}GiB. [reset]"
            ))
        
        # 提前终止
        if 'patience' in locals() and patience <= 0:
            _logger.warning(tag2ansi(f"Early stopping at epoch [lightred]{epoch}/{args.epochs}[reset], "))
            break

    ## Log Best Result
    _logger.note(tag2ansi(
        f"[bold underline orange]best Accuracy={best_records['accuracy']:.2%}[reset] "
        f"at [#66CCFF]epoch {best_records['epoch']}[reset]. "
        f"[#66CCFF]ADE={np.mean(best_records['ade']):.4f}, "
        f"[#66CCFF]FDE={np.mean(best_records['fde']):.4f}, "
        f"[#66CCFF]X_ERROR (normal)={np.nanmean(best_records['norm_err']):.4f}, "
        f"[#66CCFF]Y_ERROR (tangential)={np.nanmean(best_records['tan_err']):.4f}, "
        f"[#66CCFF]Collision-Ped={np.mean(best_records['collision_ped']):.2%}, "
        f"[#66CCFF]Collision-Veh={np.mean(best_records['collision_veh']):.2%}, "
        f"[#66CCFF]Collision-Map={np.mean(best_records['collision_map']):.2%}, "
        f"[#66CCFF]AvgLen={np.mean(best_records['trajlen']):.4f}, "
        f"[#66CCFF]Loss={np.mean(best_records['loss']):.4f}, "
        f"[#66CCFF]PedNum={np.mean(best_records['ped_num']):.1f}, "
        f"[#66CCFF]VehNum={np.mean(best_records['veh_num']):.1f}"
    ))
    if len(set(best_records['dataset_class'])) > 1:
        for klass in sorted(list(set(best_records['dataset_class']))):
            idxs = [i for i, k in enumerate(best_records['dataset_class']) if k == klass]
            ade = np.array([best_records['ade'][i] for i in idxs])
            fde = np.array([best_records['fde'][i] for i in idxs])
            trajlen = np.array([best_records['trajlen'][i] for i in idxs])
            ped_num = np.array([best_records['ped_num'][i] for i in idxs])
            veh_num = np.array([best_records['veh_num'][i] for i in idxs])
            rollout_time = np.array([best_records['rollout_time'][i] for i in idxs])
            w = np.array([best_records['sample_nums'][i] for i in idxs], dtype=float)
            w /= w.sum()
            acc = 1 - np.sum(w * ade) / np.sum(w * trajlen)
            _logger.info(tag2ansi(
                f"[#66CCFF][Epoch {best_records['epoch']}/{args.epochs}] Overall on {klass} datasets: "
                f"[bold underline orange]Accuracy={acc:.2%}[reset], "
                f"[#66CCFF]ADE={np.sum(w * ade):.4f}, "
                f"[#66CCFF]FDE={np.sum(w * fde):.4f}, "
                f"[#66CCFF]X_ERROR (normal)={np.nansum(w * best_records['norm_err']) / np.sum(w * np.isfinite(best_records['norm_err'])):.4f}, "
                f"[#66CCFF]Y_ERROR (tangential)={np.nansum(w * best_records['tan_err']) / np.sum(w * np.isfinite(best_records['tan_err'])):.4f}, "
                f"[#66CCFF]Collision-Ped={np.sum(w * best_records['collision_ped']):.2%}, "
                f"[#66CCFF]Collision-Veh={np.sum(w * best_records['collision_veh']):.2%}, "
                f"[#66CCFF]Collision-Map={np.sum(w * best_records['collision_map']):.2%}, "
                f"[#66CCFF]AvgLen={np.sum(w * trajlen):.4f}, "
                f"[#66CCFF]PedNum={np.sum(w * ped_num):.4f}, "
                f"[#66CCFF]VehNum={np.sum(w * veh_num):.4f}, "
                f"[#66CCFF]RolloutTime={np.mean(rollout_time)*1000:.2f}ms "
                f"([bold underline orange]FPS={1/np.mean(rollout_time):.2f} Hz[reset])"
            ))
    _logger.note(f"Training finished. Re-run: {args.command}")


if __name__ == "__main__":
    parser = ArgumentParser()
    # 基础配置
    parser.add_argument("--name", type=str, default="train_spdiff", help="实验任务名称，用于生成实验ID")
    parser.add_argument("--exp_name", type=str, default=None, help="手动指定实验名称（若指定则覆盖自动生成的名称）")
    parser.add_argument("--device", type=str, default="auto", help="计算设备，可选 'cpu', 'cuda:0' 或 'auto'（自动选择显存充足的 GPU）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，固定以复现实验结果")
    parser.add_argument("--save_dir", type=str, default="./logs/train_spdiff", help="日志和模型权重的保存根目录")
    parser.add_argument("--debug", action="store_true", help="是否开启调试模式（输出更多日志，不保存部分文件）")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader 的工作线程数（0 表示主线程）")
    parser.add_argument("--minimize_gpu", action="store_true", default=False, help="是否在每个 epoch 结束后尽可能释放显存以供其他进程使用")
    
    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=128, help="训练批次大小")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率 (Learning Rate)")
    parser.add_argument("--epochs", type=int, default=10000, help="最大训练轮数")
    parser.add_argument('--patience', type=int, default=20, help="Early Stopping 的耐心值（多少个 epoch 验证集指标不提升则停止）")
    parser.add_argument('--loss_type', type=str, default='noise', choices=['position', 'accelerate', 'noise'], help="损失函数计算的目标类型")
    parser.add_argument('--reload_checkpoint', type=str, default=None, help="断点续训的 checkpoint 路径（.pth 文件）")
    parser.add_argument('--required_memory_MB', type=int, default=5000, help="自动选择 GPU 时要求的最小剩余显存 (MB)")

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
    parser.add_argument('--p_drop_map', type=float, default=0.1, help="以一定概率丢弃地图信息（实现无地图引导的生成）")
    parser.add_argument('--p_drop_destination', type=float, default=0.1, help="以一定概率丢弃目的地条件（实现无目标引导的生成）")
    parser.add_argument('--p_drop_speed', type=float, default=0.1, help="以一定概率丢弃初始速度条件")
    parser.add_argument('--dropout', type=float, default=0.5, help="模型中的 Dropout 比率")

    # 数据集配置
    parser.add_argument('--train_datasets', type=str, default=[], nargs='*', choices=['eth', 'hotel', 'zara01', 'zara02', 'univ', 'GC_train', 'SDD_train', 'WayMo_train'], help="使用的训练数据集列表 (KDD)")
    parser.add_argument('--test_datasets', type=str, default=[], nargs='*', choices=['eth', 'hotel', 'zara01', 'zara02', 'univ', 'GC_test', 'SDD_test', 'WayMo_test'], help="使用的训练数据集列表 (KDD)")
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
    parser.add_argument('--test_before_train', action='store_true', default=True, help="是否在训练开始前先运行一次测试")
    parser.add_argument('--test_per_epoch', type=int, default=10, help="每隔多少个 epoch 运行一次测试")
    parser.add_argument('--collision_threshold', type=float, default=0.6, help="碰撞检测的距离阈值（单位：米）")

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

    # MID 参数
    parser.add_argument('--config', type=str, default='./baselines/mid/configs/baseline.yaml', help="MID 模型配置文件路径")

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

    ## Start Training
    setproctitle(f"{args.exp_name}@ZihanYu")
    main(args)
