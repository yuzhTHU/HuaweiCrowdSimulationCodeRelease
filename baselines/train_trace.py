import os
import re
import sys
import json
import time
import torch
import shlex
import random
import logging
import numpy as np
import torch.nn as nn
import torch.utils.data as D
import torch.distributions.multivariate_normal as torchdist
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from socket import gethostname
from argparse import ArgumentParser
from setproctitle import setproctitle
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

from baselines.social_stgcnn.model import social_stgcnn
from baselines.social_stgcnn.metrics import bivariate_loss, seq_to_nodes, nodes_rel_to_nodes_abs, ade, fde
from baselines.social_stgcnn.utils import seq_to_graph

_logger = logging.getLogger("src.train_trace")


import torch
import torch.nn.functional as F

def get_local_map_data(
    global_map_tensor, 
    pos, 
    yaw, 
    map_xmin, map_xmax, map_ymin, map_ymax,
    raster_size=224, 
    pixel_size=1.0/2.0, # 默认 2 pixels per meter (根据需求调整，TRACE 可能用 1/2 或 1/4)
    ego_center=(0.5, 0.5), # Agent 在局部图中心
    device='cuda'
):
    """
    Args:
        global_map_tensor: (W, H) float tensor. 0=Free, 1=Obstacle, NaN=Void.
        pos: (B*N, 2) Global positions.
        yaw: (B*N, 1) Global headings (radians).
        map_xmin, ...: Global map boundaries.
        raster_size: Output image size (H, W).
        pixel_size: Meters per pixel.
        
    Returns:
        image: (B*N, 1, raster_size, raster_size)
               Ch0: Obstacle (1.0 where map==1)
               Ch1: Drivable (1.0 where map==0)
               Ch2: Valid (1.0 where map is not NaN)
        raster_from_agent: (B*N, 3, 3) Transformation matrix.
    """
    num_agents = pos.shape[0]
    
    # 1. 准备 Global Map 用于 grid_sample
    # map 是 (W, H)，对应 (X, Y)。
    # PyTorch image 是 (H, W) 即 (Rows, Cols)。
    # grid_sample 的 grid 是 (x, y)，对应 (Width, Height)。
    # 所以我们需要把 map 转置为 (H, W) 以匹配 grid_sample 的 (y, x) 逻辑
    # Input: (1, 1, H, W)
    map_in = global_map_tensor.T.unsqueeze(0).unsqueeze(0).to(device) # (1, 1, H_global, W_global)

    # 2. 构建 raster_from_agent (Agent Frame -> Image Pixel Frame)
    # 定义: Agent X+ (Forward) -> Image U+ (Right)
    #       Agent Y+ (Left)    -> Image V- (Up)  (图像坐标系 Y 向下)
    s = 1.0 / pixel_size
    u0 = raster_size * ego_center[0]
    v0 = raster_size * ego_center[1]
    
    # M = [[s, 0, u0], [0, -s, v0], [0, 0, 1]]
    raster_from_agent = torch.tensor([
        [s,   0.0, u0],
        [0.0, -s,  v0],
        [0.0, 0.0, 1.0]
    ], device=device).unsqueeze(0).expand(num_agents, -1, -1)

    # 3. 生成采样网格 (Inverse Warp: Pixel -> Global)
    # (A) Generate Pixel Grid (u, v)
    u_grid, v_grid = torch.meshgrid(
        torch.arange(raster_size, device=device, dtype=torch.float32),
        torch.arange(raster_size, device=device, dtype=torch.float32),
        indexing='xy'
    )
    # (N, H, W, 3) Homogeneous
    pixel_grid = torch.stack([u_grid, v_grid, torch.ones_like(u_grid)], dim=-1)
    pixel_grid = pixel_grid.unsqueeze(0).expand(num_agents, -1, -1, -1)
    
    # (B) Pixel -> Agent (M_inv)
    M_inv = torch.inverse(raster_from_agent) # (N, 3, 3)
    
    # Reshape for matmul: (N, H*W, 3)
    pixel_flat = pixel_grid.reshape(num_agents, -1, 3)
    agent_flat = torch.bmm(pixel_flat, M_inv.transpose(1, 2)) # (x_a, y_a, 1)

    # (C) Agent -> Global
    # Global = R(yaw) * Agent + Pos
    c = torch.cos(yaw).squeeze(-1).unsqueeze(1) # (N, 1)
    s = torch.sin(yaw).squeeze(-1).unsqueeze(1)
    p_x = pos[..., 0].unsqueeze(1)
    p_y = pos[..., 1].unsqueeze(1)
    
    x_a = agent_flat[..., 0]
    y_a = agent_flat[..., 1]
    
    x_global = x_a * c - y_a * s + p_x
    y_global = x_a * s + y_a * c + p_y
    
    # (D) Global -> Normalized Grid [-1, 1] for grid_sample
    # map index i = (x - xmin) / (xmax - xmin) * W
    # normalized = (i / W) * 2 - 1 = (x - xmin)/(range) * 2 - 1
    
    norm_x = (x_global - map_xmin) / (map_xmax - map_xmin) * 2.0 - 1.0
    norm_y = (y_global - map_ymin) / (map_ymax - map_ymin) * 2.0 - 1.0
    
    # Stack into (N, H, W, 2)
    # 注意: grid_sample 期望最后一个维度是 (x, y)，对应 map_in 的 (Width, Height)
    # 我们的 map_in 是 (H_global, W_global)。
    # grid.x 采样 Width (dim 3), grid.y 采样 Height (dim 2)
    # 这里的 norm_x 对应 W_global，norm_y 对应 H_global，顺序正确
    grid = torch.stack([norm_x, norm_y], dim=-1).reshape(num_agents, raster_size, raster_size, 2)

    # 4. 采样
    # map_in: (1, 1, H, W) -> expand N
    # mode='nearest' 因为是分类地图
    # padding_mode='zeros' 越界视为 0 (空地/未知)
    # sampled: (N, 1, H, W)
    raw_crop = F.grid_sample(map_in.expand(num_agents, -1, -1, -1), grid, mode='nearest', padding_mode='zeros', align_corners=False)

    # 5. 构建多通道 Image
    # 原始值: 0=空, 1=障, NaN=缺失
    # 通道设计:
    # Ch0: Obstacle (1 if val==1)
    # Ch1: Drivable (1 if val==0)
    # Ch2: Valid Area (1 if not NaN)
    
    is_nan = torch.isnan(raw_crop)
    is_obstacle = (raw_crop == 1.0) & (~is_nan)
    is_drivable = (raw_crop == 0.0) & (~is_nan)
    is_valid = ~is_nan
    
    image = (
        1 * is_obstacle.float() +
        0 * is_drivable.float() +
        0 * is_valid.float() 
    ) # (N, 1, H, W)

    return image, raster_from_agent


def parse_data(args, batch, map_data):
    # 1. 读取基础数据
    pos = batch['pos'].to(args.device)  # (B, N, 2)
    vel = batch['vel'].to(args.device)  # (B, N, 2)
    hst = batch['hst'].to(args.device)  # (B, N, 8, 2)
    future_pos = batch['future_pos'].to(args.device) # (B, N, 12, 2)
    ped_length = batch['ped_length'].to(args.device) # (B,) 每个场景的有效行人数

    B, N, _ = pos.shape
    
    # === [关键修正 1] 构建精确的 Valid Mask ===
    # 形状: (B, N)
    # 对应位置为 True 表示该行人是真实存在的，False 表示是 Padding
    range_tensor = torch.arange(N, device=args.device).unsqueeze(0) # (1, N)
    valid_mask = range_tensor < ped_length.unsqueeze(1) # (B, 1) -> (B, N)

    # 2. 拼接与切片 History
    # hst(8) + pos(1) -> 9 帧 -> 切片取后 8 帧
    # 注意: Padding 的数据在这里依然保留，但在最后一步会被过滤掉
    history_full = torch.concat([hst, pos.unsqueeze(-2)], dim=-2)
    required_frames = 8 
    history_global = history_full[..., -required_frames:, :] # (B, N, 8, 2)

    # 3. 建立局部坐标系参数
    origin = pos.view(B, N, 1, 1, 2)
    theta = torch.atan2(vel[..., 1], vel[..., 0]).view(B, N, 1, 1)
    c, s = torch.cos(theta), torch.sin(theta)

    # 辅助函数
    def global_to_local(points_global):
        delta = points_global - origin
        x = delta[..., 0] * c + delta[..., 1] * s
        y = delta[..., 0] * (-s) + delta[..., 1] * c
        return torch.stack([x, y], dim=-1)

    def normalize_yaw(yaw_global):
        if yaw_global.dim() == 3: 
            ref = theta.view(B, N, 1)
        elif yaw_global.dim() == 4:
            ref = theta # (B, N, 1, 1)
        else:
            ref = theta
        yaw_rel = yaw_global - ref
        return (yaw_rel + torch.pi) % (2 * torch.pi) - torch.pi

    # 4. Target 处理
    target_pos_local = global_to_local(future_pos.unsqueeze(2)).squeeze(2)
    target_vel_local = target_pos_local.diff(dim=2, prepend=torch.zeros(B, N, 1, 2, device=args.device)) * args.fps
    target_yaws_local = torch.atan2(target_vel_local[..., 1], target_vel_local[..., 0]).unsqueeze(-1)
    # Target Avail: 只要是 Valid Agent，我们假设其 Future 都是需要预测的 (除非 Future 也是 NaN)
    # 这里简单处理，稍后统一用 valid_mask 过滤

    # 5. Ego History 处理
    history_pos_local = global_to_local(history_global.unsqueeze(2)).squeeze(2)
    
    history_vel_global = history_global.diff(dim=-2, prepend=torch.zeros(B, N, 1, 2, device=args.device)) * args.fps
    # 修正首帧速度 (复制第二帧)
    # history_vel_global[:, :, 0] = history_vel_global[:, :, 1] # 可选优化
    
    history_speeds = history_vel_global.norm(dim=-1)
    history_yaw_global = torch.atan2(history_vel_global[..., 1], history_vel_global[..., 0])
    history_yaws_local = normalize_yaw(history_yaw_global).unsqueeze(-1)
    
    curr_speed = vel.norm(dim=-1)

    # 6. Neighbor History 处理
    # 构造邻居索引 (N, N-1)
    indices = torch.arange(N, device=args.device)
    mask_eye = ~torch.eye(N, dtype=torch.bool, device=args.device)
    neighbor_indices = indices.unsqueeze(0).expand(N, N)[mask_eye].reshape(N, N-1)
    
    raw_neighbor_pos = history_global[:, neighbor_indices] # (B, N, N-1, 8, 2)
    neighbor_pos_local = global_to_local(raw_neighbor_pos)

    raw_neighbor_vel = raw_neighbor_pos.diff(dim=-2) * args.fps
    raw_neighbor_vel = torch.cat([raw_neighbor_vel[..., :1, :], raw_neighbor_vel], dim=-2)
    
    neighbor_speeds = raw_neighbor_vel.norm(dim=-1)
    raw_neighbor_yaw = torch.atan2(raw_neighbor_vel[..., 1], raw_neighbor_vel[..., 0])
    neighbor_yaws_local = normalize_yaw(raw_neighbor_yaw).unsqueeze(-1)
    neighbor_extents = torch.zeros(B, N, N-1, 3, device=args.device)

    # === [关键修正 2] Neighbor Mask 处理 ===
    # 我们需要判断取到的 "邻居" 是否是 Padding 里的无效行人
    # valid_mask: (B, N)
    # 我们需要将其扩展并 gather 到 (B, N, N-1)
    
    # 1. 扩展 Valid Mask: (B, N) -> (B, N, N) (每个 Ego 看到的 N 个潜在邻居)
    valid_mask_expanded = valid_mask.unsqueeze(1).expand(B, N, N)
    
    # 2. 使用 neighbor_indices 提取 N-1 个邻居的有效性
    # neighbor_indices: (N, N-1) -> (B, N, N-1)
    idx_expanded = neighbor_indices.unsqueeze(0).expand(B, N, N-1)
    
    # 3. Gather 得到邻居的有效性 Mask
    # shape: (B, N, N-1)
    # True 表示该邻居是真实存在的，False 表示该邻居是 Padding
    neighbor_validity_mask = torch.gather(valid_mask_expanded, 2, idx_expanded)
    
    # 4. 结合原本的数据 NaN 检查 (双重保险)
    neigh_data_valid = ~raw_neighbor_pos.isnan().any(dim=-1) # (B, N, N-1, 8)
    
    # 最终 Neighbor Avail: (数据非NaN) AND (邻居是真实行人)
    # 注意广播: (B, N, N-1, 8) & (B, N, N-1, 1)
    neighbor_avail = (neigh_data_valid & neighbor_validity_mask.unsqueeze(-1)).float()


    # 7. Ego Availability
    ego_data_valid = ~history_global.isnan().any(dim=-1) # (B, N, 8)
    # Ego Avail 不需要在这里与 valid_mask 结合，因为最后我们会直接丢弃无效 Ego
    history_avail = ego_data_valid.float()

    # 8. Clean Data (置零)
    def clean_data(data, mask):
        while mask.dim() < data.dim():
            mask = mask.unsqueeze(-1)
        return data * mask
    
    history_pos_local = clean_data(history_pos_local, history_avail)
    history_yaws_local = clean_data(history_yaws_local, history_avail)
    history_speeds = clean_data(history_speeds, history_avail)

    neighbor_pos_local = clean_data(neighbor_pos_local, neighbor_avail)
    neighbor_yaws_local = clean_data(neighbor_yaws_local, neighbor_avail)
    neighbor_speeds = clean_data(neighbor_speeds, neighbor_avail)
    
    target_pos_local[torch.isnan(target_pos_local)] = 0.0
    target_yaws_local[torch.isnan(target_yaws_local)] = 0.0
    extent = torch.zeros(B, N, 3, device=args.device)

    # === [关键修正 3] 只保留 Valid Agents ===
    # 使用 valid_mask (B, N) 对所有数据进行索引
    # 结果将自动 Flatten 为 (Total_Valid_Agents, ...)
    
    # 提取有效行人的数据
    valid_ego_pos = history_pos_local[valid_mask]         # (M, 8, 2)
    valid_ego_yaws = history_yaws_local[valid_mask]       # (M, 8, 1)
    valid_ego_speeds = history_speeds[valid_mask]         # (M, 8)
    valid_ego_avail = history_avail[valid_mask]           # (M, 8)
    valid_extent = extent[valid_mask]                     # (M, 3)
    valid_curr_speed = curr_speed[valid_mask]             # (M)
    
    valid_target_pos = target_pos_local[valid_mask]       # (M, 12, 2)
    valid_target_yaws = target_yaws_local[valid_mask]     # (M, 12, 1)
    
    # 邻居数据也要对应提取 (只保留有效 Ego 的邻居数据)
    # 这一步过滤的是 "Ego"，保留的是 "该有效 Ego 的 N-1 个邻居"
    # 至于这 N-1 个邻居里哪些是 Padding，已经由 neighbor_avail 处理为 0 了
    valid_neigh_pos = neighbor_pos_local[valid_mask]      # (M, N-1, 8, 2)
    valid_neigh_yaws = neighbor_yaws_local[valid_mask]    # (M, N-1, 8, 1)
    valid_neigh_speeds = neighbor_speeds[valid_mask]      # (M, N-1, 8)
    valid_neigh_avail = neighbor_avail[valid_mask]        # (M, N-1, 8)
    valid_neigh_extents = neighbor_extents[valid_mask]    # (M, N-1, 3)

    # 处理 Single Agent 情况 (防止 reshape 报错)
    if valid_neigh_pos.shape[1] == 0:
         curr_bs = valid_neigh_pos.shape[0]
         valid_neigh_pos = torch.zeros(curr_bs, 1, 8, 2, device=args.device)
         valid_neigh_yaws = torch.zeros(curr_bs, 1, 8, 1, device=args.device)
         valid_neigh_speeds = torch.zeros(curr_bs, 1, 8, device=args.device)
         valid_neigh_extents = torch.zeros(curr_bs, 1, 3, device=args.device)
         valid_neigh_avail = torch.zeros(curr_bs, 1, 8, device=args.device)

    # 9. Pack
    model_input = {
        'target_positions': valid_target_pos.nan_to_num(0.0),
        'target_yaws': valid_target_yaws.nan_to_num(0.0),
        'curr_speed': valid_curr_speed.nan_to_num(0.0),
        
        'history_positions': valid_ego_pos.nan_to_num(0.0),
        'history_yaws': valid_ego_yaws.nan_to_num(0.0),
        'history_speeds': valid_ego_speeds.nan_to_num(0.0),
        'extent': valid_extent.nan_to_num(0.0),
        'history_availabilities': valid_ego_avail.nan_to_num(0.0),
        
        'all_other_agents_history_positions': valid_neigh_pos.nan_to_num(0.0),
        'all_other_agents_history_yaws': valid_neigh_yaws.nan_to_num(0.0),
        'all_other_agents_history_speeds': valid_neigh_speeds.nan_to_num(0.0),
        'all_other_agents_extents': valid_neigh_extents.nan_to_num(0.0),
        'all_other_agents_history_availabilities': valid_neigh_avail.nan_to_num(0.0),
        
        # Image 和 Raster 也要只保留 Valid Ego 的
        'image': torch.zeros(valid_ego_pos.shape[0], 1, 224, 224, device=args.device),
        'raster_from_agent': torch.eye(3, device=args.device).unsqueeze(0).expand(valid_ego_pos.shape[0], -1, -1),
    }

    sample_valid_mask = torch.arange(N, device=args.device).unsqueeze(0) < ped_length.unsqueeze(1)
    yaw = torch.atan2(vel[..., 1], vel[..., 0]).unsqueeze(-1)
    valid_pos = pos[sample_valid_mask]
    valid_yaw = yaw[sample_valid_mask]
    img, r_f_a = get_local_map_data(
        global_map_tensor=torch.from_numpy(map_data.map).float().to(args.device),
        pos=valid_pos,
        yaw=valid_yaw,
        map_xmin=map_data.xmin,
        map_xmax=map_data.xmax, 
        map_ymin=map_data.ymin, 
        map_ymax=map_data.ymax,
        device=args.device
    )
    model_input['image'] = img
    model_input['raster_from_agent'] = r_f_a

    return model_input


import torch

def convert_predictions_to_global(pred_trajs, batch, args):
    """
    将模型输出的局部坐标轨迹转换为与 future_pos 对齐的全局坐标轨迹。
    
    Args:
        pred_trajs: (Valid_Ped, Sample, T, 2) 模型预测的局部轨迹
                    其中 Valid_Ped = sum(batch['ped_length'])
        batch: DataLoader yield 的 batch 字典，必须包含 'pos', 'vel', 'ped_length', 'future_pos'
        args: 包含 device 设置
        
    Returns:
        pos_pred_global: (B, N, Sample, T, 2) 全局坐标下的预测轨迹 (Padding部分为NaN或0)
        pos_true_global: (B, N, 1, T, 2) 扩展维度后的 Ground Truth，可直接与前者作差
    """
    # 1. 获取 Batch 基础信息
    pos = batch['pos'].to(args.device)  # (B, N, 2) 当前全局位置
    vel = batch['vel'].to(args.device)  # (B, N, 2) 当前全局速度 (用于确定航向)
    ped_length = batch['ped_length'].to(args.device) # (B,) 每个样本的有效行人数
    
    B, N, _ = pos.shape
    num_samples = pred_trajs.shape[1]
    horizon = pred_trajs.shape[2]

    # 2. 构建有效行人掩码 (Valid Mask)
    # shape: (B, N) -> True 表示该位置是有效行人
    valid_mask = torch.arange(N, device=args.device).unsqueeze(0) < ped_length.unsqueeze(1)
    
    # 3. 提取有效行人的参考系参数 (Origin & Heading)
    # origin: (Valid_Ped, 2)
    origin_valid = pos[valid_mask] 
    
    # heading: (Valid_Ped,)
    vel_valid = vel[valid_mask]
    theta = torch.atan2(vel_valid[..., 1], vel_valid[..., 0])
    
    # 准备旋转矩阵参数，并扩展维度以支持广播: (Valid_Ped, 1, 1) -> 对应 (Sample, Time)
    c = torch.cos(theta).view(-1, 1, 1) 
    s = torch.sin(theta).view(-1, 1, 1)
    origin_valid = origin_valid.view(-1, 1, 1, 2)

    # 4. 执行逆变换 (Local -> Global)
    # 公式: 
    # x_global = x_local * cos - y_local * sin + origin_x
    # y_global = x_local * sin + y_local * cos + origin_y
    
    x_local = pred_trajs[..., 0] # (Valid_Ped, S, T)
    y_local = pred_trajs[..., 1]
    
    x_global = x_local * c - y_local * s + origin_valid[..., 0]
    y_global = x_local * s + y_local * c + origin_valid[..., 1]
    
    # (Valid_Ped, S, T, 2)
    pred_global_valid = torch.stack([x_global, y_global], dim=-1)
    
    # 5. Scatter 回 (B, N) 的网格结构
    # 初始化一个全 NaN 的容器 (避免 Padding 位置的 0 干扰 min/mean 计算)
    pos_pred_global = torch.full((B, N, num_samples, horizon, 2), float('nan'), device=args.device)
    
    # 将有效轨迹填入对应的位置
    pos_pred_global[valid_mask] = pred_global_valid
    
    return pos_pred_global

def train_once(
    args,
    train_loaders: List[D.DataLoader],
    model: Model,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    diffusion: DDPM,
    epoch: int,
) -> Dict:

    model.train()

    train_timer = NamedTimer(unit='it', mode='pace')
    records_list = []
    for loader in train_loaders:
        records = dict(loss=[])
        map_data = loader.dataset.map_data
        for cnt, batch in enumerate(tqdm(loader, disable=False, leave=False, dynamic_ncols=True)):
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

            model_input = parse_data(args, batch, map_data)            

            # Move data to device (Assuming batch_utils logic or manual move)
            # In your snippet, you unpacked specific vars. TRACE needs the dict.
            model_input = {k: v.to(args.device) if hasattr(v, 'to') else v for k, v in model_input.items()}
            
            # [TRACE] Conditioning Dropout Logic (from algos.py training_step)
            # This logic mimics the random masking of map/neighbors for classifier-free guidance
            if hasattr(model, 'use_cond') and model.use_cond:
                # 1. Map Dropout
                if hasattr(model, 'use_rasterized_map') and model.use_rasterized_map:
                    if model.cond_drop_map_p > 0:
                        # Assuming 'image' is in batch (B, C, H, W)
                        num_sem_layers = model_input['maps'].size(1) if 'maps' in model_input else 0
                        drop_mask = torch.rand((model_input["image"].size(0)), device=args.device) < model.cond_drop_map_p
                        # Fill last N layers with fill value
                        if 'image' in model_input:
                             model_input["image"][drop_mask, -num_sem_layers:] = model.cond_fill_val

                # 2. Neighbor Dropout
                if model.cond_drop_neighbor_p > 0:
                    if "all_other_agents_history_availabilities" in model_input:
                        B = model_input["all_other_agents_history_availabilities"].size(0)
                        drop_mask = torch.rand((B), device=args.device) < model.cond_drop_neighbor_p
                        model_input["all_other_agents_history_availabilities"][drop_mask] = 0
            
            # --- [TRACE] 3. Loss Calculation ---
            # The model computes losses internally based on the batch
            # Note: We access the internal policy if wrapped in LightningModule
            policy_net = model.nets["policy"] if hasattr(model, 'nets') else model
            loss_dict = policy_net.compute_losses(model_input)
            # Aggregate losses using weights from config
            loss = 0.0
            loss_weights = args.loss_weights if hasattr(args, 'loss_weights') else {}

            for lk, l in loss_dict.items():
                # Apply weight if it exists, otherwise default to 1.0
                weight = loss_weights.get(lk, 1.0)
                loss += l * weight
                
                # Optional: Record individual losses
                # records[lk] = records.get(lk, []) + [l.item()] * args.batch_size

            loss.backward()

            optimizer.step()
            # [TRACE] 4. EMA Step (Important!)
            # If using EMA, you must manually update it here
            if hasattr(model, 'use_ema') and model.use_ema:
                # Assuming 'step' is tracked, otherwise use cnt or global_step
                current_step = (epoch - 1) * len(loader) + cnt 
                if current_step % model.ema_update_every == 0:
                    model.step_ema(current_step)

            records['loss'].extend([loss.item()] * args.batch_size)
            train_timer.add('backward')

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
            if not isinstance(v, (list, tuple, np.ndarray)) or len(v) == 0:
                mean_v = np.nan
            elif isinstance(v[0], (int, float)):
                mean_v = np.mean(v)
            else: 
                mean_v = np.mean(v, axis=0).tolist()
            all_records[k].append(mean_v)
    _logger.info(tag2ansi(
        f"[#66CCFF][Epoch {epoch}/{args.epochs}] "
        f"[#66CCFF]Loss={np.mean(all_records['loss']):.4f} "
        f"[#66CCFF]Time={train_timer}"
    ))
    return records


def test_once(
    args,
    test_loaders: List[D.DataLoader],
    model: Model,
    criterion: nn.Module,
    diffusion: DDPM,
    epoch: int,
) -> Dict:
    
    model.eval()
    test_timer = NamedTimer(unit='it', mode='pace')
    records_list = []
    step = 0
    for loader in test_loaders:
        map_data = loader.dataset.map_data
        records = dict(loss=[], ade=[], fde=[], trajlen=[], ped_num=[], veh_num=[], rollout_time=[], collision_ped=[], collision_veh=[], collision_map=[])
        for batch in tqdm(loader, leave=False, disable=False, dynamic_ncols=True): 
            step+=1

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
            test_timer.add('prepare_data')

            start_time = time.time()
            model_input = parse_data(args, batch, map_data)
            output = model(
                model_input, 
                num_samp=args.sample_num, 
                class_free_guide_w=0.0,   # 如果需要 CFG 可以设为 > 0
                return_diffusion=False,
            )
            # [Fix 3] 直接从 output 中获取 positions，无需再通过 ['predictions']
            # DiffuserTrafficModel.forward 返回的是 cur_policy(...)["predictions"]
            pred_trajs = output['positions'] # (B*N, S, T, 2) 或 (B*N, T, 2) 取决于模型输出
            pos_pred = convert_predictions_to_global(pred_trajs, batch, args)
            pos_pred = pos_pred.permute(2, 0, 1, 3, 4)
            rollout_time = time.time() - start_time
            test_timer.add('sample')

            # 获取有效的行人掩模
            mask = torch.arange(pos.shape[1], device=args.device).expand(pos.shape[0], pos.shape[1]) < ped_length.unsqueeze(-1)  # (B, #pedestrian)
            # 计算 pos_true 和 vel_true
            acc_true = future_acc # (B, #pedestrian, roll_step*pred_step, 2)
            vel_true = vel.unsqueeze(-2) + acc_true.cumsum(dim=-2) / args.fps # (B, #pedestrian, roll_step*pred_step, 2)
            pos_true = pos.unsqueeze(-2) + vel_true.cumsum(dim=-2) / args.fps # (B, #pedestrian, roll_step*pred_step, 2)
            # 计算 distance error
            dis_err = (pos_pred - pos_true).norm(dim=-1) # (S, B, #pedestrian, roll_step*pred_step)
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
            test_timer.add('evaluate')
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
        f"[#66CCFF]Collision-Ped={np.sum(w * all_records['collision_ped']):.2%}, "
        f"[#66CCFF]Collision-Veh={np.sum(w * all_records['collision_veh']):.2%}, "
        f"[#66CCFF]Collision-Map={np.sum(w * all_records['collision_map']):.2%}, "
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
                f"[#66CCFF]Collision-Ped={np.sum(w * collision_ped):.2%}, "
                f"[#66CCFF]Collision-Veh={np.sum(w * collision_veh):.2%}, "
                f"[#66CCFF]Collision-Map={np.sum(w * collision_map):.2%}, "
                f"[#66CCFF]APD={np.sum(w * apd):.4f}, "
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

        # 将每个场景的特定比例样本划分到测试集
        dataset_list = train_dataset
        train_dataset = []
        eval_dataset = []
        for d in dataset_list:
            d1 = d
            d2 = deepcopy(d)
            train_num = int(len(d) * (1-args.eval_ratio))
            d1.name = d1.name + f"_{100-args.eval_ratio*100:.0f}train"
            d2.name = d2.name + f"_{args.eval_ratio*100:.0f}eval"
            d1.samples = d1.samples[:train_num]
            d2.samples = d2.samples[train_num:]
            train_dataset.append(d1)
            eval_dataset.append(d2)
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
        eval_dataset = test_dataset
    # 创建数据加载器
    train_loaders = []
    eval_loaders = []
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
    for dataset in eval_dataset:
        if len(dataset) == 0:
            _logger.warning(f"Dataset {dataset.name} has no evaluation samples!")
            continue
        eval_loaders.append(D.DataLoader(
            dataset,
            shuffle=False,
            batch_size=args.batch_size, # //args.sample_num,
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
            batch_size=args.batch_size, # //args.sample_num,
            num_workers=args.num_workers,
            collate_fn=dataset.collate_fn,
        ))
    _logger.note(
        "Datasets:\n"
        f"Train on {[d.name for d in train_dataset]} datasets ({sum([len(d) for d in train_dataset]):,} samples in total)\n"
        f"Eval on {[d.name for d in eval_dataset]} datasets ({sum([len(d) for d in eval_dataset]):,} samples in total)\n"
        f"Test on {[d.name for d in test_dataset]} datasets ({sum([len(d) for d in test_dataset]):,} samples in total)"
    )

    ## Load Model
    # --- [TRACE] 1. Model & Optimizer Definition ---
    from baselines.trace.tbsim.algos.algos import DiffuserTrafficModel
    from baselines.trace.tbsim.configs.registry import get_registered_experiment_config
    
    # Load Config
    # Assuming args.config_name corresponds to a registered TRACE config
    cfg = get_registered_experiment_config('mixed_ped_diff')
    cfg.algo.history_num_frames = 7
    cfg.algo.history_num_frames_ego = 7
    cfg.algo.history_num_frames_agents = 7
    cfg.algo.future_num_frames = 12
    cfg.algo.horizon = 12
    
    # 1. Define Model
    # We instantiate the LightningModule wrapper as it contains the init logic
    # modality_shapes must be known (usually from datamodule.modality_shapes)
    # You might need to hardcode this or fetch from your dataset
    modality_shapes = {
        "image": (1, 224, 224), # Example shape
        "history": (8, 2)
    }
    model = DiffuserTrafficModel(algo_config=cfg.algo, modality_shapes=modality_shapes)
    model.to(args.device)

    # 2. Define Optimizer
    # TRACE uses Adam with params from the config
    optim_params = cfg.algo.optim_params["policy"]
    optim_params["learning_rate"]["initial"] = args.lr
    optimizer = torch.optim.Adam(
        params=model.nets["policy"].parameters(),
        lr=args.lr
    )
    
    criterion = None # Loss is computed inside the model
    diffusion = None # Diffusion logic is inside the model
    _logger.note(
        "Model Parameters:\n"
        f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n"
        f"Total: {sum(p.numel() for p in model.parameters()):,}"
    )

    ## Reload Checkpoint
    if args.force_new_experiment:
        checkpoint_path = None
    elif args.reload_checkpoint is not None:
        # 如果指定了 checkpoint 路径，则从该路径加载
        checkpoint_path = Path(args.reload_checkpoint)
    elif args.test and (Path(args.save_path) / "best.pth").exists():
        # 测试模式优先加载 Best checkpoint
        checkpoint_path = Path(args.save_path) / "best.pth"
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
                if key in [
                    'device', 'save_dir', 'save_path', 'command', 'name', 'exp_name',
                    'seed', 'reload_checkpoint', 'test_before_train', 'test_per_epoch',
                ]:
                    continue
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
        # 只测试
        if args.test:
            break
        
        # 训练一个 epoch
        if epoch > 0:
            torch.set_grad_enabled(True)
            model.train()
            train_records = train_once(args, train_loaders, model, optimizer, criterion, diffusion, epoch)
            timer.add('train')
        else:
            train_records = None

        # 测试一个 epoch
        if (epoch > 0 and not epoch % args.test_per_epoch) or (epoch == 0 and args.test_before_train):
            torch.set_grad_enabled(False)
            model.eval()
            with npu_attention_fallback_context(model, enable=USE_NPU):
                test_records = test_once(args, eval_loaders, model, criterion, diffusion, epoch)
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
                    f"[#66CCFF]APD={np.mean(best_records['apd']):.4f}, "
                    f"[#66CCFF]AvgLen={np.mean(best_records['trajlen']):.4f}, "
                    f"[#66CCFF]Loss={np.mean(best_records['loss']):.4f}, "
                    f"[#66CCFF]PedNum={np.mean(best_records['ped_num']):.1f}, "
                    f"[#66CCFF]VehNum={np.mean(best_records['veh_num']):.1f}), "
                    f"[#66CCFF]RolloutTime={np.mean(best_records['rollout_time'])*1000:.2f}ms "
                    f"([bold underline orange]FPS={1/np.mean(best_records['rollout_time']):.2f} Hz[reset])"
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
    if not args.test:
        _logger.note(tag2ansi(
            f"[bold underline orange]best Evaluation Accuracy={best_records['accuracy']:.2%}[reset] "
            f"at [#66CCFF]epoch {best_records['epoch']}[reset]. "
            f"[#66CCFF]ADE={np.mean(best_records['ade']):.4f}, "
            f"[#66CCFF]FDE={np.mean(best_records['fde']):.4f}, "
            f"[#66CCFF]X_ERROR (normal)={np.nanmean(best_records['norm_err']):.4f}, "
            f"[#66CCFF]Y_ERROR (tangential)={np.nanmean(best_records['tan_err']):.4f}, "
            f"[#66CCFF]Collision-Ped={np.mean(best_records['collision_ped']):.2%}, "
            f"[#66CCFF]Collision-Veh={np.mean(best_records['collision_veh']):.2%}, "
            f"[#66CCFF]Collision-Map={np.mean(best_records['collision_map']):.2%}, "
            f"[#66CCFF]APD={np.mean(best_records['apd']):.4f}, "
            f"[#66CCFF]AvgLen={np.mean(best_records['trajlen']):.4f}, "
            f"[#66CCFF]Loss={np.mean(best_records['loss']):.4f}, "
            f"[#66CCFF]PedNum={np.mean(best_records['ped_num']):.1f}, "
            f"[#66CCFF]VehNum={np.mean(best_records['veh_num']):.1f}, "
            f"[#66CCFF]RolloutTime={np.mean(best_records['rollout_time'])*1000:.2f}ms "
            f"([bold underline orange]FPS={1/np.mean(best_records['rollout_time']):.2f} Hz[reset])"
        ))
        if len(set(best_records['dataset_class'])) > 1:
            for klass in sorted(list(set(best_records['dataset_class']))):
                idxs = [i for i, k in enumerate(best_records['dataset_class']) if k == klass]
                ade = np.array([best_records['ade'][i] for i in idxs])
                fde = np.array([best_records['fde'][i] for i in idxs])
                trajlen = np.array([best_records['trajlen'][i] for i in idxs])
                ped_num = np.array([best_records['ped_num'][i] for i in idxs])
                veh_num = np.array([best_records['veh_num'][i] for i in idxs])
                norm_err = np.array([best_records['norm_err'][i] for i in idxs])
                tan_err = np.array([best_records['tan_err'][i] for i in idxs])
                collision_ped = np.array([best_records['collision_ped'][i] for i in idxs])
                collision_veh = np.array([best_records['collision_veh'][i] for i in idxs])
                collision_map = np.array([best_records['collision_map'][i] for i in idxs])
                apd = np.array([best_records['apd'][i] for i in idxs])
                rollout_time = np.array([best_records['rollout_time'][i] for i in idxs])
                w = np.array([best_records['sample_nums'][i] for i in idxs], dtype=float)
                w /= w.sum()
                acc = 1 - np.sum(w * ade) / np.sum(w * trajlen)
                _logger.info(tag2ansi(
                    f"[#66CCFF][Epoch {best_records['epoch']}/{args.epochs}] Overall on {klass} datasets: "
                    f"[bold underline orange]Accuracy={acc:.2%}[reset], "
                    f"[#66CCFF]ADE={np.sum(w * ade):.4f}, "
                    f"[#66CCFF]FDE={np.sum(w * fde):.4f}, "
                    f"[#66CCFF]X_ERROR (normal)={np.nansum(w * norm_err) / np.sum(w * np.isfinite(norm_err)):.4f}, "
                    f"[#66CCFF]Y_ERROR (tangential)={np.nansum(w * tan_err) / np.sum(w * np.isfinite(tan_err)):.4f}, "
                    f"[#66CCFF]Collision-Ped={np.sum(w * collision_ped):.2%}, "
                    f"[#66CCFF]Collision-Veh={np.sum(w * collision_veh):.2%}, "
                    f"[#66CCFF]Collision-Map={np.sum(w * collision_map):.2%}, "
                    f"[#66CCFF]APD={np.sum(w * apd):.4f}, "
                    f"[#66CCFF]AvgLen={np.sum(w * trajlen):.4f}, "
                    f"[#66CCFF]PedNum={np.sum(w * ped_num):.4f}, "
                    f"[#66CCFF]VehNum={np.sum(w * veh_num):.4f}, "
                    f"[#66CCFF]RolloutTime={np.mean(rollout_time)*1000:.2f}ms "
                    f"([bold underline orange]FPS={1/np.mean(rollout_time):.2f} Hz[reset])"
                ))
        best_path = Path(args.save_path) / 'best.pth'
        checkpoint = torch.load(best_path, map_location=args.device)
        if checkpoint['epoch'] != best_records['epoch']:
            _logger.warning(tag2ansi(
                f"Best epoch in records.jsonl ({best_records['epoch']}) does not match that in best.pth ({checkpoint['epoch']})!"
            ))
        model.load_state_dict(checkpoint["model"])
        _logger.note(f'Load best model from epoch {best_records["epoch"]} ({best_path}) for final test.')
    else:
        best_records = {'epoch': start_epoch - 1}

    ## Test
    torch.set_grad_enabled(False)
    model.eval()
    with npu_attention_fallback_context(model, enable=USE_NPU):
        test_records = test_once(args, test_loaders, model, criterion, diffusion, best_records['epoch'])
    with open(f"{args.save_path}/records.jsonl", "a") as f:
        if test_records is not None:
            f.write(json.dumps(test_records) + "\n")

    ## Log Test Result
    w = np.array(test_records['sample_nums'], dtype=float)
    w /= w.sum()
    test_records['accuracy'] = 1 - np.sum(w * test_records['ade']) / np.sum(w * test_records['trajlen'])
    test_records['unweighted_accuracy'] = 1 - np.mean(test_records['ade']) / np.mean(test_records['trajlen'])
    _logger.note(tag2ansi(
        f"[bold underline orange]Test Accuracy={test_records['accuracy']:.2%}[reset] (unweighted={test_records['unweighted_accuracy']:.2%}), "
        f"at [#66CCFF]epoch {test_records['epoch']}[reset]. "
        f"[#66CCFF]ADE={np.sum(w * test_records['ade']):.4f}, "
        f"[#66CCFF]FDE={np.sum(w * test_records['fde']):.4f}, "
        f"[#66CCFF]X_ERROR (normal)={np.nansum(w * test_records['norm_err']) / np.sum(w * np.isfinite(test_records['norm_err'])):.4f}, "
        f"[#66CCFF]Y_ERROR (tangential)={np.nansum(w * test_records['tan_err']) / np.sum(w * np.isfinite(test_records['tan_err'])):.4f}, "
        f"[#66CCFF]Collision-Ped={np.sum(w * test_records['collision_ped']):.2%}, "
        f"[#66CCFF]Collision-Veh={np.sum(w * test_records['collision_veh']):.2%}, "
        f"[#66CCFF]Collision-Map={np.sum(w * test_records['collision_map']):.2%}, "
        f"[#66CCFF]APD={np.sum(w * test_records['apd']):.4f}, "
        f"[#66CCFF]AvgLen={np.sum(w * test_records['trajlen']):.4f}, "
        f"[#66CCFF]Loss={np.sum(w * test_records['loss']):.4f}, "
        f"[#66CCFF]PedNum={np.sum(w * test_records['ped_num']):.1f}, "
        f"[#66CCFF]VehNum={np.sum(w * test_records['veh_num']):.1f}, "
        f"[#66CCFF]RolloutTime={np.mean(test_records['rollout_time'])*1000:.2f}ms "
        f"([bold underline orange]FPS={1/np.mean(test_records['rollout_time']):.2f} Hz)[reset])"
    ))
    if len(set(test_records['dataset_class'])) > 1:
        for klass in sorted(list(set(test_records['dataset_class']))):
            idxs = [i for i, k in enumerate(test_records['dataset_class']) if k == klass]
            ade = np.array([test_records['ade'][i] for i in idxs])
            fde = np.array([test_records['fde'][i] for i in idxs])
            trajlen = np.array([test_records['trajlen'][i] for i in idxs])
            ped_num = np.array([test_records['ped_num'][i] for i in idxs])
            veh_num = np.array([test_records['veh_num'][i] for i in idxs])
            norm_err = np.array([test_records['norm_err'][i] for i in idxs])
            tan_err = np.array([test_records['tan_err'][i] for i in idxs])
            collision_ped = np.array([test_records['collision_ped'][i] for i in idxs])
            collision_veh = np.array([test_records['collision_veh'][i] for i in idxs])
            collision_map = np.array([test_records['collision_map'][i] for i in idxs])
            apd = np.array([test_records['apd'][i] for i in idxs])
            rollout_time = np.array([test_records['rollout_time'][i] for i in idxs])
            w = np.array([test_records['sample_nums'][i] for i in idxs], dtype=float)
            w /= w.sum()
            acc = 1 - np.sum(w * ade) / np.sum(w * trajlen)
            _logger.info(tag2ansi(
                f"[#66CCFF][Final Test] Overall on {klass} datasets: "
                f"[bold underline orange]Accuracy={acc:.2%}[reset], "
                f"[#66CCFF]ADE={np.sum(w * ade):.4f}, "
                f"[#66CCFF]FDE={np.sum(w * fde):.4f}, "
                f"[#66CCFF]X_ERROR (normal)={np.nansum(w * norm_err) / np.sum(w * np.isfinite(norm_err)):.4f}, "
                f"[#66CCFF]Y_ERROR (tangential)={np.nansum(w * tan_err) / np.sum(w * np.isfinite(tan_err)):.4f}, "
                f"[#66CCFF]Collision-Ped={np.sum(w * collision_ped):.2%}, "
                f"[#66CCFF]Collision-Veh={np.sum(w * collision_veh):.2%}, "
                f"[#66CCFF]Collision-Map={np.sum(w * collision_map):.2%}, "
                f"[#66CCFF]APD={np.sum(w * apd):.4f}, "
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
    parser.add_argument('--test', action='store_true', help="是否仅运行测试流程（跳过训练）")
    parser.add_argument("--name", type=str, default="train_trace", help="实验任务名称，用于生成实验ID")
    parser.add_argument("--exp_name", type=str, default=None, help="手动指定实验名称（若指定则覆盖自动生成的名称）")
    parser.add_argument("--device", type=str, default="auto", help="计算设备，可选 'cpu', 'cuda:0' 或 'auto'（自动选择显存充足的 GPU）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，固定以复现实验结果")
    parser.add_argument("--save_dir", type=str, default="./logs/train_trace", help="日志和模型权重的保存根目录")
    parser.add_argument("--debug", action="store_true", help="是否开启调试模式（输出更多日志，不保存部分文件）")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader 的工作线程数（0 表示主线程）")
    parser.add_argument("--minimize_gpu", action="store_true", default=False, help="是否在每个 epoch 结束后尽可能释放显存以供其他进程使用")
    
    # 训练超参数
    # parser.add_argument("--batch_size", type=int, default=128, help="训练批次大小")
    # parser.add_argument("--lr", type=float, default=2e-4, help="学习率 (Learning Rate)")
    # parser.add_argument("--epochs", type=int, default=10000, help="最大训练轮数")
    parser.add_argument('--patience', type=int, default=20, help="Early Stopping 的耐心值（多少个 epoch 验证集指标不提升则停止）")
    parser.add_argument('--loss_type', type=str, default='noise', choices=['position', 'accelerate', 'noise'], help="损失函数计算的目标类型")
    parser.add_argument('--reload_checkpoint', type=str, default=None, help="断点续训的 checkpoint 路径（.pth 文件）")
    parser.add_argument('--force_new_experiment', action='store_true', help="是否强制不使用 checkpoint 继续训练，即使存在 checkpoint 文件")
    parser.add_argument('--required_memory_MB', type=int, default=6500, help="自动选择 GPU 时要求的最小剩余显存 (MB)")

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
    parser.add_argument('--eval_ratio', type=float, default=0.2, help="训练集划分为训练/验证集的比例 (KDD)")
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
    parser.add_argument('--test_before_train', action='store_true', default=False, help="是否在训练开始前先运行一次测试")
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

    # TRACE 特定参数
    parser.add_argument('--batch_size', type=int, default=8, help='minibatch size')
    parser.add_argument('--epochs', type=int, default=10000, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')

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
