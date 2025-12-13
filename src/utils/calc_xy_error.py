import torch


def calc_xy_error(traj_diff, ped_pos, veh_pos, veh_vel):
    """
    计算基于车辆方向的法向和切向误差
    
    Args:
        traj_diff: (N, T, 2) 预测轨迹误差向量
        ped_pos:   (N, 2) 行人初始位置
        veh_pos:   (N, 2) 最近车辆位置 (若无效则为 NaN)
        veh_vel:   (N, 2) 最近车辆速度 (若无效则为 NaN)
        
    Returns:
        avg_norm_err: 平均法向误差
        avg_tang_err: 平均切向误差
    """
    # 场景无车辆的无效数据
    valid_data_mask = ~veh_pos.isnan().any(dim=-1) # (N,)

    # 计算射线方向向量
    speed = veh_vel.norm(dim=-1, keepdim=True) # (N, 1)
    vel_dir = veh_vel / (speed + 1e-8) # (N, 2)

    # 计算行人到射线的距离
    rel_pos = ped_pos - veh_pos # (N, 2)
    proj_len = (rel_pos * vel_dir).sum(dim=-1, keepdim=True) # (N, 1)
    perp_vec = rel_pos - proj_len * vel_dir
    dist_to_ray = torch.norm(perp_vec, dim=-1) # (N,)
    
    # 综合筛选条件 (1. 数据有效  2. 车辆在移动  3. 行人距离车辆前进射线距离 < 10m)
    final_mask = valid_data_mask & (speed.squeeze(-1) > 0.01) & (dist_to_ray < 10.0)
    if not final_mask.any():
        return float('nan'), float('nan')
        
    # 误差分解
    target_diff = traj_diff[final_mask]
    target_dir = vel_dir[final_mask].unsqueeze(1)
    # 切向误差 (Tangential): 在方向向量上的投影
    tang_err = torch.abs((target_diff * target_dir).sum(dim=-1)) # (N_subset, T)
    # 法向误差 (Normal): 在法向量上的投影
    target_norm_vec = torch.stack([-target_dir[..., 1], target_dir[..., 0]], dim=-1)
    norm_err = torch.abs((target_diff * target_norm_vec).sum(dim=-1)) # (N_subset, T)
    
    # 返回平均值
    return norm_err.mean().item(), tang_err.mean().item()
