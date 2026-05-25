import torch


def calc_xy_error(traj_diff, ped_pos, veh_pos, veh_vel):
    """
    Compute normal and tangential errors relative to the vehicle direction.
    
    Args:
        traj_diff: (N, T, 2) Predicted trajectory error vectors.
        ped_pos:   (N, 2) Initial pedestrian positions.
        veh_pos:   (N, 2) Nearest vehicle positions (NaN if invalid).
        veh_vel:   (N, 2) Nearest vehicle velocities (NaN if invalid).

    Returns:
        avg_norm_err: Average normal error.
        avg_tang_err: Average tangential error.
    """
    # Invalid data for scenes without vehicles.
    valid_data_mask = ~veh_pos.isnan().any(dim=-1) # (N,)

    # Compute the ray direction vector.
    speed = veh_vel.norm(dim=-1, keepdim=True) # (N, 1)
    vel_dir = veh_vel / (speed + 1e-8) # (N, 2)

    # Compute the pedestrian-to-ray distance.
    rel_pos = ped_pos - veh_pos # (N, 2)
    proj_len = (rel_pos * vel_dir).sum(dim=-1, keepdim=True) # (N, 1)
    perp_vec = rel_pos - proj_len * vel_dir
    dist_to_ray = torch.norm(perp_vec, dim=-1) # (N,)
    
    # Combined filtering: valid data, moving vehicle, and pedestrian within 10m of the forward ray.
    final_mask = valid_data_mask & (speed.squeeze(-1) > 0.01) & (dist_to_ray < 10.0)
    if not final_mask.any():
        return torch.tensor([float('nan')]), torch.tensor([float('nan')])
        
    # Decompose the error.
    target_diff = traj_diff[final_mask]
    target_dir = vel_dir[final_mask].unsqueeze(1)
    # Tangential error: projection onto the direction vector.
    tang_err = torch.abs((target_diff * target_dir).sum(dim=-1)) # (N_subset, T)
    # Normal error: projection onto the normal vector.
    target_norm_vec = torch.stack([-target_dir[..., 1], target_dir[..., 0]], dim=-1)
    norm_err = torch.abs((target_diff * target_norm_vec).sum(dim=-1)) # (N_subset, T)

    # Return the averages.
    return norm_err.mean(dim=-1), tang_err.mean(dim=-1)
