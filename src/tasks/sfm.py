# import torch
# import torch.nn.functional as F
# from src.model import model
# from src.tasks.sfm_utils import get_force_map, extract_patches_torch


# def sfm_des(args, pos, vel, des, spd, veh):
#     """
#     Compute destination guidance force with the Social Force Model (SFM).
    
#     Args:
#         args: Argument object containing SFM parameters.
#         pos: Position tensor of shape `(Batch, #pedestrian, 2)`.
#         vel: Velocity tensor of shape `(Batch, #pedestrian, 2)`.
#         des: Destination tensor of shape `(Batch, #pedestrian, 2)`.
#         spd: Desired-speed tensor of shape `(Batch, #pedestrian)`.
#         veh: Vehicle-position tensor of shape `(Batch, #vehicle, pred_step, 2)`.
#     """
#     desire_vel = F.normalize(des.unsqueeze(-2) - future_pos, dim=-1) * state.spd_now.unsqueeze(-2)  # (S*B, #pedestrian, pred_step, 2)
#     des_force = (desire_vel - future_vel).nan_to_num(0.0) / args.t_des_force  # (S*B, #pedestrian, pred_step, 2)


# def sfm(args, pos, vel, des, spd, veh):
#     """
#     Compute acceleration with the Social Force Model (SFM).
    
#     Args:
#         args: Argument object containing SFM parameters.
#         pos: Position tensor of shape `(Batch, #pedestrian, 2)`.
#         vel: Velocity tensor of shape `(Batch, #pedestrian, 2)`.
#         des: Destination tensor of shape `(Batch, #pedestrian, 2)`.
#         spd: Desired-speed tensor of shape `(Batch, #pedestrian)`.
#         veh: Vehicle-position tensor of shape `(Batch, #vehicle, pred_step, 2)`.
#     """
#     future_vel = vel.unsqueeze(-2)
#     future_pos = pos.unsqueeze(-2)
#     # Destination-driven force.
#     desire_vel = F.normalize(des.unsqueeze(-2) - future_pos, dim=-1) * spd.unsqueeze(-2)  # (Batch, #pedestrian, pred_step, 2)
#     des_force = (desire_vel - future_vel).nan_to_num(0.0) / args.t_des_force  # (Batch, #pedestrian, pred_step, 2)
#     # Obstacle repulsion force from the scene map.
#     F_map = get_force_map(r=args.r, A=args.a_map_force, B=args.d_map_force, device=args.device)  # (2r+1, 2r+1, 2)
#     idx = future_pos[..., 0].sub(model.xmin).div(model.xmax - model.xmin).mul(model.map.shape[0]).round().long().clamp(0, model.map.shape[0] - 1)  # (Batch, #pedestrian, pred_step)
#     jdx = future_pos[..., 1].sub(model.ymin).div(model.ymax - model.ymin).mul(model.map.shape[1]).round().long().clamp(0, model.map.shape[1] - 1)  # (Batch, #pedestrian, pred_step)
#     patches = extract_patches_torch(model.map, idx.reshape(-1), jdx.reshape(-1), r=10).reshape(*idx.shape, 2*args.r+1, 2*args.r+1) # (Batch, #pedestrian, pred_step, 2r+1, 2r+1)
#     map_force = (patches[..., None] * F_map).nan_to_num(0.0).flatten(-3, -2).sum(-2) # (Batch, #pedestrian, pred_step, 2)
#     # Repulsion from other pedestrians.
#     p = future_pos[:, None, :, :, :] - future_pos[:, :, None, :, :] # (Batch, #focal-pedestrian, #other-pedestrian, pred_step, 2)
#     d = torch.norm(p, dim=-1, keepdim=True)
#     n = -p / d.clamp(min=1e-6)
#     F_ped = args.a_ped_force * torch.exp(-d / args.d_ped_force) * n
#     ped_force = F_ped.nan_to_num(0.0).sum(dim=1) # (Batch, #pedestrian, pred_step, 2)
#     # Repulsion from vehicles, using their last-frame positions.
#     p = veh[:, :, None, -1:, :] - future_pos[:, None, :, :, :] # (Batch, #vehicle, #pedestrian, pred_step, 2)
#     d = torch.norm(p, dim=-1, keepdim=True)
#     n = -p / d.clamp(min=1e-6)
#     F_veh = args.a_veh_force * torch.exp(-d / args.d_veh_force) * n
#     veh_force = F_veh.nan_to_num(0.0).sum(dim=1) # (Batch, #pedestrian, pred_step, 2)
#     # Damping force.
#     vel_force = -args.vel_damping * future_vel  # (Batch, #pedestrian, pred_step, 2)
#     # Total force.
#     acc = des_force + map_force + ped_force + veh_force + vel_force

#     return acc
