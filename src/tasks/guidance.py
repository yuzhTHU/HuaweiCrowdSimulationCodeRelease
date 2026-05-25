import torch
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


def guidance(args, x0, state, model, diffusion, noisy_acc, xt, denoise_t, ped_length_repeat, veh_length_repeat):
    total_guidance = 0.0
    future_acc = x0 / args.scale_accelerate  # (S*B, #pedestrian, pred_step, 2)
    future_vel = state.vel_now.unsqueeze(-2) + future_acc.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
    future_pos = state.pos_now.unsqueeze(-2) + future_vel.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
    # Destination CFG guidance.
    if args.cfg_des is not None:
        if 'model_wo_des' not in locals():
            dst_nan = torch.full_like(state.des_now, torch.nan)
            # model_wo_des = deepcopy(model)
            model_wo_des = Model(model.args).to(device=args.device)
            model_wo_des.load_state_dict(model.state_dict())
            model_wo_des.eval()
            model_wo_des.set_map_embedding(map=model.map, xmin=model.xmin, xmax=model.xmax, ymin=model.ymin, ymax=model.ymax)
            model_wo_des.set_veh_embedding(veh=state.veh_now)
            model_wo_des.set_ped_embedding(pos=state.pos_now, vel=state.vel_now, hst=state.hst_now, des=dst_nan, spd=state.spd_now)
            model_wo_des.set_sur_info()
        output_wo_des = model_wo_des(
            noisy_acc=noisy_acc, 
            denoise_t=denoise_t,
            ped_length=ped_length_repeat, 
            veh_length=veh_length_repeat,
        )  # (S*B, #pedestrian, pred_step, 2)
        x0_wo_des = diffusion.noise_to_x0(xt, denoise_t, output_wo_des) if args.predict_noise else output_wo_des
        total_guidance = total_guidance + args.cfg_des * (x0 - x0_wo_des)
    # Obstacle CFG guidance.
    if args.cfg_map is not None:
        if 'model_wo_map' not in locals():
            map_nan = torch.full_like(model.map, torch.nan)
            model_wo_map = Model(model.args).to(device=args.device)
            model_wo_map.load_state_dict(model.state_dict())
            model_wo_map.eval()
            model_wo_map.set_map_embedding(map=map_nan, xmin=model.xmin, xmax=model.xmax, ymin=model.ymin, ymax=model.ymax)
            model_wo_map.set_veh_embedding(veh=state.veh_now)
            model_wo_map.set_ped_embedding(pos=state.pos_now, vel=state.vel_now, hst=state.hst_now, des=state.des_now, spd=state.spd_now)
            model_wo_map.set_sur_info()
        output_wo_map = model_wo_map(
            noisy_acc=noisy_acc, 
            denoise_t=denoise_t,
            ped_length=ped_length_repeat, 
            veh_length=veh_length_repeat,
        )  # (S*B, #pedestrian, pred_step, 2)
        x0_wo_map = diffusion.noise_to_x0(xt, denoise_t, output_wo_map) if args.predict_noise else output_wo_map
        total_guidance = total_guidance + args.cfg_map * (x0 - x0_wo_map)
    # Destination CG guidance: acceleration should point toward the line from `pos_now` to `des_now`.
    if args.cg_dir is not None:
        direction = F.normalize(state.des_now.unsqueeze(-2) - future_pos, dim=-1).nan_to_num(0.0)  # (S*B, #pedestrian, pred_step, 2)
        # loss = F.mse_loss(future_acc, direction.detach())
        # grad = torch.autograd.grad(loss, x0)[0]
        grad = 2 * (future_acc - direction) / args.scale_accelerate  # Closed-form gradient.
        total_guidance = total_guidance - args.cg_dir * grad
    # Destination CG guidance: acceleration should reduce the energy in the destination potential well.
    if args.cg_dis is not None:
        with torch.enable_grad():
            x0_grad = x0.detach().requires_grad_(True)
            future_acc_grad = x0_grad / args.scale_accelerate  # (S*B, #pedestrian, pred_step, 2)
            future_vel_grad = state.vel_now.unsqueeze(-2) + future_acc_grad.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
            future_pos_grad = state.pos_now.unsqueeze(-2) + future_vel_grad.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
            loss = (state.des_now.unsqueeze(-2) - future_pos_grad).nan_to_num(0.0).pow(2).sum()
            grad = torch.autograd.grad(loss, x0_grad)[0]
        total_guidance = total_guidance - args.cg_dis * grad
    # Social-force destination CG guidance: acceleration should resemble the goal force in the social-force model.
    if args.cg_sfm_des is not None:
        desire_vel = F.normalize(state.des_now.unsqueeze(-2) - future_pos, dim=-1) * state.spd_now.unsqueeze(-2)  # (S*B, #pedestrian, pred_step, 2)
        des_force = (desire_vel - future_vel).nan_to_num(0.0) / args.sfm_t_des  # (S*B, #pedestrian, pred_step, 2)
        grad = 2 * (future_acc - des_force) / args.scale_accelerate
        total_guidance = total_guidance - args.cg_sfm_des * grad
    # Social-force guidance: steer acceleration away from obstacles.
    if args.cg_sfm_obs is not None:
        # Obstacle repulsion force from the scene map.
        F_map = get_force_map(r=args.sfm_r_map, A=args.sfm_a_map, B=args.sfm_b_map, device=args.device)  # (2r+1, 2r+1, 2)
        idx = future_pos[..., 0].sub(model.xmin).div(model.xmax - model.xmin).mul(model.map.shape[0]).round().long().clamp(0, model.map.shape[0] - 1)  # (S*B, #pedestrian, pred_step)
        jdx = future_pos[..., 1].sub(model.ymin).div(model.ymax - model.ymin).mul(model.map.shape[1]).round().long().clamp(0, model.map.shape[1] - 1)  # (S*B, #pedestrian, pred_step)
        patches = extract_patches_torch(model.map, idx.reshape(-1), jdx.reshape(-1), r=args.sfm_r_map).reshape(*idx.shape, 2*args.sfm_r_map+1, 2*args.sfm_r_map+1) # (S*B, #pedestrian, pred_step, 2r+1, 2r+1)
        map_force = (patches[..., None] * F_map).nan_to_num(0.0).flatten(-3, -2).sum(-2) # (S*B, #pedestrian, pred_step, 2)
        # loss = F.mse_loss(future_acc, map_force.detach())
        # grad = torch.autograd.grad(loss, x0)[0]
        grad = 2 * (future_acc - map_force) / args.scale_accelerate  # Closed-form gradient.
        total_guidance = total_guidance - args.cg_sfm_obs * grad
    # Social-force guidance: steer acceleration away from other pedestrians and vehicles.
    if args.cg_sfm_soc is not None:
        # Repulsion from other pedestrians.
        p = future_pos[:, None, :, :, :] - future_pos[:, :, None, :, :] # (S*B, #focal-pedestrian, #other-pedestrian, pred_step, 2)
        d = torch.norm(p, dim=-1, keepdim=True)
        n = -p / d.clamp(min=1e-6)
        F_ped = args.sfm_a_ped * torch.exp(-d / args.sfm_b_ped) * n
        ped_force = F_ped.nan_to_num(0.0).sum(dim=2) # (S*B, #pedestrian, pred_step, 2)
        # Repulsion from vehicles, using their last-frame positions.
        p = state.veh_now[:, :, None, -1:, :] - future_pos[:, None, :, :, :] # (S*B, #vehicle, #pedestrian, pred_step, 2)
        d = torch.norm(p, dim=-1, keepdim=True)
        n = -p / d.clamp(min=1e-6)
        F_veh = args.sfm_a_veh * torch.exp(-d / args.sfm_b_veh) * n
        veh_force = F_veh.nan_to_num(0.0).sum(dim=1) # (S*B, #pedestrian, pred_step, 2)
        # Total social force.
        social_force = ped_force + veh_force  # (S*B, #pedestrian, pred_step, 2)
        # loss = F.mse_loss(future_acc, social_force.detach())
        # grad = torch.autograd.grad(loss, x_in)[0]
        grad = 2 * (future_acc - social_force) / args.scale_accelerate  # Closed-form gradient.
        total_guidance = total_guidance - args.cg_sfm_soc * grad
    if not isinstance(total_guidance, float):
        total_guidance = total_guidance.clip(-1, 1)
    return total_guidance
