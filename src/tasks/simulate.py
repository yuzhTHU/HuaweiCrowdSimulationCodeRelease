import torch
import logging
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
from .guidance import guidance

_logger = logging.getLogger('src.simulate')

@dataclass
class SimulateState:
    df_ped: pd.DataFrame  # Pedestrian dataframe.
    df_veh: pd.DataFrame # Vehicle dataframe.
    map_data: object # Map data object.
    ped_list: List[int] # Pedestrian ID list (#pedestrian,)
    veh_list: List[int] # Vehicle ID list (#vehicle,)
    frame: int  # Current frame.
    pos_now: torch.Tensor # Current pedestrian positions (Batch, #pedestrian, 2)
    vel_now: torch.Tensor # Current pedestrian velocities (Batch, #pedestrian, 2)
    hst_now: torch.Tensor # Current pedestrian history (Batch, #pedestrian, hist_step, 2)
    des_now: torch.Tensor  # Current pedestrian destinations (Batch, #pedestrian, 2)
    spd_now: torch.Tensor  # Current pedestrian desired speeds (Batch, #pedestrian, 1)
    veh_now: torch.Tensor  # Current vehicle positions (Batch, #vehicle, hist_step + 1, 2)


def init_simulation(
    args: Namespace, dataset: BaseDataset, frame_idx: int, model: Model
) -> SimulateState:
    """
    Initialize the state required for simulation.

    Args:
        args: Configuration arguments.
        dataset: Dataset object.
        frame_idx: Initial frame index.
        model: Model object.

    Returns:
        SimulateState: Initialized simulation state.
    """
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
        # .fillna(0.0)  # There should be no NaN here.
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

    # Override destinations with user-defined values if present.
    # `user_destinations` uses string keys, so `ped_id` must be converted before lookup.
    if hasattr(dataset, 'user_destinations') and dataset.user_destinations:
        for idx, ped_id in enumerate(ped_list):
            ped_id_str = str(ped_id)
            if ped_id_str in dataset.user_destinations:
                user_des = dataset.user_destinations[ped_id_str]
                des[idx, 0] = user_des['x']
                des[idx, 1] = user_des['y']
                _logger.info(f"Using user-defined destination for pedestrian {ped_id}: ({user_des['x']}, {user_des['y']})")
    # Compute desired speed `spd`.
    # Simulated datasets may be truncated and therefore lack enough future frames.
    # In that case, use the current speed as an estimate.
    spd_data_range = range(frame_idx, frame_idx + int(5 * args.fps) + 1)
    available_frames = df_ped.index.get_level_values('f').unique()
    has_future_data = any(f in available_frames for f in spd_data_range if f > frame_idx)

    if has_future_data:
        # Enough future frames are available, compute normally.
        spd = (
            df_ped
            .reindex(pd.MultiIndex.from_product([
                spd_data_range,
                ped_list
            ], names=['f', 'id']))
            .unstack().ffill().bfill().diff().mul(args.fps).iloc[1:]
            .stack(future_stack=True)
            .pow(2).sum(axis='columns', min_count=2).pow(0.5)
            .unstack().mean(axis='rows')
            .values[:, np.newaxis]
        )
    else:
        # The simulated dataset is truncated, so use the current speed as the desired speed estimate.
        _logger.info(f"No future data for spd calculation, using current velocity as estimated desired speed")
        spd = np.linalg.norm(vel, axis=1, keepdims=True)
        # Use a reasonable default desired speed if the current speed is too small.
        default_spd = 1.0
        spd = np.where(spd < default_spd, default_spd, spd)
    assert spd.shape == (len(ped_list), 1), "spd shape mismatch"
    map_data = dataset.map_data
    if args.no_destination: 
        des *= np.nan
    if args.no_speed: 
        spd *= np.nan

    ## Simulation
    S = args.sample_num  # Number of samples.
    N = args.denoise_step  # Number of denoising steps.
    assert args.T % N == 0, f"Requested {N} sampling steps, but training steps {args.T} mod {N} is not 0."
    assert 1 <= args.step_offset <= args.T // N, f"`step_offset` must lie in {{1, ..., {args.T // N}}}."
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
    return SimulateState(
        df_ped=df_ped,
        df_veh=df_veh,
        map_data=map_data,
        ped_list=ped_list,
        veh_list=veh_list,
        frame=frame,
        pos_now=pos_now,
        vel_now=vel_now,
        hst_now=hst_now,
        des_now=des_now,
        spd_now=spd_now,
        veh_now=veh_now,
    )


def simulate_one_step(
    args: Namespace, model: Model, diffusion: DDPM, state: SimulateState
) -> Tuple[pd.DataFrame, SimulateState]:
    model.eval()
    torch.set_grad_enabled(False)

    # _logger.info(f"Simulating from frame {frame} to frame {frame + args.pred_step}...")
    S = args.sample_num  # Number of samples.
    N = args.denoise_step  # Number of denoising steps.
    ped_length_repeat = torch.full((S, ), len(state.ped_list), device=args.device, dtype=torch.long)  # (S,)
    veh_length_repeat = torch.full((S, ), len(state.veh_list), device=args.device, dtype=torch.long)  # (S,)

    # _logger.info(f"  Starting setting vehicle embeddings...")
    model.set_veh_embedding(veh=state.veh_now)
    # _logger.info(f"  Starting setting pedestrian embeddings...")
    model.set_ped_embedding(pos=state.pos_now, vel=state.vel_now, hst=state.hst_now, des=state.des_now, spd=state.spd_now)
    # _logger.info(f"  Starting setting surrounding info...")
    model.set_sur_info()

    shape = [S, len(state.ped_list), args.pred_step, 2]  # (S*1, #pedestrian, pred_step, 2)
    xt = torch.randn(shape, device=args.device)  # Start from Gaussian noise.
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

        ## Recover the estimated x0.
        x0 = diffusion.noise_to_x0(xt, denoise_t, output) if args.predict_noise else output

        ## Apply different guidance terms. The coefficients are typically around 0~1.
        x0 = x0 + guidance(args, x0, state, model, diffusion, noisy_acc, xt, denoise_t, ped_length_repeat, veh_length_repeat)

        ## Denoise.
        xt = diffusion.denoise(xt, t, x0=x0, stride=min(stride, t))
    acc_new = xt / args.scale_accelerate
    
    if args.use_sfm:
        future_vel = state.vel_now.unsqueeze(-2)
        future_pos = state.pos_now.unsqueeze(-2)
        # Destination-driven force.
        desire_vel = F.normalize(state.des_now.unsqueeze(-2) - future_pos, dim=-1) * state.spd_now.unsqueeze(-2)  # (S*B, #pedestrian, pred_step, 2)
        des_force = (desire_vel - future_vel).nan_to_num(0.0) / args.sfm_t_des  # (S*B, #pedestrian, pred_step, 2)
        # Obstacle repulsion from the scene map.
        F_map = get_force_map(r=args.sfm_r_map, A=args.sfm_a_map, B=args.sfm_b_map, device=args.device)  # (2r+1, 2r+1, 2)
        idx = future_pos[..., 0].sub(model.xmin).div(model.xmax - model.xmin).mul(model.map.shape[0]).round().long().clamp(0, model.map.shape[0] - 1)  # (S*B, #pedestrian, pred_step)
        jdx = future_pos[..., 1].sub(model.ymin).div(model.ymax - model.ymin).mul(model.map.shape[1]).round().long().clamp(0, model.map.shape[1] - 1)  # (S*B, #pedestrian, pred_step)
        patches = extract_patches_torch(model.map, idx.reshape(-1), jdx.reshape(-1), r=args.sfm_r_map).reshape(*idx.shape, 2*args.sfm_r_map+1, 2*args.sfm_r_map+1) # (S*B, #pedestrian, pred_step, 2r+1, 2r+1)
        map_force = (patches[..., None] * F_map).nan_to_num(0.0).flatten(-3, -2).sum(-2) # (S*B, #pedestrian, pred_step, 2)
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
        # Damping force.
        damp_force = -args.sfm_a_damp * future_vel  # (S*B, #pedestrian, pred_step, 2)
        # Total force.
        acc_new = des_force + map_force + ped_force + veh_force + damp_force

    vel_new = state.vel_now.unsqueeze(-2) + acc_new.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
    # Set the velocity of pedestrians who have reached their destination to 0.
    if args.threshold_of_arrive > 0:
        _pos_new = state.pos_now.unsqueeze(-2) + vel_new.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
        arrived = (_pos_new - state.des_now[:, :, None, :]).norm(dim=-1) < args.threshold_of_arrive  # (S*B, #pedestrian, pred_step)
        arrived = arrived.cumsum(dim=-1) > 0  # (S*B, #pedestrian, pred_step)
        vel_new[arrived, :] = 0.0
    pos_new = state.pos_now.unsqueeze(-2) + vel_new.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
    if state.veh_list:
        veh_new = torch.from_numpy(
            state.df_veh
            .loc[state.frame+1:state.frame+args.pred_step]  # Pandas slicing is inclusive, so this returns exactly `pred_step` frames.
            .unstack().swaplevel(axis='columns').sort_index(axis='columns')
            .reindex(
                index=range(state.frame+1, state.frame+args.pred_step+1),
                columns=pd.MultiIndex.from_product([state.veh_list, ['x', 'y']])
            )
            .values.reshape(args.pred_step, len(state.veh_list), 2) # (pred_step, #vehicle, 2)
            .transpose(1, 0, 2) # (#vehicle, pred_step, 2)
        ).to(device=args.device, dtype=torch.float32)
    des_new = state.des_now  # (S*B, #pedestrian, 2)
    spd_new = state.spd_now  # (S*B, #pedestrian, 1)
    # _logger.info(f"  Computed new positions and velocities.")

    state.hst_now = torch.cat([state.hst_now, state.pos_now.unsqueeze(-2), pos_new], dim=-2)[:, :, -args.hist_step-1:-1, :] # (S*B, #pedestrian, hist_step, 2)
    if state.veh_list:
        state.veh_now = torch.cat([state.veh_now, veh_new.unsqueeze(0).repeat(S, 1, 1, 1)], dim=-2)[:, :, -args.hist_step-1:, :] # (S*B, #vehicle, hist_step + 1, 2)
    state.pos_now = pos_new[:, :, -1, :] # (S*B, #pedestrian, 2)
    state.vel_now = vel_new[:, :, -1, :] # (S*B, #pedestrian, 2)
    state.spd_now = spd_new  # (S*B, #pedestrian, 1)
    state.des_now = des_new  # (S*B, #pedestrian, 2)
    # _logger.info(f"  Updated states for next step.")

    # _logger.info(f"  Converting new positions to CPU numpy. {pos_new.shape}, {type(pos_new)}")
    df_ped_new = pd.DataFrame([
        {
            'f': state.frame + 1 + f,
            'id': state.ped_list[i],
            'type': 'pedestrian',
            'x': float(pos_new[s, i, f, 0]),
            'y': float(pos_new[s, i, f, 1]),
            'sample': s,
        }
        for i in range(pos_new.shape[1])
        for f in range(pos_new.shape[2])
        for s in range(pos_new.shape[0])
    ])
    # _logger.info(f"  Converted new positions to CPU numpy. {pos_new.shape}, {type(pos_new)}")
    df_veh_orig = state.df_veh.loc[state.frame+1:state.frame+args.pred_step].reset_index().assign(type='vehicle')
    df_veh_new = pd.concat([
        df_veh_orig.assign(sample=s) for s in range(pos_new.shape[0])
    ], ignore_index=True)
    df_new = pd.concat([df_ped_new, df_veh_new], ignore_index=True)
    df_new['sample'] = df_new['sample'].astype(int)
    state.frame = state.frame + args.pred_step
    # _logger.info(f"Completed simulation up to frame {frame}.")

    return df_new, state
