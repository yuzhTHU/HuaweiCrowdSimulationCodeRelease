from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import asyncio, json, os
import numpy as np
import torch
from argparse import Namespace
from pathlib import Path
import logging
import pandas as pd
import torch.utils.data as D
from src.dataset import ETHDataset, UCYDataset, SDDDataset, GCDataset, WayMoDataset
from src.model.model import Model
from src.diffusion import DDPM, DDIM

# from utils.dataset_utils import load_demo_dataset
# from utils.model import DummyModel

import numpy as np
import math

def to_serializable(obj):
    if isinstance(obj, (int, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (float, np.float64, np.float32)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return to_serializable(obj.tolist())
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj

def load_demo_dataset(num_frames=200, H=100, W=150, n_peds=10, n_veh=3):
    maps = np.zeros((num_frames, H, W), dtype=np.uint8)
    agents = {}
    rng = np.random.RandomState(0)
    for i in range(n_peds):
        x0, y0 = rng.randint(0, W), rng.randint(0, H)
        vx, vy = rng.randn()*0.5, rng.randn()*0.5
        pos = [(x0 + vx*t, y0 + vy*t) for t in range(num_frames)]
        agents[f"ped_{i}"] = {"type": "ped", "positions": pos}
    for i in range(n_veh):
        x0, y0 = rng.randint(0, W), rng.randint(0, H)
        vx, vy = rng.randn()*1.0, rng.randn()*0.3
        pos = [(x0 + vx*t, y0 + vy*t) for t in range(num_frames)]
        agents[f"veh_{i}"] = {"type": "veh", "positions": pos}
    return {"maps": maps, "agents": agents, "num_frames": num_frames, "H": H, "W": W}

import numpy as np

class DummyModel:
    def rollout(self, obs_positions, num_steps=1):
        rng = np.random.RandomState(42)
        future = []
        cur = dict(obs_positions)
        for _ in range(num_steps):
            step = {}
            for k,(x,y) in cur.items():
                if np.isnan(x): continue
                nx = x + rng.randn()*1.5
                ny = y + rng.randn()*1.5
                step[k] = (float(nx), float(ny))
            future.append(step)
            cur = step
        return future

app = FastAPI()

args = Namespace(name='train', device='cuda:0', batch_size=256, lr=0.001, epochs=10000, patience=20, sampling_method='DDIM', T=100, sample_num=20, denoise_step=10, step_offset=1, antithetic_sampling=True, loss_type='noise', rollout_lambda=1.0, hist_step=8, pred_step=12, skip_step=1, roll_step=1, fps=2.5, dot_per_meter=5, seed=4280, save_dir='./logs/train', debug=False, model_dim=128, map_feature_dim=64, head_num=4, dropout=0.3, latent_token_num=16, beta_schedule='linear', num_workers=0, datasets='ETH/UCY', test_name=['zara01'], test_ratio=None, split_by_scenario=False, cache_dataset=True, test_before_train=True, test_per_epoch=50, save_per_epoch=50, reload_checkpoint=None, predict_noise=True)


# Mount static
static_dir = os.path.join(os.path.dirname(__file__), "src/web/static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Global state
STATE = {
    "dataset": None,
    "model": None,
    "running": False,
    "current_frame": 0,
    "sim_future": [],
}

@app.get("/")
async def index():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/load_dataset")
async def load_dataset():
    if args.debug:
        dataset_list = [UCYDataset.load_data(args, "./data/UCY/data/data_zara/crowds_zara01.vsp")]
    elif args.datasets == "ETH/UCY":
        dataset_list = [
            *UCYDataset.load_data_batch(args, "./data/UCY/data/"),
            *ETHDataset.load_data_batch(args, "./data/ETH/"),
        ]
    elif args.datasets == "GC":
        dataset_list = [GCDataset.load_data(args, "./data/GC/Annotation")]
    elif args.datasets == "SDD":
        dataset_list = SDDDataset.load_data_batch(args, "./data/SDD/annotations/")
    elif args.datasets == 'WayMo':
        dataset_list = WayMoDataset.load_data_batch(args, "./data/WayMo/Processed/")
    elif args.datasets == "zara01":
        dataset_list = [UCYDataset.load_data(args, "./data/UCY/data/data_zara/crowds_zara01.vsp")]
        dataset_list[0].samples = dataset_list[0].samples[:1]
    elif args.datasets == 'debug':
        # dataset_list = [SDDDataset.load_data(args, "./data/SDD/annotations/hyang/video0/annotations.txt")]
        dataset_list = [WayMoDataset.load_data(args, './data/WayMo/Processed/00002_47_93c31aa2d098f5e6/data.csv.gz')]
        dataset_list[0].samples = dataset_list[0].samples[int(len(dataset_list[0].samples) * 0.8):]
    else:
        raise ValueError(f"Unknown dataset {args.datasets}!")
    dataset = [d for d in dataset_list if 'zara01' in d.name][0]
    STATE["dataset"] = dataset
    H, W = dataset.map_data.map.shape[:2]
    num_frames = dataset.df_data['f'].nunique()
    return JSONResponse(to_serializable({"status": "ok", "num_frames": num_frames, "H": H, "W": W}))

@app.get("/load_model")
async def load_model():
    model = Model(args).to(args.device)
    if args.sampling_method == "DDIM":
        diffusion = DDIM(args)
    elif args.sampling_method == "DDPM":
        diffusion = DDPM(args, flexibility=0.0)
    else:
        raise ValueError(f"Unknown sampling_method {args.sampling_method}!")
    if args.reload_checkpoint is not None:
        # 如果指定了 checkpoint 路径，则从该路径加载
        checkpoint_path = Path(args.reload_checkpoint)
    elif (Path(args.save_path) / "checkpoint.pth").exists():
        # 如果当前保存路径下存在 checkpoint，则从该路径加载
        checkpoint_path = Path(args.save_path) / "checkpoint.pth"
    else:
        raise ValueError("No checkpoint found to load!")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found!")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    start_epoch = checkpoint["epoch"] + 1
    model.load_state_dict(checkpoint["model"])
    STATE["model"] = model
    STATE["diffusion"] = diffusion
    return JSONResponse({"status": "ok"})

@app.get("/frame/{frame_idx}")
async def get_frame(frame_idx: int):
    dataset = STATE["dataset"]
    if dataset is None:
        return JSONResponse({"error": "no dataset"}, status_code=400)
    assert isinstance(frame_idx, int), f"frame_idx must be int, got {type(frame_idx)}"
    frame_idx = int(frame_idx)
    df_data = dataset.df_data
    df_data = df_data[df_data['f'] == frame_idx]
    STATE["current_frame"] = frame_idx
    data = to_serializable({
        "frame_idx": frame_idx,
        "agents": {
            row['id']: {
                'type': row['type'],
                'positions': [row['x'], row['y']]
            } for idx, row in df_data.iterrows()
        },
        "map": dataset.map_data.map.tolist()
    })
    return JSONResponse(data)

@app.get("/agent/{agent_id}")
async def get_agent(agent_id: str):
    dataset = STATE["dataset"]
    if dataset is None:
        return JSONResponse({"error": "no dataset"}, status_code=400)
    if agent_id not in dataset.df_data['id']:
        return JSONResponse({"error": "invalid agent_id"}, status_code=400)
    df_data = dataset.df_data
    df_data = df_data[df_data['id'] == agent_id]
    return JSONResponse(to_serializable({
        "agent_id": agent_id,
        "type": df_data.iloc[0]['type'],
        "positions": df_data[['x', 'y']].values.tolist()
    }))

# ---------------- Simulation control ----------------

@app.post("/simulate/start")
async def start_simulation():
    if STATE["running"]:
        return JSONResponse({"status": "already running"})
    if STATE["model"] is None or STATE["dataset"] is None:
        return JSONResponse({"error": "no model/dataset loaded"}, status_code=400)
    STATE["running"] = True
    STATE["sim_future"] = []
    asyncio.create_task(run_simulation_loop())
    return JSONResponse({"status": "started"})

@app.post("/simulate/stop")
async def stop_simulation():
    STATE["running"] = False
    return JSONResponse({"status": "stopped"})

async def run_simulation_loop():
    """后台异步生成rollout并通过WS广播"""
    model = STATE["model"]
    dataset = STATE["dataset"]
    diffusion = STATE["diffusion"]
    frame_idx = STATE["current_frame"]
    df_data = dataset.df_data
    
    df = df_data[df_data['f'].ge(frame_idx - args.hist_step) & df_data['f'].lt(frame_idx + 1)]
    ped_data = df[df['type'].eq('pedestrian')]
    ped_list = ped_data['id'].unique().tolist()
    ped_table = (
        ped_data
        .pivot_table(index='f', columns='id', values=['x', 'y'])
        .reindex(index=range(frame_idx - args.hist_step, frame_idx + 1), 
                 columns=pd.MultiIndex.from_product([['x', 'y'], ped_list]))
        .swaplevel(axis='columns')
        .sort_index(axis='columns')
        .interpolate(method='linear', limit_area='inside', axis=0)
        .ffill().bfill()
    )
    pos = ped_table.loc[frame_idx].values.reshape(len(ped_list), 2)  # (#ped, 2)
    vel = ped_table.diff().loc[frame_idx].mul(args.fps).fillna(0).values.reshape(len(ped_list), 2)  # (#ped, 2)
    hst = (
        ped_table
        .iloc[:args.hist_step]
        .values
        .reshape(args.hist_step, len(ped_list), 2)
        .transpose(1, 0, 2)
    )  # (#ped, hist_step, 2)
    future_5s = (
        df_data[
            df_data['f'].ge(frame_idx - 1) &  # 包含当前帧，以允许计算下一步的速度
            df_data['f'].lt(frame_idx + 5 * args.fps) &
            df_data['id'].isin(ped_list)
        ]
        .pivot_table(index='f', columns='id', values=['x', 'y'])
        .reindex(index=range(frame_idx-1, int(frame_idx + 5 * args.fps) + 1), 
                    columns=pd.MultiIndex.from_product([['x', 'y'], ped_list]))
        .swaplevel(axis='columns').sort_index(axis='columns')
        .interpolate(method='linear', limit_area='inside', axis='rows')
        .ffill().bfill()
    )
    spd = (
        future_5s
        .diff().mul(args.fps).iloc[1:]  # 去掉第一行 NaN（对应于当前第 f 帧的速度），只剩未来 5s
        .swaplevel(axis='columns').stack(future_stack=True)
        .pow(2).sum(axis='columns').pow(0.5)
        .unstack()
        .mean(axis='rows').values
        [..., np.newaxis]
    )  # (#ped, 1)

    des = (
        df_data[df_data['id'].isin(ped_list)]
        .groupby('id').tail(1)
        .set_index('id').reindex(index=ped_list)
        [['x', 'y']].values
    )  # (#ped, 2)

    # 车辆信息作为条件
    veh_data = df[df['type'].eq('vehicle') & df['f'].ge(frame_idx - args.hist_step) & df['f'].lt(frame_idx + 1)]
    veh_list = veh_data['id'].unique().tolist()
    veh_table = (
        veh_data
        .pivot_table(index='f', columns='id', values=['x', 'y'])
        .reindex(index=range(frame_idx - args.hist_step, frame_idx + 1),
                    columns=pd.MultiIndex.from_product([['x', 'y'], veh_list]))
        .swaplevel(axis='columns').sort_index(axis='columns')
        .interpolate(method='linear', limit_area='inside', axis='rows')
    )
    veh = (
        veh_table
        .iloc[:args.hist_step + 1]
        .values
        .reshape(args.hist_step + 1, len(veh_list), 2)
        .transpose(1, 0, 2)
    )  # (#vehicle, hist_step + 1, 2)
    future_veh = (
        veh_table
        .iloc[-args.pred_step:]
        .values
        .reshape(args.pred_step, len(veh_list), 2)
        .transpose(1, 0, 2)
    )  # (#ped, pred_step, 2)


    for f in range(frame_idx, frame_idx + 100):
        if not STATE["running"]:
            break
        
    loader = D.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
    )
    for batch_idx, batch in enumerate(loader):
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

        S = args.sample_num  # 采样次数
        N = args.denoise_step  # 采样步数
        assert args.T % N == 0, f"试图使用 {N} 步采样，然而训练步数 {args.T} mod {N} 不等于 0!"
        assert 1 <= args.step_offset <= args.T // N, f"step_offset 应该取值于 {{1, ..., {args.T // N}}}!"
        pos_now = pos.repeat(S, 1, 1)  # (S*B, #pedestrian, 2)
        vel_now = vel.repeat(S, 1, 1)  # (S*B, #pedestrian, 2)
        hst_now = hst.repeat(S, 1, 1, 1)  # (S*B, #pedestrian, hist_step, 2)
        des_now = des.repeat(S, 1, 1)  # (S*B, #pedestrian, 2)
        spd_now = spd.repeat(S, 1, 1)  # (S*B, #pedestrian, 1)
        veh_now = veh.repeat(S, 1, 1, 1)  # (S*B, #vehicle, hist_step + 1, 2)
        ped_length_repeat = ped_length.repeat(S)  # (S*B,)
        veh_length_repeat = veh_length.repeat(S)  # (S*B,)

        for_plot = []
        acc_pred = []
        for step in range(args.roll_step):
            model.set_veh_embedding(veh=veh_now)
            model.set_ped_embedding(pos=pos_now, vel=vel_now, hst=hst_now, des=des_now, spd=spd_now)
            model.set_sur_info()

            shape = list(future_acc.shape)
            shape[0] *= S
            shape[2] = args.pred_step
            xt = torch.randn(shape, device=args.device)  # 从噪声开始
            for_plot.append([diffusion.noise_to_x0(xt=xt, denoise_t=args.T, noise=0) / args.scale_accelerate])
            stride = args.T // N
            steps = reversed(range(args.step_offset, args.T+1, stride))
            for t in steps:
                noisy_acc = xt
                denoise_t = torch.full((xt.shape[0],), t, device=args.device, dtype=torch.long)
                output = model(
                    noisy_acc=noisy_acc, 
                    denoise_t=denoise_t,
                    ped_length=ped_length_repeat, 
                    veh_length=veh_length_repeat,
                )  # (S*B, #pedestrian, pred_step, 2)
                if args.predict_noise:
                    for_plot[-1].append(diffusion.noise_to_x0(xt=xt, denoise_t=t, noise=output) / args.scale_accelerate)
                    xt = diffusion.denoise(xt, t, noise=output, stride=min(stride, t))
                else:
                    for_plot[-1].append(output / args.scale_accelerate)
                    xt = diffusion.denoise(xt, t, x0=output, stride=min(stride, t))
            acc_pred.append(xt / args.scale_accelerate)  # (S*B, #pedestrian, pred_step, 2)

            acc_new = xt / args.scale_accelerate  # (S*B, #pedestrian, pred_step, 2)
            vel_new = vel_now.unsqueeze(-2) + acc_new.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
            pos_new = pos_now.unsqueeze(-2) + vel_new.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
            veh_new = future_veh[:, :, step*args.pred_step:(step+1)*args.pred_step, :].repeat(S, 1, 1, 1)  # (S*B, #vehicle, pred_step, 2)

            hst_now = torch.cat([hst_now, pos_now.unsqueeze(-2), pos_new], dim=-2)[:, :, -args.hist_step-1:-1, :] # (S*B, #pedestrian, hist_step, 2)
            veh_now = torch.cat([veh_now, veh_new], dim=-2)[:, :, -args.hist_step-1:, :]  # (S*B, #vehicle, hist_step + 1, 2)
            pos_now = pos_new[:, :, -1, :] # (S*B, #pedestrian, 2)
            vel_now = vel_new[:, :, -1, :] # (S*B, #pedestrian, 2)

        acc_pred = torch.concat(acc_pred, dim=-2)  # (S*B, #pedestrian, roll_step*pred_step, 2)
        batch_size, ped_num, _, _ = future_acc.shape
        acc_pred = acc_pred.view(S, batch_size, ped_num, args.roll_step*args.pred_step, 2)  # (S, B, #pedestrian, roll_step*pred_step, 2)

    for step in range(50):
        if not STATE["running"]:
            break
        future_step = model.rollout(obs, num_steps=1)[0]
        STATE["sim_future"].append(future_step)
        obs = future_step
        # 推送到WebSocket
        await broadcast_ws({"step": step, "future_step": future_step})
        await asyncio.sleep(0.1)
    STATE["running"] = False

# ---------------- WebSocket broadcast ----------------
websockets = set()
"""
三个坑：
1. JS 访问时必须用 ip 而非 hostname, 否则会失败
2. HTML 中必须写 <meta http-equiv="Content-Security-Policy" content="connect-src *;">, 否则会失败
3. FastAPI 中必须写 /ws 而非 /simulate/ws, 否则会失败
4. Firefox 只能同时连接 1 个 WebSocket, Chrome 才能同时连多个
"""

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    websockets.add(ws)
    try:
        while True:
            await ws.receive_text()  # ignore incoming
    except WebSocketDisconnect:
        websockets.remove(ws)

async def broadcast_ws(data):
    if not websockets:
        return
    msg = json.dumps(data)
    for ws in list(websockets):
        try:
            await ws.send_text(msg)
        except Exception:
            websockets.remove(ws)