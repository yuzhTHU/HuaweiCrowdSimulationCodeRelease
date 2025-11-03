""" uvicorn app:app --host 0.0.0.0 --port 12345 """
import json
import torch
import asyncio
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from argparse import Namespace
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from src.model.model import Model
from src.diffusion import DDPM, DDIM
from src.dataset import BaseDataset, UCYDataset, ETHDataset, GCDataset, SDDDataset, WayMoDataset
from src.utils.logger import init_logger
from src.web.utils.json_compatible import json_compatible

_logger = logging.getLogger("src")
init_logger('src')

DATASET_LIST = pd.read_csv('./data/datasets.csv', sep='\t') # 所有可用的 Dataset
DATASET_LIST['full_name'] = '[' + DATASET_LIST['dataset'] + '] ' + DATASET_LIST['name']
MODEL_LIST = [ # 所有可用的 Model
    '20251103_new-1_104153_DL4', '20251103_new-1-BS128_104322_DL4', 
    '20251103_new-1-CFG_111635_DL4', '20251103_new-1-lr5e-4_104337_DL4',
]
DATASET_DICT = {} # 缓存加载的真实数据集以及模拟的仿真数据集
MANAGER_DICT = {} # 缓存每个 WebSocket 连接对应的仿真任务
MODEL = None
with open('logs/train/20251103_new-1-CFG_111635_DL4/args.json', 'r') as f:
    ARGS = Namespace(**json.load(f))

app = FastAPI(title="Pedestrian Simulation Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 前端可跨域访问
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="src/web/static"), name="static")


@app.get("/")
async def get_index():
    with open("src/web/static/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/api/dataset_list")
async def dataset_list():
    """ 获取所有可用的数据集列表 """
    return JSONResponse(content=json_compatible(DATASET_LIST['full_name'].to_dict()))


@app.get("/api/model_list")
async def model_list():
    """ 获取所有可用的模型列表 """
    return JSONResponse(content=json_compatible({i: s for i, s in enumerate(MODEL_LIST)}))


@app.get("/api/load_dataset")
async def load_dataset(idx: int, name: str):
    """ 加载 DATASET_LIST[idx] 对应的数据集，并保存到 DATASET_DICT[name] 中 """
    row = DATASET_LIST.loc[idx]
    if row['dataset'] == 'UCYDataset':
        dataset = UCYDataset.load_data(ARGS, row['path'])
    elif row['dataset'] == 'ETHDataset':
        dataset = ETHDataset.load_data(ARGS, row['path'])
    elif row['dataset'] == 'SDDDataset':
        dataset = SDDDataset.load_data(ARGS, row['path'])
    elif row['dataset'] == 'WayMoDataset':
        dataset = WayMoDataset.load_data(ARGS, row['path'])
    elif row['dataset'] == 'GCDataset':
        dataset = GCDataset.load_data(ARGS, row['path'])
    else:
        raise ValueError(f"Unknown dataset type: {row['dataset']}")
    global DATASET_DICT
    DATASET_DICT[name] = dataset
    response = {
        "name": name,
        "frames": { # {frame: {id: {"type":..., "x":..., "y":...}, ...}, ...}
            f: (
                group.set_index("id")
                .sort_index()[["type", "x", "y"]]
                .to_dict(orient="records")
            ) for f, group in dataset.df_data.groupby("f", sort=True)
        },
        "map": {
            "grid": dataset.map_data.map,
            "xmin": dataset.map_data.xmin,
            "xmax": dataset.map_data.xmax,
            "ymin": dataset.map_data.ymin,
            "ymax": dataset.map_data.ymax,
        }
    }
    return JSONResponse(content={"status": "ok", "response": json_compatible(response), "msg": f"Dataset {name} loaded."})


@app.get("/api/load_model")
async def load_model(idx: int):
    path = Path('logs/train') / MODEL_LIST[idx] / 'best.pth'
    checkpoint = torch.load(path, map_location=ARGS.device, weights_only=True)
    saved_args = Namespace(**checkpoint["args"])
    global MODEL
    MODEL = Model(saved_args).to(ARGS.device)
    MODEL.load_state_dict(checkpoint["model"])
    MODEL.eval()
    torch.set_grad_enabled(False)
    return JSONResponse(content={"status": "ok", "response": MODEL_LIST[idx], "msg": f"Model checkpoint loaded from {path}."})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            action = data.get("action")
            if action == "ping":
                asyncio.create_task(ws.send_json({"status": "pong"}))
            elif action == "start":
                dataset_name = data["dataset_name"]
                frame_idx = data["frame_idx"]
                save_name = f'sim-{dataset_name}-from-{frame_idx}'
                result_queue = asyncio.Queue() # maxsize=10
                sendclient_worker_task = asyncio.create_task(
                    sendclient_worker(ws, dataset_name, frame_idx, save_name, result_queue)
                )
                simulation_worker_task = asyncio.create_task(
                    simulation_worker(ws, dataset_name, frame_idx, save_name, result_queue)
                )
                MANAGER_DICT[ws] = (sendclient_worker_task, simulation_worker_task)
                _logger.info(f"Started simulation for dataset {dataset_name} from frame {frame_idx}, saving to {save_name}. Current #simulations: {len(MANAGER_DICT)}")
            elif action == "stop":
                if ws in MANAGER_DICT:
                    sendclient_worker_task, simulation_worker_task = MANAGER_DICT.pop(ws)
                    simulation_worker_task.cancel()
                    sendclient_worker_task.cancel()
                    asyncio.create_task(ws.send_json({"status": "ok", "msg": "Simulation stopped"}))
                    _logger.info("Simulation stopped by client request.")
                else:
                    asyncio.create_task(ws.send_json({"status": "error", "msg": "No simulation running"}))
                    _logger.info("No simulation to stop for this WebSocket.")
    except WebSocketDisconnect:
        if ws in MANAGER_DICT:
            sendclient_worker_task, simulation_worker_task = MANAGER_DICT.pop(ws)
            simulation_worker_task.cancel()
            sendclient_worker_task.cancel()
        _logger.info("WebSocket disconnected, cleaned up simulation tasks.")


async def sendclient_worker(ws: WebSocket, dataset_name: str, frame_idx: int, save_name: str, result_queue: asyncio.Queue):
    """ 将 result_queue 中订阅的模拟结果保存至 saved_name, 并同时发送给前端客户端 """
    try:
        if save_name not in DATASET_DICT:
            DATASET_DICT[save_name] = deepcopy(DATASET_DICT[dataset_name])
            DATASET_DICT[save_name].df_data = DATASET_DICT[save_name].df_data[DATASET_DICT[save_name].df_data['f'] <= frame_idx]
            df_data = DATASET_DICT[save_name].df_data
            map_data = DATASET_DICT[save_name].map_data
            response = {
                "name": save_name,
                "frames": {
                    f: group.set_index("id").sort_index()[["type", "x", "y"]].to_dict(orient="records")
                    for f, group in df_data.groupby("f", sort=True)
                },
                "map": {
                    "grid": map_data.map, "xmin": map_data.xmin, "xmax": map_data.xmax,
                    "ymin": map_data.ymin, "ymax": map_data.ymax,
                },
            }
            await ws.send_json(json_compatible({
                'status': 'ok', 'data': response, 'msg': f'Initialized simulation dataset {save_name} from {dataset_name} up to frame {frame_idx}.'
            }))
        while True:
            df_new_frame = await result_queue.get()
            DATASET_DICT[save_name].df_data = pd.concat([DATASET_DICT[save_name].df_data, df_new_frame], ignore_index=True)
            response = {
                'name': save_name, 'map': None, 'frames': {
                    f: group.set_index('id').sort_index()[['type', 'x', 'y']].to_dict(orient='records') 
                    for f, group in df_new_frame.groupby('f', sort=True)
                },
            }
            await ws.send_json(json_compatible({
                'status': 'ok', 'data': response, 'msg': f'Sent simulated frame {df_new_frame["f"].min()}~{df_new_frame["f"].max()} to client.'
            }))
    except asyncio.CancelledError:
        _logger.info("Sendclient worker cancelled.")
        raise
    except Exception as e:
        import traceback
        _logger.error(
            f"Simulation worker encountered an error: [{type(e)}] {e}\n"
            f"{traceback.format_exc()}"
        )
        raise

async def simulation_worker(ws: WebSocket, dataset_name: str, frame_idx: int, save_name: str, result_queue: asyncio.Queue):
    """ 从 dataset_name 的 frame_idx 帧开始进行模拟 """
    try:
        diffusion = DDIM(ARGS)
        dataset = DATASET_DICT[dataset_name]
        now = await asyncio.to_thread(init_simulation, ARGS, dataset, frame_idx, MODEL) # 运行 100~200ms
        _logger.info(f"Frame {frame_idx}: {len(now[0])} pedestrians, {len(now[1])} vehicles.")
        while True:
            df_new, now = await asyncio.to_thread(simulate_one_step, ARGS, MODEL, diffusion, *now) # 运行 100~200ms
            await result_queue.put(df_new)
    except asyncio.CancelledError:
        _logger.info("Simulation worker cancelled.")
        raise
    except Exception as e:
        import traceback
        _logger.error(
            f"Simulation worker encountered an error: [{type(e)}] {e}\n"
            f"{traceback.format_exc()}"
        )
        raise


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
        if args.predict_noise:
            xt = diffusion.denoise(xt, t, noise=output, stride=min(stride, t))
        else:
            xt = diffusion.denoise(xt, t, x0=output, stride=min(stride, t))
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
    df_veh_new = df_veh.loc[frame+1:frame+args.pred_step]
    df_new = pd.concat([df_ped_new, df_veh_new], ignore_index=True)
    frame = frame + args.pred_step
    # _logger.info(f"Completed simulation up to frame {frame}.")

    return df_new, (ped_list, veh_list, df_veh, frame, pos_now, vel_now, hst_now, des_now, spd_now, veh_now)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=12345) # , reload=True, reload_includes=["src/", 'app.py'])
