""" uvicorn app:app --host 0.0.0.0 --port 12345 """
import json
import torch
import asyncio
import logging
import traceback
import pandas as pd
from pathlib import Path
from copy import deepcopy
from argparse import Namespace
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from src.model import Model, RelativeModel, NewModel
from src.diffusion import DDPM, DDIM
from src.dataset import UCYDataset, ETHDataset, GCDataset, SDDDataset, WayMoDataset, ORCADataset
from src.utils.logger import init_logger
from src.utils.auto_gpu import AutoGPU
from src.web.utils.json_compatible import json_compatible
from src.web.utils.simulate import init_simulation, simulate_one_step
from src.utils.use_npu import USE_NPU, npu_attention_fallback

_logger = logging.getLogger("src")
init_logger('src')

DEFAULT_ARGS = Namespace(
    # **json.load(open('logs/train/20251127_RelativeModel_添加ReLU_修改Freq_添加Anchor_153734_DL4/args.json', 'r')),
    use_new_model=False,
    use_relative_model=False,
    use_spatial_anchor=False,
    dest_importance=0.0,
    des_cfg=0.0,
    map_cfg=0.0,
    use_sfm=False,
    direction_cg=0.0,
    energy_cg=0.0,
    sfm_des_cg=0.0,
    t_des_force=0.5,
    sfm_map_cg=0.0,
    r=10,
    a_map_force=3.0,
    d_map_force=0.6,
    sfm_social_cg=0.0,
    a_ped_force=2.0,
    d_ped_force=0.3,
    a_veh_force=5.0,
    d_veh_force=0.5,
    vel_damping=0.5,
)
OVERWRITE_ARGS = Namespace(
    sampling_method='DDIM',
    beta_schedule='linear',
    denoise_step=10,
    cache_dataset=True,
)
DATASET_FILE = Path('./data/datasets.csv')
DATASET_LIST = None # 所有可用的 Dataset
MODEL_DIR = Path('logs/train')
MODEL_LIST = None # 所有可用的 Model
DATASET_DICT = {} # 缓存加载的真实数据集以及模拟的仿真数据集
MANAGER_DICT = {} # 缓存每个 WebSocket 连接对应的仿真任务
MODEL = None
ARGS = None

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

@app.get("/favicon.ico")
async def get_favicon():
    with open("src/web/static/favicon.ico", "rb") as f:
        icon_content = f.read()
    return HTMLResponse(content=icon_content, status_code=200)

@app.get("/api/dataset_list")
async def dataset_list():
    """ 获取所有可用的数据集列表 """
    global DATASET_LIST
    if DATASET_LIST is None:
        DATASET_LIST = pd.read_csv(DATASET_FILE, sep='\t')
        DATASET_LIST['full_name'] = '[' + DATASET_LIST['dataset'] + '] ' + DATASET_LIST['name']
    return JSONResponse(content=json_compatible(DATASET_LIST['full_name'].to_dict()))


@app.get("/api/model_list")
async def model_list():
    """ 获取所有可用的模型列表 """
    global MODEL_LIST
    if MODEL_LIST is None:
        MODEL_LIST = [p.name for p in sorted(MODEL_DIR.glob('*'), key=lambda x: x.stat().st_mtime, reverse=False) if (p / 'best.pth').exists()][::-1]
        # if (DEFAULT_MODEL := '20251103_new-1-CFG_111635_DL4') in MODEL_LIST: # 将指定模型放在第一位
        #     MODEL_LIST.insert(0, MODEL_LIST.pop(MODEL_LIST.index(DEFAULT_MODEL)))
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
    elif row['dataset'] == 'ORCADataset':
        dataset = ORCADataset.load_data(ARGS, row['path'])
    else:
        raise ValueError(f"Unknown dataset type: {row['dataset']}")
    global DATASET_DICT
    DATASET_DICT[name] = dataset
    response = {
        "name": name,
        "fps": ARGS.fps,
        "frames": { # {frame: {id: {"type":..., "x":..., "y":...}, ...}, ...}
            f: (
                group.set_index("id")
                .sort_index()[["type", "x", "y"]]
                .to_dict(orient="index")
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
    path = MODEL_DIR / MODEL_LIST[idx] / 'best.pth'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    global ARGS
    ARGS = Namespace(**(vars(DEFAULT_ARGS) | checkpoint["args"] | vars(OVERWRITE_ARGS)))
    global MODEL
    if ARGS.use_new_model:
        MODEL = NewModel(ARGS).to(ARGS.device)
    elif ARGS.use_relative_model:
        MODEL = RelativeModel(ARGS).to(ARGS.device)
    else:
        MODEL = Model(ARGS).to(ARGS.device)
    MODEL.load_state_dict(checkpoint["model"])
    MODEL.eval()
    if USE_NPU: npu_attention_fallback(MODEL)
    torch.set_grad_enabled(False)
    return JSONResponse(content=json_compatible({
        "status": "ok", "response": vars(ARGS), "msg": f"Model checkpoint loaded from {path}."
    }))


@app.post("/api/update_args")
async def update_args(new_args: dict):
    """ 更新全局 ARGS 参数 """
    global ARGS
    if ARGS is None:
        return JSONResponse(content={"status": "error", "msg": "Model not loaded yet."})
    
    # 更新参数 (仅更新 ARGS 中已有的或新传入的)
    vars(ARGS).update(new_args)
    
    _logger.info(f"Args updated via API: {new_args}")
    return JSONResponse(content={"status": "ok", "msg": "Parameters updated successfully."})


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
                frame_num = data['frame_num']
                save_name = f'sim-{dataset_name}-from-{frame_idx}'
                result_queue = asyncio.Queue() # maxsize=10
                sendclient_worker_task = asyncio.create_task(
                    sendclient_worker(ws, dataset_name, frame_idx, save_name, result_queue)
                )
                simulation_worker_task = asyncio.create_task(
                    simulation_worker(ws, dataset_name, frame_idx, save_name, result_queue, frame_num)
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
            map_data = DATASET_DICT[save_name].map_data
            df_data = DATASET_DICT[save_name].df_data
            df_data = df_data[(df_data['f'] <= frame_idx) & (df_data['f'] >= frame_idx - ARGS.hist_step)]
            DATASET_DICT[save_name].df_data = df_data
            response = {
                "name": save_name,
                "fps": ARGS.fps,
                "frames": {
                    f: (
                        group.set_index("id")
                        .sort_index()[["type", "x", "y"]]
                        .to_dict(orient="index")
                    ) for f, group in df_data.groupby("f", sort=True)
                },
                "map": {
                    "grid": map_data.map, 
                    "xmin": map_data.xmin, 
                    "xmax": map_data.xmax,
                    "ymin": map_data.ymin, 
                    "ymax": map_data.ymax,
                },
            }
            await ws.send_json(json_compatible({
                'status': 'ok', 'data': response, 'msg': f'Initialized simulation dataset {save_name} from {dataset_name} up to frame {frame_idx}.'
            }))
        while True:
            df_new_frame = await result_queue.get()
            # _logger.info(str(df_new_frame))
            # df_new_frame['f'] = df_new_frame['f'].astype(int)
            # df_new_frame['id'] = df_new_frame['id'].astype(int)
            DATASET_DICT[save_name].df_data = pd.concat([DATASET_DICT[save_name].df_data, df_new_frame], ignore_index=True)
            response = {
                'name': save_name, 'map': None, 'frames': {
                    f: group.set_index('id').sort_index()[['type', 'x', 'y']].to_dict(orient='index') 
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
        msg = f"Simulation worker encountered an error: [{type(e)}] {e}\n{traceback.format_exc()}"
        ws.send_json({'status': 'error', 'msg': msg})
        _logger.error(msg)
        raise

async def simulation_worker(ws: WebSocket, dataset_name: str, frame_idx: int, save_name: str, result_queue: asyncio.Queue, frame_num: int):
    """ 从 dataset_name 的 frame_idx 帧开始进行模拟 """
    try:
        if ARGS.sampling_method == 'DDPM':
            diffusion = DDPM(ARGS)
        elif ARGS.sampling_method == 'DDIM':
            diffusion = DDIM(ARGS)
        else:
            raise ValueError(f"Unknown sampling method: {ARGS.sampling_method}")
        dataset = DATASET_DICT[dataset_name]
        ARGS.device = AutoGPU().choice_gpu(3000, force=False) if not USE_NPU else 'npu'
        MODEL.to(ARGS.device)
        diffusion.to(ARGS.device)
        _logger.info(f"Simulation worker using device {ARGS.device}")
        ws.send_json({'status': 'ok', 'msg': f'Simulation worker using device {ARGS.device}.'})
        now = await asyncio.to_thread(init_simulation, ARGS, dataset, frame_idx, MODEL) # 运行 100~200ms
        _logger.info(f"Frame {frame_idx}: {len(now[0])} pedestrians, {len(now[1])} vehicles.")
        for _ in range(frame_num):
            df_new, now = await asyncio.to_thread(simulate_one_step, ARGS, MODEL, diffusion, *now) # 运行 100~200ms
            await result_queue.put(df_new)
    except asyncio.CancelledError:
        _logger.info("Simulation worker cancelled.")
        raise
    except Exception as e:
        msg = f"Simulation worker encountered an error: [{type(e)}] {e}\n{traceback.format_exc()}"
        ws.send_json({'status': 'error', 'msg': msg})
        _logger.error(msg)
        raise


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=12345) # , reload=True, reload_includes=["src/", 'app.py'])
