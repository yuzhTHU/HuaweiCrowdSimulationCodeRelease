""" uvicorn app:app --host 0.0.0.0 --port 12345 """
import io
import json
import torch
import asyncio
import tarfile
import logging
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from pydantic import BaseModel
from argparse import Namespace
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from src.model import Model, RelativeModel, NewModel
from src.diffusion import DDPM, DDIM
from src.dataset import UCYDataset, ETHDataset, GCDataset, SDDDataset, WayMoDataset, ORCADataset
from src.utils.logger import init_logger
from src.utils.auto_gpu import AutoGPU
from src.utils.json_compatible import json_compatible
from src.tasks import init_simulation, simulate_one_step
from src.utils.use_npu import USE_NPU, npu_attention_fallback

_logger = logging.getLogger("src")
init_logger('src')

# 顺序即为前端参数编辑器的显示顺序
DEFAULT_ARGS = Namespace(
    # 引导参数
    cg_sfm_des=None, cg_sfm_obs=None, cg_sfm_soc=None, 
    cfg_des=None, cfg_map=None, cg_dir=None, cg_dis=None, 
    sfm_t_des=0.5, sfm_a_ped=25, sfm_a_veh=30, sfm_a_map=30, 
    sfm_b_ped=0.08, sfm_b_veh=0.10, sfm_b_map=0.10, 
    sfm_r_map=10, sfm_a_damp=0.5,
    # 消融参数
    use_sfm=False, no_map=False, no_speed=False, no_destination=False,
    # 模型架构参数
    model_dim=512, map_feature_dim=128, lstm_layer_num=2, head_num=8,
    attention_layer_num=4, latent_token_num=64, dropout=0.1, 
    use_relative_model=True, use_spatial_anchor=True, use_new_model=False,
    use_nan_embedding=True, use_latent_query=True, cache_latent_query=True,
    use_relative_features=True, use_frequency_encoding=True,
    # 扩散采样参数
    T=100, scale_accelerate=1.0, predict_noise=True,
    denoise_step=10, sample_num=10, step_offset=1,
    sampling_method='DDIM', beta_schedule='linear',
    # 数据集参数
    hist_step=8, pred_step=1, skip_step=1, roll_step=12,
    fps=2.5, dot_per_meter=1, cache_dataset=True, 
    # 其他参数
    threshold_of_arrive=3.0,
)
OVERWRITE_ARGS = Namespace(
    cg_sfm_des=0.3, 
    cg_sfm_obs=0.28454697422367176, 
    cg_sfm_soc=0.2460485410529767,
    sfm_a_ped=24.474450660285036, 
    sfm_a_veh=23.142641521977364, 
    sfm_a_map=25.71719034302977,
    sfm_b_ped=0.12717436926298342,
    sfm_b_veh=0.12878228054809737,
    sfm_b_map=0.03217431363919982,
    sfm_a_damp=0.7445477250185598,
    sampling_method='DDIM',
    beta_schedule='linear',
    denoise_step=10,
    cache_dataset=True,
    sample_num=1,
)
DATASET_FILE = Path('./data/datasets.csv')
DATASET_LIST = None # 所有可用的 Dataset
MODEL_DIR = Path('logs/train')
MODEL_LIST = None # 所有可用的 Model
DATASET_DICT = {} # 缓存加载的真实数据集以及模拟的仿真数据集
MANAGER_DICT = {} # 缓存每个 WebSocket 连接对应的仿真任务
MODEL = None
ARGS = None

# 确保保存目录存在
SAVE_DIR = Path("./logs/app/saved_trajectories")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

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
        if (DEFAULT_MODEL := 'All') in MODEL_LIST: # 将指定模型放在第一位
            MODEL_LIST.insert(0, MODEL_LIST.pop(MODEL_LIST.index(DEFAULT_MODEL)))
    return JSONResponse(content=json_compatible({i: s for i, s in enumerate(MODEL_LIST)}))


@app.get("/api/get_high_res_map")
async def get_high_res_map(dataset_name: str):
    """ 获取数据集的高分辨率原图 map.png """
    high_res_map_path = getattr(DATASET_DICT.get(dataset_name), 'high_res_map_path', None)
    if high_res_map_path and Path(high_res_map_path).exists():
        return FileResponse(high_res_map_path, media_type="image/png")
    return JSONResponse(content={"status": "error", "msg": "High-res map not found"}, status_code=404)


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
        },
        "has_high_res_map": False,
    }
    # 获取所有行人的目的地（取每个行人轨迹最后一帧的位置作为目的地）
    all_ped_ids = dataset.df_data.loc[dataset.df_data['type'] == 'pedestrian', 'id'].unique()
    df_ped_coords = dataset.df_data.loc[dataset.df_data['type'] == 'pedestrian'].set_index('id')
    destinations = {}
    for ped_id in all_ped_ids:
        if ped_id in df_ped_coords.index:
            ped_traj = df_ped_coords.loc[ped_id][['x', 'y', 'f']].sort_values('f')
            if len(ped_traj) > 0:
                last_pos = ped_traj.iloc[-1]
                destinations[str(ped_id)] = {'x': float(last_pos['x']), 'y': float(last_pos['y'])}
    response['destinations'] = destinations
    # 如果有用户自定义的目的地，覆盖默认值
    if hasattr(dataset, 'user_destinations') and dataset.user_destinations:
        for ped_id, des in dataset.user_destinations.items():
            destinations[str(ped_id)] = des
    # 初始化仿真状态（用于后续模拟）
    state = init_simulation(ARGS, dataset, 0, MODEL)
    if (map_png_path := Path(row['path']).parent / "map.png").exists():
        DATASET_DICT[name].high_res_map_path = str(map_png_path)
        response['has_high_res_map'] = True
    return JSONResponse(content={"status": "ok", "response": json_compatible(response), "msg": f"Dataset {name} loaded."})


@app.get("/api/load_model")
async def load_model(idx: int):
    path = MODEL_DIR / MODEL_LIST[idx] / 'best.pth'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    global ARGS
    # 只保留 KEEP_ARGS 中的参数
    checkpoint_args = {k: v for k, v in checkpoint["args"].items() if k in vars(DEFAULT_ARGS)}
    ARGS = Namespace(**(vars(DEFAULT_ARGS) | checkpoint_args | vars(OVERWRITE_ARGS)))
    # 设置 device 参数（运行时动态设置）
    ARGS.device = AutoGPU().choice_gpu(3000, force=False) if not USE_NPU else 'npu'
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
        "status": "ok", "response": vars(ARGS), "msg": f"Model checkpoint loaded from {path}.",
        "keep_args_order": list(vars(DEFAULT_ARGS).keys()),  # 返回参数顺序供前端使用
    }))


@app.post("/api/update_args")
async def update_args(new_args: dict):
    """ 更新全局 ARGS 参数 """
    global ARGS
    if ARGS is None:
        return JSONResponse(content={"status": "error", "msg": "Model not loaded yet."})

    # 获取原始参数的类型定义
    default_args_dict = vars(DEFAULT_ARGS)

    # 特定参数的类型映射（对于 DEFAULT_ARGS 中为 None 的参数）
    TYPE_HINTS = {
        'cg_sfm_des': float,
        'cg_sfm_obs': float,
        'cg_sfm_soc': float,
        'cfg_des': float,
        'cfg_map': float,
        'cg_dir': float,
        'cg_dis': float,
    }

    # 转换每个参数的类型
    for key, val in new_args.items():
        # 空字符串视为 None
        if val == '':
            new_args[key] = None
            continue
        elif (original_type := TYPE_HINTS.get(key, type(default_args_dict.get(key, None)))) is None:
            _logger.warning(f"Cannot determine the type of `{key}`={val}, keep it as {type(val)}")
        elif original_type == float:
            try:
                new_args[key] = float(val)
            except (ValueError, TypeError):
                new_args[key] = None
                _logger.error(f"Failed to convert `{key}`={val} into float!")
        elif original_type == int:
            try:
                new_args[key] = int(val)
            except (ValueError, TypeError):
                new_args[key] = None
                _logger.error(f"Failed to convert `{key}`={val} into int!")
        elif original_type == bool:
            if isinstance(val, bool):
                continue
            elif isinstance(val, str) and val.lower() in ('true', '1', 'yes', 'y'):
                new_args[key] = True
            elif isinstance(val, str) and val.lower() in ('false', '0', 'no', 'n'):
                new_args[key] = False
            else:
                new_args[key] = bool(val)
                _logger.warning(f"Convert `{key}`={val} into {bool(val)}")
        # 其他类型保持原样
        else:
            _logger.warning(f"Unsure how to convert `{key}`={val} into {original_type}")

    # 更新参数 (仅更新 ARGS 中已有的或新传入的)
    vars(ARGS).update(new_args)

    _logger.info(f"Args updated via API: {new_args}")
    return JSONResponse(content={"status": "ok", "msg": "Parameters updated successfully."})


class SaveTrajectoryReq(BaseModel):
    name: str
    start_frame: int
    end_frame: int
    destination: str = "server"  # 'server' or 'local'
    compress: bool = True


class UpdateDestinationReq(BaseModel):
    dataset_name: str
    pedestrian_id: str
    destination: dict  # {x: float, y: float}


@app.post("/api/save_trajectory")
async def save_trajector(req: SaveTrajectoryReq):
    if req.name not in DATASET_DICT:
        return JSONResponse(content={"status": "error", "msg": "Dataset not found."})

    dataset = DATASET_DICT[req.name]
    df = dataset.df_data

    mask = (df['f'] >= req.start_frame) & (df['f'] <= req.end_frame)
    df_save = df.loc[mask].copy()
    if df_save.empty:
         return JSONResponse(content={"status": "error", "msg": "No data in selected range."})

    date_str = datetime.now().strftime("%Y-%m-%d")
    clean_name = req.name.replace(" ", "_").replace("/", "-")
    ext = ".csv.gz" if req.compress else ".csv"
    filename = f"{date_str}-{clean_name}{ext}"

    if req.destination == 'server':
        save_file = SAVE_DIR / filename
        df_save.to_csv(save_file, index=False)
        return JSONResponse(content={"status": "ok", "msg": f"Saved to {save_file}"})
    else:
        content = io.BytesIO()
        df_save.to_csv(content, index=False, compression='gzip' if req.compress else None)
        content.seek(0)
        media_type = "application/gzip" if req.compress else "text/csv"
        return Response(
            content=content.getvalue(),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )


@app.post("/api/update_destination")
async def update_destination(req: UpdateDestinationReq):
    """更新指定行人的目的地坐标，存储在 dataset.user_destinations 中"""
    dataset_name = req.dataset_name
    pedestrian_id = req.pedestrian_id
    new_x = req.destination.get('x')
    new_y = req.destination.get('y')

    if dataset_name not in DATASET_DICT:
        return JSONResponse(content={"status": "error", "msg": f"Dataset {dataset_name} not found."})

    try:
        if new_x is None or new_y is None:
            return JSONResponse(content={"status": "error", "msg": "Destination coordinates are required."})

        dataset = DATASET_DICT[dataset_name]

        # 确保 pedestrian_id 是正确类型（可能是 int 或 str）
        try:
            ped_id = int(pedestrian_id)
        except (ValueError, TypeError):
            ped_id = pedestrian_id

        # 初始化或更新 user_destinations 字典
        if not hasattr(dataset, 'user_destinations') or dataset.user_destinations is None:
            dataset.user_destinations = {}

        dataset.user_destinations[ped_id] = {'x': float(new_x), 'y': float(new_y)}

        _logger.info(f"Updated user destination for pedestrian {ped_id} in {dataset_name} to ({new_x}, {new_y})")

        return JSONResponse(content={
            "status": "ok",
            "msg": f"Updated destination for pedestrian {ped_id} to ({new_x}, {new_y})"
        })

    except Exception as e:
        import traceback
        error_msg = f"Failed to update destination: {str(e)}\n{traceback.format_exc()}"
        _logger.error(error_msg)
        return JSONResponse(content={"status": "error", "msg": error_msg})


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
                now_str = datetime.now().strftime('%Y%m%d-%H%M%S')
                save_name = f'sim-{dataset_name}-from-{frame_idx}-{now_str}'
                result_queue = asyncio.Queue() # maxsize=10
                sendclient_worker_task = asyncio.create_task(
                    sendclient_worker(ws, dataset_name, frame_idx, save_name, result_queue)
                )
                simulation_worker_task = asyncio.create_task(
                    simulation_worker(ws, dataset_name, frame_idx, save_name, result_queue, frame_num)
                )
                MANAGER_DICT[ws] = (sendclient_worker_task, simulation_worker_task)
                _logger.info(f"Started simulation for dataset {dataset_name} from frame {frame_idx}, saving to {save_name}. Current #simulations: {len(MANAGER_DICT)}")
                await ws.send_json({"status": "ok", "msg": "Simulation started"})
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
            _, state = await result_queue.get()
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
                "has_high_res_map": getattr(DATASET_DICT[save_name], 'high_res_map_path', None),
                "destinations": {}
            }
            if state.des_now is not None:
                des = state.des_now[0].cpu().numpy()  # (ped_num, 2)
                response['destinations'] = {
                    str(ped_id): { 'x': float(des[idx, 0]), 'y': float(des[idx, 1]) }
                    for idx, ped_id in enumerate(state.ped_list)
                    if not np.isnan(des[idx]).any()
                }
            await ws.send_json(json_compatible({
                'status': 'ok', 'data': response, 'msg': f'Initialized simulation dataset {save_name} from {dataset_name} up to frame {frame_idx}.'
            }))
        while True:
            df_new_frame, state = await result_queue.get()
            # _logger.info(str(df_new_frame))
            # df_new_frame['f'] = df_new_frame['f'].astype(int)
            # df_new_frame['id'] = df_new_frame['id'].astype(int)
            DATASET_DICT[save_name].df_data = pd.concat([DATASET_DICT[save_name].df_data, df_new_frame], ignore_index=True)
            # 只取 sample=0 的数据用于前端可视化（避免同一帧同一 id 有多个位置）
            df_vis = df_new_frame[df_new_frame['sample'] == 0] if 'sample' in df_new_frame.columns else df_new_frame
            response = {
                'name': save_name, 'map': None, 'frames': {
                    f: group.set_index('id').sort_index()[['type', 'x', 'y']].to_dict(orient='index')
                    for f, group in df_vis.groupby('f', sort=True)
                },
                'destinations': {}
            }
            if state.des_now is not None:
                des = state.des_now[0].cpu().numpy()  # (ped_num, 2)
                response['destinations'] = {
                    str(ped_id): { 'x': float(des[idx, 0]), 'y': float(des[idx, 1]) }
                    for idx, ped_id in enumerate(state.ped_list)
                    if not np.isnan(des[idx]).any()
                }
            await ws.send_json(json_compatible({
                'status': 'ok', 'data': response, 'msg': f'Sent simulated frame {df_new_frame["f"].min()}~{df_new_frame["f"].max()} to client.'
            }))
    except asyncio.CancelledError:
        _logger.info("Sendclient worker cancelled.")
        raise
    except Exception as e:
        msg = f"Sendclient worker encountered an error: [{type(e)}] {e}\n{traceback.format_exc()}"
        await ws.send_json({'status': 'error', 'msg': msg})
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
        await ws.send_json({'status': 'ok', 'msg': f'Simulation worker using device {ARGS.device}.'})
        state = await asyncio.to_thread(init_simulation, ARGS, dataset, frame_idx, MODEL) # 运行 100~200ms
        await result_queue.put((None, state))
        _logger.info(f"Frame {frame_idx}: {len(state.ped_list)} pedestrians, {len(state.veh_list)} vehicles.")
        for _ in range(frame_num):
            df_new, state = await asyncio.to_thread(simulate_one_step, ARGS, MODEL, diffusion, state) # 运行 100~200ms
            await result_queue.put((df_new, state))
    except asyncio.CancelledError:
        _logger.info("Simulation worker cancelled.")
        raise
    except Exception as e:
        msg = f"Simulation worker encountered an error: [{type(e)}] {e}\n{traceback.format_exc()}"
        await ws.send_json({'status': 'error', 'msg': msg})
        _logger.error(msg)
        raise


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=12345) # , reload=True, reload_includes=["src/", 'app.py'])
