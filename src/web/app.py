from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import asyncio, json, os
import numpy as np
import torch
# from utils.dataset_utils import load_demo_dataset
# from utils.model import DummyModel

import numpy as np

def to_serializable(obj):
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
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

# Mount static
static_dir = os.path.join(os.path.dirname(__file__), "static")
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
    dataset = load_demo_dataset()
    STATE["dataset"] = dataset
    return JSONResponse({"status": "ok", "num_frames": dataset["num_frames"], "H": dataset["H"], "W": dataset["W"]})

@app.get("/load_model")
async def load_model():
    model = DummyModel()
    STATE["model"] = model
    return JSONResponse({"status": "ok"})

@app.get("/frame/{frame_idx}")
async def get_frame(frame_idx: int):
    dataset = STATE["dataset"]
    if dataset is None:
        return JSONResponse({"error": "no dataset"}, status_code=400)
    frame_idx = int(frame_idx)
    frame_idx = np.clip(frame_idx, 0, dataset["num_frames"] - 1)
    # gather current positions
    frame_data = {
        "frame_idx": frame_idx,
        "agents": {
            aid: {
                "type": info["type"],
                "pos": info["positions"][frame_idx]
            }
            for aid, info in dataset["agents"].items()
        },
        "map": dataset["maps"][frame_idx].tolist() if dataset["maps"].ndim == 3 else dataset["maps"].tolist()
    }
    STATE["current_frame"] = frame_idx
    return JSONResponse(to_serializable(frame_data))

@app.get("/agent/{agent_id}")
async def get_agent(agent_id: str):
    dataset = STATE["dataset"]
    if dataset is None:
        return JSONResponse({"error": "no dataset"}, status_code=400)
    if agent_id not in dataset["agents"]:
        return JSONResponse({"error": "invalid agent_id"}, status_code=400)
    info = dataset["agents"][agent_id]
    return JSONResponse({
        "agent_id": agent_id,
        "type": info["type"],
        "positions": info["positions"]
    })

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
    frame_idx = STATE["current_frame"]
    obs = {
        aid: info["positions"][frame_idx]
        for aid, info in dataset["agents"].items()
    }
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
1. JS 访问时必须用 ip 而非 hostname，否则会失败
2. HTML 中必须写 <meta http-equiv="Content-Security-Policy" content="connect-src *;">，否则会失败
3. FastAPI 中必须写 /ws 而非 /simulate/ws，否则会失败
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