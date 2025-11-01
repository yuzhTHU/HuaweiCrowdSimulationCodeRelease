import os
import io
import json
import base64
import asyncio
import numpy as np
from pydantic import BaseModel
from typing import Dict, Any, Optional
from starlette.requests import Request
from starlette.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File

# --- imports for your model/dataset code (adjust as needed) ---
# from src.dataset import UCYDataset, ETHDataset, ...
# from src.model.model import Model
# from src.diffusion import DDIM, DDPM
# (We'll assume you have functions to load dataset & checkpoint and run one rollout)
# -------------------------------------------------------------------

app = FastAPI()
BASE_DIR = os.path.dirname(__file__)  # project root
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "src", "web", "static"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "src", "web", "static")), name="static")

# ---------- Simple in-memory stores ----------
DATASETS: Dict[str, Dict] = {}  # dataset_id -> metadata & df/map
CHECKPOINTS: Dict[str, Dict] = {}  # ckpt_id -> metadata
SIMULATIONS: Dict[str, Dict] = {}  # sim_id -> state (history, current_step, paused, frames, map...)
WS_CLIENTS: Dict[str, WebSocket] = {}

# ---------- Pydantic models ----------
class LoadDatasetReq(BaseModel):
    dataset_path: str
    dataset_id: Optional[str] = None

class LoadCheckpointReq(BaseModel):
    checkpoint_path: str
    ckpt_id: Optional[str] = None

class StartSimReq(BaseModel):
    dataset_id: str
    frame_idx: int
    checkpoint_id: str
    sim_id: Optional[str] = None
    sample_num: int = 1
    roll_step: int = 12
    pred_step: int = 1
    future_5s_speed: Optional[float] = None  # optional override of speed
    destinations: Optional[Dict[int, list]] = None  # {ped_id: [x,y], ...} manual destinations

# ---------- Utilities ----------
def encode_map_to_png_bytes(map_array: np.ndarray) -> bytes:
    """
    map_array: H x W (0/1) or grayscale - convert to PNG bytes (so frontend can load as image)
    """
    from PIL import Image
    if map_array.dtype != np.uint8:
        # normalize to 0..255
        ma = map_array.astype(float)
        mi, ma_ = ma.min(), ma.max()
        if ma_ - mi > 0:
            ma = (ma - mi) / (ma_ - mi) * 255.0
        else:
            ma = ma * 0
        arr = ma.astype(np.uint8)
    else:
        arr = map_array
    img = Image.fromarray(arr).convert("L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ---------- Placeholder: dataset loader ----------
def load_dataset_from_path(path: str):
    """
    Replace this with your dataset load routine, returning a dict:
    {
        "map": map_data,  # numpy HxW
        "map_meta": {xmin, xmax, ymin, ymax},
        "df_data": df_data,  # pandas DataFrame like in your code with multiindex (f, id)
        "n_frames": n_frames,
        "frame_index_list": list_of_frames,
    }
    """
    # --- DUMMY example: generate empty map and no agents ---
    H, W = 400, 600
    map_arr = np.zeros((H, W), dtype=np.uint8)
    map_meta = {"xmin": 0.0, "xmax": W/10.0, "ymin": 0.0, "ymax": H/10.0}
    from pandas import DataFrame
    df = DataFrame()  # you must fill this in with your dataset's df_data
    return {
        "map": map_arr,
        "map_meta": map_meta,
        "df_data": df,
        "n_frames": 0,
        "frame_index_list": [],
    }

# ---------- Placeholder: load checkpoint ----------
def load_checkpoint(path: str):
    """
    Load your model checkpoint, return an object you will use later to instantiate model
    """
    return {"path": path}

# ---------- Placeholder: perform rollout (adapt your pasted code into this function) ----------
def run_rollout_once(checkpoint_obj, dataset_obj, frame_idx: int, *,
                    sample_num=1, roll_step=12, pred_step=1,
                    future_5s_speed=None, destinations=None):
    """
    Run model rollout starting at frame_idx and return a dictionary:
    {
    "sim_id": str,
    "ped_list": [...],  # list of ped ids
    "veh_list": [...],  # list of vehicle ids
    "hist_positions": { id: [ [x,y] ... ] }, # up to frame_idx history
    "ground_truth_future": { id: [ [x,y] ... ] }, # available true future from dataset
    "simulated_future": { sample_index: { id: [ [x,y], ... ] } } # simulated
    }
    IMPORTANT: adapt this function to call into your model code (the main() snippet you gave).
    For demo we return an empty result.
    """
    # TODO: Replace with direct call into your model's simulation logic.
    # e.g., use your code to prepare pos_now, hst_now, veh_now, then run the denoising loop and build `traj`
    result = {
        "sim_id": f"sim_{frame_idx}_{np.random.randint(1e6)}",
        "ped_list": [],
        "veh_list": [],
        "hist_positions": {},
        "ground_truth_future": {},
        "simulated_future": {},
    }
    return result

# ---------- REST endpoints ----------
@app.post("/api/load_dataset")
async def api_load_dataset(req: LoadDatasetReq):
    ds = load_dataset_from_path(req.dataset_path)
    dsid = req.dataset_id or f"ds_{len(DATASETS)+1}"
    DATASETS[dsid] = ds
    # encode map as base64 png for quick return
    png = encode_map_to_png_bytes(ds["map"])
    b64 = base64.b64encode(png).decode("ascii")
    return {"dataset_id": dsid, "n_frames": ds["n_frames"], "map_png_b64": b64, "map_meta": ds["map_meta"]}

@app.post("/api/load_checkpoint")
async def api_load_checkpoint(req: LoadCheckpointReq):
    ck = load_checkpoint(req.checkpoint_path)
    ckid = req.ckpt_id or f"ck_{len(CHECKPOINTS)+1}"
    CHECKPOINTS[ckid] = ck
    return {"checkpoint_id": ckid, "path": req.checkpoint_path}

@app.get("/api/dataset/{dataset_id}/frame_count")
async def api_frame_count(dataset_id: str):
    if dataset_id not in DATASETS:
        return JSONResponse({"error": "dataset not found"}, status_code=404)
    return {"n_frames": DATASETS[dataset_id]["n_frames"]}

@app.get("/api/dataset/{dataset_id}/frame/{frame_idx}")
async def api_frame(dataset_id: str, frame_idx: int):
    """
    Return the metadata and positions for a single frame:
    - history positions for each agent up to frame_idx (for display)
    - current positions for this frame
    - vehicle positions
    - ground-truth future (optional)
    """
    if dataset_id not in DATASETS:
        return JSONResponse({"error": "dataset not found"}, status_code=404)
    ds = DATASETS[dataset_id]
    # NOTE: adapt to your dataset DataFrame structure
    # For demo we return empty
    return {
        "frame_idx": frame_idx,
        "ped_list": [],  # [id,...]
        "veh_list": [],
        "history": {},  # id -> [[x,y],...]
        "current": {},  # id -> [x,y]
        "gt_future": {},  # id -> [[x,y], ...]
    }

# ---------- WebSocket Protocol ----------
# Frontend will open /ws and send JSON messages of form:
# { "type": "subscribe", "client_id": "abc" }
# { "type": "start_sim", "dataset_id":..., "frame_idx":..., "checkpoint_id":..., ... }
# { "type": "stop_sim", "sim_id": "..." }
# { "type": "seek_sim", "sim_id": "...", "step": N }  # to view a previous simulation step
# server sends updates:
# { "type": "frame_update", "frame_idx": N, "current": {...}, "history": {...}, "gt_future": {...} }
# { "type": "sim_step", "sim_id": ..., "sim_step_idx": K, "simulated_future": {...}, "sim_done": bool }
# { "type": "sim_state", ... }

# Helper: broadcast to single websocket
async def ws_send_safe(ws: WebSocket, data: dict):
    try:
        await ws.send_text(json.dumps(data, default=lambda o: None))
    except Exception:
        pass

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    client_id = f"client_{id(ws)}_{np.random.randint(1e6)}"
    WS_CLIENTS[client_id] = ws
    try:
        while True:
            text = await ws.receive_text()
            try:
                msg = json.loads(text)
            except Exception:
                continue
            typ = msg.get("type")
            if typ == "ping":
                await ws_send_safe(ws, {"type": "pong"})
            elif typ == "start_sim":
                # start simulation in background task (async)
                req = StartSimReq(**msg.get("payload", {}))
                sim_id = req.sim_id or f"sim_{np.random.randint(1e9)}"
                # initialize sim state
                SIMULATIONS[sim_id] = {
                    "dataset_id": req.dataset_id,
                    "frame_idx": req.frame_idx,
                    "checkpoint_id": req.checkpoint_id,
                    "sample_num": req.sample_num,
                    "roll_step": req.roll_step,
                    "pred_step": req.pred_step,
                    "future_5s_speed": req.future_5s_speed,
                    "destinations": req.destinations or {},
                    "status": "running",
                    "current_step": 0,
                    "sim_data": None,
                    "clients": [client_id],
                }
                # run rollout in a background task (non-blocking)
                asyncio.create_task(simulation_worker(sim_id, ws))
                await ws_send_safe(ws, {"type": "sim_started", "sim_id": sim_id})
            elif typ == "stop_sim":
                sim_id = msg.get("sim_id")
                if sim_id and sim_id in SIMULATIONS:
                    SIMULATIONS[sim_id]["status"] = "stopped"
                    await ws_send_safe(ws, {"type": "sim_stopped", "sim_id": sim_id})
            elif typ == "seek_sim":
                sim_id = msg.get("sim_id")
                step = int(msg.get("step", 0))
                # return stored sim up to step
                if sim_id in SIMULATIONS and SIMULATIONS[sim_id].get("sim_data") is not None:
                    simdata = SIMULATIONS[sim_id]["sim_data"]
                    # send the portion up to step
                    await ws_send_safe(ws, {"type": "sim_seek", "sim_id": sim_id, "step": step, "simulated_future": simdata.get("simulated_until", {})})
            elif typ == "request_frame":
                dataset_id = msg.get("dataset_id")
                frame_idx = int(msg.get("frame_idx", 0))
                info = await api_frame(dataset_id, frame_idx)
                await ws_send_safe(ws, {"type": "frame_info", "payload": info})
            else:
                await ws_send_safe(ws, {"type": "error", "message": f"unknown type {typ}"})
    except WebSocketDisconnect:
        del WS_CLIENTS[client_id]

# ---------- Simulation worker ----------
async def simulation_worker(sim_id: str, ws: WebSocket):
    """
    This coroutine runs the model rollout in small steps and pushes updates to the connected websocket.
    It stores the full simulation in SIMULATIONS[sim_id]['sim_data'] after each step so the frontend can seek back.
    """
    sim = SIMULATIONS[sim_id]
    dsid = sim["dataset_id"]
    ckd = sim["checkpoint_id"]
    frame_idx = sim["frame_idx"]
    # Validate dataset & checkpoint
    if dsid not in DATASETS or ckd not in CHECKPOINTS:
        await ws_send_safe(ws, {"type": "sim_error", "sim_id": sim_id, "message": "dataset or checkpoint missing"})
        sim["status"] = "stopped"
        return

    # ---------- run the actual rollout (blocking) in threadpool to avoid blocking asyncio loop ----------
    def blocking_rollout():
        # call into your run_rollout_once - must return serializable dict
        res = run_rollout_once(CHECKPOINTS[ckd], DATASETS[dsid], frame_idx,
                            sample_num=sim["sample_num"],
                            roll_step=sim["roll_step"],
                            pred_step=sim["pred_step"],
                            future_5s_speed=sim["future_5s_speed"],
                            destinations=sim["destinations"])
        return res

    # We will simulate in chunks: call the rollout function once (which may return full traj),
    # then stream step-by-step to the client. If your model supports stepwise generation, better to adapt run_rollout_once to yield steps.
    rollout_result = await run_in_threadpool(blocking_rollout)

    # store result
    sim["sim_data"] = rollout_result
    # We'll ambulate through roll steps and send updates:
    # For demonstration send a message per step (sleep a bit)
    total_steps = sim["roll_step"] * max(1, sim["pred_step"])
    # If rollout_result includes simulated frames as array, iterate through them. Here we'll just simulate sending incremental steps.
    for step in range(total_steps):
        if sim["status"] == "stopped":
            break
        # Build payload - in your real run, include simulated_future up to this step
        payload = {
            "type": "sim_step",
            "sim_id": sim_id,
            "step": step,
            "sim_done": (step == total_steps - 1),
            # include simulated trajectories up to this step. Example:
            "simulated_future": rollout_result.get("simulated_future", {}),
        }
        await ws_send_safe(ws, payload)
        sim["current_step"] = step
        await asyncio.sleep(0.1)  # small pause so frontend animates nicely
    sim["status"] = "finished" if sim["status"] != "stopped" else "stopped"
    await ws_send_safe(ws, {"type": "sim_finished", "sim_id": sim_id, "status": sim["status"]})

# ---------- Simple index page ----------
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
