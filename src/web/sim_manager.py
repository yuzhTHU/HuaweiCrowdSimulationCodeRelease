import asyncio
from typing import Dict, List, Tuple, Optional

class SimManager:
    """
    管理数据集、模型、当前模拟状态（真实轨迹 & 模拟轨迹）
    """
    def __init__(self):
        self.dataset = None
        self.model = None
        self.map_info = None
        self.real_traj = {}   # {frame: [{"id":..., "type":..., "x":..., "y":...}, ...]}
        self.sim_traj = {}    # {frame: [{"id":..., "type":..., "x":..., "y":...}, ...]}
        self.running = False
        self.current_frame = None
        self.sim_task: Optional[asyncio.Task] = None

    def load_dataset(self, dataset_data, map_info):
        self.dataset = dataset_data
        self.map_info = map_info
        self.real_traj = {f["f"]: f["objects"] for f in dataset_data}
        return True

    def load_model(self, model):
        self.model = model

    async def start_sim(self, websocket, start_frame: int):
        """模拟 rollout（这里暂时假装是模型预测出的轨迹）"""
        if self.running:
            await websocket.send_json({"status": "error", "msg": "Simulation already running"})
            return

        self.running = True
        self.current_frame = start_frame
        self.sim_traj = {}
        await websocket.send_json({"status": "ok", "msg": f"Simulation started from frame {start_frame}"})

        for i in range(1, 21):  # 模拟 rollout 20 帧
            if not self.running:
                break

            new_frame = start_frame + i
            # 这里只是随机演示：未来可以换成模型 rollout
            new_objects = [
                {"id": obj["id"], "type": obj["type"], "x": obj["x"] + 0.1*i, "y": obj["y"] + 0.05*i}
                for obj in self.real_traj.get(start_frame, [])
            ]
            self.sim_traj[new_frame] = new_objects
            await websocket.send_json({"frame": new_frame, "objects": new_objects})
            await asyncio.sleep(0.5)  # 模拟生成时间

        self.running = False
        await websocket.send_json({"status": "done", "msg": "Simulation finished"})

    def stop_sim(self):
        self.running = False
