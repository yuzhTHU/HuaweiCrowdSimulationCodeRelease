"""Small, optional bridge that mirrors generated frames into CARLA."""

from __future__ import annotations
import math
import importlib
import threading


class CarlaBridge:
    def __init__(self) -> None:
        self.client = None
        self.world = None
        self.carla = None
        self.connected = False
        self.sync_enabled = False
        self.host = None
        self.port = None
        self.actors = {}
        self.previous = {}
        self._lock = threading.RLock()

    def connect(self, host: str, port: int, timeout: float = 3.0) -> None:
        with self._lock:
            self.sync_enabled = False
            self.connected = False
            self.host, self.port = host, port
            try:
                carla = importlib.import_module("carla")
                client = carla.Client(host, port)
                client.set_timeout(timeout)
                world = client.get_world()  # Forces a round trip and verifies the endpoint.
                self._destroy_actors()
                self.carla, self.client, self.world = carla, client, world
                self.connected = True
            except Exception:
                self.client = None
                self.world = None
                raise
    
    @property
    def status(self) -> dict:
        with self._lock:
            return {
                "connected": self.connected,
                "sync_enabled": self.sync_enabled if self.connected else False,
                "host": self.host,
                "port": self.port,
            }

    def set_sync(self, enabled: bool) -> None:
        with self._lock:
            if enabled and not self.connected:
                raise RuntimeError("CARLA is not connected.")
            self.sync_enabled = enabled

    def sync_frame(self, frame) -> None:
        with self._lock:
            if not self.connected or not self.sync_enabled:
                return
            try:
                rows = frame.to_dict("records")
                current = set()
                library = self.world.get_blueprint_library()
                for row in rows:
                    actor_type = "vehicle" if str(row["type"]).lower() == "vehicle" else "pedestrian"
                    key = (actor_type, str(row["id"]))
                    current.add(key)
                    x, y = float(row["x"]), float(row["y"])
                    old = self.previous.get(key)
                    yaw = math.degrees(math.atan2(y - old[1], x - old[0])) if old else 0.0
                    z = .5 if actor_type == "vehicle" else 1.15
                    transform = self.carla.Transform(
                        self.carla.Location(x=x, y=y, z=z), self.carla.Rotation(yaw=yaw)
                    )
                    actor = self.actors.get(key)
                    if actor is None or not actor.is_alive:
                        blueprint_id = "vehicle.tesla.model3" if actor_type == "vehicle" else "walker.pedestrian.0001"
                        blueprint = library.find(blueprint_id)
                        spawn = self.carla.Transform(self.carla.Location(x=x, y=y, z=z + 100), transform.rotation)
                        actor = self.world.try_spawn_actor(blueprint, spawn)
                        if actor is None:
                            continue
                        actor.set_simulate_physics(False)
                        self.actors[key] = actor
                    actor.set_transform(transform)
                    self.previous[key] = (x, y)
                for key in set(self.actors) - current:
                    actor = self.actors.pop(key)
                    if actor.is_alive:
                        actor.destroy()
                    self.previous.pop(key, None)
            except Exception:
                self.connected = False
                self.sync_enabled = False
                raise

    def _destroy_actors(self) -> None:
        for actor in self.actors.values():
            try:
                if actor.is_alive:
                    actor.destroy()
            except Exception:
                pass
        self.actors.clear()
        self.previous.clear()
