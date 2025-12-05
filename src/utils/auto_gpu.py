import os
import time
import torch
import logging
import subprocess

__all__ = ["AutoGPU"]

_logger = logging.getLogger(__name__)


class AutoGPU:
    """
    自动显存管理工具，用于选择剩余显存充足的 GPU。
    """
    def __init__(self):
        """
        初始化 AutoGPU，获取当前可见的 CUDA 设备列表。
        """
        visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            self.gpu_list = list(map(int, visible_devices.split(",")))
        else:
            self.gpu_list = [i for i in range(torch.cuda.device_count())]
        self.free_memory = {
            i: self.query_free_memory(j) for i, j in enumerate(self.gpu_list)
        }  # cuda:i -> memory of j-th GPU

    @staticmethod
    def allocate_gpu(device, memory_MB: int, block_MB: int = None):
        """
        [内部方法] 在指定设备上分配显存占位符。
        
        用于通过实际分配显存来测试显存是否确实可用，或者用于抢占显存。
        
        Args:
            device (str or torch.device): 目标设备。
            memory_MB (int): 需要分配的显存大小 (MB)。
            block_MB (int, optional): 分块大小。如果为 None，则一次性分配。
        
        Returns:
            torch.Tensor or List[torch.Tensor]: 占用的显存张量引用。
        """
        if block_MB is None:
            return torch.zeros(memory_MB, 1024, 256, dtype=torch.float32, device=device)
        else:
            blocks = [block_MB] * (memory_MB // block_MB)
            if sum(blocks) < memory_MB:
                blocks.append(memory_MB % block_MB)
            assert (
                sum(blocks) == memory_MB
            ), f"Sum of blocks {sum(blocks)} != {memory_MB}"
            return [
                torch.zeros(block, 1024, 256, dtype=torch.float32, device=device)
                for block in blocks
            ]

    def choice_gpu(self, memory_MB, interval=600, force=True):
        """
        选择一个具有足够剩余显存的 GPU。
        
        该方法不仅查询 `nvidia-smi`，还会尝试实际分配显存以确保可用性。
        如果所有 GPU 都忙，且 force=True，则会阻塞等待。

        Args:
            memory_MB (int): 任务所需的最小显存 (MB)。
            interval (int, optional): 轮询检查的间隔时间 (秒)。默认为 600。
            force (bool, optional): 是否强制等待直到有 GPU 可用。
                如果为 False 且无可用 GPU，将返回 "cpu"。默认为 True。

        Returns:
            str: 选定的设备字符串，如 "cuda:0" 或 "cpu"。
        """
        waiting = False
        while True:
            for i, free_memory in self.free_memory.items():
                if free_memory < memory_MB:
                    continue
                try:
                    device = f"cuda:{i}"
                    free_memory1 = self.query_free_memory(self.gpu_list[i])
                    allocation = self.allocate_gpu(
                        device=device, memory_MB=memory_MB, block_MB=512
                    )
                    free_memory2 = self.query_free_memory(self.gpu_list[i])
                    (_logger.note if waiting else _logger.info)(
                        f"SubProcess[{os.getpid()}]: Choose GPU{self.gpu_list[i]} ({device}) "
                        f"with {memory_MB}MB ({free_memory1}MB -> {free_memory2}MB)"
                    )
                    del allocation
                    torch.cuda.reset_peak_memory_stats(
                        device
                    )  # 不要让 allocation 影响 torch.cuda.max_memory_allocated
                    return device
                except Exception:
                    torch.cuda.empty_cache()
                    continue
            else:
                if force:
                    if not waiting:
                        _logger.warning(f"SubProcess[{os.getpid()}]: Waiting GPU...")
                        waiting = True
                    time.sleep(interval)
                    self.update_free_memory()
                else:  # not force
                    _logger.warning(f"SubProcess[{os.getpid()}]: No available GPU!")
                    return "cpu"

    def query_free_memory(self, gpu_id):
        try:
            cmd = f"nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i {gpu_id}"
            return int(
                subprocess.check_output(cmd, shell=True).decode().strip().split("\n")[0]
            )
        except Exception as e:
            _logger.warning(f"Query CUDA (GPU{gpu_id}) Memory Failed! {e}")
            return 0

    def update_free_memory(self):
        for i, j in enumerate(self.gpu_list):
            self.free_memory[i] = self.query_free_memory(j)
