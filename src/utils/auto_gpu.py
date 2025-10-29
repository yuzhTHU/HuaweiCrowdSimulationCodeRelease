import os
import time
import torch
import logging
import subprocess

__all__ = ["AutoGPU"]

_logger = logging.getLogger(__name__)


class AutoGPU:
    """Automatically choose a GPU with enough free memory"""

    def __init__(self):
        visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            self.gpu_list = list(map(int, visible_devices.split(",")))
        else:
            self.gpu_list = [i for i in range(torch.cuda.device_count())]
        self.free_memory = {
            i: self.query_free_memory(j) for i, j in enumerate(self.gpu_list)
        }  # cuda:i -> memory of j-th GPU

    @staticmethod
    def fuck_gpu(device, memory_MB: int, block_MB: int = None):
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
        """Choose a GPU with enough free memory
        Arguments:
        - memory_MB: int, the required memory in MB
        - interval: int, the interval (in second) to check free memory
        - force: bool, whether to wait until a GPU is available
        """
        waiting = False
        while True:
            for i, free_memory in self.free_memory.items():
                if free_memory < memory_MB:
                    continue
                try:
                    device = f"cuda:{i}"
                    free_memory1 = self.query_free_memory(self.gpu_list[i])
                    fuck_cuda = self.fuck_gpu(
                        device=device, memory_MB=memory_MB, block_MB=512
                    )
                    free_memory2 = self.query_free_memory(self.gpu_list[i])
                    (_logger.note if waiting else _logger.info)(
                        f"SubProcess[{os.getpid()}]: Choose GPU{self.gpu_list[i]} ({device}) "
                        f"with {memory_MB}MB ({free_memory1}MB -> {free_memory2}MB)"
                    )
                    del fuck_cuda
                    torch.cuda.reset_peak_memory_stats(
                        device
                    )  # 不要让 fuck_cuda 影响 torch.cuda.max_memory_allocated
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
