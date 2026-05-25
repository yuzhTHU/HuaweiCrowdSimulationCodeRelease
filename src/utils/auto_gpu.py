import os
import time
import torch
import logging
import subprocess

__all__ = ["AutoGPU"]

_logger = logging.getLogger(__name__)


class AutoGPU:
    """
    Automatic GPU memory manager used to select a GPU with sufficient free memory.
    """
    def __init__(self):
        """
        Initialize AutoGPU and get the currently visible CUDA device list.
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
        [Internal method] Allocate placeholder memory on the target device.

        This is used to verify that memory is truly available by actually
        allocating it, or to proactively reserve GPU memory.

        Args:
            device (str or torch.device): Target device.
            memory_MB (int): Amount of memory to allocate in MB.
            block_MB (int, optional): Block size. If None, allocate in one shot.

        Returns:
            torch.Tensor or List[torch.Tensor]: References to the allocated tensors.
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
        Select a GPU with enough free memory.

        This method not only queries `nvidia-smi`, but also tries to allocate
        memory to verify actual availability. If all GPUs are busy and
        `force=True`, it blocks and waits.

        Args:
            memory_MB (int): Minimum memory required by the task in MB.
            interval (int, optional): Polling interval in seconds. Default is 600.
            force (bool, optional): Whether to wait until a GPU becomes available.
                If False and no GPU is available, returns "cpu". Default is True.

        Returns:
            str: Selected device string, such as "cuda:0" or "cpu".
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
                    )  # Keep allocation from affecting torch.cuda.max_memory_allocated
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
