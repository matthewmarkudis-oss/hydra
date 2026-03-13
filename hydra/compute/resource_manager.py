"""Resource detection and allocation for GPU and CPU compute.

Detects DirectML GPU, CPU cores, and memory. Provides allocation
decisions for the task execution engine.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger("hydra.compute.resource_manager")


@dataclass
class GPUInfo:
    available: bool = False
    device_name: str = ""
    device_type: str = ""  # "directml", "cuda", "cpu"
    memory_gb: float = 0.0


@dataclass
class CPUInfo:
    total_cores: int = 1
    available_workers: int = 1
    total_memory_gb: float = 0.0


@dataclass
class ResourceState:
    gpu: GPUInfo = field(default_factory=GPUInfo)
    cpu: CPUInfo = field(default_factory=CPUInfo)


class ResourceManager:
    """Detects and manages compute resources."""

    def __init__(self, max_cpu_workers: int = 6, gpu_memory_limit_gb: float = 12.0):
        self.max_cpu_workers = max_cpu_workers
        self.gpu_memory_limit_gb = gpu_memory_limit_gb
        self._state = ResourceState()
        self._detect_resources()

    def _detect_resources(self) -> None:
        """Detect available GPU and CPU resources."""
        self._detect_gpu()
        self._detect_cpu()

    def _detect_gpu(self) -> None:
        """Detect GPU availability (DirectML → CUDA → CPU fallback)."""
        # Try DirectML first (AMD GPU)
        try:
            import torch_directml
            device = torch_directml.device()
            self._state.gpu = GPUInfo(
                available=True,
                device_name="AMD DirectML",
                device_type="directml",
                memory_gb=self.gpu_memory_limit_gb,
            )
            logger.info(f"GPU detected: DirectML ({self.gpu_memory_limit_gb}GB limit)")
            return
        except (ImportError, TypeError, Exception):
            pass

        # Try CUDA
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                mem_gb = props.total_mem / (1024 ** 3)
                self._state.gpu = GPUInfo(
                    available=True,
                    device_name=props.name,
                    device_type="cuda",
                    memory_gb=mem_gb,
                )
                logger.info(f"GPU detected: CUDA {props.name} ({mem_gb:.1f}GB)")
                return
        except ImportError:
            pass

        self._state.gpu = GPUInfo(available=False, device_type="cpu")
        logger.info("No GPU detected, using CPU fallback")

    def _detect_cpu(self) -> None:
        """Detect CPU resources."""
        try:
            total_cores = os.cpu_count() or 1
        except Exception:
            total_cores = 1

        available_workers = min(total_cores - 2, self.max_cpu_workers)
        available_workers = max(available_workers, 1)

        try:
            import psutil
            mem = psutil.virtual_memory()
            total_memory_gb = mem.total / (1024 ** 3)
        except ImportError:
            total_memory_gb = 16.0  # Default assumption

        self._state.cpu = CPUInfo(
            total_cores=total_cores,
            available_workers=available_workers,
            total_memory_gb=total_memory_gb,
        )
        logger.info(f"CPU: {total_cores} cores, {available_workers} workers, {total_memory_gb:.1f}GB RAM")

    def get_device_string(self) -> str:
        """Get PyTorch device string for model placement."""
        if self._state.gpu.device_type == "directml":
            return "dml"
        elif self._state.gpu.device_type == "cuda":
            return "cuda"
        return "cpu"

    def get_torch_device(self):
        """Get actual torch device object."""
        device_str = self.get_device_string()
        if device_str == "dml":
            import torch_directml
            return torch_directml.device()
        else:
            import torch
            return torch.device(device_str)

    @property
    def gpu_available(self) -> bool:
        return self._state.gpu.available

    @property
    def cpu_workers(self) -> int:
        return self._state.cpu.available_workers

    @property
    def state(self) -> ResourceState:
        return self._state

    def get_summary(self) -> dict:
        return {
            "gpu": {
                "available": self._state.gpu.available,
                "type": self._state.gpu.device_type,
                "name": self._state.gpu.device_name,
                "memory_gb": self._state.gpu.memory_gb,
            },
            "cpu": {
                "cores": self._state.cpu.total_cores,
                "workers": self._state.cpu.available_workers,
                "memory_gb": self._state.cpu.total_memory_gb,
            },
        }
