"""@gpu_task and @cpu_task decorators for compute routing.

Routes functions to appropriate compute resources:
- @gpu_task: DirectML GPU (AMD 6900XT) with CPU fallback
- @cpu_task: ProcessPoolExecutor with configurable workers
"""

from __future__ import annotations

import functools
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable

logger = logging.getLogger("hydra.compute.decorators")

# Module-level resource state (initialized lazily)
_resource_manager = None


def _get_resource_manager():
    global _resource_manager
    if _resource_manager is None:
        from hydra.compute.resource_manager import ResourceManager
        _resource_manager = ResourceManager()
    return _resource_manager


def gpu_task(memory_gb: float = 4.0, fallback_to_cpu: bool = True):
    """Decorator to route a function to GPU (DirectML) or CPU fallback.

    Args:
        memory_gb: Estimated GPU memory requirement.
        fallback_to_cpu: If True, fall back to CPU when GPU unavailable.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            rm = _get_resource_manager()

            if rm.gpu_available:
                if rm.state.gpu.device_type == "directml":
                    from hydra.compute.dml_compat import patch_tensor_for_directml
                    patch_tensor_for_directml()
                try:
                    logger.debug(f"Running {func.__name__} on GPU ({rm.state.gpu.device_type})")
                    return func(*args, **kwargs)
                except Exception as e:
                    if fallback_to_cpu:
                        logger.warning(
                            f"GPU execution failed for {func.__name__}: {e}. "
                            "Falling back to CPU."
                        )
                        return func(*args, **kwargs)
                    raise
            elif fallback_to_cpu:
                logger.debug(f"No GPU available, running {func.__name__} on CPU")
                return func(*args, **kwargs)
            else:
                raise RuntimeError(
                    f"GPU required for {func.__name__} but not available"
                )

        wrapper._gpu_task = True
        wrapper._memory_gb = memory_gb
        return wrapper

    return decorator


def cpu_task(workers: int | None = None, use_threads: bool = False):
    """Decorator to route a function to a CPU worker pool.

    Args:
        workers: Number of workers. Defaults to ResourceManager's cpu_workers.
        use_threads: Use ThreadPoolExecutor instead of ProcessPoolExecutor.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            rm = _get_resource_manager()
            num_workers = workers or rm.cpu_workers

            logger.debug(f"Running {func.__name__} on CPU ({num_workers} workers)")
            return func(*args, **kwargs)

        wrapper._cpu_task = True
        wrapper._workers = workers
        wrapper._use_threads = use_threads

        # Add a parallel_map helper
        def parallel_map(items, map_func, max_workers=None):
            """Execute map_func on each item in parallel."""
            rm = _get_resource_manager()
            num_workers = max_workers or workers or rm.cpu_workers

            PoolClass = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
            with PoolClass(max_workers=num_workers) as executor:
                results = list(executor.map(map_func, items))
            return results

        wrapper.parallel_map = parallel_map
        return wrapper

    return decorator
