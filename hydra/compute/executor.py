"""Task execution engine for the compute pipeline."""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("hydra.compute.executor")


@dataclass
class TaskResult:
    task_id: str
    status: str  # "success", "failed", "cancelled"
    result: Any = None
    error: str = ""
    duration_ms: float = 0.0


class TaskExecutor:
    """Executes compute tasks with resource-aware scheduling."""

    def __init__(self, max_cpu_workers: int = 6, use_threads: bool = False):
        self.max_cpu_workers = max_cpu_workers
        self._use_threads = use_threads
        self._task_counter = 0
        self._results: dict[str, TaskResult] = {}

    def submit(self, func: Callable, *args: Any, task_id: str | None = None, **kwargs: Any) -> str:
        """Submit a task for execution.

        Returns task_id for tracking.
        """
        self._task_counter += 1
        tid = task_id or f"task_{self._task_counter}"

        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            duration = (time.perf_counter() - start) * 1000

            self._results[tid] = TaskResult(
                task_id=tid,
                status="success",
                result=result,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            self._results[tid] = TaskResult(
                task_id=tid,
                status="failed",
                error=str(e),
                duration_ms=duration,
            )
            logger.error(f"Task {tid} failed: {e}")

        return tid

    def submit_batch(
        self,
        func: Callable,
        args_list: list[tuple],
        max_workers: int | None = None,
    ) -> list[str]:
        """Submit a batch of tasks for parallel execution."""
        workers = max_workers or self.max_cpu_workers
        task_ids = []

        PoolClass = ThreadPoolExecutor if self._use_threads else ProcessPoolExecutor

        with PoolClass(max_workers=workers) as pool:
            futures: list[tuple[str, Future]] = []

            for args in args_list:
                self._task_counter += 1
                tid = f"task_{self._task_counter}"
                task_ids.append(tid)
                future = pool.submit(func, *args)
                futures.append((tid, future))

            for tid, future in futures:
                start = time.perf_counter()
                try:
                    result = future.result()
                    duration = (time.perf_counter() - start) * 1000
                    self._results[tid] = TaskResult(
                        task_id=tid, status="success", result=result, duration_ms=duration
                    )
                except Exception as e:
                    duration = (time.perf_counter() - start) * 1000
                    self._results[tid] = TaskResult(
                        task_id=tid, status="failed", error=str(e), duration_ms=duration
                    )

        return task_ids

    def get_result(self, task_id: str) -> TaskResult | None:
        """Get the result of a completed task."""
        return self._results.get(task_id)

    def get_all_results(self) -> dict[str, TaskResult]:
        """Get all task results."""
        return dict(self._results)

    def clear(self) -> None:
        """Clear all stored results."""
        self._results.clear()
        self._task_counter = 0
