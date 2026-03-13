"""DAG-based pipeline definition and execution.

Defines pipelines as dependency graphs with topological sort execution.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("hydra.compute.workflow")


@dataclass
class WorkflowNode:
    """A single node (task) in the workflow DAG."""
    name: str
    func: Callable
    kwargs: dict = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    result: Any = None
    status: str = "pending"  # pending, running, completed, failed
    duration_ms: float = 0.0


class Workflow:
    """DAG-based workflow engine.

    Define a pipeline as a set of named tasks with dependencies,
    then execute in topological order.
    """

    def __init__(self, name: str = "workflow"):
        self.name = name
        self._nodes: dict[str, WorkflowNode] = {}

    def add_task(
        self,
        name: str,
        func: Callable,
        dependencies: list[str] | None = None,
        **kwargs: Any,
    ) -> Workflow:
        """Add a task to the workflow.

        Args:
            name: Unique task name.
            func: Callable to execute. Receives results of dependencies as first arg.
            dependencies: List of task names that must complete first.
            **kwargs: Additional arguments passed to func.

        Returns:
            Self for chaining.
        """
        self._nodes[name] = WorkflowNode(
            name=name,
            func=func,
            kwargs=kwargs,
            dependencies=dependencies or [],
        )
        return self

    def execute(self) -> dict[str, Any]:
        """Execute the workflow in topological order.

        Returns dict of task_name → result.
        """
        order = self._topological_sort()
        results: dict[str, Any] = {}

        logger.info(f"Executing workflow '{self.name}': {len(order)} tasks")

        for task_name in order:
            node = self._nodes[task_name]
            node.status = "running"

            # Gather dependency results
            dep_results = {dep: results[dep] for dep in node.dependencies if dep in results}

            start = time.perf_counter()
            try:
                logger.info(f"  Running task '{task_name}'...")
                result = node.func(dep_results, **node.kwargs)
                node.result = result
                node.status = "completed"
                node.duration_ms = (time.perf_counter() - start) * 1000
                results[task_name] = result
                logger.info(f"  Task '{task_name}' completed ({node.duration_ms:.0f}ms)")
            except Exception as e:
                node.status = "failed"
                node.duration_ms = (time.perf_counter() - start) * 1000
                logger.error(f"  Task '{task_name}' failed: {e}")
                raise RuntimeError(f"Workflow task '{task_name}' failed: {e}") from e

        return results

    def _topological_sort(self) -> list[str]:
        """Topological sort of the DAG."""
        in_degree: dict[str, int] = defaultdict(int)
        adj: dict[str, list[str]] = defaultdict(list)

        for name, node in self._nodes.items():
            in_degree.setdefault(name, 0)
            for dep in node.dependencies:
                if dep not in self._nodes:
                    raise ValueError(f"Task '{name}' depends on unknown task '{dep}'")
                adj[dep].append(name)
                in_degree[name] += 1

        queue = deque([n for n, d in in_degree.items() if d == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self._nodes):
            raise ValueError("Workflow contains a cycle")

        return order

    def get_status(self) -> dict[str, str]:
        """Get the status of all tasks."""
        return {name: node.status for name, node in self._nodes.items()}

    def get_summary(self) -> dict[str, Any]:
        """Get workflow execution summary."""
        return {
            "name": self.name,
            "tasks": {
                name: {
                    "status": node.status,
                    "duration_ms": node.duration_ms,
                    "dependencies": node.dependencies,
                }
                for name, node in self._nodes.items()
            },
            "total_duration_ms": sum(n.duration_ms for n in self._nodes.values()),
        }
