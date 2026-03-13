"""Tests for compute orchestration."""

from __future__ import annotations

import numpy as np
import pytest

from hydra.compute.executor import TaskExecutor
from hydra.compute.resource_manager import ResourceManager
from hydra.compute.workflow import Workflow


class TestResourceManager:
    def test_creation(self):
        rm = ResourceManager(max_cpu_workers=2)
        assert rm.cpu_workers >= 1

    def test_device_string(self):
        rm = ResourceManager(max_cpu_workers=2)
        device = rm.get_device_string()
        assert device in ("cpu", "cuda", "dml")

    def test_summary(self):
        rm = ResourceManager(max_cpu_workers=2)
        summary = rm.get_summary()
        assert "gpu" in summary
        assert "cpu" in summary


class TestTaskExecutor:
    def test_submit(self):
        executor = TaskExecutor()
        tid = executor.submit(lambda: 42)
        result = executor.get_result(tid)
        assert result.status == "success"
        assert result.result == 42

    def test_submit_failure(self):
        executor = TaskExecutor()

        def failing_func():
            raise ValueError("test error")

        tid = executor.submit(failing_func)
        result = executor.get_result(tid)
        assert result.status == "failed"
        assert "test error" in result.error


class TestWorkflow:
    def test_simple_chain(self):
        wf = Workflow("test")
        wf.add_task("a", lambda deps: 1)
        wf.add_task("b", lambda deps: deps["a"] + 1, dependencies=["a"])
        wf.add_task("c", lambda deps: deps["b"] + 1, dependencies=["b"])

        results = wf.execute()
        assert results["a"] == 1
        assert results["b"] == 2
        assert results["c"] == 3

    def test_parallel_tasks(self):
        wf = Workflow("test")
        wf.add_task("a", lambda deps: 10)
        wf.add_task("b", lambda deps: 20)
        wf.add_task("c", lambda deps: deps["a"] + deps["b"], dependencies=["a", "b"])

        results = wf.execute()
        assert results["c"] == 30

    def test_cycle_detection(self):
        wf = Workflow("test")
        wf.add_task("a", lambda deps: 1, dependencies=["b"])
        wf.add_task("b", lambda deps: 2, dependencies=["a"])

        with pytest.raises(ValueError, match="cycle"):
            wf.execute()

    def test_unknown_dependency(self):
        wf = Workflow("test")
        wf.add_task("a", lambda deps: 1, dependencies=["nonexistent"])

        with pytest.raises(ValueError, match="unknown task"):
            wf.execute()

    def test_workflow_summary(self):
        wf = Workflow("test")
        wf.add_task("a", lambda deps: 1)
        wf.execute()
        summary = wf.get_summary()
        assert summary["name"] == "test"
        assert "a" in summary["tasks"]
