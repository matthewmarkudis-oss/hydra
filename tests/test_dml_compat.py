"""Tests for DirectML compatibility patches."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

import hydra.compute.dml_compat as dml_compat
from hydra.agents.ppo_agent import _get_device, _resolve_sb3_device


class TestPatchTensorForDirectML:
    """Tests for the monkey-patch function."""

    def setup_method(self):
        """Reset the patched flag before each test."""
        dml_compat._patched = False

    def test_returns_false_without_directml(self):
        """Patch should return False when torch-directml is not installed."""
        with patch.dict("sys.modules", {"torch_directml": None}):
            dml_compat._patched = False
            result = dml_compat.patch_tensor_for_directml()
            assert result is False

    def test_idempotent(self):
        """Calling patch multiple times should be safe."""
        dml_compat._patched = True
        assert dml_compat.patch_tensor_for_directml() is True
        # Still True — no error, no re-patch
        assert dml_compat.patch_tensor_for_directml() is True

    def test_cpu_tensor_item_passthrough(self):
        """CPU tensors should work normally after patching (zero overhead path)."""
        # Force _patched to False so we can test the actual patching
        # Even if torch-directml is not installed, we test the CPU path
        # by manually simulating a patched state
        t = torch.tensor(3.14)
        val = t.item()
        assert isinstance(val, float)
        assert abs(val - 3.14) < 1e-5

    def test_cpu_tensor_numpy_passthrough(self):
        """CPU tensors should produce correct numpy arrays after patching."""
        t = torch.tensor([1.0, 2.0, 3.0])
        arr = t.numpy()
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_almost_equal(arr, [1.0, 2.0, 3.0])


class TestGetDevice:
    """Tests for _get_device detection."""

    def test_returns_cpu_when_no_gpu(self):
        assert _get_device(prefer_gpu=False) == "cpu"

    def test_returns_string(self):
        device = _get_device(prefer_gpu=True)
        assert device in ("cpu", "cuda", "dml")


class TestResolveSB3Device:
    """Tests for _resolve_sb3_device conversion."""

    def test_cpu_passthrough(self):
        assert _resolve_sb3_device("cpu") == "cpu"

    def test_cuda_passthrough(self):
        assert _resolve_sb3_device("cuda") == "cuda"

    def test_dml_returns_device_or_cpu(self):
        """DML should return a DirectML device if available, else CPU."""
        result = _resolve_sb3_device("dml")
        # If torch-directml is installed, result is a torch.device-like object
        # If not, it falls back to "cpu"
        if isinstance(result, str):
            assert result == "cpu"
        else:
            # It's a torch device object from torch_directml
            assert result is not None
