"""Tests for checkpoint save/load and CPU-first loading."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hydra.agents.agent_pool import AgentPool
from hydra.agents.ppo_agent import PPOAgent
from hydra.agents.td3_agent import TD3Agent
from hydra.agents.recurrent_ppo_agent import RecurrentPPOAgent

OBS_DIM = 19
ACTION_DIM = 2


class TestPPOCheckpointRoundTrip:
    """Verify PPO save/load preserves model weights."""

    def test_save_load_preserves_weights(self, tmp_path):
        agent = PPOAgent("ppo_test", OBS_DIM, ACTION_DIM, prefer_gpu=False)
        agent._ensure_model()

        # Get initial weights
        initial_params = {
            k: v.detach().cpu().clone()
            for k, v in agent._model.policy.state_dict().items()
        }

        # Save
        save_path = tmp_path / "ppo_model"
        agent.save(save_path)

        # Load into a new agent
        agent2 = PPOAgent("ppo_test2", OBS_DIM, ACTION_DIM, prefer_gpu=False)
        agent2.load(save_path)

        # Verify weights match
        loaded_params = agent2._model.policy.state_dict()
        for key in initial_params:
            np.testing.assert_array_equal(
                initial_params[key].numpy(),
                loaded_params[key].cpu().numpy(),
                err_msg=f"Weight mismatch for {key}",
            )

    def test_load_uses_cpu_device(self, tmp_path):
        agent = PPOAgent("ppo_test", OBS_DIM, ACTION_DIM, prefer_gpu=False)
        agent._ensure_model()

        save_path = tmp_path / "ppo_model"
        agent.save(save_path)

        # Load and verify device is CPU
        agent2 = PPOAgent("ppo_test2", OBS_DIM, ACTION_DIM, prefer_gpu=True)
        agent2.load(save_path)

        import torch
        assert agent2._model.device == torch.device("cpu")


class TestTD3CheckpointRoundTrip:
    """Verify TD3 save/load preserves model weights."""

    def test_save_load_preserves_weights(self, tmp_path):
        agent = TD3Agent("td3_test", OBS_DIM, ACTION_DIM, prefer_gpu=False)
        agent._ensure_model()

        initial_params = {
            k: v.detach().cpu().clone()
            for k, v in agent._model.policy.state_dict().items()
        }

        save_path = tmp_path / "td3_model"
        agent.save(save_path)

        agent2 = TD3Agent("td3_test2", OBS_DIM, ACTION_DIM, prefer_gpu=False)
        agent2.load(save_path)

        loaded_params = agent2._model.policy.state_dict()
        for key in initial_params:
            np.testing.assert_array_equal(
                initial_params[key].numpy(),
                loaded_params[key].cpu().numpy(),
                err_msg=f"Weight mismatch for {key}",
            )


class TestRecurrentPPOCheckpointRoundTrip:
    """Verify RecurrentPPO save/load preserves model weights."""

    def test_save_load_preserves_weights(self, tmp_path):
        agent = RecurrentPPOAgent("rppo_test", OBS_DIM, ACTION_DIM, prefer_gpu=False)
        agent._ensure_model()

        initial_params = {
            k: v.detach().cpu().clone()
            for k, v in agent._model.policy.state_dict().items()
        }

        save_path = tmp_path / "rppo_model"
        agent.save(save_path)

        agent2 = RecurrentPPOAgent("rppo_test2", OBS_DIM, ACTION_DIM, prefer_gpu=False)
        agent2.load(save_path)

        loaded_params = agent2._model.policy.state_dict()
        for key in initial_params:
            np.testing.assert_array_equal(
                initial_params[key].numpy(),
                loaded_params[key].cpu().numpy(),
                err_msg=f"Weight mismatch for {key}",
            )


class TestAgentPoolLoadErrorHandling:
    """Verify agent_pool.load() reports failures properly."""

    def test_pool_save_load_round_trip(self, tmp_path):
        pool = AgentPool()
        pool.add(PPOAgent("ppo_1", OBS_DIM, ACTION_DIM, prefer_gpu=False))
        pool.save(tmp_path / "pool")

        pool2 = AgentPool()
        pool2.load(tmp_path / "pool")
        assert pool2.size == 1
        assert pool2.get("ppo_1") is not None

    def test_pool_load_logs_error_on_failure(self, tmp_path, caplog):
        """Verify pool.load() logs ERROR (not just warning) when agent load fails."""
        pool = AgentPool()
        pool.add(PPOAgent("ppo_1", OBS_DIM, ACTION_DIM, prefer_gpu=False))
        pool.save(tmp_path / "pool")

        # Corrupt the model file to force a load failure
        model_dir = tmp_path / "pool" / "ppo_1"
        for f in model_dir.iterdir():
            if f.suffix == ".zip":
                f.write_bytes(b"corrupted")

        pool2 = AgentPool()
        with caplog.at_level(logging.ERROR, logger="hydra.agents.pool"):
            pool2.load(tmp_path / "pool")

        assert pool2.size == 0
        assert any("CHECKPOINT LOAD FAILED" in r.message for r in caplog.records)


class TestTrainPhaseWeightLossLogging:
    """Verify train_phase logs clearly when adding fresh agents."""

    def test_fresh_agent_fallback_message(self):
        """The error message should mention 'WEIGHT LOSS' when fresh agents are added."""
        # This is a static check — verify the string exists in the source
        from hydra.pipeline import train_phase
        import inspect

        source = inspect.getsource(train_phase.run_training)
        assert "WEIGHT LOSS" in source
