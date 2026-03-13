"""Tests for the multi-agent environment."""

from __future__ import annotations

import numpy as np
import pytest

from hydra.agents.agent_pool import AgentPool
from hydra.agents.rule_based_agent import RuleBasedAgent
from hydra.envs.multi_agent_env import MultiAgentEnv
from hydra.envs.trading_env import TradingEnv


@pytest.fixture
def multi_agent_setup(trading_env):
    """Create a MultiAgentEnv with a small pool."""
    obs_dim = trading_env.observation_space.shape[0]
    action_dim = trading_env.action_space.shape[0]

    pool = AgentPool()
    pool.add(RuleBasedAgent("agent_1", obs_dim, action_dim))
    pool.add(RuleBasedAgent("agent_2", obs_dim, action_dim))

    return MultiAgentEnv(trading_env, pool)


class TestMultiAgentEnv:
    def test_reset(self, multi_agent_setup):
        obs, info = multi_agent_setup.reset()
        assert obs.shape == multi_agent_setup.observation_space.shape
        assert info["num_agents"] == 2

    def test_step(self, multi_agent_setup):
        obs, info = multi_agent_setup.reset()
        next_obs, reward, terminated, truncated, step_info = multi_agent_setup.step()
        assert next_obs.shape == multi_agent_setup.observation_space.shape
        assert isinstance(reward, float)
        assert "per_agent_actions" in step_info
        assert len(step_info["per_agent_actions"]) == 2

    def test_full_episode(self, multi_agent_setup):
        reward, summary = multi_agent_setup.run_episode()
        assert isinstance(reward, float)
        assert summary["steps"] > 0

    def test_action_aggregation(self, multi_agent_setup):
        obs, _ = multi_agent_setup.reset()
        _, _, _, _, info = multi_agent_setup.step()
        agg = info["aggregated_action"]
        assert agg.shape == multi_agent_setup.action_space.shape
        assert np.all(np.abs(agg) <= 1.0)
