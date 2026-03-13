"""Tests for agent implementations."""

from __future__ import annotations

import numpy as np
import pytest

from hydra.agents.agent_pool import AgentPool
from hydra.agents.base_rl_agent import BaseRLAgent
from hydra.agents.rule_based_agent import RuleBasedAgent
from hydra.agents.static_agent import StaticAgent


OBS_DIM = 19  # 7*2 + 5 for 2 stocks
ACTION_DIM = 2


class TestRuleBasedAgent:
    def test_creation(self):
        agent = RuleBasedAgent("test_rule", OBS_DIM, ACTION_DIM)
        assert agent.name == "test_rule"
        assert agent.is_frozen

    def test_select_action(self):
        agent = RuleBasedAgent("test_rule", OBS_DIM, ACTION_DIM)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action = agent.select_action(obs)
        assert action.shape == (ACTION_DIM,)
        assert action.dtype == np.float32
        assert np.all(np.abs(action) <= 1.0)

    def test_update_returns_skipped(self):
        agent = RuleBasedAgent("test_rule", OBS_DIM, ACTION_DIM)
        result = agent.update()
        assert result.get("skipped") == 1.0

    def test_rsi_based_heuristic(self):
        agent = RuleBasedAgent("test_rule", OBS_DIM, ACTION_DIM)
        # Create obs with low RSI (oversold) → should produce buy actions
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        rsi_start = 2 * ACTION_DIM + 1
        obs[rsi_start:rsi_start + ACTION_DIM] = 0.25  # RSI=25, oversold
        action = agent.select_action(obs)
        assert np.any(action > 0), "Oversold RSI should trigger buy"


class TestStaticAgent:
    def test_creation(self):
        agent = StaticAgent("test_static", OBS_DIM, ACTION_DIM)
        assert agent.is_frozen

    def test_select_action_no_model(self):
        agent = StaticAgent("test_static", OBS_DIM, ACTION_DIM)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action = agent.select_action(obs)
        assert action.shape == (ACTION_DIM,)
        # No model loaded → returns zeros (hold)
        np.testing.assert_array_equal(action, np.zeros(ACTION_DIM, dtype=np.float32))

    def test_update_raises(self):
        agent = StaticAgent("test_static", OBS_DIM, ACTION_DIM)
        with pytest.raises(RuntimeError, match="Cannot update static agent"):
            agent.update()

    def test_unfreeze_raises(self):
        agent = StaticAgent("test_static", OBS_DIM, ACTION_DIM)
        with pytest.raises(RuntimeError, match="Cannot unfreeze"):
            agent.unfreeze()


class TestAgentPool:
    def test_add_remove(self):
        pool = AgentPool()
        agent = RuleBasedAgent("test", OBS_DIM, ACTION_DIM)
        pool.add(agent)
        assert pool.size == 1
        pool.remove("test")
        assert pool.size == 0

    def test_collect_actions(self):
        pool = AgentPool()
        pool.add(RuleBasedAgent("a1", OBS_DIM, ACTION_DIM))
        pool.add(RuleBasedAgent("a2", OBS_DIM, ACTION_DIM))

        obs = np.random.randn(OBS_DIM).astype(np.float32)
        actions = pool.collect_actions(obs)
        assert len(actions) == 2
        assert "a1" in actions
        assert "a2" in actions

    def test_aggregate_actions(self):
        pool = AgentPool()
        pool.add(RuleBasedAgent("a1", OBS_DIM, ACTION_DIM))
        pool.add(RuleBasedAgent("a2", OBS_DIM, ACTION_DIM))

        obs = np.random.randn(OBS_DIM).astype(np.float32)
        agg = pool.aggregate_actions(obs)
        assert agg.shape == (ACTION_DIM,)
        assert np.all(np.abs(agg) <= 1.0)

    def test_get_weights(self):
        pool = AgentPool()
        pool.add(RuleBasedAgent("a1", OBS_DIM, ACTION_DIM), weight=2.0)
        pool.add(RuleBasedAgent("a2", OBS_DIM, ACTION_DIM), weight=1.0)

        weights = pool.get_weights()
        assert len(weights) == 2
        np.testing.assert_almost_equal(np.sum(weights), 1.0)

    def test_rankings(self):
        pool = AgentPool()
        pool.add(RuleBasedAgent("a1", OBS_DIM, ACTION_DIM))
        pool.add(RuleBasedAgent("a2", OBS_DIM, ACTION_DIM))

        pool.update_rankings({"a1": 0.5, "a2": 0.8})
        ranked = pool.get_ranked_agents()
        assert ranked[0][0] == "a2"  # Higher score first
