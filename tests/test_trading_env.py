"""Tests for the single-agent TradingEnv."""

from __future__ import annotations

import numpy as np
import pytest

from hydra.envs.trading_env import TradingEnv


class TestTradingEnv:
    """Test suite for TradingEnv gymnasium environment."""

    def test_creation(self, trading_env):
        """Environment can be created."""
        assert trading_env is not None
        assert trading_env.observation_space is not None
        assert trading_env.action_space is not None

    def test_reset(self, trading_env):
        """Reset returns valid observation and info."""
        obs, info = trading_env.reset()
        assert obs.shape == trading_env.observation_space.shape
        assert obs.dtype == np.float32
        assert isinstance(info, dict)
        assert "initial_cash" in info

    def test_step(self, trading_env):
        """Step with a valid action returns expected tuple."""
        obs, info = trading_env.reset()
        action = trading_env.action_space.sample()
        next_obs, reward, terminated, truncated, step_info = trading_env.step(action)

        assert next_obs.shape == trading_env.observation_space.shape
        assert next_obs.dtype == np.float32
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(step_info, dict)

    def test_episode_completes(self, trading_env):
        """A full episode runs to completion."""
        obs, info = trading_env.reset()
        total_reward = 0.0
        steps = 0

        while True:
            action = trading_env.action_space.sample()
            obs, reward, terminated, truncated, info = trading_env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        assert steps > 0
        assert steps <= trading_env.episode_bars
        assert "episode_summary" in info

    def test_hold_action(self, trading_env):
        """Zero action (hold) should not change holdings significantly."""
        obs, info = trading_env.reset()
        action = np.zeros(trading_env.action_space.shape, dtype=np.float32)
        next_obs, reward, terminated, truncated, info = trading_env.step(action)

        # No trades should occur
        assert info.get("transaction_cost", 0) == 0

    def test_observation_space_valid(self, trading_env):
        """Observations are within the observation space."""
        obs, _ = trading_env.reset()
        assert trading_env.observation_space.contains(obs)

        for _ in range(5):
            action = trading_env.action_space.sample()
            obs, _, terminated, truncated, _ = trading_env.step(action)
            assert trading_env.observation_space.contains(obs)
            if terminated or truncated:
                break

    def test_synthetic_data_fallback(self, small_env):
        """Environment works with synthetic data generation."""
        obs, info = small_env.reset()
        assert obs.shape == small_env.observation_space.shape

        action = small_env.action_space.sample()
        next_obs, reward, terminated, truncated, info = small_env.step(action)
        assert next_obs.shape == small_env.observation_space.shape

    def test_multiple_resets(self, trading_env):
        """Multiple resets produce valid observations."""
        for _ in range(3):
            obs, info = trading_env.reset()
            assert obs.shape == trading_env.observation_space.shape
            assert trading_env.observation_space.contains(obs)

    def test_reward_is_finite(self, trading_env):
        """Rewards should always be finite."""
        obs, _ = trading_env.reset()

        for _ in range(10):
            action = trading_env.action_space.sample()
            obs, reward, terminated, truncated, info = trading_env.step(action)
            assert np.isfinite(reward), f"Non-finite reward: {reward}"
            if terminated or truncated:
                break
