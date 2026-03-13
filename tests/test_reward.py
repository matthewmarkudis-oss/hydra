"""Tests for the reward functions."""

from __future__ import annotations

import numpy as np
import pytest

from hydra.envs.reward import DifferentialSharpeReward, compute_episode_sharpe, compute_sortino


class TestDifferentialSharpeReward:
    def test_creation(self):
        r = DifferentialSharpeReward()
        assert r is not None

    def test_reset(self):
        r = DifferentialSharpeReward()
        r.reset(100_000.0)
        assert r.peak_value == 100_000.0

    def test_positive_return_gives_positive_reward(self):
        r = DifferentialSharpeReward(eta=0.01, drawdown_penalty=0.0, transaction_penalty=0.0)
        r.reset(100.0)
        holdings = np.array([0.0], dtype=np.float32)
        prices = np.array([100.0], dtype=np.float32)

        reward, info = r.compute(105.0, 0.0, holdings, prices)
        assert info["step_return"] > 0

    def test_drawdown_penalty(self):
        r = DifferentialSharpeReward(eta=0.01, drawdown_penalty=5.0, transaction_penalty=0.0)
        r.reset(100.0)
        holdings = np.array([0.0], dtype=np.float32)
        prices = np.array([100.0], dtype=np.float32)

        # First step: gain
        r.compute(110.0, 0.0, holdings, prices)
        # Second step: loss (drawdown from peak)
        reward, info = r.compute(95.0, 0.0, holdings, prices)
        assert info["drawdown_penalty"] < 0

    def test_transaction_penalty(self):
        r = DifferentialSharpeReward(eta=0.01, drawdown_penalty=0.0, transaction_penalty=2.0)
        r.reset(100.0)
        holdings = np.array([0.0], dtype=np.float32)
        prices = np.array([100.0], dtype=np.float32)

        reward, info = r.compute(100.0, 5.0, holdings, prices)
        assert info["transaction_penalty"] < 0


class TestEpisodeMetrics:
    def test_sharpe_positive_returns(self):
        returns = np.array([0.01, 0.02, 0.01, 0.015, 0.01], dtype=np.float32)
        sharpe = compute_episode_sharpe(returns)
        assert sharpe > 0

    def test_sharpe_zero_returns(self):
        returns = np.zeros(10, dtype=np.float32)
        sharpe = compute_episode_sharpe(returns)
        assert sharpe == 0.0

    def test_sortino_positive(self):
        returns = np.array([0.01, 0.02, -0.005, 0.015, 0.01], dtype=np.float32)
        sortino = compute_sortino(returns)
        assert sortino > 0
