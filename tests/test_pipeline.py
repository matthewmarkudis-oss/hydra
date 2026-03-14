"""Tests for the pipeline phases and integration."""

from __future__ import annotations

import numpy as np
import pytest

from hydra.config.schema import HydraConfig
from hydra.data.adapter import generate_synthetic_bars
from hydra.data.indicators import (
    compute_all_indicators, rsi, macd, cci, bollinger_pct_b, volume_ratio,
    bar_body_ratio, close_range_position, bar_momentum, upper_wick_ratio,
)
from hydra.envs.market_simulator import MarketSimulator
from hydra.envs.reward import DifferentialSharpeReward
from hydra.envs.session_manager import SessionManager
from hydra.envs.state_builder import StateBuilder
from hydra.evaluation.fitness import AgentFitness, compute_fitness
from hydra.risk.env_constraints import EnvConstraints
from hydra.utils.numpy_opts import (
    extract_ohlcv_arrays,
    compute_drawdown,
    max_drawdown,
    RunningStats,
    compute_returns,
)


class TestIndicators:
    def test_rsi(self, synthetic_ohlcv):
        result = rsi(synthetic_ohlcv["close"])
        assert result.dtype == np.float32
        assert len(result) == len(synthetic_ohlcv["close"])
        # RSI should be 0-100 where not NaN
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0) and np.all(valid <= 100)

    def test_macd(self, synthetic_ohlcv):
        line, signal, hist = macd(synthetic_ohlcv["close"])
        assert line.dtype == np.float32
        assert signal.dtype == np.float32
        assert hist.dtype == np.float32

    def test_cci(self, synthetic_ohlcv):
        result = cci(synthetic_ohlcv["high"], synthetic_ohlcv["low"], synthetic_ohlcv["close"])
        assert result.dtype == np.float32

    def test_bollinger_pct_b(self, synthetic_ohlcv):
        result = bollinger_pct_b(synthetic_ohlcv["close"])
        assert result.dtype == np.float32

    def test_volume_ratio(self, synthetic_ohlcv):
        result = volume_ratio(synthetic_ohlcv["volume"])
        assert result.dtype == np.float32

    def test_bar_body_ratio(self, synthetic_ohlcv):
        result = bar_body_ratio(
            synthetic_ohlcv["open"], synthetic_ohlcv["high"],
            synthetic_ohlcv["low"], synthetic_ohlcv["close"],
        )
        assert result.dtype == np.float32
        assert len(result) == len(synthetic_ohlcv["close"])
        assert np.all(np.isfinite(result))
        assert np.all(result >= -1.0) and np.all(result <= 1.0)

    def test_close_range_position(self, synthetic_ohlcv):
        result = close_range_position(
            synthetic_ohlcv["high"], synthetic_ohlcv["low"], synthetic_ohlcv["close"],
        )
        assert result.dtype == np.float32
        assert len(result) == len(synthetic_ohlcv["close"])
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_bar_momentum(self, synthetic_ohlcv):
        result = bar_momentum(
            synthetic_ohlcv["close"], synthetic_ohlcv["high"], synthetic_ohlcv["low"],
        )
        assert result.dtype == np.float32
        assert len(result) == len(synthetic_ohlcv["close"])
        # First 14 bars should be NaN (ATR warmup)
        assert np.all(np.isnan(result[:14]))
        valid = result[~np.isnan(result)]
        assert np.all(valid >= -1.0) and np.all(valid <= 1.0)

    def test_upper_wick_ratio(self, synthetic_ohlcv):
        result = upper_wick_ratio(
            synthetic_ohlcv["open"], synthetic_ohlcv["high"],
            synthetic_ohlcv["low"], synthetic_ohlcv["close"],
        )
        assert result.dtype == np.float32
        assert len(result) == len(synthetic_ohlcv["close"])
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_price_action_flat_bar(self):
        """Verify flat bar edge case returns correct neutral values."""
        n = 5
        o = np.array([100.0] * n, dtype=np.float32)
        h = np.array([100.0] * n, dtype=np.float32)
        l = np.array([100.0] * n, dtype=np.float32)
        c = np.array([100.0] * n, dtype=np.float32)

        body = bar_body_ratio(o, h, l, c)
        np.testing.assert_array_almost_equal(body, 0.0)

        crp = close_range_position(h, l, c)
        np.testing.assert_array_almost_equal(crp, 0.5)

        wick = upper_wick_ratio(o, h, l, c)
        np.testing.assert_array_almost_equal(wick, 0.0)

    def test_compute_all(self, synthetic_ohlcv):
        indicators = compute_all_indicators(synthetic_ohlcv)
        assert "rsi" in indicators
        assert "macd_line" in indicators
        assert "cci" in indicators
        assert "bb_pct_b" in indicators
        assert "volume_ratio" in indicators
        assert "atr" in indicators
        assert "bar_body_ratio" in indicators
        assert "close_range_position" in indicators
        assert "bar_momentum" in indicators
        assert "upper_wick_ratio" in indicators


class TestMarketSimulator:
    def test_creation(self):
        sim = MarketSimulator(num_stocks=3)
        assert sim.cash == 100_000.0
        assert len(sim.holdings) == 3

    def test_reset(self):
        sim = MarketSimulator(num_stocks=3)
        prices = np.array([100.0, 50.0, 200.0], dtype=np.float32)
        sim.execute_orders(np.array([0.3, 0.2, 0.1], dtype=np.float32), prices)
        sim.reset()
        assert sim.cash == 100_000.0
        np.testing.assert_array_equal(sim.holdings, np.zeros(3, dtype=np.float32))

    def test_buy(self):
        sim = MarketSimulator(num_stocks=2, initial_cash=10_000.0, transaction_cost_bps=0, slippage_bps=0, spread_bps=0)
        prices = np.array([100.0, 50.0], dtype=np.float32)
        sim.execute_orders(np.array([0.5, 0.0], dtype=np.float32), prices)
        assert sim.holdings[0] > 0
        assert sim.cash < 10_000.0

    def test_sell(self):
        sim = MarketSimulator(num_stocks=1, initial_cash=10_000.0, transaction_cost_bps=0, slippage_bps=0, spread_bps=0)
        prices = np.array([100.0], dtype=np.float32)
        # Buy first
        sim.execute_orders(np.array([0.5], dtype=np.float32), prices)
        holdings_after_buy = sim.holdings[0]
        # Sell half
        sim.execute_orders(np.array([-0.5], dtype=np.float32), prices)
        assert sim.holdings[0] < holdings_after_buy

    def test_portfolio_value(self):
        sim = MarketSimulator(num_stocks=1, initial_cash=10_000.0, transaction_cost_bps=0, slippage_bps=0, spread_bps=0)
        prices = np.array([100.0], dtype=np.float32)
        pv = sim.get_portfolio_value(prices)
        assert pv == pytest.approx(10_000.0)


class TestSessionManager:
    def test_session_labels(self):
        sm = SessionManager(bar_interval_minutes=5)
        labels = sm.compute_session_labels(78)
        assert len(labels) == 78
        assert labels.dtype == np.int8

    def test_is_trading_day(self):
        from datetime import date
        sm = SessionManager()
        # Monday
        assert sm.is_trading_day(date(2025, 1, 6)) is True
        # Saturday
        assert sm.is_trading_day(date(2025, 1, 4)) is False


class TestStateBuilder:
    def test_obs_dim(self):
        sb = StateBuilder(num_stocks=5)
        assert sb.obs_dim == 12 * 5 + 5  # 65

    def test_build(self, synthetic_features):
        sb = StateBuilder(num_stocks=1, episode_bars=78)
        features = {"TICKER": synthetic_features}
        sb.init_episode(features, ["TICKER"])
        obs = sb.build(
            step=0, cash=100_000.0, initial_cash=100_000.0,
            holdings=np.zeros(1, dtype=np.float32),
            portfolio_value=100_000.0, peak_value=100_000.0,
        )
        assert obs.shape == (sb.obs_dim,)
        assert obs.dtype == np.float32
        assert np.all(np.isfinite(obs))


class TestEnvConstraints:
    def test_clip_actions(self):
        c = EnvConstraints(max_position_pct=0.20)
        c.reset(100_000.0)
        actions = np.array([0.5, 0.5], dtype=np.float32)
        holdings = np.zeros(2, dtype=np.float32)
        prices = np.array([100.0, 100.0], dtype=np.float32)
        clipped = c.clip_actions(actions, holdings, prices, 100_000.0)
        assert np.all(clipped <= 0.20)

    def test_drawdown_halt(self):
        c = EnvConstraints(max_drawdown_pct=0.05)
        c.reset(100.0)
        # Simulate 6% drawdown
        truncate, halted, info = c.check_constraints(94.0, -0.06)
        assert truncate is True
        assert halted is True


class TestIdleCashPenalty:
    def test_idle_cash_penalty_triggers_above_threshold(self):
        """Verify holding_penalty triggers when cash_ratio > 0.8."""
        reward_fn = DifferentialSharpeReward(
            holding_penalty=0.1,
            transaction_penalty=0.0,
            drawdown_penalty=0.0,
        )
        reward_fn.reset(100_000.0)

        # Agent holds no stocks — 100% cash, cash_ratio = 1.0
        holdings = np.zeros(3, dtype=np.float32)
        prices = np.array([100.0, 50.0, 200.0], dtype=np.float32)

        _, info = reward_fn.compute(100_000.0, 0.0, holdings, prices)
        assert info["holding_penalty"] < 0, "Expected negative penalty for idle cash"

    def test_idle_cash_penalty_inactive_below_threshold(self):
        """No idle cash penalty when cash_ratio <= 0.8."""
        reward_fn = DifferentialSharpeReward(
            holding_penalty=0.1,
            transaction_penalty=0.0,
            drawdown_penalty=0.0,
        )
        initial_cash = 100_000.0
        reward_fn.reset(initial_cash)

        # Agent holds ~50% in stocks (well below 80% cash threshold)
        holdings = np.array([200.0, 400.0, 50.0], dtype=np.float32)
        prices = np.array([100.0, 50.0, 200.0], dtype=np.float32)
        # position_value = 200*100 + 400*50 + 50*200 = 20000 + 20000 + 10000 = 50000
        # cash_ratio = (100000 - 50000) / 100000 = 0.5

        _, info = reward_fn.compute(initial_cash, 0.0, holdings, prices)
        # No concentration penalty either (max_weight = 20000/100000 = 0.2 > 0.15)
        # But idle cash penalty should NOT trigger since cash_ratio = 0.5 < 0.8
        # The holding_penalty includes concentration penalty for max_weight > 0.15
        # We just verify it's not worse than the concentration component alone
        concentration_penalty = -0.1 * (0.2 - 0.15)
        assert info["holding_penalty"] == pytest.approx(concentration_penalty, abs=1e-4)


class TestZeroTradeFitness:
    def test_zero_trade_negative_fitness(self):
        """Zero-trade agents must get -0.1 fitness (below any trading agent)."""
        metrics = AgentFitness(agent_name="idle_agent", total_trades=0)
        score, breakdown = compute_fitness(metrics)
        assert score == pytest.approx(-0.1)

    def test_nonzero_trade_beats_zero_trade(self):
        """Any agent that traded should score higher than a zero-trade agent."""
        zero_metrics = AgentFitness(agent_name="idle", total_trades=0)
        trading_metrics = AgentFitness(
            agent_name="trader", total_trades=5,
            sharpe=0.1, max_drawdown=-0.02, profit_factor=1.05,
        )
        zero_score, _ = compute_fitness(zero_metrics)
        trade_score, _ = compute_fitness(trading_metrics)
        assert trade_score > zero_score


class TestNumpyOpts:
    def test_compute_returns(self):
        prices = np.array([100, 102, 101, 105], dtype=np.float32)
        ret = compute_returns(prices)
        assert ret[0] == 0.0
        assert ret[1] == pytest.approx(0.02)

    def test_drawdown(self):
        equity = np.array([100, 110, 105, 115, 100], dtype=np.float32)
        dd = compute_drawdown(equity)
        assert dd[0] == 0.0  # No drawdown at start
        assert dd[2] < 0  # Drawdown after peak

    def test_max_drawdown(self):
        equity = np.array([100, 110, 90, 95], dtype=np.float32)
        mdd = max_drawdown(equity)
        expected = (110 - 90) / 110  # ~18.18%
        assert mdd == pytest.approx(expected, rel=0.01)

    def test_running_stats(self):
        stats = RunningStats(3)
        for _ in range(100):
            stats.update(np.ones(3, dtype=np.float32))
        np.testing.assert_array_almost_equal(stats.mean, np.ones(3))


class TestConfig:
    def test_default_config(self):
        config = HydraConfig()
        assert config.env.num_stocks == 10
        assert config.training.total_timesteps == 500_000

    def test_override(self):
        config = HydraConfig(env={"num_stocks": 5})
        assert config.env.num_stocks == 5

    def test_validation(self):
        with pytest.raises(Exception):
            HydraConfig(env={"num_stocks": -1})


class TestSyntheticData:
    def test_generate(self):
        df = generate_synthetic_bars(num_bars=78, seed=42)
        assert len(df) == 78
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df["close"].dtype == np.float32

    def test_extract_ohlcv(self):
        df = generate_synthetic_bars(num_bars=10, seed=42)
        ohlcv = extract_ohlcv_arrays(df)
        assert "open" in ohlcv
        assert ohlcv["close"].dtype == np.float32
        assert len(ohlcv["close"]) == 10
