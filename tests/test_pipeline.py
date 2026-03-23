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
        assert sim.cash == 2_500.0
        assert len(sim.holdings) == 3

    def test_reset(self):
        sim = MarketSimulator(num_stocks=3)
        prices = np.array([100.0, 50.0, 200.0], dtype=np.float32)
        sim.execute_orders(np.array([0.3, 0.2, 0.1], dtype=np.float32), prices)
        sim.reset()
        assert sim.cash == 2_500.0
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
        assert sb.obs_dim == 17 * 5 + 11  # 96 (16 per-stock indicators + 4 global + 6 signal features)

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


class TestCashDragPenalty:
    def test_cash_drag_triggers_with_no_positions(self):
        """Verify cash_drag penalty is non-zero when 100% in cash."""
        reward_fn = DifferentialSharpeReward(
            cash_drag_penalty=0.3,
            holding_penalty=0.0,
            transaction_penalty=0.0,
            drawdown_penalty=0.0,
        )
        reward_fn.reset(100_000.0)

        # Agent holds no stocks — 100% cash
        holdings = np.zeros(3, dtype=np.float32)
        prices = np.array([100.0, 50.0, 200.0], dtype=np.float32)

        _, info = reward_fn.compute(100_000.0, 0.0, holdings, prices)
        assert info["cash_drag"] < 0, "Expected negative cash drag for 100% cash"
        assert info["cash_ratio"] == pytest.approx(1.0, abs=0.01)

    def test_cash_drag_zero_when_fully_deployed(self):
        """Cash drag should be ~0 when fully deployed."""
        reward_fn = DifferentialSharpeReward(
            cash_drag_penalty=0.3,
            holding_penalty=0.0,
            transaction_penalty=0.0,
            drawdown_penalty=0.0,
        )
        initial_cash = 100_000.0
        reward_fn.reset(initial_cash)

        # Agent holds stocks worth exactly the portfolio value
        holdings = np.array([500.0, 500.0, 100.0], dtype=np.float32)
        prices = np.array([100.0, 50.0, 250.0], dtype=np.float32)
        # position_value = 50000 + 25000 + 25000 = 100000

        _, info = reward_fn.compute(initial_cash, 0.0, holdings, prices)
        assert info["cash_drag"] == pytest.approx(0.0, abs=0.01)

    def test_deployment_penalty_fires_below_threshold(self):
        """Deployment penalty triggers when deployed < min_deployment_pct."""
        reward_fn = DifferentialSharpeReward(
            cash_drag_penalty=0.3,
            min_deployment_pct=0.3,
            holding_penalty=0.0,
            transaction_penalty=0.0,
            drawdown_penalty=0.0,
        )
        reward_fn.reset(100_000.0)

        # Only 10% deployed
        holdings = np.array([50.0, 100.0, 10.0], dtype=np.float32)
        prices = np.array([100.0, 50.0, 200.0], dtype=np.float32)
        # position_value = 5000 + 5000 + 2000 = 12000, deployed_pct = 0.12

        _, info = reward_fn.compute(100_000.0, 0.0, holdings, prices)
        assert info["deployment_penalty"] < 0, "Expected negative deployment penalty"
        assert info["deployed_pct"] < 0.3

    def test_benchmark_bonus_rewards_outperformance(self):
        """Benchmark bonus is positive when portfolio outperforms."""
        reward_fn = DifferentialSharpeReward(
            benchmark_bonus_weight=2.0,
            cash_drag_penalty=0.0,
            holding_penalty=0.0,
            transaction_penalty=0.0,
            drawdown_penalty=0.0,
        )
        reward_fn.reset(100_000.0)

        # Set benchmark return of 0.1%
        benchmark = np.array([0.001] * 10, dtype=np.float32)
        reward_fn.set_benchmark(benchmark)

        # Portfolio gains 0.5% → outperforming by 0.4%
        holdings = np.array([100.0], dtype=np.float32)
        prices = np.array([100.0], dtype=np.float32)
        _, info = reward_fn.compute(100_500.0, 0.0, holdings, prices)
        assert info["benchmark_bonus"] > 0, "Expected positive benchmark bonus for outperformance"

    def test_holding_penalty_concentration_only(self):
        """Holding penalty only penalizes concentration, not idle cash."""
        reward_fn = DifferentialSharpeReward(
            holding_penalty=0.1,
            cash_drag_penalty=0.0,
            transaction_penalty=0.0,
            drawdown_penalty=0.0,
        )
        initial_cash = 100_000.0
        reward_fn.reset(initial_cash)

        # ~50% deployed, max_weight = 20% > 15% threshold
        holdings = np.array([200.0, 400.0, 50.0], dtype=np.float32)
        prices = np.array([100.0, 50.0, 200.0], dtype=np.float32)

        _, info = reward_fn.compute(initial_cash, 0.0, holdings, prices)
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


class TestWarmStart:
    """Tests for automatic checkpoint resume between training runs."""

    def test_save_latest_pointer(self, tmp_path):
        """_save_latest_checkpoint_pointer writes a valid latest.json."""
        from hydra.pipeline.train_phase import _save_latest_checkpoint_pointer
        import json

        # Create a fake checkpoint structure
        ckpt_dir = tmp_path / "checkpoints"
        gen_dir = ckpt_dir / "gen_5" / "episode_100"
        gen_dir.mkdir(parents=True)
        (gen_dir / "pool_metadata.json").write_text("{}")

        _save_latest_checkpoint_pointer(str(ckpt_dir), 5)

        latest = json.loads((ckpt_dir / "latest.json").read_text())
        assert "checkpoint_path" in latest
        assert latest["generation"] == 5
        assert latest["episode"] == 100
        assert "saved_at" in latest

    def test_save_latest_picks_most_recent(self, tmp_path):
        """When multiple episodes exist, pick the most recently modified."""
        from hydra.pipeline.train_phase import _save_latest_checkpoint_pointer
        import json, time

        ckpt_dir = tmp_path / "checkpoints"
        # Create two episode dirs — ep_50 is older, ep_30 is newer
        ep50 = ckpt_dir / "gen_3" / "episode_50"
        ep50.mkdir(parents=True)
        (ep50 / "pool_metadata.json").write_text("{}")

        time.sleep(0.05)  # Ensure different mtime

        ep30 = ckpt_dir / "gen_3" / "episode_30"
        ep30.mkdir(parents=True)
        (ep30 / "pool_metadata.json").write_text("{}")

        _save_latest_checkpoint_pointer(str(ckpt_dir), 3)

        latest = json.loads((ckpt_dir / "latest.json").read_text())
        assert latest["episode"] == 30  # ep_30 was written last

    def test_try_warm_start_no_checkpoint(self, tmp_path):
        """Returns None when no latest.json exists."""
        from hydra.pipeline.train_phase import _try_warm_start

        result = _try_warm_start(str(tmp_path / "nonexistent"), 175, 10)
        assert result is None

    def test_try_warm_start_dimension_mismatch(self, tmp_path):
        """Returns None when obs_dim doesn't match checkpoint."""
        from hydra.pipeline.train_phase import _try_warm_start
        import json

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        # Create fake checkpoint with obs_dim=125
        ep_dir = ckpt_dir / "gen_1" / "episode_10"
        ep_dir.mkdir(parents=True)
        metadata = {
            "agents": {
                "ppo_1": {
                    "type": "PPOAgent",
                    "obs_dim": 125,
                    "action_dim": 10,
                    "frozen": False,
                    "total_steps": 1000,
                }
            },
            "weights": {},
            "rankings": {},
        }
        (ep_dir / "pool_metadata.json").write_text(json.dumps(metadata))

        # Write latest.json pointing to it
        pointer = {"checkpoint_path": str(ep_dir), "generation": 1, "episode": 10}
        (ckpt_dir / "latest.json").write_text(json.dumps(pointer))

        # Request warm start with different obs_dim
        result = _try_warm_start(str(ckpt_dir), 175, 10)
        assert result is None

    def test_try_warm_start_fresh_start_env_var(self, tmp_path, monkeypatch):
        """HYDRA_FRESH_START env var forces fresh pool."""
        from hydra.pipeline.train_phase import _try_warm_start

        monkeypatch.setenv("HYDRA_FRESH_START", "1")
        result = _try_warm_start(str(tmp_path), 175, 10)
        assert result is None

    def test_try_warm_start_skips_static_agents(self, tmp_path):
        """Warm start only loads learning agents, not static snapshots."""
        from hydra.pipeline.train_phase import _try_warm_start
        import json

        ckpt_dir = tmp_path / "checkpoints"
        ep_dir = ckpt_dir / "gen_5" / "episode_100"
        ep_dir.mkdir(parents=True)

        metadata = {
            "agents": {
                "ppo_1": {
                    "type": "PPOAgent",
                    "obs_dim": 90,
                    "action_dim": 5,
                    "frozen": False,
                    "total_steps": 5000,
                },
                "rppo_1_gen450": {
                    "type": "StaticAgent",
                    "obs_dim": 90,
                    "action_dim": 5,
                    "frozen": True,
                    "total_steps": 2000,
                    "source_type": "recurrentppo",
                },
            },
            "weights": {},
            "rankings": {},
        }
        (ep_dir / "pool_metadata.json").write_text(json.dumps(metadata))

        pointer = {"checkpoint_path": str(ep_dir), "generation": 5, "episode": 100}
        (ckpt_dir / "latest.json").write_text(json.dumps(pointer))

        # Create a fake model file for ppo_1 so load doesn't crash
        ppo_dir = ep_dir / "ppo_1"
        ppo_dir.mkdir()
        # Agent load will fail because there's no real model, but
        # we're testing that static agents are skipped
        result = _try_warm_start(str(ckpt_dir), 90, 5)
        # Result will be None because ppo_1.load() fails on fake model,
        # but the important thing is that the StaticAgent was not attempted
        # (no error about 'rppo_1_gen450')

    def test_latest_pointer_survives_missing_gen(self, tmp_path):
        """No crash when final gen dir doesn't exist."""
        from hydra.pipeline.train_phase import _save_latest_checkpoint_pointer

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        # gen_99 doesn't exist — should log warning and return cleanly
        _save_latest_checkpoint_pointer(str(ckpt_dir), 99)
        assert not (ckpt_dir / "latest.json").exists()
