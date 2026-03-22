"""Tests for forward-test hardening: LiveStateBuilder, emergency halt,
slippage tracking, alerts, and config defaults."""

import json
import math
import statistics
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ── LiveStateBuilder ─────────────────────────────────────────────────────────


class TestLiveStateBuilder:
    def test_obs_dim(self):
        from hydra.forward_test.live_state_builder import LiveStateBuilder

        tickers = ["AAPL", "NVDA", "TSLA"]
        builder = LiveStateBuilder(3, tickers)
        assert builder.obs_dim == 17 * 3 + 5

    def test_not_ready_without_warmup(self):
        from hydra.forward_test.live_state_builder import LiveStateBuilder

        builder = LiveStateBuilder(2, ["AAPL", "NVDA"])
        assert not builder.is_ready
        assert builder.bars_collected == 0

    def test_ready_after_warmup(self):
        from hydra.forward_test.live_state_builder import LiveStateBuilder

        builder = LiveStateBuilder(1, ["AAPL"])
        for i in range(60):
            builder.update_bar("AAPL", {
                "open": 100 + i * 0.1,
                "high": 101 + i * 0.1,
                "low": 99 + i * 0.1,
                "close": 100.5 + i * 0.1,
                "volume": 1000000,
            })
        assert builder.is_ready
        assert builder.bars_collected == 60

    def test_build_shape(self):
        from hydra.forward_test.live_state_builder import LiveStateBuilder

        n = 2
        tickers = ["AAPL", "NVDA"]
        builder = LiveStateBuilder(n, tickers)

        # Feed enough bars
        for i in range(60):
            for t in tickers:
                builder.update_bar(t, {
                    "open": 100 + i * 0.1,
                    "high": 101 + i * 0.1,
                    "low": 99 + i * 0.1,
                    "close": 100.5 + i * 0.1,
                    "volume": 1000000,
                })

        obs = builder.build(
            cash=5000.0,
            initial_cash=10000.0,
            holdings={"AAPL": 10, "NVDA": 5},
            portfolio_value=10500.0,
            peak_value=10600.0,
        )

        assert obs.shape == (17 * n + 5,)
        assert obs.dtype == np.float32

    def test_build_populates_indicators(self):
        from hydra.forward_test.live_state_builder import LiveStateBuilder

        n = 1
        builder = LiveStateBuilder(n, ["AAPL"])

        # Feed bars with distinct price pattern
        for i in range(60):
            price = 100 + i * 0.5
            builder.update_bar("AAPL", {
                "open": price - 0.2,
                "high": price + 0.5,
                "low": price - 0.5,
                "close": price,
                "volume": 500000 + i * 1000,
            })

        obs = builder.build(
            cash=5000.0,
            initial_cash=10000.0,
            holdings={"AAPL": 10},
            portfolio_value=10500.0,
            peak_value=10600.0,
        )

        # Cash ratio should be 0.5
        assert abs(obs[0] - 0.5) < 0.01

        # Price feature (index n+1 = 2) should be > 1.0 (uptrend)
        price_feat = obs[n + 1]
        assert price_feat > 1.0, f"Price feature should be > 1.0 for uptrend, got {price_feat}"

        # RSI (index n+1+n = 3) should be > 0 in [0, 1]
        rsi_feat = obs[n + 1 + n]
        assert 0 <= rsi_feat <= 1.0, f"RSI feature out of range: {rsi_feat}"

        # The obs should not be all zeros (indicators should be populated)
        nonzero = np.count_nonzero(obs)
        assert nonzero > 10, f"Expected many non-zero features, got {nonzero}"

    def test_build_before_warmup_returns_zeros(self):
        from hydra.forward_test.live_state_builder import LiveStateBuilder

        builder = LiveStateBuilder(1, ["AAPL"])
        # Only add 3 bars (not enough for warmup)
        for i in range(3):
            builder.update_bar("AAPL", {
                "open": 100, "high": 101, "low": 99,
                "close": 100, "volume": 1000000,
            })

        assert not builder.is_ready

        # Can still build — but most features will be zero/default
        obs = builder.build(
            cash=10000.0,
            initial_cash=10000.0,
            holdings={},
            portfolio_value=10000.0,
            peak_value=10000.0,
        )
        assert obs.shape == (17 * 1 + 5,)

    def test_ignores_unknown_ticker(self):
        from hydra.forward_test.live_state_builder import LiveStateBuilder

        builder = LiveStateBuilder(1, ["AAPL"])
        builder.update_bar("UNKNOWN", {"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1})
        assert builder.bars_collected == 0


# ── Emergency Halt ───────────────────────────────────────────────────────────


class TestEmergencyHalt:
    def test_emergency_halt_cancels_orders(self):
        from hydra.forward_test.runner import ForwardTestRunner
        from hydra.forward_test.tracker import ForwardTestTracker

        broker = MagicMock()
        tracker = ForwardTestTracker(
            log_path="logs/test_ft_halt.jsonl",
            state_path="logs/test_ft_halt_state.json",
        )
        runner = ForwardTestRunner(
            agents=[],
            broker=broker,
            data_provider=MagicMock(),
            tickers=["AAPL"],
            config={},
            tracker=tracker,
        )

        runner._emergency_halt("test reason")

        broker.cancel_all_orders.assert_called_once()
        assert not runner._running
        assert runner._halted

    def test_emergency_halt_only_fires_once(self):
        from hydra.forward_test.runner import ForwardTestRunner
        from hydra.forward_test.tracker import ForwardTestTracker

        broker = MagicMock()
        tracker = ForwardTestTracker(
            log_path="logs/test_ft_halt2.jsonl",
            state_path="logs/test_ft_halt2_state.json",
        )
        runner = ForwardTestRunner(
            agents=[],
            broker=broker,
            data_provider=MagicMock(),
            tickers=["AAPL"],
            config={},
            tracker=tracker,
        )

        runner._emergency_halt("first")
        runner._emergency_halt("second")  # Should be a no-op

        # cancel_all_orders should only be called once
        assert broker.cancel_all_orders.call_count == 1

    def test_agent_error_triggers_halt(self):
        from hydra.forward_test.runner import ForwardTestRunner
        from hydra.forward_test.tracker import ForwardTestTracker

        bad_agent = MagicMock()
        bad_agent.name = "bad_agent"
        bad_agent.select_action.side_effect = RuntimeError("model crashed")

        broker = MagicMock()
        broker.is_market_open.return_value = True
        broker.get_latest_price.return_value = 100.0
        broker.get_account.return_value = {"portfolio_value": 10000, "cash": 5000}
        broker.get_open_positions.return_value = []

        tracker = ForwardTestTracker(
            log_path="logs/test_ft_agent_err.jsonl",
            state_path="logs/test_ft_agent_err_state.json",
        )
        runner = ForwardTestRunner(
            agents=[bad_agent],
            broker=broker,
            data_provider=MagicMock(),
            tickers=["AAPL"],
            config={},
            tracker=tracker,
        )

        # Run one bar — agent should crash, triggering halt
        runner._run_bar("2024-01-15T10:00:00")

        assert runner._halted
        broker.cancel_all_orders.assert_called_once()


# ── Slippage Tracking ────────────────────────────────────────────────────────


class TestSlippageTracking:
    def test_record_fill(self, tmp_path):
        from hydra.forward_test.tracker import ForwardTestTracker

        tracker = ForwardTestTracker(
            log_path=str(tmp_path / "fills.jsonl"),
            state_path=str(tmp_path / "state.json"),
        )

        tracker.record_fill(
            timestamp="2024-01-15T10:00:00",
            agent_name="ppo_1",
            ticker="NVDA",
            side="BUY",
            qty=5,
            expected_price=875.20,
            fill_price=875.45,
            slippage_bps=2.86,
        )

        log = tracker._read_log()
        assert len(log) == 1
        assert log[0]["type"] == "fill"
        assert log[0]["slippage_bps"] == 2.86
        assert log[0]["expected_price"] == 875.20
        assert log[0]["fill_price"] == 875.45

    def test_get_slippage_stats(self, tmp_path):
        from hydra.forward_test.tracker import ForwardTestTracker

        tracker = ForwardTestTracker(
            log_path=str(tmp_path / "fills.jsonl"),
            state_path=str(tmp_path / "state.json"),
        )

        # Record several fills with known slippage
        slippages = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        for i, slip in enumerate(slippages):
            tracker.record_fill(
                timestamp=f"2024-01-15T10:{i:02d}:00",
                agent_name="ppo_1",
                ticker="NVDA",
                side="BUY",
                qty=1,
                expected_price=100.0,
                fill_price=100.0 + slip * 100 / 10000,
                slippage_bps=slip,
            )

        stats = tracker.get_slippage_stats()
        assert stats["total_fills"] == 10
        assert stats["mean_slippage_bps"] == 5.5
        assert stats["median_slippage_bps"] == 5.5
        assert stats["max_slippage_bps"] == 10.0
        assert stats["training_assumption_bps"] == 12.0

    def test_empty_slippage_stats(self, tmp_path):
        from hydra.forward_test.tracker import ForwardTestTracker

        tracker = ForwardTestTracker(
            log_path=str(tmp_path / "empty.jsonl"),
            state_path=str(tmp_path / "state.json"),
        )

        stats = tracker.get_slippage_stats()
        assert stats == {}

    def test_pending_orders_tracked_in_runner(self):
        from hydra.forward_test.runner import ForwardTestRunner
        from hydra.forward_test.tracker import ForwardTestTracker

        agent = MagicMock()
        agent.name = "ppo_1"
        agent.select_action.return_value = np.array([0.5], dtype=np.float32)

        broker = MagicMock()
        broker.get_latest_price.return_value = 100.0
        broker.get_account.return_value = {"portfolio_value": 10000, "cash": 5000}
        broker.get_open_positions.return_value = []
        broker.place_order.return_value = {"id": "order123", "symbol": "AAPL", "qty": 5, "side": "buy"}
        broker.get_recent_fills.return_value = []

        tracker = ForwardTestTracker(
            log_path="logs/test_pending.jsonl",
            state_path="logs/test_pending_state.json",
        )
        runner = ForwardTestRunner(
            agents=[agent],
            broker=broker,
            data_provider=MagicMock(),
            tickers=["AAPL"],
            config={"initial_capital": 10000},
            tracker=tracker,
        )

        runner._run_bar("2024-01-15T10:00:00")

        # Should have a pending order tracked
        assert "order123" in runner._pending_orders
        assert runner._pending_orders["order123"]["expected_price"] == 100.0


# ── Alert System ─────────────────────────────────────────────────────────────


class TestAlertSystem:
    def test_daily_loss_alert(self):
        from hydra.forward_test.alerts import ForwardTestAlertManager

        mgr = ForwardTestAlertManager()
        metrics = {"total_return": -0.05, "max_drawdown": 0.02, "sharpe": 0.5, "trading_days": 10}
        thresholds = {"daily_loss_pct": 0.03, "max_drawdown_pct": 0.10}

        alerts = mgr.check_and_alert("ppo_1", metrics, thresholds)
        assert len(alerts) == 1
        assert alerts[0]["type"] == "daily_loss"

    def test_drawdown_alert(self):
        from hydra.forward_test.alerts import ForwardTestAlertManager

        mgr = ForwardTestAlertManager()
        metrics = {"total_return": 0.01, "max_drawdown": 0.15, "sharpe": 0.5, "trading_days": 10}
        thresholds = {"daily_loss_pct": 0.03, "max_drawdown_pct": 0.10}

        alerts = mgr.check_and_alert("ppo_1", metrics, thresholds)
        assert len(alerts) == 1
        assert alerts[0]["type"] == "drawdown"

    def test_negative_sharpe_alert(self):
        from hydra.forward_test.alerts import ForwardTestAlertManager

        mgr = ForwardTestAlertManager()
        metrics = {"total_return": 0.01, "max_drawdown": 0.02, "sharpe": -0.5, "trading_days": 10}
        thresholds = {"daily_loss_pct": 0.03, "max_drawdown_pct": 0.10}

        alerts = mgr.check_and_alert("ppo_1", metrics, thresholds)
        assert len(alerts) == 1
        assert alerts[0]["type"] == "negative_sharpe"

    def test_cooldown_prevents_duplicate_alerts(self):
        from hydra.forward_test.alerts import ForwardTestAlertManager

        mgr = ForwardTestAlertManager()
        metrics = {"total_return": -0.05, "max_drawdown": 0.02, "sharpe": 0.5, "trading_days": 10}
        thresholds = {"daily_loss_pct": 0.03, "max_drawdown_pct": 0.10}

        alerts1 = mgr.check_and_alert("ppo_1", metrics, thresholds)
        alerts2 = mgr.check_and_alert("ppo_1", metrics, thresholds)

        assert len(alerts1) == 1
        assert len(alerts2) == 0  # Cooldown prevents repeat

    def test_no_alert_within_thresholds(self):
        from hydra.forward_test.alerts import ForwardTestAlertManager

        mgr = ForwardTestAlertManager()
        metrics = {"total_return": 0.02, "max_drawdown": 0.05, "sharpe": 1.2, "trading_days": 10}
        thresholds = {"daily_loss_pct": 0.03, "max_drawdown_pct": 0.10}

        alerts = mgr.check_and_alert("ppo_1", metrics, thresholds)
        assert len(alerts) == 0

    def test_multiple_alerts_same_bar(self):
        from hydra.forward_test.alerts import ForwardTestAlertManager

        mgr = ForwardTestAlertManager()
        metrics = {"total_return": -0.10, "max_drawdown": 0.15, "sharpe": -1.0, "trading_days": 10}
        thresholds = {"daily_loss_pct": 0.03, "max_drawdown_pct": 0.10}

        alerts = mgr.check_and_alert("ppo_1", metrics, thresholds)
        assert len(alerts) == 3  # daily_loss + drawdown + negative_sharpe
        alert_types = {a["type"] for a in alerts}
        assert alert_types == {"daily_loss", "drawdown", "negative_sharpe"}


# ── Config Defaults ──────────────────────────────────────────────────────────


class TestConfigDefaults:
    def test_forward_test_duration_default(self):
        from hydra.forward_test.config import ForwardTestConfig

        config = ForwardTestConfig()
        assert config.duration_days == 60

    def test_schema_forward_test_duration(self):
        from hydra.config.schema import HydraConfig

        config = HydraConfig()
        assert config.forward_test.duration_days == 60

    def test_training_position_limit(self):
        from hydra.config.schema import HydraConfig

        config = HydraConfig()
        assert config.env.max_position_pct == 0.15

    def test_alert_config_fields(self):
        from hydra.forward_test.config import ForwardTestConfig

        config = ForwardTestConfig()
        assert config.alert_webhook_url == ""
        assert config.alert_daily_loss_pct == 0.03

    def test_schema_alert_config_fields(self):
        from hydra.config.schema import HydraConfig

        config = HydraConfig()
        assert config.forward_test.alert_webhook_url == ""
        assert config.forward_test.alert_daily_loss_pct == 0.03


# ── Broker Extensions ────────────────────────────────────────────────────────


class TestBrokerExtensions:
    def test_cancel_all_orders_exists(self):
        """Verify AlpacaBroker has cancel_all_orders method."""
        import sys
        from pathlib import Path
        parent = str(Path(__file__).parent.parent.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from trading_agents.utils.broker import AlpacaBroker
        assert hasattr(AlpacaBroker, "cancel_all_orders")

    def test_get_recent_fills_exists(self):
        """Verify AlpacaBroker has get_recent_fills method."""
        import sys
        from pathlib import Path
        parent = str(Path(__file__).parent.parent.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from trading_agents.utils.broker import AlpacaBroker
        assert hasattr(AlpacaBroker, "get_recent_fills")
