"""Tests for sub-account system and capital allocation."""

import pytest

from hydra.forward_test.capital_allocator import Allocation, compute_allocations
from hydra.forward_test.sub_account import Position, SubAccount


class TestCapitalAllocator:
    """Tests for Sharpe-weighted capital allocation."""

    def test_basic_sharpe_weighting(self):
        results = {
            "agent_a": {"sharpe": 1.0, "passed": True},
            "agent_b": {"sharpe": 0.5, "passed": True},
        }
        allocs = compute_allocations(results, total_capital=10000.0)
        assert len(allocs) == 2
        # agent_a should get ~2/3, agent_b ~1/3
        assert allocs[0].agent_name == "agent_a"
        assert allocs[0].weight > 0.6
        assert allocs[1].agent_name == "agent_b"
        assert allocs[1].weight < 0.4
        # Total capital should sum correctly
        total = sum(a.capital for a in allocs)
        assert abs(total - 10000.0) < 1.0

    def test_failed_agents_excluded(self):
        results = {
            "good": {"sharpe": 1.0, "passed": True},
            "bad": {"sharpe": 0.8, "passed": False},
        }
        allocs = compute_allocations(results, total_capital=10000.0)
        assert len(allocs) == 1
        assert allocs[0].agent_name == "good"
        assert abs(allocs[0].capital - 10000.0) < 1.0

    def test_below_min_sharpe_excluded(self):
        results = {
            "good": {"sharpe": 0.8, "passed": True},
            "weak": {"sharpe": 0.1, "passed": True},
        }
        allocs = compute_allocations(results, total_capital=10000.0, min_sharpe=0.3)
        assert len(allocs) == 1
        assert allocs[0].agent_name == "good"

    def test_max_agents_limits(self):
        results = {f"a{i}": {"sharpe": i * 0.5, "passed": True} for i in range(1, 6)}
        allocs = compute_allocations(results, total_capital=10000.0, max_agents=2, min_sharpe=0.0)
        assert len(allocs) == 2
        # Should be the top-2 by Sharpe
        names = {a.agent_name for a in allocs}
        assert "a5" in names  # Sharpe 2.5
        assert "a4" in names  # Sharpe 2.0

    def test_empty_results(self):
        allocs = compute_allocations({}, total_capital=10000.0)
        assert allocs == []

    def test_all_agents_below_min_sharpe(self):
        results = {
            "a": {"sharpe": 0.1, "passed": True},
            "b": {"sharpe": 0.2, "passed": True},
        }
        allocs = compute_allocations(results, total_capital=10000.0, min_sharpe=0.3)
        assert allocs == []

    def test_single_agent_gets_full_capital(self):
        results = {"solo": {"sharpe": 1.5, "passed": True}}
        allocs = compute_allocations(results, total_capital=5000.0)
        assert len(allocs) == 1
        assert allocs[0].weight == pytest.approx(1.0, abs=0.001)
        assert abs(allocs[0].capital - 5000.0) < 1.0


class TestPosition:
    """Tests for Position dataclass."""

    def test_market_value(self):
        pos = Position(ticker="AAPL", qty=10, avg_cost=150.0)
        assert pos.market_value(160.0) == 1600.0

    def test_unrealized_pnl(self):
        pos = Position(ticker="AAPL", qty=10, avg_cost=150.0)
        assert pos.unrealized_pnl(160.0) == 100.0
        assert pos.unrealized_pnl(140.0) == -100.0

    def test_zero_position_pnl(self):
        pos = Position(ticker="AAPL", qty=0, avg_cost=0.0)
        assert pos.unrealized_pnl(160.0) == 0.0


class TestSubAccount:
    """Tests for virtual sub-account portfolio tracking."""

    def test_initial_state(self):
        sub = SubAccount("agent_a", 5000.0)
        assert sub.cash == 5000.0
        assert sub.portfolio_value({}) == 5000.0
        assert sub.initial_capital == 5000.0
        assert sub.agent_name == "agent_a"

    def test_buy(self):
        sub = SubAccount("agent_a", 10000.0)
        result = sub.apply_fill("AAPL", 10, "BUY", 150.0)
        assert "error" not in result
        assert sub.cash == 8500.0  # 10000 - 1500
        assert sub.get_position_qty("AAPL") == 10
        assert sub.portfolio_value({"AAPL": 150.0}) == 10000.0  # no change at fill price

    def test_buy_and_sell(self):
        sub = SubAccount("agent_a", 10000.0)
        sub.apply_fill("AAPL", 10, "BUY", 150.0)
        assert sub.portfolio_value({"AAPL": 160.0}) == 8500 + 1600  # 10100

        result = sub.apply_fill("AAPL", 5, "SELL", 160.0)
        assert "error" not in result
        assert result["realized_pnl"] == 50.0  # (160-150)*5
        assert sub.get_position_qty("AAPL") == 5
        assert sub.cash == 8500 + 800  # 9300

    def test_insufficient_cash(self):
        sub = SubAccount("agent_a", 100.0)
        result = sub.apply_fill("AAPL", 10, "BUY", 150.0)
        assert "error" in result
        assert result["error"] == "insufficient_cash"
        assert sub.cash == 100.0  # unchanged

    def test_insufficient_position(self):
        sub = SubAccount("agent_a", 10000.0)
        sub.apply_fill("AAPL", 5, "BUY", 100.0)
        result = sub.apply_fill("AAPL", 10, "SELL", 110.0)
        assert "error" in result
        assert result["error"] == "insufficient_position"
        assert sub.get_position_qty("AAPL") == 5  # unchanged

    def test_drawdown_tracking(self):
        sub = SubAccount("agent_a", 10000.0)
        sub.apply_fill("AAPL", 50, "BUY", 100.0)  # 5000 in AAPL, 5000 cash
        sub.update_peak({"AAPL": 100.0})  # PV = 10000, peak = 10000
        assert sub.peak_value == 10000.0

        dd = sub.current_drawdown({"AAPL": 80.0})  # PV = 5000 + 4000 = 9000
        assert dd == pytest.approx(0.10, abs=0.001)

    def test_peak_updates_upward_only(self):
        sub = SubAccount("agent_a", 10000.0)
        sub.update_peak({})  # PV = 10000
        assert sub.peak_value == 10000.0

        sub.apply_fill("AAPL", 10, "BUY", 100.0)
        sub.update_peak({"AAPL": 120.0})  # PV = 9000 + 1200 = 10200
        assert sub.peak_value == 10200.0

        sub.update_peak({"AAPL": 90.0})  # PV = 9000 + 900 = 9900
        assert sub.peak_value == 10200.0  # didn't decrease

    def test_snapshot(self):
        sub = SubAccount("agent_a", 10000.0)
        sub.apply_fill("NVDA", 5, "BUY", 800.0)
        snap = sub.get_snapshot({"NVDA": 820.0})
        assert snap["agent"] == "agent_a"
        assert snap["portfolio_value"] == 6000 + 4100  # 10100
        assert snap["initial_capital"] == 10000.0
        assert snap["total_return"] == pytest.approx(0.01, abs=0.001)
        assert snap["num_positions"] == 1
        assert "NVDA" in snap["positions"]

    def test_holdings_dict(self):
        sub = SubAccount("agent_a", 10000.0)
        sub.apply_fill("AAPL", 10, "BUY", 100.0)
        sub.apply_fill("NVDA", 5, "BUY", 200.0)
        holdings = sub.get_holdings_dict()
        assert holdings == {"AAPL": 10, "NVDA": 5}

    def test_sell_clears_position(self):
        sub = SubAccount("agent_a", 10000.0)
        sub.apply_fill("AAPL", 10, "BUY", 100.0)
        sub.apply_fill("AAPL", 10, "SELL", 110.0)
        assert sub.get_position_qty("AAPL") == 0
        assert "AAPL" not in sub.get_holdings_dict()

    def test_commission_deducted(self):
        sub = SubAccount("agent_a", 10000.0)
        sub.apply_fill("AAPL", 10, "BUY", 100.0, commission=5.0)
        assert sub.cash == 10000 - 1000 - 5  # 8995
