"""Forward-test performance tracker.

Logs every action, fill, and PnL to JSONL. Computes rolling metrics,
compares forward performance to backtest expectations, and produces
GRADUATE / EXTEND / FAIL verdicts.

Backtesting and training research only.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("hydra.forward_test.tracker")


class ForwardTestTracker:
    """Tracks forward-test performance and produces graduation verdicts.

    All data is appended to a JSONL file for durability. Metrics are computed
    from the full log on demand (no in-memory accumulation that could be lost).
    """

    def __init__(
        self,
        log_path: str = "logs/forward_test_log.jsonl",
        state_path: str = "logs/forward_test_state.json",
    ):
        self._log_path = Path(log_path)
        self._state_path = Path(state_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Recording ──────────────────────────────────────────────────────────

    def record_bar(
        self,
        timestamp: str,
        agent_name: str,
        actions: dict[str, float],
        positions: dict[str, float],
        portfolio_value: float,
        cash: float,
        orders_placed: list[dict] | None = None,
    ) -> None:
        """Append a bar record to the JSONL log."""
        entry = {
            "type": "bar",
            "timestamp": timestamp,
            "agent": agent_name,
            "actions": actions,
            "positions": positions,
            "portfolio_value": portfolio_value,
            "cash": cash,
            "orders": orders_placed or [],
            "logged_at": datetime.now().isoformat(),
        }
        self._append_log(entry)

    def record_daily_snapshot(
        self,
        date: str,
        agent_name: str,
        metrics: dict[str, Any],
    ) -> None:
        """Record end-of-day summary metrics."""
        entry = {
            "type": "daily_snapshot",
            "date": date,
            "agent": agent_name,
            "metrics": metrics,
            "logged_at": datetime.now().isoformat(),
        }
        self._append_log(entry)

    def record_fill(
        self,
        timestamp: str,
        agent_name: str,
        ticker: str,
        side: str,
        qty: int,
        expected_price: float,
        fill_price: float,
        slippage_bps: float,
        commission: float = 0.0,
    ) -> None:
        """Record an order fill with slippage data."""
        entry = {
            "type": "fill",
            "timestamp": timestamp,
            "agent": agent_name,
            "ticker": ticker,
            "side": side,
            "qty": qty,
            "expected_price": round(expected_price, 4),
            "fill_price": round(fill_price, 4),
            "slippage_bps": round(slippage_bps, 2),
            "commission": round(commission, 4),
            "logged_at": datetime.now().isoformat(),
        }
        self._append_log(entry)

    def record_event(self, event_type: str, detail: dict) -> None:
        """Record a lifecycle event (start, stop, error, etc)."""
        entry = {
            "type": "event",
            "event": event_type,
            "detail": detail,
            "logged_at": datetime.now().isoformat(),
        }
        self._append_log(entry)

    # ── Metrics Computation ────────────────────────────────────────────────

    def get_agent_bars(self, agent_name: str) -> list[dict]:
        """Load all bar records for a specific agent."""
        bars = []
        for entry in self._read_log():
            if entry.get("type") == "bar" and entry.get("agent") == agent_name:
                bars.append(entry)
        return bars

    def get_daily_returns(self, agent_name: str) -> list[float]:
        """Compute daily returns from bar data."""
        bars = self.get_agent_bars(agent_name)
        if len(bars) < 2:
            return []

        # Group by date, take last bar of each day
        daily_values: dict[str, float] = {}
        for bar in bars:
            date = bar["timestamp"][:10]  # YYYY-MM-DD
            daily_values[date] = bar["portfolio_value"]

        sorted_dates = sorted(daily_values.keys())
        returns = []
        for i in range(1, len(sorted_dates)):
            prev = daily_values[sorted_dates[i - 1]]
            curr = daily_values[sorted_dates[i]]
            if prev > 0:
                returns.append((curr - prev) / prev)

        return returns

    def get_metrics(self, agent_name: str) -> dict[str, Any]:
        """Compute forward-test metrics for an agent."""
        bars = self.get_agent_bars(agent_name)
        if not bars:
            return {"error": "No bar data"}

        values = [b["portfolio_value"] for b in bars]
        initial = values[0]
        final = values[-1]
        total_return = (final - initial) / initial if initial > 0 else 0

        # Compute daily returns for Sharpe
        daily_returns = self.get_daily_returns(agent_name)

        # Sharpe ratio (annualized, 252 trading days)
        sharpe = 0.0
        if len(daily_returns) >= 5:
            mean_r = statistics.mean(daily_returns)
            std_r = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 1.0
            if std_r > 0:
                sharpe = (mean_r / std_r) * math.sqrt(252)

        # Max drawdown
        peak = values[0]
        max_dd = 0.0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Win rate from orders
        wins = 0
        losses = 0
        for bar in bars:
            for order in bar.get("orders", []):
                pnl = order.get("realized_pnl", 0)
                if pnl > 0:
                    wins += 1
                elif pnl < 0:
                    losses += 1

        total_trades = wins + losses
        win_rate = wins / total_trades if total_trades > 0 else 0.0

        # Profit factor
        gross_profit = sum(
            o.get("realized_pnl", 0)
            for b in bars for o in b.get("orders", [])
            if o.get("realized_pnl", 0) > 0
        )
        gross_loss = abs(sum(
            o.get("realized_pnl", 0)
            for b in bars for o in b.get("orders", [])
            if o.get("realized_pnl", 0) < 0
        ))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
            float("inf") if gross_profit > 0 else 0.0
        )

        # Trading days
        dates = set(b["timestamp"][:10] for b in bars)

        return {
            "total_return": round(total_return, 6),
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "total_trades": total_trades,
            "trading_days": len(dates),
            "initial_value": round(initial, 2),
            "final_value": round(final, 2),
            "bars_recorded": len(bars),
        }

    def get_slippage_stats(self) -> dict[str, Any]:
        """Compute slippage statistics across all recorded fills."""
        fills = [e for e in self._read_log() if e.get("type") == "fill"]
        if not fills:
            return {}

        slippages = [f["slippage_bps"] for f in fills]
        slippages.sort()
        n = len(slippages)

        mean_slip = statistics.mean(slippages)
        median_slip = statistics.median(slippages)
        p95_idx = min(int(n * 0.95), n - 1)
        p95_slip = slippages[p95_idx]
        max_slip = slippages[-1]

        return {
            "total_fills": n,
            "mean_slippage_bps": round(mean_slip, 2),
            "median_slippage_bps": round(median_slip, 2),
            "p95_slippage_bps": round(p95_slip, 2),
            "max_slippage_bps": round(max_slip, 2),
            "training_assumption_bps": 12.0,  # 10 slippage + 2 spread
            "excess_slippage_bps": round(mean_slip - 12.0, 2),
        }

    # ── Comparison to Backtest ─────────────────────────────────────────────

    def get_comparison(
        self,
        agent_name: str,
        backtest_expectations: dict[str, float],
        config: dict | None = None,
    ) -> dict[str, Any]:
        """Compare forward metrics against backtest expectations.

        Args:
            agent_name: Agent to evaluate.
            backtest_expectations: Dict with keys: sharpe, max_drawdown, win_rate.
            config: Forward test config dict with tolerance thresholds.

        Returns:
            Dict with per-metric comparison and overall assessment.
        """
        config = config or {}
        sharpe_retention_min = config.get("sharpe_retention_min", 0.50)
        dd_tolerance = config.get("drawdown_tolerance", 1.5)
        wr_tolerance = config.get("win_rate_tolerance", 0.80)

        forward = self.get_metrics(agent_name)
        if "error" in forward:
            return {"error": forward["error"]}

        bt_sharpe = backtest_expectations.get("sharpe", 0)
        bt_dd = backtest_expectations.get("max_drawdown", 0)
        bt_wr = backtest_expectations.get("win_rate", 0)

        # Sharpe retention
        sharpe_retention = (
            forward["sharpe"] / bt_sharpe if bt_sharpe > 0 else 1.0
        )

        # Drawdown ratio (forward DD / backtest DD)
        dd_ratio = (
            forward["max_drawdown"] / bt_dd if bt_dd > 0 else 0.0
        )

        # Win rate retention
        wr_retention = (
            forward["win_rate"] / bt_wr if bt_wr > 0 else 1.0
        )

        comparisons = {
            "sharpe": {
                "backtest": bt_sharpe,
                "forward": forward["sharpe"],
                "retention": round(sharpe_retention, 4),
                "threshold": sharpe_retention_min,
                "passed": sharpe_retention >= sharpe_retention_min,
            },
            "max_drawdown": {
                "backtest": bt_dd,
                "forward": forward["max_drawdown"],
                "ratio": round(dd_ratio, 4),
                "threshold": dd_tolerance,
                "passed": dd_ratio <= dd_tolerance,
            },
            "win_rate": {
                "backtest": bt_wr,
                "forward": forward["win_rate"],
                "retention": round(wr_retention, 4),
                "threshold": wr_tolerance,
                "passed": wr_retention >= wr_tolerance,
            },
        }

        return {
            "agent": agent_name,
            "forward_metrics": forward,
            "comparisons": comparisons,
            "all_passed": all(c["passed"] for c in comparisons.values()),
        }

    # ── Verdict ────────────────────────────────────────────────────────────

    def get_verdict(
        self,
        agent_name: str,
        backtest_expectations: dict[str, float],
        config: dict | None = None,
    ) -> dict[str, Any]:
        """Produce a GRADUATE / EXTEND / FAIL verdict.

        Logic:
        - GRADUATE: All metrics within tolerance
        - EXTEND: Any metric borderline (within 10% of threshold)
        - FAIL: Sharpe retention < 30% OR drawdown > 2x backtest
        """
        comparison = self.get_comparison(agent_name, backtest_expectations, config)
        if "error" in comparison:
            return {"verdict": "FAIL", "reason": comparison["error"]}

        comps = comparison["comparisons"]

        # Hard fail conditions
        sharpe_retention = comps["sharpe"]["retention"]
        dd_ratio = comps["max_drawdown"]["ratio"]

        if sharpe_retention < 0.30:
            return {
                "verdict": "FAIL",
                "reason": f"Sharpe retention {sharpe_retention:.0%} < 30% hard floor",
                "comparison": comparison,
            }
        if dd_ratio > 2.0:
            return {
                "verdict": "FAIL",
                "reason": f"Drawdown {dd_ratio:.1f}x backtest exceeds 2.0x hard ceiling",
                "comparison": comparison,
            }

        # Check if all pass
        if comparison["all_passed"]:
            return {
                "verdict": "GRADUATE",
                "reason": "All forward metrics within tolerance of backtest expectations",
                "comparison": comparison,
            }

        # Check for borderline (within 10% of threshold)
        borderline = False
        borderline_metrics = []
        for metric, data in comps.items():
            if not data["passed"]:
                if metric == "max_drawdown":
                    # For DD, check if ratio is within 10% above threshold
                    if data["ratio"] <= data["threshold"] * 1.10:
                        borderline = True
                        borderline_metrics.append(metric)
                else:
                    # For sharpe/win_rate, check if retention is within 10% below threshold
                    retention_key = "retention"
                    if data.get(retention_key, 0) >= data["threshold"] * 0.90:
                        borderline = True
                        borderline_metrics.append(metric)

        if borderline:
            return {
                "verdict": "EXTEND",
                "reason": f"Borderline on {', '.join(borderline_metrics)} — extend 10 more days",
                "comparison": comparison,
            }

        return {
            "verdict": "FAIL",
            "reason": "Forward metrics significantly below backtest expectations",
            "comparison": comparison,
        }

    # ── Graduation Report ──────────────────────────────────────────────────

    def get_graduation_report(
        self,
        agents: list[str],
        backtest_expectations: dict[str, dict[str, float]],
        config: dict | None = None,
    ) -> dict[str, Any]:
        """Produce a full graduation report for all agents in the forward test.

        Args:
            agents: List of agent names.
            backtest_expectations: {agent_name: {sharpe, max_drawdown, win_rate}}.
            config: Forward test config.

        Returns:
            Full report with per-agent verdicts and overall summary.
        """
        results = {}
        graduated = []
        extended = []
        failed = []

        for agent_name in agents:
            expectations = backtest_expectations.get(agent_name, {})
            verdict_data = self.get_verdict(agent_name, expectations, config)
            results[agent_name] = verdict_data

            verdict = verdict_data["verdict"]
            if verdict == "GRADUATE":
                graduated.append(agent_name)
            elif verdict == "EXTEND":
                extended.append(agent_name)
            else:
                failed.append(agent_name)

        return {
            "report_type": "forward_test_graduation",
            "generated_at": datetime.now().isoformat(),
            "agent_results": results,
            "graduated": graduated,
            "extended": extended,
            "failed": failed,
            "summary": (
                f"{len(graduated)} graduated, {len(extended)} extended, "
                f"{len(failed)} failed out of {len(agents)} agents tested."
            ),
        }

    # ── State Management ───────────────────────────────────────────────────

    def save_state(self, state: dict) -> None:
        """Save forward test state to JSON."""
        state["updated"] = datetime.now().isoformat()
        with open(self._state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def load_state(self) -> dict:
        """Load forward test state from JSON."""
        if not self._state_path.exists():
            return {}
        try:
            with open(self._state_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    # ── Internal ───────────────────────────────────────────────────────────

    def _append_log(self, entry: dict) -> None:
        """Append a single entry to the JSONL log."""
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def _read_log(self) -> list[dict]:
        """Read all entries from the JSONL log."""
        if not self._log_path.exists():
            return []
        entries = []
        with open(self._log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries
