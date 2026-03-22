"""Data loader — reads JSON state files and transforms for CEO display."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import (
    TRAINING_STATE_FILE,
    CORP_STATE_FILE,
    STARTING_CAPITAL_CAD,
    compute_portfolio_value,
    compute_dollar_pnl,
    compute_safety_score,
    friendly_name,
)


def load_dashboard_data() -> dict[str, Any]:
    """Load and transform all data for the CEO dashboard.

    Returns a dict with all pre-computed CEO-friendly values.
    """
    training = _load_json(TRAINING_STATE_FILE)
    corp = _load_json(CORP_STATE_FILE)

    if not training:
        return _empty_data()

    validation = training.get("validation", {})
    benchmark = training.get("benchmark", {})
    generations = training.get("generations", [])
    summary = training.get("summary", {})
    config = training.get("config", {})

    # Find best agent
    best_agent = ""
    best_return = 0.0
    for name, metrics in validation.items():
        total_return = metrics.get("total_return", 0)
        if total_return > best_return:
            best_return = total_return
            best_agent = name

    if not validation:
        # No validation data yet — use eval scores from generations if available
        if generations:
            last_gen = generations[-1]
            eval_scores = last_gen.get("eval_scores", {})
            if eval_scores:
                for name, score in eval_scores.items():
                    if score > best_return:
                        best_return = score / 100.0  # Scores are scaled, approximate
                        best_agent = name

    # Build agent leaderboard
    leaderboard = []
    for name, metrics in validation.items():
        total_return = metrics.get("total_return", 0)
        leaderboard.append({
            "name": friendly_name(name),
            "internal_name": name,
            "return_pct": round(total_return * 100, 2),
            "return_cad": round(compute_dollar_pnl(total_return), 2),
            "sharpe": round(metrics.get("sharpe", 0), 2),
            "max_drawdown_pct": round(abs(metrics.get("max_drawdown", 0)) * 100, 2),
            "win_rate_pct": round(metrics.get("win_rate", 0) * 100, 1),
            "profit_factor": round(metrics.get("profit_factor", 0), 2),
            "passed": metrics.get("passed", False),
            "is_best": name == best_agent,
        })
    leaderboard.sort(key=lambda x: x["return_pct"], reverse=True)

    # Benchmark data
    spy_return = benchmark.get("total_return", 0)
    excess_return = best_return - spy_return

    # Risk metrics from best agent
    best_metrics = validation.get(best_agent, {})
    max_dd = abs(best_metrics.get("max_drawdown", 0))
    win_rate = best_metrics.get("win_rate", 0)
    profit_factor = best_metrics.get("profit_factor", 0)
    safety = compute_safety_score(max_dd, win_rate, profit_factor)

    # Alerts from generation history
    alerts = _generate_alerts(generations, validation, best_agent, benchmark)

    # Equity curve data
    price_history = training.get("price_history", [])
    equity_curve = benchmark.get("equity_curve", [])

    # Build generation history for charts
    gen_history = _build_generation_history(generations)

    return {
        # Hero KPIs
        "portfolio_value": round(compute_portfolio_value(best_return), 2),
        "total_return_pct": round(best_return * 100, 2),
        "dollar_pnl": round(compute_dollar_pnl(best_return), 2),
        "best_agent": friendly_name(best_agent),
        "best_agent_return_pct": round(best_return * 100, 2),
        "spy_return_pct": round(spy_return * 100, 2),
        "excess_return_pct": round(excess_return * 100, 2),
        "excess_label": f"{'+'if excess_return > 0 else ''}{excess_return * 100:.1f}% {'ahead' if excess_return > 0 else 'behind'}",

        # Risk
        "safety_score": safety,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "max_drawdown_cad": round(STARTING_CAPITAL_CAD * max_dd, 2),
        "win_rate_pct": round(win_rate * 100, 1),
        "profit_factor": round(profit_factor, 2),

        # Tables
        "leaderboard": leaderboard,
        "passed_count": len(summary.get("passed_agents", [])),
        "total_agents": len(validation),

        # Alerts
        "alerts": alerts,

        # Charts
        "price_history": price_history,
        "spy_equity_curve": equity_curve,
        "spy_return": spy_return,
        "gen_history": gen_history,

        # Benchmark table
        "benchmark": {
            "spy_return_pct": round(spy_return * 100, 2),
            "spy_drawdown_pct": round(abs(benchmark.get("max_drawdown", 0)) * 100, 2),
            "spy_sharpe": round(benchmark.get("sharpe", 0), 2),
        },

        # Training status
        "updated": training.get("updated", "N/A"),
        "total_generations": summary.get("total_generations", 0),
        "tickers": config.get("tickers", []),
        "num_stocks": config.get("num_stocks", 0),
        "real_data": config.get("real_data", False),

        # Corp state
        "corp": corp or {},
    }


def _build_generation_history(generations: list[dict]) -> list[dict]:
    """Extract per-generation metrics for the training progress chart."""
    history = []
    for gen in generations:
        gen_num = gen.get("generation", 0)
        eval_scores = gen.get("eval_scores", {})
        mean_reward = gen.get("train_mean_reward", 0)
        pool_size = gen.get("pool_size", 0)

        # Best and worst agent this generation
        best_score = max(eval_scores.values()) if eval_scores else 0
        worst_score = min(eval_scores.values()) if eval_scores else 0
        best_name = ""
        if eval_scores:
            best_name = max(eval_scores, key=eval_scores.get)

        # Competition Sharpe if available
        comp = gen.get("competition", {})

        # Count promotions/demotions
        promoted = gen.get("promoted", [])
        demoted = gen.get("demoted", [])

        # CHIMERA diagnosis severity
        diag = gen.get("diagnosis") or {}

        history.append({
            "gen": gen_num,
            "best_eval": round(best_score, 1),
            "worst_eval": round(worst_score, 1),
            "mean_reward": round(mean_reward, 1),
            "best_agent": friendly_name(best_name),
            "pool_size": pool_size,
            "num_agents_eval": len(eval_scores),
            "promoted": len(promoted),
            "demoted": len(demoted),
            "severity": diag.get("severity", ""),
        })
    return history


def _generate_alerts(generations, validation, best_agent, benchmark) -> list[dict]:
    """Generate CEO-friendly alerts from generation history."""
    alerts = []

    for gen in generations:
        gen_num = gen.get("generation", 0)

        for agent in gen.get("promoted", []):
            alerts.append({
                "type": "promoted",
                "icon": "star",
                "color": "blue",
                "message": f"{friendly_name(agent)} earned a promotion — performing well",
                "gen": gen_num,
            })

        for agent in gen.get("demoted", []):
            alerts.append({
                "type": "demoted",
                "icon": "arrow_down",
                "color": "amber",
                "message": f"{friendly_name(agent)} removed — underperforming",
                "gen": gen_num,
            })

        diag = gen.get("diagnosis") or {}
        severity = diag.get("severity", "")
        if severity in ("severe", "critical"):
            issue = diag.get("primary_issue", "unknown issue")
            alerts.append({
                "type": "warning",
                "icon": "warning",
                "color": "red",
                "message": f"System detected an issue: {issue}",
                "gen": gen_num,
            })

    # Add benchmark comparison alert
    if best_agent and validation:
        best_return = validation.get(best_agent, {}).get("total_return", 0)
        spy_return = benchmark.get("total_return", 0)
        excess = best_return - spy_return
        if excess > 0.05:
            alerts.append({
                "type": "benchmark_beat",
                "icon": "rocket",
                "color": "green",
                "message": f"Outperforming the market by {excess * 100:.1f}%",
                "gen": -1,
            })
        elif excess < -0.05:
            alerts.append({
                "type": "benchmark_loss",
                "icon": "chart_decreasing",
                "color": "red",
                "message": f"Underperforming the market by {abs(excess) * 100:.1f}%",
                "gen": -1,
            })

    # Most recent alerts first
    alerts.reverse()
    return alerts[:20]


def _load_json(path: str) -> dict | None:
    """Load a JSON file, returning None if missing or invalid."""
    p = Path(path)
    if not p.exists():
        # Try relative to script location
        p = Path(__file__).parent.parent.parent / path
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _empty_data() -> dict[str, Any]:
    """Return empty dashboard data structure."""
    return {
        "portfolio_value": STARTING_CAPITAL_CAD,
        "total_return_pct": 0.0,
        "dollar_pnl": 0.0,
        "best_agent": "No data yet",
        "best_agent_return_pct": 0.0,
        "spy_return_pct": 0.0,
        "excess_return_pct": 0.0,
        "excess_label": "No data",
        "safety_score": 50,
        "max_drawdown_pct": 0.0,
        "max_drawdown_cad": 0.0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "leaderboard": [],
        "passed_count": 0,
        "total_agents": 0,
        "alerts": [],
        "price_history": [],
        "spy_equity_curve": [],
        "spy_return": 0,
        "gen_history": [],
        "benchmark": {"spy_return_pct": 0, "spy_drawdown_pct": 0, "spy_sharpe": 0},
        "updated": "N/A",
        "total_generations": 0,
        "tickers": [],
        "num_stocks": 0,
        "real_data": False,
        "corp": {},
    }
