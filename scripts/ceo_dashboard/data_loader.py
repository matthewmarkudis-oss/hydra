"""Data loader — reads JSON state files and transforms for CEO display."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import (
    TRAINING_STATE_FILE,
    CORP_STATE_FILE,
    FORWARD_TEST_LOG_FILE,
    FORWARD_TEST_STATE_FILE,
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
    forward_test = _load_forward_test_data()
    corp_recs = _load_corp_recommendations(corp)

    if not training:
        empty = _empty_data()
        empty["forward_test"] = forward_test
        empty["corp_recommendations"] = corp_recs
        return empty

    validation = training.get("validation", {})
    benchmark = training.get("benchmark", {})
    generations = training.get("generations", [])
    summary = training.get("summary", {})
    config = training.get("config", {})

    # Find best agent — use None sentinel so negative returns still register
    best_agent = ""
    best_return = None
    for name, metrics in validation.items():
        total_return = metrics.get("total_return", 0)
        if best_return is None or total_return > best_return:
            best_return = total_return
            best_agent = name

    has_real_pnl = bool(validation)

    if not validation and generations:
        # No validation data — try agent_pnl, then best_return_pct, then eval_scores
        last_gen = generations[-1]
        agent_pnl = last_gen.get("agent_pnl", {})
        if agent_pnl:
            has_real_pnl = True
            for name, pnl_data in agent_pnl.items():
                ret_pct = pnl_data.get("mean_return_pct", 0.0)
                ret_ratio = ret_pct / 100.0
                if best_return is None or ret_ratio > best_return:
                    best_return = ret_ratio
                    best_agent = name
        elif last_gen.get("best_return_pct") is not None:
            has_real_pnl = True
            best_return = last_gen["best_return_pct"] / 100.0
            eval_scores = last_gen.get("eval_scores", {})
            if eval_scores:
                best_agent = max(eval_scores, key=eval_scores.get)
        else:
            # Final fallback: use eval_scores — NOT real P&L
            has_real_pnl = False
            eval_scores = last_gen.get("eval_scores", {})
            if eval_scores:
                best_name = max(eval_scores, key=eval_scores.get)
                best_return = 0.0  # Don't fake P&L from reward scores
                best_agent = best_name

    if best_return is None:
        best_return = 0.0

    # Forward-test allocation data (if a graduation proposal exists)
    ft_allocations = []
    corp_data = corp or {}
    proposals = corp_data.get("proposals", [])
    for p in proposals:
        if isinstance(p, dict) and p.get("type") == "graduation":
            ft_allocations = p.get("allocations", [])
            break

    # Build agent leaderboard
    leaderboard = []
    if validation:
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
    elif generations:
        # Build leaderboard from latest generation's agent_pnl
        last_gen = generations[-1]
        agent_pnl = last_gen.get("agent_pnl", {})
        eval_scores = last_gen.get("eval_scores", {})
        for name, pnl_data in agent_pnl.items():
            ret_pct = pnl_data.get("mean_return_pct", 0.0)
            cash_ratio = pnl_data.get("mean_cash_ratio", 1.0)
            deployed = 1.0 - cash_ratio
            leaderboard.append({
                "name": friendly_name(name),
                "internal_name": name,
                "return_pct": round(ret_pct, 3),
                "return_cad": round(compute_dollar_pnl(ret_pct / 100.0), 2),
                "sharpe": round(eval_scores.get(name, 0) / 100.0, 2),
                "max_drawdown_pct": 0.0,
                "win_rate_pct": round(deployed * 100, 1),
                "profit_factor": 0.0,
                "passed": ret_pct > 0 and deployed > 0.2,
                "is_best": name == best_agent,
            })

    # Enrich leaderboard with allocation data
    alloc_lookup = {a.get("agent_name", ""): a for a in ft_allocations}
    for entry in leaderboard:
        alloc = alloc_lookup.get(entry["internal_name"], {})
        entry["allocation_pct"] = round(alloc.get("weight", 0) * 100, 1)
        entry["allocation_cad"] = round(alloc.get("capital", 0), 2)

    leaderboard.sort(key=lambda x: x["return_pct"], reverse=True)

    # Benchmark data
    spy_return = benchmark.get("total_return", 0)
    excess_return = best_return - spy_return

    # Risk metrics from best agent
    best_metrics = validation.get(best_agent, {})
    max_dd = abs(best_metrics.get("max_drawdown", 0))
    win_rate = best_metrics.get("win_rate", 0)
    profit_factor = best_metrics.get("profit_factor", 0)

    if not validation and generations:
        # Derive risk proxies from generation P&L history
        gen_returns = [
            g.get("best_return_pct", 0) for g in generations
            if g.get("best_return_pct") is not None
        ]
        if gen_returns:
            profitable_gens = sum(1 for r in gen_returns if r > 0)
            win_rate = profitable_gens / len(gen_returns)
            # Estimate profit factor from mean positive vs mean negative returns
            pos = [r for r in gen_returns if r > 0]
            neg = [abs(r) for r in gen_returns if r < 0]
            if pos and neg:
                profit_factor = (sum(pos) / len(pos)) / max(sum(neg) / len(neg), 0.01)
            elif pos:
                profit_factor = 3.0  # All profitable, cap at 3.0
            # Rough max drawdown from worst generation return
            worst = min(gen_returns)
            if worst < 0:
                max_dd = abs(worst) / 100.0  # Convert from % to ratio

    safety = compute_safety_score(max_dd, win_rate, profit_factor)

    # Alerts from generation history
    alerts = _generate_alerts(generations, validation, best_agent, benchmark)

    # Equity curve data
    price_history = training.get("price_history", [])
    equity_curve = benchmark.get("equity_curve", [])

    # Build generation history for charts
    gen_history = _build_generation_history(generations)

    # === Current Generation metrics ===
    current_gen = _compute_current_gen(generations, gen_history)

    # === All-Time Best metrics ===
    all_time_best = _compute_all_time_best(generations, gen_history)

    # === Generation Scorecard (from generation_scorer) ===
    scorecard = _compute_scorecard(generations)

    # Compute period P&L (annualized, monthly, daily) from the backtest window
    lookback_days = config.get("lookback_days", 60) or 60
    pnl_per_day = compute_dollar_pnl(best_return) / max(lookback_days, 1) if has_real_pnl else 0.0
    pnl_per_month = pnl_per_day * 21  # ~21 trading days per month
    pnl_per_year = pnl_per_day * 252   # ~252 trading days per year

    # Best eval score (always available even without P&L)
    best_eval_score = 0.0
    best_eval_agent = ""
    if generations:
        last_es = generations[-1].get("eval_scores", {})
        if last_es:
            best_eval_agent = max(last_es, key=last_es.get)
            best_eval_score = last_es[best_eval_agent]

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

        # P&L periods
        "has_real_pnl": has_real_pnl,
        "lookback_days": lookback_days,
        "pnl_per_day": round(pnl_per_day, 2),
        "pnl_per_month": round(pnl_per_month, 2),
        "pnl_per_year": round(pnl_per_year, 2),
        "best_eval_score": round(best_eval_score, 1),
        "best_eval_agent": friendly_name(best_eval_agent),

        # Risk
        "safety_score": safety,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "max_drawdown_cad": round(STARTING_CAPITAL_CAD * max_dd, 2),
        "win_rate_pct": round(win_rate * 100, 1),
        "profit_factor": round(profit_factor, 2),

        # Tables
        "leaderboard": leaderboard,
        "passed_count": len(summary.get("passed_agents", [])),
        "total_agents": len(validation) or len(leaderboard),

        # Alerts
        "alerts": alerts,

        # Charts
        "price_history": price_history,
        "spy_equity_curve": equity_curve,
        "spy_return": spy_return,
        "gen_history": gen_history,
        "ft_allocations": ft_allocations,

        # Benchmark table
        "benchmark": {
            "spy_return_pct": round(spy_return * 100, 2),
            "spy_drawdown_pct": round(abs(benchmark.get("max_drawdown", 0)) * 100, 2),
            "spy_sharpe": round(benchmark.get("sharpe", 0), 2),
        },

        # Training status
        "updated": training.get("updated", "N/A"),
        "total_generations": summary.get("total_generations", 0),
        "target_generations": config.get("num_generations", 0) or summary.get("total_generations", 0),
        "tickers": config.get("tickers", []),
        "num_stocks": config.get("num_stocks", 0),
        "real_data": config.get("real_data", False),

        # Corp state
        "corp": corp or {},

        # Forward test
        "forward_test": forward_test,

        # Corp recommendations
        "corp_recommendations": corp_recs,

        # Computed sections for two-row KPI layout
        "current_gen": current_gen,
        "all_time_best": all_time_best,
        "scorecard": scorecard,

        # Conviction data from latest generation
        "conviction": generations[-1].get("conviction", {}) if generations else {},

        # Geopolitics ticker
        "geopolitics": _load_geopolitics_ticker(corp),
    }


def _compute_current_gen(generations: list[dict], gen_history: list[dict]) -> dict:
    """Compute current-generation KPI data from latest generation."""
    if not generations:
        return {}

    last = generations[-1]
    agent_pnl = last.get("agent_pnl", {})
    eval_scores = last.get("eval_scores", {})
    diag = last.get("diagnosis") or {}

    # Best return this generation (from agent_pnl or eval_scores proxy)
    best_return_pct = last.get("best_return_pct")
    mean_return_pct = last.get("mean_return_pct")
    best_agent = ""

    if agent_pnl:
        best_name = max(agent_pnl, key=lambda a: agent_pnl[a].get("mean_return_pct", 0))
        best_return_pct = agent_pnl[best_name].get("mean_return_pct", 0.0)
        best_agent = best_name
        if mean_return_pct is None:
            rets = [p.get("mean_return_pct", 0) for p in agent_pnl.values()]
            mean_return_pct = sum(rets) / len(rets) if rets else 0.0
    elif eval_scores:
        best_name = max(eval_scores, key=eval_scores.get)
        # eval_scores are reward values — don't pretend they are return %
        best_agent = best_name
        if best_return_pct is None:
            best_return_pct = 0.0  # No real P&L available
        if mean_return_pct is None:
            mean_return_pct = 0.0

    # Deployment % from best agent
    deployed_pct = 0.0
    if agent_pnl and best_agent in agent_pnl:
        cash_ratio = agent_pnl[best_agent].get("mean_cash_ratio", 1.0)
        deployed_pct = (1.0 - cash_ratio) * 100.0

    # Scorecard verdict
    try:
        from corp.agents.generation_scorer import score_generation
        prev_gens = generations[:-1] if len(generations) > 1 else []
        sc = score_generation(last, prev_gens)
        verdict = sc.get("verdict", "")
    except Exception:
        verdict = ""

    # Best eval score (always available)
    best_eval_score = 0.0
    if eval_scores:
        best_eval_score = max(eval_scores.values())

    return {
        "gen_num": last.get("generation", 0),
        "best_return_pct": round(best_return_pct or 0.0, 3),
        "mean_return_pct": round(mean_return_pct or 0.0, 3),
        "best_agent": best_agent,
        "pool_size": last.get("pool_size", 0),
        "deployed_pct": round(deployed_pct, 1),
        "severity": diag.get("severity", ""),
        "verdict": verdict,
        "best_eval": round(best_eval_score, 1),
        "has_real_pnl": bool(agent_pnl) or last.get("best_return_pct") is not None,
    }


def _compute_all_time_best(generations: list[dict], gen_history: list[dict]) -> dict:
    """Compute all-time best metrics across all generations.

    Tracks real P&L separately from eval scores. Never fakes P&L from
    reward values — instead provides both so the dashboard can display
    whichever is available.
    """
    if not generations:
        return {}

    best_return_pct = None  # Real P&L (from agent_pnl or best_return_pct)
    best_return_agent = ""
    best_return_gen = 0
    best_eval = None  # Eval reward score (always available)
    best_eval_agent = ""
    best_eval_gen = 0
    has_real_pnl = False
    profitable_gens = 0
    total_gens = len(generations)

    for g in generations:
        gen_num = g.get("generation", 0)
        agent_pnl = g.get("agent_pnl", {})
        eval_scores = g.get("eval_scores", {})
        ret_pct = g.get("best_return_pct")

        # Track best REAL return (only from actual P&L data)
        gen_real_return = None
        gen_real_agent = ""

        if agent_pnl:
            has_real_pnl = True
            top_name = max(agent_pnl, key=lambda a: agent_pnl[a].get("mean_return_pct", 0))
            gen_real_return = agent_pnl[top_name].get("mean_return_pct", 0.0)
            gen_real_agent = top_name
        elif ret_pct is not None:
            has_real_pnl = True
            gen_real_return = ret_pct
            if eval_scores:
                gen_real_agent = max(eval_scores, key=eval_scores.get)

        if gen_real_return is not None:
            if gen_real_return > 0:
                profitable_gens += 1
            if best_return_pct is None or gen_real_return > best_return_pct:
                best_return_pct = gen_real_return
                best_return_agent = gen_real_agent
                best_return_gen = gen_num

        # Track best EVAL score (always available)
        if eval_scores:
            top_name = max(eval_scores, key=eval_scores.get)
            top_score = eval_scores[top_name]
            if top_score > 0 and not has_real_pnl:
                profitable_gens += 1  # Count positive eval as "profitable" when no P&L
            if best_eval is None or top_score > best_eval:
                best_eval = top_score
                best_eval_agent = top_name
                best_eval_gen = gen_num

    # Use real P&L if available, otherwise leave as None (dashboard shows eval scores)
    return {
        "best_return_pct": round(best_return_pct, 3) if best_return_pct is not None else None,
        "best_agent": best_return_agent or friendly_name(best_eval_agent),
        "best_gen": best_return_gen or best_eval_gen,
        "has_real_pnl": has_real_pnl,
        "best_eval": round(best_eval or 0.0, 1),
        "best_eval_agent": friendly_name(best_eval_agent),
        "best_eval_gen": best_eval_gen,
        "profitable_gens": profitable_gens,
        "total_gens": total_gens,
    }


def _compute_scorecard(generations: list[dict]) -> dict:
    """Compute generation scorecard from the latest generation."""
    if not generations:
        return {}
    try:
        from corp.agents.generation_scorer import score_generation
        last = generations[-1]
        prev = generations[:-1] if len(generations) > 1 else []
        return score_generation(last, prev)
    except Exception:
        return {}


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

        # P&L and deployment data (if available from new tracking)
        agent_pnl = gen.get("agent_pnl", {})
        best_return_pct = gen.get("best_return_pct", None)
        mean_return_pct = gen.get("mean_return_pct", None)

        # Per-agent deployment and trade info
        agent_deployment = {}
        for name, pnl_data in agent_pnl.items():
            cash_ratio = pnl_data.get("mean_cash_ratio", 1.0)
            agent_deployment[name] = {
                "return_pct": round(pnl_data.get("mean_return_pct", 0.0), 3),
                "deployed_pct": round((1.0 - cash_ratio) * 100, 1),
                "cash_pct": round(cash_ratio * 100, 1),
            }

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
            "agent_eval_scores": {k: round(v, 1) for k, v in eval_scores.items()},
            # New P&L tracking fields
            "best_return_pct": round(best_return_pct, 3) if best_return_pct is not None else None,
            "mean_return_pct": round(mean_return_pct, 3) if mean_return_pct is not None else None,
            "agent_deployment": agent_deployment,
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

    # Deployment alerts — flag agents sitting on their hands
    for gen in generations:
        gen_num = gen.get("generation", 0)
        agent_pnl = gen.get("agent_pnl", {})
        for agent_name, pnl_data in agent_pnl.items():
            cash_ratio = pnl_data.get("mean_cash_ratio", 1.0)
            if cash_ratio > 0.9:
                alerts.append({
                    "type": "idle_agent",
                    "icon": "pause",
                    "color": "red",
                    "message": f"{friendly_name(agent_name)} is {cash_ratio:.0%} in cash — not trading",
                    "gen": gen_num,
                })
            elif cash_ratio > 0.6:
                alerts.append({
                    "type": "low_deployment",
                    "icon": "trending_down",
                    "color": "amber",
                    "message": f"{friendly_name(agent_name)} only {1.0 - cash_ratio:.0%} deployed — cautious",
                    "gen": gen_num,
                })

        # P&L alerts
        best_ret = gen.get("best_return_pct")
        if best_ret is not None:
            if best_ret > 0:
                best_name = ""
                if agent_pnl:
                    best_name = max(agent_pnl, key=lambda a: agent_pnl[a].get("mean_return_pct", 0))
                alerts.append({
                    "type": "profit",
                    "icon": "trending_up",
                    "color": "green",
                    "message": f"{friendly_name(best_name)} made money: {best_ret:+.3f}% return",
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


def _load_corp_recommendations(corp: dict | None) -> list[dict]:
    """Extract actionable recommendations from corp state decisions log."""
    recs = []
    if not corp:
        return recs

    # Decision log (JSONL)
    log_path = Path("logs/corporation_decisions.jsonl")
    if not log_path.exists():
        log_path = Path(__file__).parent.parent.parent / "logs" / "corporation_decisions.jsonl"

    if log_path.exists():
        try:
            with open(log_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    agent = entry.get("agent", "")
                    action = entry.get("action", "")
                    detail = entry.get("detail", {})
                    ts = entry.get("timestamp", "")

                    # Hedge fund director strategy memos
                    if agent == "hedge_fund_director" and action == "strategy_memo":
                        recs.append({
                            "source": "Hedge Fund Director",
                            "type": "strategy",
                            "message": detail.get("memo", ""),
                            "confidence": detail.get("confidence", 0),
                            "has_action": detail.get("has_patch", False),
                            "timestamp": ts,
                        })

                    # Contrarian reviews
                    elif agent == "contrarian" and action == "contrarian_review":
                        verdict = detail.get("verdict", "")
                        score = detail.get("fragility_score", 0)
                        concerns = detail.get("num_concerns", 0)
                        recs.append({
                            "source": "Contrarian Analyst",
                            "type": "risk_review",
                            "message": f"System verdict: {verdict} (fragility: {score:.0%}, {concerns} concern{'s' if concerns != 1 else ''})",
                            "confidence": 1.0 - score,
                            "has_action": False,
                            "timestamp": ts,
                        })

                    # Performance analyst
                    elif agent == "performance_analyst" and action == "performance_analysis":
                        recs.append({
                            "source": "Performance Analyst",
                            "type": "analysis",
                            "message": (
                                f"Analyzed {detail.get('generations_analyzed', 0)} generations. "
                                f"Pool concentration (Gini): {detail.get('pool_gini', 0):.2f}. "
                                f"{detail.get('num_recommendations', 0)} recommendation(s)."
                            ),
                            "confidence": 0.7,
                            "has_action": detail.get("num_recommendations", 0) > 0,
                            "timestamp": ts,
                        })

                    # Geopolitics regime
                    elif agent == "geopolitics_expert" and action == "regime_classification":
                        recs.append({
                            "source": "Geopolitics Expert",
                            "type": "regime",
                            "message": (
                                f"Market regime: {detail.get('regime', 'unknown')} "
                                f"(confidence: {detail.get('confidence', 0):.0%}, "
                                f"from {detail.get('headlines', 0)} headlines)"
                            ),
                            "confidence": detail.get("confidence", 0),
                            "has_action": False,
                            "timestamp": ts,
                        })

                    # Operations monitor alerts
                    elif agent == "operations_monitor" and action == "operations_scan":
                        health = detail.get("health_score", 1.0)
                        if health < 0.8:
                            recs.append({
                                "source": "Operations Monitor",
                                "type": "alert",
                                "message": (
                                    f"Gen {detail.get('generation', '?')}: "
                                    f"Health {health:.0%}, {detail.get('patterns_found', 0)} issues, "
                                    f"{detail.get('proposals', 0)} fix proposals"
                                ),
                                "confidence": health,
                                "has_action": detail.get("proposals", 0) > 0,
                                "timestamp": ts,
                            })

        except OSError:
            pass

    # Innovation briefs from corp state
    for brief in corp.get("innovation_briefs", []):
        # Deduplicate by tool name — only keep the latest
        pass

    # Deduplicate innovation briefs (they repeat every pipeline run)
    seen_tools = set()
    innovation = []
    for brief in reversed(corp.get("innovation_briefs", [])):
        name = brief.get("tool_name", "")
        if name not in seen_tools:
            seen_tools.add(name)
            innovation.append({
                "source": "Innovation Scout",
                "type": "tool_recommendation",
                "message": f"**{name}** ({brief.get('priority', 'medium')} priority) — {brief.get('summary', '')}",
                "confidence": brief.get("relevance_score", 0),
                "has_action": True,
                "timestamp": brief.get("submitted", ""),
            })
    recs.extend(innovation)

    # Deduplicate hedge fund director memos (keep only latest)
    seen_memos = set()
    deduped = []
    for r in reversed(recs):
        if r["source"] == "Hedge Fund Director":
            if r["message"] not in seen_memos:
                seen_memos.add(r["message"])
                deduped.append(r)
        elif r["source"] == "Geopolitics Expert":
            # Only keep the latest regime classification
            if "Geopolitics Expert" not in seen_memos:
                seen_memos.add("Geopolitics Expert")
                deduped.append(r)
        else:
            deduped.append(r)

    deduped.reverse()
    return deduped


def _load_geopolitics_ticker(corp: dict | None) -> dict[str, Any]:
    """Load geopolitics regime data for the news ticker.

    Pulls the current regime from corporation_state.json and the history
    of regime classifications from corporation_decisions.jsonl.
    """
    result: dict[str, Any] = {
        "current_regime": "",
        "confidence": 0.0,
        "volatility_outlook": "",
        "summary": "",
        "ticker_recs": {},
        "updated": "",
        "history": [],
    }

    # Current regime from corp state
    if corp:
        regime = corp.get("regime", {})
        result["current_regime"] = regime.get("classification", "")
        result["confidence"] = regime.get("confidence", 0.0)
        result["volatility_outlook"] = regime.get("volatility_outlook", "")
        result["summary"] = regime.get("summary", "")
        result["ticker_recs"] = regime.get("ticker_recommendations", {})
        result["updated"] = regime.get("updated", "")

    # Regime history from decisions log
    log_path = Path("logs/corporation_decisions.jsonl")
    if not log_path.exists():
        log_path = Path(__file__).parent.parent.parent / "logs" / "corporation_decisions.jsonl"

    if log_path.exists():
        try:
            with open(log_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("agent") == "geopolitics_expert" and entry.get("action") == "regime_classification":
                        detail = entry.get("detail", {})
                        result["history"].append({
                            "timestamp": entry.get("timestamp", ""),
                            "regime": detail.get("regime", ""),
                            "confidence": detail.get("confidence", 0.0),
                            "headlines": detail.get("headlines", 0),
                        })
        except OSError:
            pass

    return result


def _load_forward_test_data() -> dict[str, Any]:
    """Load forward test log and state for dashboard display."""
    result = {
        "active": False,
        "status": "inactive",
        "agents": [],
        "tickers": [],
        "started_at": "",
        "duration_days": 0,
        "days_elapsed": 0,
        "days_remaining": 0,
        "total_capital": 0,
        "allocations": {},
        "daily_snapshots": [],
        "agent_metrics": {},
        "events": [],
        "combined_equity": [],
    }

    # Read forward test log (JSONL)
    log_path = Path(FORWARD_TEST_LOG_FILE)
    if not log_path.exists():
        log_path = Path(__file__).parent.parent.parent / FORWARD_TEST_LOG_FILE
    if not log_path.exists():
        return result

    entries = []
    try:
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError:
        return result

    if not entries:
        return result

    # Parse start event
    start_event = None
    daily_snapshots = []
    events = []
    bar_data = {}  # agent_name -> list of portfolio values

    for entry in entries:
        entry_type = entry.get("type", "")
        if entry_type == "event":
            event_name = entry.get("event", "")
            detail = entry.get("detail", {})
            events.append(entry)

            if event_name == "forward_test_start":
                start_event = detail
            elif event_name == "emergency_halt":
                result["status"] = "halted"

        elif entry_type == "daily_snapshot":
            daily_snapshots.append(entry)

        elif entry_type == "bar":
            agent = entry.get("agent", "")
            pv = entry.get("portfolio_value", 0)
            if agent and pv:
                bar_data.setdefault(agent, []).append(pv)

    if not start_event:
        return result

    # Basic info from start event
    from datetime import datetime, timezone

    config = start_event.get("config", {})
    allocations = start_event.get("allocations", {})
    agents = start_event.get("agents", [])
    tickers = start_event.get("tickers", [])
    started_at = start_event.get("started_at", "")
    duration_days = config.get("duration_days", 60)

    # Compute days elapsed
    days_elapsed = 0
    if started_at:
        try:
            start_dt = datetime.fromisoformat(started_at)
            now = datetime.now()
            days_elapsed = (now - start_dt).days
        except (ValueError, TypeError):
            pass

    days_remaining = max(0, duration_days - days_elapsed)
    total_capital = sum(a.get("capital", 0) for a in allocations.values())

    result["active"] = True
    result["status"] = "waiting_for_market" if not daily_snapshots and not bar_data else "trading"
    result["agents"] = agents
    result["tickers"] = tickers
    result["started_at"] = started_at
    result["duration_days"] = duration_days
    result["days_elapsed"] = days_elapsed
    result["days_remaining"] = days_remaining
    result["total_capital"] = total_capital
    result["allocations"] = allocations
    result["daily_snapshots"] = daily_snapshots
    result["events"] = events

    # Read state file for more detailed metrics if available
    state = _load_json(FORWARD_TEST_STATE_FILE)
    if state:
        result["agent_metrics"] = state.get("agent_metrics", {})

    # Build per-agent equity curves from bar data
    if bar_data:
        result["combined_equity"] = []
        # Use the longest agent's data for x-axis
        max_len = max(len(v) for v in bar_data.values())
        for i in range(max_len):
            total = 0
            for agent_vals in bar_data.values():
                if i < len(agent_vals):
                    total += agent_vals[i]
            result["combined_equity"].append(total)

    return result


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
        "has_real_pnl": False,
        "lookback_days": 60,
        "pnl_per_day": 0.0,
        "pnl_per_month": 0.0,
        "pnl_per_year": 0.0,
        "best_eval_score": 0.0,
        "best_eval_agent": "",
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
        "ft_allocations": [],
        "benchmark": {"spy_return_pct": 0, "spy_drawdown_pct": 0, "spy_sharpe": 0},
        "updated": "N/A",
        "total_generations": 0,
        "target_generations": 0,
        "tickers": [],
        "num_stocks": 0,
        "real_data": False,
        "corp": {},
        "forward_test": _load_forward_test_data(),
        "current_gen": {},
        "all_time_best": {},
        "scorecard": {},
        "conviction": {},
        "geopolitics": {
            "current_regime": "", "confidence": 0.0, "volatility_outlook": "",
            "summary": "", "ticker_recs": {}, "updated": "", "history": [],
        },
    }
