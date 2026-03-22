#!/usr/bin/env python3
"""
HYDRACORP Executive Dashboard — Navy + Gold theme
http://localhost:5010

Shows everything:
- Training progress (generations, rewards, pool evolution)
- Generation scorecard with dimension bars
- CHIMERA diagnostics & mutations
- PROMETHEUS competition weights
- ELEOS conviction calibration
- ATHENA validation (PSR, DSR, bootstrap CI)
- Equity curves, trade signals, pipeline timing

Usage:
    python scripts/hydra_dashboard.py [--port PORT] [--no-browser]
"""

import http.server
import json
import os
import time
import threading
import webbrowser
import socketserver
import sys
from datetime import datetime
from pathlib import Path

HYDRA_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = HYDRA_ROOT / "logs"
STATE_FILE = LOGS_DIR / "hydra_training_state.json"
CORP_STATE_FILE = LOGS_DIR / "corporation_state.json"
PREV_RUN_FILE = LOGS_DIR / "previous_run_best.json"
PORT = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 5010
NO_BROWSER = "--no-browser" in sys.argv

# Import generation scorer
sys.path.insert(0, str(HYDRA_ROOT))
try:
    from corp.agents.generation_scorer import score_generation
except ImportError:
    score_generation = None


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _load_previous_run_best():
    """Load the previous run's best agent score history for chart reference."""
    if not PREV_RUN_FILE.exists():
        return None
    try:
        with open(PREV_RUN_FILE) as f:
            data = json.load(f)
        if data and "best_score_history" in data:
            return data
    except Exception:
        pass
    return None


def _save_previous_run_best(best_score_history, best_agent, num_generations):
    """Save current run's best scores as previous run reference.

    Called when a completed (non-live) run is detected and no previous
    run file exists yet, or when a new run starts (the old completed
    run becomes the reference).
    """
    if not best_score_history:
        return
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "best_score_history": best_score_history,
            "best_agent": best_agent,
            "num_generations": num_generations,
            "saved_at": datetime.now().isoformat(),
        }
        with open(PREV_RUN_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


# Track whether we've archived the previous run for this session
_prev_run_archived = False


def get_dashboard_data():
    state = _load_json(STATE_FILE)
    corp_state = _load_json(CORP_STATE_FILE)
    if not state:
        return {
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "waiting",
            "status_line": "No training data yet -- run: python scripts/train.py --real-data",
            "traffic_light": "yellow",
            "training_active": False,
            "training_progress": 0,
            "scorecard": None,
            "scorecard_history": [],
            "agent_sparklines": {},
        }

    config = state.get("config", {})
    summary = state.get("summary", {})
    metrics = state.get("metrics", {})
    generations = state.get("generations", [])
    validation = state.get("validation", {})
    eval_data = state.get("eval", {})
    tasks = state.get("tasks", {})

    total_gens = summary.get("total_generations", 0)
    target_gens = config.get("num_generations", total_gens) or 1
    passed = summary.get("passed_agents", [])
    rankings = summary.get("final_rankings", {})
    thresholds = summary.get("thresholds", {})

    # Training active / progress
    training_active = state.get("live", False)
    training_progress = min(100.0, (total_gens / target_gens * 100)) if target_gens else 0

    # Best agent — prefer rankings (post-validation), fall back to latest eval scores
    if rankings:
        best_agent = max(rankings, key=rankings.get)
        best_score = rankings.get(best_agent, 0)
    elif generations and generations[-1].get("eval_scores"):
        es = generations[-1]["eval_scores"]
        best_agent = max(es, key=es.get)
        best_score = es[best_agent]
    else:
        best_agent = "N/A"
        best_score = 0

    # Reward trend
    rewards = [g.get("train_mean_reward", 0) for g in generations]
    reward_improving = len(rewards) >= 2 and rewards[-1] > rewards[0]

    # Validation stats
    val_agents = list(validation.values()) if validation else []
    best_fitness = max((a.get("fitness_score", 0) for a in val_agents), default=0)
    best_sharpe = max((a.get("sharpe", 0) for a in val_agents), default=0)
    avg_wfe = sum(a.get("wfe", 0) for a in val_agents) / max(len(val_agents), 1)

    # Traffic light
    if passed:
        traffic_light = "green"
    elif reward_improving:
        traffic_light = "yellow"
    else:
        traffic_light = "red"

    # Status line
    if passed:
        status_line = (
            f"Training complete -- {len(passed)} agent(s) passed validation! "
            f"Best: {best_agent} (fitness={best_fitness:.3f})"
        )
    elif total_gens > 0:
        last_reward = rewards[-1] if rewards else 0
        delta = ""
        if len(rewards) >= 2:
            d = rewards[-1] - rewards[0]
            delta = f" ({'improving' if d > 0 else 'declining'}: {d:+.1f})"
        live_tag = "[LIVE] " if state.get("live") else ""
        progress = f" ({total_gens}/{target_gens})" if state.get("live") else ""
        status_line = (
            f"{live_tag}Training{progress} on {len(config.get('tickers', []))} "
            f"{'real' if config.get('real_data') else 'synthetic'} tickers -- "
            f"reward={last_reward:.1f}{delta}"
        )
    else:
        status_line = "No training data yet"

    # Diagnostics summary across generations
    diagnostics_history = []
    action_diagnostics = []
    for g in generations:
        diag = g.get("diagnosis")
        comp = g.get("competition")
        conv = g.get("conviction")
        diagnostics_history.append({
            "generation": g.get("generation", 0),
            "reward": g.get("train_mean_reward", 0),
            "pool_size": g.get("pool_size", 0),
            "promoted": g.get("promoted", []),
            "demoted": g.get("demoted", []),
            "eval_scores": g.get("eval_scores", {}),
            "severity": diag.get("severity", "N/A") if diag else "N/A",
            "primary_issue": diag.get("primary_issue", "") if diag else "",
            "num_mutations": diag.get("num_mutations", 0) if diag else 0,
            "weights_after": comp.get("weights_after", {}) if comp else {},
            "converged": comp.get("converged", False) if comp else False,
            "conviction": conv or {},
        })
        # Action diagnostics per generation
        action_info = g.get("action_diagnostics", {})
        action_diagnostics.append({
            "generation": g.get("generation", 0),
            "num_trades": g.get("num_trades", action_info.get("num_trades", 0)),
            "per_agent_mean_action": action_info.get("per_agent_mean_action", {}),
            "aggregated_mean_action": action_info.get("aggregated_mean_action", 0),
            "pre_aggregation_mean": action_info.get("pre_aggregation_mean", 0),
        })

    # Scorecard via generation_scorer
    scorecard = None
    scorecard_history = []
    if score_generation and generations:
        for idx, g in enumerate(generations):
            prev = generations[:idx]
            try:
                sc = score_generation(g, prev)
                scorecard_history.append(sc)
            except Exception:
                pass
        if scorecard_history:
            scorecard = scorecard_history[-1]

    # Agent sparklines: per-agent eval score history across generations
    agent_sparklines = {}
    for g in generations:
        es = g.get("eval_scores", {})
        for agent_name, score_val in es.items():
            if agent_name not in agent_sparklines:
                agent_sparklines[agent_name] = []
            agent_sparklines[agent_name].append(score_val)

    # Corporation state: regime, portfolio value
    regime_str = "Unknown"
    portfolio_value = 0.0
    verdict_str = scorecard.get("verdict", "N/A") if scorecard else "N/A"
    if corp_state:
        regime = corp_state.get("regime", {})
        regime_str = regime.get("classification", "Unknown")
        # Compute portfolio value from shadow results or proposals
        shadow = corp_state.get("shadow_results", {})
        if shadow and isinstance(shadow, dict):
            portfolio_value = shadow.get("portfolio_value", 0.0)
        if not portfolio_value:
            proposals = corp_state.get("proposals", [])
            if proposals and isinstance(proposals, list):
                for p in proposals:
                    if isinstance(p, dict):
                        portfolio_value += p.get("value", 0)

    # Pipeline timing
    pipeline_timing = []
    for name, info in tasks.items():
        pipeline_timing.append({
            "phase": name,
            "status": info.get("status", "unknown"),
            "duration_ms": info.get("duration_ms", 0),
        })

    # Latest generation conviction + eval scores (for live agent table)
    latest_conviction = {}
    latest_eval_scores = {}
    if generations:
        last_gen = generations[-1]
        latest_conviction = last_gen.get("conviction", {})
        latest_eval_scores = last_gen.get("eval_scores", {})

    # Benchmark + price history from state
    benchmark = state.get("benchmark", {})
    price_history = state.get("price_history", [])
    trade_signals = state.get("trade_signals", [])

    # Compute excess return for hero KPI
    best_total_return = max((a.get("total_return", 0) for a in val_agents), default=0)
    benchmark_return = benchmark.get("total_return", 0)
    excess_return = best_total_return - benchmark_return

    # Best score per generation (tracks improvement trajectory)
    best_score_history = []
    for g in generations:
        es = g.get("eval_scores", {})
        if es:
            best_score_history.append(max(es.values()))
        else:
            best_score_history.append(0)

    # Portfolio value and % change calculations
    initial_cash = config.get("initial_cash", 2500.0)
    # Score change: compare first gen best to latest gen best
    score_change_pct = 0.0
    first_best = best_score_history[0] if best_score_history else 0
    latest_best = best_score_history[-1] if best_score_history else 0
    if len(best_score_history) >= 2 and abs(first_best) > 1:
        score_change_pct = ((latest_best - first_best) / abs(first_best)) * 100

    # Mean reward change for portfolio trend
    mean_reward_change = 0.0
    if len(rewards) >= 2:
        mean_reward_change = rewards[-1] - rewards[0]

    # Previous run reference line: archive old run when new run starts
    global _prev_run_archived
    prev_run = _load_previous_run_best()
    is_live = state.get("live", False)

    if is_live and total_gens <= 2 and not _prev_run_archived and prev_run is None:
        # New run just started — archive the completed run's state if it exists
        # (the state file still has the old run's data at gen 1-2)
        pass  # Will be handled below when the completed run is detected

    if not is_live and best_score_history and not _prev_run_archived:
        # Run is complete — save as previous run reference for next run
        _save_previous_run_best(best_score_history, best_agent, total_gens)
        _prev_run_archived = True

    # Build previous run data for chart
    prev_run_best = None
    if prev_run and prev_run.get("best_score_history"):
        prev_run_best = prev_run["best_score_history"]

    return {
        "updated": state.get("updated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "status": "complete",
        "status_line": status_line,
        "traffic_light": traffic_light,
        "training_active": training_active,
        "training_progress": round(training_progress, 1),
        "config": config,
        "hero": {
            "total_generations": total_gens,
            "target_generations": target_gens,
            "best_fitness": round(best_fitness, 4),
            "best_sharpe": round(best_sharpe, 3),
            "avg_wfe": round(avg_wfe, 3),
            "agents_passed": len(passed),
            "total_agents": len(validation) if validation else len(latest_eval_scores),
            "reward_trend": rewards,
            "excess_return": round(excess_return, 4),
            "best_agent": best_agent,
            "best_score": round(best_score, 2),
            "best_score_history": best_score_history,
            "regime": regime_str,
            "verdict": verdict_str,
            "portfolio_value": round(portfolio_value, 2),
            "initial_cash": initial_cash,
            "score_change_pct": round(score_change_pct, 1),
            "mean_reward_change": round(mean_reward_change, 1),
            "prev_run_best": prev_run_best,
        },
        "scorecard": scorecard,
        "scorecard_history": scorecard_history,
        "agent_sparklines": agent_sparklines,
        "generations": diagnostics_history,
        "validation": {
            name: r for name, r in validation.items()
        } if validation else {},
        "rankings": rankings,
        "thresholds": thresholds,
        "passed_agents": passed,
        "metrics": metrics,
        "pipeline": pipeline_timing,
        "action_diagnostics": action_diagnostics,
        "benchmark": benchmark,
        "price_history": price_history,
        "trade_signals": trade_signals,
        "latest_conviction": {
            name: {
                "win_rate": c.get("overall_win_rate", 0),
                "trades": c.get("total_trades", 0),
                "wins": c.get("total_wins", 0),
                "losses": c.get("total_losses", 0),
            }
            for name, c in latest_conviction.items()
        },
        "latest_eval_scores": latest_eval_scores,
    }


# ── HTML Dashboard ──────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HYDRACORP Executive Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2.2.0/dist/chartjs-plugin-datalabels.min.js"></script>
<style>
:root {
  --navy: #1B2A4A;
  --navy-light: #243558;
  --navy-dark: #111D35;
  --gold: #C8A951;
  --gold-light: #D4BC72;
  --bg: #0B1120;
  --card: #131B2E;
  --card-border: #1E2A42;
  --text: #E8ECF2;
  --text-muted: #7B8BA5;
  --green: #22C55E;
  --red: #EF4444;
  --amber: #F59E0B;
  --chart-1: #C8A951;
  --chart-2: #3B82F6;
  --chart-3: #22C55E;
  --chart-4: #A78BFA;
  --chart-5: #F472B6;
  --chart-6: #FB923C;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 13px; line-height: 1.5; min-height: 100vh; }

/* ── Header ──────────────────────────────────────────────────────── */
#header {
  background: linear-gradient(135deg, #0D1525 0%, #162240 100%);
  color: #FFFFFF;
  padding: 18px 28px;
  border-bottom: 1px solid var(--gold);
  display: flex;
  align-items: center;
  justify-content: space-between;
}
#header-left { display: flex; align-items: center; gap: 16px; }
#header-left h1 {
  font-size: 18px;
  font-weight: 700;
  letter-spacing: 2px;
  color: var(--gold);
}
#header-right {
  font-size: 11px;
  color: rgba(255,255,255,0.5);
  text-align: right;
}
#header-right .ts-value { color: rgba(255,255,255,0.8); font-weight: 500; }

/* ── Training Progress Bar ───────────────────────────────────────── */
#training-progress-bar {
  height: 4px;
  background: var(--navy-dark);
  overflow: hidden;
  display: none;
}
#training-progress-bar.active { display: block; }
#training-progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--gold), var(--gold-light));
  transition: width 0.6s ease;
  position: relative;
}
#training-progress-fill::after {
  content: '';
  position: absolute;
  top: 0; right: 0;
  width: 60px; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
  animation: shimmer 1.8s infinite;
}
@keyframes shimmer { 0%{transform:translateX(-60px)} 100%{transform:translateX(60px)} }

/* ── Wrap ────────────────────────────────────────────────────────── */
.wrap { padding: 20px 28px; max-width: 1440px; margin: 0 auto; }

/* ── Card ────────────────────────────────────────────────────────── */
.card {
  background: var(--card);
  border: 1px solid var(--card-border);
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.3);
  overflow: hidden;
}
.card-header {
  padding: 16px 20px 12px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  color: var(--text-muted);
  border-bottom: 1px solid var(--card-border);
}
.card-body { padding: 16px 20px; }
.chart-wrap { padding: 12px 16px; height: 240px; position: relative; }

/* ── Grids ───────────────────────────────────────────────────────── */
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 16px; }
.grid-2-1 { display: grid; grid-template-columns: 2fr 1fr; gap: 16px; margin-bottom: 16px; }
.mb-16 { margin-bottom: 16px; }

/* ── KPI Strip ───────────────────────────────────────────────────── */
.kpi-strip {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 16px;
  margin-bottom: 20px;
}
.kpi-card {
  background: var(--card);
  border: 1px solid var(--card-border);
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.3);
  padding: 16px 20px;
  text-align: center;
}
.kpi-label {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: 6px;
}
.kpi-value {
  font-size: 26px;
  font-weight: 700;
  color: var(--gold);
  line-height: 1.2;
}
.kpi-sub {
  font-size: 10px;
  color: var(--text-muted);
  margin-top: 4px;
}

/* ── Verdict Badges ──────────────────────────────────────────────── */
.verdict-badge {
  display: inline-block;
  padding: 3px 12px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.3px;
}
.verdict-continue { background: rgba(34,197,94,0.15); color: #22C55E; }
.verdict-retune { background: rgba(245,158,11,0.15); color: #F59E0B; }
.verdict-halt { background: rgba(239,68,68,0.15); color: #EF4444; }

/* ── Scorecard ───────────────────────────────────────────────────── */
.scorecard-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
.dim-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 10px;
}
.dim-label {
  width: 140px;
  font-size: 11px;
  font-weight: 500;
  color: var(--text-muted);
  flex-shrink: 0;
}
.dim-bar-track {
  flex: 1;
  height: 10px;
  background: #1A2338;
  border-radius: 5px;
  overflow: hidden;
}
.dim-bar-fill {
  height: 100%;
  border-radius: 5px;
  transition: width 0.5s ease;
}
.dim-score {
  width: 32px;
  font-size: 12px;
  font-weight: 600;
  text-align: right;
  flex-shrink: 0;
}
.overall-score {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 140px;
}
.overall-number {
  font-size: 64px;
  font-weight: 700;
  color: var(--gold);
  line-height: 1;
}
.overall-label {
  font-size: 13px;
  color: var(--text-muted);
  font-weight: 500;
  margin-top: 4px;
}
.overall-of { font-size: 22px; font-weight: 400; color: var(--text-muted); }
.gap-list { margin-top: 12px; }
.gap-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 0;
  border-bottom: 1px solid var(--card-border);
  font-size: 12px;
}
.gap-item:last-child { border-bottom: none; }
.gap-priority {
  display: inline-block;
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.3px;
  flex-shrink: 0;
}
.gap-high { background: rgba(239,68,68,0.15); color: #EF4444; }
.gap-medium { background: rgba(245,158,11,0.15); color: #F59E0B; }
.gap-low { background: rgba(34,197,94,0.15); color: #22C55E; }

/* ── Tables ──────────────────────────────────────────────────────── */
table { width: 100%; border-collapse: collapse; }
th {
  text-align: left;
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-muted);
  padding: 10px 14px;
  border-bottom: 2px solid var(--card-border);
  white-space: nowrap;
}
td {
  padding: 8px 14px;
  border-bottom: 1px solid var(--card-border);
  white-space: nowrap;
  font-size: 12px;
}
tr:last-child td { border-bottom: none; }
tr:hover td { background: rgba(200,169,81,0.05); }
.num { text-align: right; }
.clr-green { color: var(--green); }
.clr-red { color: var(--red); }
.clr-amber { color: var(--amber); }
.clr-navy { color: var(--gold-light); }
.clr-muted { color: var(--text-muted); }
.clr-gold { color: var(--gold); }
.fw-600 { font-weight: 600; }

/* ── Pixel Hydra ─────────────────────────────────────────────────── */
#hydra-container {
  width: 224px;
  height: 112px;
  display: grid;
  grid-template-columns: repeat(32, 7px);
  grid-template-rows: repeat(16, 7px);
  flex-shrink: 0;
}
#hydra-container .px {
  width: 7px; height: 7px;
  border-radius: 1px;
}
.px-navy { background: var(--navy); }
.px-navy-light { background: var(--navy-light); }
.px-navy-dark { background: var(--navy-dark); }
.px-gold { background: var(--gold); }
.px-gold-light { background: var(--gold-light); }
.px-eye { background: var(--gold); }
.px-bg { background: transparent; }

/* Hydra animations — active (training) */
#hydra-container.training-active .head-1 {
  animation: bob1 1.8s ease-in-out infinite;
}
#hydra-container.training-active .head-2 {
  animation: bob2 2.0s ease-in-out infinite;
}
#hydra-container.training-active .head-3 {
  animation: bob3 2.5s ease-in-out infinite;
}
#hydra-container.training-active .body-px {
  animation: sway 3s ease-in-out infinite;
}
#hydra-container.training-active .eye-px {
  animation: blink 4s step-end infinite;
}

/* Hydra animations — sleeping (not training) */
#hydra-container:not(.training-active) .px {
  animation: sleepPulse 5s ease-in-out infinite;
}

@keyframes bob1 {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-2px); }
}
@keyframes bob2 {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-3px); }
}
@keyframes bob3 {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-2px); }
}
@keyframes sway {
  0%, 100% { transform: translateX(0); }
  50% { transform: translateX(1px); }
}
@keyframes blink {
  0%, 90%, 100% { opacity: 1; }
  92%, 98% { opacity: 0.1; }
}
@keyframes sleepPulse {
  0%, 100% { opacity: 0.4; }
  50% { opacity: 0.7; }
}

/* ── Empty state ─────────────────────────────────────────────────── */
.empty {
  padding: 32px;
  text-align: center;
  color: var(--text-muted);
  font-size: 12px;
}

/* ── Footer ──────────────────────────────────────────────────────── */
#footer {
  text-align: center;
  color: var(--text-muted);
  font-size: 10px;
  padding: 14px;
  border-top: 1px solid var(--card-border);
  letter-spacing: 0.3px;
}

/* ── Dark theme positive/negative indicators ─────────────────────── */
.indicator-up { color: var(--green); font-weight: 600; }
.indicator-down { color: var(--red); font-weight: 600; }
.indicator-flat { color: var(--text-muted); font-weight: 600; }
</style>
</head>
<body>

<!-- ═══ HEADER ═══════════════════════════════════════════════════════ -->
<div id="header">
  <div id="header-left">
    <h1>HYDRACORP</h1>
    <div id="hydra-container"></div>
  </div>
  <div id="header-right">
    Last updated<br>
    <span class="ts-value" id="ts">--</span>
  </div>
</div>

<!-- ═══ TRAINING PROGRESS BAR ════════════════════════════════════════ -->
<div id="training-progress-bar">
  <div id="training-progress-fill" style="width:0%"></div>
</div>

<div class="wrap">

  <!-- ═══ KPI STRIP ════════════════════════════════════════════════════ -->
  <div class="kpi-strip">
    <div class="kpi-card">
      <div class="kpi-label">Portfolio Value</div>
      <div class="kpi-value" id="kpi-portfolio">--</div>
      <div class="kpi-sub" id="kpi-portfolio-sub">&nbsp;</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Best Agent</div>
      <div class="kpi-value" id="kpi-best-agent">--</div>
      <div class="kpi-sub" id="kpi-best-agent-sub">&nbsp;</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Score Change</div>
      <div class="kpi-value" id="kpi-change">--</div>
      <div class="kpi-sub" id="kpi-change-sub">&nbsp;</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Win Rate</div>
      <div class="kpi-value" id="kpi-winrate">--</div>
      <div class="kpi-sub" id="kpi-winrate-sub">&nbsp;</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Generation</div>
      <div class="kpi-value" id="kpi-gen">--</div>
      <div class="kpi-sub" id="kpi-gen-sub">&nbsp;</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Verdict</div>
      <div class="kpi-value" id="kpi-verdict">--</div>
      <div class="kpi-sub" id="kpi-verdict-sub">&nbsp;</div>
    </div>
  </div>

  <!-- ═══ GENERATION SCORECARD ═════════════════════════════════════════ -->
  <div class="card mb-16" id="scorecard-card">
    <div class="card-header">Generation Scorecard</div>
    <div class="card-body" id="scorecard-body">
      <div class="empty">No scorecard data yet.</div>
    </div>
  </div>

  <!-- ═══ CHARTS ROW 1: Reward Trend + Agent Weights ═══════════════════ -->
  <div class="grid-2">
    <div class="card">
      <div class="card-header">Performance: Mean vs Best Agent</div>
      <div class="chart-wrap"><canvas id="reward-chart"></canvas></div>
    </div>
    <div class="card">
      <div class="card-header">Agent Competition Weights</div>
      <div class="chart-wrap"><canvas id="weights-chart"></canvas></div>
    </div>
  </div>

  <!-- ═══ AGENT PERFORMANCE TABLE ══════════════════════════════════════ -->
  <div class="card mb-16">
    <div class="card-header">Agent Performance</div>
    <div id="agent-table-body"><div class="empty">No agent data.</div></div>
  </div>

  <!-- ═══ CHARTS ROW 2: Equity Curves + Trade Count ════════════════════ -->
  <div class="grid-2">
    <div class="card">
      <div class="card-header">Equity Curves</div>
      <div class="chart-wrap" style="height:260px"><canvas id="equity-chart"></canvas></div>
    </div>
    <div class="card">
      <div class="card-header">Trade Count by Generation</div>
      <div class="chart-wrap" style="height:260px"><canvas id="trades-chart"></canvas></div>
    </div>
  </div>

  <!-- ═══ PRICE + SIGNALS ══════════════════════════════════════════════ -->
  <div class="card mb-16">
    <div class="card-header">Price Trend + Trade Signals</div>
    <div class="chart-wrap" style="height:300px"><canvas id="price-chart"></canvas></div>
  </div>

  <!-- ═══ BOTTOM ROW: Pipeline Timing + Config ═════════════════════════ -->
  <div class="grid-2">
    <div class="card">
      <div class="card-header">Pipeline Timing</div>
      <div id="pipeline-body"><div class="empty">No pipeline data.</div></div>
    </div>
    <div class="card">
      <div class="card-header">Training Configuration</div>
      <div id="config-body"><div class="empty">No configuration data.</div></div>
    </div>
  </div>

</div>

<div id="footer">HYDRACORP Executive Dashboard &mdash; CHIMERA + PROMETHEUS + ELEOS + ATHENA + KRONOS</div>

<script>
// ═══════════════════════════════════════════════════════════════════
// Chart.js Defaults
// ═══════════════════════════════════════════════════════════════════
Chart.defaults.color = '#7B8BA5';
Chart.defaults.borderColor = '#1E2A42';
Chart.defaults.font.family = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.display = false;
Chart.defaults.animation = { duration: 400 };
Chart.register(ChartDataLabels);

// Color constants
const NAVY = '#1B2A4A';
const NAVY_LIGHT = '#243558';
const GOLD = '#C8A951';
const GOLD_LIGHT = '#D4BC72';
const GREEN = '#22C55E';
const RED = '#EF4444';
const AMBER = '#F59E0B';
const CHART_COLORS = ['#C8A951','#3B82F6','#22C55E','#A78BFA','#F472B6','#FB923C','#06B6D4','#E879F9'];

// Chart instances
let rewardChart = null, weightsChart = null, tradesChart = null;
let equityChart = null, priceChart = null;

// ═══════════════════════════════════════════════════════════════════
// Pixel Hydra
// ═══════════════════════════════════════════════════════════════════
function buildHydra(trainingActive) {
  const c = document.getElementById('hydra-container');
  c.innerHTML = '';
  if (trainingActive) {
    c.classList.add('training-active');
  } else {
    c.classList.remove('training-active');
  }

  // 32x16 grid. Coordinates are [row][col], 0-indexed.
  // pixel classes: 'navy','navy-light','navy-dark','gold','gold-light','eye','bg'
  const grid = [];
  for (let r = 0; r < 16; r++) {
    grid[r] = [];
    for (let c = 0; c < 32; c++) grid[r][c] = { cls: 'bg', group: '' };
  }

  function set(r, col, cls, group) {
    if (r >= 0 && r < 16 && col >= 0 && col < 32) {
      grid[r][col] = { cls: cls, group: group };
    }
  }

  // Head 1 (left) — 3x3 at row 0, col 6
  for (let dr = 0; dr < 3; dr++) for (let dc = 0; dc < 3; dc++) set(dr, 6+dc, 'navy', 'head-1');
  set(0, 7, 'navy-dark', 'head-1');
  set(1, 7, 'eye', 'head-1 eye-px');

  // Head 2 (center) — 3x3 at row 0, col 14
  for (let dr = 0; dr < 3; dr++) for (let dc = 0; dc < 3; dc++) set(dr, 14+dc, 'navy', 'head-2');
  set(0, 15, 'navy-dark', 'head-2');
  set(1, 15, 'eye', 'head-2 eye-px');

  // Head 3 (right) — 3x3 at row 0, col 22
  for (let dr = 0; dr < 3; dr++) for (let dc = 0; dc < 3; dc++) set(dr, 22+dc, 'navy', 'head-3');
  set(0, 23, 'navy-dark', 'head-3');
  set(1, 23, 'eye', 'head-3 eye-px');

  // Necks — connect heads to body
  // Neck 1: col 7, rows 3-5
  for (let r = 3; r <= 5; r++) { set(r, 7, 'navy', 'head-1'); set(r, 8, 'navy-light', 'head-1'); }
  // Neck 2: col 15, rows 3-5
  for (let r = 3; r <= 5; r++) { set(r, 15, 'navy', 'head-2'); set(r, 14, 'navy-light', 'head-2'); }
  // Neck 3: col 23, rows 3-4
  for (let r = 3; r <= 4; r++) { set(r, 23, 'navy', 'head-3'); set(r, 22, 'navy-light', 'head-3'); }

  // Body mass — rows 5-11, cols 6-25
  for (let r = 5; r <= 11; r++) {
    for (let col = 6; col <= 25; col++) {
      set(r, col, 'navy', 'body-px');
    }
  }
  // Gold belly accents — rows 9-10, cols 10-21
  for (let col = 10; col <= 21; col++) {
    set(9, col, 'gold', 'body-px');
    set(10, col, 'gold-light', 'body-px');
  }
  // Navy-light shading on top of body
  for (let col = 8; col <= 23; col++) {
    set(6, col, 'navy-light', 'body-px');
  }

  // Legs — rows 12-13
  // Left leg
  set(12, 8, 'navy', 'body-px'); set(12, 9, 'navy', 'body-px');
  set(13, 7, 'navy-dark', 'body-px'); set(13, 8, 'navy-dark', 'body-px');
  // Right leg
  set(12, 22, 'navy', 'body-px'); set(12, 23, 'navy', 'body-px');
  set(13, 23, 'navy-dark', 'body-px'); set(13, 24, 'navy-dark', 'body-px');

  // Tail — rows 10-13, curving left from col 5 down
  set(10, 5, 'navy', 'tail'); set(11, 4, 'navy', 'tail'); set(11, 5, 'navy', 'tail');
  set(12, 3, 'navy-light', 'tail'); set(12, 4, 'navy', 'tail');
  set(13, 2, 'navy-light', 'tail'); set(13, 3, 'navy-light', 'tail');
  set(14, 1, 'gold', 'tail'); set(14, 2, 'navy-light', 'tail');

  // Render
  for (let r = 0; r < 16; r++) {
    for (let col = 0; col < 32; col++) {
      const px = document.createElement('div');
      const g = grid[r][col];
      px.className = 'px px-' + g.cls + (g.group ? ' ' + g.group : '');
      c.appendChild(px);
    }
  }
}

// ═══════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════
function fmtMoney(v) {
  if (!v && v !== 0) return '--';
  return '$' + Number(v).toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 });
}

function fmtNum(v, dec) {
  if (v === null || v === undefined || v === 'N/A') return '<span class="clr-muted">--</span>';
  const n = parseFloat(v);
  if (isNaN(n)) return '<span class="clr-muted">--</span>';
  const sign = n > 0 ? '+' : '';
  const cls = n > 0 ? 'clr-green' : n < 0 ? 'clr-red' : 'clr-muted';
  return '<span class="' + cls + '">' + sign + n.toFixed(dec) + '</span>';
}

function fmtPct(v) {
  if (v === null || v === undefined) return '<span class="clr-muted">--</span>';
  const n = parseFloat(v) * 100;
  if (isNaN(n)) return '<span class="clr-muted">--</span>';
  const cls = n >= 50 ? 'clr-green' : n >= 35 ? 'clr-amber' : 'clr-red';
  return '<span class="' + cls + '">' + n.toFixed(1) + '%</span>';
}

function dimColor(score) {
  if (score >= 4) return GREEN;
  if (score >= 3) return GOLD;
  return RED;
}

function verdictClass(v) {
  if (!v) return '';
  const vl = v.toUpperCase();
  if (vl === 'CONTINUE') return 'verdict-continue';
  if (vl === 'RETUNE') return 'verdict-retune';
  if (vl === 'HALT') return 'verdict-halt';
  return '';
}

// ═══════════════════════════════════════════════════════════════════
// KPI Strip
// ═══════════════════════════════════════════════════════════════════
function updateKPIs(d) {
  const hero = d.hero || {};
  const conv = d.latest_conviction || {};
  const evalScores = d.latest_eval_scores || {};
  const gens = d.generations || [];

  // 1. Portfolio Value — starting capital + mean reward change indicator
  var initialCash = hero.initial_cash || 2500;
  var portfolioEl = document.getElementById('kpi-portfolio');
  portfolioEl.textContent = '$' + initialCash.toLocaleString();
  var mrc = hero.mean_reward_change || 0;
  var mrcCls = mrc > 0 ? 'indicator-up' : mrc < 0 ? 'indicator-down' : 'indicator-flat';
  var mrcArrow = mrc > 0 ? '\u25B2' : mrc < 0 ? '\u25BC' : '';
  document.getElementById('kpi-portfolio-sub').innerHTML =
    'Reward trend: <span class="' + mrcCls + '">' + mrcArrow + ' ' + (mrc > 0 ? '+' : '') + mrc.toFixed(0) + '</span>';

  // 2. Best Agent — score with +/- color, name as subtitle
  var bs = hero.best_score || 0;
  var ba = hero.best_agent || 'N/A';
  var bestEl = document.getElementById('kpi-best-agent');
  if (bs) {
    var bCls = bs > 0 ? 'indicator-up' : 'indicator-down';
    bestEl.innerHTML = '<span class="' + bCls + '">' + (bs > 0 ? '+' : '') + bs.toFixed(1) + '</span>';
  } else {
    bestEl.textContent = '--';
  }
  var baShort = ba.length > 18 ? ba.substring(0, 16) + '..' : ba;
  document.getElementById('kpi-best-agent-sub').textContent = baShort;

  // 3. Score Change — % improvement from gen 1 to latest
  var changePct = hero.score_change_pct || 0;
  var changeEl = document.getElementById('kpi-change');
  var chCls = changePct > 0 ? 'indicator-up' : changePct < 0 ? 'indicator-down' : 'indicator-flat';
  var chArrow = changePct > 0 ? '\u25B2' : changePct < 0 ? '\u25BC' : '';
  changeEl.innerHTML = '<span class="' + chCls + '">' + chArrow + ' ' + (changePct > 0 ? '+' : '') + changePct.toFixed(1) + '%</span>';
  var bsh = hero.best_score_history || [];
  if (bsh.length >= 2) {
    document.getElementById('kpi-change-sub').textContent =
      'Gen 1: ' + bsh[0].toFixed(0) + ' \u2192 Now: ' + bsh[bsh.length-1].toFixed(0);
  } else {
    document.getElementById('kpi-change-sub').textContent = 'vs first generation';
  }

  // 4. Win Rate — aggregate from conviction
  var totalWins = 0, totalLosses = 0;
  for (var agent in conv) {
    totalWins += (conv[agent].wins || 0);
    totalLosses += (conv[agent].losses || 0);
  }
  var totalGames = totalWins + totalLosses;
  var winPct = totalGames > 0 ? (totalWins / totalGames * 100) : 0;
  var wrEl = document.getElementById('kpi-winrate');
  var wrCls = winPct >= 50 ? 'indicator-up' : winPct >= 30 ? 'indicator-flat' : 'indicator-down';
  wrEl.innerHTML = '<span class="' + wrCls + '">' + winPct.toFixed(1) + '%</span>';
  document.getElementById('kpi-winrate-sub').textContent = totalWins + 'W / ' + totalLosses + 'L (' + totalGames + ' total)';

  // 5. Generation — progress
  var tg = hero.total_generations || 0;
  var target = hero.target_generations || 0;
  var pct = target > 0 ? ((tg / target) * 100).toFixed(0) : 0;
  document.getElementById('kpi-gen').textContent = tg + '/' + target;
  document.getElementById('kpi-gen-sub').textContent = d.training_active ? pct + '% complete' : 'Finished';

  // 6. Verdict — with regime subtitle
  var verdict = hero.verdict || 'N/A';
  var vEl = document.getElementById('kpi-verdict');
  vEl.innerHTML = '<span class="verdict-badge ' + verdictClass(verdict) + '">' + verdict + '</span>';
  var regime = hero.regime || 'Unknown';
  document.getElementById('kpi-verdict-sub').textContent = regime;
}

// ═══════════════════════════════════════════════════════════════════
// Scorecard
// ═══════════════════════════════════════════════════════════════════
function updateScorecard(scorecard) {
  const el = document.getElementById('scorecard-body');
  if (!scorecard) {
    el.innerHTML = '<div class="empty">No scorecard data yet.</div>';
    return;
  }

  const dims = scorecard.dimension_scores || {};
  const overall = scorecard.overall || 0;
  const verdict = scorecard.verdict || 'N/A';
  const gaps = scorecard.critical_gaps || [];

  const dimLabels = {
    'reward_trend': 'Reward Trend',
    'top_agent_quality': 'Top Agent Quality',
    'pool_diversity': 'Pool Diversity',
    'positive_rate': 'Positive Rate',
    'conviction_strength': 'Conviction Strength',
    'stability': 'Stability'
  };

  let leftHtml = '';
  for (const [key, label] of Object.entries(dimLabels)) {
    const score = dims[key] || 0;
    const pct = (score / 5 * 100).toFixed(0);
    const color = dimColor(score);
    leftHtml += '<div class="dim-row">' +
      '<div class="dim-label">' + label + '</div>' +
      '<div class="dim-bar-track"><div class="dim-bar-fill" style="width:' + pct + '%;background:' + color + '"></div></div>' +
      '<div class="dim-score" style="color:' + color + '">' + score + '/5</div>' +
      '</div>';
  }

  let rightHtml = '<div class="overall-score">' +
    '<div class="overall-number">' + overall + '<span class="overall-of">/10</span></div>' +
    '<div class="overall-label">Overall Score</div>' +
    '<div style="margin-top:8px"><span class="verdict-badge ' + verdictClass(verdict) + '">' + verdict + '</span></div>' +
    '</div>';

  if (gaps.length > 0) {
    rightHtml += '<div class="gap-list">';
    for (const g of gaps) {
      const pcls = g.priority === 'HIGH' ? 'gap-high' : g.priority === 'MEDIUM' ? 'gap-medium' : 'gap-low';
      rightHtml += '<div class="gap-item">' +
        '<span class="gap-priority ' + pcls + '">' + g.priority + '</span>' +
        '<span>' + g.title + '</span>' +
        '</div>';
    }
    rightHtml += '</div>';
  }

  el.innerHTML = '<div class="scorecard-grid"><div>' + leftHtml + '</div><div>' + rightHtml + '</div></div>';
}

// ═══════════════════════════════════════════════════════════════════
// Reward Trend Chart
// ═══════════════════════════════════════════════════════════════════
function updateRewardChart(rewards, bestScores, prevRunBest) {
  if (rewardChart) { rewardChart.destroy(); rewardChart = null; }
  const el = document.getElementById('reward-chart');
  if (!rewards || rewards.length === 0) return;

  const labels = rewards.map(function(_, i) { return 'Gen ' + (i + 1); });
  const improving = rewards[rewards.length - 1] > rewards[0];

  var datasets = [{
    label: 'Mean Reward',
    data: rewards,
    borderColor: improving ? GREEN : RED,
    backgroundColor: improving ? 'rgba(34,197,94,0.08)' : 'rgba(239,68,68,0.08)',
    borderWidth: 2,
    fill: true,
    tension: 0.3,
    pointRadius: 0,
    datalabels: {
      display: function(ctx) {
        return ctx.dataIndex === ctx.dataset.data.length - 1;
      },
      color: improving ? GREEN : RED,
      anchor: 'end', align: 'left',
      font: { size: 10, weight: 600 },
      formatter: function(v) { return 'Avg ' + v.toFixed(0); }
    }
  }];

  if (bestScores && bestScores.length > 0) {
    var bestImproving = bestScores[bestScores.length - 1] > bestScores[0];
    datasets.push({
      label: 'Best Agent',
      data: bestScores,
      borderColor: GOLD,
      backgroundColor: 'rgba(200,169,81,0.08)',
      borderWidth: 2,
      borderDash: [5, 3],
      fill: true,
      tension: 0.3,
      pointRadius: 0,
      datalabels: {
        display: function(ctx) {
          return ctx.dataIndex === ctx.dataset.data.length - 1;
        },
        color: GOLD,
        anchor: 'end', align: 'left',
        font: { size: 10, weight: 600 },
        formatter: function(v) { return 'Best ' + v.toFixed(0); }
      }
    });
  }

  // Previous run's best agent line (reference from last completed run)
  if (prevRunBest && prevRunBest.length > 0) {
    // Pad or trim to match current run's generation count
    var prevData = [];
    for (var i = 0; i < labels.length; i++) {
      if (i < prevRunBest.length) {
        prevData.push(prevRunBest[i]);
      } else {
        // Extend with last known value as flat reference
        prevData.push(prevRunBest[prevRunBest.length - 1]);
      }
    }
    datasets.push({
      label: 'Prev Run Best',
      data: prevData,
      borderColor: '#6B7FA3',
      backgroundColor: 'rgba(107,127,163,0.05)',
      borderWidth: 1.5,
      borderDash: [8, 4],
      fill: false,
      tension: 0.3,
      pointRadius: 0,
      datalabels: {
        display: function(ctx) {
          return ctx.dataIndex === ctx.dataset.data.length - 1;
        },
        color: '#6B7FA3',
        anchor: 'end', align: 'left',
        font: { size: 9, weight: 500 },
        formatter: function(v) { return 'Prev ' + v.toFixed(0); }
      }
    });
  }

  // Zero line reference
  datasets.push({
    label: 'Breakeven',
    data: rewards.map(function() { return 0; }),
    borderColor: 'rgba(123,139,165,0.3)',
    borderWidth: 1,
    borderDash: [3, 3],
    pointRadius: 0,
    fill: false,
    datalabels: { display: false }
  });

  rewardChart = new Chart(el.getContext('2d'), {
    type: 'line',
    data: { labels: labels, datasets: datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { color: '#1A2338' }, ticks: { font: { size: 10 } } },
        y: { grid: { color: '#1A2338' }, ticks: { font: { size: 10 }, callback: function(v) { return v.toFixed(0); } } }
      },
      plugins: {
        legend: { display: true, position: 'top', labels: { color: '#7B8BA5', boxWidth: 12, font: { size: 10 } } },
        tooltip: { mode: 'index', intersect: false }
      }
    }
  });
}

// ═══════════════════════════════════════════════════════════════════
// Agent Weights Chart (stacked bar)
// ═══════════════════════════════════════════════════════════════════
function updateWeightsChart(generations) {
  if (weightsChart) { weightsChart.destroy(); weightsChart = null; }
  const el = document.getElementById('weights-chart');
  if (!generations || generations.length === 0) return;

  var allAgents = {};
  for (var i = 0; i < generations.length; i++) {
    var wa = generations[i].weights_after || {};
    for (var a in wa) { allAgents[a] = true; }
  }
  var agents = Object.keys(allAgents).sort();
  if (agents.length === 0) return;

  var labels = generations.map(function(g) { return 'Gen ' + g.generation; });
  var datasets = agents.map(function(a, idx) {
    return {
      label: a,
      data: generations.map(function(g) { return (g.weights_after || {})[a] || 0; }),
      backgroundColor: CHART_COLORS[idx % CHART_COLORS.length] + 'A0',
      borderColor: CHART_COLORS[idx % CHART_COLORS.length],
      borderWidth: 1,
      datalabels: {
        display: function(ctx) {
          return ctx.dataset.data[ctx.dataIndex] > 0.15;
        },
        color: '#FFFFFF',
        font: { size: 9, weight: 600 },
        formatter: function(v) { return (v * 100).toFixed(0) + '%'; }
      }
    };
  });

  weightsChart = new Chart(el.getContext('2d'), {
    type: 'bar',
    data: { labels: labels, datasets: datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { stacked: true, grid: { color: '#1A2338' }, ticks: { font: { size: 10 } } },
        y: { stacked: true, grid: { color: '#1A2338' }, max: 1, ticks: { font: { size: 10 }, callback: function(v) { return (v * 100).toFixed(0) + '%'; } } }
      },
      plugins: { legend: { display: false }, tooltip: { enabled: true } }
    }
  });
}

// ═══════════════════════════════════════════════════════════════════
// Agent Performance Table with Sparklines
// ═══════════════════════════════════════════════════════════════════
function renderAgentTable(d) {
  var el = document.getElementById('agent-table-body');
  var evalScores = d.latest_eval_scores || {};
  var conviction = d.latest_conviction || {};
  var validation = d.validation || {};
  var sparklines = d.agent_sparklines || {};

  // Merge all known agent names from live data
  var agentNames = {};
  for (var n in evalScores) agentNames[n] = true;
  for (var n in conviction) agentNames[n] = true;
  for (var n in validation) agentNames[n] = true;
  for (var n in sparklines) agentNames[n] = true;

  var names = Object.keys(agentNames);
  // Sort by eval score descending
  names.sort(function(a, b) { return (evalScores[b] || -9999) - (evalScores[a] || -9999); });

  if (names.length === 0) {
    el.innerHTML = '<div class="empty">No agent data.</div>';
    return;
  }

  var html = '<table><thead><tr>' +
    '<th>Agent</th><th class="num">Eval Score</th><th class="num">Win Rate</th>' +
    '<th class="num">Gens</th><th class="num">W/L</th><th>Trend</th>' +
    '</tr></thead><tbody>';

  for (var i = 0; i < names.length; i++) {
    var name = names[i];
    var score = evalScores[name];
    var conv = conviction[name] || {};
    var val = validation[name] || {};
    // Prefer live conviction data, fall back to validation
    var wr = conv.win_rate != null ? conv.win_rate : val.win_rate;
    var trades = conv.trades || val.total_trades || 0;
    var wins = conv.wins || 0;
    var losses = conv.losses || 0;
    var sparkData = sparklines[name] || [];
    var scoreClass = score != null && score > 0 ? ' style="color:#15803D"' : '';

    html += '<tr>' +
      '<td class="fw-600 clr-navy">' + name + '</td>' +
      '<td class="num"' + scoreClass + '>' + fmtNum(score, 1) + '</td>' +
      '<td class="num">' + fmtPct(wr) + '</td>' +
      '<td class="num clr-muted">' + trades + '</td>' +
      '<td class="num clr-muted">' + wins + '/' + losses + '</td>' +
      '<td><canvas id="spark-' + i + '" width="60" height="20" data-spark=\'' + JSON.stringify(sparkData) + '\'></canvas></td>' +
      '</tr>';
  }

  el.innerHTML = html + '</tbody></table>';

  // Draw sparklines
  for (var i = 0; i < names.length; i++) {
    var canvas = document.getElementById('spark-' + i);
    if (!canvas) continue;
    var data;
    try { data = JSON.parse(canvas.getAttribute('data-spark')); } catch(e) { continue; }
    if (!data || data.length < 2) continue;
    drawSparkline(canvas, data);
  }
}

function drawSparkline(canvas, data) {
  var ctx = canvas.getContext('2d');
  var w = canvas.width;
  var h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  var min = Math.min.apply(null, data);
  var max = Math.max.apply(null, data);
  var range = max - min || 1;
  var trending = data[data.length - 1] >= data[0];
  var color = trending ? GREEN : RED;

  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;

  for (var i = 0; i < data.length; i++) {
    var x = (i / (data.length - 1)) * w;
    var y = h - ((data[i] - min) / range) * (h - 4) - 2;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

// ═══════════════════════════════════════════════════════════════════
// Equity Curves Chart
// ═══════════════════════════════════════════════════════════════════
function updateEquityChart(benchmark, validation) {
  if (equityChart) { equityChart.destroy(); equityChart = null; }
  var el = document.getElementById('equity-chart');
  if (!benchmark || !benchmark.equity_curve || benchmark.equity_curve.length === 0) return;

  var spyEquity = benchmark.equity_curve;
  var labels = spyEquity.map(function(_, i) { return i; });

  // Approximate best agent equity from validation return
  var bestReturn = 0;
  if (validation) {
    for (var name in validation) {
      var r = validation[name];
      if ((r.fitness_score || 0) > bestReturn || bestReturn === 0) {
        bestReturn = r.total_return || 0;
      }
    }
  }
  var hydraEquity = spyEquity.map(function(_, i) {
    return 1.0 + bestReturn * (i / Math.max(spyEquity.length - 1, 1));
  });

  equityChart = new Chart(el.getContext('2d'), {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'HYDRA',
          data: hydraEquity,
          borderColor: NAVY,
          backgroundColor: 'rgba(27,42,74,0.06)',
          borderWidth: 2, fill: true, tension: 0.3, pointRadius: 0,
          datalabels: {
            display: function(ctx) { return ctx.dataIndex === ctx.dataset.data.length - 1; },
            color: NAVY, anchor: 'end', align: 'left',
            font: { size: 10, weight: 600 },
            formatter: function(v) { return (v * 100).toFixed(0) + '%'; }
          }
        },
        {
          label: benchmark.ticker || 'SPY',
          data: spyEquity,
          borderColor: GOLD,
          backgroundColor: 'rgba(200,169,81,0.06)',
          borderWidth: 2, fill: true, tension: 0.3, pointRadius: 0,
          datalabels: {
            display: function(ctx) { return ctx.dataIndex === ctx.dataset.data.length - 1; },
            color: GOLD, anchor: 'end', align: 'left',
            font: { size: 10, weight: 600 },
            formatter: function(v) { return (v * 100).toFixed(0) + '%'; }
          }
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { display: false, grid: { color: '#1A2338' } },
        y: { grid: { color: '#1A2338' }, ticks: { font: { size: 10 }, callback: function(v) { return (v * 100).toFixed(0) + '%'; } } }
      },
      plugins: { legend: { display: false }, tooltip: { mode: 'index', intersect: false } }
    }
  });
}

// ═══════════════════════════════════════════════════════════════════
// Trade Count Chart (bar)
// ═══════════════════════════════════════════════════════════════════
function updateTradesChart(generations) {
  if (tradesChart) { tradesChart.destroy(); tradesChart = null; }
  var el = document.getElementById('trades-chart');
  if (!generations || generations.length === 0) return;

  var labels = generations.map(function(d) { return 'Gen ' + d.generation; });
  // Sum total_trades across all agents in each generation's conviction data
  var trades = generations.map(function(d) {
    var conv = d.conviction || {};
    var total = 0;
    for (var agent in conv) {
      if (conv[agent] && conv[agent].total_trades) total += conv[agent].total_trades;
    }
    return total;
  });

  tradesChart = new Chart(el.getContext('2d'), {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        data: trades,
        backgroundColor: NAVY + 'B0',
        borderColor: NAVY,
        borderWidth: 1,
        borderRadius: 3,
        datalabels: {
          display: function(ctx) { return ctx.dataset.data[ctx.dataIndex] > 0; },
          color: NAVY,
          anchor: 'end',
          align: 'top',
          font: { size: 9, weight: 600 },
          formatter: function(v) { return v; }
        }
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { grid: { color: '#1A2338' }, ticks: { font: { size: 10 } } },
        y: { grid: { color: '#1A2338' }, beginAtZero: true, ticks: { font: { size: 10 }, callback: function(v) { return v.toFixed(0); } } }
      },
      plugins: { legend: { display: false } }
    }
  });
}

// ═══════════════════════════════════════════════════════════════════
// Price + Signals Chart
// ═══════════════════════════════════════════════════════════════════
function updatePriceChart(priceHistory, tradeSignals) {
  if (priceChart) { priceChart.destroy(); priceChart = null; }
  var el = document.getElementById('price-chart');
  if (!priceHistory || priceHistory.length === 0) return;

  var labels = priceHistory.map(function(p) { return p.step; });
  var avgPrices = priceHistory.map(function(p) {
    var cp = p.close_prices || [];
    return cp.length > 0 ? cp.reduce(function(a, b) { return a + b; }, 0) / cp.length : 0;
  });

  // SMA helper
  function sma(data, period) {
    return data.map(function(_, i) {
      if (i < period - 1) return null;
      var sum = 0;
      for (var j = i - period + 1; j <= i; j++) sum += data[j];
      return sum / period;
    });
  }
  var sma20 = sma(avgPrices, 20);
  var sma50 = sma(avgPrices, 50);

  // Buy/sell scatter
  var buys = [], sells = [];
  if (tradeSignals) {
    for (var s = 0; s < tradeSignals.length; s++) {
      var sig = tradeSignals[s];
      var idx = -1;
      for (var p = 0; p < priceHistory.length; p++) {
        if (priceHistory[p].step === sig.step) { idx = p; break; }
      }
      if (idx < 0) continue;
      var pt = { x: idx, y: avgPrices[idx] };
      if (sig.type === 'buy') buys.push(pt);
      else sells.push(pt);
    }
  }

  var datasets = [
    {
      label: 'Price',
      data: avgPrices,
      borderColor: NAVY,
      backgroundColor: 'rgba(27,42,74,0.04)',
      borderWidth: 1.5, fill: true, tension: 0.2, pointRadius: 0, order: 3,
      datalabels: { display: false }
    },
    {
      label: 'SMA20',
      data: sma20,
      borderColor: GOLD,
      borderWidth: 1, borderDash: [4, 2], fill: false, tension: 0.3, pointRadius: 0, order: 4,
      datalabels: { display: false }
    },
    {
      label: 'SMA50',
      data: sma50,
      borderColor: NAVY_LIGHT,
      borderWidth: 1, borderDash: [4, 2], fill: false, tension: 0.3, pointRadius: 0, order: 5,
      datalabels: { display: false }
    }
  ];

  if (buys.length > 0) {
    datasets.push({
      label: 'Buy',
      data: buys,
      type: 'scatter',
      pointStyle: 'triangle',
      pointRadius: 7,
      pointBackgroundColor: GREEN,
      pointBorderColor: GREEN,
      order: 1,
      datalabels: { display: false }
    });
  }
  if (sells.length > 0) {
    datasets.push({
      label: 'Sell',
      data: sells,
      type: 'scatter',
      pointStyle: 'triangle',
      rotation: 180,
      pointRadius: 7,
      pointBackgroundColor: RED,
      pointBorderColor: RED,
      order: 2,
      datalabels: { display: false }
    });
  }

  priceChart = new Chart(el.getContext('2d'), {
    type: 'line',
    data: { labels: labels.map(function(_, i) { return i; }), datasets: datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { grid: { color: '#1A2338' }, ticks: { maxTicksLimit: 15, font: { size: 10 }, callback: function(v, i) { return labels[i] || ''; } } },
        y: { grid: { color: '#1A2338' }, ticks: { font: { size: 10 }, callback: function(v) { return '$' + v.toFixed(0); } } }
      },
      plugins: {
        legend: { display: false },
        tooltip: { mode: 'index', intersect: false }
      }
    }
  });
}

// ═══════════════════════════════════════════════════════════════════
// Pipeline Timing (horizontal bar)
// ═══════════════════════════════════════════════════════════════════
function renderPipeline(phases) {
  var el = document.getElementById('pipeline-body');
  if (!phases || phases.length === 0) { el.innerHTML = '<div class="empty">No pipeline data.</div>'; return; }

  var maxDur = 1;
  for (var i = 0; i < phases.length; i++) {
    if ((phases[i].duration_ms || 0) > maxDur) maxDur = phases[i].duration_ms;
  }

  var html = '<div class="card-body">';
  for (var i = 0; i < phases.length; i++) {
    var p = phases[i];
    var dur = p.duration_ms || 0;
    var pct = (dur / maxDur * 100).toFixed(0);
    var durStr = dur > 60000 ? (dur / 60000).toFixed(1) + 'm' : dur > 1000 ? (dur / 1000).toFixed(1) + 's' : dur + 'ms';
    var ok = p.status === 'completed';
    var barColor = ok ? NAVY : RED;
    var statusDot = ok ? '<span class="clr-green fw-600">OK</span>' : '<span class="clr-red fw-600">ERR</span>';

    html += '<div style="margin-bottom:10px">' +
      '<div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:3px">' +
      '<span>' + statusDot + ' ' + p.phase + '</span>' +
      '<span class="clr-muted">' + durStr + '</span>' +
      '</div>' +
      '<div style="height:6px;background:#F0F0F3;border-radius:3px;overflow:hidden">' +
      '<div style="height:100%;width:' + pct + '%;background:' + barColor + ';border-radius:3px;transition:width 0.3s"></div>' +
      '</div></div>';
  }
  el.innerHTML = html + '</div>';
}

// ═══════════════════════════════════════════════════════════════════
// Training Config Table
// ═══════════════════════════════════════════════════════════════════
function renderConfig(cfg) {
  var el = document.getElementById('config-body');
  if (!cfg) { el.innerHTML = '<div class="empty">No configuration data.</div>'; return; }

  var rows = [
    ['Tickers', (cfg.tickers || []).join(', ') || '--'],
    ['Data Source', cfg.real_data ? '<span class="clr-green fw-600">Real (Alpaca)</span>' : '<span class="clr-amber fw-600">Synthetic</span>'],
    ['Generations', cfg.num_generations || '--'],
    ['Episodes / Gen', cfg.episodes_per_generation || '--'],
    ['Stocks', cfg.num_stocks || '--'],
    ['Seed', cfg.seed !== undefined ? cfg.seed : '--']
  ];

  var html = '<table>';
  for (var i = 0; i < rows.length; i++) {
    html += '<tr><td class="clr-muted">' + rows[i][0] + '</td><td class="num fw-600">' + rows[i][1] + '</td></tr>';
  }
  el.innerHTML = html + '</table>';
}

// ═══════════════════════════════════════════════════════════════════
// Main Refresh
// ═══════════════════════════════════════════════════════════════════
var hydraBuilt = false;

async function refresh() {
  try {
    var resp = await fetch('/api/data');
    var d = await resp.json();

    // Timestamp
    document.getElementById('ts').textContent = d.updated || '--';

    // Training progress bar
    var progressBar = document.getElementById('training-progress-bar');
    var progressFill = document.getElementById('training-progress-fill');
    if (d.training_active) {
      progressBar.classList.add('active');
      progressFill.style.width = (d.training_progress || 0) + '%';
    } else {
      progressBar.classList.remove('active');
    }

    // Build hydra (once, then just toggle class)
    if (!hydraBuilt) {
      buildHydra(d.training_active);
      hydraBuilt = true;
    } else {
      var hc = document.getElementById('hydra-container');
      if (d.training_active) hc.classList.add('training-active');
      else hc.classList.remove('training-active');
    }

    // KPIs
    updateKPIs(d);

    // Scorecard
    updateScorecard(d.scorecard);

    // Charts
    if (d.hero && d.hero.reward_trend) updateRewardChart(d.hero.reward_trend, d.hero.best_score_history, d.hero.prev_run_best);
    updateWeightsChart(d.generations);
    updateEquityChart(d.benchmark, d.validation);
    updateTradesChart(d.generations);
    updatePriceChart(d.price_history, d.trade_signals);

    // Agent table
    renderAgentTable(d);

    // Pipeline + Config
    renderPipeline(d.pipeline);
    renderConfig(d.config);

  } catch (e) {
    console.error('Dashboard fetch error:', e);
  }
}

refresh();
setInterval(refresh, 15000);
</script>
</body>
</html>
"""


# ── HTTP Server ──────────────────────────────────────────────────────────────

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/data":
            data = get_dashboard_data()
            body = json.dumps(data, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass


class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def main():
    server = ThreadedServer(("localhost", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"HYDRACORP Executive Dashboard  ->  {url}")
    print("Ctrl+C to stop.")
    if not NO_BROWSER:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()


if __name__ == "__main__":
    main()
