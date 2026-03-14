#!/usr/bin/env python3
"""
HYDRA Training Dashboard — single consolidated view
http://localhost:5010

Shows everything:
- Training progress (generations, rewards, pool evolution)
- CHIMERA diagnostics & mutations
- PROMETHEUS competition weights
- ELEOS conviction calibration
- ATHENA validation (PSR, DSR, bootstrap CI)
- KRONOS WFE overfitting detection
- CHIMERA fitness decomposition

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

HYDRA_ROOT = Path(__file__).parent.parent
LOGS_DIR = HYDRA_ROOT / "logs"
STATE_FILE = LOGS_DIR / "hydra_training_state.json"
PORT = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 5010
NO_BROWSER = "--no-browser" in sys.argv


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def get_dashboard_data():
    state = _load_json(STATE_FILE)
    if not state:
        return {
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "waiting",
            "status_line": "No training data yet -- run: python scripts/train.py --real-data",
            "traffic_light": "yellow",
        }

    config = state.get("config", {})
    summary = state.get("summary", {})
    metrics = state.get("metrics", {})
    generations = state.get("generations", [])
    validation = state.get("validation", {})
    eval_data = state.get("eval", {})
    tasks = state.get("tasks", {})

    total_gens = summary.get("total_generations", 0)
    passed = summary.get("passed_agents", [])
    rankings = summary.get("final_rankings", {})
    thresholds = summary.get("thresholds", {})

    # Best agent
    best_agent = max(rankings, key=rankings.get) if rankings else "N/A"
    best_score = rankings.get(best_agent, 0) if rankings else 0

    # Reward trend
    rewards = [g.get("train_mean_reward", 0) for g in generations]
    reward_improving = len(rewards) >= 2 and rewards[-1] > rewards[0]

    # Validation stats
    val_agents = list(validation.values())
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
        status_line = (
            f"Trained {total_gens} generations on {len(config.get('tickers', []))} "
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

    # Pipeline timing
    pipeline_timing = []
    for name, info in tasks.items():
        pipeline_timing.append({
            "phase": name,
            "status": info.get("status", "unknown"),
            "duration_ms": info.get("duration_ms", 0),
        })

    return {
        "updated": state.get("updated", ""),
        "status": "complete",
        "status_line": status_line,
        "traffic_light": traffic_light,
        "config": config,
        "hero": {
            "total_generations": total_gens,
            "best_fitness": round(best_fitness, 4),
            "best_sharpe": round(best_sharpe, 3),
            "avg_wfe": round(avg_wfe, 3),
            "agents_passed": len(passed),
            "total_agents": len(validation),
            "reward_trend": rewards,
        },
        "generations": diagnostics_history,
        "validation": {
            name: r for name, r in validation.items()
        },
        "rankings": rankings,
        "thresholds": thresholds,
        "passed_agents": passed,
        "metrics": metrics,
        "pipeline": pipeline_timing,
        "action_diagnostics": action_diagnostics,
    }


# ── HTML Dashboard ──────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HYDRA Training Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg: #0a0a0a; --card: #131313; --border: #222;
  --cyan: #00e5ff; --green: #00e676; --red: #ff5252;
  --yellow: #ffeb3b; --purple: #ce93d8; --blue: #4fc3f7;
  --orange: #ff9800; --text: #e0e0e0; --muted: #606060; --muted2: #2e2e2e;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: 'Consolas','Courier New',monospace; font-size: 13px; }

/* Header */
#header { background: #0e0e0e; border-bottom: 1px solid var(--border); padding: 10px 20px; display: flex; align-items: center; justify-content: space-between; }
#header h1 { color: var(--cyan); font-size: 17px; font-weight: bold; letter-spacing: 1px; }
.hdr-meta { color: var(--muted); font-size: 12px; }

.wrap { padding: 14px 20px; max-width: 1400px; margin: 0 auto; }

/* Hero */
#hero { background: linear-gradient(135deg, #131313 0%, #0e1a0e 100%); border: 1px solid var(--border); border-radius: 6px; padding: 20px 24px; margin-bottom: 14px; }
#hero.light-green { border-left: 4px solid var(--green); }
#hero.light-yellow { border-left: 4px solid var(--yellow); }
#hero.light-red { border-left: 4px solid var(--red); }

.hero-status { font-size: 15px; color: var(--text); margin-bottom: 14px; line-height: 1.6; }
.traffic { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; vertical-align: middle; }
.traffic-green { background: var(--green); box-shadow: 0 0 6px var(--green); }
.traffic-yellow { background: var(--yellow); box-shadow: 0 0 6px var(--yellow); }
.traffic-red { background: var(--red); box-shadow: 0 0 6px var(--red); }

.hero-numbers { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
.hero-num { text-align: center; }
.hero-num .big { font-size: 30px; font-weight: bold; }
.hero-num .label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.hero-num .explain { font-size: 10px; color: #444; margin-top: 3px; }

/* Grid layouts */
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 14px; }
.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; margin-bottom: 14px; }
.grid-2-1 { display: grid; grid-template-columns: 2fr 1fr; gap: 14px; margin-bottom: 14px; }

/* Card */
.card { background: var(--card); border: 1px solid var(--border); border-radius: 4px; margin-bottom: 14px; overflow: hidden; }
.card-title { font-size: 10px; letter-spacing: 1.5px; text-transform: uppercase; color: var(--muted); padding: 9px 14px 7px; border-bottom: 1px solid var(--border); background: #0e0e0e; }
.card-title .protocol-tag { display: inline-block; padding: 1px 6px; border-radius: 2px; font-size: 9px; margin-left: 8px; letter-spacing: 0; }
.tag-chimera { background: #2d1b3a; color: var(--purple); }
.tag-prometheus { background: #3a2d1b; color: var(--orange); }
.tag-eleos { background: #1b3a2d; color: var(--green); }
.tag-athena { background: #1b2d3a; color: var(--blue); }
.tag-kronos { background: #3a1b1b; color: var(--red); }
.card-body { padding: 14px; }
.chart-wrap { padding: 12px 14px; height: 200px; position: relative; }

/* Table */
table { width: 100%; border-collapse: collapse; }
th { text-align: left; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted); padding: 7px 12px; border-bottom: 1px solid var(--border); white-space: nowrap; }
td { padding: 6px 12px; border-bottom: 1px solid #191919; white-space: nowrap; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #1c1c1c; }
.num { text-align: right; }
.g { color: var(--green); } .r { color: var(--red); } .y { color: var(--yellow); }
.c { color: var(--cyan); } .m { color: var(--muted); } .p { color: var(--purple); }
.o { color: var(--orange); } .b { color: var(--blue); }

/* Badges */
.badge { display: inline-block; padding: 1px 7px; border-radius: 2px; font-size: 11px; font-weight: bold; }
.badge-pass { background: #1b3a1b; color: var(--green); }
.badge-fail { background: #3a1b1b; color: var(--red); }
.badge-minor { background: #1b3a2d; color: var(--green); }
.badge-moderate { background: #3a3a1b; color: var(--yellow); }
.badge-severe { background: #3a2d1b; color: var(--orange); }
.badge-critical { background: #3a1b1b; color: var(--red); }

/* Progress bar */
.progress-bar { height: 8px; background: #222; border-radius: 4px; overflow: hidden; margin: 4px 0; }
.progress-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }

/* Fitness breakdown bar */
.fitness-bar { display: flex; height: 16px; border-radius: 3px; overflow: hidden; margin: 4px 0; }
.fitness-bar div { height: 100%; }

/* WR bar inline */
.wr-wrap { display: inline-flex; align-items: center; gap: 6px; }
.wr-num { min-width: 36px; text-align: right; display: inline-block; }
.wr-bar { width: 44px; height: 5px; background: #222; border-radius: 2px; overflow: hidden; }
.wr-fill { height: 100%; border-radius: 2px; }

.empty { padding: 28px; text-align: center; color: var(--muted); font-size: 12px; }
#footer { text-align: center; color: var(--muted2); font-size: 11px; padding: 10px; border-top: 1px solid var(--border); }
</style>
</head>
<body>

<div id="header">
  <h1>HYDRA Training Dashboard</h1>
  <div class="hdr-meta">Updated: <span id="ts">--</span> &nbsp; auto-refresh 15s</div>
</div>

<div class="wrap">

  <!-- Hero -->
  <div id="hero">
    <div class="hero-status"><span class="traffic" id="traffic-dot"></span><span id="hero-status-text">Loading...</span></div>
    <div class="hero-numbers">
      <div class="hero-num"><div class="label">Generations</div><div class="big c" id="hero-gens">--</div><div class="explain">Training rounds completed</div></div>
      <div class="hero-num"><div class="label">Best Fitness</div><div class="big" id="hero-fitness">--</div><div class="explain">CHIMERA multi-objective score (0-1)</div></div>
      <div class="hero-num"><div class="label">Best Sharpe</div><div class="big" id="hero-sharpe">--</div><div class="explain">Risk-adjusted return (above 0.3 = good)</div></div>
      <div class="hero-num"><div class="label">Agents Passed</div><div class="big" id="hero-passed">--</div><div class="explain">ATHENA validation gate</div></div>
    </div>
  </div>

  <!-- Charts row -->
  <div class="grid-2">
    <div class="card" style="margin:0">
      <div class="card-title">Reward Trend by Generation</div>
      <div class="chart-wrap"><canvas id="reward-chart"></canvas></div>
    </div>
    <div class="card" style="margin:0">
      <div class="card-title">Agent Competition Weights <span class="protocol-tag tag-prometheus">PROMETHEUS</span></div>
      <div class="chart-wrap"><canvas id="weights-chart"></canvas></div>
    </div>
  </div>

  <!-- Action Diagnostics -->
  <div class="grid-2">
    <div class="card" style="margin:0">
      <div class="card-title">Trade Count by Generation</div>
      <div class="chart-wrap"><canvas id="trades-chart"></canvas></div>
    </div>
    <div class="card" style="margin:0">
      <div class="card-title">Per-Agent Action Magnitude</div>
      <div class="chart-wrap"><canvas id="action-mag-chart"></canvas></div>
    </div>
  </div>

  <!-- Generation History -->
  <div class="card">
    <div class="card-title">Generation History <span class="protocol-tag tag-chimera">CHIMERA</span> <span class="protocol-tag tag-prometheus">PROMETHEUS</span> <span class="protocol-tag tag-eleos">ELEOS</span></div>
    <div id="gen-body"><div class="empty">No data yet.</div></div>
  </div>

  <!-- Validation + Fitness -->
  <div class="grid-2-1">
    <div class="card" style="margin:0">
      <div class="card-title">Agent Validation <span class="protocol-tag tag-athena">ATHENA</span> <span class="protocol-tag tag-kronos">KRONOS</span></div>
      <div id="val-body"><div class="empty">No validation data.</div></div>
    </div>
    <div class="card" style="margin:0">
      <div class="card-title">Fitness Decomposition <span class="protocol-tag tag-chimera">CHIMERA</span></div>
      <div id="fitness-body"><div class="empty">No fitness data.</div></div>
    </div>
  </div>

  <!-- Pipeline Timing + Config -->
  <div class="grid-2">
    <div class="card" style="margin:0">
      <div class="card-title">Pipeline Timing</div>
      <div id="pipeline-body"><div class="empty">No data.</div></div>
    </div>
    <div class="card" style="margin:0">
      <div class="card-title">Training Configuration</div>
      <div id="config-body"><div class="empty">No data.</div></div>
    </div>
  </div>

</div>
<div id="footer">HYDRA Training Dashboard -- CHIMERA + PROMETHEUS + ELEOS + ATHENA + KRONOS</div>

<script>
Chart.defaults.color = '#606060';
Chart.defaults.borderColor = '#222';
Chart.defaults.font.family = "Consolas, 'Courier New', monospace";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.display = false;
Chart.defaults.animation = { duration: 400 };

let rewardChart = null, weightsChart = null, tradesChart = null, actionMagChart = null;

function fmtNum(v, dec) {
  if (v === null || v === undefined || v === 'N/A') return '<span class="m">--</span>';
  const n = parseFloat(v), sign = n > 0 ? '+' : '';
  const cls = n > 0 ? 'g' : n < 0 ? 'r' : 'm';
  return `<span class="${cls}">${sign}${n.toFixed(dec)}</span>`;
}

function fmtPct(v) {
  if (v === null || v === undefined) return '<span class="m">--</span>';
  const n = parseFloat(v) * 100;
  const cls = n >= 50 ? 'g' : n >= 35 ? 'y' : 'r';
  return `<span class="${cls}">${n.toFixed(1)}%</span>`;
}

function severityBadge(s) {
  if (!s || s === 'N/A') return '<span class="m">--</span>';
  return `<span class="badge badge-${s}">${s.toUpperCase()}</span>`;
}

function wfeBadge(wfe, diag) {
  if (wfe === undefined || wfe === null) return '<span class="m">--</span>';
  const w = parseFloat(wfe);
  const cls = w >= 0.6 ? 'g' : w >= 0.4 ? 'y' : w >= 0.25 ? 'o' : 'r';
  const label = diag && diag.severity ? diag.severity : '';
  return `<span class="${cls}">${w.toFixed(2)}</span>` + (label ? ` <span class="m" style="font-size:10px">${label}</span>` : '');
}

// Reward trend chart
function updateRewardChart(rewards) {
  if (rewardChart) { rewardChart.destroy(); rewardChart = null; }
  const el = document.getElementById('reward-chart');
  if (!rewards || rewards.length === 0) return;
  const labels = rewards.map((_, i) => `Gen ${i+1}`);
  const improving = rewards[rewards.length-1] > rewards[0];
  rewardChart = new Chart(el.getContext('2d'), {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Mean Reward',
        data: rewards,
        borderColor: improving ? '#00e676' : '#ff5252',
        backgroundColor: improving ? 'rgba(0,230,118,0.08)' : 'rgba(255,82,82,0.08)',
        borderWidth: 2, fill: true, tension: 0.3, pointRadius: 4,
        pointBackgroundColor: improving ? '#00e676' : '#ff5252',
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { grid: { color: '#1a1a1a' } },
        y: { grid: { color: '#1a1a1a' }, ticks: { callback: v => v.toFixed(0) } }
      },
      plugins: { legend: { display: true, labels: { color: '#606060' } } }
    }
  });
}

// Weight evolution chart (stacked bar)
function updateWeightsChart(generations) {
  if (weightsChart) { weightsChart.destroy(); weightsChart = null; }
  const el = document.getElementById('weights-chart');
  if (!generations || generations.length === 0) return;

  // Collect all agent names
  const allAgents = new Set();
  for (const g of generations) {
    for (const a of Object.keys(g.weights_after || {})) allAgents.add(a);
  }
  const agents = [...allAgents].sort();
  if (agents.length === 0) return;

  const colors = ['#00e5ff','#00e676','#ff5252','#ffeb3b','#ce93d8','#4fc3f7','#ff9800','#76ff03'];
  const labels = generations.map(g => `Gen ${g.generation}`);
  const datasets = agents.map((a, i) => ({
    label: a,
    data: generations.map(g => (g.weights_after || {})[a] || 0),
    backgroundColor: colors[i % colors.length] + '60',
    borderColor: colors[i % colors.length],
    borderWidth: 1,
  }));

  weightsChart = new Chart(el.getContext('2d'), {
    type: 'bar',
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { stacked: true, grid: { color: '#1a1a1a' } },
        y: { stacked: true, grid: { color: '#1a1a1a' }, max: 1, ticks: { callback: v => (v*100).toFixed(0) + '%' } }
      },
      plugins: { legend: { display: true, labels: { color: '#606060', font: { size: 10 } } } }
    }
  });
}

// Trade count chart
function updateTradesChart(actionDiag) {
  if (tradesChart) { tradesChart.destroy(); tradesChart = null; }
  const el = document.getElementById('trades-chart');
  if (!actionDiag || actionDiag.length === 0) return;
  const labels = actionDiag.map(d => `Gen ${d.generation}`);
  const trades = actionDiag.map(d => d.num_trades || 0);
  const hasTrading = trades.some(t => t > 0);
  tradesChart = new Chart(el.getContext('2d'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Trades',
        data: trades,
        backgroundColor: hasTrading ? 'rgba(0,230,118,0.4)' : 'rgba(255,82,82,0.4)',
        borderColor: hasTrading ? '#00e676' : '#ff5252',
        borderWidth: 1,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { grid: { color: '#1a1a1a' } },
        y: { grid: { color: '#1a1a1a' }, beginAtZero: true, ticks: { callback: v => v.toFixed(0) } }
      },
      plugins: { legend: { display: true, labels: { color: '#606060' } } }
    }
  });
}

// Per-agent action magnitude chart
function updateActionMagChart(actionDiag) {
  if (actionMagChart) { actionMagChart.destroy(); actionMagChart = null; }
  const el = document.getElementById('action-mag-chart');
  if (!actionDiag || actionDiag.length === 0) return;

  const allAgents = new Set();
  for (const d of actionDiag) {
    for (const a of Object.keys(d.per_agent_mean_action || {})) allAgents.add(a);
  }
  const agents = [...allAgents].sort();
  if (agents.length === 0) {
    // Fallback: show aggregated mean action
    const labels = actionDiag.map(d => `Gen ${d.generation}`);
    const aggData = actionDiag.map(d => d.aggregated_mean_action || 0);
    const preData = actionDiag.map(d => d.pre_aggregation_mean || 0);
    actionMagChart = new Chart(el.getContext('2d'), {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label: 'Pre-Aggregation', data: preData, borderColor: '#ffeb3b', borderWidth: 2, fill: false, tension: 0.3, pointRadius: 3 },
          { label: 'Post-Aggregation', data: aggData, borderColor: '#00e5ff', borderWidth: 2, fill: false, tension: 0.3, pointRadius: 3 },
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          x: { grid: { color: '#1a1a1a' } },
          y: { grid: { color: '#1a1a1a' }, beginAtZero: true }
        },
        plugins: { legend: { display: true, labels: { color: '#606060', font: { size: 10 } } } }
      }
    });
    return;
  }

  const colors = ['#00e5ff','#00e676','#ff5252','#ffeb3b','#ce93d8','#4fc3f7','#ff9800','#76ff03'];
  const labels = actionDiag.map(d => `Gen ${d.generation}`);
  const datasets = agents.map((a, i) => ({
    label: a,
    data: actionDiag.map(d => (d.per_agent_mean_action || {})[a] || 0),
    borderColor: colors[i % colors.length],
    backgroundColor: colors[i % colors.length] + '20',
    borderWidth: 2, fill: false, tension: 0.3, pointRadius: 3,
  }));

  actionMagChart = new Chart(el.getContext('2d'), {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { grid: { color: '#1a1a1a' } },
        y: { grid: { color: '#1a1a1a' }, beginAtZero: true, title: { display: true, text: 'Mean |action|', color: '#606060' } }
      },
      plugins: { legend: { display: true, labels: { color: '#606060', font: { size: 10 } } } }
    }
  });
}

// Generation table
function renderGenerations(gens) {
  const el = document.getElementById('gen-body');
  if (!gens || gens.length === 0) { el.innerHTML = '<div class="empty">No generation data.</div>'; return; }
  let html = `<table><thead><tr>
    <th>Gen</th><th class="num">Reward</th><th>Pool</th>
    <th>Diagnosis <span class="p">[CHIMERA]</span></th><th>Mutations</th>
    <th>Converged <span class="o">[PROMETHEUS]</span></th>
    <th>Promoted</th><th>Demoted</th>
  </tr></thead><tbody>`;
  for (const g of gens) {
    const reward = g.reward;
    const rewardCls = reward > 0 ? 'g' : reward < -50 ? 'r' : 'y';
    html += `<tr>
      <td class="c" style="font-weight:bold">${g.generation}</td>
      <td class="num ${rewardCls}">${reward.toFixed(1)}</td>
      <td class="m">${g.pool_size}</td>
      <td>${severityBadge(g.severity)} <span class="m" style="font-size:10px">${(g.primary_issue||'').substring(0,50)}</span></td>
      <td class="num">${g.num_mutations || 0}</td>
      <td>${g.converged ? '<span class="g">YES</span>' : '<span class="m">no</span>'}</td>
      <td class="m" style="font-size:10px">${(g.promoted||[]).join(', ') || '--'}</td>
      <td class="m" style="font-size:10px">${(g.demoted||[]).join(', ') || '--'}</td>
    </tr>`;
  }
  el.innerHTML = html + '</tbody></table>';
}

// Validation table
function renderValidation(val) {
  const el = document.getElementById('val-body');
  if (!val || Object.keys(val).length === 0) { el.innerHTML = '<div class="empty">No validation data.</div>'; return; }
  let html = `<table><thead><tr>
    <th>Agent</th><th class="num">Sharpe</th><th class="num">PSR</th><th class="num">DSR</th>
    <th>WFE</th><th>Win Rate</th><th class="num">PF</th><th class="num">Trades</th>
    <th>Calibration</th><th>Result</th>
  </tr></thead><tbody>`;
  for (const [name, r] of Object.entries(val)) {
    const badge = r.passed ? '<span class="badge badge-pass">PASS</span>' : '<span class="badge badge-fail">FAIL</span>';
    const calBadge = r.calibration_verdict === 'PASS'
      ? '<span class="g">PASS</span>'
      : r.calibration_verdict === 'FAIL' ? '<span class="r">FAIL</span>' : '<span class="m">N/A</span>';
    html += `<tr>
      <td class="c" style="font-weight:bold">${name}</td>
      <td class="num">${fmtNum(r.sharpe, 3)}</td>
      <td class="num">${fmtNum(r.psr, 3)}</td>
      <td class="num">${fmtNum(r.dsr, 3)}</td>
      <td>${wfeBadge(r.wfe, r.wfe_diagnosis)}</td>
      <td>${fmtPct(r.win_rate)}</td>
      <td class="num">${fmtNum(r.profit_factor, 2)}</td>
      <td class="num m">${r.total_trades || 0}</td>
      <td>${calBadge}</td>
      <td>${badge}</td>
    </tr>`;
  }
  el.innerHTML = html + '</tbody></table>';
}

// Fitness decomposition
function renderFitness(val) {
  const el = document.getElementById('fitness-body');
  if (!val || Object.keys(val).length === 0) { el.innerHTML = '<div class="empty">No fitness data.</div>'; return; }

  let html = '';
  const colors = { sharpe: '#00e5ff', max_dd: '#ff5252', profit_factor: '#00e676', wfe: '#ffeb3b', consistency: '#ce93d8' };
  const labels = { sharpe: 'Sharpe', max_dd: 'Drawdown', profit_factor: 'Profit Factor', wfe: 'WFE', consistency: 'Consistency' };

  for (const [name, r] of Object.entries(val)) {
    const bd = r.fitness_breakdown || {};
    const score = r.fitness_score || 0;
    const stability = bd.stability_multiplier || 1;
    html += `<div style="margin-bottom:14px">
      <div style="display:flex;justify-content:space-between;margin-bottom:4px">
        <span class="c" style="font-weight:bold">${name}</span>
        <span style="font-weight:bold">${score.toFixed(4)}</span>
      </div>
      <div class="fitness-bar">`;
    for (const [k, color] of Object.entries(colors)) {
      const w = (bd[k] || 0) * 100;
      if (w > 0) {
        html += `<div style="width:${Math.max(w*2, 2)}%;background:${color}" title="${labels[k]}: ${bd[k]?.toFixed(4) || 0}"></div>`;
      }
    }
    html += `</div>
      <div style="font-size:10px;color:var(--muted);display:flex;gap:12px;flex-wrap:wrap">`;
    for (const [k, label] of Object.entries(labels)) {
      html += `<span><span style="color:${colors[k]}">\u25A0</span> ${label}: ${(bd[k] || 0).toFixed(3)}</span>`;
    }
    html += `<span>Stability: ${stability.toFixed(3)}</span></div></div>`;
  }
  el.innerHTML = `<div class="card-body">${html}</div>`;
}

// Pipeline timing
function renderPipeline(phases) {
  const el = document.getElementById('pipeline-body');
  if (!phases || phases.length === 0) { el.innerHTML = '<div class="empty">No pipeline data.</div>'; return; }
  const maxDur = Math.max(...phases.map(p => p.duration_ms || 0), 1);
  let html = '<div class="card-body">';
  for (const p of phases) {
    const dur = p.duration_ms || 0;
    const pct = (dur / maxDur * 100).toFixed(0);
    const durStr = dur > 60000 ? `${(dur/60000).toFixed(1)}m` : dur > 1000 ? `${(dur/1000).toFixed(1)}s` : `${dur}ms`;
    const ok = p.status === 'completed';
    html += `<div style="margin-bottom:8px">
      <div style="display:flex;justify-content:space-between;font-size:11px">
        <span>${ok ? '<span class="g">OK</span>' : '<span class="r">ERR</span>'} ${p.phase}</span>
        <span class="m">${durStr}</span>
      </div>
      <div class="progress-bar"><div class="progress-fill" style="width:${pct}%;background:${ok ? 'var(--cyan)' : 'var(--red)'}"></div></div>
    </div>`;
  }
  el.innerHTML = html + '</div>';
}

// Config display
function renderConfig(cfg) {
  const el = document.getElementById('config-body');
  if (!cfg) { el.innerHTML = '<div class="empty">No config.</div>'; return; }
  let html = '<div class="card-body" style="font-size:12px">';
  const rows = [
    ['Tickers', (cfg.tickers || []).join(', ')],
    ['Data Source', cfg.real_data ? '<span class="g">Real (Alpaca)</span>' : '<span class="y">Synthetic</span>'],
    ['Generations', cfg.num_generations],
    ['Episodes/Gen', cfg.episodes_per_generation],
    ['Seed', cfg.seed],
  ];
  for (const [k, v] of rows) {
    html += `<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #191919"><span class="m">${k}</span><span>${v}</span></div>`;
  }
  el.innerHTML = html + '</div>';
}

// Main refresh
async function refresh() {
  try {
    const resp = await fetch('/api/data');
    const d = await resp.json();

    document.getElementById('ts').textContent = d.updated || '--';

    // Hero
    const hero = document.getElementById('hero');
    hero.className = `light-${d.traffic_light || 'yellow'}`;
    document.getElementById('traffic-dot').className = `traffic traffic-${d.traffic_light || 'yellow'}`;
    document.getElementById('hero-status-text').textContent = d.status_line || 'Loading...';

    if (d.hero) {
      document.getElementById('hero-gens').textContent = d.hero.total_generations || '--';

      const fitEl = document.getElementById('hero-fitness');
      const fit = d.hero.best_fitness || 0;
      fitEl.textContent = fit.toFixed(3);
      fitEl.style.color = fit > 0.5 ? 'var(--green)' : fit > 0.2 ? 'var(--yellow)' : 'var(--red)';

      const shEl = document.getElementById('hero-sharpe');
      const sh = d.hero.best_sharpe || 0;
      shEl.textContent = (sh > 0 ? '+' : '') + sh.toFixed(3);
      shEl.style.color = sh >= 0.3 ? 'var(--green)' : sh >= 0 ? 'var(--yellow)' : 'var(--red)';

      const passEl = document.getElementById('hero-passed');
      const pass = d.hero.agents_passed || 0;
      const total = d.hero.total_agents || 0;
      passEl.textContent = `${pass}/${total}`;
      passEl.style.color = pass > 0 ? 'var(--green)' : 'var(--yellow)';

      updateRewardChart(d.hero.reward_trend);
    }

    // Charts + Tables
    updateWeightsChart(d.generations);
    updateTradesChart(d.action_diagnostics);
    updateActionMagChart(d.action_diagnostics);
    renderGenerations(d.generations);
    renderValidation(d.validation);
    renderFitness(d.validation);
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
    print(f"HYDRA Training Dashboard  ->  {url}")
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
