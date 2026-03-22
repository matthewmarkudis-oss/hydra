"""Hydra Capital CEO Dashboard — simplified, dollar-focused trading dashboard.

Launch: streamlit run scripts/ceo_dashboard/app.py --server.port 5050
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root and dashboard dir to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import COLORS, STARTING_CAPITAL_CAD, safety_label, REFRESH_INTERVAL_MS
from data_loader import load_dashboard_data

# --- Page config ---
st.set_page_config(
    page_title="Hydra Capital",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Auto-refresh ---
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=REFRESH_INTERVAL_MS, key="ceo_refresh")
except ImportError:
    pass  # Works without auto-refresh

# --- Load data ---
data = load_dashboard_data()

# === HEADER ===
col_logo, col_status, col_updated = st.columns([3, 5, 2])
with col_logo:
    st.markdown("### HYDRA CAPITAL")
with col_status:
    gen_history = data.get("gen_history", [])
    n_gens = len(gen_history)
    if data["passed_count"] > 0:
        st.markdown(f":green_circle: **System Healthy** — {data['passed_count']} agents validated")
    elif n_gens > 0:
        st.markdown(f":orange_circle: **Training in progress** — Gen {gen_history[-1]['gen']}/{data.get('total_generations', '?')}")
    elif data["total_agents"] > 0:
        st.markdown(":orange_circle: **Training in progress** — no agents validated yet")
    else:
        st.markdown(":red_circle: **No data** — run a training cycle first")
with col_updated:
    st.caption(f"Updated: {data['updated']}")

st.divider()

# === HERO KPI CARDS ===
c1, c2, c3, c4, c5 = st.columns(5)

# Compute improvement delta from gen history
gen_improve = ""
if len(gen_history) >= 2:
    first_best = gen_history[0]["best_eval"]
    last_best = gen_history[-1]["best_eval"]
    gen_improve = f"{last_best - first_best:+.0f} since Gen 1"

with c1:
    st.metric(
        "Portfolio Value",
        f"${data['portfolio_value']:,.2f}",
        delta=f"${data['dollar_pnl']:+,.2f}",
    )
with c2:
    st.metric(
        "Best Agent Return",
        f"{data['total_return_pct']:+.2f}%",
        delta=f"${data['dollar_pnl']:+,.2f} CAD",
        help="Return of the single best agent from the latest evaluation. Not cumulative across generations.",
    )
with c3:
    pool_delta = f"Pool: {gen_history[-1]['pool_size']}" if gen_history else ""
    st.metric(
        "Best Agent",
        data["best_agent"],
        delta=pool_delta,
    )
with c4:
    st.metric(
        "vs S&P 500",
        data["excess_label"],
        delta=f"{data['excess_return_pct']:+.2f}%",
    )
with c5:
    score = data["safety_score"]
    label, color = safety_label(score)
    st.metric(
        "Safety Score",
        f"{score}/100",
        delta=label,
        delta_color="normal" if score >= 60 else "inverse",
    )

st.divider()

# === TRAINING PROGRESS CHART (NEW) ===
st.subheader("Training Progress by Generation")

if gen_history and len(gen_history) >= 2:
    gens = [g["gen"] for g in gen_history]
    best_evals = [g["best_eval"] for g in gen_history]
    worst_evals = [g["worst_eval"] for g in gen_history]
    mean_rewards = [g["mean_reward"] for g in gen_history]
    pool_sizes = [g["pool_size"] for g in gen_history]
    best_agents = [g["best_agent"] for g in gen_history]

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Agent Performance (Eval Reward)", "Pool Size & Events"),
    )

    # --- Top chart: Performance ---

    # Shaded range between worst and best
    fig.add_trace(go.Scatter(
        x=gens + gens[::-1],
        y=best_evals + worst_evals[::-1],
        fill="toself",
        fillcolor="rgba(41, 121, 255, 0.08)",
        line=dict(width=0),
        name="Best-Worst Range",
        showlegend=True,
        hoverinfo="skip",
    ), row=1, col=1)

    # Best agent per generation
    fig.add_trace(go.Scatter(
        x=gens,
        y=best_evals,
        mode="lines+markers",
        name="Best Agent",
        line=dict(color=COLORS["blue"], width=3),
        marker=dict(size=8),
        customdata=best_agents,
        hovertemplate="Gen %{x}<br>Best: %{y:.0f}<br>Agent: %{customdata}<extra></extra>",
    ), row=1, col=1)

    # Mean reward
    fig.add_trace(go.Scatter(
        x=gens,
        y=mean_rewards,
        mode="lines",
        name="Pool Mean Reward",
        line=dict(color=COLORS["orange"], width=2, dash="dash"),
        hovertemplate="Gen %{x}<br>Mean: %{y:.0f}<extra></extra>",
    ), row=1, col=1)

    # Worst agent
    fig.add_trace(go.Scatter(
        x=gens,
        y=worst_evals,
        mode="lines",
        name="Worst Agent",
        line=dict(color=COLORS["red"], width=1, dash="dot"),
        hovertemplate="Gen %{x}<br>Worst: %{y:.0f}<extra></extra>",
    ), row=1, col=1)

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                  annotation_text="Break-even", row=1, col=1)

    # --- Bottom chart: Pool size bars with promotion/demotion markers ---
    promoted_counts = [g["promoted"] for g in gen_history]
    demoted_counts = [g["demoted"] for g in gen_history]

    fig.add_trace(go.Bar(
        x=gens,
        y=pool_sizes,
        name="Pool Size",
        marker_color="rgba(41, 121, 255, 0.5)",
        hovertemplate="Gen %{x}<br>Pool: %{y} agents<extra></extra>",
    ), row=2, col=1)

    # Promotion markers on pool chart
    fig.add_trace(go.Scatter(
        x=gens,
        y=promoted_counts,
        mode="markers",
        name="Promotions",
        marker=dict(symbol="triangle-up", size=10, color=COLORS["green"]),
        hovertemplate="Gen %{x}<br>Promoted: %{y}<extra></extra>",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=gens,
        y=[-d for d in demoted_counts],
        mode="markers",
        name="Demotions",
        marker=dict(symbol="triangle-down", size=10, color=COLORS["red"]),
        hovertemplate="Gen %{x}<br>Demoted: %{y}<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        height=550,
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Eval Reward", row=1, col=1)
    fig.update_yaxes(title_text="Agents", row=2, col=1)
    fig.update_xaxes(title_text="Generation", row=2, col=1)
    fig.update_xaxes(dtick=1, row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Summary stats below chart
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        improvement = best_evals[-1] - best_evals[0]
        st.metric("Reward Improvement", f"{improvement:+.0f}",
                   delta=f"Gen 1: {best_evals[0]:.0f} -> Gen {gens[-1]}: {best_evals[-1]:.0f}")
    with s2:
        st.metric("Current Pool", f"{pool_sizes[-1]} agents",
                   delta=f"+{pool_sizes[-1] - pool_sizes[0]} since start")
    with s3:
        above_zero = sum(1 for b in best_evals if b > 0)
        st.metric("Profitable Gens", f"{above_zero}/{len(gens)}",
                   delta=f"{above_zero/len(gens)*100:.0f}% of generations")
    with s4:
        current_phase = "warmup" if gens[-1] <= 2 else "exploration" if gens[-1] <= 10 else "exploitation"
        st.metric("Training Phase", current_phase.title(),
                   delta=f"Gen {gens[-1]}/{data.get('total_generations', '?')}")

elif gen_history and len(gen_history) == 1:
    st.info(f"Generation 1 complete (best eval: {gen_history[0]['best_eval']:.0f}). Chart appears after Gen 2.")
else:
    st.info("No generation data yet. The chart will appear as training progresses.")

st.divider()

# === PORTFOLIO GROWTH CHART ===
if data["spy_equity_curve"]:
    st.subheader("Portfolio Growth vs Benchmark")
    fig = go.Figure()

    # Hydra equity (reconstruct from best return applied linearly)
    n_points = len(data["spy_equity_curve"])
    best_ret = data["total_return_pct"] / 100
    hydra_equity = [
        STARTING_CAPITAL_CAD * (1 + best_ret * (i / max(n_points - 1, 1)))
        for i in range(n_points)
    ]

    # SPY equity
    spy_equity = [STARTING_CAPITAL_CAD * v for v in data["spy_equity_curve"]]

    fig.add_trace(go.Scatter(
        y=hydra_equity,
        mode="lines",
        name="Hydra Portfolio",
        line=dict(color=COLORS["blue"], width=3),
        fill="tozeroy",
        fillcolor="rgba(41, 121, 255, 0.1)",
    ))
    fig.add_trace(go.Scatter(
        y=spy_equity,
        mode="lines",
        name="S&P 500 (Buy & Hold)",
        line=dict(color=COLORS["orange"], width=2, dash="dash"),
    ))

    fig.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title="Portfolio Value (CAD)",
        xaxis_title="Trading Days",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode="x unified",
        yaxis_tickprefix="$",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

# === AGENT LEADERBOARD + RISK ===
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Agent Leaderboard")
    if data["leaderboard"]:
        for i, agent in enumerate(data["leaderboard"][:10]):
            color = COLORS["green"] if agent["return_pct"] > 0 else COLORS["red"]
            status = "Validated" if agent["passed"] else "Training"
            star = " :star:" if agent["is_best"] else ""
            st.markdown(
                f"**#{i+1}** {agent['name']}{star} — "
                f"<span style='color:{color}'>{agent['return_pct']:+.2f}% "
                f"(${agent['return_cad']:+,.2f})</span> "
                f"| Win Rate: {agent['win_rate_pct']:.0f}% "
                f"| *{status}*",
                unsafe_allow_html=True,
            )
    else:
        st.info("No agents evaluated yet.")

with col_right:
    st.subheader("Risk Dashboard")

    # Max Drawdown
    dd = data["max_drawdown_pct"]
    dd_cad = data["max_drawdown_cad"]
    st.markdown(f"**Worst Dip from Peak:** {dd:.1f}% (${dd_cad:,.2f})")
    dd_pct = min(dd / 25.0, 1.0)  # 25% is the limit
    st.progress(dd_pct, text=f"{dd:.1f}% of 25% limit")

    st.markdown("")

    # Win Rate
    wr = data["win_rate_pct"]
    st.markdown(f"**Win Rate:** {wr:.0f}% of trades made money")
    st.progress(min(wr / 100.0, 1.0), text=f"{wr:.0f}%")

    st.markdown("")

    # Profit Factor
    pf = data["profit_factor"]
    pf_label = "Good" if pf > 1.5 else "OK" if pf > 1.0 else "Losing"
    st.markdown(f"**Profit Ratio:** {pf:.2f}x ({pf_label})")
    st.caption("For every $1 lost, you earn ${:.2f}".format(pf))

st.divider()

# === ALERTS ===
st.subheader("Recent Events")
if data["alerts"]:
    for alert in data["alerts"][:10]:
        icon_map = {
            "star": ":star:",
            "arrow_down": ":arrow_down:",
            "warning": ":warning:",
            "rocket": ":rocket:",
            "chart_decreasing": ":chart_with_downwards_trend:",
        }
        icon = icon_map.get(alert.get("icon", ""), ":information_source:")
        st.markdown(f"{icon} {alert['message']}")
else:
    st.info("No events yet. Events appear after training cycles complete.")

st.divider()

# === BENCHMARK COMPARISON ===
st.subheader("How We Compare")

bm = data["benchmark"]
if data["total_agents"] > 0:
    col_metric, col_hydra, col_spy, col_verdict = st.columns(4)

    with col_metric:
        st.markdown("**Metric**")
        st.markdown("Total Return")
        st.markdown("Risk (Max Drawdown)")
        st.markdown("Risk-Adjusted Score")

    with col_hydra:
        st.markdown("**Hydra**")
        st.markdown(f"{data['total_return_pct']:+.2f}%")
        st.markdown(f"-{data['max_drawdown_pct']:.1f}%")
        best_sharpe = 0
        if data["leaderboard"]:
            best_sharpe = data["leaderboard"][0].get("sharpe", 0)
        st.markdown(f"{best_sharpe:.2f}")

    with col_spy:
        st.markdown("**S&P 500**")
        st.markdown(f"{bm['spy_return_pct']:+.2f}%")
        st.markdown(f"-{bm['spy_drawdown_pct']:.1f}%")
        st.markdown(f"{bm['spy_sharpe']:.2f}")

    with col_verdict:
        st.markdown("**Verdict**")
        ret_win = data["total_return_pct"] > bm["spy_return_pct"]
        st.markdown(f":{'green' if ret_win else 'red'}[{'WINNING' if ret_win else 'LOSING'}]")
        dd_win = data["max_drawdown_pct"] < bm["spy_drawdown_pct"]
        st.markdown(f":{'green' if dd_win else 'red'}[{'SAFER' if dd_win else 'RISKIER'}]")
        sh_win = best_sharpe > bm["spy_sharpe"]
        st.markdown(f":{'green' if sh_win else 'red'}[{'BETTER' if sh_win else 'WORSE'}]")
else:
    st.info("Run a training cycle to see benchmark comparison.")

# === TRAINING STATUS (collapsed) ===
with st.expander("System Details"):
    st.markdown(f"**Tickers:** {', '.join(data['tickers'])}")
    st.markdown(f"**Number of stocks:** {data['num_stocks']}")
    st.markdown(f"**Data mode:** {'Real Market Data' if data['real_data'] else 'Backtesting Simulation'}")
    st.markdown(f"**Generations completed:** {data['total_generations']}")
    st.markdown(f"**Agents in pool:** {data['total_agents']} "
                f"({data['passed_count']} validated)")
    st.markdown(f"**Starting capital:** ${STARTING_CAPITAL_CAD:,.2f} CAD")

    # Corp state if available
    corp = data.get("corp", {})
    if corp:
        st.markdown("---")
        st.markdown(f"**Corp pipeline runs:** {corp.get('pipeline_run_count', 0)}")
        regime = corp.get("regime", {})
        if regime.get("classification") and regime["classification"] != "unknown":
            st.markdown(f"**Market regime:** {regime['classification']}")
        blacklisted = len(corp.get("proposals", []))
        if blacklisted:
            st.markdown(f"**Pending proposals:** {blacklisted}")

# === FOOTER ===
st.divider()
st.caption(
    f"Hydra Capital Dashboard v2.1 | "
    f"Auto-refresh: {REFRESH_INTERVAL_MS // 1000}s | "
    f"Data: {'Real Market' if data['real_data'] else 'Backtesting Simulation'}"
)
