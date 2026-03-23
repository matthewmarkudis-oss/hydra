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

from datetime import datetime

from config import (
    COLORS, STARTING_CAPITAL_CAD, safety_label, REFRESH_INTERVAL_MS,
    friendly_name, CHART_COLORS,
)
from data_loader import load_dashboard_data

# --- Page config ---
st.set_page_config(
    page_title="HYDRACORP",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Navy + Gold Theme CSS ---
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
/* ── Global overrides ──────────────────────────────────────────── */
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
}

/* Base styling */
.stApp, [data-testid="stAppViewContainer"], section[data-testid="stSidebar"],
.main .block-container {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* Remove Streamlit default padding */
.main .block-container {
  padding-top: 0 !important;
  max-width: 1440px !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header[data-testid="stHeader"] {
  visibility: hidden !important;
  height: 0 !important;
}

/* ── Typography ────────────────────────────────────────────────── */
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
  color: var(--gold) !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: 1px !important;
}

p, span, label, .stMarkdown, .stCaption, div {
  font-family: 'Inter', sans-serif !important;
}

/* Muted text */
.stCaption, [data-testid="stCaptionContainer"] {
  color: var(--text-muted) !important;
}

/* ── Metric cards ──────────────────────────────────────────────── */
[data-testid="stMetric"] {
  background: var(--card) !important;
  border: 1px solid var(--card-border) !important;
  border-radius: 8px !important;
  padding: 16px 20px !important;
  box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
}

[data-testid="stMetricLabel"] {
  font-size: 10px !important;
  font-weight: 600 !important;
  letter-spacing: 0.5px !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
}

[data-testid="stMetricValue"] {
  font-size: 24px !important;
  font-weight: 700 !important;
  color: var(--gold) !important;
}

[data-testid="stMetricDelta"] {
  font-size: 11px !important;
}

/* ── Dividers ──────────────────────────────────────────────────── */
hr, [data-testid="stDivider"] {
  border-color: var(--card-border) !important;
  opacity: 0.6;
}

/* ── Info/Warning boxes ────────────────────────────────────────── */
.stAlert {
  background: var(--card) !important;
  border: 1px solid var(--card-border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
}

/* ── Expander ──────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: var(--card) !important;
  border: 1px solid var(--card-border) !important;
  border-radius: 8px !important;
}

[data-testid="stExpander"] summary {
  color: var(--text-muted) !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.5px !important;
}

/* ── Progress bars ─────────────────────────────────────────────── */
.stProgress > div > div > div {
  background: linear-gradient(90deg, var(--gold), var(--gold-light)) !important;
}
.stProgress > div > div {
  background: var(--navy-dark) !important;
}

/* ── Tabs ──────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--card) !important;
  border-bottom: 1px solid var(--card-border) !important;
}
.stTabs [data-baseweb="tab"] {
  color: var(--text-muted) !important;
}
.stTabs [aria-selected="true"] {
  color: var(--gold) !important;
  border-bottom-color: var(--gold) !important;
}

/* Header + Hydra + Progress bar are rendered via components.html() */

/* ── Section headers ───────────────────────────────────────────── */
.section-header {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  color: var(--text-muted);
  border-bottom: 1px solid var(--card-border);
  padding-bottom: 8px;
  margin-bottom: 16px;
}

/* ── Footer ────────────────────────────────────────────────────── */
.hydra-footer {
  text-align: center;
  color: var(--text-muted);
  font-size: 10px;
  padding: 14px;
  border-top: 1px solid var(--card-border);
  letter-spacing: 0.3px;
  margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# --- Auto-refresh ---
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=REFRESH_INTERVAL_MS, key="ceo_refresh")
except ImportError:
    pass  # Works without auto-refresh

# --- Load data ---
data = load_dashboard_data()

# --- Build pixel hydra as SVG ---
def _build_hydra_svg(active: bool) -> str:
    """Generate the 32x16 pixel art hydra as an inline SVG."""
    PX = 5  # pixel size
    W, H = 32 * PX, 16 * PX
    color_map = {
        "bg": "none",
        "navy": "#1B2A4A",
        "navy-light": "#243558",
        "navy-dark": "#111D35",
        "gold": "#C8A951",
        "gold-light": "#D4BC72",
        "eye": "#C8A951",
    }

    grid = [["bg"] * 32 for _ in range(16)]

    def s(r, c, cls):
        if 0 <= r < 16 and 0 <= c < 32:
            grid[r][c] = cls

    # Head 1 (left)
    for dr in range(3):
        for dc in range(3):
            s(dr, 6+dc, "navy")
    s(0, 7, "navy-dark")
    s(1, 7, "eye")

    # Head 2 (center)
    for dr in range(3):
        for dc in range(3):
            s(dr, 14+dc, "navy")
    s(0, 15, "navy-dark")
    s(1, 15, "eye")

    # Head 3 (right)
    for dr in range(3):
        for dc in range(3):
            s(dr, 22+dc, "navy")
    s(0, 23, "navy-dark")
    s(1, 23, "eye")

    # Necks
    for r in range(3, 6):
        s(r, 7, "navy"); s(r, 8, "navy-light")
    for r in range(3, 6):
        s(r, 15, "navy"); s(r, 14, "navy-light")
    for r in range(3, 5):
        s(r, 23, "navy"); s(r, 22, "navy-light")

    # Body
    for r in range(5, 12):
        for c in range(6, 26):
            s(r, c, "navy")
    for c in range(10, 22):
        s(9, c, "gold")
        s(10, c, "gold-light")
    for c in range(8, 24):
        s(6, c, "navy-light")

    # Legs
    s(12, 8, "navy"); s(12, 9, "navy")
    s(13, 7, "navy-dark"); s(13, 8, "navy-dark")
    s(12, 22, "navy"); s(12, 23, "navy")
    s(13, 23, "navy-dark"); s(13, 24, "navy-dark")

    # Tail
    s(10, 5, "navy"); s(11, 4, "navy"); s(11, 5, "navy")
    s(12, 3, "navy-light"); s(12, 4, "navy")
    s(13, 2, "navy-light"); s(13, 3, "navy-light")
    s(14, 1, "gold"); s(14, 2, "navy-light")

    # Build SVG rects
    rects = []
    for r in range(16):
        for c in range(32):
            cls = grid[r][c]
            if cls != "bg":
                fill = color_map[cls]
                rects.append(
                    f'<rect x="{c*PX}" y="{r*PX}" width="{PX}" height="{PX}" '
                    f'rx="1" fill="{fill}"/>'
                )

    anim_style = ""
    if active:
        anim_style = """
        <style>
          @keyframes hpulse { 0%,100%{opacity:0.85} 50%{opacity:1} }
          svg { animation: hpulse 2s ease-in-out infinite; }
        </style>
        """
    else:
        anim_style = """
        <style>
          @keyframes hsleep { 0%,100%{opacity:0.4} 50%{opacity:0.7} }
          svg { animation: hsleep 5s ease-in-out infinite; }
        </style>
        """

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}">{anim_style}{"".join(rects)}</svg>'
    )


# === HEADER ===
gen_history = data.get("gen_history", [])
n_gens = len(gen_history)
is_training = n_gens > 0 and data.get("total_generations", 0) > 0

# Status text
if data["passed_count"] > 0:
    status_dot = "#22C55E"
    status_text = f"System Healthy — {data['passed_count']} agents validated"
elif n_gens > 0:
    status_dot = "#F59E0B"
    status_text = f"Training in progress — Gen {gen_history[-1]['gen']}/{data.get('total_generations', '?')}"
elif data["total_agents"] > 0:
    status_dot = "#F59E0B"
    status_text = "Training in progress"
else:
    status_dot = "#EF4444"
    status_text = "No data — run a training cycle first"

hydra_svg = _build_hydra_svg(active=is_training)

st.markdown(
    f'<div style="background:linear-gradient(135deg,#0D1525 0%,#162240 100%);'
    f'border-bottom:2px solid #C8A951;padding:18px 28px;display:flex;'
    f'align-items:center;justify-content:space-between;margin:-1rem -1rem 1rem -1rem;">'
    f'<div style="display:flex;align-items:center;gap:16px;">'
    f'<span style="font-size:20px;font-weight:700;letter-spacing:3px;color:#C8A951;'
    f'font-family:Inter,sans-serif;">HYDRACORP</span>'
    f'{hydra_svg}'
    f'</div>'
    f'<div style="text-align:right;">'
    f'<div style="font-size:12px;color:rgba(255,255,255,0.7);">'
    f'<span style="color:{status_dot}">&#9679;</span> {status_text}</div>'
    f'<div style="font-size:11px;color:rgba(255,255,255,0.5);">'
    f'Last updated: <span style="color:rgba(255,255,255,0.8);font-weight:500;">'
    f'{data["updated"]}</span></div>'
    f'</div></div>',
    unsafe_allow_html=True,
)

# Training progress bar
if is_training and n_gens > 0:
    total_target = data.get("total_generations", 1) or 1
    progress_pct = min(100, (gen_history[-1]["gen"] / total_target) * 100) if total_target else 0
    st.markdown(
        f'<div style="height:4px;background:#111D35;margin:-1rem -1rem 1rem -1rem;">'
        f'<div style="height:100%;width:{progress_pct:.1f}%;'
        f'background:linear-gradient(90deg,#C8A951,#D4BC72);"></div></div>',
        unsafe_allow_html=True,
    )

# --- Plotly Navy+Gold theme helper ---
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#131B2E",
    plot_bgcolor="#0B1120",
    font=dict(family="Inter, sans-serif", color="#E8ECF2", size=11),
    xaxis=dict(gridcolor="#1E2A42", zerolinecolor="#1E2A42"),
    yaxis=dict(gridcolor="#1E2A42", zerolinecolor="#1E2A42"),
    legend=dict(
        font=dict(color="#7B8BA5", size=10),
        bgcolor="rgba(0,0,0,0)",
    ),
)

def apply_navy_theme(fig):
    """Apply Navy+Gold theme to a plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="#1E2A42", zerolinecolor="#1E2A42")
    fig.update_yaxes(gridcolor="#1E2A42", zerolinecolor="#1E2A42")
    return fig

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
        subplot_titles=("Individual Agent Performance", "Pool Size & Events"),
    )

    # --- Top chart: Performance ---

    # Shaded range between worst and best
    fig.add_trace(go.Scatter(
        x=gens + gens[::-1],
        y=best_evals + worst_evals[::-1],
        fill="toself",
        fillcolor="rgba(200, 169, 81, 0.08)",
        line=dict(width=0),
        name="Best-Worst Range",
        showlegend=True,
        hoverinfo="skip",
    ), row=1, col=1)

    # Per-agent individual lines
    agent_colors = CHART_COLORS
    # Collect all agent names across generations
    all_agents_set = set()
    for g in gen_history:
        for name in g.get("agent_eval_scores", {}):
            all_agents_set.add(name)
    all_agent_names = sorted(all_agents_set)

    for idx, agent_name in enumerate(all_agent_names):
        agent_scores = []
        agent_gens = []
        for g in gen_history:
            scores = g.get("agent_eval_scores", {})
            if agent_name in scores:
                agent_gens.append(g["gen"])
                agent_scores.append(scores[agent_name])
        color = agent_colors[idx % len(agent_colors)]
        fig.add_trace(go.Scatter(
            x=agent_gens,
            y=agent_scores,
            mode="lines",
            name=agent_name,
            line=dict(color=color, width=1.5),
            hovertemplate="Gen %{x}<br>" + agent_name + ": %{y:.0f}<extra></extra>",
            opacity=0.7,
        ), row=1, col=1)

    # Best agent envelope (gold dashed)
    fig.add_trace(go.Scatter(
        x=gens,
        y=best_evals,
        mode="lines+markers",
        name="Best Agent",
        line=dict(color=COLORS["gold"], width=3),
        marker=dict(size=6, color=COLORS["gold"]),
        customdata=best_agents,
        hovertemplate="Gen %{x}<br>Best: %{y:.0f}<br>Agent: %{customdata}<extra></extra>",
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
        marker_color="rgba(200, 169, 81, 0.4)",
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
    apply_navy_theme(fig)
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

    # --- P&L and Deployment Monitor ---
    latest_gen = gen_history[-1]
    agent_deployment = latest_gen.get("agent_deployment", {})
    best_ret = latest_gen.get("best_return_pct")
    mean_ret = latest_gen.get("mean_return_pct")

    if agent_deployment or best_ret is not None:
        st.markdown('<div class="section-header">AGENT P&L & DEPLOYMENT</div>', unsafe_allow_html=True)

        # P&L headline
        if best_ret is not None:
            p1, p2 = st.columns(2)
            with p1:
                color = COLORS["green"] if best_ret > 0 else COLORS["red"]
                st.metric(
                    "Best Agent Return",
                    f"{best_ret:+.3f}%",
                    delta="Making money" if best_ret > 0 else "Losing money",
                    delta_color="normal" if best_ret > 0 else "inverse",
                )
            with p2:
                if mean_ret is not None:
                    color = COLORS["green"] if mean_ret > 0 else COLORS["red"]
                    st.metric(
                        "Pool Average Return",
                        f"{mean_ret:+.3f}%",
                        delta="Pool profitable" if mean_ret > 0 else "Pool losing",
                        delta_color="normal" if mean_ret > 0 else "inverse",
                    )

        # Per-agent deployment table
        if agent_deployment:
            st.markdown("**Agent Deployment Status** (latest generation)")
            for agent_name, dep_data in sorted(
                agent_deployment.items(),
                key=lambda x: x[1].get("deployed_pct", 0),
                reverse=True,
            ):
                deployed = dep_data.get("deployed_pct", 0)
                cash = dep_data.get("cash_pct", 100)
                ret = dep_data.get("return_pct", 0)

                # Traffic light for deployment
                if deployed >= 50:
                    dot = ":green_circle:"
                    status = "Active"
                elif deployed >= 20:
                    dot = ":large_yellow_circle:"
                    status = "Cautious"
                else:
                    dot = ":red_circle:"
                    status = "Idle"

                ret_color = COLORS["green"] if ret > 0 else COLORS["red"] if ret < 0 else COLORS["text_muted"]
                st.markdown(
                    f"{dot} **{friendly_name(agent_name)}** — "
                    f"{deployed:.0f}% deployed, {cash:.0f}% cash "
                    f"| Return: <span style='color:{ret_color}'>{ret:+.3f}%</span> "
                    f"| *{status}*",
                    unsafe_allow_html=True,
                )

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
        line=dict(color=COLORS["gold"], width=3),
        fill="tozeroy",
        fillcolor="rgba(200, 169, 81, 0.1)",
    ))
    fig.add_trace(go.Scatter(
        y=spy_equity,
        mode="lines",
        name="S&P 500 (Buy & Hold)",
        line=dict(color=COLORS["blue"], width=2, dash="dash"),
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
    apply_navy_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

# === CAPITAL ALLOCATION ===
ft_allocs = data.get("ft_allocations", [])
if ft_allocs:
    st.subheader("Forward Test Capital Allocation")
    alloc_cols = st.columns(len(ft_allocs))
    for i, alloc in enumerate(ft_allocs):
        with alloc_cols[i]:
            st.metric(
                alloc.get("agent_name", "Unknown"),
                f"${alloc.get('capital', 0):,.0f}",
                delta=f"{alloc.get('weight', 0) * 100:.0f}% weight",
            )
            st.caption(f"Sharpe: {alloc.get('sharpe', 0):.2f}")

    # Donut chart
    fig_alloc = go.Figure(data=[go.Pie(
        labels=[a.get("agent_name", "") for a in ft_allocs],
        values=[a.get("capital", 0) for a in ft_allocs],
        hole=0.5,
        marker=dict(colors=[COLORS["gold"], COLORS["blue"], COLORS["green"],
                            COLORS["purple"], COLORS["pink"]][:len(ft_allocs)]),
        textinfo="label+percent",
        textfont=dict(size=12),
    )])
    fig_alloc.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    apply_navy_theme(fig_alloc)
    st.plotly_chart(fig_alloc, use_container_width=True)
    st.divider()

# === FORWARD TEST — LIVE PAPER TRADING ===
ft = data.get("forward_test", {})
if ft.get("active"):
    st.subheader("Forward Test — Live Paper Trading")

    # Status indicator
    ft_status = ft.get("status", "inactive")
    status_map = {
        "waiting_for_market": (":large_yellow_circle:", "Waiting for Market Open"),
        "trading": (":green_circle:", "Actively Trading"),
        "halted": (":red_circle:", "Emergency Halt"),
    }
    icon, label = status_map.get(ft_status, (":white_circle:", ft_status.replace("_", " ").title()))

    ft_col1, ft_col2, ft_col3, ft_col4 = st.columns(4)

    with ft_col1:
        st.metric("Status", label)
    with ft_col2:
        st.metric(
            "Day",
            f"{ft['days_elapsed']}/{ft['duration_days']}",
            delta=f"{ft['days_remaining']} days left",
        )
    with ft_col3:
        st.metric(
            "Capital Deployed",
            f"${ft['total_capital']:,.0f}",
            delta=f"{len(ft['agents'])} agents",
        )
    with ft_col4:
        started = ft.get("started_at", "")
        if started:
            try:
                start_dt = datetime.fromisoformat(started)
                st.metric("Started", start_dt.strftime("%b %d, %Y"))
            except (ValueError, TypeError):
                st.metric("Started", "Unknown")
        else:
            st.metric("Started", "Unknown")

    # Per-agent sub-account cards
    allocs = ft.get("allocations", {})
    if allocs:
        st.markdown("**Sub-Account Allocations**")
        agent_cols = st.columns(len(allocs))
        for i, (agent_name, alloc) in enumerate(allocs.items()):
            with agent_cols[i]:
                display_name = friendly_name(agent_name)
                capital = alloc.get("capital", 0)
                weight = alloc.get("weight", 0)
                sharpe = alloc.get("sharpe", 0)

                # Check if we have live metrics
                agent_metrics = ft.get("agent_metrics", {}).get(agent_name, {})
                if agent_metrics:
                    current_value = agent_metrics.get("final_value", capital)
                    pnl = current_value - capital
                    pnl_pct = (pnl / capital * 100) if capital > 0 else 0
                    st.metric(
                        display_name,
                        f"${current_value:,.0f}",
                        delta=f"${pnl:+,.0f} ({pnl_pct:+.1f}%)",
                    )
                    st.caption(
                        f"Sharpe: {agent_metrics.get('sharpe', sharpe):.2f} | "
                        f"Win: {agent_metrics.get('win_rate', 0)*100:.0f}% | "
                        f"Trades: {agent_metrics.get('total_trades', 0)}"
                    )
                else:
                    st.metric(
                        display_name,
                        f"${capital:,.0f}",
                        delta=f"{weight*100:.0f}% allocation",
                    )
                    st.caption(f"Backtest Sharpe: {sharpe:.2f}")

    # Combined equity curve (if data available)
    combined = ft.get("combined_equity", [])
    if combined and len(combined) > 1:
        st.markdown("**Portfolio Equity Curve**")
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            y=combined,
            mode="lines",
            name="Combined Portfolio",
            line=dict(color=COLORS["gold"], width=2),
            fill="tozeroy",
            fillcolor="rgba(200, 169, 81, 0.1)",
        ))
        fig_eq.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="Portfolio Value ($)",
            xaxis_title="Bars",
            yaxis_tickprefix="$",
            hovermode="x unified",
        )
        apply_navy_theme(fig_eq)
        st.plotly_chart(fig_eq, use_container_width=True)

    # Daily snapshots table (if available)
    snapshots = ft.get("daily_snapshots", [])
    if snapshots:
        st.markdown("**Daily Performance Log**")
        for snap in reversed(snapshots[-7:]):  # Show last 7 days
            date = snap.get("date", snap.get("logged_at", ""))[:10]
            pv = snap.get("portfolio_value", 0)
            daily_ret = snap.get("daily_return", 0)
            color = COLORS["green"] if daily_ret >= 0 else COLORS["red"]
            st.markdown(
                f"**{date}** — "
                f"<span style='color:{color}'>{daily_ret:+.2%}</span> "
                f"| Portfolio: ${pv:,.0f}",
                unsafe_allow_html=True,
            )

    # Waiting message if no trading data yet
    if ft_status == "waiting_for_market":
        st.info(
            "Forward test is initialized and waiting for market hours. "
            "Trading begins at the next market open (Mon-Fri, 9:30 AM ET). "
            "Data will appear here once the first trades execute."
        )

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
            alloc_text = ""
            if agent.get("allocation_pct", 0) > 0:
                alloc_text = f" | Allocation: {agent['allocation_pct']:.0f}% (${agent['allocation_cad']:,.0f})"
            st.markdown(
                f"**#{i+1}** {agent['name']}{star} — "
                f"<span style='color:{color}'>{agent['return_pct']:+.2f}% "
                f"(${agent['return_cad']:+,.2f})</span> "
                f"| Win Rate: {agent['win_rate_pct']:.0f}% "
                f"{alloc_text}"
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
    for alert in data["alerts"][:15]:
        icon_map = {
            "star": ":star:",
            "arrow_down": ":arrow_down:",
            "warning": ":warning:",
            "rocket": ":rocket:",
            "chart_decreasing": ":chart_with_downwards_trend:",
            "pause": ":no_entry_sign:",
            "trending_down": ":chart_with_downwards_trend:",
            "trending_up": ":chart_with_upwards_trend:",
        }
        icon = icon_map.get(alert.get("icon", ""), ":information_source:")
        st.markdown(f"{icon} {alert['message']}")
else:
    st.info("No events yet. Events appear after training cycles complete.")

st.divider()

# === CORP AGENT RECOMMENDATIONS ===
corp_recs = data.get("corp_recommendations", [])
if corp_recs:
    st.subheader("Agent Recommendations")

    # Group by type for cleaner display
    strategy_recs = [r for r in corp_recs if r["type"] == "strategy"]
    risk_recs = [r for r in corp_recs if r["type"] == "risk_review"]
    tool_recs = [r for r in corp_recs if r["type"] == "tool_recommendation"]
    other_recs = [r for r in corp_recs if r["type"] not in ("strategy", "risk_review", "tool_recommendation", "regime")]
    regime_recs = [r for r in corp_recs if r["type"] == "regime"]

    # Strategy recommendations from Hedge Fund Director
    if strategy_recs:
        for rec in strategy_recs[-3:]:  # Show last 3
            ts = rec["timestamp"][:10] if rec["timestamp"] else ""
            action_badge = " :arrow_forward:" if rec["has_action"] else ""
            st.markdown(
                f":chart_with_upwards_trend: **{rec['source']}** ({ts}){action_badge}\n\n"
                f"> {rec['message']}"
            )

    # Risk reviews
    if risk_recs:
        latest_risk = risk_recs[-1]
        ts = latest_risk["timestamp"][:10] if latest_risk["timestamp"] else ""
        st.markdown(
            f":shield: **{latest_risk['source']}** ({ts})\n\n"
            f"> {latest_risk['message']}"
        )

    # Market regime
    if regime_recs:
        latest_regime = regime_recs[-1]
        ts = latest_regime["timestamp"][:10] if latest_regime["timestamp"] else ""
        st.markdown(
            f":earth_americas: **{latest_regime['source']}** ({ts})\n\n"
            f"> {latest_regime['message']}"
        )

    # Tool/innovation recommendations
    if tool_recs:
        with st.expander(f"Innovation Scout ({len(tool_recs)} tool recommendations)"):
            for rec in tool_recs:
                st.markdown(f"- {rec['message']}")

    # Operations / Performance alerts
    if other_recs:
        with st.expander(f"Other Alerts ({len(other_recs)})"):
            for rec in other_recs[-5:]:
                st.markdown(f"- **{rec['source']}**: {rec['message']}")

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
st.markdown(f"""
<div class="hydra-footer">
  HYDRACORP Executive Dashboard &mdash; CHIMERA + PROMETHEUS + ELEOS + ATHENA + KRONOS
  &nbsp;|&nbsp; Auto-refresh: {REFRESH_INTERVAL_MS // 1000}s
  &nbsp;|&nbsp; Data: {'Real Market' if data['real_data'] else 'Backtesting Simulation'}
</div>
""", unsafe_allow_html=True)
