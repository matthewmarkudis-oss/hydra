"""Dashboard configuration — colors, friendly names, CAD conversion."""

# Starting capital in CAD
STARTING_CAPITAL_CAD = 2_500.00

# Internal simulation capital (used by RL environment)
INTERNAL_CAPITAL = 100_000.0

# Scale factor: convert internal dollars to CAD
SCALE_FACTOR = STARTING_CAPITAL_CAD / INTERNAL_CAPITAL  # 0.025

# Dashboard port
DASHBOARD_PORT = 5050

# Auto-refresh interval (milliseconds)
REFRESH_INTERVAL_MS = 60_000  # 60 seconds

# State file paths
TRAINING_STATE_FILE = "logs/hydra_training_state.json"
CORP_STATE_FILE = "logs/corporation_state.json"

# Color palette
COLORS = {
    "green": "#00C853",
    "red": "#FF1744",
    "amber": "#FFD600",
    "blue": "#2979FF",
    "orange": "#FF9100",
    "gray": "#616161",
    "white": "#FAFAFA",
    "bg_dark": "#0F1117",
    "bg_card": "#1E1E2E",
}

# Friendly agent names
FRIENDLY_NAMES = {
    "ppo_0": "AI Alpha",
    "ppo_1": "AI Alpha-2",
    "sac_0": "AI Beta",
    "sac_1": "AI Beta-2",
    "rppo_0": "AI Gamma",
    "rppo_1": "AI Gamma-2",
    "alpha_rule": "Momentum Strategy",
    "beta_rule": "Mean Reversion Strategy",
    "gamma_rule": "Breakout Strategy",
    "delta_rule": "VWAP Strategy",
    "epsilon_rule": "Microstructure Strategy",
    "zeta_rule": "Regime Strategy",
    "eta_rule": "Liquidity Strategy",
    "theta_rule": "Stat Arb Strategy",
    "iota_rule": "Order Flow Strategy",
    "static_0": "Baseline Snapshot A",
    "static_1": "Baseline Snapshot B",
}


def friendly_name(internal_name: str) -> str:
    """Convert internal agent names to CEO-friendly names."""
    if internal_name in FRIENDLY_NAMES:
        return FRIENDLY_NAMES[internal_name]

    # Handle generational snapshots like ppo_1_gen2080
    for key, name in FRIENDLY_NAMES.items():
        if internal_name.startswith(key + "_gen"):
            gen_num = internal_name.split("_gen")[-1]
            return f"{name} (v{gen_num})"

    return internal_name


def compute_portfolio_value(total_return: float) -> float:
    """Convert internal return ratio to CAD portfolio value."""
    return STARTING_CAPITAL_CAD * (1 + total_return)


def compute_dollar_pnl(total_return: float) -> float:
    """Convert return ratio to dollar P&L in CAD."""
    return STARTING_CAPITAL_CAD * total_return


def compute_safety_score(max_drawdown: float, win_rate: float, profit_factor: float) -> int:
    """Compute 0-100 safety score from risk metrics."""
    dd_score = max(0, 100 - (abs(max_drawdown) / 0.25 * 100))
    wr_score = max(0, (win_rate - 0.30) / 0.30 * 100)
    pf_score = max(0, (profit_factor - 0.8) / 1.2 * 100)
    return int(min(100, dd_score * 0.4 + wr_score * 0.3 + pf_score * 0.3))


def safety_label(score: int) -> tuple[str, str]:
    """Return (label, color) for a safety score."""
    if score >= 80:
        return "Looking Good", COLORS["green"]
    elif score >= 60:
        return "Watch Closely", COLORS["amber"]
    elif score >= 40:
        return "Elevated Risk", COLORS["orange"]
    else:
        return "High Risk", COLORS["red"]
