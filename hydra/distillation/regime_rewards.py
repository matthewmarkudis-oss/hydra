"""Regime-conditional reward weight multipliers.

Each regime defines multipliers applied to base reward weights.
Base weights come from RewardConfig; these are multiplicative overlays.

Derived from analysis of how top equity hedge funds adjust risk parameters
across different macro regimes (Fung-Hsieh factor analysis + HFRI data).

The "antifragile" regime is inspired by Nassim Taleb's barbell strategy:
protect the core aggressively (tight drawdown limits) but reward outsized
wins (high P&L bonus) and let winners run (low holding penalty). This
trains agents to find asymmetric bets during volatile markets rather than
sitting in cash.

Backtesting and training only.
"""

from __future__ import annotations


REGIME_MULTIPLIERS: dict[str, dict[str, float]] = {
    "risk_on": {
        # Bull market: lean into winners, loosen drawdown, push for returns
        "drawdown_penalty": 0.5,        # Was 0.8 — accept more DD in favorable conditions
        "transaction_penalty": 0.8,     # Was 1.0 — cheaper to rebalance
        "holding_penalty": 0.5,         # Was 1.0 — allow concentration on winners
        "pnl_bonus_weight": 1.8,        # Was 1.2 — aggressively reward profits
        "sharpe_eta": 0.8,              # Was 1.0 — slower EMA = smoother in trending markets
        "reward_scale": 1.2,            # Was 1.0 — amplify signal during good times
    },
    "risk_off": {
        # Defensive: tighten risk, reduce reward for small wins, protect capital
        "drawdown_penalty": 3.0,        # Was 1.5 — hard punishment for losses
        "transaction_penalty": 0.5,     # Was 0.8 — let agents exit positions cheaply
        "holding_penalty": 1.5,         # Was 1.2 — diversify away concentration risk
        "pnl_bonus_weight": 0.3,        # Was 0.7 — don't chase small wins in bad markets
        "sharpe_eta": 1.5,              # Was 1.2 — faster EMA to react to regime shifts
        "reward_scale": 0.7,            # Was 0.9 — dampen signal to prevent overtrading
    },
    "crisis": {
        # Survival: capital preservation above all, only trade to exit
        "drawdown_penalty": 5.0,        # Was 2.5 — maximum loss aversion
        "transaction_penalty": 0.3,     # Was 0.5 — free to liquidate positions
        "holding_penalty": 2.0,         # Was 1.5 — no concentrated bets
        "pnl_bonus_weight": 0.1,        # Was 0.3 — almost no reward for profits
        "sharpe_eta": 2.0,              # Was 1.5 — very fast EMA, respond immediately
        "reward_scale": 0.4,            # Was 0.7 — heavy signal dampening
    },
    "antifragile": {
        # Taleb barbell: tight ruin protection + massive upside rewards
        # The asymmetry is the point: protect the core, then swing hard
        "drawdown_penalty": 3.0,        # Was 2.0 — tight ruin protection (safe end)
        "transaction_penalty": 1.5,     # Was 1.3 — force conviction, no panic trading
        "holding_penalty": 0.3,         # Was 0.6 — LET WINNERS RUN
        "pnl_bonus_weight": 3.0,        # Was 1.8 — massive reward for outsized wins
        "sharpe_eta": 1.3,              # Slightly faster EMA for volatile conditions
        "reward_scale": 1.5,            # Was 1.1 — amplify everything in volatile markets
    },
}


def get_multipliers(regime: str) -> dict[str, float]:
    """Get reward weight multipliers for a given market regime.

    Args:
        regime: One of "risk_on", "risk_off", "crisis", "antifragile".

    Returns:
        Dict mapping reward parameter names to multiplicative factors.
        Unknown regimes default to risk_on (near-neutral multipliers).
    """
    return REGIME_MULTIPLIERS.get(regime, REGIME_MULTIPLIERS["risk_on"])
