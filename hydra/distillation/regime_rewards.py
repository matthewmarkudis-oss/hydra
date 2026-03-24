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
        "drawdown_penalty": 0.5,        # Accept more DD in favorable conditions
        "transaction_penalty": 0.8,     # Cheaper to rebalance
        "holding_penalty": 0.5,         # Allow concentration on winners
        "pnl_bonus_weight": 1.8,        # Aggressively reward profits
        "sharpe_eta": 0.8,              # Slower EMA = smoother in trending markets
        "reward_scale": 1.2,            # Amplify signal during good times
        "cash_drag_penalty": 1.5,       # Push harder to deploy in bull markets
    },
    "risk_off": {
        # Defensive: tighter risk but still allow learning
        # Previous values (3.0/0.3/0.7) were too punitive — agents couldn't
        # learn anything useful because profit signals were crushed (0.3x)
        # while loss penalties were amplified (3.0x).
        "drawdown_penalty": 1.5,        # Was 3.0 — cautious but not punitive
        "transaction_penalty": 0.7,     # Was 0.5 — moderate exit cost
        "holding_penalty": 1.2,         # Was 1.5 — mild diversification pressure
        "pnl_bonus_weight": 0.8,        # Was 0.3 — agents still need to learn from profits
        "sharpe_eta": 1.3,              # Was 1.5 — slightly faster EMA
        "reward_scale": 0.85,           # Was 0.7 — slight dampening, not crushing
        "cash_drag_penalty": 0.5,       # Reduce deployment pressure in defensive mode
    },
    "crisis": {
        # Survival: protect capital but still allow learning
        # Previous 0.1x profit / 5.0x drawdown made learning impossible.
        "drawdown_penalty": 2.5,        # Was 5.0 — strong but not crushing
        "transaction_penalty": 0.5,     # Was 0.3 — moderate exit cost
        "holding_penalty": 1.5,         # Was 2.0 — some diversification
        "pnl_bonus_weight": 0.5,        # Was 0.1 — agents must still learn from profits
        "sharpe_eta": 1.5,              # Was 2.0 — fast EMA
        "reward_scale": 0.6,            # Was 0.4 — dampened but not invisible
        "cash_drag_penalty": 0.2,       # Allow defensive cash positioning in crisis
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
        "cash_drag_penalty": 0.8,       # Some pressure but allow dry powder for vol plays
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
