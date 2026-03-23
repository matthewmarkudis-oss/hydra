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
        "drawdown_penalty": 0.8,
        "transaction_penalty": 1.0,
        "holding_penalty": 1.0,
        "pnl_bonus_weight": 1.2,
        "sharpe_eta": 1.0,
        "reward_scale": 1.0,
    },
    "risk_off": {
        "drawdown_penalty": 1.5,
        "transaction_penalty": 0.8,
        "holding_penalty": 1.2,
        "pnl_bonus_weight": 0.7,
        "sharpe_eta": 1.2,
        "reward_scale": 0.9,
    },
    "crisis": {
        "drawdown_penalty": 2.5,
        "transaction_penalty": 0.5,
        "holding_penalty": 1.5,
        "pnl_bonus_weight": 0.3,
        "sharpe_eta": 1.5,
        "reward_scale": 0.7,
    },
    "antifragile": {
        # Taleb barbell: protect the core, swing hard on asymmetric bets
        "drawdown_penalty": 2.0,       # Tight ruin protection (the "safe" end)
        "transaction_penalty": 1.3,     # Discourage panic trading, force conviction
        "holding_penalty": 0.6,         # Let winners run — don't punish holding
        "pnl_bonus_weight": 1.8,        # Reward outsized wins (the "aggressive" end)
        "sharpe_eta": 1.3,              # Slightly faster EMA for volatile conditions
        "reward_scale": 1.1,            # Amplify signal — volatile markets need stronger gradients
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
