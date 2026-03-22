"""Regime-conditional reward weight multipliers.

Each regime defines multipliers applied to base reward weights.
Base weights come from RewardConfig; these are multiplicative overlays.

Derived from analysis of how top equity hedge funds adjust risk parameters
across different macro regimes (Fung-Hsieh factor analysis + HFRI data).

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
}


def get_multipliers(regime: str) -> dict[str, float]:
    """Get reward weight multipliers for a given market regime.

    Args:
        regime: One of "risk_on", "risk_off", "crisis".

    Returns:
        Dict mapping reward parameter names to multiplicative factors.
        Unknown regimes default to risk_on (near-neutral multipliers).
    """
    return REGIME_MULTIPLIERS.get(regime, REGIME_MULTIPLIERS["risk_on"])
