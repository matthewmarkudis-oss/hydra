"""Multi-objective fitness decomposition ported from CHIMERA.

Scores agents on 5 weighted components: sharpe, drawdown, profit_factor,
walk-forward efficiency, and consistency. A stability multiplier penalizes
high variance across evaluation windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# Default fitness weights (same as CHIMERA's EvolutionState defaults)
DEFAULT_FITNESS_WEIGHTS: dict[str, float] = {
    "sharpe": 0.35,
    "max_dd": 0.20,
    "profit_factor": 0.20,
    "wfe": 0.15,
    "consistency": 0.10,
}

_STABILITY_FLOOR = 0.25


@dataclass
class AgentFitness:
    """Per-agent fitness metrics for a generation."""

    agent_name: str
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    wfe: float = 0.0
    consistency: float = 0.0
    window_sharpes: list[float] = field(default_factory=list)
    total_trades: int = 0
    total_return: float = 0.0
    win_rate: float = 0.0


def compute_fitness(
    metrics: AgentFitness,
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute composite fitness score with multi-objective decomposition.

    Each component is normalized to [0, 1] then weighted. A stability
    multiplier penalizes high variance across evaluation windows.

    Args:
        metrics: Agent performance metrics.
        weights: Optional override for component weights.

    Returns:
        Tuple of (composite_score, breakdown_dict).
    """
    w = weights or DEFAULT_FITNESS_WEIGHTS

    zero_breakdown = {k: 0.0 for k in w}
    zero_breakdown["stability_multiplier"] = 0.0

    # Zero-trade agents get negative fitness — must rank below agents that tried
    if metrics.total_trades == 0:
        return -0.1, zero_breakdown

    def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, v))

    # Normalize each component to [0, 1]
    # Sharpe: range [-2, 2] → [0, 1]
    n_sharpe = clamp(metrics.sharpe / 2.0)

    # Max drawdown: penalty, lower is better
    # Range [0%, 30%] → [0, 1] (inverted)
    n_max_dd = clamp(1.0 - abs(metrics.max_drawdown) / 0.30)

    # Profit factor: range [1, 3], excess above 1.0 is edge
    n_pf = clamp((metrics.profit_factor - 1.0) / 1.0)

    # WFE: range [0, 0.8]
    n_wfe = clamp(metrics.wfe / 0.8)

    # Consistency: already in [0, 1] (fraction of windows profitable)
    n_consistency = clamp(metrics.consistency)

    # Stability multiplier: penalize high variance across windows
    stability = 1.0
    if metrics.window_sharpes and len(metrics.window_sharpes) > 1:
        arr = np.array(metrics.window_sharpes)
        mean_sh = np.mean(arr)
        std_sh = np.std(arr)
        if abs(mean_sh) > 0.01:
            cv = std_sh / abs(mean_sh)  # Coefficient of variation
            stability = clamp(1.0 - cv * 0.3)  # 30% penalty per unit of CV
        stability = max(stability, _STABILITY_FLOOR)

    # Build breakdown
    breakdown = {
        "sharpe": round(n_sharpe * w.get("sharpe", 0), 4),
        "max_dd": round(n_max_dd * w.get("max_dd", 0), 4),
        "profit_factor": round(n_pf * w.get("profit_factor", 0), 4),
        "wfe": round(n_wfe * w.get("wfe", 0), 4),
        "consistency": round(n_consistency * w.get("consistency", 0), 4),
        "stability_multiplier": round(stability, 4),
    }

    # Raw weighted sum, then apply stability
    raw = sum(v for k, v in breakdown.items() if k != "stability_multiplier")
    score = round(raw * stability, 4)

    return score, breakdown


def rank_agents(
    agent_fitnesses: list[AgentFitness],
    weights: dict[str, float] | None = None,
) -> list[tuple[str, float, dict[str, float]]]:
    """Rank agents by composite fitness score (best first).

    Args:
        agent_fitnesses: List of per-agent fitness metrics.
        weights: Optional weight overrides.

    Returns:
        List of (agent_name, score, breakdown) sorted by score descending.
    """
    scored = []
    for af in agent_fitnesses:
        score, breakdown = compute_fitness(af, weights)
        scored.append((af.agent_name, score, breakdown))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def compare_generations(
    current: list[tuple[str, float, dict]],
    previous: list[tuple[str, float, dict]] | None,
) -> dict[str, Any]:
    """Compare fitness between current and previous generation.

    Returns:
        Dict with deltas, best_agent, direction (improving/declining/stagnant).
    """
    if not current:
        return {"direction": "no_data", "deltas": {}}

    curr_best_score = current[0][1]
    curr_mean_score = sum(s for _, s, _ in current) / len(current)

    if not previous:
        return {
            "direction": "baseline",
            "best_agent": current[0][0],
            "best_score": curr_best_score,
            "mean_score": curr_mean_score,
            "deltas": {},
        }

    prev_best_score = previous[0][1]
    prev_mean_score = sum(s for _, s, _ in previous) / len(previous)

    best_delta = curr_best_score - prev_best_score
    mean_delta = curr_mean_score - prev_mean_score

    # Direction classification
    if abs(mean_delta) < 0.005:
        direction = "stagnant"
    elif mean_delta > 0:
        direction = "improving"
    else:
        direction = "declining"

    return {
        "direction": direction,
        "best_agent": current[0][0],
        "best_score": curr_best_score,
        "mean_score": curr_mean_score,
        "best_delta": round(best_delta, 4),
        "mean_delta": round(mean_delta, 4),
        "deltas": {
            "best_score": round(best_delta, 4),
            "mean_score": round(mean_delta, 4),
        },
    }
