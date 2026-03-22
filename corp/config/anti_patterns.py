"""Anti-Pattern Library — structured runbook of known operational problems.

Each pattern has:
- name: unique identifier
- description: human-readable explanation
- detect(state) -> severity | None: checks training state, returns severity if triggered
- action: prescribed intervention (config patch or alert)
- auto_fix: whether the system can apply the fix without CEO approval

Backtesting and training only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class AntiPattern:
    """A known operational anti-pattern with detection and remediation."""

    name: str
    description: str
    detect: Callable[[dict], dict | None]
    severity: str = "warning"  # "info", "warning", "critical"
    auto_fix: bool = False
    patch: dict[str, Any] = field(default_factory=dict)


def _detect_pool_bloat(state: dict) -> dict | None:
    """Pool growing without bound, slowing generation time."""
    gens = state.get("generations", [])
    if len(gens) < 5:
        return None

    pool_sizes = [g.get("pool_size", 0) for g in gens]
    current = pool_sizes[-1]
    initial = pool_sizes[0]

    if current <= 20:
        return None

    growth_rate = (current - initial) / max(len(pool_sizes) - 1, 1)
    dead_weight = 0
    latest_scores = gens[-1].get("eval_scores", {})
    if latest_scores:
        dead_weight = sum(1 for s in latest_scores.values() if s < -500)

    if current > 25 or (growth_rate > 0.5 and current > 15):
        return {
            "pattern": "pool_bloat",
            "severity": "critical" if current > 30 else "warning",
            "message": (
                f"Pool has {current} agents (started at {initial}), "
                f"growing {growth_rate:+.1f}/gen. "
                f"{dead_weight} agents score below -500."
            ),
            "auto_fix": True,
            "patch": {
                "max_pool_size": 20,
                "bottom_k_demote": 3,
            },
            "metrics": {
                "pool_size": current,
                "initial_size": initial,
                "growth_rate": round(growth_rate, 2),
                "dead_weight": dead_weight,
            },
        }
    return None


def _detect_reward_plateau(state: dict) -> dict | None:
    """Reward not improving over extended window."""
    gens = state.get("generations", [])
    if len(gens) < 12:
        return None

    recent = gens[-10:]
    rewards = [g.get("train_mean_reward", 0) for g in recent]
    first_half = sum(rewards[:5]) / 5
    second_half = sum(rewards[5:]) / 5

    # Less than 5% improvement over 10 gens
    if first_half == 0:
        return None
    improvement = abs((second_half - first_half) / abs(first_half))

    if improvement < 0.05:
        return {
            "pattern": "reward_plateau",
            "severity": "warning",
            "message": (
                f"Mean reward plateaued: {first_half:.0f} -> {second_half:.0f} "
                f"({improvement:.1%} change over 10 gens). "
                f"Consider increasing exploration or mutation rate."
            ),
            "auto_fix": False,
            "patch": {},
            "metrics": {
                "early_reward": round(first_half, 1),
                "late_reward": round(second_half, 1),
                "improvement_pct": round(improvement * 100, 1),
            },
        }
    return None


def _detect_single_agent_dominance(state: dict) -> dict | None:
    """One agent has disproportionate weight for too long."""
    gens = state.get("generations", [])
    if len(gens) < 5:
        return None

    dominant_streak = 0
    dominant_name = None

    for g in gens[-5:]:
        comp = g.get("competition", {})
        weights = comp.get("weights_after", {})
        if not weights:
            continue
        top_agent = max(weights, key=weights.get)
        top_weight = weights[top_agent]
        if top_weight > 0.40:
            if dominant_name == top_agent:
                dominant_streak += 1
            else:
                dominant_name = top_agent
                dominant_streak = 1

    if dominant_streak >= 4 and dominant_name:
        return {
            "pattern": "single_agent_dominance",
            "severity": "warning",
            "message": (
                f"'{dominant_name}' has dominated weights (>40%) "
                f"for {dominant_streak} consecutive gens. "
                f"Pool diversity may be collapsing."
            ),
            "auto_fix": False,
            "patch": {},
            "metrics": {
                "dominant_agent": dominant_name,
                "streak": dominant_streak,
            },
        }
    return None


def _detect_dead_learner(state: dict) -> dict | None:
    """A learning agent with near-zero win rate after many generations."""
    gens = state.get("generations", [])
    if len(gens) < 15:
        return None

    latest = gens[-1]
    conviction = latest.get("conviction", {})

    dead_learners = []
    for name, conv in conviction.items():
        if not isinstance(conv, dict):
            continue
        total = conv.get("total_trades", 0)
        wr = conv.get("overall_win_rate", 0)
        # Only flag learning agents (no _gen suffix = live learner)
        if "_gen" not in name and total >= 15 and wr < 0.05:
            dead_learners.append({"name": name, "win_rate": wr, "trades": total})

    if dead_learners:
        names = ", ".join(d["name"] for d in dead_learners)
        return {
            "pattern": "dead_learner",
            "severity": "warning",
            "message": (
                f"Learning agent(s) {names} have <5% win rate after "
                f"{len(gens)} generations. Consider replacing architecture "
                f"or resetting weights."
            ),
            "auto_fix": False,
            "patch": {},
            "metrics": {"dead_learners": dead_learners},
        }
    return None


def _detect_conviction_collapse(state: dict) -> dict | None:
    """All agents have near-zero conviction — nobody is confident."""
    gens = state.get("generations", [])
    if len(gens) < 10:
        return None

    latest = gens[-1]
    conviction = latest.get("conviction", {})
    if not conviction:
        return None

    convictions = []
    for conv in conviction.values():
        if isinstance(conv, dict):
            convictions.append(conv.get("conviction_scale", conv.get("conviction", 0)))

    if not convictions:
        return None

    avg_conviction = sum(convictions) / len(convictions)
    max_conviction = max(convictions)

    if max_conviction < 0.85 and avg_conviction < 0.82 and len(gens) >= 15:
        return {
            "pattern": "conviction_collapse",
            "severity": "info",
            "message": (
                f"Low conviction across all agents: "
                f"avg={avg_conviction:.3f}, max={max_conviction:.3f}. "
                f"No agent has demonstrated reliable performance."
            ),
            "auto_fix": False,
            "patch": {},
            "metrics": {
                "avg_conviction": round(avg_conviction, 3),
                "max_conviction": round(max_conviction, 3),
            },
        }
    return None


def _detect_no_positive_agents(state: dict) -> dict | None:
    """No agent has a positive eval score after significant training."""
    gens = state.get("generations", [])
    if len(gens) < 20:
        return None

    latest_scores = gens[-1].get("eval_scores", {})
    if not latest_scores:
        return None

    positive = sum(1 for s in latest_scores.values() if s > 0)
    total = len(latest_scores)

    # Also check if ANY gen in last 5 had a positive agent
    any_recent_positive = False
    for g in gens[-5:]:
        es = g.get("eval_scores", {})
        if any(s > 0 for s in es.values()):
            any_recent_positive = True
            break

    if positive == 0 and not any_recent_positive:
        return {
            "pattern": "no_positive_agents",
            "severity": "critical",
            "message": (
                f"No agent has scored positive in the last 5 generations "
                f"(0/{total} agents). After {len(gens)} gens, this may indicate "
                f"the problem is too hard for current config. Consider: "
                f"simpler tickers (ETFs), shorter lookback, or reward tuning."
            ),
            "auto_fix": False,
            "patch": {},
            "metrics": {
                "positive_agents": 0,
                "total_agents": total,
                "generations_trained": len(gens),
            },
        }
    return None


# ── Master Registry ──────────────────────────────────────────────────────

ANTI_PATTERNS: list[Callable[[dict], dict | None]] = [
    _detect_pool_bloat,
    _detect_reward_plateau,
    _detect_single_agent_dominance,
    _detect_dead_learner,
    _detect_conviction_collapse,
    _detect_no_positive_agents,
]


def scan_all(training_state: dict) -> list[dict]:
    """Run all anti-pattern detectors against the current training state.

    Returns list of triggered patterns, sorted by severity (critical first).
    """
    triggered = []
    for detector in ANTI_PATTERNS:
        try:
            result = detector(training_state)
            if result:
                triggered.append(result)
        except Exception:
            pass  # Don't let a broken detector crash the monitor

    severity_order = {"critical": 0, "warning": 1, "info": 2}
    triggered.sort(key=lambda x: severity_order.get(x.get("severity", "info"), 3))
    return triggered
