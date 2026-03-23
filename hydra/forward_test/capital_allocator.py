"""Sharpe-weighted capital allocator for forward test sub-accounts.

Takes ATHENA validation results and allocates capital to agents that
passed validation, proportional to their Sharpe ratio. Agents below
the minimum Sharpe threshold receive zero allocation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("hydra.forward_test.capital_allocator")


@dataclass
class Allocation:
    """Capital allocation for a single agent."""

    agent_name: str
    sharpe: float
    capital: float  # absolute dollar amount
    weight: float  # fraction of total capital [0, 1]
    passed_validation: bool


def compute_allocations(
    agent_results: dict[str, dict[str, Any]],
    total_capital: float,
    min_sharpe: float = 0.3,
    max_agents: int = 3,
    min_allocation_pct: float = 0.05,
) -> list[Allocation]:
    """Compute Sharpe-weighted capital allocations.

    Args:
        agent_results: Dict of agent_name -> validation result dict.
            Each must have 'sharpe' and 'passed' keys.
        total_capital: Total capital to allocate across agents.
        min_sharpe: Minimum Sharpe ratio to receive any allocation.
        max_agents: Maximum number of agents to allocate to (top-K).
        min_allocation_pct: Minimum allocation percentage. Agents below
            this threshold after Sharpe-weighting get folded into others.

    Returns:
        List of Allocation objects, sorted by weight descending.
        Only agents with weight > 0 are included.
    """
    # Step 1: Filter to passed agents above min Sharpe
    candidates = []
    for name, result in agent_results.items():
        if not result.get("passed", False):
            continue
        sharpe = result.get("sharpe", 0.0)
        if sharpe < min_sharpe:
            continue
        candidates.append((name, sharpe))

    if not candidates:
        logger.warning("No agents passed Sharpe threshold %.2f", min_sharpe)
        return []

    # Step 2: Sort by Sharpe descending, take top-K
    candidates.sort(key=lambda x: x[1], reverse=True)
    candidates = candidates[:max_agents]

    # Step 3: Compute raw weights proportional to Sharpe
    total_sharpe = sum(s for _, s in candidates)
    if total_sharpe <= 0:
        logger.warning("Total Sharpe <= 0 among candidates")
        return []

    raw_weights = [(name, sharpe, sharpe / total_sharpe) for name, sharpe in candidates]

    # Step 4: Enforce minimum allocation — redistribute crumbs
    above_min = [(name, sharpe, w) for name, sharpe, w in raw_weights if w >= min_allocation_pct]
    below_min = [(name, sharpe, w) for name, sharpe, w in raw_weights if w < min_allocation_pct]

    if not above_min:
        # All below minimum — give equal weight to all candidates
        equal_w = 1.0 / len(candidates)
        above_min = [(name, sharpe, equal_w) for name, sharpe, _ in raw_weights]
        below_min = []

    # Redistribute crumbs proportionally among remaining
    redistrib = sum(w for _, _, w in below_min)
    if redistrib > 0 and above_min:
        remaining_total = sum(w for _, _, w in above_min)
        above_min = [
            (name, sharpe, w + redistrib * (w / remaining_total))
            for name, sharpe, w in above_min
        ]

    # Step 5: Build Allocation objects
    allocations = []
    for name, sharpe, weight in above_min:
        allocations.append(Allocation(
            agent_name=name,
            sharpe=sharpe,
            capital=round(total_capital * weight, 2),
            weight=round(weight, 6),
            passed_validation=True,
        ))

    allocations.sort(key=lambda a: a.weight, reverse=True)

    for a in allocations:
        logger.info(
            "Allocation: %s — Sharpe=%.3f, weight=%.1f%%, capital=$%.2f",
            a.agent_name, a.sharpe, a.weight * 100, a.capital,
        )

    return allocations
