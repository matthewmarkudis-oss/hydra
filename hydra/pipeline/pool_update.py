"""Phase 5: Pool update — promote/demote/freeze agents based on eval results."""

from __future__ import annotations

import logging
from typing import Any

from hydra.agents.agent_pool import AgentPool

logger = logging.getLogger("hydra.pipeline.pool_update")


def update_pool(
    deps: dict[str, Any],
    top_k_promote: int = 2,
    bottom_k_demote: int = 1,
) -> dict[str, Any]:
    """Update agent pool based on evaluation results.

    Ranks agents by evaluation performance, promotes best learning agents
    to static snapshots, and removes worst static agents.

    Args:
        deps: Must contain 'eval_phase' with rl_eval results,
              and 'train_phase' with pool.

    Returns:
        Dict with promotion/demotion results and updated rankings.
    """
    eval_result = deps.get("eval_phase", {})
    train_result = deps.get("train_phase", {})

    pool: AgentPool = train_result.get("pool")
    rl_eval = eval_result.get("rl_eval", {})

    if pool is None:
        raise ValueError("No pool found in dependencies")

    # Extract scores (use mean reward as primary metric)
    scores = {}
    for agent_name, metrics in rl_eval.items():
        scores[agent_name] = metrics.get("mean_reward", 0.0)

    # Update rankings
    pool.update_rankings(scores)

    logger.info("Agent rankings:")
    for name, score in pool.get_ranked_agents():
        agent = pool.get(name)
        agent_type = agent.__class__.__name__ if agent else "unknown"
        logger.info(f"  {name} ({agent_type}): {score:.4f}")

    # Promote and demote
    promoted = pool.promote_top(top_k_promote)
    demoted = pool.demote_bottom(bottom_k_demote)

    return {
        "scores": scores,
        "rankings": dict(pool.get_ranked_agents()),
        "promoted": promoted,
        "demoted": demoted,
        "pool_size": pool.size,
        "pool": pool,
    }
