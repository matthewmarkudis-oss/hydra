"""Competition-based weight rebalancing ported from PROMETHEUS.

Agents compete each generation. Weights are adjusted based on composite
performance scores (Sharpe, win rate, profit factor). Convergence is
checked against target thresholds over consecutive generations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("hydra.evaluation.competition")


@dataclass
class AgentCompetitionScore:
    """Per-agent competition results for one generation."""

    agent_name: str
    sharpe: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0
    trades: int = 0
    composite: float = 0.0  # computed by score()


@dataclass
class GenerationCompetitionResult:
    """Competition results for one generation."""

    generation: int
    agent_scores: list[AgentCompetitionScore] = field(default_factory=list)
    weights_before: dict[str, float] = field(default_factory=dict)
    weights_after: dict[str, float] = field(default_factory=dict)
    aggregate_sharpe: float = 0.0
    aggregate_win_rate: float = 0.0
    aggregate_max_dd: float = 0.0
    aggregate_pf: float = 0.0
    converged: bool = False


class CompetitionRebalancer:
    """PROMETHEUS-style competition-based weight rebalancing.

    Each generation, agents are scored on a composite metric and
    weights are linearly adjusted by rank. Convergence is declared
    when target thresholds are met for N consecutive generations.

    Args:
        min_weight: Floor weight for any agent (prevents zeroing out).
        max_weight: Ceiling weight for any agent.
        adjustment_step: Max weight change per generation rank position.
        sharpe_threshold: Min Sharpe for convergence.
        min_win_rate: Min win rate for convergence.
        max_drawdown_threshold: Max drawdown for convergence.
        min_profit_factor: Min profit factor for convergence.
        required_consecutive: How many consecutive gens must pass.
    """

    def __init__(
        self,
        min_weight: float = 0.01,
        max_weight: float = 0.60,
        adjustment_step: float = 0.10,
        sharpe_threshold: float = 1.0,
        min_win_rate: float = 0.52,
        max_drawdown_threshold: float = 0.18,
        min_profit_factor: float = 1.50,
        required_consecutive: int = 3,
        ema_alpha: float = 0.70,
        max_weight_change: float = 0.30,
    ):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.adjustment_step = adjustment_step
        self.sharpe_threshold = sharpe_threshold
        self.min_win_rate = min_win_rate
        self.max_drawdown_threshold = max_drawdown_threshold
        self.min_profit_factor = min_profit_factor
        self.required_consecutive = required_consecutive

        # Rebalance smoothing (production-readiness)
        # ema_alpha: blend factor — 0.3 means 30% new weights + 70% old weights.
        # max_weight_change: cap per-agent weight change per generation (e.g. 0.10 = ±10pp).
        self.ema_alpha = ema_alpha
        self.max_weight_change = max_weight_change

        self._history: list[GenerationCompetitionResult] = []

    def score_agents(
        self, agents: list[AgentCompetitionScore]
    ) -> list[AgentCompetitionScore]:
        """Compute composite competition score for each agent.

        Composite = 0.40 * sharpe_norm + 0.30 * win_rate + 0.30 * pf_norm

        Returns:
            Agents sorted by composite (best first), with .composite set.
        """
        for a in agents:
            sharpe_score = min(max(a.sharpe, -2), 3) / 3
            wr_score = a.win_rate
            pf_score = min(a.profit_factor / 2, 1.0)
            a.composite = sharpe_score * 0.40 + wr_score * 0.30 + pf_score * 0.30

        agents.sort(key=lambda x: x.composite, reverse=True)
        return agents

    def rebalance_weights(
        self,
        current_weights: dict[str, float],
        ranked_agents: list[AgentCompetitionScore],
    ) -> dict[str, float]:
        """Rebalance weights based on competition rankings.

        Linear adjustment: top-ranked agents gain weight, bottom lose.
        EMA smoothing blends new target with current weights to prevent
        wild swings. Per-generation change cap limits how much any single
        agent's weight can move.

        Args:
            current_weights: Current agent → weight mapping.
            ranked_agents: Agents sorted by composite (best first).

        Returns:
            New normalized weight dict.
        """
        n = len(ranked_agents)
        if n == 0:
            return current_weights

        # Step 1: compute raw target weights from ranking
        raw_weights = dict(current_weights)

        for i, agent in enumerate(ranked_agents):
            name = agent.agent_name
            if name not in raw_weights:
                raw_weights[name] = 1.0 / n

            # Linear: top gets +step, bottom gets -step
            if n > 1:
                adjustment = self.adjustment_step * (1 - 2 * i / (n - 1))
            else:
                adjustment = 0.0

            raw_weights[name] = max(
                self.min_weight,
                min(self.max_weight, raw_weights[name] + adjustment)
            )

        # Step 2: EMA smoothing — blend raw target with current weights
        # new = alpha * raw + (1 - alpha) * current
        smoothed_weights = {}
        for name in raw_weights:
            old_w = current_weights.get(name, raw_weights[name])
            smoothed_weights[name] = (
                self.ema_alpha * raw_weights[name]
                + (1.0 - self.ema_alpha) * old_w
            )

        # Step 3: cap per-generation change
        capped_weights = {}
        for name, new_w in smoothed_weights.items():
            old_w = current_weights.get(name, new_w)
            delta = new_w - old_w
            if abs(delta) > self.max_weight_change:
                capped_delta = self.max_weight_change if delta > 0 else -self.max_weight_change
                capped_weights[name] = old_w + capped_delta
            else:
                capped_weights[name] = new_w
            # Enforce floor/ceiling after capping
            capped_weights[name] = max(self.min_weight, min(self.max_weight, capped_weights[name]))

        # Step 4: normalize to sum to 1.0
        total = sum(capped_weights.values())
        if total > 0:
            capped_weights = {k: round(v / total, 4) for k, v in capped_weights.items()}

        return capped_weights

    def evaluate_generation(
        self,
        generation: int,
        agent_scores: list[AgentCompetitionScore],
        current_weights: dict[str, float],
    ) -> GenerationCompetitionResult:
        """Run full competition for one generation.

        Scores agents, rebalances weights, checks convergence.

        Args:
            generation: Generation number.
            agent_scores: Raw agent metrics for this generation.
            current_weights: Current weight allocation.

        Returns:
            GenerationCompetitionResult with new weights and convergence status.
        """
        # Score and rank
        ranked = self.score_agents(agent_scores)

        # Compute aggregates (weighted by current weights)
        total_w = sum(current_weights.get(a.agent_name, 0) for a in ranked)
        if total_w > 0:
            agg_sharpe = sum(
                a.sharpe * current_weights.get(a.agent_name, 0) for a in ranked
            ) / total_w
            agg_wr = sum(
                a.win_rate * current_weights.get(a.agent_name, 0) for a in ranked
            ) / total_w
            agg_dd = max(
                (abs(a.max_drawdown) for a in ranked), default=0
            )
            agg_pf = sum(
                a.profit_factor * current_weights.get(a.agent_name, 0) for a in ranked
            ) / total_w
        else:
            agg_sharpe = agg_wr = agg_dd = agg_pf = 0.0

        # Rebalance
        new_weights = self.rebalance_weights(current_weights, ranked)

        result = GenerationCompetitionResult(
            generation=generation,
            agent_scores=ranked,
            weights_before=dict(current_weights),
            weights_after=new_weights,
            aggregate_sharpe=agg_sharpe,
            aggregate_win_rate=agg_wr,
            aggregate_max_dd=agg_dd,
            aggregate_pf=agg_pf,
        )

        self._history.append(result)

        # Check convergence
        result.converged = self._check_convergence()

        if result.converged:
            logger.info(
                f"Generation {generation}: CONVERGED — "
                f"Sharpe={agg_sharpe:.3f}, WR={agg_wr:.2%}, "
                f"DD={agg_dd:.2%}, PF={agg_pf:.2f}"
            )
        else:
            logger.info(
                f"Generation {generation}: competition complete — "
                f"Sharpe={agg_sharpe:.3f}, WR={agg_wr:.2%}"
            )

        return result

    def _check_convergence(self) -> bool:
        """Check if last N generations all pass convergence thresholds."""
        if len(self._history) < self.required_consecutive:
            return False

        recent = self._history[-self.required_consecutive:]
        for r in recent:
            if r.aggregate_sharpe < self.sharpe_threshold:
                return False
            if r.aggregate_win_rate < self.min_win_rate:
                return False
            if r.aggregate_max_dd > self.max_drawdown_threshold:
                return False
            if r.aggregate_pf < self.min_profit_factor:
                return False

        return True

    @property
    def history(self) -> list[GenerationCompetitionResult]:
        return list(self._history)

    def get_weight_trajectory(self, agent_name: str) -> list[float]:
        """Get weight history for a specific agent across generations."""
        return [
            r.weights_after.get(agent_name, 0.0) for r in self._history
        ]
