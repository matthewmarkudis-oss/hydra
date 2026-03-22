"""Bayesian conviction calibration ported from ELEOS.

Calibrates agent action confidence using a Beta posterior over historical
win/loss outcomes. Agents that historically perform well in certain
conditions get amplified; those that don't get dampened.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("hydra.evaluation.conviction")


@dataclass
class TradeOutcome:
    """Record of a single trade outcome for conviction tracking."""

    agent_name: str
    step: int = 0
    reward: float = 0.0
    is_win: bool = False
    action_magnitude: float = 0.0
    market_regime: str = "unknown"


class AgentConvictionTracker:
    """Bayesian conviction calibration for a single RL agent.

    Uses a Beta distribution posterior to calibrate how much to trust
    an agent's actions based on its historical performance.

    Prior: Beta(alpha_prior, beta_prior) — defaults to Beta(2, 2).
    Updated with wins/losses from evaluation episodes.
    Posterior mean → conviction scale applied to agent weight.

    Trust gating (production-readiness): new agents with fewer than
    `trust_threshold` observations start at `new_agent_scale` (default 0.50)
    so they don't get full allocation until they've proven themselves.

    Args:
        agent_name: Name of the tracked agent.
        alpha_prior: Beta prior alpha (wins).
        beta_prior: Beta prior beta (losses).
        learning_rate: Dampening factor for conviction adjustments (0-1).
        max_adjustment: Maximum scale factor change from 1.0.
        new_agent_scale: Starting allocation multiplier for untrusted agents.
        trust_threshold: Min observations before agent gets full conviction scaling.
    """

    def __init__(
        self,
        agent_name: str,
        alpha_prior: float = 2.0,
        beta_prior: float = 2.0,
        learning_rate: float = 0.25,
        max_adjustment: float = 0.20,
        new_agent_scale: float = 0.50,
        trust_threshold: int = 5,
    ):
        self.agent_name = agent_name
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.learning_rate = learning_rate
        self.max_adjustment = max_adjustment
        self.new_agent_scale = new_agent_scale
        self.trust_threshold = trust_threshold

        # Running evidence counters
        self._total_wins: int = 0
        self._total_losses: int = 0
        self._by_regime: dict[str, dict[str, int]] = {}
        self._recent_outcomes: list[bool] = []  # last N outcomes
        self._max_recent: int = 100

    def record_outcome(self, outcome: TradeOutcome) -> None:
        """Record a trade outcome for future conviction updates."""
        if outcome.is_win:
            self._total_wins += 1
        else:
            self._total_losses += 1

        # Track by regime
        regime = outcome.market_regime
        if regime not in self._by_regime:
            self._by_regime[regime] = {"wins": 0, "losses": 0}
        if outcome.is_win:
            self._by_regime[regime]["wins"] += 1
        else:
            self._by_regime[regime]["losses"] += 1

        # Rolling recent window
        self._recent_outcomes.append(outcome.is_win)
        if len(self._recent_outcomes) > self._max_recent:
            self._recent_outcomes.pop(0)

    @property
    def is_trusted(self) -> bool:
        """Whether this agent has enough history to be fully trusted."""
        return self.total_trades >= self.trust_threshold

    def get_conviction_scale(
        self,
        regime: str | None = None,
        min_trades: int | None = None,
    ) -> float:
        """Compute Bayesian conviction scale factor for this agent.

        Scale > 1.0 means amplify this agent's weight.
        Scale < 1.0 means dampen this agent's weight.

        Trust gating: agents below trust_threshold get new_agent_scale
        (default 0.50) instead of 1.0, so new/unproven agents start at
        reduced allocation rather than full weight.

        Args:
            regime: Optional market regime for regime-specific calibration.
            min_trades: Deprecated — use trust_threshold in __init__ instead.
                If provided, overrides trust_threshold for backward compat.

        Returns:
            Scale factor in [new_agent_scale, 1 + max_adjustment].
        """
        threshold = min_trades if min_trades is not None else self.trust_threshold
        total_trades = self._total_wins + self._total_losses
        if total_trades < threshold:
            return self.new_agent_scale

        # Base posterior from all trades
        alpha_post = self.alpha_prior + self._total_wins
        beta_post = self.beta_prior + self._total_losses

        # If regime specified, weight regime evidence 2x
        if regime and regime in self._by_regime:
            rb = self._by_regime[regime]
            alpha_post += rb["wins"]  # double-counted on purpose (2x regime weight)
            beta_post += rb["losses"]

        # Posterior mean of Beta distribution
        posterior_mean = alpha_post / (alpha_post + beta_post)

        # Scale relative to 0.5 (neutral)
        raw_scale = posterior_mean / 0.50
        # Apply learning rate dampening
        scale = 1.0 + self.learning_rate * (raw_scale - 1.0)
        # Clamp
        scale = max(1.0 - self.max_adjustment, min(1.0 + self.max_adjustment, scale))

        return round(scale, 4)

    def get_rolling_win_rate(self, window: int = 30) -> float:
        """Get recent win rate over last N outcomes."""
        if not self._recent_outcomes:
            return 0.5
        recent = self._recent_outcomes[-window:]
        return sum(recent) / len(recent)

    @property
    def total_trades(self) -> int:
        return self._total_wins + self._total_losses

    @property
    def overall_win_rate(self) -> float:
        total = self.total_trades
        if total == 0:
            return 0.5
        return self._total_wins / total

    def get_summary(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "total_trades": self.total_trades,
            "total_wins": self._total_wins,
            "total_losses": self._total_losses,
            "overall_win_rate": round(self.overall_win_rate, 4),
            "rolling_win_rate": round(self.get_rolling_win_rate(), 4),
            "conviction_scale": self.get_conviction_scale(),
            "trusted": self.is_trusted,
            "regimes": dict(self._by_regime),
        }


class ConvictionCalibrator:
    """Manages conviction tracking for all agents in the pool.

    Integrates with AgentPool to apply Bayesian conviction scaling
    to agent weights during action aggregation.
    """

    def __init__(
        self,
        alpha_prior: float = 2.0,
        beta_prior: float = 2.0,
        learning_rate: float = 0.25,
        max_adjustment: float = 0.20,
        new_agent_scale: float = 0.50,
        trust_threshold: int = 5,
        persistence_dir: str | None = None,
    ):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.learning_rate = learning_rate
        self.max_adjustment = max_adjustment
        self.new_agent_scale = new_agent_scale
        self.trust_threshold = trust_threshold
        self.persistence_dir = Path(persistence_dir) if persistence_dir else None

        self._trackers: dict[str, AgentConvictionTracker] = {}

    def get_tracker(self, agent_name: str) -> AgentConvictionTracker:
        """Get or create conviction tracker for an agent."""
        if agent_name not in self._trackers:
            self._trackers[agent_name] = AgentConvictionTracker(
                agent_name=agent_name,
                alpha_prior=self.alpha_prior,
                beta_prior=self.beta_prior,
                learning_rate=self.learning_rate,
                max_adjustment=self.max_adjustment,
                new_agent_scale=self.new_agent_scale,
                trust_threshold=self.trust_threshold,
            )
        return self._trackers[agent_name]

    def record_episode_outcomes(
        self,
        agent_rewards: dict[str, list[float]],
        threshold: float = 0.0,
        regime: str = "unknown",
    ) -> None:
        """Record outcomes from an evaluation episode for all agents.

        Args:
            agent_rewards: Dict of agent_name → list of per-step rewards.
            threshold: Reward threshold for a "win".
            regime: Current market regime label.
        """
        for agent_name, rewards in agent_rewards.items():
            tracker = self.get_tracker(agent_name)
            for step, r in enumerate(rewards):
                outcome = TradeOutcome(
                    agent_name=agent_name,
                    step=step,
                    reward=r,
                    is_win=r > threshold,
                    action_magnitude=abs(r),
                    market_regime=regime,
                )
                tracker.record_outcome(outcome)

    def get_conviction_weights(
        self,
        base_weights: dict[str, float],
        regime: str | None = None,
    ) -> dict[str, float]:
        """Apply conviction scaling to base weights.

        Args:
            base_weights: Current agent → weight mapping.
            regime: Optional market regime for regime-specific scaling.

        Returns:
            Conviction-adjusted weight dict (normalized to sum to 1).
        """
        adjusted = {}
        for agent_name, base_w in base_weights.items():
            tracker = self.get_tracker(agent_name)
            scale = tracker.get_conviction_scale(regime=regime)
            adjusted[agent_name] = base_w * scale

        # Normalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: round(v / total, 6) for k, v in adjusted.items()}

        return adjusted

    def get_all_summaries(self) -> dict[str, dict]:
        """Get conviction summaries for all tracked agents."""
        return {name: t.get_summary() for name, t in self._trackers.items()}

    def save(self, path: Path | None = None) -> None:
        """Persist conviction state to JSON."""
        path = path or (self.persistence_dir / "conviction_state.json" if self.persistence_dir else None)
        if path is None:
            return

        state = {}
        for name, tracker in self._trackers.items():
            state[name] = {
                "total_wins": tracker._total_wins,
                "total_losses": tracker._total_losses,
                "by_regime": tracker._by_regime,
                "recent_outcomes": tracker._recent_outcomes[-50:],
            }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load(self, path: Path | None = None) -> None:
        """Load conviction state from JSON."""
        path = path or (self.persistence_dir / "conviction_state.json" if self.persistence_dir else None)
        if path is None or not path.exists():
            return

        with open(path) as f:
            state = json.load(f)

        for name, data in state.items():
            tracker = self.get_tracker(name)
            tracker._total_wins = data.get("total_wins", 0)
            tracker._total_losses = data.get("total_losses", 0)
            tracker._by_regime = data.get("by_regime", {})
            tracker._recent_outcomes = data.get("recent_outcomes", [])
