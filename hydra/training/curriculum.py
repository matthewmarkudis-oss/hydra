"""Curriculum for pool composition schedule.

Controls how the agent pool evolves across generations: when to add
new agent types, adjust exploration rates, or change training parameters.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("hydra.training.curriculum")


class Curriculum:
    """Manages the training curriculum across generations.

    Defines phases of training with different pool compositions
    and hyperparameter schedules.
    """

    def __init__(
        self,
        warmup_generations: int = 2,
        exploration_decay: float = 0.95,
        min_exploration: float = 0.01,
    ):
        self.warmup_generations = warmup_generations
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration

        self._current_exploration = 1.0
        self._phase = "warmup"
        self._generation = 0
        self._regime = "risk_on"

    def on_generation(self, generation: int, eval_scores: dict[str, float]) -> dict[str, Any]:
        """Update curriculum state after a generation.

        Args:
            generation: Current generation number.
            eval_scores: Agent evaluation scores from this generation.

        Returns:
            Dict of curriculum adjustments to apply.
        """
        self._generation = generation
        adjustments = {}

        # Phase transitions
        if generation <= self.warmup_generations:
            self._phase = "warmup"
        elif generation <= self.warmup_generations * 3:
            self._phase = "exploration"
        else:
            self._phase = "exploitation"

        # Decay exploration
        self._current_exploration = max(
            self._current_exploration * self.exploration_decay,
            self.min_exploration,
        )

        adjustments["phase"] = self._phase
        adjustments["exploration_rate"] = self._current_exploration
        adjustments["regime"] = self._regime

        logger.info(
            f"Curriculum gen {generation}: phase={self._phase}, "
            f"exploration={self._current_exploration:.4f}"
        )

        return adjustments

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def exploration_rate(self) -> float:
        return self._current_exploration

    @property
    def regime(self) -> str:
        return self._regime

    def set_regime(self, regime: str) -> None:
        """Set the market regime for regime-conditional rewards.

        Invalid regimes are rejected and the regime resets to ``risk_on``.
        """
        valid = ("risk_on", "risk_off", "crisis")
        if regime in valid:
            self._regime = regime
            logger.info(f"Curriculum regime set to: {regime}")
        else:
            self._regime = "risk_on"
            logger.warning(f"Invalid regime '{regime}', reset to 'risk_on'")

    def get_pool_schedule(self, generation: int) -> dict[str, int]:
        """Get target pool composition for a generation.

        Returns target number of each agent type.
        """
        if generation <= self.warmup_generations:
            return {
                "learning": 2,
                "static": 0,
                "rule_based": 2,
            }
        elif generation <= self.warmup_generations * 3:
            return {
                "learning": 2,
                "static": 2,
                "rule_based": 2,
            }
        else:
            return {
                "learning": 2,
                "static": 3,
                "rule_based": 1,
            }
