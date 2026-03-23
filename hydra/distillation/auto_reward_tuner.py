"""Auto reward tuner — maps CHIMERA diagnostic mutations to reward parameter adjustments.

Reads mutation recommendations from the diagnostic engine each generation
and applies bounded, directional adjustments to the reward function weights.
This closes the loop between diagnostics and training, replacing manual tuning.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from hydra.evolution.mutation_engine import MutationRecord

logger = logging.getLogger("hydra.distillation.auto_reward_tuner")

# Schema defaults (from hydra/config/schema.py RewardConfig)
_DEFAULTS = {
    "drawdown_penalty": 0.15,
    "transaction_penalty": 0.01,
    "holding_penalty": 0.02,
    "pnl_bonus_weight": 5.0,
    "reward_scale": 100.0,
    "cash_drag_penalty": 0.3,
    "benchmark_bonus_weight": 2.0,
    "min_deployment_pct": 0.3,
}

# Schema bounds (min, max) for each tunable parameter
_BOUNDS = {
    "drawdown_penalty": (0.0, 2.0),
    "transaction_penalty": (0.0, 1.0),
    "holding_penalty": (0.0, 1.0),
    "pnl_bonus_weight": (0.5, 20.0),  # floor at 0.5 to prevent "don't trade"
    "reward_scale": (10.0, 500.0),
    "cash_drag_penalty": (0.05, 1.0),
    "benchmark_bonus_weight": (0.5, 10.0),
    "min_deployment_pct": (0.1, 0.7),
}

# Mutation type → (parameter_name, direction)
# direction: +1 = increase, -1 = decrease
_MUTATION_MAP: dict[str, list[tuple[str, int]]] = {
    "increase_drawdown_penalty": [("drawdown_penalty", +1)],
    "decrease_drawdown_penalty": [("drawdown_penalty", -1)],
    "tighten_risk": [("drawdown_penalty", +1)],
    "loosen_risk": [("drawdown_penalty", -1), ("transaction_penalty", -1)],
    "prioritize_consistency": [("holding_penalty", +1), ("transaction_penalty", +1)],
    "increase_deployment": [("cash_drag_penalty", +1), ("min_deployment_pct", +1)],
    "reward_outperformance": [("benchmark_bonus_weight", +1), ("pnl_bonus_weight", +1)],
}


class AutoRewardTuner:
    """Maps CHIMERA diagnostic mutations to bounded reward parameter adjustments.

    Called every N generations by the population trainer. Reads the list of
    recommended mutations from diagnostics, counts the net direction for each
    reward parameter, and applies a bounded step.
    """

    def __init__(
        self,
        tune_every_n: int = 5,
        max_delta_pct: float = 0.20,
        enabled: bool = True,
    ):
        self.tune_every_n = max(1, tune_every_n)
        self.max_delta_pct = max(0.01, min(max_delta_pct, 0.50))
        self.enabled = enabled
        self._history: list[dict[str, float]] = []

    def should_tune(self, generation: int) -> bool:
        """Check whether tuning should run this generation."""
        if not self.enabled:
            return False
        return generation > 0 and generation % self.tune_every_n == 0

    def apply_mutations(
        self,
        current_params: dict[str, float],
        mutations: list[MutationRecord],
    ) -> dict[str, float]:
        """Compute new reward parameters from diagnostic mutations.

        Args:
            current_params: Current reward params from reward_fn.get_params().
            mutations: List of MutationRecord from diagnostics.

        Returns:
            Updated params dict (same keys as current_params).
        """
        if not mutations:
            return dict(current_params)

        # Check for restore_defaults
        for m in mutations:
            if m.mutation_type == "restore_defaults":
                logger.info("AutoRewardTuner: restoring defaults per diagnostic recommendation")
                self._history.append(dict(_DEFAULTS))
                return dict(_DEFAULTS)

        # Count net direction per parameter
        param_votes: dict[str, int] = Counter()
        total_relevant = 0

        for m in mutations:
            adjustments = _MUTATION_MAP.get(m.mutation_type, [])
            for param_name, direction in adjustments:
                param_votes[param_name] += direction
                total_relevant += 1

        if total_relevant == 0:
            return dict(current_params)

        # Apply adjustments
        new_params = dict(current_params)
        changes: list[str] = []

        for param_name, net_vote in param_votes.items():
            if net_vote == 0 or param_name not in current_params:
                continue

            current_val = current_params[param_name]
            lo, hi = _BOUNDS.get(param_name, (0.0, 100.0))

            # Step proportional to net vote strength relative to total mutations
            strength = abs(net_vote) / max(total_relevant, 1)
            step = self.max_delta_pct * current_val * strength

            if net_vote > 0:
                new_val = current_val + step
            else:
                new_val = current_val - step

            # Clamp to bounds
            new_val = max(lo, min(new_val, hi))
            new_params[param_name] = round(new_val, 6)

            if new_val != current_val:
                direction = "+" if net_vote > 0 else "-"
                changes.append(
                    f"{param_name}: {current_val:.4f} -> {new_val:.4f} ({direction})"
                )

        if changes:
            logger.info(
                f"AutoRewardTuner applied {len(changes)} adjustment(s): "
                + "; ".join(changes)
            )
        else:
            logger.debug("AutoRewardTuner: no effective changes this cycle")

        self._history.append(dict(new_params))
        return new_params

    @property
    def tuning_history(self) -> list[dict[str, float]]:
        """Return the history of applied parameter sets."""
        return list(self._history)
