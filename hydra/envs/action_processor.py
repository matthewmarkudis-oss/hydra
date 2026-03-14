"""Continuous action to order conversion with constraint enforcement.

Translates the agent's continuous [-1, +1] action space into concrete
buy/sell fractions, applying risk constraints before execution.
"""

from __future__ import annotations

import numpy as np

from hydra.risk.env_constraints import EnvConstraints


class ActionProcessor:
    """Processes raw agent actions into executable order fractions.

    Action space: Box([-1, +1], shape=(num_stocks,))
    - Positive values: buy fraction of available cash allocated to this stock
    - Negative values: sell fraction of current holdings for this stock
    - Values near zero: hold (dead zone applied)
    """

    def __init__(
        self,
        num_stocks: int,
        constraints: EnvConstraints,
        dead_zone: float = 0.0,
        min_action: float = 0.02,
    ):
        self.num_stocks = num_stocks
        self.constraints = constraints
        self.dead_zone = np.float32(dead_zone)
        self.min_action = np.float32(min_action)

    def process(
        self,
        raw_actions: np.ndarray,
        holdings: np.ndarray,
        prices: np.ndarray,
        portfolio_value: float,
    ) -> np.ndarray:
        """Process raw actions into clipped order fractions.

        Args:
            raw_actions: Agent output, shape (num_stocks,), values in [-1, 1].
            holdings: Current share holdings per stock.
            prices: Current prices per stock.
            portfolio_value: Current total portfolio value.

        Returns:
            Processed action fractions, shape (num_stocks,), values in [-1, 1].
        """
        # Clip to valid range
        actions = np.clip(raw_actions, -1.0, 1.0).astype(np.float32)

        # Apply dead zone — small actions become zero (reduces churn)
        mask = np.abs(actions) < self.dead_zone
        actions[mask] = 0.0

        # Rescale surviving actions to remove dead zone gap
        pos_mask = actions > 0
        neg_mask = actions < 0
        if np.any(pos_mask):
            actions[pos_mask] = (actions[pos_mask] - self.dead_zone) / (1.0 - self.dead_zone)
        if np.any(neg_mask):
            actions[neg_mask] = (actions[neg_mask] + self.dead_zone) / (1.0 - self.dead_zone)

        # Bump weak-but-nonzero actions up to minimum magnitude
        if self.min_action > 0:
            weak = (np.abs(actions) > 0) & (np.abs(actions) < self.min_action)
            if np.any(weak):
                actions[weak] = np.sign(actions[weak]) * self.min_action

        # Apply risk constraints (position size limits)
        actions = self.constraints.clip_actions(actions, holdings, prices, portfolio_value)

        return actions

    def get_action_space_info(self) -> dict:
        """Return info about the action space for logging."""
        return {
            "shape": (self.num_stocks,),
            "low": -1.0,
            "high": 1.0,
            "dead_zone": float(self.dead_zone),
            "min_action": float(self.min_action),
        }
