"""Risk constraints enforced inside the environment step function.

These are not post-hoc checks — they directly modify actions and can
terminate episodes, so the agent learns to respect limits.
"""

from __future__ import annotations

import numpy as np


class EnvConstraints:
    """Risk constraints applied within env.step().

    Clips actions to respect position limits, halts trading on drawdown
    or daily loss breaches, and implements circuit breakers.
    """

    def __init__(
        self,
        max_position_pct: float = 0.10,
        max_drawdown_pct: float = 0.10,
        max_daily_loss_pct: float = 0.03,
        circuit_breaker_pct: float = 0.05,
    ):
        self.max_position_pct = np.float32(max_position_pct)
        self.max_drawdown_pct = np.float32(max_drawdown_pct)
        self.max_daily_loss_pct = np.float32(max_daily_loss_pct)
        self.circuit_breaker_pct = np.float32(circuit_breaker_pct)

        # State
        self._peak_value = np.float32(0.0)
        self._initial_value = np.float32(0.0)
        self._halted = False
        self._halt_reason = ""

    def reset(self, initial_value: float) -> None:
        """Reset constraints for a new episode."""
        self._peak_value = np.float32(initial_value)
        self._initial_value = np.float32(initial_value)
        self._halted = False
        self._halt_reason = ""

    def clip_actions(
        self,
        actions: np.ndarray,
        holdings: np.ndarray,
        prices: np.ndarray,
        portfolio_value: float,
    ) -> np.ndarray:
        """Clip actions to respect position size limits.

        Args:
            actions: Raw actions from agent, [-1, +1] per stock.
            holdings: Current share holdings per stock.
            prices: Current prices per stock.
            portfolio_value: Current total portfolio value.

        Returns:
            Clipped actions array.
        """
        if self._halted:
            # If halted, only allow sells (negative actions) to reduce risk
            return np.minimum(actions, np.float32(0.0))

        clipped = actions.copy()
        pv = max(portfolio_value, 1e-8)
        max_value = np.float32(pv * self.max_position_pct)

        for i in range(len(actions)):
            current_value = holdings[i] * prices[i]

            if actions[i] > 0:
                # Buying — check if it would exceed max position
                headroom = max_value - current_value
                if headroom <= 0:
                    clipped[i] = 0.0  # Already at max, can't buy more
                else:
                    # Scale down buy action if needed
                    max_buy_frac = headroom / pv
                    clipped[i] = min(actions[i], max_buy_frac)

        return clipped

    def check_constraints(
        self,
        portfolio_value: float,
        step_return: float,
    ) -> tuple[bool, bool, dict[str, float]]:
        """Check all risk constraints after a step.

        Args:
            portfolio_value: Current portfolio value.
            step_return: Return for this step.

        Returns:
            Tuple of (should_truncate, should_halt, info_dict).
        """
        pv = np.float32(portfolio_value)
        self._peak_value = max(self._peak_value, pv)

        info = {}

        # Drawdown check
        drawdown = (pv - self._peak_value) / max(float(self._peak_value), 1e-8)
        info["drawdown"] = float(drawdown)

        if -drawdown >= self.max_drawdown_pct:
            self._halted = True
            self._halt_reason = "max_drawdown"
            info["halt_reason"] = "max_drawdown"
            return True, True, info

        # Daily loss check
        daily_loss = (pv - self._initial_value) / max(float(self._initial_value), 1e-8)
        info["daily_pnl"] = float(daily_loss)

        if -daily_loss >= self.max_daily_loss_pct:
            self._halted = True
            self._halt_reason = "max_daily_loss"
            info["halt_reason"] = "max_daily_loss"
            return True, True, info

        # Circuit breaker (large single-step loss)
        if step_return < -self.circuit_breaker_pct:
            self._halted = True
            self._halt_reason = "circuit_breaker"
            info["halt_reason"] = "circuit_breaker"
            return False, True, info  # Don't truncate, but halt further trading

        return False, self._halted, info

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason
