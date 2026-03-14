"""Reward functions for the trading environment.

Differential Sharpe ratio as the primary reward signal,
with penalty terms for drawdown and transaction costs.
"""

from __future__ import annotations

import numpy as np


class DifferentialSharpeReward:
    """Differential Sharpe Ratio reward function.

    Computes an incremental approximation to the Sharpe ratio at each step,
    using exponential moving averages of returns and squared returns.
    This provides a dense reward signal that incentivizes risk-adjusted returns.

    Reference: Moody & Saffell (2001) "Learning to Trade via Direct Reinforcement"
    """

    def __init__(
        self,
        eta: float = 0.05,
        drawdown_penalty: float = 2.0,
        transaction_penalty: float = 0.1,
        holding_penalty: float = 0.0,
        reward_scale: float = 100.0,
    ):
        self.eta = np.float32(eta)
        self.drawdown_penalty = np.float32(drawdown_penalty)
        self.transaction_penalty = np.float32(transaction_penalty)
        self.holding_penalty = np.float32(holding_penalty)
        self.reward_scale = np.float32(reward_scale)

        # EMA state for differential Sharpe
        self._ema_return = np.float32(0.0)
        self._ema_return_sq = np.float32(0.0)
        self._prev_portfolio_value = np.float32(0.0)
        self._peak_value = np.float32(0.0)

    def reset(self, initial_value: float) -> None:
        """Reset reward state at episode start."""
        self._ema_return = np.float32(0.0)
        self._ema_return_sq = np.float32(0.0)
        self._prev_portfolio_value = np.float32(initial_value)
        self._peak_value = np.float32(initial_value)

    def compute(
        self,
        portfolio_value: float,
        transaction_cost: float,
        holdings: np.ndarray,
        prices: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        """Compute reward for the current step.

        Args:
            portfolio_value: Current total portfolio value.
            transaction_cost: Transaction costs incurred this step.
            holdings: Current holdings array.
            prices: Current prices array.

        Returns:
            Tuple of (total_reward, info_dict with component breakdown).
        """
        pv = np.float32(portfolio_value)

        # Step return
        if self._prev_portfolio_value > 1e-8:
            step_return = (pv - self._prev_portfolio_value) / self._prev_portfolio_value
        else:
            step_return = np.float32(0.0)

        # Update EMAs
        self._ema_return = self.eta * step_return + (1 - self.eta) * self._ema_return
        self._ema_return_sq = self.eta * (step_return ** 2) + (1 - self.eta) * self._ema_return_sq

        # Differential Sharpe
        variance = self._ema_return_sq - self._ema_return ** 2
        if variance > 1e-10:
            denom = np.float32(np.sqrt(float(variance)))
            # Numerator: incremental improvement in Sharpe
            numerator = (
                denom * (step_return - self._ema_return)
                - (self._ema_return / (2 * denom)) * (step_return ** 2 - self._ema_return_sq)
            )
            sharpe_reward = float(numerator / (denom ** 2))
        else:
            sharpe_reward = float(step_return)

        # Drawdown penalty
        self._peak_value = max(self._peak_value, pv)
        drawdown = float((pv - self._peak_value) / max(float(self._peak_value), 1e-8))
        dd_penalty = float(self.drawdown_penalty) * min(drawdown, 0.0)  # negative when in drawdown

        # Transaction cost penalty
        tc_penalty = -float(self.transaction_penalty) * float(transaction_cost) / max(float(self._prev_portfolio_value), 1e-8)

        # Holding penalty (penalize large concentrated positions)
        hold_penalty = 0.0
        if self.holding_penalty > 0 and len(holdings) > 0:
            position_values = np.abs(holdings * prices)
            total_value = max(float(pv), 1e-8)
            max_weight = float(np.max(position_values)) / total_value
            if max_weight > 0.15:  # Penalize >15% concentration
                hold_penalty = -float(self.holding_penalty) * (max_weight - 0.15)

            # Idle cash penalty: penalize holding too much cash
            cash_ratio = float(self._prev_portfolio_value - np.sum(np.abs(holdings * prices))) / max(float(self._prev_portfolio_value), 1e-8)
            if cash_ratio > 0.8:
                hold_penalty += -float(self.holding_penalty) * (cash_ratio - 0.8)

        total_reward = sharpe_reward + dd_penalty + tc_penalty + hold_penalty
        total_reward *= float(self.reward_scale)

        self._prev_portfolio_value = pv

        info = {
            "sharpe_reward": sharpe_reward,
            "drawdown_penalty": dd_penalty,
            "transaction_penalty": tc_penalty,
            "holding_penalty": hold_penalty,
            "step_return": float(step_return),
            "drawdown": drawdown,
            "total_reward": total_reward,
        }

        return total_reward, info

    @property
    def peak_value(self) -> float:
        return float(self._peak_value)


def compute_episode_sharpe(returns: np.ndarray, annualization: float = np.sqrt(252 * 78)) -> float:
    """Compute Sharpe ratio for a full episode's returns.

    Args:
        returns: Array of per-step returns.
        annualization: Annualization factor (default for 5-min bars).

    Returns:
        Annualized Sharpe ratio.
    """
    if len(returns) < 2:
        return 0.0
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    if std_ret < 1e-10:
        return 0.0
    return float(mean_ret / std_ret * annualization)


def compute_sortino(returns: np.ndarray, annualization: float = np.sqrt(252 * 78)) -> float:
    """Compute Sortino ratio (penalizes only downside volatility)."""
    if len(returns) < 2:
        return 0.0
    mean_ret = np.mean(returns)
    downside = returns[returns < 0]
    if len(downside) < 1:
        return float(mean_ret * annualization) if mean_ret > 0 else 0.0
    downside_std = np.std(downside) if len(downside) > 1 else float(np.abs(downside[0]))
    if downside_std < 1e-10:
        return 0.0
    return float(mean_ret / downside_std * annualization)
