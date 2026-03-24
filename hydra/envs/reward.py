"""Reward functions for the trading environment.

Differential Sharpe ratio as the primary reward signal,
with penalty terms for drawdown and transaction costs.
"""

from __future__ import annotations

import numpy as np

from hydra.distillation.regime_rewards import get_multipliers


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
        drawdown_penalty: float = 0.15,
        transaction_penalty: float = 0.01,
        holding_penalty: float = 0.02,
        pnl_bonus_weight: float = 5.0,
        reward_scale: float = 100.0,
        cash_drag_penalty: float = 0.3,
        benchmark_bonus_weight: float = 2.0,
        min_deployment_pct: float = 0.3,
    ):
        self.eta = np.float32(eta)
        self.drawdown_penalty = np.float32(drawdown_penalty)
        self.transaction_penalty = np.float32(transaction_penalty)
        self.holding_penalty = np.float32(holding_penalty)
        self.pnl_bonus_weight = np.float32(pnl_bonus_weight)
        self.reward_scale = np.float32(reward_scale)
        self.cash_drag_penalty = np.float32(cash_drag_penalty)
        self.benchmark_bonus_weight = np.float32(benchmark_bonus_weight)
        self.min_deployment_pct = np.float32(min_deployment_pct)

        # Regime multipliers (default: neutral)
        self._regime = "risk_on"

        # Benchmark returns for the current episode (set via set_benchmark)
        self._benchmark_returns: np.ndarray | None = None
        self._step_idx = 0

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
        self._step_idx = 0

    def set_benchmark(self, benchmark_returns: np.ndarray | None) -> None:
        """Set per-bar benchmark returns for the current episode.

        Args:
            benchmark_returns: Array of per-bar returns for the benchmark
                              (e.g. SPY). None to disable benchmark bonus.
        """
        self._benchmark_returns = benchmark_returns

    @property
    def regime(self) -> str:
        """Current market regime used for reward weight adjustment."""
        return self._regime

    def set_regime(self, regime: str) -> None:
        """Set the market regime for reward weight multipliers.

        Args:
            regime: One of "risk_on", "risk_off", "crisis".
                    Unknown values fall back to "risk_on" multipliers.
        """
        self._regime = regime

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

        self._step_idx += 1

        # Load regime multipliers (applied on top of base weights)
        mult = get_multipliers(self._regime)

        # Effective weights = base weight * regime multiplier
        eff_drawdown_penalty = float(self.drawdown_penalty) * mult["drawdown_penalty"]
        eff_transaction_penalty = float(self.transaction_penalty) * mult["transaction_penalty"]
        eff_holding_penalty = float(self.holding_penalty) * mult["holding_penalty"]
        eff_pnl_bonus_weight = float(self.pnl_bonus_weight) * mult["pnl_bonus_weight"]
        eff_reward_scale = float(self.reward_scale) * mult["reward_scale"]
        eff_cash_drag_penalty = float(self.cash_drag_penalty)
        eff_benchmark_bonus_weight = float(self.benchmark_bonus_weight)

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
        dd_penalty = eff_drawdown_penalty * min(drawdown, 0.0)  # negative when in drawdown

        # Transaction cost penalty
        tc_penalty = -eff_transaction_penalty * float(transaction_cost) / max(float(self._prev_portfolio_value), 1e-8)

        # Holding penalty (penalize extreme concentrated positions)
        hold_penalty = 0.0
        if eff_holding_penalty > 0 and len(holdings) > 0:
            position_values = np.abs(holdings * prices)
            total_value = max(float(pv), 1e-8)
            max_weight = float(np.max(position_values)) / total_value
            if max_weight > 0.35:  # Was 0.15 — allow concentrated conviction bets
                hold_penalty = -eff_holding_penalty * (max_weight - 0.35)

        # P&L bonus: direct reward for positive returns, penalty for losses.
        # This complements the Sharpe component (which rewards consistency)
        # by also rewarding raw profitability.
        pnl_bonus = eff_pnl_bonus_weight * float(step_return)

        # Cash drag penalty — continuous penalty for undeployed capital
        position_value = float(np.sum(np.abs(holdings * prices)))
        cash_ratio = max(0.0, 1.0 - position_value / max(float(pv), 1e-8))
        cash_drag = -eff_cash_drag_penalty * cash_ratio

        # Minimum deployment target — escalating penalty below threshold
        deployment_penalty = 0.0
        deployed_pct = position_value / max(float(pv), 1e-8)
        if deployed_pct < float(self.min_deployment_pct):
            shortfall = float(self.min_deployment_pct) - deployed_pct
            deployment_penalty = -eff_cash_drag_penalty * shortfall * 2.0

        # Benchmark bonus — reward outperforming the benchmark
        benchmark_bonus = 0.0
        if self._benchmark_returns is not None and self._step_idx - 1 < len(self._benchmark_returns):
            bench_return = float(self._benchmark_returns[self._step_idx - 1])
            excess_return = float(step_return) - bench_return
            benchmark_bonus = eff_benchmark_bonus_weight * excess_return

        total_reward = (
            sharpe_reward + pnl_bonus + dd_penalty + tc_penalty
            + hold_penalty + cash_drag + deployment_penalty + benchmark_bonus
        )
        total_reward *= eff_reward_scale

        self._prev_portfolio_value = pv

        info = {
            "sharpe_reward": sharpe_reward,
            "pnl_bonus": pnl_bonus,
            "drawdown_penalty": dd_penalty,
            "transaction_penalty": tc_penalty,
            "holding_penalty": hold_penalty,
            "cash_drag": cash_drag,
            "deployment_penalty": deployment_penalty,
            "benchmark_bonus": benchmark_bonus,
            "step_return": float(step_return),
            "drawdown": drawdown,
            "cash_ratio": cash_ratio,
            "deployed_pct": deployed_pct,
            "total_reward": total_reward,
            "regime": self._regime,
        }

        return total_reward, info

    def get_params(self) -> dict[str, float]:
        """Return current tunable reward parameters."""
        return {
            "drawdown_penalty": float(self.drawdown_penalty),
            "transaction_penalty": float(self.transaction_penalty),
            "holding_penalty": float(self.holding_penalty),
            "pnl_bonus_weight": float(self.pnl_bonus_weight),
            "reward_scale": float(self.reward_scale),
            "cash_drag_penalty": float(self.cash_drag_penalty),
            "benchmark_bonus_weight": float(self.benchmark_bonus_weight),
            "min_deployment_pct": float(self.min_deployment_pct),
        }

    def update_params(self, params: dict[str, float]) -> None:
        """Update tunable reward parameters in-place.

        Only updates keys that exist as attributes. Ignores unknown keys.
        """
        for key in ("drawdown_penalty", "transaction_penalty", "holding_penalty",
                     "pnl_bonus_weight", "reward_scale",
                     "cash_drag_penalty", "benchmark_bonus_weight", "min_deployment_pct"):
            if key in params:
                setattr(self, key, np.float32(params[key]))

    @property
    def peak_value(self) -> float:
        return float(self._peak_value)


def compute_episode_sharpe(
    returns: np.ndarray,
    bars_per_day: int = 78,
    annualization: float | None = None,
) -> float:
    """Compute Sharpe ratio for a full episode's returns.

    Args:
        returns: Array of per-step returns.
        bars_per_day: Bars per trading day (78 for 5-min, 1 for daily).
        annualization: Annualization factor. Defaults to sqrt(252 * bars_per_day).

    Returns:
        Annualized Sharpe ratio.
    """
    if annualization is None:
        annualization = float(np.sqrt(252 * bars_per_day))
    if len(returns) < 2:
        return 0.0
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    if std_ret < 1e-10:
        return 0.0
    return float(mean_ret / std_ret * annualization)


def compute_sortino(
    returns: np.ndarray,
    bars_per_day: int = 78,
    annualization: float | None = None,
) -> float:
    """Compute Sortino ratio (penalizes only downside volatility).

    Args:
        returns: Array of per-step returns.
        bars_per_day: Bars per trading day (78 for 5-min, 1 for daily).
        annualization: Annualization factor. Defaults to sqrt(252 * bars_per_day).
    """
    if annualization is None:
        annualization = float(np.sqrt(252 * bars_per_day))
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
