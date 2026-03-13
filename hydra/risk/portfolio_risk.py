"""Portfolio-level risk monitoring.

Tracks aggregate risk metrics across the portfolio for logging and alerts.
"""

from __future__ import annotations

import numpy as np

from hydra.utils.numpy_opts import compute_drawdown, max_drawdown


class PortfolioRiskMonitor:
    """Monitors portfolio-level risk metrics during an episode."""

    def __init__(self, num_stocks: int, initial_cash: float = 100_000.0):
        self.num_stocks = num_stocks
        self.initial_cash = np.float32(initial_cash)
        self._equity_history: list[float] = []
        self._return_history: list[float] = []
        self._prev_value = np.float32(initial_cash)

    def reset(self, initial_cash: float | None = None) -> None:
        """Reset for a new episode."""
        if initial_cash is not None:
            self.initial_cash = np.float32(initial_cash)
        self._equity_history = [float(self.initial_cash)]
        self._return_history = []
        self._prev_value = self.initial_cash

    def update(self, portfolio_value: float) -> dict[str, float]:
        """Record new portfolio value and compute risk metrics.

        Returns dict of current risk metrics.
        """
        pv = np.float32(portfolio_value)
        self._equity_history.append(float(pv))

        step_return = float((pv - self._prev_value) / max(float(self._prev_value), 1e-8))
        self._return_history.append(step_return)
        self._prev_value = pv

        equity = np.array(self._equity_history, dtype=np.float32)

        return {
            "portfolio_value": float(pv),
            "total_return": float((pv - self.initial_cash) / self.initial_cash),
            "step_return": step_return,
            "max_drawdown": max_drawdown(equity),
            "current_drawdown": float(compute_drawdown(equity)[-1]) if len(equity) > 0 else 0.0,
            "volatility": float(np.std(self._return_history)) if len(self._return_history) > 1 else 0.0,
            "num_steps": len(self._return_history),
        }

    def get_summary(self) -> dict[str, float]:
        """Get end-of-episode summary metrics."""
        if not self._equity_history:
            return {}

        equity = np.array(self._equity_history, dtype=np.float32)
        returns = np.array(self._return_history, dtype=np.float32)

        mean_ret = float(np.mean(returns)) if len(returns) > 0 else 0.0
        std_ret = float(np.std(returns)) if len(returns) > 1 else 1e-8

        sharpe = mean_ret / max(std_ret, 1e-8) * np.sqrt(252 * 78)

        downside = returns[returns < 0]
        downside_std = float(np.std(downside)) if len(downside) > 1 else 1e-8
        sortino = mean_ret / max(downside_std, 1e-8) * np.sqrt(252 * 78)

        return {
            "total_return": float((equity[-1] - equity[0]) / equity[0]),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": max_drawdown(equity),
            "volatility": std_ret,
            "num_steps": len(returns),
            "final_value": float(equity[-1]),
        }
