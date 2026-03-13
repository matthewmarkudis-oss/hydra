"""Numpy-optimized market execution simulator.

Handles order filling with slippage, spread modeling, transaction costs,
and position tracking. All operations use float32 arrays.
For backtesting/simulation only.
"""

from __future__ import annotations

import numpy as np


class MarketSimulator:
    """Simulates order execution with realistic market microstructure.

    Tracks portfolio state: cash, holdings, and portfolio value.
    All monetary values in float32.
    """

    def __init__(
        self,
        num_stocks: int,
        initial_cash: float = 100_000.0,
        transaction_cost_bps: float = 5.0,
        slippage_bps: float = 10.0,
        spread_bps: float = 2.0,
    ):
        self.num_stocks = num_stocks
        self.initial_cash = np.float32(initial_cash)
        self.transaction_cost_rate = np.float32(transaction_cost_bps / 10_000.0)
        self.slippage_rate = np.float32(slippage_bps / 10_000.0)
        self.spread_rate = np.float32(spread_bps / 10_000.0)

        # Portfolio state
        self.cash = np.float32(initial_cash)
        self.holdings = np.zeros(num_stocks, dtype=np.float32)
        self.avg_entry_prices = np.zeros(num_stocks, dtype=np.float32)

        # Tracking
        self.total_transaction_costs = np.float32(0.0)
        self.total_slippage_costs = np.float32(0.0)
        self.num_trades = 0

    def reset(self) -> None:
        """Reset simulator to initial state."""
        self.cash = self.initial_cash
        self.holdings[:] = 0.0
        self.avg_entry_prices[:] = 0.0
        self.total_transaction_costs = np.float32(0.0)
        self.total_slippage_costs = np.float32(0.0)
        self.num_trades = 0

    def execute_orders(
        self,
        target_fractions: np.ndarray,
        current_prices: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Execute orders based on target position fractions.

        Args:
            target_fractions: Array of [-1, +1] per stock.
                Positive = buy fraction of available cash.
                Negative = sell fraction of current holdings.
            current_prices: Current close prices per stock (float32).

        Returns:
            Tuple of (actual shares traded per stock, total transaction cost).
        """
        shares_traded = np.zeros(self.num_stocks, dtype=np.float32)
        total_cost = np.float32(0.0)

        # Process sells first (frees up cash)
        sell_mask = target_fractions < 0
        if np.any(sell_mask):
            sell_fracs = np.abs(target_fractions[sell_mask])
            sell_shares = self.holdings[sell_mask] * sell_fracs

            # Only sell what we actually hold
            sell_shares = np.minimum(sell_shares, np.maximum(self.holdings[sell_mask], 0.0))

            if np.any(sell_shares > 0):
                fill_prices = self._get_sell_fill_prices(current_prices[sell_mask])
                proceeds = sell_shares * fill_prices

                # Transaction costs
                costs = proceeds * self.transaction_cost_rate
                net_proceeds = proceeds - costs

                self.cash += np.sum(net_proceeds)
                self.holdings[sell_mask] -= sell_shares
                shares_traded[sell_mask] = -sell_shares
                total_cost += np.sum(costs)
                self.total_transaction_costs += np.sum(costs)
                self.num_trades += int(np.sum(sell_shares > 0))

        # Process buys
        buy_mask = target_fractions > 0
        if np.any(buy_mask) and self.cash > 0:
            buy_fracs = target_fractions[buy_mask]

            # Allocate cash proportionally
            total_frac = np.sum(buy_fracs)
            if total_frac > 1.0:
                buy_fracs = buy_fracs / total_frac

            cash_per_stock = self.cash * buy_fracs
            fill_prices = self._get_buy_fill_prices(current_prices[buy_mask])

            # Shares to buy (fractional allowed for simplicity)
            buy_shares = cash_per_stock / np.maximum(fill_prices, np.float32(1e-8))

            # Transaction costs
            trade_values = buy_shares * fill_prices
            costs = trade_values * self.transaction_cost_rate
            total_spend = np.sum(trade_values + costs)

            # Scale down if we'd overspend
            if total_spend > self.cash:
                scale = self.cash / max(total_spend, 1e-8)
                buy_shares *= scale
                trade_values *= scale
                costs *= scale
                total_spend = self.cash

            self.cash -= total_spend
            self.cash = max(self.cash, np.float32(0.0))

            # Update average entry prices
            old_value = self.holdings[buy_mask] * self.avg_entry_prices[buy_mask]
            new_value = buy_shares * fill_prices
            new_total = self.holdings[buy_mask] + buy_shares
            safe_total = np.maximum(new_total, np.float32(1e-8))
            self.avg_entry_prices[buy_mask] = (old_value + new_value) / safe_total

            self.holdings[buy_mask] += buy_shares
            shares_traded[buy_mask] = buy_shares
            total_cost += np.sum(costs)
            self.total_transaction_costs += np.sum(costs)
            self.num_trades += int(np.sum(buy_shares > 0))

        return shares_traded, float(total_cost)

    def get_portfolio_value(self, current_prices: np.ndarray) -> float:
        """Compute total portfolio value (cash + holdings)."""
        holdings_value = np.sum(self.holdings * current_prices)
        return float(self.cash + holdings_value)

    def get_holdings_value(self, current_prices: np.ndarray) -> np.ndarray:
        """Get per-stock holdings value."""
        return self.holdings * current_prices

    def get_position_weights(self, current_prices: np.ndarray) -> np.ndarray:
        """Get per-stock position weights (fraction of portfolio)."""
        pv = self.get_portfolio_value(current_prices)
        if pv < 1e-8:
            return np.zeros(self.num_stocks, dtype=np.float32)
        return (self.holdings * current_prices) / np.float32(pv)

    def liquidate_all(self, current_prices: np.ndarray) -> float:
        """Sell all holdings at current prices. Returns total proceeds."""
        if np.sum(np.abs(self.holdings)) < 1e-8:
            return 0.0
        sell_fracs = -np.ones(self.num_stocks, dtype=np.float32)
        sell_fracs[self.holdings <= 0] = 0.0
        self.execute_orders(sell_fracs, current_prices)
        return float(self.cash)

    def _get_buy_fill_prices(self, mid_prices: np.ndarray) -> np.ndarray:
        """Apply spread and slippage to buy orders (price goes up)."""
        spread_adj = mid_prices * (1 + self.spread_rate / 2)
        slippage_adj = spread_adj * (1 + self.slippage_rate)
        return slippage_adj

    def _get_sell_fill_prices(self, mid_prices: np.ndarray) -> np.ndarray:
        """Apply spread and slippage to sell orders (price goes down)."""
        spread_adj = mid_prices * (1 - self.spread_rate / 2)
        slippage_adj = spread_adj * (1 - self.slippage_rate)
        return slippage_adj

    def get_state(self) -> dict:
        """Snapshot of current portfolio state."""
        return {
            "cash": float(self.cash),
            "holdings": self.holdings.copy(),
            "avg_entry_prices": self.avg_entry_prices.copy(),
            "total_transaction_costs": float(self.total_transaction_costs),
            "num_trades": self.num_trades,
        }
