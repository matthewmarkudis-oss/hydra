"""Virtual sub-account for per-agent forward test tracking.

Each agent gets its own SubAccount with independent cash, positions,
and portfolio value. The SubAccount does NOT place orders — it tracks
the virtual state. The runner mediates between sub-accounts and the
real broker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("hydra.forward_test.sub_account")


@dataclass
class Position:
    """A position in a single ticker."""

    ticker: str
    qty: float = 0.0
    avg_cost: float = 0.0

    @property
    def cost_basis(self) -> float:
        return self.qty * self.avg_cost

    def market_value(self, price: float) -> float:
        return self.qty * price

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.avg_cost) * self.qty if self.qty != 0 else 0.0


class SubAccount:
    """Virtual portfolio for a single agent.

    Tracks cash, positions, and portfolio value independently.
    All state mutations go through explicit methods so the runner
    can log and verify each change.
    """

    def __init__(self, agent_name: str, initial_capital: float):
        self.agent_name = agent_name
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self._positions: dict[str, Position] = {}
        self._peak_value = initial_capital
        self._realized_pnl = 0.0

    @property
    def positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def get_position_qty(self, ticker: str) -> float:
        pos = self._positions.get(ticker)
        return pos.qty if pos else 0.0

    def portfolio_value(self, prices: dict[str, float]) -> float:
        """Compute total portfolio value = cash + sum(position market values)."""
        total = self.cash
        for ticker, pos in self._positions.items():
            price = prices.get(ticker, 0.0)
            total += pos.market_value(price)
        return total

    def update_peak(self, prices: dict[str, float]) -> None:
        """Update peak portfolio value for drawdown tracking."""
        current = self.portfolio_value(prices)
        if current > self._peak_value:
            self._peak_value = current

    @property
    def peak_value(self) -> float:
        return self._peak_value

    def current_drawdown(self, prices: dict[str, float]) -> float:
        """Current drawdown from peak as a positive fraction."""
        current = self.portfolio_value(prices)
        if self._peak_value <= 0:
            return 0.0
        return max(0.0, (self._peak_value - current) / self._peak_value)

    def apply_fill(
        self,
        ticker: str,
        qty: int,
        side: str,
        fill_price: float,
        commission: float = 0.0,
    ) -> dict[str, Any]:
        """Apply a fill to update positions and cash.

        Args:
            ticker: Symbol.
            qty: Absolute quantity filled.
            side: 'BUY' or 'SELL'.
            fill_price: Execution price.
            commission: Trading cost.

        Returns:
            Dict with fill details and PnL, or error dict.
        """
        pos = self._positions.get(ticker, Position(ticker=ticker))
        realized_pnl = 0.0

        if side.upper() == "BUY":
            cost = qty * fill_price + commission
            if cost > self.cash:
                logger.warning(
                    "%s: Insufficient cash for BUY %d %s @ %.2f (need %.2f, have %.2f)",
                    self.agent_name, qty, ticker, fill_price, cost, self.cash,
                )
                return {"error": "insufficient_cash"}

            # Update average cost
            old_value = pos.qty * pos.avg_cost
            new_value = old_value + qty * fill_price
            pos.qty += qty
            pos.avg_cost = new_value / pos.qty if pos.qty > 0 else 0.0
            self.cash -= cost

        elif side.upper() == "SELL":
            if pos.qty < qty:
                logger.warning(
                    "%s: Cannot sell %d %s, only hold %.0f",
                    self.agent_name, qty, ticker, pos.qty,
                )
                return {"error": "insufficient_position"}

            realized_pnl = (fill_price - pos.avg_cost) * qty - commission
            self._realized_pnl += realized_pnl
            pos.qty -= qty
            self.cash += qty * fill_price - commission

        self._positions[ticker] = pos

        # Clean up zero positions
        if pos.qty == 0:
            del self._positions[ticker]

        return {
            "agent": self.agent_name,
            "ticker": ticker,
            "side": side,
            "qty": qty,
            "fill_price": fill_price,
            "commission": commission,
            "realized_pnl": round(realized_pnl, 4),
            "cash_after": round(self.cash, 2),
        }

    def get_holdings_dict(self) -> dict[str, float]:
        """Return {ticker: qty} for observation building."""
        return {t: p.qty for t, p in self._positions.items() if p.qty != 0}

    def get_snapshot(self, prices: dict[str, float]) -> dict[str, Any]:
        """Full snapshot for logging/dashboard."""
        pv = self.portfolio_value(prices)
        total_return = (
            (pv - self.initial_capital) / self.initial_capital
            if self.initial_capital > 0
            else 0.0
        )
        return {
            "agent": self.agent_name,
            "cash": round(self.cash, 2),
            "portfolio_value": round(pv, 2),
            "initial_capital": self.initial_capital,
            "total_return": round(total_return, 6),
            "realized_pnl": round(self._realized_pnl, 2),
            "peak_value": round(self._peak_value, 2),
            "current_drawdown": round(self.current_drawdown(prices), 4),
            "positions": {
                t: {
                    "qty": p.qty,
                    "avg_cost": round(p.avg_cost, 4),
                    "market_value": round(p.market_value(prices.get(t, 0)), 2),
                    "unrealized_pnl": round(p.unrealized_pnl(prices.get(t, 0)), 2),
                }
                for t, p in self._positions.items()
            },
            "num_positions": len(self._positions),
        }
