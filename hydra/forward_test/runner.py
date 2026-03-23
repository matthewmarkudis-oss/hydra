"""Forward-test runner — executes agents against sandbox broker.

Loads StaticAgent checkpoints, polls bars during market hours,
translates continuous RL actions to discrete orders, and routes
through AlpacaBroker (sandbox enforced).

Each agent trades against its own virtual sub-account with capital
allocated proportional to Sharpe ratio from ATHENA validation.

Backtesting and training research only. All broker interactions
are through AlpacaBroker which enforces sandbox mode in its constructor.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime
from typing import Any

import numpy as np

from hydra.forward_test.sub_account import SubAccount
from hydra.forward_test.tracker import ForwardTestTracker

logger = logging.getLogger("hydra.forward_test.runner")


class ForwardTestRunner:
    """Runs graduated agents against a sandbox broker for forward testing.

    The runner:
    1. Loads StaticAgent checkpoints
    2. Polls bars at configured intervals during market hours
    3. Feeds observations to each agent from its own sub-account state
    4. Translates continuous actions [-1, 1] into position targets
    5. Applies fills to virtual sub-accounts (optionally routes to broker)
    6. Logs everything to tracker
    7. Tracks slippage between expected and actual fill prices
    8. Alerts on daily loss / drawdown breaches
    """

    def __init__(
        self,
        agents: list,
        broker,
        data_provider,
        tickers: list[str],
        config: dict,
        tracker: ForwardTestTracker,
        allocations: list | None = None,
    ):
        """Initialize the forward test runner.

        Args:
            agents: List of StaticAgent instances to test.
            broker: AlpacaBroker instance (sandbox mode enforced).
            data_provider: MarketDataProvider or similar with get_bars/get_latest_price.
            tickers: List of ticker symbols to trade.
            config: Forward test config dict.
            tracker: ForwardTestTracker for logging.
            allocations: List of Allocation objects from capital_allocator.
                If None, capital is split equally among agents.
        """
        self._agents = agents
        self._broker = broker
        self._data_provider = data_provider
        self._tickers = tickers
        self._config = config
        self._tracker = tracker
        self._running = False
        self._halted = False
        self._day_count = 0
        self._max_days = config.get("duration_days", 60)
        self._max_position_pct = config.get("max_position_pct", 0.20)
        self._poll_interval = config.get("poll_interval_minutes", 5) * 60  # seconds
        self._route_to_broker = config.get("route_to_broker", False)

        # Pending order prices for slippage tracking
        self._pending_orders: dict[str, dict] = {}
        self._last_fill_check: str | None = None

        # Sub-account management — each agent gets its own virtual portfolio
        self._allocations = allocations or []
        self._sub_accounts: dict[str, SubAccount] = {}
        self._init_sub_accounts()

        # Live state builder for full observation vector
        self._live_state_builder = None
        try:
            from hydra.forward_test.live_state_builder import LiveStateBuilder
            self._live_state_builder = LiveStateBuilder(len(tickers), tickers)
            logger.info("LiveStateBuilder initialized — full observation vector enabled")
        except Exception as e:
            logger.warning("LiveStateBuilder unavailable, using simplified obs: %s", e)

        # Alert manager
        self._alert_manager = None
        try:
            from hydra.forward_test.alerts import ForwardTestAlertManager
            self._alert_manager = ForwardTestAlertManager(
                webhook_url=config.get("alert_webhook_url", ""),
                corp_state=None,  # Will be set externally if available
            )
        except Exception as e:
            logger.debug("Alert manager unavailable: %s", e)

    def _init_sub_accounts(self) -> None:
        """Initialize sub-accounts from allocations or equal-split."""
        total_capital = self._config.get("initial_capital", 10000.0)

        if self._allocations:
            for alloc in self._allocations:
                self._sub_accounts[alloc.agent_name] = SubAccount(
                    agent_name=alloc.agent_name,
                    initial_capital=alloc.capital,
                )
                logger.info(
                    "Sub-account %s: $%.2f (%.1f%% of total)",
                    alloc.agent_name, alloc.capital, alloc.weight * 100,
                )
        else:
            # Equal split fallback
            per_agent = total_capital / max(len(self._agents), 1)
            for agent in self._agents:
                self._sub_accounts[agent.name] = SubAccount(
                    agent_name=agent.name,
                    initial_capital=per_agent,
                )
            logger.info(
                "Sub-accounts: equal split $%.2f each across %d agents",
                per_agent, len(self._agents),
            )

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Begin the forward test loop.

        Runs until duration_days is reached or stop() is called.
        This is a blocking call — run in a thread if needed.
        """
        self._running = True

        # Include allocation data in start event
        alloc_data = {}
        if self._allocations:
            alloc_data = {
                a.agent_name: {
                    "sharpe": a.sharpe,
                    "capital": a.capital,
                    "weight": a.weight,
                }
                for a in self._allocations
            }

        self._tracker.record_event("forward_test_start", {
            "agents": [a.name for a in self._agents],
            "tickers": self._tickers,
            "config": self._config,
            "started_at": datetime.now().isoformat(),
            "allocations": alloc_data,
        })

        logger.info(
            "Forward test started: %d agents, %d tickers, %d day limit",
            len(self._agents), len(self._tickers), self._max_days,
        )

        last_date = None

        try:
            while self._running and self._day_count < self._max_days:
                # Check if market is open
                if not self._broker.is_market_open():
                    time.sleep(60)  # Check again in 1 minute
                    continue

                current_date = datetime.now().strftime("%Y-%m-%d")
                if current_date != last_date:
                    if last_date is not None:
                        self._end_of_day(last_date)
                        self._day_count += 1
                    last_date = current_date

                # Check fills from previous bar (slippage tracking)
                self._check_fills()

                # Run one bar for all agents
                timestamp = datetime.now().isoformat()
                self._run_bar(timestamp)

                # Wait for next bar
                time.sleep(self._poll_interval)

        except Exception as e:
            logger.error("Forward test error: %s", e)
            self._emergency_halt(f"Unhandled error: {e}")
            self._tracker.record_event("forward_test_error", {"error": str(e)})
        finally:
            self._running = False
            self._produce_final_report()

    def stop(self) -> dict:
        """Stop the forward test and produce a partial report."""
        self._running = False
        self._tracker.record_event("forward_test_stopped", {
            "days_completed": self._day_count,
            "stopped_at": datetime.now().isoformat(),
        })
        return self._produce_final_report()

    def _emergency_halt(self, reason: str = "unknown") -> None:
        """Cancel all pending orders and halt trading.

        Called on any unhandled error to prevent orphaned orders
        from executing without monitoring.
        """
        if self._halted:
            return  # Already halted, don't spam broker API
        self._halted = True
        logger.critical("EMERGENCY HALT: %s", reason)
        try:
            self._broker.cancel_all_orders()
            logger.warning("All pending orders cancelled via emergency halt")
        except Exception as e:
            logger.error("Failed to cancel orders during emergency halt: %s", e)
        self._tracker.record_event("emergency_halt", {
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        })
        self._running = False

    def _run_bar(self, timestamp: str) -> None:
        """Execute one bar cycle — each agent trades its own sub-account."""
        if self._halted:
            return

        # 1. Fetch latest prices for all tickers (shared market data)
        prices = {}
        for ticker in self._tickers:
            price = self._broker.get_latest_price(ticker)
            if price is not None:
                prices[ticker] = price

        if not prices:
            logger.warning("No prices available, skipping bar")
            return

        # 2. Update live state builder with bar data
        self._update_live_bars(prices)

        # 3. Run each agent against its own sub-account
        for agent in self._agents:
            sub = self._sub_accounts.get(agent.name)
            if sub is None:
                continue
            try:
                self._run_agent_bar_independent(agent, sub, prices, timestamp)
            except Exception as e:
                logger.error("Agent %s bar error: %s", agent.name, e)
                self._emergency_halt(f"Agent {agent.name} error: {e}")
                break

    def _run_agent_bar_independent(
        self,
        agent,
        sub: SubAccount,
        prices: dict[str, float],
        timestamp: str,
    ) -> None:
        """Run one bar for a single agent against its own sub-account."""
        portfolio_value = sub.portfolio_value(prices)
        cash = sub.cash
        current_positions = sub.get_holdings_dict()

        if portfolio_value <= 0:
            logger.warning("Agent %s portfolio value <= 0, skipping", agent.name)
            return

        # Build observation from this agent's own portfolio state
        obs = self._build_full_observation(
            prices, current_positions, portfolio_value, cash,
            peak_value=sub.peak_value,
            initial_cash=sub.initial_capital,
        )

        # Get action from agent
        action = agent.select_action(obs, deterministic=True)

        # Translate to orders and apply to sub-account
        orders_placed = []
        action_dict = {}

        for i, ticker in enumerate(self._tickers):
            if i >= len(action):
                break

            action_val = float(action[i])
            action_dict[ticker] = action_val
            price = prices.get(ticker)
            if price is None or price <= 0:
                continue

            # Target shares based on this agent's own portfolio value
            target_value = action_val * self._max_position_pct * portfolio_value
            target_shares = math.floor(target_value / price)

            current_shares = current_positions.get(ticker, 0)
            delta = target_shares - current_shares

            if abs(delta) < 1:
                continue

            side = "BUY" if delta > 0 else "SELL"
            qty = abs(int(delta))

            # Apply to sub-account (virtual accounting)
            fill_result = sub.apply_fill(
                ticker=ticker,
                qty=qty,
                side=side,
                fill_price=price,
                commission=0.0,
            )

            if "error" not in fill_result:
                order_record = {
                    "ticker": ticker,
                    "side": side,
                    "qty": qty,
                    "expected_price": price,
                    "fill_price": price,
                    "target_shares": target_shares,
                    "current_shares": current_shares,
                    "realized_pnl": fill_result.get("realized_pnl", 0),
                }
                orders_placed.append(order_record)

                # Optionally also route through real broker for paper trading
                if self._route_to_broker:
                    order = self._broker.place_order(ticker, qty, side)
                    if order:
                        order_id = order.get("id")
                        if order_id:
                            self._pending_orders[str(order_id)] = {
                                "ticker": ticker,
                                "expected_price": price,
                                "side": side,
                                "qty": qty,
                                "agent": agent.name,
                                "timestamp": timestamp,
                            }

        # Update peak value for drawdown tracking
        sub.update_peak(prices)

        # Record to tracker with sub-account specific values
        self._tracker.record_bar(
            timestamp=timestamp,
            agent_name=agent.name,
            actions=action_dict,
            positions=current_positions,
            portfolio_value=sub.portfolio_value(prices),
            cash=sub.cash,
            orders_placed=orders_placed,
        )

    def _check_fills(self) -> None:
        """Query broker for recent fills and compute slippage."""
        if not self._pending_orders:
            return

        try:
            fills = self._broker.get_recent_fills(since=self._last_fill_check)
        except Exception:
            return

        if not fills:
            return

        self._last_fill_check = datetime.now().isoformat()

        for fill in fills:
            order_id = fill.get("order_id", "")
            pending = self._pending_orders.pop(order_id, None)
            if pending is None:
                continue

            expected_price = pending["expected_price"]
            fill_price = fill.get("price", expected_price)

            slippage_bps = 0.0
            if expected_price > 0:
                slippage_bps = abs(fill_price - expected_price) / expected_price * 10_000

            self._tracker.record_fill(
                timestamp=fill.get("timestamp", datetime.now().isoformat()),
                agent_name=pending["agent"],
                ticker=pending["ticker"],
                side=pending["side"],
                qty=pending["qty"],
                expected_price=expected_price,
                fill_price=fill_price,
                slippage_bps=round(slippage_bps, 2),
            )

    def _update_live_bars(self, prices: dict[str, float]) -> None:
        """Fetch latest OHLCV bar data and feed to live state builder."""
        if self._live_state_builder is None:
            return

        for ticker in self._tickers:
            try:
                # Try to get recent bars from data provider
                bar_data = None
                if hasattr(self._data_provider, "get_latest_bar"):
                    bar_data = self._data_provider.get_latest_bar(ticker)
                elif hasattr(self._broker, "get_bars"):
                    bars_df = self._broker.get_bars(ticker, timeframe="5Min", limit=1)
                    if bars_df is not None and len(bars_df) > 0:
                        row = bars_df.iloc[-1]
                        bar_data = {
                            "open": float(row.get("open", 0)),
                            "high": float(row.get("high", 0)),
                            "low": float(row.get("low", 0)),
                            "close": float(row.get("close", 0)),
                            "volume": float(row.get("volume", 0)),
                        }

                if bar_data is None:
                    # Fallback: synthetic bar from latest price
                    price = prices.get(ticker, 0)
                    if price > 0:
                        bar_data = {
                            "open": price,
                            "high": price,
                            "low": price,
                            "close": price,
                            "volume": 0.0,
                        }

                if bar_data:
                    self._live_state_builder.update_bar(ticker, bar_data)
            except Exception as e:
                logger.debug("Failed to fetch bar for %s: %s", ticker, e)

    def _build_full_observation(
        self,
        prices: dict[str, float],
        positions: dict[str, float],
        portfolio_value: float,
        cash: float,
        peak_value: float | None = None,
        initial_cash: float | None = None,
    ) -> np.ndarray:
        """Build observation vector using LiveStateBuilder if available.

        Falls back to simplified observation if live state builder
        is not initialized or doesn't have enough data yet.
        """
        if peak_value is None:
            peak_value = portfolio_value
        if initial_cash is None:
            initial_cash = self._config.get("initial_capital", 10000.0)

        if self._live_state_builder is not None and self._live_state_builder.is_ready:
            try:
                return self._live_state_builder.build(
                    cash=cash,
                    initial_cash=initial_cash,
                    holdings=positions,
                    portfolio_value=portfolio_value,
                    peak_value=peak_value,
                )
            except Exception as e:
                logger.debug("LiveStateBuilder.build() failed, using fallback: %s", e)

        return self._build_observation(prices, positions, portfolio_value, cash)

    def _build_observation(
        self,
        prices: dict[str, float],
        positions: dict[str, float],
        portfolio_value: float,
        cash: float,
    ) -> np.ndarray:
        """Build a simplified observation vector (fallback).

        Used when LiveStateBuilder is unavailable or warming up.
        Produces correct shape (17*N+5) but only populates price and
        weight features — indicators are zeros.
        """
        n = len(self._tickers)
        obs = np.zeros(17 * n + 5, dtype=np.float32)

        for i, ticker in enumerate(self._tickers):
            price = prices.get(ticker, 0)
            pos_qty = positions.get(ticker, 0)
            pos_value = pos_qty * price if price > 0 else 0
            weight = pos_value / portfolio_value if portfolio_value > 0 else 0

            base = i * 17
            obs[base] = price / 1000.0
            obs[base + 1] = weight

        # Global features at the end
        obs[17 * n] = cash / portfolio_value if portfolio_value > 0 else 1.0
        obs[17 * n + 1] = 1.0 - (cash / portfolio_value if portfolio_value > 0 else 1.0)

        return obs

    def _end_of_day(self, date: str) -> None:
        """Record end-of-day snapshot per agent and combined portfolio."""
        prices = {}
        for ticker in self._tickers:
            price = self._broker.get_latest_price(ticker)
            if price is not None:
                prices[ticker] = price

        combined_value = 0.0

        for agent in self._agents:
            sub = self._sub_accounts.get(agent.name)
            metrics = self._tracker.get_metrics(agent.name)

            # Merge sub-account snapshot into metrics
            if sub:
                snapshot = sub.get_snapshot(prices)
                metrics.update(snapshot)
                combined_value += sub.portfolio_value(prices)

            self._tracker.record_daily_snapshot(date, agent.name, metrics)

            # Check alert thresholds
            if self._alert_manager:
                try:
                    thresholds = {
                        "daily_loss_pct": self._config.get("alert_daily_loss_pct", 0.03),
                        "max_drawdown_pct": 0.10,
                    }
                    self._alert_manager.check_and_alert(agent.name, metrics, thresholds)
                except Exception as e:
                    logger.debug("Alert check failed: %s", e)

        # Record combined portfolio snapshot
        self._tracker.record_event("combined_daily_snapshot", {
            "date": date,
            "combined_portfolio_value": round(combined_value, 2),
            "sub_accounts": {
                a.name: self._sub_accounts[a.name].get_snapshot(prices)
                for a in self._agents
                if a.name in self._sub_accounts
            },
        })

    def _produce_final_report(self) -> dict:
        """Produce the final graduation report."""
        state = self._tracker.load_state()
        backtest_expectations = state.get("backtest_expectations", {})

        report = self._tracker.get_graduation_report(
            agents=[a.name for a in self._agents],
            backtest_expectations=backtest_expectations,
            config=self._config,
        )

        # Add slippage stats to report
        slippage_stats = self._tracker.get_slippage_stats()
        if slippage_stats:
            report["slippage_stats"] = slippage_stats

        # Add sub-account summaries
        prices = {}
        for ticker in self._tickers:
            price = self._broker.get_latest_price(ticker)
            if price is not None:
                prices[ticker] = price

        report["sub_accounts"] = {
            name: sub.get_snapshot(prices)
            for name, sub in self._sub_accounts.items()
        }

        # Combined portfolio value
        combined = sum(sub.portfolio_value(prices) for sub in self._sub_accounts.values())
        initial_total = sum(sub.initial_capital for sub in self._sub_accounts.values())
        report["combined_portfolio"] = {
            "value": round(combined, 2),
            "initial_capital": round(initial_total, 2),
            "total_return": round((combined - initial_total) / max(initial_total, 1e-8), 6),
        }

        self._tracker.record_event("forward_test_complete", {
            "days_completed": self._day_count,
            "report_summary": report.get("summary", ""),
            "slippage_stats": slippage_stats,
            "combined_portfolio": report["combined_portfolio"],
        })

        # Save final state
        state["status"] = "completed"
        state["final_report"] = report
        state["completed_at"] = datetime.now().isoformat()
        self._tracker.save_state(state)

        logger.info("Forward test complete: %s", report.get("summary", ""))
        return report
