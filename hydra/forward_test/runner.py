"""Forward-test runner — executes agents against sandbox broker.

Loads StaticAgent checkpoints, polls bars during market hours,
translates continuous RL actions to discrete orders, and routes
through AlpacaBroker (sandbox enforced).

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

from hydra.forward_test.tracker import ForwardTestTracker

logger = logging.getLogger("hydra.forward_test.runner")


class ForwardTestRunner:
    """Runs graduated agents against a sandbox broker for forward testing.

    The runner:
    1. Loads StaticAgent checkpoints
    2. Polls bars at configured intervals during market hours
    3. Feeds observations to each agent via select_action
    4. Translates continuous actions [-1, 1] into position targets
    5. Computes order deltas and routes through broker
    6. Logs everything to tracker
    """

    def __init__(
        self,
        agents: list,
        broker,
        data_provider,
        tickers: list[str],
        config: dict,
        tracker: ForwardTestTracker,
    ):
        """Initialize the forward test runner.

        Args:
            agents: List of StaticAgent instances to test.
            broker: AlpacaBroker instance (sandbox mode enforced).
            data_provider: MarketDataProvider or similar with get_bars/get_latest_price.
            tickers: List of ticker symbols to trade.
            config: Forward test config dict.
            tracker: ForwardTestTracker for logging.
        """
        self._agents = agents
        self._broker = broker
        self._data_provider = data_provider
        self._tickers = tickers
        self._config = config
        self._tracker = tracker
        self._running = False
        self._day_count = 0
        self._max_days = config.get("duration_days", 20)
        self._max_position_pct = config.get("max_position_pct", 0.20)
        self._poll_interval = config.get("poll_interval_minutes", 5) * 60  # seconds

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Begin the forward test loop.

        Runs until duration_days is reached or stop() is called.
        This is a blocking call — run in a thread if needed.
        """
        self._running = True
        self._tracker.record_event("forward_test_start", {
            "agents": [a.name for a in self._agents],
            "tickers": self._tickers,
            "config": self._config,
            "started_at": datetime.now().isoformat(),
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

                # Run one bar for all agents
                timestamp = datetime.now().isoformat()
                self._run_bar(timestamp)

                # Wait for next bar
                time.sleep(self._poll_interval)

        except Exception as e:
            logger.error("Forward test error: %s", e)
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

    def _run_bar(self, timestamp: str) -> None:
        """Execute one bar cycle for all agents."""
        # 1. Fetch latest prices for all tickers
        prices = {}
        for ticker in self._tickers:
            price = self._broker.get_latest_price(ticker)
            if price is not None:
                prices[ticker] = price

        if not prices:
            logger.warning("No prices available, skipping bar")
            return

        # 2. Get current account state
        account = self._broker.get_account()
        portfolio_value = account.get("portfolio_value", 0)
        cash = account.get("cash", 0)

        if portfolio_value <= 0:
            logger.warning("Portfolio value is zero or negative, skipping bar")
            return

        # 3. Get current positions
        positions_list = self._broker.get_open_positions()
        current_positions = {p["symbol"]: p["qty"] for p in positions_list}

        # 4. Build observation vector (simplified — matches env state builder)
        obs = self._build_observation(prices, current_positions, portfolio_value, cash)

        # 5. Run each agent
        for agent in self._agents:
            try:
                self._run_agent_bar(
                    agent, obs, prices, current_positions,
                    portfolio_value, cash, timestamp,
                )
            except Exception as e:
                logger.error("Agent %s bar error: %s", agent.name, e)

    def _run_agent_bar(
        self,
        agent,
        obs: np.ndarray,
        prices: dict[str, float],
        current_positions: dict[str, float],
        portfolio_value: float,
        cash: float,
        timestamp: str,
    ) -> None:
        """Run one bar for a single agent."""
        # Get action from agent
        action = agent.select_action(obs, deterministic=True)

        # Translate continuous action to target positions
        # action[i] in [-1, 1] = target portfolio weight for ticker i
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

            # Target shares = action * max_position_pct * portfolio_value / price
            target_value = action_val * self._max_position_pct * portfolio_value
            target_shares = math.floor(target_value / price)

            # Current shares
            current_shares = current_positions.get(ticker, 0)
            delta = target_shares - current_shares

            # Only route when delta >= 1 share
            if abs(delta) < 1:
                continue

            side = "BUY" if delta > 0 else "SELL"
            qty = abs(int(delta))

            order = self._broker.place_order(ticker, qty, side)
            if order:
                orders_placed.append({
                    "ticker": ticker,
                    "side": side,
                    "qty": qty,
                    "price": price,
                    "target_shares": target_shares,
                    "current_shares": current_shares,
                })

        # Record to tracker
        self._tracker.record_bar(
            timestamp=timestamp,
            agent_name=agent.name,
            actions=action_dict,
            positions=current_positions,
            portfolio_value=portfolio_value,
            cash=cash,
            orders_placed=orders_placed,
        )

    def _build_observation(
        self,
        prices: dict[str, float],
        positions: dict[str, float],
        portfolio_value: float,
        cash: float,
    ) -> np.ndarray:
        """Build a simplified observation vector.

        This is a minimal obs for forward testing. The full TradingEnv
        observation includes technical indicators, volume, etc. For
        forward testing we provide:
        - Per ticker: normalized price, position weight (2 features)
        - Global: cash ratio, total position ratio (2 features)

        Note: This produces a smaller obs than the training env (17*N+5).
        For agents that expect the full obs, the runner should be extended
        to build the complete feature vector from bar data.
        """
        n = len(self._tickers)
        obs = np.zeros(17 * n + 5, dtype=np.float32)

        for i, ticker in enumerate(self._tickers):
            price = prices.get(ticker, 0)
            pos_qty = positions.get(ticker, 0)
            pos_value = pos_qty * price if price > 0 else 0
            weight = pos_value / portfolio_value if portfolio_value > 0 else 0

            # Fill first two features per ticker (price norm, weight)
            # Rest stay at 0 — agent should handle gracefully
            base = i * 17
            obs[base] = price / 1000.0  # crude normalization
            obs[base + 1] = weight

        # Global features at the end
        obs[17 * n] = cash / portfolio_value if portfolio_value > 0 else 1.0
        obs[17 * n + 1] = 1.0 - (cash / portfolio_value if portfolio_value > 0 else 1.0)

        return obs

    def _end_of_day(self, date: str) -> None:
        """Record end-of-day snapshot for each agent."""
        for agent in self._agents:
            metrics = self._tracker.get_metrics(agent.name)
            self._tracker.record_daily_snapshot(date, agent.name, metrics)

    def _produce_final_report(self) -> dict:
        """Produce the final graduation report."""
        state = self._tracker.load_state()
        backtest_expectations = state.get("backtest_expectations", {})

        report = self._tracker.get_graduation_report(
            agents=[a.name for a in self._agents],
            backtest_expectations=backtest_expectations,
            config=self._config,
        )

        self._tracker.record_event("forward_test_complete", {
            "days_completed": self._day_count,
            "report_summary": report.get("summary", ""),
        })

        # Save final state
        state["status"] = "completed"
        state["final_report"] = report
        state["completed_at"] = datetime.now().isoformat()
        self._tracker.save_state(state)

        logger.info("Forward test complete: %s", report.get("summary", ""))
        return report
