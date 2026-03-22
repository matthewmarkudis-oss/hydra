"""Live observation vector builder for forward testing.

Mirrors the training StateBuilder's 17N+5 observation layout using a
rolling window of live OHLCV bars. Reuses hydra.data.indicators for
all technical indicator computation.

This ensures agents receive the same observation distribution during
forward testing as they did during training.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from hydra.data.indicators import compute_all_indicators

logger = logging.getLogger("hydra.forward_test.live_state_builder")

# Minimum bars needed before indicators are reliable
_MIN_WARMUP_BARS = 55  # SMA50 + a few extra


class LiveStateBuilder:
    """Builds full 17N+5 observation vectors from live bar data.

    Maintains a rolling window of OHLCV bars per ticker and computes
    all 16 indicator groups on each build() call. This is slower than
    the training StateBuilder (which pre-computes per-episode), but
    produces the identical feature vector.

    Observation layout matches StateBuilder exactly:
        [0]              cash_ratio
        [1:N+1]          holdings (normalized)
        [N+1:2N+1]       prices (close / start_close)
        [2N+1:3N+1]      RSI / 100
        ...              (see StateBuilder docstring for full layout)
        [17N+1:17N+5]    global features (drawdown, pnl, session, progress)
    """

    def __init__(self, num_stocks: int, tickers: list[str], buffer_size: int = 60):
        """Initialize the live state builder.

        Args:
            num_stocks: Number of tickers being traded.
            tickers: Ordered list of ticker symbols.
            buffer_size: Rolling window size (bars). 60 covers all
                indicator warmup periods (max is SMA50 = 50 bars).
        """
        self.num_stocks = num_stocks
        self.obs_dim = 17 * num_stocks + 5
        self._tickers = tickers
        self._buffer_size = buffer_size
        self._bar_count = 0

        # Per-ticker ring buffers: {ticker: deque of {open, high, low, close, volume}}
        self._buffers: dict[str, deque[dict[str, float]]] = {
            t: deque(maxlen=buffer_size) for t in tickers
        }

        # Start prices (first close seen per ticker, for normalization)
        self._start_prices: dict[str, float] = {}

    @property
    def is_ready(self) -> bool:
        """True when enough bars have accumulated for indicator warmup."""
        return all(
            len(self._buffers[t]) >= _MIN_WARMUP_BARS
            for t in self._tickers
        )

    @property
    def bars_collected(self) -> int:
        """Minimum number of bars across all tickers."""
        if not self._buffers:
            return 0
        return min(len(buf) for buf in self._buffers.values())

    def update_bar(self, ticker: str, ohlcv: dict[str, float]) -> None:
        """Append one OHLCV bar for a ticker.

        Args:
            ohlcv: Dict with keys: open, high, low, close, volume.
        """
        if ticker not in self._buffers:
            return

        self._buffers[ticker].append({
            "open": float(ohlcv.get("open", 0)),
            "high": float(ohlcv.get("high", 0)),
            "low": float(ohlcv.get("low", 0)),
            "close": float(ohlcv.get("close", 0)),
            "volume": float(ohlcv.get("volume", 0)),
        })

        # Record first close as start price
        if ticker not in self._start_prices:
            close = ohlcv.get("close", 0)
            if close > 0:
                self._start_prices[ticker] = close

        self._bar_count += 1

    def build(
        self,
        cash: float,
        initial_cash: float,
        holdings: dict[str, float],
        portfolio_value: float,
        peak_value: float,
    ) -> np.ndarray:
        """Build the full 17N+5 observation vector.

        Uses the rolling bar buffers to compute all indicators, then
        applies the same normalization as StateBuilder.

        Args:
            cash: Current cash balance.
            initial_cash: Starting cash.
            holdings: {ticker: share_count} dict.
            portfolio_value: Current total portfolio value.
            peak_value: Peak portfolio value (for drawdown).

        Returns:
            Float32 array of shape (obs_dim,).
        """
        n = self.num_stocks
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        # [0] Cash ratio
        obs[0] = np.float32(cash / max(initial_cash, 1e-8))

        # [1:N+1] Holdings (normalized)
        for i, ticker in enumerate(self._tickers):
            obs[1 + i] = np.float32(holdings.get(ticker, 0)) / np.float32(100.0)

        # [N+1:17N+1] Per-ticker indicator features
        for i, ticker in enumerate(self._tickers):
            buf = self._buffers.get(ticker)
            if not buf or len(buf) < 3:
                continue

            indicators = self._compute_ticker_indicators(ticker)
            if indicators is None:
                continue

            start_price = self._start_prices.get(ticker, 1.0)
            safe_start = max(start_price, 1e-8)

            base = n + 1 + i  # Starting index for this ticker's features
            stride = n  # Features are interleaved: all tickers' price, then all RSI, etc.

            # Feature 0: price (close / start_close)
            obs[base + 0 * stride] = indicators["close"] / safe_start

            # Feature 1: RSI / 100
            obs[base + 1 * stride] = indicators["rsi"] / 100.0

            # Feature 2: MACD histogram (z-scored, clipped)
            obs[base + 2 * stride] = np.clip(indicators["macd_hist"] / 5.0, -1.0, 1.0)

            # Feature 3: CCI / 200
            obs[base + 3 * stride] = np.clip(indicators["cci"] / 200.0, -1.0, 1.0)

            # Feature 4: Bollinger %B
            obs[base + 4 * stride] = np.clip(indicators["bb_pct_b"], 0.0, 1.0)

            # Feature 5: Volume ratio / 5
            obs[base + 5 * stride] = np.clip(indicators["volume_ratio"], 0.0, 5.0) / 5.0

            # Feature 6: Trend direction
            obs[base + 6 * stride] = indicators["trend_direction"]

            # Feature 7: Bar body ratio
            obs[base + 7 * stride] = indicators["bar_body_ratio"]

            # Feature 8: Close range position
            obs[base + 8 * stride] = indicators["close_range_position"]

            # Feature 9: Bar momentum
            obs[base + 9 * stride] = indicators["bar_momentum"]

            # Feature 10: Upper wick ratio
            obs[base + 10 * stride] = indicators["upper_wick_ratio"]

            # Feature 11: Vol regime / 3
            obs[base + 11 * stride] = np.clip(indicators["vol_regime"], 0.0, 3.0) / 3.0

            # Feature 12: Trend strength
            obs[base + 12 * stride] = np.clip(indicators["trend_strength"], 0.0, 1.0)

            # Feature 13: Mean reversion z / 3
            obs[base + 13 * stride] = np.clip(indicators["mean_reversion_z"], -3.0, 3.0) / 3.0

            # Feature 14: News sentiment (zero — not available live)
            obs[base + 14 * stride] = 0.0

            # Feature 15: Sentiment momentum (zero — not available live)
            obs[base + 15 * stride] = 0.0

        # Global features
        idx = 17 * n + 1
        obs[idx] = np.float32((portfolio_value - peak_value) / max(peak_value, 1e-8))
        obs[idx + 1] = np.float32((portfolio_value - initial_cash) / max(initial_cash, 1e-8))
        obs[idx + 2] = np.float32(0.5)  # Session label — not applicable live, use neutral
        obs[idx + 3] = np.float32(0.5)  # Time progress — not applicable live, use midpoint

        return obs

    def _compute_ticker_indicators(self, ticker: str) -> dict[str, float] | None:
        """Compute all indicators for a ticker from its ring buffer.

        Returns the latest (most recent) value of each indicator.
        """
        buf = self._buffers.get(ticker)
        if not buf or len(buf) < 3:
            return None

        # Convert buffer to arrays
        bars = list(buf)
        n = len(bars)
        ohlcv = {
            "open": np.array([b["open"] for b in bars], dtype=np.float32),
            "high": np.array([b["high"] for b in bars], dtype=np.float32),
            "low": np.array([b["low"] for b in bars], dtype=np.float32),
            "close": np.array([b["close"] for b in bars], dtype=np.float32),
            "volume": np.array([b["volume"] for b in bars], dtype=np.float32),
        }

        try:
            all_ind = compute_all_indicators(ohlcv)
        except Exception as e:
            logger.debug("Indicator computation failed for %s: %s", ticker, e)
            return None

        # Extract latest (last) value of each indicator, replacing NaN with neutral
        def _last(arr, default=0.0):
            if arr is None or len(arr) == 0:
                return default
            val = float(arr[-1])
            return default if np.isnan(val) else val

        return {
            "close": float(ohlcv["close"][-1]),
            "rsi": _last(all_ind.get("rsi"), 50.0),
            "macd_hist": _last(all_ind.get("macd_hist"), 0.0),
            "cci": _last(all_ind.get("cci"), 0.0),
            "bb_pct_b": _last(all_ind.get("bb_pct_b"), 0.5),
            "volume_ratio": _last(all_ind.get("volume_ratio"), 1.0),
            "trend_direction": _last(all_ind.get("trend_direction"), 0.0),
            "bar_body_ratio": _last(all_ind.get("bar_body_ratio"), 0.0),
            "close_range_position": _last(all_ind.get("close_range_position"), 0.5),
            "bar_momentum": _last(all_ind.get("bar_momentum"), 0.0),
            "upper_wick_ratio": _last(all_ind.get("upper_wick_ratio"), 0.0),
            "vol_regime": _last(all_ind.get("vol_regime"), 1.0),
            "trend_strength": _last(all_ind.get("trend_strength"), 0.0),
            "mean_reversion_z": _last(all_ind.get("mean_reversion_z"), 0.0),
        }
