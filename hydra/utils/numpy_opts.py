"""Numpy optimization patterns for high-performance environment execution.

All patterns target float32 for memory efficiency and GPU compatibility.
Designed for intraday 5-min bar data (~78 bars/episode).
"""

from __future__ import annotations

import numpy as np
from datetime import date


# --- Pattern 1: Pre-extract OHLCV as contiguous float32 arrays ---

def extract_ohlcv_arrays(df) -> dict[str, np.ndarray]:
    """Extract OHLCV columns from DataFrame as contiguous float32 numpy arrays.

    Call once at episode start to avoid repeated DataFrame indexing.
    """
    return {
        "open": np.ascontiguousarray(df["open"].values, dtype=np.float32),
        "high": np.ascontiguousarray(df["high"].values, dtype=np.float32),
        "low": np.ascontiguousarray(df["low"].values, dtype=np.float32),
        "close": np.ascontiguousarray(df["close"].values, dtype=np.float32),
        "volume": np.ascontiguousarray(df["volume"].values, dtype=np.float32),
    }


# --- Pattern 2: Vectorized indicator computation ---

def vectorized_sma(close: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average via cumsum trick. O(n) instead of O(n*period)."""
    out = np.full_like(close, np.nan)
    cumsum = np.cumsum(close)
    out[period - 1:] = (cumsum[period - 1:] - np.concatenate(([0.0], cumsum[:-period]))) / period
    return out


def vectorized_ema(close: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average. Vectorized initialization, scalar loop for EMA."""
    alpha = np.float32(2.0 / (period + 1))
    out = np.empty_like(close)
    out[0] = close[0]
    for i in range(1, len(close)):
        out[i] = alpha * close[i] + (1 - alpha) * out[i - 1]
    return out


# --- Pattern 3: Pre-allocated output arrays ---

def preallocate_episode_arrays(num_bars: int, num_stocks: int) -> dict[str, np.ndarray]:
    """Pre-allocate arrays for a full episode to avoid per-step allocation."""
    return {
        "portfolio_values": np.zeros(num_bars, dtype=np.float32),
        "returns": np.zeros(num_bars, dtype=np.float32),
        "positions": np.zeros((num_bars, num_stocks), dtype=np.float32),
        "actions": np.zeros((num_bars, num_stocks), dtype=np.float32),
        "rewards": np.zeros(num_bars, dtype=np.float32),
        "cash": np.zeros(num_bars, dtype=np.float32),
    }


# --- Pattern 4: Vectorized session classification ---

# US market sessions (Eastern Time, in minutes from midnight)
PREMARKET_START = 4 * 60       # 04:00
MARKET_OPEN = 9 * 60 + 30     # 09:30
MORNING_END = 12 * 60 + 30    # 12:30
AFTERNOON_START = 13 * 60 + 30  # 13:30
MARKET_CLOSE = 16 * 60         # 16:00
POWER_HOUR = 15 * 60           # 15:00


def classify_sessions_vectorized(minutes_from_midnight: np.ndarray) -> np.ndarray:
    """Classify trading session for each bar. Returns int8 labels.

    0=pre-market, 1=open_auction, 2=morning, 3=midday, 4=afternoon, 5=power_hour, 6=close
    """
    labels = np.zeros(len(minutes_from_midnight), dtype=np.int8)
    labels[minutes_from_midnight < MARKET_OPEN] = 0
    labels[(minutes_from_midnight >= MARKET_OPEN) & (minutes_from_midnight < MARKET_OPEN + 30)] = 1
    labels[(minutes_from_midnight >= MARKET_OPEN + 30) & (minutes_from_midnight < MORNING_END)] = 2
    labels[(minutes_from_midnight >= MORNING_END) & (minutes_from_midnight < AFTERNOON_START)] = 3
    labels[(minutes_from_midnight >= AFTERNOON_START) & (minutes_from_midnight < POWER_HOUR)] = 4
    labels[(minutes_from_midnight >= POWER_HOUR) & (minutes_from_midnight < MARKET_CLOSE)] = 5
    labels[minutes_from_midnight >= MARKET_CLOSE] = 6
    return labels


# --- Pattern 5: frozenset holiday lookups ---

US_MARKET_HOLIDAYS = frozenset({
    # 2024
    date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19),
    date(2024, 3, 29), date(2024, 5, 27), date(2024, 6, 19),
    date(2024, 7, 4), date(2024, 9, 2), date(2024, 11, 28),
    date(2024, 12, 25),
    # 2025
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17),
    date(2025, 4, 18), date(2025, 5, 26), date(2025, 6, 19),
    date(2025, 7, 4), date(2025, 9, 1), date(2025, 11, 27),
    date(2025, 12, 25),
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16),
    date(2026, 4, 3), date(2026, 5, 25), date(2026, 6, 19),
    date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26),
    date(2026, 12, 25),
})


def is_market_holiday(d: date) -> bool:
    """O(1) holiday check via frozenset."""
    return d in US_MARKET_HOLIDAYS


# --- Pattern 6: Signal index maps for O(1) lookup ---

def build_signal_index_map(tickers: list[str]) -> dict[str, int]:
    """Map ticker symbols to contiguous integer indices for array indexing."""
    return {ticker: i for i, ticker in enumerate(tickers)}


# --- Pattern 7: Shared OHLC arrays across agents ---

class SharedMarketData:
    """Immutable shared market data for multi-agent environments.

    All agents reference the same underlying numpy arrays — no copies.
    """

    __slots__ = ("_ohlcv", "_indicators", "_timestamps", "_tickers", "_ticker_idx")

    def __init__(
        self,
        ohlcv: dict[str, dict[str, np.ndarray]],
        indicators: dict[str, dict[str, np.ndarray]],
        timestamps: np.ndarray,
        tickers: list[str],
    ):
        self._ohlcv = ohlcv
        self._indicators = indicators
        self._timestamps = timestamps
        self._tickers = tuple(tickers)
        self._ticker_idx = build_signal_index_map(tickers)

    @property
    def tickers(self) -> tuple[str, ...]:
        return self._tickers

    @property
    def num_stocks(self) -> int:
        return len(self._tickers)

    @property
    def num_bars(self) -> int:
        return len(self._timestamps)

    def get_close(self, ticker: str) -> np.ndarray:
        return self._ohlcv[ticker]["close"]

    def get_ohlcv(self, ticker: str) -> dict[str, np.ndarray]:
        return self._ohlcv[ticker]

    def get_indicator(self, ticker: str, name: str) -> np.ndarray:
        return self._indicators[ticker][name]

    def get_all_closes_matrix(self) -> np.ndarray:
        """Return (num_bars, num_stocks) close price matrix."""
        return np.column_stack([self._ohlcv[t]["close"] for t in self._tickers])

    def ticker_index(self, ticker: str) -> int:
        return self._ticker_idx[ticker]


# --- Pattern 8: Vectorized returns computation ---

def compute_returns(prices: np.ndarray) -> np.ndarray:
    """Vectorized simple returns. First element is 0."""
    ret = np.empty_like(prices)
    ret[0] = 0.0
    ret[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
    return ret


def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """Vectorized log returns. First element is 0."""
    ret = np.empty_like(prices)
    ret[0] = 0.0
    ret[1:] = np.log(prices[1:] / prices[:-1])
    return ret


# --- Pattern 9: Rolling statistics without pandas ---

def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean via cumsum. NaN for first window-1 elements."""
    out = np.full_like(arr, np.nan)
    cs = np.cumsum(arr)
    out[window - 1:] = (cs[window - 1:] - np.concatenate(([0.0], cs[:-window]))) / window
    return out


def rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation. NaN for first window-1 elements."""
    out = np.full_like(arr, np.nan)
    cs = np.cumsum(arr)
    cs2 = np.cumsum(arr ** 2)
    n = np.float32(window)
    mean = (cs[window - 1:] - np.concatenate(([0.0], cs[:-window]))) / n
    mean2 = (cs2[window - 1:] - np.concatenate(([0.0], cs2[:-window]))) / n
    var = mean2 - mean ** 2
    var = np.maximum(var, 0.0)  # Numerical stability
    out[window - 1:] = np.sqrt(var)
    return out


# --- Pattern 10: Vectorized position sizing ---

def clip_position_sizes(
    target_positions: np.ndarray,
    current_positions: np.ndarray,
    max_position_value: float,
    prices: np.ndarray,
) -> np.ndarray:
    """Clip target positions to respect max position size constraint."""
    max_shares = max_position_value / np.maximum(prices, 1e-8)
    clipped = np.clip(target_positions, -max_shares, max_shares)
    return clipped


# --- Pattern 11: Fast drawdown computation ---

def compute_drawdown(equity_curve: np.ndarray) -> np.ndarray:
    """Vectorized drawdown from equity curve. Returns negative fractions."""
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / np.maximum(running_max, 1e-8)
    return drawdown


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum drawdown as a positive fraction."""
    dd = compute_drawdown(equity_curve)
    return float(-np.min(dd)) if len(dd) > 0 else 0.0


# --- Pattern 12: Batch normalization for observations ---

def normalize_observation(obs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Normalize observation vector with running statistics."""
    return (obs - mean) / np.maximum(std, 1e-8)


class RunningStats:
    """Welford's online algorithm for running mean/variance. Float32."""

    __slots__ = ("_count", "_mean", "_m2")

    def __init__(self, size: int):
        self._count = 0
        self._mean = np.zeros(size, dtype=np.float32)
        self._m2 = np.zeros(size, dtype=np.float32)

    def update(self, x: np.ndarray) -> None:
        self._count += 1
        delta = x - self._mean
        self._mean += delta / self._count
        delta2 = x - self._mean
        self._m2 += delta * delta2

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    @property
    def std(self) -> np.ndarray:
        if self._count < 2:
            return np.ones_like(self._mean)
        return np.sqrt(self._m2 / (self._count - 1))

    @property
    def count(self) -> int:
        return self._count
