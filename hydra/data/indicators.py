"""Vectorized technical indicator library.

All functions operate on float32 numpy arrays and return float32 arrays.
NaN values are used for warmup periods where the indicator is undefined.
"""

from __future__ import annotations

import numpy as np


def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index (0-100). Vectorized Wilder smoothing."""
    delta = np.empty_like(close)
    delta[0] = 0.0
    delta[1:] = close[1:] - close[:-1]

    gain = np.where(delta > 0, delta, 0.0).astype(np.float32)
    loss = np.where(delta < 0, -delta, 0.0).astype(np.float32)

    out = np.full_like(close, np.nan)

    # Seed with simple average
    if len(close) < period + 1:
        return out

    avg_gain = np.mean(gain[1:period + 1])
    avg_loss = np.mean(loss[1:period + 1])

    alpha = np.float32(1.0 / period)

    for i in range(period, len(close)):
        avg_gain = alpha * gain[i] + (1 - alpha) * avg_gain
        avg_loss = alpha * loss[i] + (1 - alpha) * avg_loss
        if avg_loss < 1e-10:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = np.float32(100.0 - 100.0 / (1.0 + rs))

    return out


def macd(
    close: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD line, signal line, histogram. Returns tuple of 3 float32 arrays."""
    fast_ema = _ema(close, fast_period)
    slow_ema = _ema(close, slow_period)

    macd_line = fast_ema - slow_ema
    signal_line = _ema(macd_line, signal_period)
    histogram = macd_line - signal_line

    # NaN out warmup
    macd_line[:slow_period - 1] = np.nan
    signal_line[:slow_period + signal_period - 2] = np.nan
    histogram[:slow_period + signal_period - 2] = np.nan

    return macd_line, signal_line, histogram


def cci(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    """Commodity Channel Index. Vectorized."""
    tp = (high + low + close) / np.float32(3.0)

    out = np.full_like(close, np.nan)

    for i in range(period - 1, len(tp)):
        window = tp[i - period + 1:i + 1]
        sma_val = np.mean(window)
        mad = np.mean(np.abs(window - sma_val))
        if mad < 1e-10:
            out[i] = 0.0
        else:
            out[i] = (tp[i] - sma_val) / (np.float32(0.015) * mad)

    return out


def bollinger_bands(
    close: np.ndarray,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands: upper, middle (SMA), lower. Float32."""
    middle = _sma(close, period)
    std = _rolling_std(close, period)

    upper = middle + np.float32(num_std) * std
    lower = middle - np.float32(num_std) * std

    return upper, middle, lower


def bollinger_pct_b(close: np.ndarray, period: int = 20, num_std: float = 2.0) -> np.ndarray:
    """Bollinger %B: position within bands (0=lower, 1=upper)."""
    upper, middle, lower = bollinger_bands(close, period, num_std)
    width = upper - lower
    pct_b = np.where(width > 1e-10, (close - lower) / width, np.float32(0.5))
    pct_b[:period - 1] = np.nan
    return pct_b.astype(np.float32)


def volume_ratio(volume: np.ndarray, period: int = 20) -> np.ndarray:
    """Current volume / SMA(volume). Values > 1 indicate above-average volume."""
    avg_vol = _sma(volume, period)
    ratio = np.where(avg_vol > 1e-10, volume / avg_vol, np.float32(1.0))
    ratio[:period - 1] = np.nan
    return ratio.astype(np.float32)


def atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Average True Range. Wilder smoothing."""
    tr = np.empty_like(close)
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    tr = tr.astype(np.float32)

    out = np.full_like(close, np.nan)
    if len(close) < period:
        return out

    out[period - 1] = np.mean(tr[:period])
    alpha = np.float32(1.0 / period)
    for i in range(period, len(close)):
        out[i] = alpha * tr[i] + (1 - alpha) * out[i - 1]

    return out


def compute_all_indicators(
    ohlcv: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute all standard indicators for one ticker's OHLCV data.

    Args:
        ohlcv: Dict with keys 'open', 'high', 'low', 'close', 'volume' (float32 arrays).

    Returns:
        Dict of indicator name → float32 array.
    """
    c = ohlcv["close"]
    h = ohlcv["high"]
    l = ohlcv["low"]
    v = ohlcv["volume"]

    macd_line, macd_signal, macd_hist = macd(c)

    return {
        "rsi": rsi(c),
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "cci": cci(h, l, c),
        "bb_pct_b": bollinger_pct_b(c),
        "volume_ratio": volume_ratio(v),
        "atr": atr(h, l, c),
    }


# --- Internal helpers ---

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    alpha = np.float32(2.0 / (period + 1))
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average via cumsum."""
    out = np.full_like(arr, np.nan)
    cs = np.cumsum(arr)
    out[period - 1:] = (cs[period - 1:] - np.concatenate(([np.float32(0.0)], cs[:-period]))) / np.float32(period)
    return out


def _rolling_std(arr: np.ndarray, period: int) -> np.ndarray:
    """Rolling standard deviation."""
    out = np.full_like(arr, np.nan)
    cs = np.cumsum(arr)
    cs2 = np.cumsum(arr ** 2)
    n = np.float32(period)
    mean = (cs[period - 1:] - np.concatenate(([np.float32(0.0)], cs[:-period]))) / n
    mean2 = (cs2[period - 1:] - np.concatenate(([np.float32(0.0)], cs2[:-period]))) / n
    var = np.maximum(mean2 - mean ** 2, 0.0)
    out[period - 1:] = np.sqrt(var)
    return out
