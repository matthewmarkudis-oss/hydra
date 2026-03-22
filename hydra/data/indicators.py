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


def trend_direction(close: np.ndarray, fast_period: int = 20, slow_period: int = 50) -> np.ndarray:
    """Trend direction via SMA crossover.

    Returns +1 (bullish: fast SMA > slow SMA), -1 (bearish), or 0 (neutral/warmup).
    """
    fast_sma = _sma(close, fast_period)
    slow_sma = _sma(close, slow_period)

    out = np.zeros_like(close)
    valid = ~(np.isnan(fast_sma) | np.isnan(slow_sma))
    diff = fast_sma - slow_sma
    out[valid & (diff > 0)] = np.float32(1.0)
    out[valid & (diff < 0)] = np.float32(-1.0)
    return out.astype(np.float32)


def bar_body_ratio(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """Bar body ratio: (close - open) / bar range.

    Measures direction and conviction of each bar.
    Positive = bullish, negative = bearish, magnitude = strength.
    Flat bars (high == low) → 0.0.

    Returns:
        Float32 array in [-1, 1]. No warmup, no NaN.
    """
    bar_range = high - low
    eps = np.float32(1e-10)
    ratio = (close - open_) / np.maximum(bar_range, eps)
    return np.clip(ratio, -1.0, 1.0).astype(np.float32)


def close_range_position(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """Close range position: where close settled within the bar's range.

    Near 1.0 = closed near high (bullish), near 0.0 = closed near low (bearish).
    Flat bars (high == low) → 0.5 (neutral).

    Returns:
        Float32 array in [0, 1]. No warmup, no NaN.
    """
    bar_range = high - low
    eps = np.float32(1e-10)
    return np.where(
        bar_range > eps,
        (close - low) / np.maximum(bar_range, eps),
        np.float32(0.5),
    ).astype(np.float32)


def bar_momentum(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr_period: int = 14,
    atr_values: np.ndarray | None = None,
) -> np.ndarray:
    """ATR-normalized bar-over-bar momentum.

    Measures directional move relative to recent volatility.
    First `atr_period` bars are NaN (from ATR warmup).

    Args:
        close: Close prices.
        high: High prices.
        low: Low prices.
        atr_period: ATR lookback period.
        atr_values: Optional pre-computed ATR to avoid recomputation.

    Returns:
        Float32 array in [-1, 1] (clipped). First atr_period bars NaN.
    """
    atr_vals = atr_values if atr_values is not None else atr(high, low, close, atr_period)
    eps = np.float32(1e-10)

    delta = np.empty_like(close)
    delta[0] = np.nan
    delta[1:] = close[1:] - close[:-1]

    momentum = delta / np.maximum(atr_vals, eps)
    momentum = np.clip(momentum, -1.0, 1.0)
    # Preserve NaN from ATR warmup
    momentum[:atr_period] = np.nan
    return momentum.astype(np.float32)


def upper_wick_ratio(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """Upper wick ratio: upper shadow relative to bar range.

    Large upper wick = rejection at highs, potential reversal signal.
    Flat bars (high == low) → 0.0.

    Returns:
        Float32 array in [0, 1]. No warmup, no NaN.
    """
    bar_range = high - low
    eps = np.float32(1e-10)
    upper_shadow = np.maximum(high - np.maximum(open_, close), np.float32(0.0))
    return (upper_shadow / np.maximum(bar_range, eps)).astype(np.float32)


def vol_regime(
    atr_values: np.ndarray,
    slow_period: int = 50,
) -> np.ndarray:
    """Volatility expansion/contraction ratio.

    Current ATR relative to its own long-term average.
    >1.0 = volatility expanding, <1.0 = contracting.

    Returns:
        Float32 array. NaN during warmup (first slow_period bars).
    """
    # Fill NaN from ATR warmup so cumsum-based _sma doesn't propagate NaN
    atr_clean = atr_values.copy()
    nan_mask = np.isnan(atr_clean)
    first_valid = np.argmin(nan_mask)  # index of first non-NaN
    if nan_mask.all():
        return np.full_like(atr_values, np.nan)
    atr_clean[:first_valid] = atr_clean[first_valid]

    sma_atr = _sma(atr_clean, slow_period)
    eps = np.float32(1e-10)
    ratio = atr_clean / np.maximum(sma_atr, eps)
    # NaN warmup: need slow_period bars of valid ATR for meaningful SMA
    warmup = max(slow_period, first_valid + slow_period)
    ratio[:warmup] = np.nan
    return ratio.astype(np.float32)


def trend_strength(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Kaufman Efficiency Ratio — directional strength of price movement.

    abs(net price change over N bars) / sum(abs(bar-to-bar changes)).
    Close to 1.0 = strong trend, close to 0.0 = choppy/ranging.

    Returns:
        Float32 array in [0, 1]. NaN during warmup (first period bars).
    """
    out = np.full_like(close, np.nan)
    if len(close) <= period:
        return out.astype(np.float32)

    abs_diff = np.abs(np.diff(close))
    for i in range(period, len(close)):
        net_change = abs(close[i] - close[i - period])
        sum_changes = np.sum(abs_diff[i - period:i])
        if sum_changes > 1e-10:
            out[i] = net_change / sum_changes
        else:
            out[i] = 0.0

    return out.astype(np.float32)


def mean_reversion_z(
    close: np.ndarray,
    atr_values: np.ndarray,
    sma_period: int = 50,
) -> np.ndarray:
    """ATR-normalized deviation from moving average.

    (close - SMA) / ATR — how far price has stretched from its mean,
    measured in volatility units. Positive = overbought, negative = oversold.

    Returns:
        Float32 array. NaN during warmup (first sma_period bars).
    """
    sma_close = _sma(close, sma_period)
    # Use cleaned ATR (fill NaN warmup) to avoid division issues
    atr_clean = atr_values.copy()
    nan_mask = np.isnan(atr_clean)
    if not nan_mask.all():
        first_valid = np.argmin(nan_mask)
        atr_clean[:first_valid] = atr_clean[first_valid]
    eps = np.float32(1e-10)
    z = (close - sma_close) / np.maximum(atr_clean, eps)
    z[:sma_period] = np.nan
    return z.astype(np.float32)


def compute_all_indicators(
    ohlcv: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute all standard indicators for one ticker's OHLCV data.

    Args:
        ohlcv: Dict with keys 'open', 'high', 'low', 'close', 'volume' (float32 arrays).

    Returns:
        Dict of indicator name → float32 array.
    """
    o = ohlcv["open"]
    c = ohlcv["close"]
    h = ohlcv["high"]
    l = ohlcv["low"]
    v = ohlcv["volume"]

    macd_line, macd_signal, macd_hist = macd(c)
    atr_vals = atr(h, l, c)

    result = {
        "rsi": rsi(c),
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "cci": cci(h, l, c),
        "bb_pct_b": bollinger_pct_b(c),
        "volume_ratio": volume_ratio(v),
        "atr": atr_vals,
        "trend_direction": trend_direction(c),
        "bar_body_ratio": bar_body_ratio(o, h, l, c),
        "close_range_position": close_range_position(h, l, c),
        "bar_momentum": bar_momentum(c, h, l, atr_values=atr_vals),
        "upper_wick_ratio": upper_wick_ratio(o, h, l, c),
        # Regime features
        "vol_regime": vol_regime(atr_vals),
        "trend_strength": trend_strength(c),
        "mean_reversion_z": mean_reversion_z(c, atr_vals),
    }

    # Sentiment features (injected from external data; zero-fallback if absent)
    n = len(c)
    if "news_sentiment" in ohlcv:
        result["news_sentiment"] = ohlcv["news_sentiment"].astype(np.float32)
        result["sentiment_momentum"] = _sma(ohlcv["news_sentiment"].astype(np.float32), 5)
        # Fill NaN warmup with zeros
        result["sentiment_momentum"] = np.nan_to_num(result["sentiment_momentum"], nan=0.0).astype(np.float32)
    else:
        result["news_sentiment"] = np.zeros(n, dtype=np.float32)
        result["sentiment_momentum"] = np.zeros(n, dtype=np.float32)

    return result


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
