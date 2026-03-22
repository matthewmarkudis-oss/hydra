"""Phase 1: Data preparation — fetch historical bars, compute features, cache.

CPU-bound. Uses the DataAdapter for historical data and FeatureStore for caching.
For backtesting only.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import numpy as np

from hydra.compute.decorators import cpu_task
from hydra.data.adapter import DataAdapter, generate_synthetic_bars
from hydra.data.feature_store import FeatureStore
from hydra.data.indicators import compute_all_indicators
from hydra.utils.numpy_opts import extract_ohlcv_arrays, SharedMarketData

logger = logging.getLogger("hydra.pipeline.data_prep")


@cpu_task(workers=6)
def prepare_data(
    deps: dict[str, Any],
    tickers: list[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    episode_bars: int = 78,
    cache_dir: str = "data/cache",
    use_synthetic: bool = False,
    num_days: int = 20,
    seed: int = 42,
    adapter_config: dict | None = None,
) -> dict[str, Any]:
    """Prepare market data for RL training.

    Fetches historical 5-min bars, computes indicators, and caches as .npy.
    Returns a SharedMarketData object for the environment.

    Args:
        deps: Results from upstream workflow tasks (unused for first phase).
        tickers: List of ticker symbols.
        start_date: Start date for historical data.
        end_date: End date for historical data.
        episode_bars: Bars per episode.
        cache_dir: Directory for cached features.
        use_synthetic: If True, generate synthetic data (for testing).
        num_days: Number of trading days to prepare.
        seed: Random seed for synthetic data.
        adapter_config: Alpaca credentials dict (api_key, secret_key, base_url).

    Returns:
        Dict with 'market_data' (SharedMarketData), 'trading_dates', etc.
    """
    tickers = tickers or ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
                          "META", "TSLA", "JPM", "V", "UNH"]

    feature_store = FeatureStore(cache_dir)

    if use_synthetic:
        return _prepare_synthetic(tickers, episode_bars, num_days, seed, feature_store)

    return _prepare_historical(tickers, start_date, end_date, episode_bars, feature_store,
                               adapter_config=adapter_config)


def _prepare_synthetic(
    tickers: list[str],
    episode_bars: int,
    num_days: int,
    seed: int,
    feature_store: FeatureStore,
) -> dict[str, Any]:
    """Generate synthetic data for testing/development."""
    logger.info(f"Generating synthetic data: {len(tickers)} tickers, {num_days} days")

    all_ohlcv: dict[str, dict[str, np.ndarray]] = {}
    all_indicators: dict[str, dict[str, np.ndarray]] = {}

    total_bars = episode_bars * num_days

    for i, ticker in enumerate(tickers):
        df = generate_synthetic_bars(
            num_bars=total_bars,
            base_price=50.0 + i * 20.0,
            volatility=0.01 + i * 0.002,
            seed=seed + i,
        )

        ohlcv = extract_ohlcv_arrays(df)
        indicators = compute_all_indicators(ohlcv)

        all_ohlcv[ticker] = ohlcv
        all_indicators[ticker] = indicators

        # Cache
        feature_store.get_or_compute(ticker, df, cache_key=f"synthetic_{seed}")

    timestamps = np.arange(total_bars, dtype=np.int32)

    market_data = SharedMarketData(
        ohlcv=all_ohlcv,
        indicators=all_indicators,
        timestamps=timestamps,
        tickers=tickers,
    )

    # Generate synthetic benchmark series (simulates SPY-like index)
    benchmark_data = _generate_synthetic_benchmark(total_bars, seed)

    return {
        "market_data": market_data,
        "tickers": tickers,
        "num_days": num_days,
        "episode_bars": episode_bars,
        "total_bars": total_bars,
        "source": "synthetic",
        "benchmark_data": benchmark_data,
    }


def _prepare_historical(
    tickers: list[str],
    start_date: date | None,
    end_date: date | None,
    episode_bars: int,
    feature_store: FeatureStore,
    adapter_config: dict | None = None,
) -> dict[str, Any]:
    """Fetch and prepare historical data.

    Uses local Parquet cache (data/bars_cache/) so Alpaca API is only
    hit on the first run for a given ticker + date range.  Subsequent
    runs load from disk in seconds instead of minutes.
    """
    adapter = DataAdapter(config=adapter_config)

    start = start_date or date(2024, 1, 2)
    end = end_date or date(2024, 12, 31)
    trading_dates = adapter.get_trading_dates(start, end)

    logger.info(f"Fetching historical data: {len(tickers)} tickers, {len(trading_dates)} dates")

    all_ohlcv: dict[str, dict[str, np.ndarray]] = {}
    all_indicators: dict[str, dict[str, np.ndarray]] = {}

    for ticker in tickers:
        # Bulk fetch: one Alpaca call per ticker (or instant cache hit)
        df = adapter.get_bars_range(ticker, start, end, interval_minutes=5)

        if df is None or len(df) == 0:
            logger.warning(f"No data for {ticker}, using synthetic fallback")
            df = generate_synthetic_bars(
                num_bars=episode_bars * len(trading_dates),
                seed=hash(ticker) & 0xFFFFFFFF,
            )

        features = feature_store.get_or_compute(
            ticker, df, cache_key=f"hist_{start}_{end}"
        )

        ohlcv_keys = {"open", "high", "low", "close", "volume"}
        all_ohlcv[ticker] = {k: v for k, v in features.items() if k in ohlcv_keys}
        all_indicators[ticker] = {k: v for k, v in features.items() if k not in ohlcv_keys}

    total_bars = min(len(v["close"]) for v in all_ohlcv.values()) if all_ohlcv else 0
    timestamps = np.arange(total_bars, dtype=np.int32)

    market_data = SharedMarketData(
        ohlcv=all_ohlcv,
        indicators=all_indicators,
        timestamps=timestamps,
        tickers=tickers,
    )

    # Fetch SPY as benchmark (not part of trading universe)
    benchmark_data = _fetch_benchmark("SPY", adapter, start, end, trading_dates, episode_bars)

    return {
        "market_data": market_data,
        "tickers": tickers,
        "trading_dates": trading_dates,
        "num_days": len(trading_dates),
        "episode_bars": episode_bars,
        "total_bars": total_bars,
        "source": "historical",
        "benchmark_data": benchmark_data,
    }


def _fetch_benchmark(
    ticker: str,
    adapter: DataAdapter,
    start_date: date,
    end_date: date,
    trading_dates: list[date],
    episode_bars: int,
) -> dict[str, Any]:
    """Fetch benchmark ticker (e.g. SPY) close prices for comparison."""
    df = adapter.get_bars_range(ticker, start_date, end_date, interval_minutes=5)

    if df is None or len(df) == 0:
        logger.warning(f"No benchmark data for {ticker}, using synthetic fallback")
        return _generate_synthetic_benchmark(episode_bars * len(trading_dates), seed=99)

    close = df["close"].values.astype(np.float32)
    return {
        "ticker": ticker,
        "close": close.tolist(),
    }


def _generate_synthetic_benchmark(total_bars: int, seed: int) -> dict[str, Any]:
    """Generate a synthetic benchmark series (SPY-like index) for testing."""
    rng = np.random.default_rng(seed + 9999)
    # SPY-like: ~10% annual return, ~16% annual vol
    # At 5-min bars, ~78 bars/day, ~252 days/year => ~19,656 bars/year
    daily_return = 0.10 / 252
    daily_vol = 0.16 / np.sqrt(252)
    bar_return = daily_return / 78
    bar_vol = daily_vol / np.sqrt(78)

    returns = rng.normal(bar_return, bar_vol, total_bars).astype(np.float32)
    close = float(450.0) * np.cumprod(1 + returns).astype(np.float32)
    return {
        "ticker": "SPY_SYNTHETIC",
        "close": close.tolist(),
    }
