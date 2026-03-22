"""Bridge to trading_agents data infrastructure.

Wraps MarketDataProvider for Hydra's needs: historical 5-min bars as float32
numpy arrays for backtesting and RL training only.

Includes a local Parquet cache so historical data is downloaded from Alpaca
once and reused across training runs without repeated API calls.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("hydra.data.adapter")

# Default directory for cached Parquet bar files
_DEFAULT_CACHE_DIR = Path("data/bars_cache")


class DataAdapter:
    """Fetches and normalizes historical market data for the RL environment.

    Uses trading_agents.data.market_data.MarketDataProvider for historical data,
    or falls back to synthetic data generation for testing.
    Backtesting use only.

    Local Parquet cache: bars are stored per-ticker in
    ``data/bars_cache/{ticker}_{timeframe}_{start}_{end}.parquet``
    so subsequent runs skip Alpaca entirely for the same date range.
    """

    def __init__(self, config: dict | None = None, cache_dir: str | Path | None = None):
        self._config = config or {}
        self._provider = None
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._init_provider()

    def _init_provider(self) -> None:
        """Initialize the underlying data provider."""
        try:
            from trading_agents.data.market_data import MarketDataProvider
            self._provider = MarketDataProvider(self._config)
            logger.info("Using trading_agents MarketDataProvider")
        except (ImportError, Exception) as e:
            logger.warning(f"MarketDataProvider not available ({e}), using synthetic fallback")
            self._provider = None

    def get_intraday_bars(
        self,
        ticker: str,
        trade_date: date,
        interval_minutes: int = 5,
    ) -> Optional[pd.DataFrame]:
        """Fetch historical intraday bars for a single date (backtesting).

        Returns DataFrame with columns: open, high, low, close, volume
        All prices are float32.
        """
        if self._provider is not None:
            return self._fetch_via_provider(ticker, trade_date, interval_minutes)
        logger.warning(f"No data provider available for {ticker}, returning None")
        return None

    def get_daily_bars(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        limit: int = 252,
    ) -> Optional[pd.DataFrame]:
        """Fetch historical daily bars for a date range (backtesting)."""
        if self._provider is not None:
            try:
                df = self._provider.get_bars(ticker, timeframe="1Day", limit=limit)
                if df is not None and len(df) > 0:
                    return self._normalize_df(df)
            except Exception as e:
                logger.warning(f"Failed to fetch daily bars for {ticker}: {e}")
        return None

    def get_trading_dates(self, start_date: date, end_date: date) -> list[date]:
        """Return list of valid trading dates in range (excludes weekends and holidays)."""
        from hydra.utils.numpy_opts import is_market_holiday

        dates = []
        current = start_date
        while current <= end_date:
            if current.weekday() < 5 and not is_market_holiday(current):
                dates.append(current)
            current += timedelta(days=1)
        return dates

    def get_bars_range(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        interval_minutes: int = 5,
    ) -> Optional[pd.DataFrame]:
        """Fetch all intraday bars for a ticker over a date range.

        Uses local Parquet cache — downloads from Alpaca only on first call
        for a given ticker + date range.  Subsequent calls return instantly.

        Returns DataFrame with columns: open, high, low, close, volume (float32).
        """
        timeframe = f"{interval_minutes}Min"
        cache_key = f"{ticker}_{timeframe}_{start_date}_{end_date}"
        cache_path = self._cache_dir / f"{cache_key}.parquet"

        # --- Cache hit ---
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                logger.info(f"  Cache hit: {ticker} ({len(df)} bars)")
                return df
            except Exception as e:
                logger.warning(f"  Corrupt cache for {ticker}, re-fetching: {e}")
                cache_path.unlink(missing_ok=True)

        # --- Cache miss: fetch from Alpaca ---
        if self._provider is None:
            logger.warning(f"No data provider for {ticker}, cannot fetch")
            return None

        logger.info(f"  Downloading {ticker} {timeframe} bars: {start_date} to {end_date}")
        try:
            # Use the Alpaca REST API directly with start/end dates
            # to get the full range in one call instead of per-date.
            api = self._provider.api
            bars_df = api.get_bars(
                ticker,
                timeframe,
                start=start_date.isoformat(),
                end=(end_date + timedelta(days=1)).isoformat(),  # end is exclusive
                limit=50000,  # generous; ~78 bars/day * 250 days = ~19,500
                feed="iex",
                adjustment="split",
            ).df

            if bars_df is None or bars_df.empty:
                logger.warning(f"  No bars returned for {ticker}")
                return None

            df = self._normalize_df(bars_df)

            # Save to Parquet cache (non-fatal if it fails)
            try:
                df.to_parquet(cache_path, index=True)
                logger.info(f"  Cached {ticker}: {len(df)} bars -> {cache_path.name}")
            except Exception as cache_err:
                logger.warning(f"  Cache write failed for {ticker}: {cache_err}")

            return df

        except Exception as e:
            logger.warning(f"  Bulk fetch failed for {ticker}: {e}")
            return None

    def _fetch_via_provider(
        self, ticker: str, trade_date: date, interval_minutes: int
    ) -> Optional[pd.DataFrame]:
        """Fetch historical data via trading_agents MarketDataProvider."""
        try:
            timeframe = f"{interval_minutes}Min"
            # 78 bars/day * potential multi-day fetch; use generous limit
            df = self._provider.get_bars(ticker, timeframe=timeframe, limit=5000)
            if df is None or len(df) == 0:
                return None
            return self._normalize_df(df)
        except Exception as e:
            logger.warning(f"Provider fetch failed for {ticker} on {trade_date}: {e}")
            return None

    @staticmethod
    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame columns and dtypes."""
        col_map = {}
        for col in df.columns:
            lower = col.lower()
            if lower in ("open", "high", "low", "close", "volume"):
                col_map[col] = lower
        df = df.rename(columns=col_map)

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)

        return df[["open", "high", "low", "close", "volume"]]


def generate_synthetic_bars(
    num_bars: int = 78,
    num_days: int = 1,
    base_price: float = 100.0,
    volatility: float = 0.02,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate synthetic 5-min OHLCV bars for backtesting and testing.

    Returns DataFrame with columns: open, high, low, close, volume.
    """
    rng = np.random.default_rng(seed)
    total_bars = num_bars * num_days

    returns = rng.normal(0, volatility / np.sqrt(78), total_bars).astype(np.float32)
    close = np.float32(base_price) * np.cumprod(1 + returns).astype(np.float32)

    noise = rng.uniform(0.001, 0.005, total_bars).astype(np.float32)
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_price = np.empty_like(close)
    open_price[0] = np.float32(base_price)
    open_price[1:] = close[:-1]

    volume = (rng.lognormal(10, 1, total_bars) * 100).astype(np.float32)

    return pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
