"""Feature caching layer — parquet and .npy storage for backtesting data.

Caches computed features to avoid redundant indicator calculation across
training episodes.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from hydra.data.indicators import compute_all_indicators
from hydra.utils.numpy_opts import extract_ohlcv_arrays

logger = logging.getLogger("hydra.data.feature_store")


class FeatureStore:
    """Caches OHLCV data and computed indicators as .npy files."""

    def __init__(self, cache_dir: str = "data/cache"):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, dict[str, np.ndarray]] = {}

    def get_or_compute(
        self,
        ticker: str,
        df: pd.DataFrame,
        cache_key: str | None = None,
    ) -> dict[str, np.ndarray]:
        """Get cached features or compute and cache them.

        Args:
            ticker: Stock ticker symbol.
            df: OHLCV DataFrame (columns: open, high, low, close, volume).
            cache_key: Optional cache key (e.g. date string). Auto-generated if None.

        Returns:
            Dict with OHLCV arrays + all indicator arrays, all float32.
        """
        key = cache_key or self._make_key(ticker, df)
        full_key = f"{ticker}_{key}"

        # Check memory cache first
        if full_key in self._memory_cache:
            return self._memory_cache[full_key]

        # Check disk cache
        disk_path = self._cache_dir / f"{full_key}.npz"
        if disk_path.exists():
            try:
                cached = dict(np.load(str(disk_path)))
                self._memory_cache[full_key] = cached
                return cached
            except Exception as e:
                logger.warning(f"Failed to load cache {disk_path}: {e}")

        # Compute features
        ohlcv = extract_ohlcv_arrays(df)
        indicators = compute_all_indicators(ohlcv)

        features = {**ohlcv, **indicators}

        # Cache to disk
        try:
            np.savez_compressed(str(disk_path), **features)
        except Exception as e:
            logger.warning(f"Failed to save cache {disk_path}: {e}")

        # Cache in memory
        self._memory_cache[full_key] = features
        return features

    def get_cached(self, ticker: str, cache_key: str) -> Optional[dict[str, np.ndarray]]:
        """Get features from cache only (no computation)."""
        full_key = f"{ticker}_{cache_key}"

        if full_key in self._memory_cache:
            return self._memory_cache[full_key]

        disk_path = self._cache_dir / f"{full_key}.npz"
        if disk_path.exists():
            try:
                cached = dict(np.load(str(disk_path)))
                self._memory_cache[full_key] = cached
                return cached
            except Exception:
                pass
        return None

    def clear_memory_cache(self) -> None:
        """Clear in-memory cache (disk cache persists)."""
        self._memory_cache.clear()

    def clear_all(self) -> None:
        """Clear both memory and disk cache."""
        self._memory_cache.clear()
        for f in self._cache_dir.glob("*.npz"):
            f.unlink()

    @staticmethod
    def _make_key(ticker: str, df: pd.DataFrame) -> str:
        """Generate a cache key from data content."""
        content = f"{ticker}_{len(df)}_{df.iloc[0].values.tobytes()}_{df.iloc[-1].values.tobytes()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
