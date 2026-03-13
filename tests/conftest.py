"""Shared test fixtures for the Hydra test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hydra.config.schema import HydraConfig
from hydra.data.adapter import generate_synthetic_bars
from hydra.data.indicators import compute_all_indicators
from hydra.envs.trading_env import TradingEnv
from hydra.utils.numpy_opts import extract_ohlcv_arrays, SharedMarketData


@pytest.fixture
def test_config():
    """Lightweight config for testing."""
    return HydraConfig(
        env={"num_stocks": 3, "episode_bars": 20, "initial_cash": 10_000.0},
        training={"total_timesteps": 100, "eval_interval": 50, "checkpoint_interval": 100,
                  "num_generations": 2, "episodes_per_generation": 5},
        data={"tickers": ["SYN000", "SYN001", "SYN002"], "lookback_days": 5},
        compute={"cpu_workers": 2, "prefer_gpu": False},
    )


@pytest.fixture
def synthetic_bars():
    """Generate synthetic OHLCV bars for one ticker."""
    return generate_synthetic_bars(num_bars=78, seed=42)


@pytest.fixture
def synthetic_ohlcv(synthetic_bars):
    """Float32 OHLCV numpy arrays."""
    return extract_ohlcv_arrays(synthetic_bars)


@pytest.fixture
def synthetic_features(synthetic_ohlcv):
    """OHLCV + all indicators."""
    indicators = compute_all_indicators(synthetic_ohlcv)
    return {**synthetic_ohlcv, **indicators}


@pytest.fixture
def shared_market_data():
    """SharedMarketData for 3 synthetic tickers."""
    tickers = ["SYN000", "SYN001", "SYN002"]
    num_bars = 20

    ohlcv = {}
    indicators = {}

    for i, ticker in enumerate(tickers):
        df = generate_synthetic_bars(num_bars=num_bars, base_price=50 + i * 20, seed=42 + i)
        o = extract_ohlcv_arrays(df)
        ind = compute_all_indicators(o)
        ohlcv[ticker] = o
        indicators[ticker] = ind

    return SharedMarketData(
        ohlcv=ohlcv,
        indicators=indicators,
        timestamps=np.arange(num_bars, dtype=np.int32),
        tickers=tickers,
    )


@pytest.fixture
def trading_env(shared_market_data):
    """A TradingEnv with synthetic data for testing."""
    return TradingEnv(
        market_data=shared_market_data,
        num_stocks=3,
        episode_bars=20,
        initial_cash=10_000.0,
        seed=42,
    )


@pytest.fixture
def small_env():
    """Minimal TradingEnv with synthetic data generation (no injected data)."""
    return TradingEnv(
        num_stocks=2,
        episode_bars=10,
        initial_cash=5_000.0,
        seed=42,
    )
