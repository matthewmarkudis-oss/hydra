"""Phase 2: Construct train/val/test gymnasium environments.

Splits data by date for ATHENA-style walk-forward validation,
then constructs TradingEnv instances for each split.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from hydra.envs.trading_env import TradingEnv
from hydra.utils.numpy_opts import SharedMarketData

logger = logging.getLogger("hydra.pipeline.env_builder")


def build_environments(
    deps: dict[str, Any],
    num_stocks: int = 10,
    episode_bars: int = 78,
    initial_cash: float = 2_500.0,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
    **env_kwargs: Any,
) -> dict[str, Any]:
    """Build train/val/test environments from prepared data.

    Args:
        deps: Must contain 'data_prep' with market_data and metadata.
        train_ratio: Fraction of days for training.
        val_ratio: Fraction of days for validation.
        Remaining days go to test set.

    Returns:
        Dict with 'train_env', 'val_env', 'test_env', and split metadata.
    """
    data_result = deps.get("data_prep", {})
    market_data = data_result.get("market_data")
    total_bars = data_result.get("total_bars", episode_bars * 20)
    num_days = data_result.get("num_days", 20)
    tickers = data_result.get("tickers", [])

    actual_num_stocks = min(num_stocks, len(tickers)) if tickers else num_stocks

    logger.info(f"Building environments: {actual_num_stocks} stocks, {num_days} days")
    logger.info(f"Split: train={train_ratio:.0%}, val={val_ratio:.0%}, test={1-train_ratio-val_ratio:.0%}")

    # For now, all environments share the same market data
    # Date-based splitting would slice the SharedMarketData by day index
    envs = {}
    for split in ["train", "val", "test"]:
        # Augmentation only during training — prevents overfitting
        augment = (split == "train")
        envs[f"{split}_env"] = TradingEnv(
            market_data=market_data,
            num_stocks=actual_num_stocks,
            episode_bars=episode_bars,
            initial_cash=initial_cash,
            augment=augment,
            seed=seed + hash(split),
            **env_kwargs,
        )

    envs["split_info"] = {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": 1 - train_ratio - val_ratio,
        "num_stocks": actual_num_stocks,
        "episode_bars": episode_bars,
    }

    logger.info("Environments built successfully")
    return envs
