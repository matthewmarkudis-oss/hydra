"""Quick real-data training run with restructured reward function.

Uses cached 5-min Alpaca data from Jan-March 2026.
Loads parquet files directly (no trading_agents dependency needed).
"""
import logging
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Force fresh start (no warm-start from old checkpoints)
os.environ["HYDRA_FRESH_START"] = "1"

# Configure logging so we see progress
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/real_data_training.log", mode="w"),
    ],
)
logger = logging.getLogger("run_real_data")

from hydra.data.indicators import compute_all_indicators
from hydra.envs.trading_env import TradingEnv
from hydra.pipeline.env_builder import build_environments, _compute_benchmark_returns
from hydra.pipeline.train_phase import run_training
from hydra.utils.numpy_opts import extract_ohlcv_arrays, SharedMarketData

# ── Config ──────────────────────────────────────────────────────────────
TICKERS = ["NVDA", "TSLA", "AMD", "MARA", "COIN",
           "META", "AMZN", "GOOGL", "NFLX"]  # 9 high-vol tickers + SPY benchmark
EPISODE_BARS = 78          # 1 trading day
INITIAL_CASH = 10_000.0
NUM_GENERATIONS = 15
EPISODES_PER_GEN = 20
CACHE_DIR = Path("data/bars_cache")
START_DATE = "2026-01-01"
END_DATE = "2026-03-21"

# ── Load cached parquet data ────────────────────────────────────────────
def load_cached_bars(ticker: str) -> pd.DataFrame | None:
    """Load the most recent cached parquet file for a ticker."""
    candidates = sorted(CACHE_DIR.glob(f"{ticker}_5Min_*.parquet"))
    if not candidates:
        logger.warning(f"No cached data for {ticker}")
        return None
    # Use the latest file (sorted by name, last one has newest date range)
    path = candidates[-1]
    df = pd.read_parquet(path)
    # Filter to 2026 date range
    df = df[df.index >= START_DATE]
    df = df[df.index < END_DATE]
    if len(df) == 0:
        logger.warning(f"No 2026 data for {ticker} in {path.name}")
        return None
    logger.info(f"  {ticker}: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")
    return df


def main():
    logger.info("=" * 60)
    logger.info("REAL DATA TRAINING — Jan-Mar 2026, Restructured Rewards")
    logger.info("=" * 60)

    # Load market data
    logger.info(f"\nLoading {len(TICKERS)} tickers from cache...")
    all_ohlcv = {}
    all_indicators = {}
    min_bars = float("inf")

    for ticker in TICKERS:
        df = load_cached_bars(ticker)
        if df is None:
            continue
        ohlcv = extract_ohlcv_arrays(df)
        indicators = compute_all_indicators(ohlcv)
        all_ohlcv[ticker] = ohlcv
        all_indicators[ticker] = indicators
        min_bars = min(min_bars, len(ohlcv["close"]))

    available_tickers = list(all_ohlcv.keys())
    if len(available_tickers) < 2:
        logger.error("Not enough tickers with data!")
        return

    # Truncate all arrays to the same length
    min_bars = int(min_bars)
    for ticker in available_tickers:
        for key in all_ohlcv[ticker]:
            all_ohlcv[ticker][key] = all_ohlcv[ticker][key][:min_bars]
        for key in all_indicators[ticker]:
            all_indicators[ticker][key] = all_indicators[ticker][key][:min_bars]

    logger.info(f"\nLoaded {len(available_tickers)} tickers, {min_bars} bars each")
    num_days = min_bars // EPISODE_BARS
    logger.info(f"Episodes available: {num_days} (at {EPISODE_BARS} bars/episode)")

    # Build SharedMarketData
    timestamps = np.arange(min_bars, dtype=np.int32)
    market_data = SharedMarketData(
        ohlcv=all_ohlcv,
        indicators=all_indicators,
        timestamps=timestamps,
        tickers=available_tickers,
    )

    # Load SPY benchmark
    spy_df = load_cached_bars("SPY")
    benchmark_data = None
    if spy_df is not None:
        spy_close = spy_df["close"].values.astype(np.float32)[:min_bars]
        benchmark_data = {"ticker": "SPY", "close": spy_close.tolist()}
        logger.info(f"  SPY benchmark: {len(spy_close)} bars")

    benchmark_returns = _compute_benchmark_returns(benchmark_data)

    # Build environments
    logger.info("\nBuilding environments...")
    num_stocks = len(available_tickers)

    env_kwargs = {
        "market_data": market_data,
        "num_stocks": num_stocks,
        "episode_bars": EPISODE_BARS,
        "initial_cash": INITIAL_CASH,
        "benchmark_returns": benchmark_returns,
    }

    envs = {}
    for split in ["train", "val", "test"]:
        augment = (split == "train")
        envs[f"{split}_env"] = TradingEnv(
            augment=augment,
            seed=42 + hash(split),
            **env_kwargs,
        )
    envs["split_info"] = {
        "num_stocks": num_stocks,
        "episode_bars": EPISODE_BARS,
    }

    logger.info(f"Environments built: {num_stocks} stocks, {EPISODE_BARS} bars/episode")
    logger.info(f"Obs dim: {envs['train_env'].observation_space.shape[0]}")

    # Build data_prep result dict for train_phase compatibility
    data_result = {
        "market_data": market_data,
        "tickers": available_tickers,
        "num_days": num_days,
        "episode_bars": EPISODE_BARS,
        "total_bars": min_bars,
        "source": "historical_cache",
        "benchmark_data": benchmark_data,
    }

    # Run training
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING: {NUM_GENERATIONS} generations x {EPISODES_PER_GEN} episodes")
    logger.info(f"{'='*60}\n")

    results = run_training(
        deps={"env_builder": envs, "data_prep": data_result},
        num_generations=NUM_GENERATIONS,
        episodes_per_generation=EPISODES_PER_GEN,
        top_k_promote=2,
        bottom_k_demote=2,
        max_pool_size=15,
        checkpoint_dir="checkpoints_real_2026",
        tensorboard_dir="logs/tb_real_2026",
        prefer_gpu=False,
    )

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETE — Results Summary")
    logger.info(f"{'='*60}")

    for gen in results["training_results"]["generations"]:
        scores = gen["eval_scores"]
        best = max(scores, key=scores.get) if scores else "N/A"
        best_score = max(scores.values()) if scores else 0
        diag = gen.get("diagnosis", {})
        logger.info(
            f"Gen {gen['generation']:2d}: best={best:20s} ({best_score:8.1f}), "
            f"pool={gen['pool_size']}, severity={diag.get('severity', 'n/a')}"
        )

    # Final deployment check
    logger.info(f"\n{'='*60}")
    logger.info("DEPLOYMENT CHECK — Agent Behavior")
    logger.info(f"{'='*60}")

    from hydra.agents.agent_pool import AgentPool
    ckpt = Path(f"checkpoints_real_2026/gen_{NUM_GENERATIONS}/episode_{EPISODES_PER_GEN}")
    if ckpt.exists():
        pool = AgentPool()
        pool.load(ckpt)
        env = envs["test_env"]

        for agent in pool.get_all():
            obs, info = env.reset(seed=999)
            cash_ratios = []
            total_reward = 0.0
            trades = 0

            for step in range(EPISODE_BARS):
                action = agent.select_action(obs, deterministic=True)
                obs, reward, term, trunc, step_info = env.step(action)
                cash_ratios.append(step_info.get("cash_ratio", 1.0))
                total_reward += reward
                if step_info.get("transaction_cost", 0) > 0:
                    trades += 1
                if term or trunc:
                    break

            mean_deploy = 1.0 - np.mean(cash_ratios)
            logger.info(
                f"  {agent.name:20s}: deploy={mean_deploy:5.1%}, "
                f"reward={total_reward:8.1f}, trades={trades:3d}"
            )

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
