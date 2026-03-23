"""Daily bar training — lower friction, longer horizons.

Resamples cached 5-min Alpaca data to daily OHLCV bars.
Uses DailyEnvConfig defaults: 60-bar episodes (~3 months),
100K initial cash, 3.5 BPS round-trip friction.
"""
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Force fresh start (no warm-start from old checkpoints)
os.environ["HYDRA_FRESH_START"] = "1"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/daily_training.log", mode="w"),
    ],
)
logger = logging.getLogger("run_daily")

from hydra.data.indicators import compute_all_indicators
from hydra.envs.trading_env import TradingEnv
from hydra.pipeline.env_builder import _compute_benchmark_returns
from hydra.pipeline.train_phase import run_training
from hydra.utils.numpy_opts import extract_ohlcv_arrays, SharedMarketData

# ── Config ──────────────────────────────────────────────────────────────
TICKERS = ["NVDA", "TSLA", "AMD", "MARA", "COIN",
           "META", "AMZN", "GOOGL", "NFLX"]
EPISODE_BARS = 55          # ~55 trading days in Jan-Mar 2026 (use all data as 1 episode)
INITIAL_CASH = 100_000.0   # 100K starting capital (DailyEnvConfig default)
NUM_GENERATIONS = 20
EPISODES_PER_GEN = 30
CACHE_DIR = Path("data/bars_cache")
START_DATE = "2026-01-01"
END_DATE = "2026-03-21"

# Daily bar friction (much lower than 5-min)
TRANSACTION_COST_BPS = 2.0  # vs 5.0 for 5-min
SLIPPAGE_BPS = 1.0          # vs 2.0 for 5-min
SPREAD_BPS = 0.5            # vs 1.0 for 5-min
BAR_INTERVAL_MINUTES = 1440 # 1 day


# ── Resample 5-min bars to daily ──────────────────────────────────────
def load_daily_bars(ticker: str) -> pd.DataFrame | None:
    """Load cached 5-min parquet, resample to daily OHLCV."""
    candidates = sorted(CACHE_DIR.glob(f"{ticker}_5Min_*.parquet"))
    if not candidates:
        logger.warning(f"No cached data for {ticker}")
        return None

    path = candidates[-1]
    df = pd.read_parquet(path)

    # Filter to date range
    df = df[df.index >= START_DATE]
    df = df[df.index < END_DATE]
    if len(df) == 0:
        logger.warning(f"No 2026 data for {ticker} in {path.name}")
        return None

    # Resample 5-min bars to daily
    df_daily = df.resample("1D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    logger.info(
        f"  {ticker}: {len(df)} 5-min bars -> {len(df_daily)} daily bars "
        f"({df_daily.index[0].date()} to {df_daily.index[-1].date()})"
    )
    return df_daily


def main():
    logger.info("=" * 60)
    logger.info("DAILY BAR TRAINING — Jan-Mar 2026, Lower Friction")
    logger.info("=" * 60)
    logger.info(f"  Friction: {TRANSACTION_COST_BPS} + {SLIPPAGE_BPS} + {SPREAD_BPS} = "
                f"{TRANSACTION_COST_BPS + SLIPPAGE_BPS + SPREAD_BPS} BPS one-way "
                f"({2*(TRANSACTION_COST_BPS + SLIPPAGE_BPS + SPREAD_BPS)} BPS round-trip)")
    logger.info(f"  Episode: {EPISODE_BARS} daily bars (~{EPISODE_BARS // 21:.0f} months)")
    logger.info(f"  Capital: ${INITIAL_CASH:,.0f}")

    # Load and resample market data
    logger.info(f"\nLoading {len(TICKERS)} tickers from cache (resampling to daily)...")
    all_ohlcv = {}
    all_indicators = {}
    min_bars = float("inf")

    for ticker in TICKERS:
        df = load_daily_bars(ticker)
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

    # Truncate all arrays to same length
    min_bars = int(min_bars)
    for ticker in available_tickers:
        for key in all_ohlcv[ticker]:
            all_ohlcv[ticker][key] = all_ohlcv[ticker][key][:min_bars]
        for key in all_indicators[ticker]:
            all_indicators[ticker][key] = all_indicators[ticker][key][:min_bars]

    logger.info(f"\nLoaded {len(available_tickers)} tickers, {min_bars} daily bars each")

    # Adjust episode_bars if we have fewer bars than configured
    episode_bars = min(EPISODE_BARS, min_bars - 1)
    num_episodes_available = min_bars // episode_bars
    logger.info(f"Episode bars: {episode_bars} (of {min_bars} available)")
    logger.info(f"Episodes available: {num_episodes_available}")

    # Build SharedMarketData
    timestamps = np.arange(min_bars, dtype=np.int32)
    market_data = SharedMarketData(
        ohlcv=all_ohlcv,
        indicators=all_indicators,
        timestamps=timestamps,
        tickers=available_tickers,
    )

    # Load SPY benchmark (also resampled to daily)
    spy_df = load_daily_bars("SPY")
    benchmark_data = None
    if spy_df is not None:
        spy_close = spy_df["close"].values.astype(np.float32)[:min_bars]
        benchmark_data = {"ticker": "SPY", "close": spy_close.tolist()}
        logger.info(f"  SPY benchmark: {len(spy_close)} daily bars")

    benchmark_returns = _compute_benchmark_returns(benchmark_data)

    # Build environments with daily bar settings
    logger.info("\nBuilding daily bar environments...")
    num_stocks = len(available_tickers)

    env_kwargs = {
        "market_data": market_data,
        "num_stocks": num_stocks,
        "episode_bars": episode_bars,
        "initial_cash": INITIAL_CASH,
        "benchmark_returns": benchmark_returns,
        "bar_interval_minutes": BAR_INTERVAL_MINUTES,
        "transaction_cost_bps": TRANSACTION_COST_BPS,
        "slippage_bps": SLIPPAGE_BPS,
        "spread_bps": SPREAD_BPS,
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
        "episode_bars": episode_bars,
    }

    logger.info(f"Environments built: {num_stocks} stocks, {episode_bars} daily bars/episode")
    logger.info(f"Obs dim: {envs['train_env'].observation_space.shape[0]}")
    logger.info(f"Bar interval: {BAR_INTERVAL_MINUTES} min (daily)")
    logger.info(f"Friction: {TRANSACTION_COST_BPS}/{SLIPPAGE_BPS}/{SPREAD_BPS} BPS "
                f"(tc/slip/spread)")

    # Build data_prep result dict
    data_result = {
        "market_data": market_data,
        "tickers": available_tickers,
        "num_days": num_episodes_available,
        "episode_bars": episode_bars,
        "total_bars": min_bars,
        "source": "historical_cache_daily",
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
        checkpoint_dir="checkpoints_daily_2026",
        tensorboard_dir="logs/tb_daily_2026",
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
        pnl = gen.get("agent_pnl", {})
        best_ret = gen.get("best_return_pct", 0.0)
        logger.info(
            f"Gen {gen['generation']:2d}: best={best:20s} (Sharpe={best_score:8.1f}), "
            f"return={best_ret:+.3f}%, pool={gen['pool_size']}, "
            f"severity={diag.get('severity', 'n/a')}"
        )

    # Final deployment check
    logger.info(f"\n{'='*60}")
    logger.info("DEPLOYMENT CHECK — Agent Behavior (Daily Bars)")
    logger.info(f"{'='*60}")

    from hydra.agents.agent_pool import AgentPool
    ckpt = Path(f"checkpoints_daily_2026/gen_{NUM_GENERATIONS}/episode_{EPISODES_PER_GEN}")
    if ckpt.exists():
        pool = AgentPool()
        pool.load(ckpt)
        env = envs["test_env"]

        for agent in pool.get_all():
            obs, info = env.reset(seed=999)
            cash_ratios = []
            step_returns = []
            total_reward = 0.0
            trades = 0

            for step in range(episode_bars):
                action = agent.select_action(obs, deterministic=True)
                obs, reward, term, trunc, step_info = env.step(action)
                cash_ratios.append(step_info.get("cash_ratio", 1.0))
                step_returns.append(step_info.get("step_return", 0.0))
                total_reward += reward
                if step_info.get("transaction_cost", 0) > 0:
                    trades += 1
                if term or trunc:
                    break

            mean_deploy = 1.0 - np.mean(cash_ratios)
            cumulative_return = float(np.prod([1 + r for r in step_returns]) - 1) * 100
            logger.info(
                f"  {agent.name:20s}: deploy={mean_deploy:5.1%}, "
                f"return={cumulative_return:+.2f}%, "
                f"reward={total_reward:8.1f}, trades={trades:3d}"
            )

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
