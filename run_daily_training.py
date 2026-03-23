"""Daily bar training — 4-year history with temporal train/val/test splits.

Loads daily bars from Alpaca cache (fetched by fetch_daily_history.py).
Splits data temporally: train (2022-2025H1), val (2025H2), test (2026Q1).
With ~875 training bars and 55-bar episodes, agents see hundreds of
unique episodes across bull/bear/sideways markets.
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
EPISODE_BARS = 55          # ~2.5 months per episode
INITIAL_CASH = 100_000.0   # 100K starting capital
NUM_GENERATIONS = 30
EPISODES_PER_GEN = 40
CACHE_DIR = Path("data/bars_cache")

# Full date range (4+ years of history)
START_DATE = "2022-01-01"
END_DATE = "2026-03-21"

# Temporal split boundaries
TRAIN_END = "2025-07-01"   # Train: 2022-01 to 2025-06 (~875 bars)
VAL_END = "2026-01-01"     # Val:   2025-07 to 2025-12 (~125 bars)
                            # Test:  2026-01 to 2026-03 (~55 bars)

# Daily bar friction (much lower than 5-min)
TRANSACTION_COST_BPS = 2.0  # vs 5.0 for 5-min
SLIPPAGE_BPS = 1.0          # vs 2.0 for 5-min
SPREAD_BPS = 0.5            # vs 1.0 for 5-min
BAR_INTERVAL_MINUTES = 1440 # 1 day


# ── Load daily bars from cache ──────────────────────────────────────────
def load_daily_bars(ticker: str) -> pd.DataFrame | None:
    """Load cached daily parquet file for a ticker.

    Looks for 1Day parquets first (from fetch_daily_history.py),
    falls back to resampling from 5-min cache if needed.
    """
    # Try daily cache first
    candidates = sorted(CACHE_DIR.glob(f"{ticker}_1Day_*.parquet"))
    if candidates:
        path = candidates[-1]
        df = pd.read_parquet(path)
        df.columns = [c.lower() for c in df.columns]
        df = df[df.index >= START_DATE]
        df = df[df.index < END_DATE]
        if len(df) > 0:
            logger.info(
                f"  {ticker}: {len(df)} daily bars "
                f"({df.index[0].date()} to {df.index[-1].date()})"
            )
            return df

    # Fallback: resample from 5-min cache
    candidates_5m = sorted(CACHE_DIR.glob(f"{ticker}_5Min_*.parquet"))
    if not candidates_5m:
        logger.warning(f"No cached data for {ticker}. Run fetch_daily_history.py first.")
        return None

    path = candidates_5m[-1]
    df = pd.read_parquet(path)
    df = df[df.index >= START_DATE]
    df = df[df.index < END_DATE]
    if len(df) == 0:
        logger.warning(f"No data for {ticker} in range")
        return None

    df_daily = df.resample("1D").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()

    logger.info(
        f"  {ticker}: {len(df)} 5-min bars -> {len(df_daily)} daily bars "
        f"({df_daily.index[0].date()} to {df_daily.index[-1].date()}) [resampled]"
    )
    return df_daily


def split_market_data(
    all_ohlcv: dict,
    all_indicators: dict,
    daily_index: pd.DatetimeIndex,
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> tuple[dict, dict, int]:
    """Slice all ticker data to [start_date, end_date) by index position.

    Returns (ohlcv_dict, indicators_dict, num_bars).
    """
    mask = (daily_index >= start_date) & (daily_index < end_date)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return {}, {}, 0
    start_idx, end_idx = int(indices[0]), int(indices[-1]) + 1

    sliced_ohlcv = {}
    sliced_indicators = {}
    for ticker in tickers:
        sliced_ohlcv[ticker] = {
            k: v[start_idx:end_idx].copy() for k, v in all_ohlcv[ticker].items()
        }
        sliced_indicators[ticker] = {
            k: v[start_idx:end_idx].copy() for k, v in all_indicators[ticker].items()
        }
    return sliced_ohlcv, sliced_indicators, end_idx - start_idx


def main():
    logger.info("=" * 60)
    logger.info("DAILY BAR TRAINING — 4-Year History, Temporal Splits")
    logger.info("=" * 60)
    logger.info(f"  Date range: {START_DATE} to {END_DATE}")
    logger.info(f"  Train: {START_DATE} to {TRAIN_END}")
    logger.info(f"  Val:   {TRAIN_END} to {VAL_END}")
    logger.info(f"  Test:  {VAL_END} to {END_DATE}")
    logger.info(f"  Friction: {TRANSACTION_COST_BPS} + {SLIPPAGE_BPS} + {SPREAD_BPS} = "
                f"{TRANSACTION_COST_BPS + SLIPPAGE_BPS + SPREAD_BPS} BPS one-way "
                f"({2*(TRANSACTION_COST_BPS + SLIPPAGE_BPS + SPREAD_BPS)} BPS round-trip)")
    logger.info(f"  Episode: {EPISODE_BARS} daily bars (~{EPISODE_BARS // 21:.0f} months)")
    logger.info(f"  Capital: ${INITIAL_CASH:,.0f}")

    # ── Load market data ────────────────────────────────────────────────
    logger.info(f"\nLoading {len(TICKERS)} tickers from cache...")
    all_dfs = {}
    for ticker in TICKERS:
        df = load_daily_bars(ticker)
        if df is not None:
            all_dfs[ticker] = df

    available_tickers = list(all_dfs.keys())
    if len(available_tickers) < 2:
        logger.error("Not enough tickers with data!")
        return

    # Find common date range (intersection of all ticker indices)
    common_index = all_dfs[available_tickers[0]].index
    for ticker in available_tickers[1:]:
        common_index = common_index.intersection(all_dfs[ticker].index)
    common_index = common_index.sort_values()
    min_bars = len(common_index)

    logger.info(f"\nLoaded {len(available_tickers)} tickers, {min_bars} common daily bars")
    logger.info(f"  Date range: {common_index[0].date()} to {common_index[-1].date()}")

    # Align all dataframes to common index, compute OHLCV and indicators
    all_ohlcv = {}
    all_indicators = {}
    for ticker in available_tickers:
        df_aligned = all_dfs[ticker].reindex(common_index).ffill().bfill()
        ohlcv = extract_ohlcv_arrays(df_aligned)
        indicators = compute_all_indicators(ohlcv)
        all_ohlcv[ticker] = ohlcv
        all_indicators[ticker] = indicators

    # ── Temporal split ──────────────────────────────────────────────────
    logger.info("\nSplitting data temporally...")
    splits = {
        "train": (START_DATE, TRAIN_END),
        "val":   (TRAIN_END, VAL_END),
        "test":  (VAL_END, END_DATE),
    }

    split_data = {}
    for name, (s, e) in splits.items():
        ohlcv_s, ind_s, n_bars = split_market_data(
            all_ohlcv, all_indicators, common_index, available_tickers, s, e
        )
        if n_bars == 0:
            logger.warning(f"  {name}: NO DATA in range {s} to {e}")
            continue
        ts = np.arange(n_bars, dtype=np.int32)
        split_data[name] = {
            "market_data": SharedMarketData(
                ohlcv=ohlcv_s, indicators=ind_s,
                timestamps=ts, tickers=available_tickers,
            ),
            "num_bars": n_bars,
        }
        windows = max(1, n_bars - EPISODE_BARS + 1)
        logger.info(f"  {name:5s}: {n_bars} bars, {windows} possible episode windows")

    if "train" not in split_data:
        logger.error("No training data!")
        return

    # Adjust episode_bars if any split is too small
    train_bars = split_data["train"]["num_bars"]
    episode_bars = min(EPISODE_BARS, train_bars - 1)
    logger.info(f"\nEpisode bars: {episode_bars}")

    # ── SPY benchmark (split per env) ───────────────────────────────────
    spy_df = load_daily_bars("SPY")
    benchmark_splits = {}
    if spy_df is not None:
        spy_aligned = spy_df.reindex(common_index).ffill().bfill()
        spy_close = spy_aligned["close"].values.astype(np.float32)
        for name, (s, e) in splits.items():
            mask = (common_index >= s) & (common_index < e)
            indices = np.where(mask)[0]
            if len(indices) > 0:
                spy_slice = spy_close[indices[0]:indices[-1]+1]
                bm_data = {"ticker": "SPY", "close": spy_slice.tolist()}
                benchmark_splits[name] = _compute_benchmark_returns(bm_data)
                logger.info(f"  SPY benchmark ({name}): {len(spy_slice)} bars")
            else:
                benchmark_splits[name] = None
    else:
        for name in splits:
            benchmark_splits[name] = None

    # ── Build environments ──────────────────────────────────────────────
    logger.info("\nBuilding daily bar environments...")
    num_stocks = len(available_tickers)

    envs = {}
    for split_name in ["train", "val", "test"]:
        if split_name not in split_data:
            continue
        sd = split_data[split_name]
        augment = (split_name == "train")
        ep = min(episode_bars, sd["num_bars"] - 1)
        envs[f"{split_name}_env"] = TradingEnv(
            market_data=sd["market_data"],
            num_stocks=num_stocks,
            episode_bars=ep,
            initial_cash=INITIAL_CASH,
            augment=augment,
            seed=42 + hash(split_name),
            benchmark_returns=benchmark_splits.get(split_name),
            bar_interval_minutes=BAR_INTERVAL_MINUTES,
            transaction_cost_bps=TRANSACTION_COST_BPS,
            slippage_bps=SLIPPAGE_BPS,
            spread_bps=SPREAD_BPS,
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
    train_md = split_data["train"]["market_data"]
    data_result = {
        "market_data": train_md,
        "tickers": available_tickers,
        "num_days": train_bars // episode_bars,
        "episode_bars": episode_bars,
        "total_bars": train_bars,
        "source": "historical_cache_daily_4yr",
        "benchmark_data": {"ticker": "SPY"} if spy_df is not None else None,
    }

    # ── Run training ────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING: {NUM_GENERATIONS} generations x {EPISODES_PER_GEN} episodes")
    logger.info(f"  Train data: {train_bars} bars, ~{max(1, train_bars - episode_bars + 1)} episode windows")
    logger.info(f"{'='*60}\n")

    results = run_training(
        deps={"env_builder": envs, "data_prep": data_result},
        num_generations=NUM_GENERATIONS,
        episodes_per_generation=EPISODES_PER_GEN,
        top_k_promote=2,
        bottom_k_demote=2,
        max_pool_size=15,
        checkpoint_dir="checkpoints_daily_4yr",
        tensorboard_dir="logs/tb_daily_4yr",
        prefer_gpu=False,
    )

    # ── Print summary ───────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETE — Results Summary")
    logger.info(f"{'='*60}")

    for gen in results["training_results"]["generations"]:
        scores = gen["eval_scores"]
        best = max(scores, key=scores.get) if scores else "N/A"
        best_score = max(scores.values()) if scores else 0
        diag = gen.get("diagnosis", {})
        best_ret = gen.get("best_return_pct", 0.0)
        logger.info(
            f"Gen {gen['generation']:2d}: best={best:20s} (Sharpe={best_score:8.1f}), "
            f"return={best_ret:+.3f}%, pool={gen['pool_size']}, "
            f"severity={diag.get('severity', 'n/a')}"
        )

    # ── Final deployment check on TEST data ─────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("DEPLOYMENT CHECK — Agent Behavior (Test: 2026 Q1)")
    logger.info(f"{'='*60}")

    from hydra.agents.agent_pool import AgentPool
    ckpt = Path(f"checkpoints_daily_4yr/gen_{NUM_GENERATIONS}/episode_{EPISODES_PER_GEN}")
    if ckpt.exists():
        pool = AgentPool()
        pool.load(ckpt)
        env = envs["test_env"]
        test_bars = split_data.get("test", {}).get("num_bars", episode_bars)

        for agent in pool.get_all():
            obs, info = env.reset(seed=999)
            cash_ratios = []
            step_returns = []
            total_reward = 0.0
            trades = 0

            for step in range(min(episode_bars, test_bars)):
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
