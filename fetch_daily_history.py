"""Fetch 4+ years of daily bars from Alpaca and cache as parquet.

Run once before training. Fetches daily OHLCV for all tickers from
2022-01-01 to 2026-03-21, saving to data/bars_cache/.

Usage: python fetch_daily_history.py
"""

import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# Add project root and trading_agents parent to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Config ──────────────────────────────────────────────────────────────
TICKERS = [
    "NVDA", "TSLA", "AMD", "MARA", "COIN",
    "META", "AMZN", "GOOGL", "NFLX", "SPY",
]
START_DATE = date(2022, 1, 1)
END_DATE = date(2026, 3, 21)
CACHE_DIR = Path("data/bars_cache")
RATE_LIMIT_SECONDS = 3  # IEX free tier rate limit


def _load_env_file() -> None:
    """Load all keys from trading_agents/.env into os.environ."""
    env_path = Path(__file__).parent.parent / "trading_agents" / ".env"
    if not env_path.exists():
        print(f"WARNING: .env not found at {env_path}")
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and value and key not in os.environ:
                os.environ[key] = value


def main():
    _load_env_file()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        print("ERROR: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
        print("       Check trading_agents/.env")
        sys.exit(1)

    from alpaca_trade_api.rest import REST
    api = REST(
        key_id=api_key,
        secret_key=secret_key,
        base_url=os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
    )

    print(f"Fetching daily bars: {START_DATE} to {END_DATE}")
    print(f"Tickers: {', '.join(TICKERS)}")
    print(f"Cache dir: {CACHE_DIR}")
    print()

    results = {}
    for i, ticker in enumerate(TICKERS):
        cache_path = CACHE_DIR / f"{ticker}_1Day_{START_DATE}_{END_DATE}.parquet"

        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            print(f"  [{i+1}/{len(TICKERS)}] {ticker}: cache hit ({len(df)} bars)")
            results[ticker] = len(df)
            continue

        print(f"  [{i+1}/{len(TICKERS)}] {ticker}: downloading...", end=" ", flush=True)
        try:
            bars_df = api.get_bars(
                ticker,
                "1Day",
                start=START_DATE.isoformat(),
                end=(END_DATE + timedelta(days=1)).isoformat(),
                limit=50000,
                feed="iex",
                adjustment="split",
            ).df

            if bars_df is None or bars_df.empty:
                print("NO DATA")
                continue

            # Normalize columns
            bars_df.columns = [c.lower() for c in bars_df.columns]
            df = bars_df[["open", "high", "low", "close", "volume"]].copy()
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype("float32")

            df.to_parquet(cache_path, index=True)
            print(f"{len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")
            results[ticker] = len(df)

        except Exception as e:
            print(f"FAILED: {e}")

        # Rate limit between API calls
        if i < len(TICKERS) - 1:
            time.sleep(RATE_LIMIT_SECONDS)

    # Summary
    print(f"\n{'='*50}")
    print("FETCH COMPLETE")
    print(f"{'='*50}")
    for ticker, bars in results.items():
        print(f"  {ticker:6s}: {bars:5d} daily bars")
    if results:
        min_bars = min(results.values())
        max_bars = max(results.values())
        print(f"\n  Min bars: {min_bars}, Max bars: {max_bars}")
        print(f"  With 55-bar episodes: ~{max(1, min_bars - 55 + 1)} possible windows")
    else:
        print("  No data fetched!")


if __name__ == "__main__":
    main()
