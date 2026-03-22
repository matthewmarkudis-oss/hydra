"""Historical news sentiment data fetcher.

Fetches pre-computed sentiment scores from free APIs (Alpha Vantage, Finnhub)
and aligns them to 5-min bar timestamps for use as RL observation features.

Caches results as .npy files alongside price data to avoid redundant API calls.
Gracefully returns zero arrays when sentiment is unavailable.

Backtesting and training only.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("hydra.data.sentiment")

_DEFAULT_CACHE_DIR = Path("data/cache")


class SentimentAdapter:
    """Fetches and caches historical news sentiment per ticker.

    Supports two free-tier providers:
    - Alpha Vantage News Sentiment (25 req/day free)
    - Finnhub News Sentiment (60 req/min free)

    Sentiment is aggregated to 5-min bar resolution by averaging all
    news sentiment within each bar's time window.
    """

    def __init__(
        self,
        provider: str = "alphavantage",
        api_key: str | None = None,
        cache_dir: str | Path | None = None,
        rate_limit_delay: float = 2.0,
    ):
        self.provider = provider
        self.api_key = api_key or self._load_api_key(provider)
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._rate_limit_delay = rate_limit_delay

    @staticmethod
    def _load_api_key(provider: str) -> str:
        """Try to load API key from environment."""
        import os

        if provider == "alphavantage":
            return os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        elif provider == "finnhub":
            return os.environ.get("FINNHUB_API_KEY", "")
        return ""

    def get_sentiment(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        num_bars: int,
        bar_interval_minutes: int = 5,
    ) -> np.ndarray:
        """Get historical sentiment aligned to bar timestamps.

        Args:
            ticker: Stock ticker symbol.
            start_date: Start date for sentiment data.
            end_date: End date for sentiment data.
            num_bars: Number of bars to produce sentiment for.
            bar_interval_minutes: Bar interval (default 5 min).

        Returns:
            float32 array of shape (num_bars,) with values in [-1, +1].
            Returns zeros if sentiment is unavailable.
        """
        # Check cache first
        cache_path = self._cache_path(ticker, start_date, end_date)
        if cache_path.exists():
            cached = np.load(cache_path)
            if len(cached) >= num_bars:
                return cached[:num_bars].astype(np.float32)

        # Fetch from API
        if not self.api_key:
            logger.debug(f"No API key for {self.provider}, returning zero sentiment for {ticker}")
            return np.zeros(num_bars, dtype=np.float32)

        try:
            raw_sentiment = self._fetch_raw_sentiment(ticker, start_date, end_date)
            if not raw_sentiment:
                logger.info(f"No sentiment data found for {ticker}")
                return np.zeros(num_bars, dtype=np.float32)

            aligned = self._align_to_bars(
                raw_sentiment, start_date, end_date, num_bars, bar_interval_minutes
            )

            # Cache the result
            np.save(cache_path, aligned)
            logger.info(f"Cached sentiment for {ticker}: {len(aligned)} bars")

            return aligned

        except Exception as e:
            logger.warning(f"Sentiment fetch failed for {ticker}: {e}")
            return np.zeros(num_bars, dtype=np.float32)

    def _fetch_raw_sentiment(
        self, ticker: str, start_date: date, end_date: date
    ) -> list[dict]:
        """Fetch raw sentiment data from API.

        Returns list of {timestamp: datetime, score: float} dicts.
        """
        if self.provider == "alphavantage":
            return self._fetch_alphavantage(ticker, start_date, end_date)
        elif self.provider == "finnhub":
            return self._fetch_finnhub(ticker, start_date, end_date)
        else:
            logger.warning(f"Unknown provider: {self.provider}")
            return []

    def _fetch_alphavantage(
        self, ticker: str, start_date: date, end_date: date
    ) -> list[dict]:
        """Fetch from Alpha Vantage News Sentiment API."""
        import urllib.request
        import urllib.parse

        results = []
        time_from = start_date.strftime("%Y%m%dT0000")
        time_to = end_date.strftime("%Y%m%dT2359")

        url = (
            f"https://www.alphavantage.co/query?"
            f"function=NEWS_SENTIMENT&tickers={ticker}"
            f"&time_from={time_from}&time_to={time_to}"
            f"&apikey={self.api_key}&limit=200"
        )

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())

            feed = data.get("feed", [])
            for article in feed:
                ts_str = article.get("time_published", "")
                if not ts_str:
                    continue

                # Parse timestamp: "20240315T120000"
                try:
                    ts = datetime.strptime(ts_str[:15], "%Y%m%dT%H%M%S")
                except ValueError:
                    continue

                # Find ticker-specific sentiment
                for ts_item in article.get("ticker_sentiment", []):
                    if ts_item.get("ticker") == ticker:
                        score = float(ts_item.get("ticker_sentiment_score", 0))
                        results.append({"timestamp": ts, "score": score})
                        break

            time.sleep(self._rate_limit_delay)

        except Exception as e:
            logger.warning(f"Alpha Vantage request failed: {e}")

        return results

    def _fetch_finnhub(
        self, ticker: str, start_date: date, end_date: date
    ) -> list[dict]:
        """Fetch from Finnhub Company News API."""
        import urllib.request

        results = []
        url = (
            f"https://finnhub.io/api/v1/company-news?"
            f"symbol={ticker}"
            f"&from={start_date.isoformat()}&to={end_date.isoformat()}"
            f"&token={self.api_key}"
        )

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                articles = json.loads(resp.read().decode())

            for article in articles:
                ts_unix = article.get("datetime", 0)
                if not ts_unix:
                    continue
                ts = datetime.fromtimestamp(ts_unix)

                # Finnhub doesn't provide pre-computed sentiment.
                # Use headline length as a crude proxy, or default to neutral.
                # A proper implementation would use a local sentiment model.
                headline = article.get("headline", "")
                # Simple heuristic: positive words score +0.3, negative -0.3
                score = self._simple_headline_score(headline)
                results.append({"timestamp": ts, "score": score})

            time.sleep(self._rate_limit_delay)

        except Exception as e:
            logger.warning(f"Finnhub request failed: {e}")

        return results

    @staticmethod
    def _simple_headline_score(headline: str) -> float:
        """Very basic headline sentiment heuristic.

        This is a placeholder. A production system would use a proper
        sentiment model (FinBERT, etc.) or a pre-scored API.
        """
        headline_lower = headline.lower()
        positive = ["surge", "rally", "gain", "profit", "beat", "upgrade",
                     "growth", "record", "boost", "bullish", "soar", "jump"]
        negative = ["crash", "fall", "drop", "loss", "miss", "downgrade",
                     "decline", "plunge", "bearish", "sell", "warn", "cut"]

        pos_count = sum(1 for w in positive if w in headline_lower)
        neg_count = sum(1 for w in negative if w in headline_lower)

        if pos_count + neg_count == 0:
            return 0.0

        raw = (pos_count - neg_count) / (pos_count + neg_count)
        return max(-1.0, min(1.0, raw))

    def _align_to_bars(
        self,
        raw_sentiment: list[dict],
        start_date: date,
        end_date: date,
        num_bars: int,
        bar_interval_minutes: int,
    ) -> np.ndarray:
        """Align raw sentiment events to fixed-interval bar timestamps.

        Aggregates by averaging all sentiment scores within each bar's
        time window. Bars with no news get the previous bar's value
        (forward fill) to avoid sudden zeros.
        """
        result = np.zeros(num_bars, dtype=np.float32)

        if not raw_sentiment:
            return result

        # Build bar timestamps (simplified: assume market hours 9:30-16:00 ET)
        bars_per_day = int(390 / bar_interval_minutes)  # 78 bars for 5-min

        # Sort sentiment by timestamp
        sorted_sent = sorted(raw_sentiment, key=lambda x: x["timestamp"])

        # Create a simple mapping: for each bar index, collect matching sentiment
        current_date = start_date
        bar_idx = 0

        while current_date <= end_date and bar_idx < num_bars:
            # Skip weekends
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue

            market_open = datetime(current_date.year, current_date.month,
                                   current_date.day, 9, 30)

            for bar_in_day in range(bars_per_day):
                if bar_idx >= num_bars:
                    break

                bar_start = market_open + timedelta(minutes=bar_in_day * bar_interval_minutes)
                bar_end = bar_start + timedelta(minutes=bar_interval_minutes)

                # Collect sentiment in this window (including some pre-market)
                # Expand window to capture news from overnight/pre-market
                if bar_in_day == 0:
                    window_start = bar_start - timedelta(hours=14)  # previous day's close
                else:
                    window_start = bar_start

                scores = [
                    s["score"] for s in sorted_sent
                    if window_start <= s["timestamp"] < bar_end
                ]

                if scores:
                    result[bar_idx] = np.mean(scores).astype(np.float32)

                bar_idx += 1

            current_date += timedelta(days=1)

        # Forward fill: propagate last known sentiment to bars with no news
        last_val = np.float32(0.0)
        for i in range(num_bars):
            if result[i] != 0.0:
                last_val = result[i]
            else:
                result[i] = last_val * 0.95  # Decay towards zero

        return result

    def _cache_path(self, ticker: str, start_date: date, end_date: date) -> Path:
        """Generate cache file path for sentiment data."""
        return self._cache_dir / f"{ticker}_sentiment_{start_date}_{end_date}.npy"
