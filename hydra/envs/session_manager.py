"""Vectorized session classification for intraday trading.

Pre-computes session labels for all bars at episode start.
Uses frozenset holiday lookups and numpy vectorized operations.
"""

from __future__ import annotations

from datetime import date, time

import numpy as np
import pandas as pd

from hydra.utils.numpy_opts import (
    MARKET_OPEN,
    MORNING_END,
    AFTERNOON_START,
    POWER_HOUR,
    MARKET_CLOSE,
    classify_sessions_vectorized,
    is_market_holiday,
)


# Session label constants
SESSION_PREMARKET = 0
SESSION_OPEN_AUCTION = 1
SESSION_MORNING = 2
SESSION_MIDDAY = 3
SESSION_AFTERNOON = 4
SESSION_POWER_HOUR = 5
SESSION_CLOSE = 6

SESSION_NAMES = {
    0: "premarket",
    1: "open_auction",
    2: "morning",
    3: "midday",
    4: "afternoon",
    5: "power_hour",
    6: "close",
}


class SessionManager:
    """Pre-computes session labels and trading schedule for episodes."""

    def __init__(self, bar_interval_minutes: int = 5):
        self.bar_interval = bar_interval_minutes

    def compute_session_labels(self, num_bars: int, market_open_minutes: int = MARKET_OPEN) -> np.ndarray:
        """Pre-compute session labels for an episode's worth of bars.

        Args:
            num_bars: Number of bars in the episode.
            market_open_minutes: Market open time in minutes from midnight (default 9:30 = 570).

        Returns:
            Int8 array of session labels, one per bar.
        """
        minutes = np.arange(num_bars, dtype=np.int32) * self.bar_interval + market_open_minutes
        return classify_sessions_vectorized(minutes)

    def compute_session_labels_from_timestamps(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Compute session labels from actual bar timestamps.

        Args:
            timestamps: DatetimeIndex of bar timestamps (Eastern Time assumed).

        Returns:
            Int8 array of session labels.
        """
        minutes = (timestamps.hour * 60 + timestamps.minute).values.astype(np.int32)
        return classify_sessions_vectorized(minutes)

    def is_trading_day(self, d: date) -> bool:
        """Check if a date is a valid trading day."""
        if d.weekday() >= 5:  # Weekend
            return False
        return not is_market_holiday(d)

    def get_session_weights(self, labels: np.ndarray) -> np.ndarray:
        """Return per-bar weights based on session (higher for high-volume sessions).

        Open auction and power hour get higher weight reflecting
        higher volume/volatility during those periods.
        """
        weights = np.ones(len(labels), dtype=np.float32)
        weights[labels == SESSION_OPEN_AUCTION] = 1.5
        weights[labels == SESSION_MORNING] = 1.2
        weights[labels == SESSION_MIDDAY] = 0.8
        weights[labels == SESSION_AFTERNOON] = 1.0
        weights[labels == SESSION_POWER_HOUR] = 1.3
        return weights

    def get_bar_timestamps(
        self,
        trade_date: date,
        num_bars: int,
    ) -> np.ndarray:
        """Generate bar timestamps for an episode (for indexing/logging).

        Returns array of minutes-from-midnight as int32.
        """
        start_minutes = MARKET_OPEN
        return np.arange(num_bars, dtype=np.int32) * self.bar_interval + start_minutes
