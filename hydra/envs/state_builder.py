"""Observation vector construction for the trading environment.

Builds a flat float32 numpy array from raw OHLCV data, indicators,
portfolio state, and global features. Pre-computes all per-episode
data at episode start for O(1) per-step observation construction.
"""

from __future__ import annotations

import numpy as np


class StateBuilder:
    """Constructs the observation vector for the RL agent.

    Observation layout (per num_stocks=N):
        [0]          cash_ratio          (cash / initial_cash)
        [1:N+1]      holdings            (shares held per stock, normalized)
        [N+1:2N+1]   prices              (current close / episode start close)
        [2N+1:3N+1]  rsi                 (RSI / 100, so in [0, 1])
        [3N+1:4N+1]  macd_hist           (MACD histogram, z-scored)
        [4N+1:5N+1]  cci                 (CCI / 200, clipped to [-1, 1])
        [5N+1:6N+1]  bb_pct_b            (Bollinger %B, already [0, 1])
        [6N+1:7N+1]  volume_ratio        (vol / avg_vol, clipped to [0, 5])
        [7N+1:8N+1]  trend_direction     (SMA20/SMA50 crossover: -1, 0, +1)
        [8N+1]       drawdown            (current drawdown, negative)
        [8N+2]       daily_pnl           (today's P&L as fraction of initial)
        [8N+3]       session_label       (session type / 6.0, normalized to ~[0, 1])
        [8N+4]       time_progress       (bar_index / total_bars, [0, 1])

    Total dims = 1 + 8*N + 4 = 8*N + 5
    """

    def __init__(self, num_stocks: int, episode_bars: int = 78, normalize: bool = True):
        self.num_stocks = num_stocks
        self.episode_bars = episode_bars
        self.normalize = normalize
        self.obs_dim = 8 * num_stocks + 5

        # Pre-computed episode data (set at episode start)
        self._close_matrix: np.ndarray | None = None  # (bars, stocks)
        self._rsi_matrix: np.ndarray | None = None
        self._macd_matrix: np.ndarray | None = None
        self._cci_matrix: np.ndarray | None = None
        self._bb_matrix: np.ndarray | None = None
        self._vol_matrix: np.ndarray | None = None
        self._trend_matrix: np.ndarray | None = None
        self._session_labels: np.ndarray | None = None
        self._start_prices: np.ndarray | None = None

    def init_episode(
        self,
        features: dict[str, dict[str, np.ndarray]],
        tickers: list[str],
        session_labels: np.ndarray | None = None,
    ) -> None:
        """Pre-compute all indicator matrices at episode start.

        Args:
            features: {ticker: {indicator_name: array}} — all float32.
            tickers: Ordered list of ticker symbols.
            session_labels: Pre-computed session labels (int8 array, len=episode_bars).
        """
        n = self.num_stocks
        bars = self.episode_bars

        self._close_matrix = np.column_stack(
            [features[t]["close"][:bars] for t in tickers]
        ).astype(np.float32)

        self._rsi_matrix = np.column_stack(
            [features[t]["rsi"][:bars] for t in tickers]
        ).astype(np.float32)

        self._macd_matrix = np.column_stack(
            [features[t]["macd_hist"][:bars] for t in tickers]
        ).astype(np.float32)

        self._cci_matrix = np.column_stack(
            [features[t]["cci"][:bars] for t in tickers]
        ).astype(np.float32)

        self._bb_matrix = np.column_stack(
            [features[t]["bb_pct_b"][:bars] for t in tickers]
        ).astype(np.float32)

        self._vol_matrix = np.column_stack(
            [features[t]["volume_ratio"][:bars] for t in tickers]
        ).astype(np.float32)

        self._trend_matrix = np.column_stack(
            [features[t].get("trend_direction", np.zeros(bars, dtype=np.float32))[:bars] for t in tickers]
        ).astype(np.float32)

        self._session_labels = session_labels if session_labels is not None else np.zeros(bars, dtype=np.int8)
        self._start_prices = self._close_matrix[0].copy()

        # Replace NaNs with neutral values
        np.nan_to_num(self._rsi_matrix, copy=False, nan=50.0)
        np.nan_to_num(self._macd_matrix, copy=False, nan=0.0)
        np.nan_to_num(self._cci_matrix, copy=False, nan=0.0)
        np.nan_to_num(self._bb_matrix, copy=False, nan=0.5)
        np.nan_to_num(self._vol_matrix, copy=False, nan=1.0)
        np.nan_to_num(self._trend_matrix, copy=False, nan=0.0)

    def build(
        self,
        step: int,
        cash: float,
        initial_cash: float,
        holdings: np.ndarray,
        portfolio_value: float,
        peak_value: float,
    ) -> np.ndarray:
        """Build observation vector for the current step.

        Args:
            step: Current bar index within the episode.
            cash: Current cash balance.
            initial_cash: Starting cash.
            holdings: Array of share counts per stock (float32, len=num_stocks).
            portfolio_value: Current total portfolio value.
            peak_value: Peak portfolio value so far this episode.

        Returns:
            Float32 array of shape (obs_dim,).
        """
        n = self.num_stocks
        obs = np.empty(self.obs_dim, dtype=np.float32)

        # Cash ratio
        obs[0] = np.float32(cash / max(initial_cash, 1e-8))

        # Holdings (normalized by a reference scale)
        obs[1:n + 1] = holdings.astype(np.float32) / np.float32(100.0)

        # Prices (normalized by episode start price)
        prices = self._close_matrix[step]
        obs[n + 1:2 * n + 1] = prices / np.maximum(self._start_prices, np.float32(1e-8))

        # RSI (0-100 → 0-1)
        obs[2 * n + 1:3 * n + 1] = self._rsi_matrix[step] / np.float32(100.0)

        # MACD histogram (z-score-like: divide by mean absolute value)
        macd_vals = self._macd_matrix[step]
        obs[3 * n + 1:4 * n + 1] = np.clip(macd_vals / np.float32(5.0), -1.0, 1.0)

        # CCI (-200 to 200 → -1 to 1)
        obs[4 * n + 1:5 * n + 1] = np.clip(self._cci_matrix[step] / np.float32(200.0), -1.0, 1.0)

        # Bollinger %B (already ~[0, 1])
        obs[5 * n + 1:6 * n + 1] = np.clip(self._bb_matrix[step], 0.0, 1.0)

        # Volume ratio (clipped to [0, 5], then /5 to normalize)
        obs[6 * n + 1:7 * n + 1] = np.clip(self._vol_matrix[step], 0.0, 5.0) / np.float32(5.0)

        # Trend direction (already -1, 0, +1)
        obs[7 * n + 1:8 * n + 1] = self._trend_matrix[step]

        # Global features
        idx = 8 * n + 1
        drawdown = (portfolio_value - peak_value) / max(peak_value, 1e-8)
        obs[idx] = np.float32(drawdown)

        daily_pnl = (portfolio_value - initial_cash) / max(initial_cash, 1e-8)
        obs[idx + 1] = np.float32(daily_pnl)

        obs[idx + 2] = np.float32(self._session_labels[step]) / np.float32(6.0)

        obs[idx + 3] = np.float32(step) / np.float32(max(self.episode_bars - 1, 1))

        return obs

    @property
    def observation_shape(self) -> tuple[int]:
        return (self.obs_dim,)

    def get_current_prices(self, step: int) -> np.ndarray:
        """Get close prices at the given step."""
        return self._close_matrix[step].copy()
