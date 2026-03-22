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
        [0]            cash_ratio          (cash / initial_cash)
        [1:N+1]        holdings            (shares held per stock, normalized)
        [N+1:2N+1]     prices              (current close / episode start close)
        [2N+1:3N+1]    rsi                 (RSI / 100, so in [0, 1])
        [3N+1:4N+1]    macd_hist           (MACD histogram, z-scored)
        [4N+1:5N+1]    cci                 (CCI / 200, clipped to [-1, 1])
        [5N+1:6N+1]    bb_pct_b            (Bollinger %B, already [0, 1])
        [6N+1:7N+1]    volume_ratio        (vol / avg_vol, clipped to [0, 5])
        [7N+1:8N+1]    trend_direction     (SMA20/SMA50 crossover: -1, 0, +1)
        [8N+1:9N+1]    bar_body_ratio      (bar direction/conviction, [-1, 1])
        [9N+1:10N+1]   close_range_pos     (settlement strength, [0, 1])
        [10N+1:11N+1]  bar_momentum        (ATR-normalized momentum, [-1, 1])
        [11N+1:12N+1]  upper_wick_ratio    (rejection at highs, [0, 1])
        --- Regime features ---
        [12N+1:13N+1]  vol_regime          (ATR / long-term ATR, [0, 1])
        [13N+1:14N+1]  trend_strength      (Kaufman efficiency ratio, [0, 1])
        [14N+1:15N+1]  mean_reversion_z    (price deviation z-score, [-1, 1])
        --- Sentiment features ---
        [15N+1:16N+1]  news_sentiment      (news sentiment score, [-1, 1])
        [16N+1:17N+1]  sentiment_momentum  (5-bar SMA of sentiment, [-1, 1])
        --- Global features ---
        [17N+1]        drawdown            (current drawdown, negative)
        [17N+2]        daily_pnl           (today's P&L as fraction of initial)
        [17N+3]        session_label       (session type / 6.0, normalized to ~[0, 1])
        [17N+4]        time_progress       (bar_index / total_bars, [0, 1])

    Total dims = 1 + 17*N + 4 = 17*N + 5
    """

    def __init__(self, num_stocks: int, episode_bars: int = 78, normalize: bool = True):
        self.num_stocks = num_stocks
        self.episode_bars = episode_bars
        self.normalize = normalize
        self.obs_dim = 17 * num_stocks + 5

        # Pre-computed episode data (set at episode start)
        self._close_matrix: np.ndarray | None = None  # (bars, stocks)
        self._rsi_matrix: np.ndarray | None = None
        self._macd_matrix: np.ndarray | None = None
        self._cci_matrix: np.ndarray | None = None
        self._bb_matrix: np.ndarray | None = None
        self._vol_matrix: np.ndarray | None = None
        self._trend_matrix: np.ndarray | None = None
        self._body_ratio_matrix: np.ndarray | None = None
        self._close_range_matrix: np.ndarray | None = None
        self._momentum_matrix: np.ndarray | None = None
        self._upper_wick_matrix: np.ndarray | None = None
        # Regime features
        self._vol_regime_matrix: np.ndarray | None = None
        self._trend_strength_matrix: np.ndarray | None = None
        self._mr_z_matrix: np.ndarray | None = None
        # Sentiment features
        self._sentiment_matrix: np.ndarray | None = None
        self._sentiment_momentum_matrix: np.ndarray | None = None
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

        self._body_ratio_matrix = np.column_stack(
            [features[t].get("bar_body_ratio", np.zeros(bars, dtype=np.float32))[:bars] for t in tickers]
        ).astype(np.float32)

        self._close_range_matrix = np.column_stack(
            [features[t].get("close_range_position", np.full(bars, 0.5, dtype=np.float32))[:bars] for t in tickers]
        ).astype(np.float32)

        self._momentum_matrix = np.column_stack(
            [features[t].get("bar_momentum", np.zeros(bars, dtype=np.float32))[:bars] for t in tickers]
        ).astype(np.float32)

        self._upper_wick_matrix = np.column_stack(
            [features[t].get("upper_wick_ratio", np.zeros(bars, dtype=np.float32))[:bars] for t in tickers]
        ).astype(np.float32)

        # Regime features
        self._vol_regime_matrix = np.column_stack(
            [features[t].get("vol_regime", np.ones(bars, dtype=np.float32))[:bars] for t in tickers]
        ).astype(np.float32)

        self._trend_strength_matrix = np.column_stack(
            [features[t].get("trend_strength", np.zeros(bars, dtype=np.float32))[:bars] for t in tickers]
        ).astype(np.float32)

        self._mr_z_matrix = np.column_stack(
            [features[t].get("mean_reversion_z", np.zeros(bars, dtype=np.float32))[:bars] for t in tickers]
        ).astype(np.float32)

        # Sentiment features (zero-fallback when unavailable)
        self._sentiment_matrix = np.column_stack(
            [features[t].get("news_sentiment", np.zeros(bars, dtype=np.float32))[:bars] for t in tickers]
        ).astype(np.float32)

        self._sentiment_momentum_matrix = np.column_stack(
            [features[t].get("sentiment_momentum", np.zeros(bars, dtype=np.float32))[:bars] for t in tickers]
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
        np.nan_to_num(self._body_ratio_matrix, copy=False, nan=0.0)
        np.nan_to_num(self._close_range_matrix, copy=False, nan=0.5)
        np.nan_to_num(self._momentum_matrix, copy=False, nan=0.0)
        np.nan_to_num(self._upper_wick_matrix, copy=False, nan=0.0)
        np.nan_to_num(self._vol_regime_matrix, copy=False, nan=1.0)
        np.nan_to_num(self._trend_strength_matrix, copy=False, nan=0.0)
        np.nan_to_num(self._mr_z_matrix, copy=False, nan=0.0)
        np.nan_to_num(self._sentiment_matrix, copy=False, nan=0.0)
        np.nan_to_num(self._sentiment_momentum_matrix, copy=False, nan=0.0)

        # Pre-compute normalized observation template for all steps.
        # This moves 16 indicator normalizations from per-step to per-episode,
        # so build() just copies one row instead of doing 16 indexing + clip ops.
        safe_start = np.maximum(self._start_prices, np.float32(1e-8))
        template = np.empty((bars, 16 * n), dtype=np.float32)
        template[:, 0:n] = self._close_matrix / safe_start[np.newaxis, :]
        template[:, n:2*n] = self._rsi_matrix / np.float32(100.0)
        template[:, 2*n:3*n] = np.clip(self._macd_matrix / np.float32(5.0), -1.0, 1.0)
        template[:, 3*n:4*n] = np.clip(self._cci_matrix / np.float32(200.0), -1.0, 1.0)
        template[:, 4*n:5*n] = np.clip(self._bb_matrix, 0.0, 1.0)
        template[:, 5*n:6*n] = np.clip(self._vol_matrix, 0.0, 5.0) / np.float32(5.0)
        template[:, 6*n:7*n] = self._trend_matrix
        template[:, 7*n:8*n] = self._body_ratio_matrix
        template[:, 8*n:9*n] = self._close_range_matrix
        template[:, 9*n:10*n] = self._momentum_matrix
        template[:, 10*n:11*n] = self._upper_wick_matrix
        template[:, 11*n:12*n] = np.clip(self._vol_regime_matrix, 0.0, 3.0) / np.float32(3.0)
        template[:, 12*n:13*n] = np.clip(self._trend_strength_matrix, 0.0, 1.0)
        template[:, 13*n:14*n] = np.clip(self._mr_z_matrix, -3.0, 3.0) / np.float32(3.0)
        # Sentiment features (already in [-1, 1])
        template[:, 14*n:15*n] = np.clip(self._sentiment_matrix, -1.0, 1.0)
        template[:, 15*n:16*n] = np.clip(self._sentiment_momentum_matrix, -1.0, 1.0)
        self._obs_template = template

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

        Uses the pre-computed observation template for all indicator values
        (single memcpy), then fills in runtime portfolio state.

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

        # All 16 indicator groups — pre-computed and normalized at episode start
        obs[n + 1:17 * n + 1] = self._obs_template[step]

        # Global features
        idx = 17 * n + 1
        obs[idx] = np.float32((portfolio_value - peak_value) / max(peak_value, 1e-8))
        obs[idx + 1] = np.float32((portfolio_value - initial_cash) / max(initial_cash, 1e-8))
        obs[idx + 2] = np.float32(self._session_labels[step]) / np.float32(6.0)
        obs[idx + 3] = np.float32(step) / np.float32(max(self.episode_bars - 1, 1))

        return obs

    @property
    def observation_shape(self) -> tuple[int]:
        return (self.obs_dim,)

    def get_current_prices(self, step: int) -> np.ndarray:
        """Get close prices at the given step."""
        return self._close_matrix[step].copy()
