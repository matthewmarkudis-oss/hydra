"""Single-agent trading environment.

Gymnasium-compatible environment for RL training on 5-min bar data.
Each episode spans episode_bars bars (default 390 = 1 trading week).
All internals use float32 numpy.
For backtesting/simulation only.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from hydra.data.adapter import generate_synthetic_bars
from hydra.data.indicators import compute_all_indicators
from hydra.envs.action_processor import ActionProcessor
from hydra.envs.market_simulator import MarketSimulator
from hydra.envs.reward import DifferentialSharpeReward
from hydra.envs.session_manager import SessionManager
from hydra.envs.state_builder import StateBuilder
from hydra.risk.env_constraints import EnvConstraints
from hydra.risk.portfolio_risk import PortfolioRiskMonitor
from hydra.utils.numpy_opts import extract_ohlcv_arrays, SharedMarketData


class TradingEnv(gym.Env):
    """Single-agent trading environment.

    Observation: float32 vector of dimension 17*num_stocks + 5.
    Action: continuous Box[-1, +1] per stock.
    Reward: Differential Sharpe + P&L bonus + penalties.
    Episode: configurable window of 5-min bars (default 390 = 1 week).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        market_data: SharedMarketData | None = None,
        num_stocks: int = 10,
        episode_bars: int = 390,
        initial_cash: float = 100_000.0,
        transaction_cost_bps: float = 5.0,
        slippage_bps: float = 2.0,
        spread_bps: float = 1.0,
        max_position_pct: float = 0.30,
        max_drawdown_pct: float = 0.20,
        max_daily_loss_pct: float = 0.05,
        sharpe_eta: float = 0.05,
        drawdown_penalty: float = 0.15,
        transaction_penalty: float = 0.01,
        holding_penalty: float = 0.02,
        pnl_bonus_weight: float = 5.0,
        reward_scale: float = 100.0,
        cash_drag_penalty: float = 0.3,
        benchmark_bonus_weight: float = 2.0,
        min_deployment_pct: float = 0.3,
        dead_zone: float = 0.0,
        normalize_obs: bool = True,
        augment: bool = False,
        seed: int | None = None,
        render_mode: str | None = None,
        signal_provider=None,
        benchmark_returns: np.ndarray | None = None,
        bar_interval_minutes: int = 5,
    ):
        super().__init__()

        self.num_stocks = num_stocks
        self.episode_bars = episode_bars
        self.initial_cash = initial_cash
        self.render_mode = render_mode
        self._seed = seed
        self._augment = augment
        self._signal_provider = signal_provider
        self._benchmark_returns = benchmark_returns
        self._bar_interval_minutes = bar_interval_minutes

        # Store constructor kwargs for creating vectorized copies
        self._init_kwargs = {
            "market_data": market_data,
            "num_stocks": num_stocks,
            "episode_bars": episode_bars,
            "initial_cash": initial_cash,
            "transaction_cost_bps": transaction_cost_bps,
            "slippage_bps": slippage_bps,
            "spread_bps": spread_bps,
            "max_position_pct": max_position_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "max_daily_loss_pct": max_daily_loss_pct,
            "sharpe_eta": sharpe_eta,
            "drawdown_penalty": drawdown_penalty,
            "transaction_penalty": transaction_penalty,
            "holding_penalty": holding_penalty,
            "pnl_bonus_weight": pnl_bonus_weight,
            "reward_scale": reward_scale,
            "cash_drag_penalty": cash_drag_penalty,
            "benchmark_bonus_weight": benchmark_bonus_weight,
            "min_deployment_pct": min_deployment_pct,
            "dead_zone": dead_zone,
            "normalize_obs": normalize_obs,
            "augment": augment,
            "signal_provider": signal_provider,
            "benchmark_returns": benchmark_returns,
            "bar_interval_minutes": bar_interval_minutes,
        }

        # Components
        self.simulator = MarketSimulator(
            num_stocks=num_stocks,
            initial_cash=initial_cash,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            spread_bps=spread_bps,
        )

        self.constraints = EnvConstraints(
            max_position_pct=max_position_pct,
            max_drawdown_pct=max_drawdown_pct,
            max_daily_loss_pct=max_daily_loss_pct,
        )

        self.action_processor = ActionProcessor(
            num_stocks=num_stocks,
            constraints=self.constraints,
            dead_zone=dead_zone,
        )

        self.reward_fn = DifferentialSharpeReward(
            eta=sharpe_eta,
            drawdown_penalty=drawdown_penalty,
            transaction_penalty=transaction_penalty,
            holding_penalty=holding_penalty,
            pnl_bonus_weight=pnl_bonus_weight,
            reward_scale=reward_scale,
            cash_drag_penalty=cash_drag_penalty,
            benchmark_bonus_weight=benchmark_bonus_weight,
            min_deployment_pct=min_deployment_pct,
        )

        self.state_builder = StateBuilder(
            num_stocks=num_stocks,
            episode_bars=episode_bars,
            normalize=normalize_obs,
            signal_provider=signal_provider,
        )

        self.session_manager = SessionManager(bar_interval_minutes=bar_interval_minutes)
        self.risk_monitor = PortfolioRiskMonitor(num_stocks, initial_cash)

        # Market data (can be injected or generated synthetically)
        self._market_data = market_data
        self._synthetic_tickers = [f"SYN{i:03d}" for i in range(num_stocks)]

        # Spaces
        obs_dim = self.state_builder.obs_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(num_stocks,), dtype=np.float32,
        )

        # Episode state
        self._step_count = 0
        self._episode_features: dict[str, dict[str, np.ndarray]] = {}
        self._tickers: list[str] = []
        self._episode_day_index = 0
        self._num_episodes = 0
        self._total_days_available = 0

    def get_init_kwargs(self) -> dict[str, Any]:
        """Return constructor kwargs for creating matching env copies (e.g. for VecEnv)."""
        return dict(self._init_kwargs)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment for a new episode."""
        super().reset(seed=seed)

        self._step_count = 0
        self.simulator.reset()
        self.constraints.reset(self.initial_cash)
        self.reward_fn.reset(self.initial_cash)
        self.risk_monitor.reset(self.initial_cash)

        # Load or generate market data for this episode
        self._load_episode_data(options)

        # Pre-compute session labels (meaningless for daily bars)
        if self._bar_interval_minutes >= 1440:
            session_labels = np.zeros(self.episode_bars, dtype=np.int8)
        else:
            session_labels = self.session_manager.compute_session_labels(self.episode_bars)

        # Wire benchmark returns for the episode window
        if self._episode_benchmark_returns is not None:
            self.reward_fn.set_benchmark(self._episode_benchmark_returns)
        else:
            self.reward_fn.set_benchmark(None)

        # Initialize state builder with pre-computed features
        self.state_builder.init_episode(
            features=self._episode_features,
            tickers=self._tickers,
            session_labels=session_labels,
        )

        obs = self.state_builder.build(
            step=0,
            cash=float(self.simulator.cash),
            initial_cash=self.initial_cash,
            holdings=self.simulator.holdings,
            portfolio_value=self.initial_cash,
            peak_value=self.initial_cash,
        )

        info = {
            "episode": self._num_episodes,
            "tickers": self._tickers,
            "initial_cash": self.initial_cash,
        }

        self._num_episodes += 1
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step (one 5-min bar).

        Args:
            action: Continuous actions per stock, shape (num_stocks,).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        self._step_count += 1

        # Get current prices
        prices = self.state_builder.get_current_prices(
            min(self._step_count, self.episode_bars - 1)
        )

        # Get veto mask from signal provider (ZETA fundamental veto)
        veto_mask = None
        if self._signal_provider is not None:
            try:
                step_idx = min(self._step_count, self.episode_bars - 1)
                signals = self._signal_provider(step_idx)
                # Index 5 = zeta_veto_active (1 = veto is ON = block buys)
                if len(signals) > 5 and signals[5] > 0.5:
                    veto_mask = np.ones(self.num_stocks, dtype=np.float32)
                    # Veto blocks positive (buy) actions; sells still allowed
                    veto_mask[:] = -1.0  # Sentinel: only block buys
            except Exception:
                pass

        # Process actions through constraint system
        processed_actions = self.action_processor.process(
            raw_actions=action,
            holdings=self.simulator.holdings,
            prices=prices,
            portfolio_value=self.simulator.get_portfolio_value(prices),
            veto_mask=veto_mask,
        )

        # Execute orders
        shares_traded, transaction_cost = self.simulator.execute_orders(
            target_fractions=processed_actions,
            current_prices=prices,
        )

        # Compute portfolio value
        portfolio_value = self.simulator.get_portfolio_value(prices)

        # Compute reward
        reward, reward_info = self.reward_fn.compute(
            portfolio_value=portfolio_value,
            transaction_cost=transaction_cost,
            holdings=self.simulator.holdings,
            prices=prices,
        )

        # Check risk constraints
        step_return = reward_info.get("step_return", 0.0)
        should_truncate, is_halted, constraint_info = self.constraints.check_constraints(
            portfolio_value=portfolio_value,
            step_return=step_return,
        )

        # Update risk monitor
        risk_metrics = self.risk_monitor.update(portfolio_value)

        # Episode termination
        terminated = False
        truncated = should_truncate or (self._step_count >= self.episode_bars)

        # If halted by constraints, apply negative penalty (preserves learning signal)
        if is_halted and not should_truncate:
            reward = min(reward, -0.01 * float(self.reward_fn.reward_scale))

        # Build next observation
        obs_step = min(self._step_count, self.episode_bars - 1)
        obs = self.state_builder.build(
            step=obs_step,
            cash=float(self.simulator.cash),
            initial_cash=self.initial_cash,
            holdings=self.simulator.holdings,
            portfolio_value=portfolio_value,
            peak_value=self.reward_fn.peak_value,
        )

        info = {
            **reward_info,
            **constraint_info,
            **risk_metrics,
            "shares_traded": shares_traded.copy(),
            "transaction_cost": transaction_cost,
            "step": self._step_count,
            "halted": is_halted,
        }

        # End-of-episode summary
        if terminated or truncated:
            # Liquidate all positions at end of day
            self.simulator.liquidate_all(prices)
            info["episode_summary"] = self.risk_monitor.get_summary()
            info["total_transaction_costs"] = float(self.simulator.total_transaction_costs)
            info["num_trades"] = self.simulator.num_trades

        return obs, float(reward), terminated, truncated, info

    def _load_episode_data(self, options: dict | None) -> None:
        """Load market data for the current episode.

        When real market data is available, picks a random window of
        ``episode_bars`` bars from the full dataset.  Each reset() with a
        different RNG state produces a different window, so training and
        eval episodes see diverse market conditions instead of always
        replaying the first 78 bars.
        """
        self._episode_benchmark_returns = None

        if self._market_data is not None:
            self._tickers = list(self._market_data.tickers)[:self.num_stocks]
            total_bars = self._market_data.num_bars
            self._total_days_available = total_bars // self.episode_bars

            # Pick a random start offset within the data
            if total_bars > self.episode_bars:
                max_start = total_bars - self.episode_bars
                start = int(self.np_random.integers(0, max_start + 1))
            else:
                start = 0
            end = start + self.episode_bars

            # Slice benchmark returns to match the episode window
            if self._benchmark_returns is not None and len(self._benchmark_returns) > end:
                self._episode_benchmark_returns = self._benchmark_returns[start:end]

            self._episode_features = {}
            for ticker in self._tickers:
                ohlcv = self._market_data.get_ohlcv(ticker)
                features = {}
                for key, arr in ohlcv.items():
                    features[key] = arr[start:end]
                for ind_name in ("rsi", "macd_hist", "cci", "bb_pct_b", "volume_ratio", "trend_direction",
                                "bar_body_ratio", "close_range_position", "bar_momentum", "upper_wick_ratio",
                                "news_sentiment", "sentiment_momentum"):
                    try:
                        arr = self._market_data.get_indicator(ticker, ind_name)
                        features[ind_name] = arr[start:end]
                    except KeyError:
                        features[ind_name] = np.zeros(self.episode_bars, dtype=np.float32)
                self._episode_features[ticker] = features
        else:
            # Generate synthetic data for training/testing
            self._tickers = self._synthetic_tickers[:self.num_stocks]
            self._episode_features = {}
            rng_seed = (self._seed or 0) + self._num_episodes
            for i, ticker in enumerate(self._tickers):
                df = generate_synthetic_bars(
                    num_bars=self.episode_bars,
                    base_price=50.0 + i * 20.0,
                    volatility=0.01 + i * 0.002,
                    seed=rng_seed + i,
                )
                ohlcv = extract_ohlcv_arrays(df)
                indicators = compute_all_indicators(ohlcv)
                self._episode_features[ticker] = {**ohlcv, **indicators}

        # Apply data augmentation (training only)
        self._augment_episode_data()

    def _augment_episode_data(self) -> None:
        """Apply lightweight episode-level data augmentation.

        Only active when ``self._augment`` is True (training mode).
        Three augmentations are applied in sequence:

        1. **Price noise injection** -- multiplicative Normal(0, 0.001) noise
           on OHLC prices (same noise per bar to preserve relationships) and
           independent Normal(0, 0.01) noise on volume.
        2. **Ticker shuffle** -- 30% chance to randomly permute ticker order,
           preventing the agent from memorising positional stock identity.
        3. **Time masking** -- 20% chance to zero-out a contiguous block of
           5--15 bars across all technical indicator channels, forcing the
           agent to be robust to missing indicator data.
        """
        if not self._augment:
            return

        rng = self.np_random
        n_bars = self.episode_bars
        price_keys = ("open", "high", "low", "close")
        indicator_keys = (
            "rsi", "macd_hist", "cci", "bb_pct_b", "volume_ratio",
            "bar_body_ratio", "close_range_position", "bar_momentum",
            "upper_wick_ratio",
        )

        # --- 1) Price noise injection (multiplicative) ---
        for ticker in self._tickers:
            feats = self._episode_features[ticker]
            # Same noise for OHLC within each bar, different across bars
            price_noise = rng.standard_normal(n_bars).astype(np.float32) * 0.001
            price_mult = np.float32(1.0) + price_noise
            for key in price_keys:
                if key in feats:
                    feats[key] = feats[key] * price_mult
            # Independent noise for volume
            if "volume" in feats:
                vol_noise = rng.standard_normal(n_bars).astype(np.float32) * 0.01
                vol_mult = np.float32(1.0) + vol_noise
                feats["volume"] = feats["volume"] * vol_mult
                # Volume must stay non-negative
                np.maximum(feats["volume"], 0.0, out=feats["volume"])

        # --- 2) Ticker shuffle (30% probability) ---
        if rng.random() < 0.3 and len(self._tickers) > 1:
            perm = rng.permutation(len(self._tickers))
            shuffled_tickers = [self._tickers[i] for i in perm]
            shuffled_features = {
                shuffled_tickers[j]: self._episode_features[self._tickers[perm[j]]]
                for j in range(len(shuffled_tickers))
            }
            self._tickers = shuffled_tickers
            self._episode_features = shuffled_features

        # --- 3) Time masking (20% probability) ---
        if rng.random() < 0.2:
            mask_len = int(rng.integers(5, 16))  # 5..15 inclusive
            max_start = max(n_bars - mask_len, 0)
            mask_start = int(rng.integers(0, max_start + 1))
            mask_end = mask_start + mask_len
            for ticker in self._tickers:
                feats = self._episode_features[ticker]
                for ind in indicator_keys:
                    if ind in feats:
                        feats[ind][mask_start:mask_end] = 0.0

    def render(self) -> None:
        """Render current state (human-readable)."""
        if self.render_mode != "human":
            return
        prices = self.state_builder.get_current_prices(
            min(self._step_count, self.episode_bars - 1)
        )
        pv = self.simulator.get_portfolio_value(prices)
        print(
            f"Step {self._step_count}/{self.episode_bars} | "
            f"Cash: ${self.simulator.cash:.2f} | "
            f"Portfolio: ${pv:.2f} | "
            f"Trades: {self.simulator.num_trades}"
        )
