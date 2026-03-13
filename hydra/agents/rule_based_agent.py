"""Rule-based agent adapter.

Wraps existing trading_agents Alpha/Beta strategies to produce
continuous actions compatible with the RL environment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from hydra.agents.base_rl_agent import BaseRLAgent

logger = logging.getLogger("hydra.agents.rule_based")


class RuleBasedAgent(BaseRLAgent):
    """Adapter wrapping existing BaseAgent subclasses from trading_agents.

    Translates TradeSignal outputs into continuous [-1, +1] actions.
    """

    def __init__(
        self,
        name: str,
        obs_dim: int,
        action_dim: int,
        agent_class_path: str | None = None,
        tickers: list[str] | None = None,
        config: dict | None = None,
    ):
        super().__init__(name, obs_dim, action_dim)
        self._frozen = True  # Rule-based agents don't learn
        self._agent_class_path = agent_class_path
        self._tickers = tickers or []
        self._config = config or {}
        self._wrapped_agent = None
        self._ticker_to_idx: dict[str, int] = {}

        if tickers:
            self._ticker_to_idx = {t: i for i, t in enumerate(tickers)}

        if agent_class_path:
            self._init_wrapped_agent()

    def _init_wrapped_agent(self) -> None:
        """Initialize the wrapped trading_agents agent."""
        try:
            if "alpha_momentum" in self._agent_class_path.lower():
                from trading_agents.agents.alpha_momentum import AlphaMomentum
                self._wrapped_agent = AlphaMomentum(
                    config=self._config,
                    data_providers=self._config.get("data_providers", {}),
                )
            elif "beta_mean_reversion" in self._agent_class_path.lower():
                from trading_agents.agents.beta_mean_reversion import BetaMeanReversion
                self._wrapped_agent = BetaMeanReversion(
                    config=self._config,
                    data_providers=self._config.get("data_providers", {}),
                )
            else:
                logger.warning(f"Unknown agent class: {self._agent_class_path}")
        except ImportError as e:
            logger.warning(f"Could not import wrapped agent: {e}")
        except Exception as e:
            logger.warning(f"Could not initialize wrapped agent: {e}")

    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Generate actions from the rule-based strategy.

        If the wrapped agent is available and tickers are set, it runs
        the analyze() method. Otherwise, falls back to a simple
        RSI-based heuristic extracted from the observation vector.
        """
        actions = np.zeros(self.action_dim, dtype=np.float32)

        if self._wrapped_agent is not None and self._tickers:
            return self._action_from_wrapped_agent()

        # Fallback: extract RSI from observation and use simple rules
        return self._action_from_observation(observation)

    def _action_from_wrapped_agent(self) -> np.ndarray:
        """Get actions by running the wrapped agent's analyze()."""
        actions = np.zeros(self.action_dim, dtype=np.float32)

        try:
            from trading_agents.agents.base_agent import MarketContext
            context = MarketContext()

            signals = self._wrapped_agent.analyze(self._tickers, context)
            for signal in signals:
                if signal.ticker in self._ticker_to_idx:
                    idx = self._ticker_to_idx[signal.ticker]
                    if signal.action == "BUY":
                        actions[idx] = np.float32(signal.conviction * signal.suggested_size_pct)
                    elif signal.action == "SELL":
                        actions[idx] = np.float32(-signal.conviction * signal.suggested_size_pct)
                    elif signal.action == "SCALE_OUT":
                        actions[idx] = np.float32(-0.5 * signal.conviction)
        except Exception as e:
            logger.debug(f"Wrapped agent analyze failed: {e}")

        return np.clip(actions, -1.0, 1.0)

    def _action_from_observation(self, obs: np.ndarray) -> np.ndarray:
        """Simple RSI-based heuristic from the observation vector.

        Observation layout: RSI is at indices [2N+1 : 3N+1] (normalized to [0,1]).
        """
        n = self.action_dim
        actions = np.zeros(n, dtype=np.float32)

        # RSI position in observation
        rsi_start = 2 * n + 1
        rsi_end = 3 * n + 1

        if rsi_end <= len(obs):
            rsi_values = obs[rsi_start:rsi_end]  # Normalized [0, 1], so 0.5 = RSI 50

            for i in range(n):
                rsi = rsi_values[i]
                if rsi < 0.30:
                    actions[i] = 0.3  # Oversold → buy
                elif rsi < 0.35:
                    actions[i] = 0.15
                elif rsi > 0.70:
                    actions[i] = -0.3  # Overbought → sell
                elif rsi > 0.65:
                    actions[i] = -0.15

        return actions

    def update(self, **kwargs: Any) -> dict[str, float]:
        """Rule-based agents do not update."""
        return {"skipped": 1.0}

    def save(self, path: str | Path) -> None:
        """Save agent metadata (rule-based agents have no learned params)."""
        from hydra.utils.serialization import save_json
        save_json({
            "name": self.name,
            "agent_class_path": self._agent_class_path,
            "tickers": self._tickers,
        }, path)

    def load(self, path: str | Path) -> None:
        """Load agent metadata."""
        from hydra.utils.serialization import load_json
        data = load_json(path)
        self._agent_class_path = data.get("agent_class_path")
        self._tickers = data.get("tickers", [])
        self._ticker_to_idx = {t: i for i, t in enumerate(self._tickers)}
        if self._agent_class_path:
            self._init_wrapped_agent()

    def set_tickers(self, tickers: list[str]) -> None:
        """Update the ticker list and index mapping."""
        self._tickers = tickers
        self._ticker_to_idx = {t: i for i, t in enumerate(tickers)}
