"""TensorBoard-compatible metrics logging.

Tracks per-agent rewards, portfolio value, P&L, Sharpe ratio,
drawdown, win rate — organized into layman-friendly categories.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("hydra.training.metrics")


class MetricsTracker:
    """Tracks and logs training metrics to TensorBoard.

    Dashboard categories (layman-friendly):
      Performance/     — Portfolio value, P&L, total return
      Risk/            — Drawdown, volatility, Sharpe ratio
      Trading/         — Win rate, number of trades, transaction costs
      Learning/        — Episode reward, policy loss, value loss
      Agents/          — Per-agent comparison metrics
      Population/      — Generation-level evolution stats
    """

    def __init__(self, log_dir: str = "logs/tensorboard", use_tensorboard: bool = True):
        self._log_dir = Path(log_dir)
        self._writer = None
        self._use_tensorboard = use_tensorboard

        # In-memory tracking
        self._episode_rewards: list[float] = []
        self._eval_rewards: list[float] = []
        self._episode_data: list[dict] = []
        self._portfolio_values: list[float] = []
        self._winning_episodes = 0
        self._total_episodes = 0

        if use_tensorboard:
            self._init_tensorboard()

    def _init_tensorboard(self) -> None:
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(str(self._log_dir))
            logger.info(f"TensorBoard logging to {self._log_dir}")
        except ImportError:
            logger.warning("TensorBoard not available, using in-memory logging only")
            self._use_tensorboard = False

    def log_episode(self, episode: int, reward: float, info: dict[str, Any]) -> None:
        """Log metrics for a training episode."""
        self._episode_rewards.append(reward)
        self._episode_data.append({"episode": episode, "reward": reward, **info})
        self._total_episodes += 1

        # Extract episode summary from env
        summary = info.get("episode_summary", {}) if isinstance(info, dict) else {}

        # Track wins (positive return episodes)
        total_return = summary.get("total_return", 0.0)
        if total_return > 0:
            self._winning_episodes += 1

        final_value = summary.get("final_value", 0.0)
        if final_value > 0:
            self._portfolio_values.append(final_value)

        if not self._writer:
            return

        # === Performance (How is my money doing?) ===
        if final_value > 0:
            self._writer.add_scalar("Performance/Portfolio Value ($)", final_value, episode)
            pnl = final_value - 100_000.0  # assumes 100k starting
            self._writer.add_scalar("Performance/Profit or Loss ($)", pnl, episode)
        if total_return != 0:
            self._writer.add_scalar("Performance/Return (%)", total_return * 100, episode)

        # === Risk (How safe is my money?) ===
        mdd = summary.get("max_drawdown", 0.0)
        if mdd != 0:
            self._writer.add_scalar("Risk/Max Drawdown (%)", mdd * 100, episode)
        sharpe = summary.get("sharpe_ratio", 0.0)
        if sharpe != 0:
            self._writer.add_scalar("Risk/Sharpe Ratio", sharpe, episode)
        sortino = summary.get("sortino_ratio", 0.0)
        if sortino != 0:
            self._writer.add_scalar("Risk/Sortino Ratio", sortino, episode)
        vol = summary.get("volatility", 0.0)
        if vol != 0:
            self._writer.add_scalar("Risk/Volatility", vol, episode)

        # === Trading (What is the agent doing?) ===
        num_trades = info.get("num_trades", 0)
        if num_trades > 0:
            self._writer.add_scalar("Trading/Trades per Episode", num_trades, episode)
        tc = info.get("total_transaction_costs", 0.0)
        if tc > 0:
            self._writer.add_scalar("Trading/Transaction Costs ($)", tc, episode)
        if self._total_episodes > 0:
            win_rate = self._winning_episodes / self._total_episodes * 100
            self._writer.add_scalar("Trading/Win Rate (%)", win_rate, episode)

        # === Learning (Is the agent improving?) ===
        self._writer.add_scalar("Learning/Episode Reward", reward, episode)
        if len(self._episode_rewards) >= 10:
            avg = float(np.mean(self._episode_rewards[-10:]))
            self._writer.add_scalar("Learning/Reward (10-episode avg)", avg, episode)

        # Per-agent update metrics
        update_metrics = info.get("update_metrics", {})
        for agent_name, metrics in update_metrics.items():
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        self._writer.add_scalar(
                            f"Agents/{agent_name}/{k}", v, episode
                        )

    def log_eval(self, episode: int, eval_result: dict[str, float]) -> None:
        """Log evaluation metrics."""
        self._eval_rewards.append(eval_result.get("mean_reward", 0.0))

        if self._writer:
            mean_r = eval_result.get("mean_reward", 0.0)
            self._writer.add_scalar("Learning/Eval Reward (mean)", mean_r, episode)
            std_r = eval_result.get("std_reward", 0.0)
            self._writer.add_scalar("Learning/Eval Reward (std)", std_r, episode)

    def log_generation(self, generation: int, metrics: dict[str, Any]) -> None:
        """Log population-based training generation metrics."""
        if not self._writer:
            return

        train_mr = metrics.get("train_mean_reward", 0.0)
        self._writer.add_scalar("Population/Mean Reward", train_mr, generation)
        pool_size = metrics.get("pool_size", 0)
        self._writer.add_scalar("Population/Pool Size", pool_size, generation)
        best = metrics.get("best_eval_score", 0.0)
        self._writer.add_scalar("Population/Best Agent Score", best, generation)

    def log_agent_eval(
        self, generation: int, agent_name: str, score: float, extra: dict[str, float] | None = None,
    ) -> None:
        """Log per-agent evaluation results at generation level."""
        if not self._writer:
            return
        self._writer.add_scalar(f"Agents/{agent_name}/eval_score", score, generation)
        if extra:
            for k, v in extra.items():
                if isinstance(v, (int, float)):
                    self._writer.add_scalar(f"Agents/{agent_name}/{k}", v, generation)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log an arbitrary scalar."""
        if self._writer:
            self._writer.add_scalar(tag, value, step)

    def get_recent_reward(self, window: int = 100) -> float:
        """Get mean reward over the last `window` episodes."""
        if not self._episode_rewards:
            return 0.0
        recent = self._episode_rewards[-window:]
        return float(np.mean(recent))

    def get_summary(self) -> dict[str, float]:
        """Get overall training summary."""
        rewards = np.array(self._episode_rewards) if self._episode_rewards else np.array([0.0])
        return {
            "total_episodes": len(self._episode_rewards),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "best_reward": float(np.max(rewards)),
            "worst_reward": float(np.min(rewards)),
            "recent_mean_reward": self.get_recent_reward(100),
            "win_rate": self._winning_episodes / max(self._total_episodes, 1) * 100,
        }

    def close(self) -> None:
        """Flush and close the TensorBoard writer."""
        if self._writer:
            self._writer.flush()
            self._writer.close()
