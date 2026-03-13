"""TensorBoard-compatible metrics logging.

Tracks per-agent rewards, losses, action distributions, and
episode-level performance metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("hydra.training.metrics")


class MetricsTracker:
    """Tracks and logs training metrics."""

    def __init__(self, log_dir: str = "logs/tensorboard", use_tensorboard: bool = True):
        self._log_dir = Path(log_dir)
        self._writer = None
        self._use_tensorboard = use_tensorboard

        # In-memory tracking
        self._episode_rewards: list[float] = []
        self._eval_rewards: list[float] = []
        self._episode_data: list[dict] = []

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

        if self._writer:
            self._writer.add_scalar("train/episode_reward", reward, episode)

            # Log episode summary if available
            summary = info.get("episode_summary", {}) if isinstance(info, dict) else {}
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    self._writer.add_scalar(f"train/{key}", value, episode)

            # Log per-agent update metrics
            update_metrics = info.get("update_metrics", {})
            for agent_name, metrics in update_metrics.items():
                if isinstance(metrics, dict):
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            self._writer.add_scalar(
                                f"agent/{agent_name}/{k}", v, episode
                            )

    def log_eval(self, episode: int, eval_result: dict[str, float]) -> None:
        """Log evaluation metrics."""
        self._eval_rewards.append(eval_result.get("mean_reward", 0.0))

        if self._writer:
            for key, value in eval_result.items():
                if isinstance(value, (int, float)):
                    self._writer.add_scalar(f"eval/{key}", value, episode)

    def log_generation(self, generation: int, metrics: dict[str, Any]) -> None:
        """Log population-based training generation metrics."""
        if self._writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._writer.add_scalar(f"generation/{key}", value, generation)

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
        }

    def close(self) -> None:
        """Flush and close the TensorBoard writer."""
        if self._writer:
            self._writer.flush()
            self._writer.close()
