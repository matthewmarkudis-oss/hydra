"""Abstract base class for all RL agents in the Hydra system.

Defines the common interface for learning agents, static agents,
and rule-based adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseRLAgent(ABC):
    """Abstract RL agent interface.

    All agents in the pool implement this interface, whether they are
    learning (PPO/SAC/A2C), frozen (static), or rule-based adapters.
    """

    def __init__(self, name: str, obs_dim: int, action_dim: int):
        self.name = name
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._frozen = False
        self._total_steps = 0
        self._episode_count = 0

    @abstractmethod
    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select an action given an observation.

        Args:
            observation: Float32 array of shape (obs_dim,).
            deterministic: If True, use greedy/mean action (no exploration noise).

        Returns:
            Float32 array of shape (action_dim,), values in [-1, 1].
        """

    @abstractmethod
    def update(self, **kwargs: Any) -> dict[str, float]:
        """Update the agent's policy from collected experience.

        Returns:
            Dict of training metrics (loss, entropy, etc.).
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save agent state to disk."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load agent state from disk."""

    def freeze(self) -> None:
        """Freeze the agent (no more updates). Creates a static snapshot."""
        self._frozen = True

    def unfreeze(self) -> None:
        """Unfreeze the agent (allow updates again)."""
        self._frozen = False

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    @property
    def is_learning(self) -> bool:
        """True if this agent actively updates its policy."""
        return not self._frozen

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def episode_count(self) -> int:
        return self._episode_count

    def on_episode_start(self) -> None:
        """Called at the beginning of each episode."""
        self._episode_count += 1

    def on_step(self) -> None:
        """Called after each environment step."""
        self._total_steps += 1

    def get_info(self) -> dict[str, Any]:
        """Return agent metadata for logging."""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "frozen": self._frozen,
            "total_steps": self._total_steps,
            "episode_count": self._episode_count,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }
