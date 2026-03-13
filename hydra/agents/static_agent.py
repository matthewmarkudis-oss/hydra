"""Static (frozen) agent — inference only.

A snapshot of a previously trained RL agent. Raises on update() calls.
Used as part of the diverse agent pool to provide stable opponents.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from hydra.agents.base_rl_agent import BaseRLAgent

logger = logging.getLogger("hydra.agents.static")


class StaticAgent(BaseRLAgent):
    """Frozen agent that only performs inference.

    Can be created from a saved PPO/SAC/A2C checkpoint.
    """

    def __init__(
        self,
        name: str,
        obs_dim: int,
        action_dim: int,
        source_type: str = "ppo",
        checkpoint_path: str | Path | None = None,
    ):
        super().__init__(name, obs_dim, action_dim)
        self._frozen = True
        self._source_type = source_type
        self._model = None
        self._fallback_action = np.zeros(action_dim, dtype=np.float32)

        if checkpoint_path is not None:
            self.load(checkpoint_path)

    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using the frozen policy."""
        if self._model is None:
            # No model loaded — return zero action (hold everything)
            return self._fallback_action.copy()

        import torch

        obs_tensor = torch.as_tensor(observation.reshape(1, -1), dtype=torch.float32)
        obs_tensor = obs_tensor.to(self._model.device)

        with torch.no_grad():
            action = self._model.policy.predict(obs_tensor, deterministic=True)

        if isinstance(action, tuple):
            action = action[0]

        action_np = action.cpu().numpy().flatten().astype(np.float32) if hasattr(action, 'cpu') else np.asarray(action, dtype=np.float32).flatten()
        return np.clip(action_np, -1.0, 1.0)

    def update(self, **kwargs: Any) -> dict[str, float]:
        """Static agents do not update. Raises if called unintentionally."""
        raise RuntimeError(
            f"Cannot update static agent '{self.name}'. "
            "Static agents are frozen snapshots — inference only."
        )

    def freeze(self) -> None:
        """No-op — static agents are always frozen."""
        self._frozen = True

    def unfreeze(self) -> None:
        """Static agents cannot be unfrozen."""
        raise RuntimeError(f"Cannot unfreeze static agent '{self.name}'.")

    def save(self, path: str | Path) -> None:
        """Save the underlying model checkpoint."""
        if self._model is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._model.save(str(path))

    def load(self, path: str | Path) -> None:
        """Load model from a checkpoint."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Checkpoint not found: {path}")
            return

        try:
            if self._source_type == "ppo":
                from stable_baselines3 import PPO
                self._model = PPO.load(str(path), device="cpu")
            elif self._source_type == "sac":
                from stable_baselines3 import SAC
                self._model = SAC.load(str(path), device="cpu")
            elif self._source_type == "a2c":
                from stable_baselines3 import A2C
                self._model = A2C.load(str(path), device="cpu")
            else:
                logger.warning(f"Unknown source type '{self._source_type}'")

            logger.info(f"Loaded static agent '{self.name}' from {path}")
        except Exception as e:
            logger.error(f"Failed to load static agent '{self.name}': {e}")

    @classmethod
    def from_agent(cls, agent: BaseRLAgent, name: str | None = None) -> StaticAgent:
        """Create a static snapshot from a learning agent."""
        import tempfile

        snapshot_name = name or f"{agent.name}_static"
        static = cls(
            name=snapshot_name,
            obs_dim=agent.obs_dim,
            action_dim=agent.action_dim,
            source_type=agent.__class__.__name__.lower().replace("agent", ""),
        )

        # Save and reload to create independent copy
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            agent.save(tmp.name)
            static.load(tmp.name)

        return static
