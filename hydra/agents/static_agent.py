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

    Can be created from a saved PPO/SAC/A2C/RecurrentPPO checkpoint.
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
        self._lstm_states = None  # For RecurrentPPO snapshots

        if checkpoint_path is not None:
            self.load(checkpoint_path)

    def on_episode_start(self) -> None:
        """Reset LSTM state for RecurrentPPO snapshots."""
        super().on_episode_start()
        self._lstm_states = None

    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using the frozen policy."""
        if self._model is None:
            # No model loaded — return zero action (hold everything)
            return self._fallback_action.copy()

        # RecurrentPPO models need LSTM state management via model.predict()
        if self._source_type == "recurrentppo":
            return self._select_action_recurrent(observation, deterministic)

        import torch

        obs_tensor = torch.as_tensor(observation.reshape(1, -1), dtype=torch.float32)
        obs_tensor = obs_tensor.to(self._model.device)

        with torch.no_grad():
            action = self._model.policy.predict(obs_tensor, deterministic=deterministic)

        if isinstance(action, tuple):
            action = action[0]

        action_np = action.cpu().numpy().flatten().astype(np.float32) if hasattr(action, 'cpu') else np.asarray(action, dtype=np.float32).flatten()
        return np.clip(action_np, -1.0, 1.0)

    def _select_action_recurrent(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action from a RecurrentPPO model with LSTM state tracking."""
        obs_np = observation.reshape(1, -1).astype(np.float32)
        episode_starts = np.array([self._lstm_states is None])

        action, self._lstm_states = self._model.predict(
            obs_np,
            state=self._lstm_states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )

        action_np = np.asarray(action, dtype=np.float32).flatten()
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
        """Load model from a checkpoint.

        Static agents always load on CPU. They are inference-only and
        DirectML's device comparison operators are broken for model loading
        (``'>=' not supported between 'torch.device' and 'int'``).
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"Checkpoint not found: {path}")
            return

        try:
            self._model = self._load_sb3_model(path, "cpu")
            logger.info(f"Loaded static agent '{self.name}' from {path}")
        except Exception as e:
            logger.error(f"Failed to load static agent '{self.name}': {e}")

    def _load_sb3_model(self, path: Path, device):
        """Load an SB3 model by source type."""
        if self._source_type == "ppo":
            from stable_baselines3 import PPO
            return PPO.load(str(path), device=device)
        elif self._source_type == "sac":
            from stable_baselines3 import SAC
            return SAC.load(str(path), device=device)
        elif self._source_type == "a2c":
            from stable_baselines3 import A2C
            return A2C.load(str(path), device=device)
        elif self._source_type == "recurrentppo":
            from sb3_contrib import RecurrentPPO
            return RecurrentPPO.load(str(path), device=device)
        else:
            logger.warning(f"Unknown source type '{self._source_type}'")
            return None

    @classmethod
    def from_agent(cls, agent: BaseRLAgent, name: str | None = None) -> StaticAgent:
        """Create a static snapshot from a learning agent.

        Two-step process:
        1. File-based save/load to get the correct SB3 model structure
           (hyperparams, network architecture, optimizer state).
        2. Direct state_dict copy with explicit CPU transfer to capture
           the *actual* trained weights from GPU memory.

        Step 2 is critical because DirectML's torch.save() doesn't properly
        sync device memory to CPU during serialization — without it, all
        snapshots would contain the initial untrained weights.
        """
        import os
        import tempfile

        snapshot_name = name or f"{agent.name}_static"
        static = cls(
            name=snapshot_name,
            obs_dim=agent.obs_dim,
            action_dim=agent.action_dim,
            source_type=agent.__class__.__name__.lower().replace("agent", ""),
        )

        src_model = getattr(agent, '_model', None)
        if src_model is None:
            logger.warning(f"Agent '{agent.name}' has no model to snapshot")
            return static

        # Step 1: File-based save/load for model structure
        fd, tmp_path = tempfile.mkstemp(suffix=".zip")
        os.close(fd)
        try:
            agent.save(tmp_path)
            static.load(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        # Step 2: Overwrite policy weights with live trained weights.
        # state_dict() + .cpu().clone() guarantees we get the actual GPU
        # weights, not stale serialized copies from DirectML.
        if static._model is not None:
            try:
                src_state = src_model.policy.state_dict()
                cpu_state = {k: v.detach().cpu().clone() for k, v in src_state.items()}
                static._model.policy.load_state_dict(cpu_state)
                logger.info(f"  Copied live weights to snapshot '{snapshot_name}'")
            except Exception as e:
                logger.warning(f"  Direct weight copy failed for '{snapshot_name}': {e}")

            _log_weight_checksum(snapshot_name, static._model)

        return static


def _log_weight_checksum(name: str, model) -> None:
    """Log a hash of model weights to verify snapshots have different parameters."""
    try:
        import hashlib
        params = model.policy.state_dict()
        h = hashlib.md5()
        for key in sorted(params.keys()):
            h.update(params[key].cpu().numpy().tobytes())
        logger.info(f"  Snapshot '{name}' weight checksum: {h.hexdigest()[:12]}")
    except Exception:
        pass
