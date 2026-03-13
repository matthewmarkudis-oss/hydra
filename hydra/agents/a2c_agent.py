"""A2C agent wrapper using stable-baselines3.

Advantage Actor-Critic with DirectML-aware device selection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from hydra.agents.base_rl_agent import BaseRLAgent
from hydra.agents.ppo_agent import _get_device, _make_dummy_env

logger = logging.getLogger("hydra.agents.a2c")


class A2CAgent(BaseRLAgent):
    """A2C agent using stable-baselines3."""

    def __init__(
        self,
        name: str,
        obs_dim: int,
        action_dim: int,
        learning_rate: float = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.01,
        vf_coef: float = 0.25,
        max_grad_norm: float = 0.5,
        prefer_gpu: bool = True,
    ):
        super().__init__(name, obs_dim, action_dim)
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self._device = _get_device(prefer_gpu)
        self._model = None

        logger.info(f"A2CAgent '{name}' using device: {self._device}")

    def _ensure_model(self) -> None:
        """Lazy-initialize the SB3 A2C model."""
        if self._model is not None:
            return

        try:
            import gymnasium as gym
            from stable_baselines3 import A2C

            obs_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32,
            )
            action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32,
            )

            device = self._device if self._device != "dml" else "cpu"
            dummy_env = _make_dummy_env(obs_space, action_space)

            self._model = A2C(
                "MlpPolicy",
                env=dummy_env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                ent_coef=self.ent_coef,
                vf_coef=self.vf_coef,
                max_grad_norm=self.max_grad_norm,
                device=device,
                verbose=0,
            )

        except ImportError as e:
            logger.error(f"stable-baselines3 not available: {e}")
            raise

    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using the A2C policy."""
        self._ensure_model()
        import torch

        obs_tensor = torch.as_tensor(observation.reshape(1, -1), dtype=torch.float32)
        obs_tensor = obs_tensor.to(self._model.device)

        with torch.no_grad():
            action = self._model.policy.predict(
                obs_tensor, deterministic=deterministic
            )

        if isinstance(action, tuple):
            action = action[0]

        action_np = action.cpu().numpy().flatten().astype(np.float32) if hasattr(action, 'cpu') else np.asarray(action, dtype=np.float32).flatten()
        return np.clip(action_np, -1.0, 1.0)

    def update(self, **kwargs: Any) -> dict[str, float]:
        """Update A2C policy."""
        if self._frozen:
            return {"skipped": 1.0}
        self._ensure_model()
        return {"updated": 1.0}

    def train_on_env(self, env, total_timesteps: int) -> dict[str, float]:
        """Train directly on a gymnasium environment."""
        self._ensure_model()
        if self._frozen:
            return {"skipped": 1.0}

        self._model.set_env(env)
        self._model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
        self._total_steps += total_timesteps

        return {"total_timesteps": float(self._total_steps)}

    def save(self, path: str | Path) -> None:
        """Save A2C model."""
        self._ensure_model()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path))

    def load(self, path: str | Path) -> None:
        """Load A2C model."""
        from stable_baselines3 import A2C

        device = self._device if self._device != "dml" else "cpu"
        self._model = A2C.load(str(path), device=device)
