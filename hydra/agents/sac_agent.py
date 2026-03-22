"""SAC agent wrapper using stable-baselines3.

Soft Actor-Critic with automatic entropy tuning.
DirectML-aware device selection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from hydra.agents.base_rl_agent import BaseRLAgent
from hydra.agents.ppo_agent import _get_device, _make_dummy_env, _resolve_sb3_device

logger = logging.getLogger("hydra.agents.sac")


class SACAgent(BaseRLAgent):
    """SAC agent using stable-baselines3."""

    def __init__(
        self,
        name: str,
        obs_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        buffer_size: int = 100_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        ent_coef: str = "auto",
        learning_starts: int = 100,
        net_arch: list[int] | None = None,
        prefer_gpu: bool = True,
    ):
        super().__init__(name, obs_dim, action_dim)
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.ent_coef = ent_coef
        self.learning_starts = learning_starts
        self.net_arch = net_arch or [256, 256]

        self._device = _get_device(prefer_gpu)
        self._model = None

        logger.info(f"SACAgent '{name}' using device: {self._device}")

    def _ensure_model(self) -> None:
        """Lazy-initialize the SB3 SAC model."""
        if self._model is not None:
            return

        try:
            import gymnasium as gym
            from stable_baselines3 import SAC

            obs_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32,
            )
            action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32,
            )

            device = _resolve_sb3_device(self._device)
            dummy_env = _make_dummy_env(obs_space, action_space)

            policy_kwargs = {"net_arch": self.net_arch}

            self._model = SAC(
                "MlpPolicy",
                env=dummy_env,
                learning_rate=self.learning_rate,
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                gamma=self.gamma,
                tau=self.tau,
                ent_coef=self.ent_coef,
                learning_starts=self.learning_starts,
                policy_kwargs=policy_kwargs,
                device=device,
                verbose=0,
            )

        except ImportError as e:
            logger.error(f"stable-baselines3 not available: {e}")
            raise

    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using the SAC policy."""
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
        """Update SAC policy."""
        if self._frozen:
            return {"skipped": 1.0}
        self._ensure_model()
        return {"updated": 1.0}

    def train_on_env(self, env, total_timesteps: int) -> dict[str, float]:
        """Train directly on a gymnasium environment.

        Only calls set_env() when the env reference actually changed, to
        preserve the replay buffer across generations.  When switching from
        n_envs=1 (dummy) to n_envs=4 (VecEnv), uses save/load to resize
        internal buffers (set_env rejects n_envs changes).
        """
        self._ensure_model()
        if self._frozen:
            return {"skipped": 1.0}

        # Only set_env if env changed (preserves replay buffer)
        if self._model.get_env() is not env:
            current_n = getattr(self._model.get_env(), "num_envs", 1)
            new_n = getattr(env, "num_envs", 1)
            if current_n != new_n:
                import tempfile, os
                from stable_baselines3 import SAC
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
                    tmp_path = f.name
                self._model.save(tmp_path)
                self._model = SAC.load(tmp_path, env=env, device=self._model.device)
                os.unlink(tmp_path)
            else:
                self._model.set_env(env)
        self._model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
        self._total_steps += total_timesteps

        return {"total_timesteps": float(self._total_steps)}

    def save(self, path: str | Path) -> None:
        """Save SAC model."""
        self._ensure_model()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path))

    def load(self, path: str | Path) -> None:
        """Load SAC model."""
        from stable_baselines3 import SAC

        device = _resolve_sb3_device(self._device)
        self._model = SAC.load(str(path), device=device)
