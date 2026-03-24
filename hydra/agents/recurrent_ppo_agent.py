"""RecurrentPPO agent wrapper using sb3-contrib.

LSTM-based PPO that can learn temporal patterns across bars.
DirectML-aware device selection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from hydra.agents.base_rl_agent import BaseRLAgent
from hydra.agents.ppo_agent import _get_device, _make_dummy_env, _resolve_sb3_device

logger = logging.getLogger("hydra.agents.recurrent_ppo")


class RecurrentPPOAgent(BaseRLAgent):
    """RecurrentPPO agent using sb3-contrib.

    Uses an LSTM policy network that can learn temporal dependencies
    across time steps — e.g., trend patterns, mean-reversion timing,
    and regime shifts that MLP policies cannot capture.
    """

    def __init__(
        self,
        name: str,
        obs_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 512,
        n_epochs: int = 10,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        lstm_hidden_size: int = 128,
        n_lstm_layers: int = 1,
        net_arch: list[int] | None = None,
        prefer_gpu: bool = True,
    ):
        super().__init__(name, obs_dim, action_dim)
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.net_arch = net_arch or [512, 256]

        self._device = _get_device(prefer_gpu)
        self._model = None
        self._lstm_states = None

        logger.info(f"RecurrentPPOAgent '{name}' using device: {self._device}")

    def _ensure_model(self) -> None:
        """Lazy-initialize the sb3-contrib RecurrentPPO model."""
        if self._model is not None:
            return

        try:
            import gymnasium as gym
            from sb3_contrib import RecurrentPPO

            obs_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32,
            )
            action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32,
            )

            device = _resolve_sb3_device(self._device)
            dummy_env = _make_dummy_env(obs_space, action_space)

            policy_kwargs = {
                "lstm_hidden_size": self.lstm_hidden_size,
                "n_lstm_layers": self.n_lstm_layers,
                "net_arch": self.net_arch,
            }

            self._model = RecurrentPPO(
                "MlpLstmPolicy",
                env=dummy_env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
                ent_coef=self.ent_coef,
                vf_coef=self.vf_coef,
                max_grad_norm=self.max_grad_norm,
                policy_kwargs=policy_kwargs,
                device=device,
                verbose=0,
            )

        except ImportError as e:
            logger.error(f"sb3-contrib not available: {e}")
            raise

    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using the LSTM policy.

        sb3-contrib RecurrentPPO.predict() expects:
        - numpy observation (NOT torch tensor)
        - episode_starts: boolean array indicating new episodes
        - state: LSTM hidden state tuple from previous call
        """
        self._ensure_model()

        obs_np = observation.reshape(1, -1).astype(np.float32)

        # episode_starts signals whether the LSTM should reset.
        # True on the very first call (no prior state), False otherwise.
        episode_starts = np.array([self._lstm_states is None])

        action, self._lstm_states = self._model.predict(
            obs_np,
            state=self._lstm_states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )

        action_np = np.asarray(action, dtype=np.float32).flatten()
        return np.clip(action_np, -1.0, 1.0)

    def on_episode_start(self) -> None:
        """Reset LSTM hidden state at the start of each episode."""
        super().on_episode_start()
        self._lstm_states = None

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition. LSTM state is managed internally."""
        if done:
            self._lstm_states = None

    def update(self, **kwargs: Any) -> dict[str, float]:
        """Update policy."""
        if self._frozen:
            return {"skipped": 1.0}
        self._ensure_model()
        return {"updated": 1.0}

    def train_on_env(self, env, total_timesteps: int) -> dict[str, float]:
        """Train directly on a gymnasium environment using sb3-contrib's learn().

        When switching from n_envs=1 (dummy) to n_envs=4 (VecEnv), uses
        save/load to resize internal buffers (set_env rejects n_envs changes).
        """
        self._ensure_model()

        if self._frozen:
            return {"skipped": 1.0}

        if self._model.get_env() is not env:
            current_n = getattr(self._model.get_env(), "num_envs", 1)
            new_n = getattr(env, "num_envs", 1)
            if current_n != new_n:
                import tempfile, os
                from sb3_contrib import RecurrentPPO
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
                    tmp_path = f.name
                self._model.save(tmp_path)
                self._model = RecurrentPPO.load(tmp_path, env=env, device=self._model.device)
                os.unlink(tmp_path)
            else:
                self._model.set_env(env)
        self._model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
        self._total_steps += total_timesteps
        self._lstm_states = None  # Reset LSTM state after training

        return {"total_timesteps": float(self._total_steps)}

    def save(self, path: str | Path) -> None:
        """Save RecurrentPPO model to disk."""
        self._ensure_model()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path))
        logger.info(f"Saved RecurrentPPO agent '{self.name}' to {path}")

    def load(self, path: str | Path) -> None:
        """Load RecurrentPPO model from disk.

        Always loads on CPU first to avoid DirectML device comparison
        bugs in SB3's load().
        """
        from sb3_contrib import RecurrentPPO

        self._model = RecurrentPPO.load(str(path), device="cpu")
        self._lstm_states = None
        logger.info(f"Loaded RecurrentPPO agent '{self.name}' from {path} (device=cpu)")
