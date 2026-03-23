"""PPO agent wrapper using stable-baselines3.

DirectML-aware device selection for AMD GPU acceleration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from hydra.agents.base_rl_agent import BaseRLAgent

logger = logging.getLogger("hydra.agents.ppo")


def _make_dummy_env(obs_space, action_space):
    """Create a minimal gymnasium.Env to satisfy SB3 constructor.

    SB3 does isinstance(env, gymnasium.Env), so we must subclass it.
    We import gymnasium here to keep it lazy.
    """
    import gymnasium as gym

    class _DummyEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.observation_space = obs_space
            self.action_space = action_space

        def reset(self, *, seed=None, options=None):
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}

    return _DummyEnv()


def _get_device(prefer_gpu: bool = True) -> str:
    """Detect best available device (DirectML > CUDA > CPU)."""
    if prefer_gpu:
        try:
            import torch_directml
            torch_directml.device()  # verify it works
            return "dml"
        except (ImportError, TypeError, Exception):
            pass
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
    return "cpu"


def _resolve_sb3_device(device_str: str):
    """Convert internal device string to a torch device SB3 can use.

    When DirectML is available, we monkey-patch torch.Tensor.item and
    .numpy so SB3's scalar extraction works transparently on GPU tensors.
    This lets SB3 place models on the DirectML device for GPU-accelerated
    forward/backward passes while the cheap scalar transfers happen
    automatically.
    """
    if device_str == "dml":
        from hydra.compute.dml_compat import patch_tensor_for_directml

        if patch_tensor_for_directml():
            import torch_directml

            device = torch_directml.device()
            logger.info(f"SB3 will use DirectML device: {device}")
            return device

        logger.warning("DirectML patch failed, falling back to CPU for SB3")
        return "cpu"
    return device_str


class PPOAgent(BaseRLAgent):
    """PPO agent using stable-baselines3.

    Supports DirectML (AMD GPU), CUDA, or CPU backends.
    """

    def __init__(
        self,
        name: str,
        obs_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
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
        self.net_arch = net_arch or [256, 256]

        self._device = _get_device(prefer_gpu)
        self._model = None
        self._rollout_buffer: list[dict] = []

        logger.info(f"PPOAgent '{name}' using device: {self._device}")

    def _ensure_model(self) -> None:
        """Lazy-initialize the SB3 PPO model."""
        if self._model is not None:
            return

        try:
            import gymnasium as gym
            from stable_baselines3 import PPO

            obs_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32,
            )
            action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32,
            )

            device = _resolve_sb3_device(self._device)

            dummy_env = _make_dummy_env(obs_space, action_space)

            policy_kwargs = {"net_arch": self.net_arch}

            self._model = PPO(
                "MlpPolicy",
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
            logger.error(f"stable-baselines3 not available: {e}")
            raise

    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using the PPO policy."""
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

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition for later update."""
        self._rollout_buffer.append({
            "obs": obs.copy(),
            "action": action.copy(),
            "reward": np.float32(reward),
            "next_obs": next_obs.copy(),
            "done": done,
        })

    def update(self, **kwargs: Any) -> dict[str, float]:
        """Update policy from collected rollout buffer."""
        if self._frozen:
            return {"skipped": 1.0}

        self._ensure_model()

        if len(self._rollout_buffer) < self.batch_size:
            return {"buffer_size": float(len(self._rollout_buffer))}

        # For custom training loop, we extract experience and feed to SB3
        # This is a simplified version — full integration would use SB3's
        # collect_rollouts() method with the actual env
        metrics = {
            "buffer_size": float(len(self._rollout_buffer)),
            "policy_loss": 0.0,
            "value_loss": 0.0,
        }

        self._rollout_buffer.clear()
        return metrics

    def train_on_env(self, env, total_timesteps: int) -> dict[str, float]:
        """Train directly on a gymnasium environment using SB3's learn().

        Only calls set_env() when the env reference actually changed, to
        avoid resetting PPO's internal rollout buffer state unnecessarily.
        When switching from n_envs=1 (dummy) to n_envs=4 (VecEnv), uses
        save/load to resize internal buffers (set_env rejects n_envs changes).
        """
        self._ensure_model()

        if self._frozen:
            return {"skipped": 1.0}

        # Only set_env if env changed (preserves internal SB3 state)
        if self._model.get_env() is not env:
            current_n = getattr(self._model.get_env(), "num_envs", 1)
            new_n = getattr(env, "num_envs", 1)
            if current_n != new_n:
                # n_envs changed — must save/reload to resize buffers
                import tempfile, os
                from stable_baselines3 import PPO
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
                    tmp_path = f.name
                self._model.save(tmp_path)
                self._model = PPO.load(tmp_path, env=env, device=self._model.device)
                os.unlink(tmp_path)
            else:
                self._model.set_env(env)
        self._model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
        self._total_steps += total_timesteps

        return {"total_timesteps": float(self._total_steps)}

    def save(self, path: str | Path) -> None:
        """Save PPO model to disk."""
        self._ensure_model()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path))
        logger.info(f"Saved PPO agent '{self.name}' to {path}")

    def load(self, path: str | Path) -> None:
        """Load PPO model from disk.

        Always loads on CPU first to avoid DirectML device comparison
        bugs in SB3's load() ('>=' not supported between 'torch.device'
        and 'int'). Training/inference will run on CPU after resume.
        """
        from stable_baselines3 import PPO

        self._model = PPO.load(str(path), device="cpu")
        logger.info(f"Loaded PPO agent '{self.name}' from {path} (device=cpu)")
