"""CMA-ES evolutionary agent — gradient-free policy optimization.

Uses Covariance Matrix Adaptation Evolution Strategy to optimize a
linear policy by evaluating complete episode rollouts. Immune to
reward non-stationarity because it never uses a replay buffer or
per-step gradients — it scores entire trajectories under the current
reward function.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from hydra.agents.base_rl_agent import BaseRLAgent

logger = logging.getLogger("hydra.agents.cmaes")


class CMAESAgent(BaseRLAgent):
    """Evolutionary agent using CMA-ES for gradient-free optimization.

    Policy: linear  tanh(obs @ W + b) → actions in [-1, 1].
    Training: CMA-ES samples candidate weight vectors, evaluates each
    as a complete episode rollout, and updates the search distribution.
    """

    def __init__(
        self,
        name: str,
        obs_dim: int,
        action_dim: int,
        population_size: int = 16,
        sigma0: float = 0.5,
        max_evals_per_gen: int = 64,
        prefer_gpu: bool = False,  # CMA-ES is CPU-only
    ):
        super().__init__(name, obs_dim, action_dim)
        self.population_size = population_size
        self.sigma0 = sigma0
        self.max_evals_per_gen = max_evals_per_gen

        # Linear policy: W (obs_dim x action_dim) + b (action_dim)
        self._num_params = obs_dim * action_dim + action_dim
        self._weights = np.zeros(self._num_params, dtype=np.float64)
        self._es = None  # Lazy-initialized CMA-ES optimizer

        logger.info(
            f"CMAESAgent '{name}': {self._num_params} params "
            f"(policy: {obs_dim}x{action_dim}+{action_dim}), "
            f"pop_size={population_size}, sigma0={sigma0}"
        )

    def _decode_weights(self, flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decode flat weight vector into W matrix and b bias."""
        w_size = self.obs_dim * self.action_dim
        W = flat[:w_size].reshape(self.obs_dim, self.action_dim)
        b = flat[w_size:]
        return W.astype(np.float32), b.astype(np.float32)

    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Linear policy: tanh(obs @ W + b)."""
        W, b = self._decode_weights(self._weights)
        raw = observation.astype(np.float32) @ W + b
        return np.clip(np.tanh(raw), -1.0, 1.0).astype(np.float32)

    def _evaluate_candidate(self, flat_weights: np.ndarray, env) -> float:
        """Run one episode with candidate weights and return total reward."""
        W, b = self._decode_weights(flat_weights)

        # Handle both VecEnv and regular Env
        is_vec = hasattr(env, "num_envs")
        if is_vec:
            n_envs = env.num_envs
            # VecEnv — reset returns (obs_array, info_dict) tuple or just obs
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs = reset_result[0]  # Extract observations, not info
            else:
                obs = reset_result
            if obs.ndim > 1:
                obs = obs[0]  # Use first sub-env only
        else:
            obs, _ = env.reset()

        total_reward = 0.0
        max_steps = 2000  # Safety limit

        for _ in range(max_steps):
            raw = obs.astype(np.float32) @ W + b
            action = np.clip(np.tanh(raw), -1.0, 1.0).astype(np.float32)

            if is_vec:
                # VecEnv expects (n_envs, action_dim) — tile same action
                vec_action = np.tile(action.reshape(1, -1), (n_envs, 1))
                obs_all, rewards, dones, infos = env.step(vec_action)
                obs = obs_all[0]  # Track first sub-env
                total_reward += float(rewards[0])
                if dones[0]:
                    break
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break

        return total_reward

    def train_on_env(self, env, total_timesteps: int) -> dict[str, float]:
        """Evolutionary training: sample candidates, evaluate episodes, update.

        Unlike gradient-based agents, CMA-ES evaluates complete trajectories.
        The total_timesteps param is used to determine how many candidate
        evaluations to run (each eval is one full episode).
        """
        if self._frozen:
            return {"skipped": 1.0}

        try:
            import cma
        except ImportError:
            logger.warning("pycma not installed — skipping CMA-ES training")
            return {"skipped": 1.0, "reason": "pycma_missing"}

        # Initialize or continue the CMA-ES optimizer
        if self._es is None:
            opts = {
                "popsize": self.population_size,
                "maxiter": 0,  # We control iteration manually
                "verbose": -9,  # Suppress CMA output
                "seed": 42,
                "bounds": [-3.0, 3.0],  # Keep weights bounded
            }
            self._es = cma.CMAEvolutionStrategy(
                self._weights.tolist(), self.sigma0, opts
            )

        # Run evolutionary iterations
        num_evals = min(self.max_evals_per_gen, self.population_size * 4)
        iterations = max(1, num_evals // self.population_size)
        best_fitness = -float("inf")

        for _ in range(iterations):
            candidates = self._es.ask()
            fitnesses = []

            for candidate in candidates:
                candidate_arr = np.array(candidate, dtype=np.float64)
                reward = self._evaluate_candidate(candidate_arr, env)
                # CMA-ES minimizes, so negate reward
                fitnesses.append(-reward)

                if reward > best_fitness:
                    best_fitness = reward

            self._es.tell(candidates, fitnesses)

        # Update weights to the current best
        self._weights = np.array(self._es.result.xbest, dtype=np.float64)
        self._total_steps += num_evals

        logger.info(
            f"  CMA-ES '{self.name}': {num_evals} evals, "
            f"best_fitness={best_fitness:.3f}, "
            f"sigma={self._es.sigma:.4f}"
        )

        return {
            "total_timesteps": float(self._total_steps),
            "best_fitness": best_fitness,
            "sigma": self._es.sigma,
            "evals": float(num_evals),
        }

    def update(self, **kwargs: Any) -> dict[str, float]:
        """CMA-ES updates happen in train_on_env, not here."""
        if self._frozen:
            return {"skipped": 1.0}
        return {"updated": 1.0}

    def save(self, path: str | Path) -> None:
        """Save CMA-ES state: weights + optimizer params."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "name": self.name,
            "weights": self._weights,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "population_size": self.population_size,
            "sigma0": self.sigma0,
        }

        # Save CMA-ES optimizer state if available
        if self._es is not None:
            state["es_mean"] = np.array(self._es.mean)
            state["es_sigma"] = self._es.sigma

        np.savez(str(path), **state)
        logger.info(f"Saved CMA-ES agent '{self.name}' to {path}")

    def load(self, path: str | Path) -> None:
        """Load CMA-ES state from disk."""
        path = Path(path)
        if not str(path).endswith(".npz"):
            path = Path(str(path) + ".npz")

        data = np.load(str(path), allow_pickle=True)
        self._weights = data["weights"].astype(np.float64)

        if "es_mean" in data and "es_sigma" in data:
            try:
                import cma
                opts = {
                    "popsize": self.population_size,
                    "maxiter": 0,
                    "verbose": -9,
                    "seed": 42,
                    "bounds": [-3.0, 3.0],
                }
                self._es = cma.CMAEvolutionStrategy(
                    data["es_mean"].tolist(),
                    float(data["es_sigma"]),
                    opts,
                )
            except ImportError:
                logger.warning("pycma not installed — loaded weights only")

        logger.info(f"Loaded CMA-ES agent '{self.name}' from {path}")
