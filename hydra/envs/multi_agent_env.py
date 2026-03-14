"""Multi-agent shared-portfolio trading environment.

Wraps TradingEnv to support multiple agents sharing a single portfolio.
Each agent observes the same state and independently selects actions.
Actions are aggregated via weighted sum. The shared reward signal
(differential Sharpe of combined portfolio) incentivizes cooperation.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from hydra.agents.agent_pool import AgentPool
from hydra.envs.trading_env import TradingEnv
from hydra.utils.numpy_opts import SharedMarketData


class MultiAgentEnv:
    """Multi-agent wrapper over TradingEnv.

    All agents share one portfolio. Actions are aggregated before
    being passed to the underlying TradingEnv. Each agent receives
    the same observation and the same shared reward.

    This is NOT a gymnasium.Env subclass — it has its own step/reset
    interface that returns per-agent data. The underlying TradingEnv
    handles all the actual market simulation.
    """

    def __init__(
        self,
        env: TradingEnv,
        pool: AgentPool,
        exploration_noise: float = 0.1,
    ):
        self.env = env
        self.pool = pool
        self.exploration_noise = np.float32(exploration_noise)
        self._last_obs: np.ndarray | None = None
        self._episode_count = 0
        self._rng = np.random.default_rng()

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self.env.observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return self.env.action_space

    @property
    def num_agents(self) -> int:
        return self.pool.size

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment. All agents see the same initial observation."""
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self._episode_count += 1

        # Notify all agents of episode start
        for agent in self.pool.get_all():
            agent.on_episode_start()

        info["num_agents"] = self.num_agents
        info["agent_names"] = self.pool.agent_names
        return obs, info

    def step(
        self,
        external_actions: dict[str, np.ndarray] | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one multi-agent step.

        Each agent independently selects an action from the shared observation.
        Actions are aggregated via weighted mean and passed to the underlying env.

        Args:
            external_actions: Optional pre-computed actions per agent.
                If None, agents select actions from the current observation.
            deterministic: If True, disable exploration noise in aggregation.

        Returns:
            Tuple of (observation, shared_reward, terminated, truncated, info).
            Info contains per-agent action breakdown.
        """
        obs = self._last_obs

        # Collect actions from all agents
        if external_actions is not None:
            per_agent_actions = external_actions
        else:
            per_agent_actions = self.pool.collect_actions(obs)

        # Aggregate actions
        aggregated = self._aggregate_actions(per_agent_actions, deterministic=deterministic)

        # Step the underlying environment
        next_obs, reward, terminated, truncated, info = self.env.step(aggregated)
        self._last_obs = next_obs

        # Notify agents
        for agent in self.pool.get_all():
            agent.on_step()

        # Per-agent info
        info["per_agent_actions"] = {
            name: action.copy() for name, action in per_agent_actions.items()
        }
        info["aggregated_action"] = aggregated.copy()
        info["agent_weights"] = self.pool.get_weights().copy()

        return next_obs, reward, terminated, truncated, info

    def _aggregate_actions(
        self,
        per_agent_actions: dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> np.ndarray:
        """Aggregate per-agent actions into a single action via weighted mean.

        Includes magnitude-preserving rescaling with a dynamic cap based on
        pool size, and exploration noise during training to ensure non-zero
        actions even when untrained networks output near-zero values.
        """
        if not per_agent_actions:
            return np.zeros(self.env.action_space.shape, dtype=np.float32)

        names = list(per_agent_actions.keys())
        weights = self.pool.get_weights()
        n_agents = len(names)

        action_matrix = np.stack([per_agent_actions[n] for n in names])
        aggregated = np.average(action_matrix, axis=0, weights=weights)

        # Magnitude-preserving rescaling — cap scales with pool size
        max_abs_per_stock = np.max(np.abs(action_matrix), axis=0)
        agg_abs = np.abs(aggregated)
        max_scale = np.float32(max(n_agents, 3))
        meaningful = max_abs_per_stock > 0.01
        if np.any(meaningful):
            scale = np.ones_like(aggregated)
            safe_agg = np.where(agg_abs > 1e-8, agg_abs, np.float32(1.0))
            scale[meaningful] = np.minimum(
                max_abs_per_stock[meaningful] / safe_agg[meaningful],
                max_scale,
            )
            aggregated = aggregated * scale

        # Exploration noise during training — ensures non-zero actions
        if not deterministic and self.exploration_noise > 0:
            noise = self._rng.normal(
                0, float(self.exploration_noise), size=aggregated.shape,
            ).astype(np.float32)
            aggregated = aggregated + noise

        return np.clip(aggregated, -1.0, 1.0).astype(np.float32)

    def collect_experience(
        self,
        num_steps: int,
        deterministic: bool = False,
    ) -> dict[str, list[dict]]:
        """Collect experience tuples for all learning agents.

        Runs num_steps in the environment and returns per-agent
        (obs, action, reward, next_obs, done) tuples.

        Returns:
            Dict of agent_name → list of transition dicts.
        """
        experience: dict[str, list[dict]] = {
            a.name: [] for a in self.pool.get_learning_agents()
        }

        obs = self._last_obs
        if obs is None:
            obs, _ = self.reset()

        for _ in range(num_steps):
            # Each agent selects independently
            per_agent_actions = {}
            for agent in self.pool.get_all():
                per_agent_actions[agent.name] = agent.select_action(
                    obs, deterministic=deterministic
                )

            next_obs, reward, terminated, truncated, info = self.step(
                external_actions=per_agent_actions
            )
            done = terminated or truncated

            # Store experience for learning agents
            for agent in self.pool.get_learning_agents():
                experience[agent.name].append({
                    "obs": obs.copy(),
                    "action": per_agent_actions[agent.name].copy(),
                    "reward": reward,
                    "next_obs": next_obs.copy(),
                    "done": done,
                })

            if done:
                obs, _ = self.reset()
            else:
                obs = next_obs

        return experience

    def run_episode(
        self, deterministic: bool = False
    ) -> tuple[float, dict[str, Any]]:
        """Run a full episode and return total reward and summary.

        Returns:
            Tuple of (total_reward, episode_info).
        """
        obs, info = self.reset()
        total_reward = 0.0
        step_count = 0

        while True:
            next_obs, reward, terminated, truncated, step_info = self.step(
                deterministic=deterministic,
            )
            total_reward += reward
            step_count += 1

            if terminated or truncated:
                break

        return total_reward, {
            "total_reward": total_reward,
            "steps": step_count,
            "episode_summary": step_info.get("episode_summary", {}),
        }
