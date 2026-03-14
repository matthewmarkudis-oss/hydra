"""Core training loop for multi-agent RL.

All agents observe → act → env steps → learning agents update.
Tracks per-agent statistics.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from hydra.agents.agent_pool import AgentPool
from hydra.envs.multi_agent_env import MultiAgentEnv
from hydra.envs.trading_env import TradingEnv
from hydra.training.metrics_tracker import MetricsTracker

logger = logging.getLogger("hydra.training.trainer")


class Trainer:
    """Training loop for the multi-agent system.

    Orchestrates the interaction between agents and the environment,
    collects experience, updates learning agents, and logs metrics.
    """

    def __init__(
        self,
        env: TradingEnv,
        pool: AgentPool,
        metrics: MetricsTracker | None = None,
        eval_interval: int = 10,
        checkpoint_interval: int = 50,
        checkpoint_dir: str = "checkpoints",
    ):
        self.multi_env = MultiAgentEnv(env, pool)
        self.pool = pool
        self.metrics = metrics or MetricsTracker()
        self.eval_interval = eval_interval
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir

        self._global_step = 0
        self._episode_count = 0

    def train_episodes(
        self,
        num_episodes: int,
        deterministic: bool = False,
    ) -> dict[str, Any]:
        """Train for a fixed number of episodes.

        Each episode:
        1. All agents observe the same state
        2. Each agent independently selects actions
        3. Actions are aggregated and environment steps
        4. Learning agents receive (obs, action, reward, next_obs, done)
        5. At episode end, learning agents update their policies

        Returns:
            Training summary with per-agent metrics.
        """
        episode_rewards = []
        episode_summaries = []

        for ep in range(num_episodes):
            reward, summary = self._train_one_episode(deterministic)
            episode_rewards.append(reward)
            episode_summaries.append(summary)
            self._episode_count += 1

            # Log metrics
            self.metrics.log_episode(
                episode=self._episode_count,
                reward=reward,
                info=summary,
            )

            # Periodic evaluation
            if self._episode_count % self.eval_interval == 0:
                eval_result = self.evaluate(num_episodes=3)
                self.metrics.log_eval(self._episode_count, eval_result)
                logger.info(
                    f"Episode {self._episode_count}: "
                    f"train_reward={reward:.4f}, "
                    f"eval_reward={eval_result['mean_reward']:.4f}"
                )

            # Periodic checkpoint
            if self._episode_count % self.checkpoint_interval == 0:
                self._save_checkpoint()

        total_trades = sum(s.get("num_trades", 0) for s in episode_summaries)

        return {
            "episodes": num_episodes,
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "total_steps": self._global_step,
            "total_trades": total_trades,
        }

    def _train_one_episode(self, deterministic: bool = False) -> tuple[float, dict]:
        """Run one training episode."""
        obs, info = self.multi_env.reset()
        total_reward = 0.0
        episode_transitions: dict[str, list[dict]] = {
            a.name: [] for a in self.pool.get_learning_agents()
        }

        while True:
            # Collect individual actions
            per_agent_actions = {}
            for agent in self.pool.get_all():
                per_agent_actions[agent.name] = agent.select_action(
                    obs, deterministic=deterministic
                )

            # Step environment
            next_obs, reward, terminated, truncated, step_info = self.multi_env.step(
                external_actions=per_agent_actions
            )

            done = terminated or truncated
            total_reward += reward
            self._global_step += 1

            # Store transitions for learning agents
            for agent in self.pool.get_learning_agents():
                episode_transitions[agent.name].append({
                    "obs": obs.copy(),
                    "action": per_agent_actions[agent.name].copy(),
                    "reward": reward,
                    "next_obs": next_obs.copy(),
                    "done": done,
                })

            if done:
                break

            obs = next_obs

        # Update learning agents
        update_metrics = {}
        for agent in self.pool.get_learning_agents():
            transitions = episode_transitions[agent.name]
            if hasattr(agent, "store_transition"):
                for t in transitions:
                    agent.store_transition(
                        t["obs"], t["action"], t["reward"], t["next_obs"], t["done"]
                    )
            metrics = agent.update()
            update_metrics[agent.name] = metrics

        summary = step_info.get("episode_summary", {})
        summary["update_metrics"] = update_metrics
        summary["num_trades"] = step_info.get("num_trades", 0)
        summary["total_transaction_costs"] = step_info.get("total_transaction_costs", 0.0)
        return total_reward, summary

    def evaluate(
        self,
        num_episodes: int = 5,
    ) -> dict[str, float]:
        """Evaluate current pool performance (deterministic actions)."""
        rewards = []

        for _ in range(num_episodes):
            reward, summary = self.multi_env.run_episode(deterministic=True)
            rewards.append(reward)

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "num_episodes": num_episodes,
        }

    def _save_checkpoint(self) -> None:
        """Save pool checkpoint."""
        from pathlib import Path

        ckpt_dir = Path(self.checkpoint_dir) / f"episode_{self._episode_count}"
        self.pool.save(ckpt_dir)
        logger.info(f"Saved checkpoint at episode {self._episode_count}")

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def episode_count(self) -> int:
        return self._episode_count
