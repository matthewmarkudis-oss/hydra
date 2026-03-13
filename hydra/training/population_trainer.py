"""Population-based training with generational evolution.

Trains agents in generations: each generation runs a fixed number of episodes,
then the best agents are promoted (frozen as static opponents) and the worst
are demoted (removed from pool). New agents are added according to the
curriculum schedule.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from hydra.agents.agent_pool import AgentPool
from hydra.envs.trading_env import TradingEnv
from hydra.training.curriculum import Curriculum
from hydra.training.metrics_tracker import MetricsTracker
from hydra.training.trainer import Trainer

logger = logging.getLogger("hydra.training.population")


class PopulationTrainer:
    """Generational population-based training.

    Each generation:
    1. Train all learning agents for N episodes
    2. Evaluate all agents
    3. Rank by evaluation performance
    4. Promote top-K learning agents → static snapshots
    5. Demote bottom-K static agents (remove from pool)
    6. Apply curriculum adjustments to pool composition
    """

    def __init__(
        self,
        env: TradingEnv,
        pool: AgentPool,
        curriculum: Curriculum | None = None,
        metrics: MetricsTracker | None = None,
        episodes_per_generation: int = 100,
        eval_episodes: int = 10,
        num_generations: int = 10,
        top_k_promote: int = 2,
        bottom_k_demote: int = 1,
        checkpoint_dir: str = "checkpoints",
    ):
        self.env = env
        self.pool = pool
        self.curriculum = curriculum or Curriculum()
        self.metrics = metrics or MetricsTracker()
        self.episodes_per_generation = episodes_per_generation
        self.eval_episodes = eval_episodes
        self.num_generations = num_generations
        self.top_k_promote = top_k_promote
        self.bottom_k_demote = bottom_k_demote
        self.checkpoint_dir = checkpoint_dir

        self._generation = 0

    def train(self) -> dict[str, Any]:
        """Run full population-based training."""
        generation_results = []

        for gen in range(self.num_generations):
            self._generation = gen + 1
            logger.info(f"=== Generation {self._generation}/{self.num_generations} ===")
            logger.info(f"Pool: {self.pool.size} agents ({len(self.pool.get_learning_agents())} learning)")

            # 1. Train
            trainer = Trainer(
                env=self.env,
                pool=self.pool,
                metrics=self.metrics,
                eval_interval=max(self.episodes_per_generation // 5, 1),
                checkpoint_interval=self.episodes_per_generation,
                checkpoint_dir=f"{self.checkpoint_dir}/gen_{self._generation}",
            )

            train_result = trainer.train_episodes(self.episodes_per_generation)
            logger.info(f"Training: mean_reward={train_result['mean_reward']:.4f}")

            # 2. Evaluate all agents individually
            eval_scores = self._evaluate_agents()

            # 3. Update rankings
            self.pool.update_rankings(eval_scores)

            # 4. Promote top learning agents
            promoted = self.pool.promote_top(self.top_k_promote)

            # 5. Demote bottom static agents (keep pool size manageable)
            demoted = self.pool.demote_bottom(self.bottom_k_demote)

            # 6. Apply curriculum
            self.curriculum.on_generation(self._generation, eval_scores)

            gen_result = {
                "generation": self._generation,
                "train_mean_reward": train_result["mean_reward"],
                "eval_scores": eval_scores,
                "promoted": promoted,
                "demoted": demoted,
                "pool_size": self.pool.size,
            }
            generation_results.append(gen_result)

            self.metrics.log_generation(self._generation, {
                "train_mean_reward": train_result["mean_reward"],
                "pool_size": float(self.pool.size),
                "best_eval_score": max(eval_scores.values()) if eval_scores else 0.0,
            })

            logger.info(
                f"Gen {self._generation}: promoted={promoted}, demoted={demoted}, "
                f"pool_size={self.pool.size}"
            )

        return {
            "total_generations": self.num_generations,
            "generations": generation_results,
            "final_rankings": dict(self.pool.get_ranked_agents()),
        }

    def _evaluate_agents(self) -> dict[str, float]:
        """Evaluate each agent by running episodes with only that agent active."""
        scores: dict[str, float] = {}

        for agent in self.pool.get_all():
            episode_rewards = []

            for _ in range(self.eval_episodes):
                obs, info = self.env.reset()
                total_reward = 0.0

                while True:
                    action = agent.select_action(obs, deterministic=True)
                    obs, reward, terminated, truncated, step_info = self.env.step(action)
                    total_reward += reward
                    if terminated or truncated:
                        break

                episode_rewards.append(total_reward)

            scores[agent.name] = float(np.mean(episode_rewards))

        return scores

    @property
    def generation(self) -> int:
        return self._generation
