"""Population-based training with generational evolution.

Trains agents in generations: each generation runs episodes for ranking,
then does actual gradient-based learning via SB3's train_on_env(),
and finally promotes/demotes agents based on evaluation scores.

Integrates:
- CHIMERA diagnostics + mutation engine (adaptive evolution)
- PROMETHEUS competition-based weight rebalancing
- ELEOS Bayesian conviction calibration
- CHIMERA multi-objective fitness decomposition
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from hydra.agents.agent_pool import AgentPool
from hydra.envs.trading_env import TradingEnv
from hydra.evaluation.competition import AgentCompetitionScore, CompetitionRebalancer
from hydra.evaluation.conviction import ConvictionCalibrator
from hydra.evolution.diagnostics import DiagnosticEngine, GenerationMetrics
from hydra.evolution.mutation_engine import MutationEngine, generate_random_variant
from hydra.training.curriculum import Curriculum
from hydra.training.metrics_tracker import MetricsTracker
from hydra.training.trainer import Trainer

logger = logging.getLogger("hydra.training.population")

# Timesteps of SB3 gradient training per learning agent per generation
_DEFAULT_TRAIN_TIMESTEPS = 500


class PopulationTrainer:
    """Generational population-based training with CHIMERA/PROMETHEUS/ELEOS integration.

    Each generation:
    1. Run multi-agent episodes (collect experience + metrics)
    2. Run SB3 gradient training on each learning agent (real weight updates)
    3. Evaluate all agents individually
    4. CHIMERA diagnostics: analyze results and recommend mutations
    5. PROMETHEUS competition: rebalance agent weights by rank
    6. ELEOS conviction: calibrate weights by Bayesian win/loss tracking
    7. Rank by evaluation performance
    8. Promote top-K learning agents → static snapshots
    9. Demote bottom-K static agents (remove from pool)
    10. Apply curriculum adjustments
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
        train_timesteps: int = _DEFAULT_TRAIN_TIMESTEPS,
        enable_diagnostics: bool = True,
        enable_competition: bool = True,
        enable_conviction: bool = True,
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
        self.train_timesteps = train_timesteps

        self._generation = 0

        # CHIMERA diagnostics + mutation engine
        self.enable_diagnostics = enable_diagnostics
        self._diagnostics = DiagnosticEngine() if enable_diagnostics else None
        self._overlay: dict = {
            "agent_weights": {},
            "agent_params": {},
            "risk": {},
            "reward": {},
            "benched_agents": [],
            "fitness_weights": {},
        }
        self._mutations_applied: list = []

        # PROMETHEUS competition rebalancer
        self.enable_competition = enable_competition
        self._competition = CompetitionRebalancer() if enable_competition else None

        # ELEOS Bayesian conviction calibrator
        self.enable_conviction = enable_conviction
        self._conviction = ConvictionCalibrator() if enable_conviction else None

    def train(self) -> dict[str, Any]:
        """Run full population-based training with CHIMERA/PROMETHEUS/ELEOS integration."""
        generation_results = []

        for gen in range(self.num_generations):
            self._generation = gen + 1
            logger.info(f"=== Generation {self._generation}/{self.num_generations} ===")
            logger.info(f"Pool: {self.pool.size} agents ({len(self.pool.get_learning_agents())} learning)")

            # 1. Run multi-agent episodes (collect experience + metrics)
            trainer = Trainer(
                env=self.env,
                pool=self.pool,
                metrics=self.metrics,
                eval_interval=max(self.episodes_per_generation // 5, 1),
                checkpoint_interval=self.episodes_per_generation,
                checkpoint_dir=f"{self.checkpoint_dir}/gen_{self._generation}",
            )

            train_result = trainer.train_episodes(self.episodes_per_generation)
            logger.info(f"Multi-agent episodes: mean_reward={train_result['mean_reward']:.4f}")

            # 2. SB3 gradient training — real neural network learning
            self._train_agents_on_env()

            # 3. Evaluate all agents individually
            eval_scores = self._evaluate_agents()

            # Log per-agent eval scores
            for agent_name, score in eval_scores.items():
                self.metrics.log_agent_eval(self._generation, agent_name, score)

            # --- CHIMERA: Diagnostics + Mutations ---
            diagnosis = None
            if self._diagnostics:
                diagnosis = self._run_diagnostics(train_result, eval_scores)

            # --- PROMETHEUS: Competition-based weight rebalancing ---
            competition_result = None
            if self._competition:
                competition_result = self._run_competition(eval_scores)

            # --- ELEOS: Bayesian conviction calibration ---
            if self._conviction:
                self._update_conviction(eval_scores)

            # 4. Update rankings
            self.pool.update_rankings(eval_scores)

            # 5. Promote top learning agents
            promoted = self.pool.promote_top(self.top_k_promote)

            # 6. Demote bottom static agents (keep pool size manageable)
            demoted = self.pool.demote_bottom(self.bottom_k_demote)

            # 7. Apply curriculum
            self.curriculum.on_generation(self._generation, eval_scores)

            gen_result = {
                "generation": self._generation,
                "train_mean_reward": train_result["mean_reward"],
                "eval_scores": eval_scores,
                "promoted": promoted,
                "demoted": demoted,
                "pool_size": self.pool.size,
            }

            # Attach diagnostic/competition/conviction results
            if diagnosis:
                gen_result["diagnosis"] = {
                    "severity": diagnosis["severity"],
                    "primary_issue": diagnosis["primary_issue"],
                    "num_mutations": len(diagnosis["recommended_mutations"]),
                }
            if competition_result:
                gen_result["competition"] = {
                    "weights_after": competition_result.weights_after,
                    "converged": competition_result.converged,
                }
            if self._conviction:
                gen_result["conviction"] = self._conviction.get_all_summaries()

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

    def _train_agents_on_env(self) -> None:
        """Run SB3 gradient-based training for each learning agent.

        This is where actual neural network weight updates happen.
        Each learning agent runs train_on_env() which calls SB3's learn()
        method, performing real policy gradient descent.
        """
        for agent in self.pool.get_learning_agents():
            if not hasattr(agent, "train_on_env"):
                continue
            try:
                result = agent.train_on_env(self.env, total_timesteps=self.train_timesteps)
                logger.info(
                    f"  SB3 training '{agent.name}': "
                    f"{self.train_timesteps} timesteps, "
                    f"total={result.get('total_timesteps', 0):.0f}"
                )
            except Exception as e:
                logger.warning(f"  SB3 training '{agent.name}' failed: {e}")

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

    # -------------------------------------------------------------------
    # CHIMERA: Diagnostics + Mutation integration
    # -------------------------------------------------------------------

    def _run_diagnostics(
        self, train_result: dict, eval_scores: dict[str, float]
    ) -> dict | None:
        """Run CHIMERA diagnostic engine and apply recommended mutations."""
        try:
            gen_metrics = GenerationMetrics(
                generation=self._generation,
                mean_reward=train_result.get("mean_reward", 0.0),
                best_reward=max(eval_scores.values()) if eval_scores else 0.0,
                agent_scores=eval_scores,
                total_trades=train_result.get("total_trades", 0),
            )

            self._diagnostics.add_generation(gen_metrics)
            diagnosis = self._diagnostics.diagnose(gen_metrics)

            if diagnosis["severity"] in ("moderate", "severe", "critical"):
                logger.info(
                    f"  CHIMERA [{diagnosis['severity']}]: {diagnosis['primary_issue']}"
                )
                # Apply recommended mutations to overlay
                mutations = diagnosis["recommended_mutations"]
                if mutations:
                    engine = MutationEngine(self._overlay)
                    self._overlay = engine.apply_mutations(mutations)
                    self._mutations_applied.extend(mutations)
                    logger.info(f"  Applied {len(mutations)} mutations")

                    # If competition rebalancer exists, apply weight mutations
                    new_weights = self._overlay.get("agent_weights", {})
                    if new_weights:
                        for agent_name, weight in new_weights.items():
                            if weight > 0:
                                self.pool.set_weight(agent_name, weight)

            return diagnosis
        except Exception as e:
            logger.warning(f"Diagnostics failed: {e}")
            return None

    # -------------------------------------------------------------------
    # PROMETHEUS: Competition-based weight rebalancing
    # -------------------------------------------------------------------

    def _run_competition(self, eval_scores: dict[str, float]) -> Any:
        """Run PROMETHEUS competition and rebalance pool weights."""
        try:
            agent_scores = []
            for name, score in eval_scores.items():
                agent_scores.append(AgentCompetitionScore(
                    agent_name=name,
                    sharpe=score / 100,  # normalize reward to Sharpe-like scale
                    win_rate=0.5 + score / 200,  # rough proxy
                    profit_factor=max(0.5, 1.0 + score / 100),
                    max_drawdown=max(0, -score / 100),
                ))

            current_weights = {
                name: self.pool._weights.get(name, 1.0 / max(self.pool.size, 1))
                for name in self.pool.agent_names
            }

            result = self._competition.evaluate_generation(
                generation=self._generation,
                agent_scores=agent_scores,
                current_weights=current_weights,
            )

            # Apply new weights to pool
            for agent_name, weight in result.weights_after.items():
                self.pool.set_weight(agent_name, weight)

            logger.info(
                f"  PROMETHEUS: rebalanced weights — "
                f"top={list(result.weights_after.items())[:2]}"
            )

            return result
        except Exception as e:
            logger.warning(f"Competition rebalancing failed: {e}")
            return None

    # -------------------------------------------------------------------
    # ELEOS: Bayesian conviction calibration
    # -------------------------------------------------------------------

    def _update_conviction(self, eval_scores: dict[str, float]) -> None:
        """Update ELEOS conviction trackers from evaluation results."""
        try:
            # Convert eval scores into win/loss outcomes
            agent_rewards: dict[str, list[float]] = {}
            for name, score in eval_scores.items():
                # Each eval score represents mean reward — treat positive as win
                agent_rewards[name] = [score]

            self._conviction.record_episode_outcomes(agent_rewards, threshold=0.0)

            # Apply conviction-adjusted weights
            base_weights = {
                name: self.pool._weights.get(name, 1.0 / max(self.pool.size, 1))
                for name in self.pool.agent_names
            }
            adjusted = self._conviction.get_conviction_weights(base_weights)
            for agent_name, weight in adjusted.items():
                self.pool.set_weight(agent_name, weight)

            # Log conviction state
            for name, tracker in self._conviction._trackers.items():
                scale = tracker.get_conviction_scale()
                if abs(scale - 1.0) > 0.01:
                    logger.info(
                        f"  ELEOS: {name} conviction={scale:.3f} "
                        f"(WR={tracker.overall_win_rate:.1%})"
                    )
        except Exception as e:
            logger.warning(f"Conviction calibration failed: {e}")

    @property
    def generation(self) -> int:
        return self._generation
