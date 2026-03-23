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
from hydra.distillation.auto_reward_tuner import AutoRewardTuner
from hydra.envs.trading_env import TradingEnv
from hydra.evaluation.competition import AgentCompetitionScore, CompetitionRebalancer
from hydra.evaluation.conviction import ConvictionCalibrator
from hydra.evolution.diagnostics import DiagnosticEngine, GenerationMetrics
from hydra.agents.td3_agent import TD3Agent
from hydra.training.curriculum import Curriculum
from hydra.training.metrics_tracker import MetricsTracker
from hydra.training.trainer import Trainer

logger = logging.getLogger("hydra.training.population")

# Timesteps of SB3 gradient training per learning agent per generation.
# Must be >= PPO n_steps * 2 (4096 ≥ 2048 * 2) to ensure at least two
# full rollout buffer fills and meaningful gradient updates per generation.
_DEFAULT_TRAIN_TIMESTEPS = 4096


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
        bottom_k_demote: int = 3,
        max_pool_size: int = 20,
        checkpoint_dir: str = "checkpoints",
        train_timesteps: int = _DEFAULT_TRAIN_TIMESTEPS,
        enable_diagnostics: bool = True,
        enable_competition: bool = True,
        enable_conviction: bool = True,
        start_generation: int = 0,
        auto_reward_tuner: AutoRewardTuner | None = None,
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
        self.max_pool_size = max_pool_size
        self.checkpoint_dir = checkpoint_dir
        self.train_timesteps = train_timesteps
        self.on_generation: Any = None  # Optional callback(gen_idx, gen_result)
        self.on_intervention: Any = None  # Optional callback(gen_idx, gen_results) -> intervention dict

        self._generation = start_generation
        self._start_generation = start_generation
        self._agent_envs: dict[str, TradingEnv] = {}  # Persistent per-agent envs

        # CHIMERA diagnostics (monitoring + weak agent detection)
        self.enable_diagnostics = enable_diagnostics
        self._diagnostics = DiagnosticEngine() if enable_diagnostics else None

        # PROMETHEUS competition rebalancer
        self.enable_competition = enable_competition
        self._competition = CompetitionRebalancer() if enable_competition else None

        # ELEOS Bayesian conviction calibrator
        self.enable_conviction = enable_conviction
        self._conviction = ConvictionCalibrator() if enable_conviction else None

        # Auto reward tuner (CHIMERA-driven)
        self._auto_reward_tuner = auto_reward_tuner

    def train(self) -> dict[str, Any]:
        """Run full population-based training with CHIMERA/PROMETHEUS/ELEOS integration."""
        generation_results = []

        if self._start_generation > 0:
            logger.info(f"Resuming from generation {self._start_generation}")

        for gen in range(self._start_generation, self.num_generations):
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

            # --- CHIMERA: Diagnostics + Circuit Breakers ---
            diagnosis = None
            if self._diagnostics:
                diagnosis = self._run_diagnostics(train_result, eval_scores)

            # --- Auto Reward Tuning (CHIMERA-driven) ---
            if (
                self._auto_reward_tuner
                and diagnosis
                and self._auto_reward_tuner.should_tune(self._generation)
                and hasattr(self.env, "reward_fn")
            ):
                try:
                    mutations = diagnosis.get("recommended_mutations", [])
                    current_params = self.env.reward_fn.get_params()
                    new_params = self._auto_reward_tuner.apply_mutations(
                        current_params, mutations
                    )
                    if new_params != current_params:
                        self.env.reward_fn.update_params(new_params)
                        # Also update per-agent envs
                        for agent_env in self._agent_envs.values():
                            for i in range(getattr(agent_env, "num_envs", 0)):
                                sub_env = agent_env.envs[i]
                                if hasattr(sub_env, "reward_fn"):
                                    sub_env.reward_fn.update_params(new_params)
                        logger.info(f"  Reward params auto-tuned at gen {self._generation}")
                        # Flush off-policy replay buffers so agents don't
                        # train on experiences scored under the old reward.
                        for agent in self.pool.agents:
                            if isinstance(agent, TD3Agent) and agent._model is not None:
                                agent._model.replay_buffer.reset()
                                logger.info(f"  Flushed replay buffer for {agent.name}")
                except Exception as e:
                    logger.warning(f"Auto reward tuning failed: {e}")

            # --- ELEOS: Bayesian conviction calibration (runs BEFORE PROMETHEUS) ---
            # Trust gating: new/unproven agents start at 50% allocation.
            # Running before PROMETHEUS ensures conviction-adjusted weights are
            # the input to competition rebalancing, not overwritten by it.
            if self._conviction:
                self._update_conviction(eval_scores)

            # --- PROMETHEUS: Competition-based weight rebalancing ---
            # Now operates on conviction-gated weights with EMA smoothing.
            competition_result = None
            if self._competition:
                competition_result = self._run_competition(eval_scores)

            # 4. Update rankings
            self.pool.update_rankings(eval_scores)

            # 5. Promote top learning agents
            promoted = self.pool.promote_top(self.top_k_promote)

            # 6. Demote bottom static agents (keep pool size manageable)
            demoted = self.pool.demote_bottom(self.bottom_k_demote)

            # 6b. Enforce max pool size cap — demote excess bottom agents
            if self.pool.size > self.max_pool_size:
                excess = self.pool.size - self.max_pool_size
                additional = self.pool.demote_bottom(excess)
                demoted.extend(additional)
                logger.info(
                    f"Pool cap enforcement: demoted {len(additional)} extra "
                    f"agents to reach max_pool_size={self.max_pool_size}"
                )

            # 7. Apply curriculum
            adjustments = self.curriculum.on_generation(self._generation, eval_scores)

            # Apply regime-conditional rewards if regime changed
            regime = adjustments.get("regime")
            if regime and hasattr(self.env, "reward_fn"):
                try:
                    self.env.reward_fn.set_regime(regime)
                    logger.info(f"  Reward regime set to: {regime}")
                except Exception as e:
                    logger.debug(f"  Could not set reward regime: {e}")

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
                    "circuit_breaker_actions": [
                        {"action": cb.action, "target": cb.target_agent, "reduction_pct": cb.reduction_pct}
                        for cb in diagnosis.get("circuit_breaker_actions", [])
                    ],
                }
            if competition_result:
                gen_result["competition"] = {
                    "weights_after": competition_result.weights_after,
                    "converged": competition_result.converged,
                }
            if self._conviction:
                gen_result["conviction"] = self._conviction.get_all_summaries()

            generation_results.append(gen_result)

            best_agent_name = max(eval_scores, key=eval_scores.get) if eval_scores else "N/A"
            best_agent_score = max(eval_scores.values()) if eval_scores else 0.0

            self.metrics.log_generation(self._generation, {
                "train_mean_reward": train_result["mean_reward"],
                "pool_size": float(self.pool.size),
                "best_eval_score": best_agent_score,
                "best_agent": best_agent_name,
            })

            logger.info(
                f"Gen {self._generation}: best={best_agent_name} "
                f"(Sharpe={best_agent_score:.3f}), promoted={promoted}, "
                f"demoted={demoted}, pool_size={self.pool.size}"
            )

            # Fire generation callback (used for live dashboard updates)
            if self.on_generation:
                try:
                    self.on_generation(self._generation, generation_results)
                except Exception as e:
                    logger.warning(f"Generation callback failed: {e}")

            # Fire intervention hook (Operations Monitor + Regime + Risk Manager)
            if self.on_intervention:
                try:
                    intervention = self.on_intervention(
                        self._generation, generation_results
                    )
                    if intervention:
                        # Config patches from Operations Monitor
                        if intervention.get("type") == "config_patch":
                            patches = intervention.get("patches", {})
                            if "max_pool_size" in patches:
                                self.max_pool_size = patches["max_pool_size"]
                            if "bottom_k_demote" in patches:
                                self.bottom_k_demote = patches["bottom_k_demote"]
                            logger.info(
                                f"Intervention applied at gen {self._generation}: "
                                f"{list(patches.keys())}"
                            )

                        # Regime injection from Geopolitics Expert via corp state
                        int_regime = intervention.get("regime")
                        if int_regime:
                            self.curriculum.set_regime(int_regime)
                            if hasattr(self.env, "reward_fn"):
                                try:
                                    self.env.reward_fn.set_regime(int_regime)
                                    logger.info(f"  Reward regime set to: {int_regime} (from corp state)")
                                except Exception as e:
                                    logger.debug(f"  Could not set reward regime: {e}")

                        # Weight overrides from Risk Manager (circuit breaker enforcement)
                        weight_overrides = intervention.get("weight_overrides")
                        if weight_overrides:
                            for agent_name, new_weight in weight_overrides.items():
                                try:
                                    self.pool.set_weight(agent_name, new_weight)
                                    logger.info(
                                        f"  RISK MANAGER: {agent_name} weight → {new_weight:.4f}"
                                    )
                                except Exception:
                                    pass
                except Exception as e:
                    logger.warning(f"Intervention hook failed: {e}")

        return {
            "total_generations": self.num_generations,
            "generations": generation_results,
            "final_rankings": dict(self.pool.get_ranked_agents()),
        }

    def _train_agents_on_env(self) -> None:
        """Run SB3 gradient-based training for learning agents.

        Each learning agent gets a persistent env copy (created once, reused
        across generations) to avoid state contention AND preserve SB3
        internal state (SAC replay buffer, PPO rollout buffer).

        Uses ThreadPoolExecutor for parallelism on CPU.  When a GPU device
        (DirectML / CUDA) is detected, workers are serialized (max_workers=1)
        because DirectML crashes on concurrent GPU access from threads.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        learning_agents = [
            a for a in self.pool.get_learning_agents()
            if hasattr(a, "train_on_env")
        ]
        if not learning_agents:
            return

        # Detect GPU usage — DirectML and CUDA can't safely share a single
        # device across threads, so serialize training in that case.
        uses_gpu = any(
            getattr(a, "_device", "cpu") in ("dml", "cuda")
            for a in learning_agents
        )
        max_workers = 1 if uses_gpu else len(learning_agents)
        mode = "sequential (GPU)" if uses_gpu else f"parallel ({len(learning_agents)} workers)"
        logger.info(f"  SB3 training mode: {mode}")

        def train_one(agent):
            env = self._get_agent_env(agent.name)
            result = agent.train_on_env(env, total_timesteps=self.train_timesteps)
            return agent.name, result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(train_one, agent): agent
                for agent in learning_agents
            }
            for future in as_completed(futures):
                agent = futures[future]
                try:
                    name, result = future.result()
                    logger.info(
                        f"  SB3 training '{name}': "
                        f"{self.train_timesteps} timesteps, "
                        f"total={result.get('total_timesteps', 0):.0f}"
                    )
                except Exception as e:
                    logger.warning(f"  SB3 training '{agent.name}' failed: {e}")

    def _get_agent_env(self, agent_name: str):
        """Get or create a persistent vectorized env for a specific agent.

        Creates a DummyVecEnv with n_envs copies of TradingEnv for batched
        SB3 training.  Envs are created once and reused across generations
        so that SB3's set_env() identity check works correctly — preserving
        SAC's replay buffer and PPO's rollout state between generations.
        """
        if agent_name not in self._agent_envs:
            from stable_baselines3.common.vec_env import DummyVecEnv

            kwargs = self.env.get_init_kwargs()
            n_envs = 4

            def _make_env(seed_offset, kw=kwargs):
                def _init():
                    return TradingEnv(seed=seed_offset, **kw)
                return _init

            self._agent_envs[agent_name] = DummyVecEnv(
                [_make_env(i) for i in range(n_envs)]
            )
        return self._agent_envs[agent_name]

    def _evaluate_agents(self) -> dict[str, float]:
        """Evaluate each agent by running episodes with only that agent active.

        Each eval episode is seeded so that:
        - Each agent gets a reproducible but unique set of market windows
        - Different generations evaluate on different windows (prevents overfitting)
        - Per-agent seed offsets ensure even near-identical policies produce
          different scores (they face different market conditions)

        The per-agent offset uses hash(agent.name) % 10000 so results are
        deterministic and reproducible, but each agent sees different data.
        With eval_episodes=10, the mean across diverse windows provides a
        robust ranking even when policies are similar.
        """
        scores: dict[str, float] = {}
        # Base seed changes per generation so eval windows rotate
        gen_seed = 42 + self._generation * 1000

        for agent in self.pool.get_all():
            episode_rewards = []
            # Per-agent offset ensures different agents see different windows,
            # revealing policy differences that identical seeds would hide.
            agent_offset = hash(agent.name) % 10000

            for ep in range(self.eval_episodes):
                obs, info = self.env.reset(seed=gen_seed + ep * 100 + agent_offset)
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
        """Run CHIMERA diagnostic engine — monitoring + weak agent detection.

        Logs issues and severity. Also generates circuit breaker actions
        for use in forward-test / paper trading mode.
        """
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
                for mut in diagnosis["recommended_mutations"]:
                    logger.info(f"  Recommended [{mut.category}] {mut.mutation_type}: {mut.description}")

                # Circuit breaker actions — logged for backtesting,
                # acted upon in forward-test/paper mode by the caller.
                cb_actions = self._diagnostics.get_circuit_breaker_actions(diagnosis)
                diagnosis["circuit_breaker_actions"] = cb_actions
                for cb in cb_actions:
                    target = cb.target_agent or "portfolio"
                    logger.info(
                        f"  CIRCUIT BREAKER: {cb.action} → {target} "
                        f"(reduction={cb.reduction_pct:.0%})"
                    )

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
