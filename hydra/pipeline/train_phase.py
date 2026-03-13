"""Phase 3: RL training — wraps population_trainer with GPU task routing."""

from __future__ import annotations

import logging
from typing import Any

from hydra.agents.agent_pool import AgentPool
from hydra.agents.ppo_agent import PPOAgent
from hydra.agents.sac_agent import SACAgent
from hydra.agents.rule_based_agent import RuleBasedAgent
from hydra.compute.decorators import gpu_task
from hydra.envs.trading_env import TradingEnv
from hydra.training.metrics_tracker import MetricsTracker
from hydra.training.population_trainer import PopulationTrainer

logger = logging.getLogger("hydra.pipeline.train_phase")


@gpu_task(memory_gb=4)
def run_training(
    deps: dict[str, Any],
    num_generations: int = 10,
    episodes_per_generation: int = 100,
    top_k_promote: int = 2,
    bottom_k_demote: int = 1,
    checkpoint_dir: str = "checkpoints",
    tensorboard_dir: str = "logs/tensorboard",
    prefer_gpu: bool = True,
) -> dict[str, Any]:
    """Run the full population-based training pipeline.

    Args:
        deps: Must contain 'env_builder' with train_env.

    Returns:
        Training results including final rankings and metrics.
    """
    env_result = deps.get("env_builder", {})
    train_env: TradingEnv = env_result.get("train_env")
    split_info = env_result.get("split_info", {})

    if train_env is None:
        raise ValueError("No training environment found in dependencies")

    num_stocks = split_info.get("num_stocks", train_env.num_stocks)
    obs_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]

    logger.info(f"Initializing agent pool: obs_dim={obs_dim}, action_dim={action_dim}")

    # Build agent pool
    pool = AgentPool()

    # Learning agents
    pool.add(PPOAgent("ppo_1", obs_dim, action_dim, prefer_gpu=prefer_gpu))
    pool.add(SACAgent("sac_1", obs_dim, action_dim, prefer_gpu=prefer_gpu))

    # Rule-based agents
    pool.add(RuleBasedAgent(
        "alpha_rule", obs_dim, action_dim,
        agent_class_path="alpha_momentum.AlphaMomentum",
    ))
    pool.add(RuleBasedAgent(
        "beta_rule", obs_dim, action_dim,
        agent_class_path="beta_mean_reversion.BetaMeanReversion",
    ))

    logger.info(f"Pool initialized: {pool.size} agents")

    # Metrics
    metrics = MetricsTracker(log_dir=tensorboard_dir)

    # Each learning agent gets gradient training per generation.
    # timesteps = episode_bars * episodes gives enough experience for SB3's learn().
    train_timesteps = train_env.episode_bars * episodes_per_generation

    # Run population-based training
    pop_trainer = PopulationTrainer(
        env=train_env,
        pool=pool,
        metrics=metrics,
        episodes_per_generation=episodes_per_generation,
        num_generations=num_generations,
        top_k_promote=top_k_promote,
        bottom_k_demote=bottom_k_demote,
        checkpoint_dir=checkpoint_dir,
        train_timesteps=train_timesteps,
    )

    results = pop_trainer.train()

    metrics.close()

    return {
        "training_results": results,
        "pool": pool,
        "metrics_summary": metrics.get_summary(),
    }
