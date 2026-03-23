"""Phase 3: RL training — wraps population_trainer with GPU task routing."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from hydra.agents.agent_pool import AgentPool
from hydra.agents.cmaes_agent import CMAESAgent
from hydra.agents.ppo_agent import PPOAgent
from hydra.agents.td3_agent import TD3Agent
from hydra.agents.recurrent_ppo_agent import RecurrentPPOAgent
from hydra.agents.rule_based_agent import RuleBasedAgent
from hydra.compute.decorators import gpu_task
from hydra.envs.trading_env import TradingEnv
from hydra.training.metrics_tracker import MetricsTracker
from hydra.training.population_trainer import PopulationTrainer

logger = logging.getLogger("hydra.pipeline.train_phase")

# Dashboard state file — written incrementally after each generation
_STATE_FILE = Path(__file__).parent.parent.parent / "logs" / "hydra_training_state.json"

# Pointer to the latest checkpoint — used for automatic warm-start between runs
_LATEST_CKPT_FILE = "latest.json"


@gpu_task(memory_gb=4)
def run_training(
    deps: dict[str, Any],
    num_generations: int = 10,
    episodes_per_generation: int = 50,
    top_k_promote: int = 2,
    bottom_k_demote: int = 3,
    max_pool_size: int = 20,
    checkpoint_dir: str = "checkpoints",
    tensorboard_dir: str = "logs/tensorboard",
    prefer_gpu: bool = True,
    resume_from: str | None = None,
) -> dict[str, Any]:
    """Run the full population-based training pipeline.

    Args:
        deps: Must contain 'env_builder' with train_env.
        resume_from: Path to checkpoint directory to resume from.

    Returns:
        Training results including final rankings and metrics.
    """
    env_result = deps.get("env_builder", {})
    data_result = deps.get("data_prep", env_result.get("data_prep", {}))
    train_env: TradingEnv = env_result.get("train_env")
    split_info = env_result.get("split_info", {})

    if train_env is None:
        raise ValueError("No training environment found in dependencies")

    num_stocks = split_info.get("num_stocks", train_env.num_stocks)
    obs_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]

    start_generation = 0

    # Resume from checkpoint or build fresh pool
    if resume_from:
        # Explicit resume: load full pool and continue generation numbering
        ckpt_path = Path(resume_from)

        if not ckpt_path.exists():
            logger.warning(f"Checkpoint path not found: {ckpt_path}, starting fresh")
            pool = _build_fresh_pool(obs_dim, action_dim)
        else:
            pool = AgentPool()
            pool.load(ckpt_path)

            # Infer generation from checkpoint path: .../gen_20/episode_65
            start_generation = _infer_generation(ckpt_path)
            logger.info(
                f"Resumed from checkpoint: {ckpt_path} "
                f"(gen {start_generation}, {pool.size} agents)"
            )

            # Ensure learning agents exist (they may not be in the checkpoint
            # if only static snapshots were saved, or if all loads failed)
            if not pool.get_learning_agents():
                logger.warning(
                    "No learning agents after initial load from "
                    f"{ckpt_path}. Attempting CPU-only recovery..."
                )
                recovered = _try_recover_learning_agents(pool, ckpt_path, obs_dim, action_dim)
                if not recovered:
                    logger.error(
                        "WEIGHT LOSS: No learning agents survived checkpoint load or "
                        f"CPU recovery from {ckpt_path}. All learned weights are LOST. "
                        "Adding fresh random-weight agents — training restarts from scratch."
                    )
                    pool.add(PPOAgent("ppo_1", obs_dim, action_dim, prefer_gpu=False))
                    pool.add(PPOAgent("ppo_2", obs_dim, action_dim, net_arch=[128, 128],
                                       learning_rate=1e-3, ent_coef=0.02, prefer_gpu=False))
                    pool.add(TD3Agent("td3_1", obs_dim, action_dim, prefer_gpu=False))
                    pool.add(RecurrentPPOAgent("rppo_1", obs_dim, action_dim, prefer_gpu=False))

            # Add ppo_2 variant if not already in pool (new agent added post-checkpoint)
            if not pool.get("ppo_2"):
                pool.add(PPOAgent("ppo_2", obs_dim, action_dim, net_arch=[128, 128],
                                   learning_rate=1e-3, ent_coef=0.02, prefer_gpu=False))
                logger.info("Added new learning agent 'ppo_2' (128x128, lr=1e-3, ent=0.02)")
    else:
        # Auto warm-start: load learning agents from latest checkpoint if available
        pool = _try_warm_start(checkpoint_dir, obs_dim, action_dim)
        if pool is None:
            pool = _build_fresh_pool(obs_dim, action_dim)

    logger.info(f"Pool initialized: {pool.size} agents")

    # Metrics
    metrics = MetricsTracker(log_dir=tensorboard_dir)

    # Each learning agent gets gradient training per generation.
    # Ensure enough timesteps for PPO to fill its rollout buffer multiple times
    # (n_steps=2048 default → 4 fills = 8192) AND enough for ~100 episodes worth
    # of experience.  Decoupled from eval episodes so reducing eval doesn't
    # reduce training budget.
    ppo_n_steps = 2048  # PPO default; must match PPOAgent.n_steps
    train_timesteps = max(ppo_n_steps * 4, train_env.episode_bars * 100)
    logger.info(
        f"Train timesteps per agent per generation: {train_timesteps} "
        f"(env_based={train_env.episode_bars * 100}, ppo_min={ppo_n_steps * 4})"
    )

    # Auto reward tuner (CHIMERA-driven)
    auto_tuner = None
    try:
        from hydra.distillation.auto_reward_tuner import AutoRewardTuner
        auto_tuner = AutoRewardTuner(
            tune_every_n=5,
            max_delta_pct=0.20,
            enabled=True,
        )
        logger.info("Auto reward tuner enabled (tune every 5 generations, max delta 20%)")
    except Exception as e:
        logger.warning(f"Could not initialize auto reward tuner: {e}")

    # Run population-based training
    pop_trainer = PopulationTrainer(
        env=train_env,
        pool=pool,
        metrics=metrics,
        episodes_per_generation=episodes_per_generation,
        num_generations=num_generations,
        top_k_promote=top_k_promote,
        bottom_k_demote=bottom_k_demote,
        max_pool_size=max_pool_size,
        checkpoint_dir=checkpoint_dir,
        train_timesteps=train_timesteps,
        start_generation=start_generation,
        auto_reward_tuner=auto_tuner,
    )

    # Live dashboard updates — write partial state after each generation.
    # Capture env + config info so the dashboard shows correct context.
    env_config = {
        "num_stocks": train_env.num_stocks,
        "episode_bars": train_env.episode_bars,
        "tickers": data_result.get("tickers", []),
        "real_data": bool(data_result.get("trading_dates")),  # real data has trading_dates
    }
    pop_trainer.on_generation = lambda gen_idx, gen_results: _write_live_state(
        gen_idx, num_generations, gen_results, episodes_per_generation,
        env_config, train_env,
    )

    # Wire Operations Monitor for mid-training intervention
    pop_trainer.on_intervention = _create_intervention_hook()

    results = pop_trainer.train()

    metrics.close()

    # Save pointer to latest checkpoint for automatic warm-start on next run
    _save_latest_checkpoint_pointer(checkpoint_dir, num_generations)

    return {
        "training_results": results,
        "pool": pool,
        "metrics_summary": metrics.get_summary(),
    }


def _build_fresh_pool(obs_dim: int, action_dim: int) -> AgentPool:
    """Create a fresh agent pool with default learning + rule-based agents."""
    pool = AgentPool()

    # Allow meta-optimizer to override network architecture via env var
    arch_size = int(os.environ.get("HYDRA_NET_ARCH_SIZE", "256"))
    net_arch = [arch_size, arch_size]

    # Learning agents — force CPU for SB3 MLP training.
    # DirectML's backward pass doesn't properly update weights (gradient
    # chain is broken across CPU fallback ops like aten::std.correction),
    # and SB3 itself warns that MLP policies train faster on CPU anyway.
    pool.add(PPOAgent("ppo_1", obs_dim, action_dim, net_arch=net_arch, prefer_gpu=False))
    pool.add(PPOAgent("ppo_2", obs_dim, action_dim, net_arch=[128, 128],
                       learning_rate=1e-3, ent_coef=0.02, prefer_gpu=False))
    pool.add(TD3Agent("td3_1", obs_dim, action_dim, net_arch=net_arch, prefer_gpu=False))
    pool.add(RecurrentPPOAgent("rppo_1", obs_dim, action_dim, prefer_gpu=False))

    # CMA-ES — gradient-free evolutionary agent, immune to reward non-stationarity
    pool.add(CMAESAgent("cmaes_1", obs_dim, action_dim))

    # Rule-based agents
    pool.add(RuleBasedAgent(
        "alpha_rule", obs_dim, action_dim,
        agent_class_path="alpha_momentum.AlphaMomentum",
    ))
    pool.add(RuleBasedAgent(
        "beta_rule", obs_dim, action_dim,
        agent_class_path="beta_mean_reversion.BetaMeanReversion",
    ))
    pool.add(RuleBasedAgent(
        "theta_rule", obs_dim, action_dim,
        agent_class_path="theta.ThetaAgent",
    ))

    return pool


def _save_latest_checkpoint_pointer(checkpoint_dir: str, num_generations: int) -> None:
    """Write a latest.json pointer so the next run can warm-start automatically.

    Finds the most recently written checkpoint from the final generation
    (by file modification time, not episode number) to avoid picking up
    stale checkpoints from earlier runs that share the same gen directory.
    """
    ckpt_root = Path(checkpoint_dir)
    latest_file = ckpt_root / _LATEST_CKPT_FILE

    # Find the final generation's latest checkpoint by modification time
    final_gen_dir = ckpt_root / f"gen_{num_generations}"
    if not final_gen_dir.exists():
        logger.warning(f"Final generation dir not found: {final_gen_dir}")
        return

    best_mtime = 0.0
    best_path = None
    best_episode = 0
    for entry in final_gen_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("episode_"):
            meta_file = entry / "pool_metadata.json"
            if meta_file.exists():
                mtime = meta_file.stat().st_mtime
                if mtime > best_mtime:
                    best_mtime = mtime
                    best_path = entry
                    try:
                        best_episode = int(entry.name.split("_")[1])
                    except (ValueError, IndexError):
                        best_episode = 0

    if best_path is None:
        logger.warning(f"No valid checkpoints found in {final_gen_dir}")
        return

    pointer = {
        "checkpoint_path": str(best_path),
        "generation": num_generations,
        "episode": best_episode,
        "saved_at": datetime.now().isoformat(),
    }

    try:
        ckpt_root.mkdir(parents=True, exist_ok=True)
        with open(latest_file, "w") as f:
            json.dump(pointer, f, indent=2)
        logger.info(f"Saved latest checkpoint pointer: {best_path}")
    except Exception as e:
        logger.warning(f"Failed to save latest checkpoint pointer: {e}")


def _try_recover_learning_agents(
    pool: AgentPool,
    ckpt_path: Path,
    obs_dim: int,
    action_dim: int,
) -> bool:
    """Attempt CPU-only reload of learning agents from checkpoint.

    Tries to recreate each expected learning agent (PPO, TD3, RecurrentPPO)
    and load their weights with explicit CPU device. This handles cases where
    the initial pool.load() silently failed due to device compatibility issues.

    Returns True if at least one learning agent was recovered.
    """
    agent_specs = [
        ("ppo_1", PPOAgent, {}),
        ("ppo_2", PPOAgent, {"net_arch": [128, 128], "learning_rate": 1e-3, "ent_coef": 0.02}),
        ("td3_1", TD3Agent, {}),
        ("rppo_1", RecurrentPPOAgent, {}),
    ]

    recovered = 0
    for name, agent_cls, kwargs in agent_specs:
        if pool.get(name):
            # Already loaded successfully
            if hasattr(pool.get(name), "frozen") and not pool.get(name).frozen:
                recovered += 1
            continue

        model_path = ckpt_path / name / "model"
        if not model_path.exists() and not model_path.with_suffix(".zip").exists():
            continue

        try:
            agent = agent_cls(name, obs_dim, action_dim, prefer_gpu=False, **kwargs)
            agent.load(model_path)
            pool.add(agent)
            recovered += 1
            logger.info(f"CPU recovery succeeded for agent '{name}'")
        except Exception as e:
            logger.error(f"CPU recovery failed for agent '{name}': {e}")

    if recovered > 0:
        logger.info(f"Recovered {recovered} learning agent(s) from checkpoint via CPU fallback")
    return recovered > 0


def _try_warm_start(
    checkpoint_dir: str,
    obs_dim: int,
    action_dim: int,
) -> AgentPool | None:
    """Try to warm-start from the latest checkpoint.

    Loads only learning agents (PPO, SAC, RecurrentPPO) with their trained
    weights from the previous run. Static snapshots and rule-based agents
    are rebuilt fresh — snapshots will be recreated through promotion, and
    rule-based agents don't carry state.

    Generation counter resets to 0 (this is a new run, not a continuation).

    Returns None if no checkpoint is available or if dimensions don't match.
    """
    if os.environ.get("HYDRA_FRESH_START"):
        logger.info("HYDRA_FRESH_START set — skipping warm-start")
        return None

    ckpt_root = Path(checkpoint_dir)
    latest_file = ckpt_root / _LATEST_CKPT_FILE

    if not latest_file.exists():
        logger.info("No previous checkpoint found — starting fresh")
        return None

    try:
        with open(latest_file) as f:
            pointer = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read latest checkpoint pointer: {e}")
        return None

    ckpt_path = Path(pointer["checkpoint_path"])
    if not ckpt_path.exists():
        logger.warning(f"Checkpoint path from pointer does not exist: {ckpt_path}")
        return None

    # Load the full pool to inspect it
    from hydra.utils.serialization import load_json
    metadata_file = ckpt_path / "pool_metadata.json"
    if not metadata_file.exists():
        logger.warning(f"No pool_metadata.json in {ckpt_path}")
        return None

    metadata = load_json(metadata_file)

    # Verify dimension compatibility — if obs_dim or action_dim changed
    # (e.g., different number of tickers), the saved weights are incompatible
    for name, info in metadata.get("agents", {}).items():
        if info.get("obs_dim") != obs_dim or info.get("action_dim") != action_dim:
            logger.warning(
                f"Dimension mismatch for '{name}': "
                f"checkpoint=({info.get('obs_dim')}, {info.get('action_dim')}), "
                f"current=({obs_dim}, {action_dim}). Starting fresh."
            )
            return None

    # Load learning agents only — force CPU to avoid DirectML load issues.
    # SAC is replaced by CMA-ES (SAC's entropy tuning is incompatible with
    # non-stationary rewards from auto reward tuning).
    learning_constructors = {
        "PPOAgent": lambda n, o, a: PPOAgent(n, o, a, prefer_gpu=False),
        "TD3Agent": lambda n, o, a: TD3Agent(n, o, a, prefer_gpu=False),
        "RecurrentPPOAgent": lambda n, o, a: RecurrentPPOAgent(n, o, a, prefer_gpu=False),
        "CMAESAgent": lambda n, o, a: CMAESAgent(n, o, a),
    }

    # When warm-starting from an old run that had SAC, replace with CMA-ES
    _AGENT_REPLACEMENTS = {
        "SACAgent": ("CMAESAgent", "cmaes_1"),
    }

    pool = AgentPool()
    loaded_count = 0
    has_td3 = False

    for name, info in metadata.get("agents", {}).items():
        agent_type = info["type"]

        # Check if this agent type should be replaced
        if agent_type in _AGENT_REPLACEMENTS:
            new_type, new_name = _AGENT_REPLACEMENTS[agent_type]
            logger.info(
                f"  Replacing '{name}' ({agent_type}) with fresh '{new_name}' ({new_type})"
            )
            constructor = learning_constructors.get(new_type)
            if constructor:
                agent = constructor(new_name, obs_dim, action_dim)
                pool.add(agent)
                loaded_count += 1
                has_td3 = True
            continue

        constructor = learning_constructors.get(agent_type)
        if constructor is None:
            continue

        if agent_type == "TD3Agent":
            has_td3 = True

        agent_dir = ckpt_path / name
        try:
            agent = constructor(name, obs_dim, action_dim)
            agent.load(agent_dir / "model")
            pool.add(agent)
            loaded_count += 1
            logger.info(
                f"  Warm-loaded '{name}' ({info['type']}, "
                f"{info.get('total_steps', 0)} steps from previous run)"
            )
        except Exception as e:
            logger.warning(f"  Failed to warm-load '{name}': {e}")

    # Ensure pool has a TD3 agent even if neither SAC nor TD3 was in checkpoint
    if not has_td3:
        pool.add(TD3Agent("td3_1", obs_dim, action_dim, prefer_gpu=False))
        loaded_count += 1
        logger.info("  Added fresh TD3 agent (no SAC/TD3 found in checkpoint)")

    if loaded_count == 0:
        logger.warning("No learning agents loaded from checkpoint — starting fresh")
        return None

    # Add fresh rule-based agents
    pool.add(RuleBasedAgent(
        "alpha_rule", obs_dim, action_dim,
        agent_class_path="alpha_momentum.AlphaMomentum",
    ))
    pool.add(RuleBasedAgent(
        "beta_rule", obs_dim, action_dim,
        agent_class_path="beta_mean_reversion.BetaMeanReversion",
    ))

    prev_gen = pointer.get("generation", "?")
    prev_ep = pointer.get("episode", "?")
    logger.info(
        f"Warm-start from previous run (gen {prev_gen}, episode {prev_ep}): "
        f"loaded {loaded_count} learning agents, added 2 rule-based agents"
    )

    return pool


def _infer_generation(ckpt_path: Path) -> int:
    """Infer generation number from checkpoint path.

    Checkpoint paths look like: checkpoints/gen_20/episode_65
    Walks up the path looking for 'gen_N' components.
    """
    import re
    for part in ckpt_path.parts:
        m = re.match(r"gen_(\d+)", part)
        if m:
            return int(m.group(1))
    return 0


def _write_live_state(
    gen_idx: int,
    total_generations: int,
    gen_results: list[dict],
    episodes_per_generation: int,
    env_config: dict | None = None,
    train_env: TradingEnv | None = None,
) -> None:
    """Write partial training state for live dashboard consumption.

    Called after each generation so the dashboard's 15-second polling
    can show progress during training (not just after the pipeline ends).
    Includes price data from the training env for the trend chart.

    Skipped when running under the meta-optimizer (HYDRA_META_TRIAL env var)
    to avoid clobbering the dashboard state from the main training run.
    """
    import os
    if os.environ.get("HYDRA_META_TRIAL"):
        return

    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    env_config = env_config or {}

    generations = []
    for gen in gen_results:
        gen_entry = {
            "generation": gen.get("generation", 0),
            "train_mean_reward": gen.get("train_mean_reward", 0),
            "eval_scores": gen.get("eval_scores", {}),
            "promoted": gen.get("promoted", []),
            "demoted": gen.get("demoted", []),
            "pool_size": gen.get("pool_size", 0),
            "diagnosis": gen.get("diagnosis"),
            "competition": gen.get("competition"),
            "conviction": gen.get("conviction"),
        }
        # P&L tracking data (per-agent returns, deployment, trades)
        if gen.get("agent_pnl"):
            gen_entry["agent_pnl"] = gen["agent_pnl"]
            gen_entry["best_return_pct"] = gen.get("best_return_pct", 0.0)
            gen_entry["mean_return_pct"] = gen.get("mean_return_pct", 0.0)
        generations.append(gen_entry)

    rewards = [g["train_mean_reward"] for g in generations]
    best_reward = rewards[-1] if rewards else 0
    first_reward = rewards[0] if rewards else 0

    # Extract price data from the training env for the trend direction chart.
    # Run one episode to capture close prices at each step.
    price_history = _extract_price_history(train_env)

    tickers = env_config.get("tickers", [])
    real_data = env_config.get("real_data", False)

    state = {
        "updated": datetime.now().isoformat(),
        "live": True,  # Signals dashboard this is partial/in-progress data
        "config": {
            "tickers": tickers,
            "num_stocks": env_config.get("num_stocks", 0),
            "num_generations": total_generations,
            "episodes_per_generation": episodes_per_generation,
            "real_data": real_data,
        },
        "summary": {
            "total_generations": gen_idx,
            "status": f"Training gen {gen_idx}/{total_generations}",
        },
        "metrics": {
            "mean_reward": best_reward,
            "reward_delta": best_reward - first_reward,
        },
        "generations": generations,
        "validation": {},
        "benchmark": {},
        "price_history": price_history,
        "trade_signals": [],
        "eval": {},
        "tasks": {
            "train_phase": {"status": "running", "duration_ms": 0},
        },
    }

    try:
        with open(_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
        logger.info(f"  Dashboard state updated (gen {gen_idx}/{total_generations})")
    except Exception as e:
        logger.warning(f"  Failed to write dashboard state: {e}")


def _extract_price_history(env: TradingEnv | None) -> list[dict]:
    """Extract price data from env by running one silent episode.

    Returns a list of {step, close_prices} dicts for the trend chart.
    Downsampled to ~200 points to keep the JSON small.
    """
    if env is None:
        return []
    try:
        import numpy as np
        obs, info = env.reset(seed=42)
        history = []
        step = 0
        while True:
            # Null action (hold) — we just want the price data
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            obs, reward, terminated, truncated, step_info = env.step(action)
            try:
                close_prices = env.state_builder.get_current_prices(
                    min(step, env.episode_bars - 1)
                ).tolist()
            except Exception:
                close_prices = []
            history.append({"step": step, "close_prices": close_prices})
            step += 1
            if terminated or truncated:
                break
        # Downsample to ~200 points
        n = max(1, len(history) // 200)
        return history[::n]
    except Exception as e:
        logger.warning(f"  Failed to extract price history: {e}")
        return []


def _create_intervention_hook():
    """Create a combined intervention hook for mid-training use.

    Chains three systems:
    1. Operations Monitor — anti-pattern detection + auto-fixes
    2. Regime injection — reads live regime from corp state (set by Geopolitics Expert)
    3. Risk Manager — circuit breaker enforcement (weight cuts)

    Returns a callback function compatible with PopulationTrainer.on_intervention.
    """
    try:
        from corp.agents.operations_monitor import OperationsMonitor
        from corp.state.corporation_state import CorporationState
        from corp.state.decision_log import DecisionLog

        state = CorporationState()
        decision_log = DecisionLog()
        monitor = OperationsMonitor(state, decision_log)

        # Optional: Risk Manager for circuit breaker enforcement
        risk_manager = None
        try:
            from corp.agents.risk_manager import RiskManager
            risk_manager = RiskManager(state, decision_log)
            logger.info("Risk Manager wired for mid-training circuit breaker enforcement")
        except Exception:
            pass

        logger.info("Operations Monitor wired for mid-training intervention")

        def hook(generation: int, generation_results: list) -> dict | None:
            result: dict = {}

            # 1. Operations Monitor: anti-pattern scan
            ops_result = monitor.on_generation_complete(
                generation, generation_results
            )
            if ops_result:
                result.update(ops_result)

            # 2. Regime injection from corp state (set by Geopolitics Expert)
            try:
                regime_data = state.get_regime()
                regime = regime_data.get("classification", "risk_on")
                if regime and regime != "unknown":
                    result["regime"] = regime
            except Exception:
                pass

            # 3. Risk Manager: circuit breaker enforcement
            if risk_manager:
                try:
                    risk_result = risk_manager.on_generation_complete(
                        generation, generation_results
                    )
                    if risk_result:
                        # Merge weight overrides
                        if "weight_overrides" in risk_result:
                            result["weight_overrides"] = risk_result["weight_overrides"]
                        if "alerts" in risk_result:
                            result.setdefault("alerts", []).extend(risk_result["alerts"])
                except Exception as e:
                    logger.debug(f"Risk Manager hook failed: {e}")

            return result if result else None

        return hook
    except Exception as e:
        logger.warning(f"Could not wire intervention hooks: {e}")
        return None
