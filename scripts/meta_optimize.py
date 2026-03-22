"""Meta-optimizer: Automated hyperparameter search for Hydra.

Uses Optuna TPE (Tree-structured Parzen Estimator) to run short training
experiments with different hyperparameter configurations, evaluate results,
and converge on high-performing settings.

Usage:
    python scripts/meta_optimize.py --trials 30 --gens-per-trial 5 --real-data
    python scripts/meta_optimize.py --resume   # continue from prior JSONL log
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root + parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import optuna
from optuna.samplers import TPESampler

from hydra.config.schema import HydraConfig
from hydra.pipeline.orchestrator import PipelineOrchestrator
from hydra.utils.logging import setup_logging

logger = logging.getLogger("hydra.meta_optimize")

LOG_DIR = Path(__file__).parent.parent / "logs"
JSONL_PATH = LOG_DIR / "meta_optimize.jsonl"
BEST_YAML_PATH = LOG_DIR / "meta_optimize_best.yaml"


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

def sample_hyperparams(trial: optuna.Trial) -> dict:
    """Sample hyperparameters from Optuna search space."""
    return {
        # Reward shaping
        "reward.pnl_bonus_weight": trial.suggest_float("reward.pnl_bonus_weight", 0.1, 5.0),
        "reward.drawdown_penalty": trial.suggest_float("reward.drawdown_penalty", 0.1, 3.0),
        "reward.holding_penalty": trial.suggest_float("reward.holding_penalty", 0.0, 0.5),
        "reward.sharpe_eta": trial.suggest_float("reward.sharpe_eta", 0.01, 0.2, log=True),
        "reward.reward_scale": trial.suggest_float("reward.reward_scale", 10.0, 500.0),
        # Environment risk limits
        "env.max_position_pct": trial.suggest_float("env.max_position_pct", 0.10, 0.50),
        "env.max_drawdown_pct": trial.suggest_float("env.max_drawdown_pct", 0.10, 0.40),
        # PPO hyperparams
        "ppo.learning_rate": trial.suggest_float("ppo.learning_rate", 1e-5, 1e-2, log=True),
        "ppo.ent_coef": trial.suggest_float("ppo.ent_coef", 0.001, 0.1, log=True),
        "ppo.clip_range": trial.suggest_float("ppo.clip_range", 0.05, 0.4),
        # SAC hyperparams
        "sac.learning_rate": trial.suggest_float("sac.learning_rate", 1e-5, 1e-2, log=True),
        # Network architecture
        "net_arch_size": trial.suggest_categorical("net_arch_size", [128, 256, 512]),
    }


def apply_params_to_config(config: HydraConfig, params: dict) -> HydraConfig:
    """Apply sampled hyperparameters to a HydraConfig."""
    # Reward
    config.reward.pnl_bonus_weight = params["reward.pnl_bonus_weight"]
    config.reward.drawdown_penalty = params["reward.drawdown_penalty"]
    config.reward.holding_penalty = params["reward.holding_penalty"]
    config.reward.sharpe_eta = params["reward.sharpe_eta"]
    config.reward.reward_scale = params["reward.reward_scale"]

    # Environment
    config.env.max_position_pct = params["env.max_position_pct"]
    config.env.max_drawdown_pct = params["env.max_drawdown_pct"]

    # Agent configs — update PPO and SAC defaults in the pool
    for agent_cfg in config.pool.agents:
        if agent_cfg.agent_type == "ppo":
            agent_cfg.learning_rate = params["ppo.learning_rate"]
            agent_cfg.ent_coef = params["ppo.ent_coef"]
            agent_cfg.clip_range = params["ppo.clip_range"]
        elif agent_cfg.agent_type == "sac":
            agent_cfg.learning_rate = params["sac.learning_rate"]

    # net_arch_size is applied via environment variable that agents read.
    # HYDRA_META_TRIAL suppresses dashboard state file writes so the
    # meta-optimizer doesn't clobber the main training run's dashboard.
    import os
    os.environ["HYDRA_NET_ARCH_SIZE"] = str(params["net_arch_size"])
    os.environ["HYDRA_META_TRIAL"] = "1"

    return config


# ---------------------------------------------------------------------------
# Fitness scoring
# ---------------------------------------------------------------------------

class FitnessTracker:
    """Track min/max across trials for normalization."""

    def __init__(self):
        self.reward_min = float("inf")
        self.reward_max = float("-inf")
        self.improvement_min = float("inf")
        self.improvement_max = float("-inf")
        self.drawdown_min = float("inf")
        self.drawdown_max = float("-inf")
        self.trades_min = float("inf")
        self.trades_max = float("-inf")

    def update(self, reward: float, improvement: float, drawdown: float, trades: float):
        self.reward_min = min(self.reward_min, reward)
        self.reward_max = max(self.reward_max, reward)
        self.improvement_min = min(self.improvement_min, improvement)
        self.improvement_max = max(self.improvement_max, improvement)
        self.drawdown_min = min(self.drawdown_min, drawdown)
        self.drawdown_max = max(self.drawdown_max, drawdown)
        self.trades_min = min(self.trades_min, trades)
        self.trades_max = max(self.trades_max, trades)

    def normalize(self, value: float, vmin: float, vmax: float) -> float:
        if vmax - vmin < 1e-10:
            return 0.5
        return (value - vmin) / (vmax - vmin)

    def compute_fitness(
        self, reward: float, improvement: float, drawdown: float, trades: float
    ) -> float:
        self.update(reward, improvement, drawdown, trades)
        return (
            0.40 * self.normalize(reward, self.reward_min, self.reward_max)
            + 0.25 * self.normalize(improvement, self.improvement_min, self.improvement_max)
            + 0.20 * (1.0 - self.normalize(drawdown, self.drawdown_min, self.drawdown_max))
            + 0.15 * self.normalize(trades, self.trades_min, self.trades_max)
        )


def extract_metrics(results: dict) -> dict:
    """Extract key metrics from pipeline results."""
    train_result = results.get("train_phase", {})
    training_data = train_result.get("training_results", {})
    generations = training_data.get("generations", [])

    if not generations:
        return {
            "best_eval_reward": -1000,
            "reward_improvement": 0,
            "max_drawdown": 1.0,
            "total_trades": 0,
            "num_generations": 0,
        }

    # Best eval reward across all generations (mean of top agent per gen)
    eval_rewards = []
    for gen in generations:
        scores = gen.get("eval_scores", {})
        if scores:
            eval_rewards.append(max(scores.values()))

    best_eval = max(eval_rewards) if eval_rewards else -1000
    first_eval = eval_rewards[0] if eval_rewards else -1000
    improvement = best_eval - first_eval

    # Training mean rewards for trend
    train_rewards = [g.get("train_mean_reward", -1000) for g in generations]

    # Max drawdown from diagnostics
    max_dd = 0.0
    for gen in generations:
        diag = gen.get("diagnosis") or {}
        issues = diag.get("issues", [])
        for issue in issues:
            if "drawdown" in str(issue).lower():
                max_dd = max(max_dd, 0.15)  # At least moderate if flagged

    # Total trades from competition results
    total_trades = 0
    for gen in generations:
        comp = gen.get("competition") or {}
        for agent_score in comp.get("agent_scores", []):
            total_trades += agent_score.get("trades", 0)

    # If no competition data, estimate from pool activity
    if total_trades == 0:
        total_trades = len(generations) * 50  # rough estimate

    return {
        "best_eval_reward": float(best_eval),
        "first_eval_reward": float(first_eval),
        "reward_improvement": float(improvement),
        "final_train_reward": float(train_rewards[-1]) if train_rewards else -1000,
        "max_drawdown": float(max_dd),
        "total_trades": int(total_trades),
        "num_generations": len(generations),
    }


# ---------------------------------------------------------------------------
# JSONL logging + resume
# ---------------------------------------------------------------------------

def log_trial(trial_id: int, params: dict, fitness: float, metrics: dict, duration_s: float):
    """Append trial result to JSONL log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "trial_id": trial_id,
        "timestamp": datetime.now().isoformat(),
        "params": params,
        "fitness": fitness,
        "metrics": metrics,
        "duration_s": round(duration_s, 1),
    }
    with open(JSONL_PATH, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def load_prior_trials() -> list[dict]:
    """Load completed trials from JSONL log."""
    if not JSONL_PATH.exists():
        return []
    trials = []
    with open(JSONL_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    trials.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return trials


def save_best_config(study: optuna.Study, base_config: HydraConfig):
    """Save the best trial's config as a YAML file."""
    best = study.best_trial
    config = HydraConfig()
    apply_params_to_config(config, best.params)
    config.to_yaml(BEST_YAML_PATH)
    logger.info(f"Best config saved to {BEST_YAML_PATH}")


# ---------------------------------------------------------------------------
# Alpaca config loader (same as train.py)
# ---------------------------------------------------------------------------

def _load_alpaca_config() -> dict | None:
    env_path = Path(__file__).parent.parent.parent / "trading_agents" / ".env"
    if not env_path.exists():
        return None
    config = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key == "ALPACA_API_KEY" and value:
                config["api_key"] = value
            elif key == "ALPACA_SECRET_KEY" and value:
                config["secret_key"] = value
            elif key == "ALPACA_BASE_URL" and value:
                config["base_url"] = value
    if "api_key" in config and "secret_key" in config:
        return config
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hydra Meta-Optimizer (Optuna TPE)")
    parser.add_argument("--trials", type=int, default=30, help="Number of trials to run")
    parser.add_argument("--gens-per-trial", type=int, default=5, help="Generations per trial")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per generation")
    parser.add_argument("--real-data", action="store_true", help="Use real Alpaca data")
    parser.add_argument("--resume", action="store_true", help="Resume from prior JSONL log")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Optuna sampler")
    parser.add_argument("--log-level", type=str, default="WARNING",
                        help="Log level (WARNING to reduce noise per trial)")
    args = parser.parse_args()

    setup_logging(args.log_level)
    # Keep meta-optimizer logs visible
    logging.getLogger("hydra.meta_optimize").setLevel(logging.INFO)

    # Alpaca setup
    alpaca_config = None
    if args.real_data:
        alpaca_config = _load_alpaca_config()
        if alpaca_config is None:
            print("ERROR: --real-data requires Alpaca credentials in trading_agents/.env")
            sys.exit(1)

    # Create Optuna study
    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Suppress Optuna's own logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Resume: inject prior trials
    prior_trials = []
    if args.resume:
        prior_trials = load_prior_trials()
        if prior_trials:
            print(f"Resuming: loaded {len(prior_trials)} prior trials from {JSONL_PATH}")
            for pt in prior_trials:
                try:
                    study.add_trial(
                        optuna.trial.create_trial(
                            params=pt["params"],
                            distributions={
                                "reward.pnl_bonus_weight": optuna.distributions.FloatDistribution(0.1, 5.0),
                                "reward.drawdown_penalty": optuna.distributions.FloatDistribution(0.1, 3.0),
                                "reward.holding_penalty": optuna.distributions.FloatDistribution(0.0, 0.5),
                                "reward.sharpe_eta": optuna.distributions.FloatDistribution(0.01, 0.2, log=True),
                                "reward.reward_scale": optuna.distributions.FloatDistribution(10.0, 500.0),
                                "env.max_position_pct": optuna.distributions.FloatDistribution(0.10, 0.50),
                                "env.max_drawdown_pct": optuna.distributions.FloatDistribution(0.10, 0.40),
                                "ppo.learning_rate": optuna.distributions.FloatDistribution(1e-5, 1e-2, log=True),
                                "ppo.ent_coef": optuna.distributions.FloatDistribution(0.001, 0.1, log=True),
                                "ppo.clip_range": optuna.distributions.FloatDistribution(0.05, 0.4),
                                "sac.learning_rate": optuna.distributions.FloatDistribution(1e-5, 1e-2, log=True),
                                "net_arch_size": optuna.distributions.CategoricalDistribution([128, 256, 512]),
                            },
                            values=[pt["fitness"]],
                        )
                    )
                except Exception as e:
                    logger.warning(f"Skipping prior trial {pt.get('trial_id')}: {e}")
        else:
            print("No prior trials found, starting fresh")

    # Fitness tracker for normalization
    fitness_tracker = FitnessTracker()

    # Seed the tracker with prior trial metrics
    for pt in prior_trials:
        m = pt.get("metrics", {})
        fitness_tracker.update(
            m.get("best_eval_reward", -1000),
            m.get("reward_improvement", 0),
            m.get("max_drawdown", 1.0),
            m.get("total_trades", 0),
        )

    trials_to_run = args.trials - len(prior_trials)
    if trials_to_run <= 0:
        print(f"Already have {len(prior_trials)} trials >= requested {args.trials}")
        _print_summary(study)
        return

    print(f"\n{'='*60}")
    print(f"  Hydra Meta-Optimizer")
    print(f"  Trials: {trials_to_run} new + {len(prior_trials)} prior = {args.trials} total")
    print(f"  Generations per trial: {args.gens_per_trial}")
    print(f"  Episodes per generation: {args.episodes}")
    print(f"  Data: {'Real Alpaca' if args.real_data else 'Synthetic'}")
    print(f"{'='*60}\n")

    trial_counter = len(prior_trials)

    def objective(trial: optuna.Trial) -> float:
        nonlocal trial_counter
        trial_counter += 1
        trial_num = trial_counter

        # Sample hyperparameters
        params = sample_hyperparams(trial)

        print(f"\n--- Trial {trial_num}/{args.trials} ---")
        print(f"  pnl_bonus={params['reward.pnl_bonus_weight']:.2f}, "
              f"dd_pen={params['reward.drawdown_penalty']:.2f}, "
              f"lr_ppo={params['ppo.learning_rate']:.1e}, "
              f"lr_sac={params['sac.learning_rate']:.1e}, "
              f"arch={params['net_arch_size']}")

        # Build config
        config = HydraConfig()
        config.training.num_generations = args.gens_per_trial
        config.training.episodes_per_generation = args.episodes
        config.training.checkpoint_dir = f"checkpoints/meta_trial_{trial_num}"
        config.seed = args.seed + trial_num
        apply_params_to_config(config, params)

        # Run experiment
        t0 = time.time()
        try:
            orchestrator = PipelineOrchestrator(
                config,
                alpaca_config=alpaca_config,
                use_real_data=args.real_data,
            )
            results = orchestrator.run()
        except Exception as e:
            logger.error(f"  Trial {trial_num} FAILED: {e}")
            duration = time.time() - t0
            log_trial(trial_num, params, float("-inf"), {"error": str(e)}, duration)
            return float("-inf")

        duration = time.time() - t0

        # Extract metrics and compute fitness
        metrics = extract_metrics(results)
        fitness = fitness_tracker.compute_fitness(
            metrics["best_eval_reward"],
            metrics["reward_improvement"],
            metrics["max_drawdown"],
            metrics["total_trades"],
        )

        print(f"  Result: fitness={fitness:.4f}, "
              f"best_reward={metrics['best_eval_reward']:.1f}, "
              f"improvement={metrics['reward_improvement']:.1f}, "
              f"duration={duration:.0f}s")

        # Log to JSONL
        log_trial(trial_num, params, fitness, metrics, duration)

        return fitness

    # Run optimization
    study.optimize(objective, n_trials=trials_to_run)

    # Save best config
    try:
        save_best_config(study, HydraConfig())
    except Exception as e:
        logger.warning(f"Failed to save best config: {e}")

    _print_summary(study)


def _print_summary(study: optuna.Study):
    """Print top-5 trial summary."""
    print(f"\n{'='*60}")
    print(f"  META-OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total trials: {len(study.trials)}")

    if not study.trials:
        return

    best = study.best_trial
    print(f"\n  BEST TRIAL (#{best.number + 1}):")
    print(f"    Fitness: {best.value:.4f}")
    print(f"    Parameters:")
    for k, v in sorted(best.params.items()):
        if isinstance(v, float):
            print(f"      {k}: {v:.6g}")
        else:
            print(f"      {k}: {v}")

    # Top 5
    sorted_trials = sorted(
        [t for t in study.trials if t.value is not None and t.value > float("-inf")],
        key=lambda t: t.value,
        reverse=True,
    )
    if len(sorted_trials) > 1:
        print(f"\n  TOP 5 TRIALS:")
        for i, t in enumerate(sorted_trials[:5]):
            print(f"    #{t.number + 1}: fitness={t.value:.4f}"
                  f"  lr_ppo={t.params.get('ppo.learning_rate', 0):.1e}"
                  f"  pnl_bonus={t.params.get('reward.pnl_bonus_weight', 0):.2f}"
                  f"  arch={t.params.get('net_arch_size', 0)}")

    print(f"\n  Best config saved to: {BEST_YAML_PATH}")
    print(f"  Full log: {JSONL_PATH}")
    print(f"\n  To run full training with best config:")
    print(f"    python scripts/train.py --config {BEST_YAML_PATH} --real-data --generations 60")
    print()


if __name__ == "__main__":
    main()
