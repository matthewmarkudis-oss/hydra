"""Multi-seed training curriculum.

Runs the full Hydra pipeline across multiple seeds to build robust agents
that generalize across different market scenarios. Each seed produces a
different synthetic market environment, exposing agents to diverse
price dynamics, volatility regimes, and trend patterns.

The curriculum progressively increases difficulty:
  Phase 1 (Seeds 1-3):  Warmup — few generations, learn basic dynamics
  Phase 2 (Seeds 4-7):  Main   — more generations, deeper learning
  Phase 3 (Seeds 8-10): Stress — high volatility seeds, test robustness

Results are aggregated across all seeds to identify agents that
consistently perform well regardless of market conditions.

Usage:
  python scripts/train_curriculum.py
  python scripts/train_curriculum.py --phases 1 2
  python scripts/train_curriculum.py --seeds 42 43 44 45
  python scripts/train_curriculum.py --generations 5 --episodes 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class SeedRun:
    """Results from a single seed training run."""
    seed: int
    phase: str
    generations: int
    episodes: int
    duration_secs: float = 0.0
    passed_agents: list[str] = field(default_factory=list)
    agent_scores: dict[str, float] = field(default_factory=dict)
    mean_reward: float = 0.0
    best_sharpe: float = 0.0
    error: str | None = None


@dataclass
class CurriculumConfig:
    """Configuration for the multi-seed curriculum."""
    phases: dict[str, dict] = field(default_factory=lambda: {
        "warmup": {
            "seeds": [42, 43, 44],
            "generations": 3,
            "episodes": 15,
            "description": "Warmup: learn basic market dynamics",
        },
        "main": {
            "seeds": [100, 101, 102, 103],
            "generations": 5,
            "episodes": 20,
            "description": "Main: deeper learning across diverse markets",
        },
        "stress": {
            "seeds": [200, 201, 202],
            "generations": 5,
            "episodes": 25,
            "description": "Stress: high-volatility regime testing",
        },
    })


def run_single_seed(
    seed: int,
    phase_name: str,
    generations: int,
    episodes: int,
    log_level: str = "INFO",
    tensorboard_base: str = "logs/tensorboard",
) -> SeedRun:
    """Run a single training pipeline with the given seed."""
    from hydra.config.schema import HydraConfig
    from hydra.pipeline.orchestrator import PipelineOrchestrator
    from hydra.utils.logging import setup_logging

    setup_logging(log_level)

    result = SeedRun(
        seed=seed,
        phase=phase_name,
        generations=generations,
        episodes=episodes,
    )

    try:
        config = HydraConfig()
        config.seed = seed
        config.training.num_generations = generations
        config.training.episodes_per_generation = episodes

        orchestrator = PipelineOrchestrator(config)

        start = time.time()
        orchestrator.run()
        result.duration_secs = time.time() - start

        summary = orchestrator.get_summary()
        result.passed_agents = summary.get("passed_agents", [])

        # Extract results from the pipeline run
        pipeline_results = orchestrator._results or {}

        # Get train phase metrics
        train_data = pipeline_results.get("train_phase", {})
        if isinstance(train_data, dict):
            metrics_summary = train_data.get("metrics_summary", {})
            result.mean_reward = metrics_summary.get("mean_reward", 0.0)
            result.win_rate = metrics_summary.get("win_rate", 0.0)

        # Get validation results for per-agent scores
        val_data = pipeline_results.get("validation", {})
        if isinstance(val_data, dict):
            for agent_name, agent_info in val_data.get("agent_results", {}).items():
                if isinstance(agent_info, dict):
                    sharpe = agent_info.get("sharpe", 0.0)
                    result.agent_scores[agent_name] = sharpe
                    if sharpe > result.best_sharpe:
                        result.best_sharpe = sharpe

    except Exception as e:
        result.error = str(e)

    return result


def print_seed_result(run: SeedRun) -> None:
    """Print results for a single seed run."""
    status = "PASS" if run.passed_agents else "FAIL"
    if run.error:
        status = "ERROR"

    print(f"\n  Seed {run.seed} ({run.phase}): {status}")
    print(f"    Duration: {run.duration_secs:.1f}s")
    print(f"    Generations: {run.generations}, Episodes: {run.episodes}")

    if run.error:
        print(f"    Error: {run.error}")
    else:
        print(f"    Mean reward: {run.mean_reward:.2f}")
        n_passed = len(run.passed_agents)
        print(f"    Agents passed validation: {n_passed}")
        if run.passed_agents:
            print(f"    Passed: {', '.join(run.passed_agents)}")


def print_curriculum_summary(runs: list[SeedRun]) -> None:
    """Print aggregate summary across all seed runs."""
    print("\n" + "=" * 60)
    print("CURRICULUM SUMMARY")
    print("=" * 60)

    total_seeds = len(runs)
    successful = [r for r in runs if r.error is None]
    failed = [r for r in runs if r.error is not None]
    total_time = sum(r.duration_secs for r in runs)

    print(f"\nSeeds completed: {len(successful)}/{total_seeds}")
    print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} min)")

    if failed:
        print(f"\nFailed seeds: {[r.seed for r in failed]}")

    if not successful:
        print("\nNo successful runs to analyze.")
        return

    # Aggregate pass rates
    seeds_with_passes = [r for r in successful if r.passed_agents]
    print(f"Seeds with passing agents: {len(seeds_with_passes)}/{len(successful)}")

    # Count how often each base agent TYPE has at least one passing variant per seed
    agent_pass_counts: dict[str, int] = {}
    for r in successful:
        # Deduplicate: count each base agent at most once per seed
        bases_seen = set()
        for agent_name in r.passed_agents:
            base = agent_name.split("_gen")[0]
            bases_seen.add(base)
        for base in bases_seen:
            agent_pass_counts[base] = agent_pass_counts.get(base, 0) + 1

    if agent_pass_counts:
        print("\nAgent consistency (seeds where at least one variant passes):")
        for agent, count in sorted(agent_pass_counts.items(), key=lambda x: -x[1]):
            pct = count / len(successful) * 100
            bar = "#" * int(pct / 5)
            print(f"  {agent:20s}  {count}/{len(successful)} seeds ({pct:.0f}%)  {bar}")

    # Reward progression by phase
    phases_seen = []
    for r in successful:
        if r.phase not in phases_seen:
            phases_seen.append(r.phase)

    print("\nReward by phase:")
    for phase in phases_seen:
        phase_runs = [r for r in successful if r.phase == phase]
        rewards = [r.mean_reward for r in phase_runs]
        import numpy as np
        mean_r = np.mean(rewards)
        std_r = np.std(rewards)
        print(f"  {phase:10s}  mean_reward={mean_r:+.2f} +/- {std_r:.2f}  (n={len(phase_runs)})")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed training curriculum for Hydra agents"
    )
    parser.add_argument(
        "--phases", nargs="+", default=None,
        help="Which phases to run (warmup, main, stress). Default: all"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Override: run these specific seeds (ignores phases)"
    )
    parser.add_argument(
        "--generations", type=int, default=None,
        help="Override generations per seed"
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Override episodes per generation"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        help="Log level (DEBUG, INFO, WARNING)"
    )
    parser.add_argument(
        "--output", type=str, default="logs/curriculum_results.json",
        help="Path to save results JSON"
    )
    args = parser.parse_args()

    curriculum = CurriculumConfig()
    all_runs: list[SeedRun] = []

    print("=" * 60)
    print("HYDRA MULTI-SEED TRAINING CURRICULUM")
    print("=" * 60)

    if args.seeds:
        # Custom seed list — single flat phase
        gens = args.generations or 3
        eps = args.episodes or 15
        print(f"\nCustom seeds: {args.seeds}")
        print(f"Generations: {gens}, Episodes: {eps}")

        for i, seed in enumerate(args.seeds):
            print(f"\n--- Seed {seed} ({i+1}/{len(args.seeds)}) ---")
            run = run_single_seed(
                seed=seed,
                phase_name="custom",
                generations=gens,
                episodes=eps,
                log_level=args.log_level,
            )
            all_runs.append(run)
            print_seed_result(run)

    else:
        # Run curriculum phases
        phases_to_run = args.phases or list(curriculum.phases.keys())
        total_seeds = sum(
            len(curriculum.phases[p]["seeds"])
            for p in phases_to_run if p in curriculum.phases
        )
        seed_num = 0

        for phase_name in phases_to_run:
            if phase_name not in curriculum.phases:
                print(f"\nUnknown phase: {phase_name}, skipping")
                continue

            phase = curriculum.phases[phase_name]
            gens = args.generations or phase["generations"]
            eps = args.episodes or phase["episodes"]

            print(f"\n{'='*40}")
            print(f"PHASE: {phase_name.upper()}")
            print(f"  {phase['description']}")
            print(f"  Seeds: {phase['seeds']}")
            print(f"  Generations: {gens}, Episodes: {eps}")
            print(f"{'='*40}")

            for seed in phase["seeds"]:
                seed_num += 1
                print(f"\n--- Seed {seed} ({seed_num}/{total_seeds}) ---")
                run = run_single_seed(
                    seed=seed,
                    phase_name=phase_name,
                    generations=gens,
                    episodes=eps,
                    log_level=args.log_level,
                )
                all_runs.append(run)
                print_seed_result(run)

    # Print aggregate summary
    print_curriculum_summary(all_runs)

    # Save results to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_json = {
        "runs": [
            {
                "seed": r.seed,
                "phase": r.phase,
                "generations": r.generations,
                "episodes": r.episodes,
                "duration_secs": r.duration_secs,
                "passed_agents": r.passed_agents,
                "agent_scores": r.agent_scores,
                "mean_reward": r.mean_reward,
                "best_sharpe": r.best_sharpe,
                "error": r.error,
            }
            for r in all_runs
        ],
    }

    output_path.write_text(json.dumps(results_json, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
