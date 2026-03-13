"""CLI entry point: Full training pipeline.

Usage: python scripts/train.py [--config path/to/config.yaml] [--synthetic]
Backtesting and training only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Hydra RL Training Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--generations", type=int, default=None, help="Override num_generations")
    parser.add_argument("--episodes", type=int, default=None, help="Override episodes_per_generation")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    args = parser.parse_args()

    from hydra.config.schema import HydraConfig
    from hydra.pipeline.orchestrator import PipelineOrchestrator
    from hydra.utils.logging import setup_logging

    setup_logging(args.log_level)

    # Load config
    if args.config:
        config = HydraConfig.from_yaml(args.config)
    else:
        config = HydraConfig()

    # Apply overrides
    if args.generations is not None:
        config.training.num_generations = args.generations
    if args.episodes is not None:
        config.training.episodes_per_generation = args.episodes
    if args.seed is not None:
        config.seed = args.seed

    # Run pipeline
    orchestrator = PipelineOrchestrator(config)
    results = orchestrator.run()

    # Print summary
    summary = orchestrator.get_summary()
    print("\n=== Training Complete ===")
    print(f"Total tasks: {len(summary.get('tasks', {}))}")
    for task_name, task_info in summary.get("tasks", {}).items():
        status = task_info.get("status", "unknown")
        duration = task_info.get("duration_ms", 0)
        print(f"  {task_name}: {status} ({duration:.0f}ms)")

    passed = summary.get("passed_agents", [])
    if passed:
        print(f"\nPassed validation: {passed}")
    else:
        print("\nNo agents passed validation")


if __name__ == "__main__":
    main()
