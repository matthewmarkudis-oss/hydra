"""CLI entry point: Full training pipeline.

Usage: python scripts/train.py [--config path/to/config.yaml] [--synthetic]
       python scripts/train.py --real-data --tickers XLK,XLF,XLE
Backtesting and training only.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add TradingAgents parent so `import trading_agents` works
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

SECTOR_ETFS = "XLK,XLF,XLE,XLV,XLI,XLU,XLP,XLY,XLB,XLRE"


def _load_alpaca_config() -> dict | None:
    """Load Alpaca credentials from trading_agents/.env."""
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


def main():
    parser = argparse.ArgumentParser(description="Hydra RL Training Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--real-data", action="store_true", help="Use real Alpaca market data")
    parser.add_argument("--tickers", type=str, default=None,
                        help=f"Comma-separated tickers (default: {SECTOR_ETFS})")
    parser.add_argument("--generations", type=int, default=None, help="Override num_generations")
    parser.add_argument("--episodes", type=int, default=None, help="Override episodes_per_generation")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint dir (e.g., checkpoints/gen_20/episode_100)")
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

    # Apply ticker override
    if args.tickers:
        config.data.tickers = [t.strip() for t in args.tickers.split(",")]
        config.env.num_stocks = len(config.data.tickers)

    # Real data setup
    alpaca_config = None
    use_real_data = args.real_data
    if use_real_data:
        alpaca_config = _load_alpaca_config()
        if alpaca_config is None:
            print("ERROR: --real-data requires Alpaca credentials in trading_agents/.env")
            sys.exit(1)
        # Default to sector ETFs if no tickers specified
        if not args.tickers:
            config.data.tickers = [t.strip() for t in SECTOR_ETFS.split(",")]
            config.env.num_stocks = len(config.data.tickers)
        print(f"Using REAL Alpaca data for {len(config.data.tickers)} tickers: "
              f"{', '.join(config.data.tickers)}")

    # Run pipeline
    orchestrator = PipelineOrchestrator(
        config,
        alpaca_config=alpaca_config,
        use_real_data=use_real_data,
        resume_checkpoint=args.resume,
    )
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

    # Save results to JSON for dashboard consumption
    _save_dashboard_state(config, results, summary, use_real_data)


def _save_dashboard_state(config, results, summary, use_real_data):
    """Save training results to JSON for the Hydra dashboard."""
    import json
    from datetime import datetime

    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    state_file = logs_dir / "hydra_training_state.json"

    # Extract training results
    train_result = results.get("train_phase", {})
    training_data = train_result.get("training_results", {})
    eval_result = results.get("eval_phase", {})
    validation_result = results.get("validation", {})
    metrics_summary = train_result.get("metrics_summary", {})

    # Build generation history
    generations = []
    for gen in training_data.get("generations", []):
        generations.append({
            "generation": gen.get("generation", 0),
            "train_mean_reward": gen.get("train_mean_reward", 0),
            "eval_scores": gen.get("eval_scores", {}),
            "promoted": gen.get("promoted", []),
            "demoted": gen.get("demoted", []),
            "pool_size": gen.get("pool_size", 0),
            "diagnosis": gen.get("diagnosis"),
            "competition": gen.get("competition"),
            "conviction": gen.get("conviction"),
        })

    # Build agent validation results
    agent_results = {}
    for name, r in validation_result.get("agent_results", {}).items():
        agent_results[name] = {
            "sharpe": r.get("sharpe", 0),
            "sharpe_ci_low": r.get("sharpe_ci_low", 0),
            "sharpe_ci_high": r.get("sharpe_ci_high", 0),
            "psr": r.get("psr", 0),
            "dsr": r.get("dsr", 0),
            "max_drawdown": r.get("max_drawdown", 0),
            "win_rate": r.get("win_rate", 0),
            "profit_factor": r.get("profit_factor", 0),
            "wfe": r.get("wfe", 0),
            "wfe_diagnosis": r.get("wfe_diagnosis", {}),
            "fitness_score": r.get("fitness_score", 0),
            "fitness_breakdown": r.get("fitness_breakdown", {}),
            "total_return": r.get("total_return", 0),
            "total_trades": r.get("total_trades", 0),
            "consistency": r.get("consistency", 0),
            "calibration_verdict": r.get("calibration_verdict", "N/A"),
            "passed": r.get("passed", False),
        }

    # Benchmark data from SPY buy-and-hold
    benchmark = validation_result.get("benchmark", {})

    # Price history from best agent's first validation episode
    price_history = validation_result.get("price_history", [])
    trade_signals = validation_result.get("trade_signals", [])

    state = {
        "updated": datetime.now().isoformat(),
        "config": {
            "tickers": config.data.tickers,
            "num_stocks": config.env.num_stocks,
            "num_generations": config.training.num_generations,
            "episodes_per_generation": config.training.episodes_per_generation,
            "real_data": use_real_data,
            "seed": config.seed,
        },
        "summary": {
            "total_generations": training_data.get("total_generations", 0),
            "final_rankings": training_data.get("final_rankings", {}),
            "passed_agents": validation_result.get("passed_agents", []),
            "thresholds": validation_result.get("thresholds", {}),
        },
        "metrics": metrics_summary,
        "generations": generations,
        "validation": agent_results,
        "benchmark": benchmark,
        "price_history": price_history,
        "trade_signals": trade_signals,
        "eval": {
            name: metrics
            for name, metrics in eval_result.get("rl_eval", {}).items()
        },
        "tasks": {
            name: {
                "status": info.get("status", "unknown"),
                "duration_ms": info.get("duration_ms", 0),
            }
            for name, info in summary.get("tasks", {}).items()
        },
    }

    with open(state_file, "w") as f:
        json.dump(state, f, indent=2, default=str)
    print(f"\nDashboard state saved to {state_file}")


if __name__ == "__main__":
    main()
