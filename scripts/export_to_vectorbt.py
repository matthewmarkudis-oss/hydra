"""CLI entry point: Export RL strategy signals for VectorBT backtesting.

Usage: python scripts/export_to_vectorbt.py --checkpoint path/to/checkpoint --output signals.json
Backtesting only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Export RL signals for VectorBT")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--agent", type=str, default=None, help="Agent name (default: best)")
    parser.add_argument("--output", type=str, default="signals.json", help="Output file")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes to export")
    parser.add_argument("--threshold", type=float, default=0.3, help="Action threshold for signals")
    parser.add_argument("--config", type=str, default=None, help="Config YAML")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    from hydra.agents.agent_pool import AgentPool
    from hydra.config.schema import HydraConfig
    from hydra.envs.trading_env import TradingEnv
    from hydra.pipeline.eval_phase import export_signals
    from hydra.utils.logging import setup_logging

    setup_logging(args.log_level)

    config = HydraConfig.from_yaml(args.config) if args.config else HydraConfig()

    # Load pool
    pool = AgentPool()
    pool.load(Path(args.checkpoint) / "pool")

    # Select agent
    if args.agent:
        agent = pool.get(args.agent)
        if agent is None:
            print(f"Agent '{args.agent}' not found. Available: {pool.agent_names}")
            sys.exit(1)
    else:
        ranked = pool.get_ranked_agents()
        if ranked:
            agent = pool.get(ranked[0][0])
        else:
            agent = pool.get_all()[0] if pool.size > 0 else None

    if agent is None:
        print("No agents available")
        sys.exit(1)

    print(f"Exporting signals from agent '{agent.name}'")

    # Create environment
    env = TradingEnv(
        num_stocks=config.env.num_stocks,
        episode_bars=config.env.episode_bars,
        initial_cash=config.env.initial_cash,
        seed=config.seed,
    )

    # Export signals
    signals = export_signals(agent, env, args.episodes, args.threshold)

    # Write to file
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(signals, f, indent=2, default=str)

    total_signals = sum(len(s) for s in signals.values())
    print(f"Exported {total_signals} signals across {len(signals)} tickers to {output_path}")


if __name__ == "__main__":
    main()
