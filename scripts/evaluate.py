"""CLI entry point: Evaluate trained agents.

Usage: python scripts/evaluate.py --checkpoint path/to/checkpoint [--episodes 20]
Backtesting only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Hydra Agent Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--episodes", type=int, default=20, help="Number of eval episodes")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    args = parser.parse_args()

    from hydra.agents.agent_pool import AgentPool
    from hydra.config.schema import HydraConfig
    from hydra.envs.trading_env import TradingEnv
    from hydra.utils.logging import setup_logging

    setup_logging(args.log_level)

    config = HydraConfig.from_yaml(args.config) if args.config else HydraConfig()

    # Load pool from checkpoint
    pool = AgentPool()
    pool.load(Path(args.checkpoint) / "pool")

    print(f"Loaded {pool.size} agents from {args.checkpoint}")

    # Create eval environment (synthetic)
    env = TradingEnv(
        num_stocks=config.env.num_stocks,
        episode_bars=config.env.episode_bars,
        initial_cash=config.env.initial_cash,
        seed=config.seed,
    )

    # Evaluate each agent
    import numpy as np

    for agent in pool.get_all():
        rewards = []
        for _ in range(args.episodes):
            obs, info = env.reset()
            total_reward = 0.0
            while True:
                action = agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, step_info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            rewards.append(total_reward)

        mean_r = np.mean(rewards)
        std_r = np.std(rewards)
        print(f"  {agent.name} ({agent.__class__.__name__}): "
              f"reward={mean_r:.4f} +/- {std_r:.4f}")


if __name__ == "__main__":
    main()
