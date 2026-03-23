"""Launch forward test for graduated agents.

Reads the approved graduation proposal from corp state,
loads agent checkpoints, initializes the broker and sub-accounts,
and starts the ForwardTestRunner.

Usage:
    python scripts/launch_forward_test.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Ensure project roots are on sys.path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from dotenv import load_dotenv

load_dotenv(ROOT.parent / "trading_agents" / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "logs" / "forward_test.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("forward_test.launch")


def main():
    from corp.state.corporation_state import CorporationState
    from hydra.forward_test.capital_allocator import Allocation
    from hydra.forward_test.runner import ForwardTestRunner
    from hydra.forward_test.tracker import ForwardTestTracker

    # 1. Read approved graduation proposal
    state = CorporationState()
    all_proposals = state._read_state()["proposals"]
    graduation = None
    for p in reversed(all_proposals):
        if p.get("type") == "graduation" and p.get("status") == "approved":
            graduation = p
            break

    if not graduation:
        logger.error("No approved graduation proposal found. Run graduation_manager first.")
        sys.exit(1)

    candidates = graduation["candidates"]
    alloc_list = graduation["allocations"]
    ft_config = graduation["forward_test_config"]

    logger.info("=== Forward Test Launch ===")
    logger.info(f"Candidates: {[c['name'] for c in candidates]}")
    for a in alloc_list:
        logger.info(f"  {a['agent_name']}: {a['weight']:.0%} (${a['capital']:,.0f})")
    logger.info(f"Duration: {ft_config['duration_days']} days")
    logger.info(f"Capital: ${ft_config['initial_capital']:,.0f}")

    # 2. Load agent checkpoints
    from hydra.agents.ppo_agent import PPOAgent
    from hydra.agents.td3_agent import TD3Agent
    from hydra.agents.recurrent_ppo_agent import RecurrentPPOAgent
    from hydra.agents.static_agent import StaticAgent

    # Find latest checkpoint
    ckpt_base = ROOT / "checkpoints"
    gen_dirs = sorted(ckpt_base.glob("gen_*"), key=lambda p: int(p.name.split("_")[1]))
    if not gen_dirs:
        logger.error("No checkpoint directories found.")
        sys.exit(1)

    latest_gen = gen_dirs[-1]
    episode_dirs = sorted(latest_gen.glob("episode_*"))
    if not episode_dirs:
        logger.error(f"No episode checkpoints in {latest_gen}")
        sys.exit(1)

    ckpt_dir = episode_dirs[-1]
    logger.info(f"Loading agents from checkpoint: {ckpt_dir}")

    # Read pool metadata for obs/action dims
    meta_path = ckpt_dir / "pool_metadata.json"
    with open(meta_path) as f:
        pool_meta = json.load(f)

    agents = []
    for candidate in candidates:
        name = candidate["name"]
        if name not in pool_meta.get("agents", {}):
            logger.warning(f"Agent '{name}' not in checkpoint metadata, skipping.")
            continue

        info = pool_meta["agents"][name]
        obs_dim = info["obs_dim"]
        action_dim = info["action_dim"]
        agent_type = info["type"]

        # Create a StaticAgent snapshot for inference
        agent = StaticAgent(name, obs_dim, action_dim)
        model_path = ckpt_dir / name / "model"

        if model_path.exists() or (ckpt_dir / name).exists():
            try:
                agent.load(model_path)
                logger.info(f"Loaded agent '{name}' ({agent_type}) from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load agent '{name}': {e}")
                continue
        else:
            logger.error(f"No model file for agent '{name}' at {model_path}")
            continue

        agents.append(agent)

    if not agents:
        logger.error("No agents loaded successfully. Aborting.")
        sys.exit(1)

    # 3. Build allocations
    allocations = []
    for a in alloc_list:
        if any(agent.name == a["agent_name"] for agent in agents):
            allocations.append(Allocation(
                agent_name=a["agent_name"],
                sharpe=a["sharpe"],
                capital=a["capital"],
                weight=a["weight"],
                passed_validation=True,
            ))

    # 4. Initialize broker
    from trading_agents.utils.broker import AlpacaBroker

    broker_config = {
        "api_key": os.environ.get("ALPACA_API_KEY", ""),
        "secret_key": os.environ.get("ALPACA_SECRET_KEY", ""),
        "base_url": os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
        "paper": True,
    }

    if not broker_config["api_key"]:
        logger.error("ALPACA_API_KEY not set. Check .env file.")
        sys.exit(1)

    broker = AlpacaBroker(broker_config)
    logger.info("Broker initialized (paper mode)")

    # 5. Initialize data provider (broker doubles as price source)
    data_provider = broker

    # 6. Get tickers from training state
    with open(ROOT / "logs" / "hydra_training_state.json") as f:
        train_state = json.load(f)
    train_config = train_state.get("config", {})
    tickers = (
        train_config.get("tickers", [])
        or train_config.get("data", {}).get("tickers", [])
        or ft_config.get("tickers", [])
    )

    if not tickers:
        logger.error("No tickers configured for forward test.")
        sys.exit(1)

    logger.info(f"Tickers: {tickers}")

    # 7. Initialize tracker
    tracker = ForwardTestTracker(
        log_path=str(ROOT / "logs" / "forward_test" / "forward_test_log.jsonl"),
        state_path=str(ROOT / "logs" / "forward_test" / "forward_test_state.json"),
    )

    # 8. Launch forward test runner
    runner = ForwardTestRunner(
        agents=agents,
        broker=broker,
        data_provider=data_provider,
        tickers=tickers,
        config=ft_config,
        tracker=tracker,
        allocations=allocations,
    )

    logger.info("Starting forward test (blocking — Ctrl+C to stop)")
    logger.info(f"Market hours only. Polling every {ft_config.get('poll_interval_minutes', 5)} minutes.")

    try:
        runner.start()
    except KeyboardInterrupt:
        logger.info("Forward test interrupted by user.")
        report = runner.stop()
        logger.info(f"Partial report: {json.dumps(report, indent=2, default=str)[:500]}")


if __name__ == "__main__":
    main()
