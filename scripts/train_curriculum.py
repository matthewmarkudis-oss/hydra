"""Staged Lookback Curriculum Training — 2020 to present.

Progressive curriculum that trains agents across expanding historical windows:
  Stage 1: Recent 504 days  (~2 years) — learn current regime patterns
  Stage 2: Extend to 1008 days (~4 years) — add 2022 bear + 2023 recovery
  Stage 3: Full 1512 days (~6 years) — add COVID crash + stimulus rally

Each stage resumes from the previous stage's checkpoint. Episodes per
generation reduced to 70 (Option A) since broader data provides more
variety per episode.

Usage:
  python scripts/train_curriculum.py
  python scripts/train_curriculum.py --stages 1 2 3
  python scripts/train_curriculum.py --stages 3        # only final stage
  python scripts/train_curriculum.py --gens-per-stage 50
  python scripts/train_curriculum.py --episodes 80

Backtesting and training only.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Project root setup
HYDRA_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(HYDRA_ROOT))
sys.path.insert(0, str(HYDRA_ROOT.parent))

# Load all API keys from .env
_env_path = HYDRA_ROOT.parent / "trading_agents" / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _, _v = _line.partition("=")
            _k, _v = _k.strip(), _v.strip()
            if _k and _v and _k not in os.environ:
                os.environ[_k] = _v

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLU", "XLP", "XLY", "XLB", "XLRE"]

# ── Curriculum Stages ─────────────────────────────────────────────────────

STAGES = {
    1: {
        "name": "Recent Regime",
        "lookback_days": 504,
        "description": "2024-2026: Current market regime, tariffs, AI boom/correction",
        "generations": 40,
        "episodes": 70,
    },
    2: {
        "name": "Bear + Recovery",
        "lookback_days": 1008,
        "description": "2022-2026: Rate hike bear market, AI rally, current instability",
        "generations": 40,
        "episodes": 70,
    },
    3: {
        "name": "Full Spectrum",
        "lookback_days": 1512,
        "description": "2020-2026: COVID crash, stimulus rally, bear market, AI boom, crisis",
        "generations": 40,
        "episodes": 70,
    },
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            HYDRA_ROOT / "logs" / "curriculum_training.log", encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger("curriculum")


def _load_alpaca_config() -> dict | None:
    """Load Alpaca credentials from environment."""
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if api_key and secret_key:
        return {
            "api_key": api_key,
            "secret_key": secret_key,
            "base_url": os.environ.get("ALPACA_BASE_URL", ""),
        }
    return None


def _find_latest_checkpoint() -> str | None:
    """Find the most recent checkpoint directory."""
    ckpt_root = HYDRA_ROOT / "checkpoints"
    latest_file = ckpt_root / "latest.json"
    if latest_file.exists():
        try:
            with open(latest_file) as f:
                pointer = json.load(f)
            path = pointer.get("path", "")
            if path and Path(path).exists():
                return path
        except Exception:
            pass
    return None


def run_stage(
    stage_num: int,
    stage_config: dict,
    resume_from: str | None = None,
    alpaca_config: dict | None = None,
    tickers: list[str] | None = None,
    gens_override: int | None = None,
    episodes_override: int | None = None,
) -> dict:
    """Run a single curriculum stage."""
    from hydra.config.schema import HydraConfig
    from hydra.pipeline.orchestrator import PipelineOrchestrator

    gens = gens_override or stage_config["generations"]
    episodes = episodes_override or stage_config["episodes"]
    lookback = stage_config["lookback_days"]
    tickers = tickers or SECTOR_ETFS

    logger.info(
        f"=== STAGE {stage_num}: {stage_config['name']} ===\n"
        f"  Lookback: {lookback} days\n"
        f"  Generations: {gens}\n"
        f"  Episodes/gen: {episodes}\n"
        f"  Resume from: {resume_from or 'fresh start'}\n"
        f"  {stage_config['description']}"
    )

    config = HydraConfig()
    config.data.tickers = tickers
    config.data.lookback_days = lookback
    config.env.num_stocks = len(tickers)
    config.training.num_generations = gens
    config.training.episodes_per_generation = episodes
    config.training.auto_tune_rewards = True
    config.seed = 42

    start = time.time()
    try:
        orchestrator = PipelineOrchestrator(
            config,
            alpaca_config=alpaca_config,
            use_real_data=True,
            resume_checkpoint=resume_from,
        )
        orchestrator.run()
        duration = time.time() - start

        summary = orchestrator.get_summary()
        latest_ckpt = _find_latest_checkpoint()

        result = {
            "stage": stage_num,
            "name": stage_config["name"],
            "lookback_days": lookback,
            "generations": gens,
            "episodes": episodes,
            "duration_secs": duration,
            "passed_agents": summary.get("passed_agents", []),
            "checkpoint": latest_ckpt,
            "status": "completed",
        }

        logger.info(
            f"Stage {stage_num} completed in {duration/60:.1f} min. "
            f"Passed: {len(result['passed_agents'])} agents. "
            f"Checkpoint: {latest_ckpt}"
        )
        return result

    except Exception as e:
        duration = time.time() - start
        logger.error(f"Stage {stage_num} failed after {duration/60:.1f} min: {e}")
        return {
            "stage": stage_num,
            "name": stage_config["name"],
            "lookback_days": lookback,
            "duration_secs": duration,
            "status": "failed",
            "error": str(e),
            "checkpoint": _find_latest_checkpoint(),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Staged lookback curriculum training (2020-2026)"
    )
    parser.add_argument(
        "--stages", nargs="+", type=int, default=None,
        help="Which stages to run (1, 2, 3). Default: all"
    )
    parser.add_argument(
        "--gens-per-stage", type=int, default=None,
        help="Override generations per stage"
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Override episodes per generation"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from specific checkpoint (skips finding latest)"
    )
    parser.add_argument(
        "--tickers", type=str, default=None,
        help=f"Comma-separated tickers (default: {','.join(SECTOR_ETFS)})"
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Start fresh (don't resume between stages)"
    )
    args = parser.parse_args()

    # Alpaca config (required for real data)
    alpaca_config = _load_alpaca_config()
    if not alpaca_config:
        print("ERROR: Alpaca credentials required. Set ALPACA_API_KEY and "
              "ALPACA_SECRET_KEY in trading_agents/.env")
        sys.exit(1)

    tickers = None
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",")]

    stages_to_run = args.stages or [1, 2, 3]
    stages_to_run = sorted(s for s in stages_to_run if s in STAGES)

    if not stages_to_run:
        print("ERROR: No valid stages specified. Available: 1, 2, 3")
        sys.exit(1)

    print("=" * 64)
    print("  HYDRA CURRICULUM TRAINING — 2020 to 2026")
    print("=" * 64)
    print(f"  Tickers: {', '.join(tickers or SECTOR_ETFS)}")
    print(f"  Stages:  {stages_to_run}")
    for s in stages_to_run:
        cfg = STAGES[s]
        gens = args.gens_per_stage or cfg["generations"]
        eps = args.episodes or cfg["episodes"]
        print(f"    Stage {s}: {cfg['name']} — {cfg['lookback_days']} days, "
              f"{gens} gens x {eps} eps")
    print(f"  Auto reward tuning: ON")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 64)

    all_results = []
    resume_ckpt = args.resume

    for stage_num in stages_to_run:
        stage_config = STAGES[stage_num]

        # Between stages, resume from previous stage's checkpoint
        if args.fresh:
            resume_ckpt = None

        result = run_stage(
            stage_num=stage_num,
            stage_config=stage_config,
            resume_from=resume_ckpt,
            alpaca_config=alpaca_config,
            tickers=tickers,
            gens_override=args.gens_per_stage,
            episodes_override=args.episodes,
        )
        all_results.append(result)

        # Use this stage's checkpoint for the next stage
        if result.get("checkpoint"):
            resume_ckpt = result["checkpoint"]

        if result["status"] == "failed":
            logger.warning(
                f"Stage {stage_num} failed. Continuing to next stage "
                f"with last known checkpoint: {resume_ckpt}"
            )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  CURRICULUM RESULTS")
    print("=" * 64)

    total_time = sum(r.get("duration_secs", 0) for r in all_results)
    print(f"  Total time: {total_time/3600:.1f} hours")

    for r in all_results:
        status = r["status"].upper()
        duration = r.get("duration_secs", 0)
        passed = len(r.get("passed_agents", []))
        lookback = r.get("lookback_days", 0)
        print(f"  Stage {r['stage']}: {r['name']:20s} | {lookback} days | "
              f"{status} | {duration/60:.0f} min | {passed} agents passed")

    all_passed = set()
    for r in all_results:
        all_passed.update(r.get("passed_agents", []))

    if all_passed:
        print(f"\n  Agents that passed ATHENA across curriculum: {sorted(all_passed)}")
    else:
        print(f"\n  No agents passed ATHENA validation (WFE threshold may need adjustment)")

    print("=" * 64)

    # Save results
    results_path = HYDRA_ROOT / "logs" / "curriculum_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "started": datetime.now().isoformat(),
            "stages": all_results,
            "total_duration_secs": total_time,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
