"""Overnight Autonomous Training Run — 10-hour pipeline toward paper trading.

Runs extended population-based training (150 generations), validates with
ATHENA, evaluates graduation readiness, and produces a morning report.

Usage:
    python scripts/overnight_run.py

Backtesting and training only.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Project root setup
HYDRA_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(HYDRA_ROOT))
sys.path.insert(0, str(HYDRA_ROOT.parent))  # TradingAgents parent

from hydra.config.schema import HydraConfig
from corp.agents.generation_scorer import score_generation, format_scorecard

# ── Configuration ──────────────────────────────────────────────────────────

OVERNIGHT_CONFIG = {
    # Training: 30 gens with pool pruning (max_pool_size=20, bottom_k_demote=3)
    "num_generations": 30,
    "episodes_per_generation": 100,
    "total_timesteps": 500_000,
    # Pool pruning — these override hydra_config.yaml
    "bottom_k_demote": 3,
    "max_pool_size": 20,
    # Checkpointing every 10 gens for safety
    "checkpoint_interval": 50_000,
    "eval_interval": 10_000,
}

REPORT_PATH = HYDRA_ROOT / "logs" / "overnight_report.json"
LOG_PATH = HYDRA_ROOT / "logs" / "overnight_run.log"

logger = logging.getLogger("overnight")


# ── Helpers ────────────────────────────────────────────────────────────────

def _load_alpaca_config() -> dict | None:
    """Load Alpaca credentials from trading_agents/.env."""
    env_path = HYDRA_ROOT.parent / "trading_agents" / ".env"
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


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    """Log to both console and file."""
    ts = _timestamp()
    line = f"[{ts}] {msg}"
    print(line)
    logger.info(msg)


def _save_report(report: dict) -> None:
    """Save the overnight report to JSON."""
    report["saved_at"] = datetime.now().isoformat()
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    _log(f"Report saved to {REPORT_PATH}")


# ── Phase 1: Training ─────────────────────────────────────────────────────

def run_training_phase(config: HydraConfig, alpaca_config: dict | None) -> dict:
    """Run extended training with real data."""
    from hydra.pipeline.orchestrator import PipelineOrchestrator

    _log(f"Starting training: {config.training.num_generations} generations, "
         f"{len(config.data.tickers)} tickers, real data")

    use_real = alpaca_config is not None
    orchestrator = PipelineOrchestrator(
        config=config,
        alpaca_config=alpaca_config,
        use_real_data=use_real,
    )

    start = time.time()
    results = orchestrator.run()
    elapsed = time.time() - start

    _log(f"Training complete in {elapsed/3600:.1f} hours")

    return {
        "phase": "training",
        "elapsed_hours": round(elapsed / 3600, 2),
        "generations": config.training.num_generations,
        "results": {
            k: v for k, v in results.items()
            if k in ("validation", "best_agent", "best_return", "status")
        },
    }


# ── Phase 2: Validation Analysis ──────────────────────────────────────────

def run_validation_analysis() -> dict:
    """Analyze training results and score generations."""
    _log("Analyzing training results...")

    ts_path = HYDRA_ROOT / "logs" / "hydra_training_state.json"
    if not ts_path.exists():
        return {"phase": "validation_analysis", "error": "No training state found"}

    with open(ts_path, encoding="utf-8") as f:
        ts = json.load(f)

    gens = ts.get("generations", [])
    if not gens:
        return {"phase": "validation_analysis", "error": "No generations found"}

    # Score the latest generation
    latest = gens[-1]
    prev = gens[:-1]
    scorecard = score_generation(latest, prev)

    _log(f"Generation {scorecard['generation']} scorecard:")
    _log(f"  Overall: {scorecard['overall']}/10  Verdict: {scorecard['verdict']}")
    _log(f"  Best agent: {scorecard['best_agent']} ({scorecard['best_score']:+.1f})")
    for gap in scorecard.get("critical_gaps", []):
        _log(f"  Gap [{gap['priority']}]: {gap['title']}")

    # Score trend across last 10 generations
    trend = []
    for i in range(max(0, len(gens) - 10), len(gens)):
        prev_gens = gens[:i]
        sc = score_generation(gens[i], prev_gens)
        trend.append({
            "generation": sc["generation"],
            "overall": sc["overall"],
            "verdict": sc["verdict"],
            "best_score": sc["best_score"],
        })

    # ATHENA validation results
    validation = ts.get("validation", {})
    passed_agents = validation.get("passed_agents", [])

    # Also check latest gen validation
    latest_validation = latest.get("validation", {})
    if not passed_agents:
        passed_agents = latest_validation.get("passed_agents", [])

    _log(f"ATHENA passed agents: {len(passed_agents)} ({passed_agents})")

    # Agent fitness ranking
    eval_scores = latest.get("eval_scores", {})
    ranked = sorted(eval_scores.items(), key=lambda x: x[1], reverse=True)

    return {
        "phase": "validation_analysis",
        "scorecard": scorecard,
        "scorecard_formatted": format_scorecard(scorecard),
        "trend": trend,
        "athena_passed": passed_agents,
        "total_agents": len(eval_scores),
        "positive_agents": sum(1 for v in eval_scores.values() if v > 0),
        "top_10_agents": [{"name": n, "score": s} for n, s in ranked[:10]],
        "conviction_summary": _summarize_conviction(latest),
    }


def _summarize_conviction(gen: dict) -> dict:
    """Summarize conviction data for the report."""
    conviction = gen.get("conviction", {})
    summary = {"agents_with_trades": 0, "best_win_rate": 0, "best_agent": "N/A"}

    for name, data in conviction.items():
        if isinstance(data, dict) and data.get("total_trades", 0) >= 3:
            summary["agents_with_trades"] += 1
            wr = data.get("overall_win_rate", 0)
            if wr > summary["best_win_rate"]:
                summary["best_win_rate"] = wr
                summary["best_agent"] = name

    return summary


# ── Phase 3: Graduation Readiness ─────────────────────────────────────────

def evaluate_graduation_readiness(validation_result: dict) -> dict:
    """Evaluate whether agents are ready for forward testing."""
    _log("Evaluating graduation readiness...")

    passed = validation_result.get("athena_passed", [])
    scorecard = validation_result.get("scorecard", {})
    verdict = scorecard.get("verdict", "HALT")
    overall = scorecard.get("overall", 0)

    readiness = {
        "phase": "graduation_readiness",
        "ready": False,
        "reason": "",
        "next_steps": [],
    }

    if not passed:
        readiness["reason"] = "No agents passed ATHENA validation."
        readiness["next_steps"] = [
            "Run more training generations to improve agent quality",
            "Review reward parameters — current agents may be undertrained",
            "Consider adjusting validation thresholds if they're too strict",
        ]

        # Check if we're close (high overall score but no ATHENA)
        if overall >= 7:
            readiness["next_steps"].insert(0,
                "Agents scoring well but ATHENA not run — "
                "run validation phase explicitly"
            )
    elif verdict == "HALT":
        readiness["reason"] = f"Agents passed ATHENA but scorecard verdict is HALT ({overall}/10)."
        readiness["next_steps"] = [
            "Investigate critical gaps in scorecard",
            "Address stability and consistency issues before forward testing",
        ]
    elif verdict == "RETUNE":
        readiness["reason"] = (
            f"{len(passed)} agents passed ATHENA. Scorecard: {overall}/10 (RETUNE). "
            "Approaching readiness but needs tuning."
        )
        readiness["next_steps"] = [
            f"Passed agents: {', '.join(passed)}",
            "Address scorecard gaps before graduating",
            "Consider running with adjusted reward parameters",
            "Forward testing can be attempted at CEO discretion",
        ]
        # Borderline ready if score >= 6
        if overall >= 6:
            readiness["ready"] = True
            readiness["next_steps"].insert(0,
                "CEO can approve graduation proposal for sandbox forward testing"
            )
    else:  # CONTINUE
        readiness["ready"] = True
        readiness["reason"] = (
            f"{len(passed)} agents passed ATHENA. Scorecard: {overall}/10 (CONTINUE). "
            "Ready for forward testing."
        )
        readiness["next_steps"] = [
            f"Graduated agents: {', '.join(passed)}",
            "Enable forward testing: set forward_test.enabled = true",
            "Approve graduation proposal via CEO CLI",
            "Agents will run for 20 days on Alpaca sandbox",
        ]

    _log(f"Graduation readiness: {readiness['ready']}")
    _log(f"  {readiness['reason']}")
    for step in readiness["next_steps"]:
        _log(f"  → {step}")

    return readiness


# ── Phase 4: Corp Agent Cycle ─────────────────────────────────────────────

def run_corp_analysis(config: HydraConfig) -> dict:
    """Run the corp agent cycle for strategy memos and proposals."""
    _log("Running corp agent analysis cycle...")

    try:
        from corp.state.corporation_state import CorporationState
        from corp.state.config_blacklist import ConfigBlacklist
        from corp.state.decision_log import DecisionLog
        from corp.scripts.run_corporation import build_all_agents, run_with_graph

        state = CorporationState()
        blacklist = ConfigBlacklist()
        decision_log = DecisionLog()

        agents = build_all_agents(state, decision_log, blacklist)

        results = run_with_graph(
            agents=agents,
            hydra_config=config,
            orchestrator=None,  # Skip pipeline — already ran
            skip_pipeline=True,
            force_all=True,
        )

        # Extract key results
        pending_proposals = state.get_pending_proposals()
        regime = state.get_regime()

        return {
            "phase": "corp_analysis",
            "regime": regime.get("classification", "unknown"),
            "regime_confidence": regime.get("confidence", 0),
            "pending_proposals": len(pending_proposals),
            "hedge_fund_memo": results.get("hedge_fund_memo", {}).get("memo", ""),
            "contrarian_fired": results.get("contrarian_review", {}).get("fired", False),
        }

    except Exception as e:
        _log(f"Corp analysis error: {e}")
        return {"phase": "corp_analysis", "error": str(e)}


# ── Morning Report ────────────────────────────────────────────────────────

def generate_morning_report(phases: dict) -> dict:
    """Compile all phase results into a morning briefing."""
    _log("=" * 60)
    _log("  GENERATING MORNING REPORT")
    _log("=" * 60)

    training = phases.get("training", {})
    validation = phases.get("validation", {})
    graduation = phases.get("graduation", {})
    corp = phases.get("corp", {})

    scorecard = validation.get("scorecard", {})

    report = {
        "title": "HydraCorp Overnight Training Report",
        "started_at": phases.get("started_at", ""),
        "completed_at": datetime.now().isoformat(),

        "training_summary": {
            "generations": training.get("generations", 0),
            "elapsed_hours": training.get("elapsed_hours", 0),
            "status": "completed" if "error" not in training else "failed",
        },

        "scorecard": {
            "overall": scorecard.get("overall", 0),
            "verdict": scorecard.get("verdict", "N/A"),
            "best_agent": scorecard.get("best_agent", "N/A"),
            "best_score": scorecard.get("best_score", 0),
            "gaps": scorecard.get("critical_gaps", []),
        },

        "agent_summary": {
            "total": validation.get("total_agents", 0),
            "positive": validation.get("positive_agents", 0),
            "athena_passed": validation.get("athena_passed", []),
            "top_10": validation.get("top_10_agents", []),
        },

        "graduation": {
            "ready": graduation.get("ready", False),
            "reason": graduation.get("reason", ""),
            "next_steps": graduation.get("next_steps", []),
        },

        "corp_analysis": {
            "regime": corp.get("regime", "unknown"),
            "pending_proposals": corp.get("pending_proposals", 0),
            "hedge_fund_memo": corp.get("hedge_fund_memo", ""),
        },

        "trend": validation.get("trend", []),
    }

    # Print formatted summary
    _log("")
    _log("==========================================================")
    _log("       HYDRACORP OVERNIGHT REPORT -- GOOD MORNING         ")
    _log("==========================================================")
    _log("")
    _log(f"  Training: {report['training_summary']['generations']} generations "
         f"in {report['training_summary']['elapsed_hours']:.1f} hours")
    _log(f"  Scorecard: {report['scorecard']['overall']}/10 — {report['scorecard']['verdict']}")
    _log(f"  Best Agent: {report['scorecard']['best_agent']} "
         f"({report['scorecard']['best_score']:+.1f})")
    _log(f"  Agents: {report['agent_summary']['positive']}/{report['agent_summary']['total']} positive")
    _log(f"  ATHENA Passed: {len(report['agent_summary']['athena_passed'])} agents")
    _log(f"  Graduation Ready: {'YES' if report['graduation']['ready'] else 'NO'}")
    _log(f"  Regime: {report['corp_analysis']['regime']}")
    _log("")

    if report["graduation"]["ready"]:
        _log("  *** AGENTS READY FOR FORWARD TESTING ***")
        _log("  Next: Open CEO CLI and approve the graduation proposal")
    else:
        _log(f"  Status: {report['graduation']['reason']}")

    _log("")
    _log("  Next Steps:")
    for step in report["graduation"]["next_steps"]:
        _log(f"    → {step}")

    if report["scorecard"]["gaps"]:
        _log("")
        _log("  Gaps:")
        for gap in report["scorecard"]["gaps"]:
            _log(f"    [{gap['priority']}] {gap['title']}")

    _log("")
    _log(f"  Full report: {REPORT_PATH}")
    _log(f"  Dashboard: python scripts/hydra_dashboard.py")
    _log(f"  CEO CLI:   python corp/scripts/ceo_cli.py")
    _log("")

    return report


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    # Setup logging
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(str(LOG_PATH), mode="w"),
            logging.StreamHandler(),
        ],
    )

    phases: dict = {"started_at": datetime.now().isoformat()}

    _log("=" * 60)
    _log("  HYDRACORP OVERNIGHT AUTONOMOUS RUN")
    _log(f"  Started: {_timestamp()}")
    _log(f"  Target: {OVERNIGHT_CONFIG['num_generations']} generations with real data")
    _log(f"  Pool pruning: max_pool_size=20, bottom_k_demote=3")
    _log("=" * 60)

    # Load and patch config for extended training
    config_path = HYDRA_ROOT / "hydra_config.yaml"
    if config_path.exists():
        config = HydraConfig.from_yaml(config_path)
    else:
        config = HydraConfig()

    # Apply overnight overrides (including pool pruning)
    config = config.apply_patch({
        "training": {
            "num_generations": OVERNIGHT_CONFIG["num_generations"],
            "episodes_per_generation": OVERNIGHT_CONFIG["episodes_per_generation"],
            "total_timesteps": OVERNIGHT_CONFIG["total_timesteps"],
            "checkpoint_interval": OVERNIGHT_CONFIG["checkpoint_interval"],
            "eval_interval": OVERNIGHT_CONFIG["eval_interval"],
            "bottom_k_demote": OVERNIGHT_CONFIG["bottom_k_demote"],
            "max_pool_size": OVERNIGHT_CONFIG["max_pool_size"],
        },
    })

    _log(f"Config: {config.training.num_generations} gens, "
         f"{len(config.data.tickers)} tickers, "
         f"seed={config.seed}")

    # Check for Alpaca credentials
    alpaca_config = _load_alpaca_config()
    if alpaca_config:
        _log("Alpaca credentials loaded — using real market data")
    else:
        _log("No Alpaca credentials — using synthetic data")

    # ── Phase 1: Training ──────────────────────────────────────────────
    _log("")
    _log("=== PHASE 1: EXTENDED TRAINING ===")
    try:
        phases["training"] = run_training_phase(config, alpaca_config)
    except Exception as e:
        _log(f"TRAINING FAILED: {e}")
        _log(traceback.format_exc())
        phases["training"] = {"phase": "training", "error": str(e)}

    # Save intermediate report
    _save_report({"phases": phases, "status": "training_complete"})

    # ── Phase 2: Validation Analysis ───────────────────────────────────
    _log("")
    _log("=== PHASE 2: VALIDATION ANALYSIS ===")
    try:
        phases["validation"] = run_validation_analysis()
    except Exception as e:
        _log(f"VALIDATION ANALYSIS FAILED: {e}")
        phases["validation"] = {"phase": "validation_analysis", "error": str(e)}

    # ── Phase 3: Graduation Readiness ──────────────────────────────────
    _log("")
    _log("=== PHASE 3: GRADUATION READINESS ===")
    try:
        phases["graduation"] = evaluate_graduation_readiness(
            phases.get("validation", {})
        )
    except Exception as e:
        _log(f"GRADUATION EVAL FAILED: {e}")
        phases["graduation"] = {"phase": "graduation_readiness", "error": str(e)}

    # ── Phase 4: Corp Agent Analysis ───────────────────────────────────
    _log("")
    _log("=== PHASE 4: CORP AGENT ANALYSIS ===")
    try:
        phases["corp"] = run_corp_analysis(config)
    except Exception as e:
        _log(f"CORP ANALYSIS FAILED: {e}")
        phases["corp"] = {"phase": "corp_analysis", "error": str(e)}

    # ── Morning Report ─────────────────────────────────────────────────
    _log("")
    report = generate_morning_report(phases)
    _save_report(report)

    _log("Overnight run complete. Good night.")


if __name__ == "__main__":
    main()
