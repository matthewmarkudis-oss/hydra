"""Main entry point for HydraCorp — runs a full corporation cycle.

Usage:
    python corp/scripts/run_corporation.py [--config path/to/config.yaml] [--analysis-only]
    python corp/scripts/run_corporation.py --use-graph  # Use LangGraph-style executor

Backtesting and training only.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# Add TradingAgents parent so `import trading_agents` works
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from hydra.config.schema import HydraConfig
from hydra.pipeline.orchestrator import PipelineOrchestrator
from corp.config.corp_config import CorporationConfig
from corp.state.corporation_state import CorporationState
from corp.state.config_blacklist import ConfigBlacklist
from corp.state.decision_log import DecisionLog
from corp.agents.chief_of_staff import ChiefOfStaff
from corp.agents.senior_dev import SeniorDev
from corp.agents.hardware_optimizer import HardwareOptimizer
from corp.agents.shadow_trader import ShadowTrader
from corp.agents.hedge_fund_director import HedgeFundDirector
from corp.agents.contrarian import Contrarian
from corp.agents.geopolitics_expert import GeopoliticsExpert
from corp.agents.innovation_scout import InnovationScout
from corp.agents.strategy_distiller import StrategyDistiller
from corp.agents.ceo_interface import CEOInterface
from corp.agents.graduation_manager import GraduationManager
from corp.agents.risk_manager import RiskManager
from corp.agents.data_quality_monitor import DataQualityMonitor
from corp.agents.performance_analyst import PerformanceAnalyst
from corp.graph.corporation_graph import CorpGraph, build_corporation_graph

logger = logging.getLogger("corp")


def _load_env_file() -> None:
    """Load all keys from trading_agents/.env into os.environ.

    Loads Alpaca, NewsAPI, Finnhub, Anthropic, and any other keys
    defined in the .env file. Does not overwrite existing env vars.
    """
    import os as _os
    env_path = Path(__file__).parent.parent.parent.parent / "trading_agents" / ".env"
    if not env_path.exists():
        return

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and value and key not in _os.environ:
                _os.environ[key] = value


# Load env vars at import time so corp agents can read API keys
_load_env_file()


def _load_alpaca_config() -> dict | None:
    """Load Alpaca credentials from environment."""
    import os as _os
    api_key = _os.environ.get("ALPACA_API_KEY", "")
    secret_key = _os.environ.get("ALPACA_SECRET_KEY", "")
    if api_key and secret_key:
        return {
            "api_key": api_key,
            "secret_key": secret_key,
            "base_url": _os.environ.get("ALPACA_BASE_URL", ""),
        }
    return None


def build_all_agents(
    state: CorporationState,
    decision_log: DecisionLog,
    blacklist: ConfigBlacklist,
    corp_config: CorporationConfig | None = None,
) -> dict[str, object]:
    """Instantiate all corporation agents.

    Returns:
        Dict mapping agent name to agent instance.
    """
    schedule = corp_config.schedule if corp_config else None

    agents = {}

    # Operations Division
    agents["senior_dev"] = SeniorDev(
        state=state,
        decision_log=decision_log,
        blacklist=blacklist,
    )
    agents["hardware_optimizer"] = HardwareOptimizer(
        state=state,
        decision_log=decision_log,
    )
    agents["shadow_trader"] = ShadowTrader(
        state=state,
        decision_log=decision_log,
    )
    agents["data_quality_monitor"] = DataQualityMonitor(
        state=state,
        decision_log=decision_log,
    )
    agents["risk_manager"] = RiskManager(
        state=state,
        decision_log=decision_log,
    )

    # Strategy Division
    agents["hedge_fund_director"] = HedgeFundDirector(
        state=state,
        decision_log=decision_log,
    )
    agents["contrarian"] = Contrarian(
        state=state,
        decision_log=decision_log,
        trigger_fitness=schedule.contrarian_trigger_fitness if schedule else 0.5,
    )

    # Intelligence Division
    agents["geopolitics_expert"] = GeopoliticsExpert(
        state=state,
        decision_log=decision_log,
        interval_hours=schedule.geopolitics_interval_hours if schedule else 24,
    )
    agents["innovation_scout"] = InnovationScout(
        state=state,
        decision_log=decision_log,
        interval_hours=schedule.innovation_scout_interval_hours if schedule else 168,
    )
    agents["strategy_distiller"] = StrategyDistiller(
        state=state,
        decision_log=decision_log,
        interval_hours=168,  # Weekly
    )

    agents["performance_analyst"] = PerformanceAnalyst(
        state=state,
        decision_log=decision_log,
    )

    # Graduation Manager
    agents["graduation_manager"] = GraduationManager(
        state=state,
        decision_log=decision_log,
    )

    # CEO Interface (interactive — not part of the automated pipeline)
    agents["ceo_interface"] = CEOInterface(
        state=state,
        decision_log=decision_log,
        senior_dev=agents["senior_dev"],
    )

    # Chief of Staff (coordinator)
    chief = ChiefOfStaff(
        state=state,
        decision_log=decision_log,
        blacklist=blacklist,
    )
    # Register all agents with the chief
    for agent in agents.values():
        chief.register_agent(agent)
    agents["chief_of_staff"] = chief

    return agents


def run_with_graph(
    agents: dict,
    hydra_config: HydraConfig,
    orchestrator: PipelineOrchestrator | None,
    use_real_data: bool = False,
    skip_pipeline: bool = False,
    force_all: bool = False,
) -> dict:
    """Run using the CorpGraph executor."""
    graph = build_corporation_graph(agents)

    initial_state = {
        "config_dict": hydra_config.model_dump(),
        "orchestrator": orchestrator,
        "use_real_data": use_real_data,
        "skip_pipeline": skip_pipeline,
        "force_all_agents": force_all,
        "alerts": [],
    }

    result = graph.execute(initial_state)
    return result


def run_with_chief(
    chief: ChiefOfStaff,
    hydra_config: HydraConfig,
    orchestrator: PipelineOrchestrator | None,
    skip_pipeline: bool = False,
) -> dict:
    """Run using the Chief of Staff directly (simpler, no graph)."""
    context = {
        "config_dict": hydra_config.model_dump(),
        "orchestrator": orchestrator,
        "skip_pipeline": skip_pipeline,
    }
    return chief.run(context)


def print_briefing(agents: dict, results: dict) -> None:
    """Print the CEO briefing to console."""
    chief = agents.get("chief_of_staff")
    if chief:
        briefing = chief.get_ceo_briefing()
    else:
        briefing = results.get("ceo_briefing", {})

    print("\n" + "=" * 60)
    print("  HYDRACORP CEO BRIEFING")
    print("=" * 60)
    print(f"  Portfolio Value:  ${briefing.get('portfolio_value_cad', 2500):,.2f} CAD")
    print(f"  Total Return:    {briefing.get('total_return_pct', 0):+.2f}%")
    print(f"  Dollar P&L:      ${briefing.get('dollar_pnl_cad', 0):+,.2f} CAD")
    print(f"  Best Agent:      {briefing.get('best_agent', 'N/A')}")
    print(f"  vs Benchmark:    {briefing.get('vs_benchmark', 'N/A')}")
    print(f"  Market Regime:   {briefing.get('regime', 'unknown')}")
    print(f"  Agents Passed:   {briefing.get('agents_passed', 0)}")
    print(f"  Pipeline Runs:   {briefing.get('pipeline_runs', 0)}")
    print("=" * 60)

    # Alerts
    alerts = results.get("alerts", [])
    if alerts:
        print("\n  ALERTS:")
        for alert in alerts:
            print(f"  [{alert.get('type', 'info').upper()}] {alert.get('message', '')}")

    # Hedge Fund Director memo
    memo = results.get("hedge_fund_memo", results.get("analysis", {}).get("hedge_fund_director", {}))
    if memo and memo.get("memo"):
        print(f"\n  STRATEGY MEMO: {memo['memo']}")
        if memo.get("confidence"):
            print(f"  Confidence: {memo['confidence']:.0%}")

    # Contrarian review
    review = results.get("contrarian_review", results.get("analysis", {}).get("contrarian", {}))
    if review and review.get("fired"):
        print(f"\n  CONTRARIAN VERDICT: {review.get('verdict', 'N/A')}")
        print(f"  Fragility Score: {review.get('fragility_score', 0):.2f}")
        for concern in review.get("concerns", [])[:3]:
            print(f"    - {concern}")

    # Geopolitics
    geo = results.get("geopolitics_regime", results.get("analysis", {}).get("geopolitics_expert", {}))
    if geo and geo.get("regime") and geo["regime"] != "unknown":
        print(f"\n  MARKET REGIME: {geo['regime']} (confidence: {geo.get('confidence', 0):.0%})")
        if geo.get("summary"):
            print(f"  {geo['summary']}")

    # Innovation briefs
    briefs = results.get("innovation_briefs", results.get("analysis", {}).get("innovation_scout", {}))
    if briefs and briefs.get("new_discoveries", 0) > 0:
        print(f"\n  INNOVATION: {briefs['new_discoveries']} new tools discovered")
        if briefs.get("top_recommendation"):
            print(f"  Top recommendation: {briefs['top_recommendation']}")

    # Shadow comparison
    shadow = results.get("shadow_comparison", {})
    if shadow and shadow.get("shadow_enabled"):
        comp = shadow.get("comparison", {})
        winner = comp.get("winner", "unknown")
        print(f"\n  SHADOW COMPARISON: {winner} wins")
        print(f"  Primary: {comp.get('primary_return_pct', 0):+.2f}%")
        print(f"  Shadow:  {comp.get('shadow_return_pct', 0):+.2f}%")

    print()


def main(argv=None):
    parser = argparse.ArgumentParser(description="HydraCorp — AI Trading Corporation (Backtesting)")
    parser.add_argument("--config", type=str, default=None, help="Path to HydraConfig YAML")
    parser.add_argument("--corp-config", type=str, default=None, help="Path to CorporationConfig YAML")
    parser.add_argument("--analysis-only", action="store_true", help="Skip pipeline, analyze existing results")
    parser.add_argument("--populate-blacklist", action="store_true", help="Pre-populate blacklist from meta-optimizer logs")
    parser.add_argument("--use-graph", action="store_true", help="Use CorpGraph executor instead of ChiefOfStaff")
    parser.add_argument("--force-all", action="store_true", help="Force all agents to run (ignore schedules)")
    parser.add_argument("--synthetic", action="store_true", default=True, help="Use synthetic data (default)")
    parser.add_argument("--real-data", action="store_true", help="Use real Alpaca data")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Load configs
    if args.config:
        hydra_config = HydraConfig.from_yaml(args.config)
    else:
        hydra_config = HydraConfig()

    corp_config = None
    if args.corp_config:
        import yaml
        with open(args.corp_config) as f:
            corp_config = CorporationConfig(**yaml.safe_load(f))

    # Initialize corp components
    state = CorporationState()
    blacklist = ConfigBlacklist()
    decision_log = DecisionLog()

    # Optionally populate blacklist from meta-optimizer history
    if args.populate_blacklist:
        meta_log = Path("logs/meta_optimize.jsonl")
        if meta_log.exists():
            from corp.agents.senior_dev import SeniorDev
            sd = SeniorDev(state=state, decision_log=decision_log, blacklist=blacklist)
            count = sd.populate_blacklist()
            logger.info(f"Populated blacklist with {count} entries")
        else:
            logger.warning("No meta_optimize.jsonl found")

    # Build all agents
    agents = build_all_agents(state, decision_log, blacklist, corp_config)

    # Build orchestrator
    use_real = args.real_data
    alpaca_config = None
    if use_real:
        alpaca_config = _load_alpaca_config()
        if alpaca_config is None:
            logger.error("--real-data requires Alpaca credentials in trading_agents/.env")
            sys.exit(1)
        logger.info("Loaded Alpaca credentials for historical data")

    orchestrator = None
    if not args.analysis_only:
        orchestrator = PipelineOrchestrator(
            config=hydra_config,
            alpaca_config=alpaca_config,
            use_real_data=use_real,
        )

    # Execute
    logger.info("=" * 60)
    logger.info("  HYDRACORP STARTING")
    logger.info(f"  Mode: {'Graph' if args.use_graph else 'Chief of Staff'}")
    logger.info(f"  Pipeline: {'Skipped' if args.analysis_only else 'Enabled'}")
    logger.info(f"  Data: {'Real' if use_real else 'Synthetic'}")
    logger.info("=" * 60)

    if args.use_graph:
        results = run_with_graph(
            agents=agents,
            hydra_config=hydra_config,
            orchestrator=orchestrator,
            use_real_data=use_real,
            skip_pipeline=args.analysis_only,
            force_all=args.force_all,
        )
    else:
        results = run_with_chief(
            chief=agents["chief_of_staff"],
            hydra_config=hydra_config,
            orchestrator=orchestrator,
            skip_pipeline=args.analysis_only,
        )

    # Print briefing
    print_briefing(agents, results)

    return results


if __name__ == "__main__":
    main()
