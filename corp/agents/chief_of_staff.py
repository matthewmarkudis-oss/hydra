"""Chief of Staff — routes events between divisions and orchestrates pipeline runs.

This is the central coordinator for HydraCorp. It:
1. Runs pre-flight checks (Senior Dev config validation)
2. Executes the Hydra pipeline
3. Routes results to Strategy Division (Hedge Fund Director, Contrarian)
4. Aggregates reports for the CEO Dashboard
5. Handles shadow trading promotion logic
"""

from __future__ import annotations

import logging
from typing import Any

from corp.agents.base_corp_agent import BaseCorpAgent
from corp.state.config_blacklist import ConfigBlacklist
from corp.state.corporation_state import CorporationState
from corp.state.decision_log import DecisionLog

logger = logging.getLogger("corp.agents.chief_of_staff")


class ChiefOfStaff(BaseCorpAgent):
    """Central coordinator for the corporation.

    Wraps the Hydra PipelineOrchestrator with pre/post hooks
    and routes outputs to the appropriate corp agents.
    """

    def __init__(
        self,
        state: CorporationState,
        decision_log: DecisionLog,
        blacklist: ConfigBlacklist,
    ):
        super().__init__("chief_of_staff", state, decision_log)
        self.blacklist = blacklist
        self._registered_agents: dict[str, BaseCorpAgent] = {}

    def register_agent(self, agent: BaseCorpAgent) -> None:
        """Register a corp agent for orchestration."""
        self._registered_agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute a full corporation cycle.

        Context should contain:
        - config_dict: The HydraConfig dict to validate
        - orchestrator: The PipelineOrchestrator instance (optional)
        - skip_pipeline: If True, only run analysis on existing results
        """
        results = {
            "phase": "chief_of_staff",
            "pre_flight": {},
            "pipeline": {},
            "analysis": {},
            "alerts": [],
        }

        config_dict = context.get("config_dict", {})

        # Phase 1: Pre-flight checks
        logger.info("=== Pre-flight checks ===")

        # Check config blacklist
        is_blacklisted, reason = self.blacklist.is_blacklisted(config_dict)
        if is_blacklisted:
            results["pre_flight"]["blacklisted"] = True
            results["pre_flight"]["reason"] = reason
            results["alerts"].append({
                "type": "veto",
                "message": f"Config blocked by Senior Dev: {reason}",
            })
            self.log_decision(
                "config_veto",
                detail={"reason": reason},
                outcome="blocked",
            )
            logger.warning(f"Config vetoed: {reason}")
            self._mark_run(results)
            return results

        results["pre_flight"]["blacklisted"] = False

        # Run hardware optimizer if registered
        hw_agent = self._registered_agents.get("hardware_optimizer")
        if hw_agent:
            try:
                hw_result = hw_agent.run(context)
                results["pre_flight"]["hardware"] = hw_result
            except Exception as e:
                logger.error(f"Hardware optimizer failed: {e}")

        # Phase 2: Pipeline execution
        if not context.get("skip_pipeline", False):
            orchestrator = context.get("orchestrator")
            if orchestrator:
                logger.info("=== Running Hydra pipeline ===")
                try:
                    pipeline_results = orchestrator.run()
                    results["pipeline"] = self._summarize_pipeline(pipeline_results)

                    # Record results
                    self.state.record_pipeline_result(results["pipeline"])
                    self.log_decision(
                        "pipeline_complete",
                        detail=results["pipeline"],
                        outcome="success",
                    )
                except Exception as e:
                    logger.error(f"Pipeline failed: {e}")
                    results["pipeline"]["error"] = str(e)
                    self.log_decision(
                        "pipeline_failed",
                        detail={"error": str(e)},
                        outcome="failed",
                    )
            else:
                logger.info("No orchestrator provided, skipping pipeline")
        else:
            logger.info("Pipeline skipped (analysis-only mode)")

        # Phase 3: Post-pipeline analysis
        logger.info("=== Post-pipeline analysis ===")

        # Run Strategy Division agents
        for agent_name in ("hedge_fund_director", "contrarian"):
            agent = self._registered_agents.get(agent_name)
            if agent:
                try:
                    agent_context = {**context, "pipeline_results": results["pipeline"]}
                    agent_result = agent.run(agent_context)
                    results["analysis"][agent_name] = agent_result
                except Exception as e:
                    logger.error(f"{agent_name} failed: {e}")
                    results["analysis"][agent_name] = {"error": str(e)}

        # Run Intelligence Division agents (if due)
        for agent_name in ("geopolitics_expert", "innovation_scout"):
            agent = self._registered_agents.get(agent_name)
            if agent:
                try:
                    agent_result = agent.run(context)
                    results["analysis"][agent_name] = agent_result
                except Exception as e:
                    logger.error(f"{agent_name} failed: {e}")

        # Phase 4: Shadow trading check
        if self.state.should_promote_shadow(
            required_wins=context.get("shadow_promote_after_wins", 3)
        ):
            results["alerts"].append({
                "type": "promotion",
                "message": "Shadow config ready for promotion — outperformed primary "
                          f"for {context.get('shadow_promote_after_wins', 3)} consecutive runs",
            })
            self.log_decision(
                "shadow_promotion_recommended",
                outcome="pending_ceo_approval",
            )

        self._mark_run(results)
        logger.info("=== Corporation cycle complete ===")
        return results

    def _summarize_pipeline(self, pipeline_results: dict) -> dict:
        """Extract CEO-relevant summary from raw pipeline results."""
        validation = pipeline_results.get("validation", {})
        agent_results = validation.get("agent_results", {})
        passed = validation.get("passed_agents", [])

        best_agent = ""
        best_return = -float("inf")
        for name, metrics in agent_results.items():
            total_return = metrics.get("total_return", 0)
            if total_return > best_return:
                best_return = total_return
                best_agent = name

        benchmark = validation.get("benchmark", {})

        return {
            "best_agent": best_agent,
            "best_return": best_return,
            "passed_count": len(passed),
            "passed_agents": passed,
            "total_agents": len(agent_results),
            "benchmark_return": benchmark.get("total_return", 0),
            "excess_return": best_return - benchmark.get("total_return", 0),
        }

    def get_ceo_briefing(self) -> dict[str, Any]:
        """Generate a CEO-friendly summary of the current state."""
        corp_state = self.state.get_full_state()
        last_result = corp_state.get("last_pipeline_result") or {}
        regime = corp_state.get("regime") or {}

        starting_capital = 2500.0
        best_return = last_result.get("best_return", 0)
        portfolio_value = starting_capital * (1 + best_return)
        dollar_pnl = starting_capital * best_return
        excess = last_result.get("excess_return", 0)

        return {
            "portfolio_value_cad": round(portfolio_value, 2),
            "total_return_pct": round(best_return * 100, 2),
            "dollar_pnl_cad": round(dollar_pnl, 2),
            "best_agent": last_result.get("best_agent", "N/A"),
            "vs_benchmark": f"{'+' if excess > 0 else ''}{excess * 100:.1f}%",
            "agents_passed": last_result.get("passed_count", 0),
            "regime": regime.get("classification", "unknown"),
            "pipeline_runs": corp_state.get("pipeline_run_count", 0),
            "pending_proposals": len(corp_state.get("proposals", [])),
            "recent_alerts": [
                m for m in corp_state.get("messages", [])[-10:]
                if m.get("msg_type") == "alert"
            ],
        }
