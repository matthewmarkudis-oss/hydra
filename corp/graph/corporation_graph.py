"""LangGraph StateGraph definition for HydraCorp.

Defines the corporation workflow as a directed graph with conditional edges.
Nodes correspond to corp agents; edges define data flow and trigger conditions.

The graph supports two modes:
1. Full cycle: pre-flight -> pipeline -> analysis -> report
2. Analysis-only: skip pipeline, analyze existing results

Backtesting and training only.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

logger = logging.getLogger("corp.graph")


class CorpGraphState(TypedDict, total=False):
    """Shared state flowing through the corporation graph.

    Each node reads from and writes to this dict.
    """
    # Inputs
    config_dict: dict
    orchestrator: Any
    use_real_data: bool
    skip_pipeline: bool
    force_all_agents: bool

    # Pre-flight outputs
    blacklist_check: dict
    hardware_analysis: dict
    senior_dev_review: dict

    # Pipeline outputs
    pipeline_results: dict
    pipeline_error: str | None

    # Analysis outputs
    hedge_fund_memo: dict
    contrarian_review: dict
    geopolitics_regime: dict
    innovation_briefs: dict

    # Shadow outputs
    shadow_comparison: dict

    # Final
    ceo_briefing: dict
    alerts: list[dict]
    cycle_complete: bool


def build_corporation_graph(agents: dict[str, Any]) -> "CorpGraph":
    """Build the corporation workflow graph.

    Args:
        agents: Dict mapping agent names to agent instances.
            Expected keys: chief_of_staff, senior_dev, hardware_optimizer,
            shadow_trader, hedge_fund_director, contrarian,
            geopolitics_expert, innovation_scout

    Returns:
        CorpGraph instance with execute() method.
    """
    return CorpGraph(agents)


class CorpGraph:
    """Corporation workflow graph — executes agents in the correct order.

    This is a lightweight graph executor that follows the planned DAG:

    pre_flight -> [pipeline] -> post_analysis -> intelligence -> shadow -> report

    Uses LangGraph-compatible state passing but does NOT require langgraph
    as a dependency. If langgraph is available, it can wrap this in a
    proper StateGraph for checkpointing and visualization.
    """

    def __init__(self, agents: dict[str, Any]):
        self._agents = agents
        self._node_order = [
            "pre_flight",
            "intelligence",
            "pipeline",
            "post_analysis",
            "shadow",
            "report",
        ]

    def execute(self, initial_state: CorpGraphState) -> CorpGraphState:
        """Execute the full corporation cycle.

        Args:
            initial_state: Starting state with config_dict, orchestrator, etc.

        Returns:
            Final state with all agent outputs.
        """
        state: dict[str, Any] = dict(initial_state)
        state.setdefault("alerts", [])
        state.setdefault("pipeline_error", None)
        state.setdefault("cycle_complete", False)

        for node_name in self._node_order:
            handler = getattr(self, f"_node_{node_name}", None)
            if handler is None:
                logger.warning(f"Unknown node: {node_name}")
                continue

            try:
                logger.info(f"=== Node: {node_name} ===")
                state = handler(state)
            except Exception as e:
                logger.error(f"Node {node_name} failed: {e}")
                state["alerts"].append({
                    "type": "error",
                    "node": node_name,
                    "message": str(e),
                })

            # Check for early termination
            if state.get("blacklist_check", {}).get("blocked"):
                logger.warning("Config blocked — terminating cycle early")
                break

        state["cycle_complete"] = True
        return state

    def _node_pre_flight(self, state: dict) -> dict:
        """Pre-flight checks: blacklist validation + hardware detection."""
        config_dict = state.get("config_dict", {})

        # Senior Dev: config validation
        senior_dev = self._agents.get("senior_dev")
        if senior_dev:
            review = senior_dev.run({"config_dict": config_dict})
            state["senior_dev_review"] = review
            if not review.get("checks_passed", True):
                state["blacklist_check"] = {"blocked": True, "vetoes": review.get("vetoes", [])}
                state["alerts"].append({
                    "type": "veto",
                    "message": f"Config blocked: {review.get('vetoes', ['unknown'])}",
                })
                return state

        state["blacklist_check"] = {"blocked": False}

        # Hardware optimizer
        hw = self._agents.get("hardware_optimizer")
        if hw:
            hw_result = hw.run({"config_dict": config_dict})
            state["hardware_analysis"] = hw_result

        # Data Quality Monitor — validate data feeds before training
        dqm = self._agents.get("data_quality_monitor")
        if dqm:
            dq_result = dqm.run({"config_dict": config_dict})
            state["data_quality"] = dq_result
            if dq_result.get("critical_failures", 0) > 0:
                state["alerts"].append({
                    "type": "data_quality",
                    "message": f"Data quality: {dq_result['critical_failures']} critical failures",
                })

        return state

    def _node_pipeline(self, state: dict) -> dict:
        """Execute the Hydra training pipeline."""
        if state.get("skip_pipeline", False):
            logger.info("Pipeline skipped (analysis-only mode)")
            return state

        orchestrator = state.get("orchestrator")
        if orchestrator is None:
            logger.info("No orchestrator provided")
            return state

        # Auto-approve eligible proposals before training starts
        try:
            from corp.state.corporation_state import CorporationState
            corp_state = CorporationState()
            approved = corp_state.auto_resolve_proposals()
            if approved:
                logger.info(f"Auto-approved {len(approved)} proposals before pipeline")
                # Apply approved patches to orchestrator config
                config = orchestrator.config
                for patch in approved:
                    for key, value in patch.items():
                        parts = key.split(".")
                        obj = config
                        for part in parts[:-1]:
                            if hasattr(obj, part):
                                obj = getattr(obj, part)
                            else:
                                break
                        else:
                            if hasattr(obj, parts[-1]):
                                setattr(obj, parts[-1], value)
                                logger.info(f"  Applied: {key} = {value}")
        except Exception as e:
            logger.debug(f"Auto-approve step skipped: {e}")

        try:
            results = orchestrator.run()
            state["pipeline_results"] = _summarize_pipeline(results)
        except Exception as e:
            state["pipeline_error"] = str(e)
            state["alerts"].append({
                "type": "error",
                "message": f"Pipeline failed: {e}",
            })

        return state

    def _node_post_analysis(self, state: dict) -> dict:
        """Strategy Division: Hedge Fund Director + Contrarian."""
        pipeline_results = state.get("pipeline_results", {})
        config_dict = state.get("config_dict", {})
        regime = state.get("geopolitics_regime", {})

        # Hedge Fund Director
        hfd = self._agents.get("hedge_fund_director")
        if hfd:
            context = {
                "pipeline_results": pipeline_results,
                "config_dict": config_dict,
                "regime": regime,
            }
            state["hedge_fund_memo"] = hfd.run(context)

        # Contrarian (conditional)
        contrarian = self._agents.get("contrarian")
        if contrarian:
            context = {
                "pipeline_results": pipeline_results,
                "config_dict": config_dict,
            }
            state["contrarian_review"] = contrarian.run(context)

        # Performance Analyst — attribution, correlation, efficiency
        pa = self._agents.get("performance_analyst")
        if pa:
            pa_context = {
                "pipeline_results": pipeline_results,
                "generation_results": [],  # loaded from training state file
            }
            state["performance_analysis"] = pa.run(pa_context)

        return state

    def _node_intelligence(self, state: dict) -> dict:
        """Intelligence Division: Geopolitics + Innovation Scout."""
        force = state.get("force_all_agents", False)

        geo = self._agents.get("geopolitics_expert")
        if geo:
            result = geo.run({"force": force})
            state["geopolitics_regime"] = result

        scout = self._agents.get("innovation_scout")
        if scout:
            result = scout.run({"force": force})
            state["innovation_briefs"] = result

        return state

    def _node_shadow(self, state: dict) -> dict:
        """Shadow trading comparison."""
        shadow = self._agents.get("shadow_trader")
        if shadow and shadow.has_shadow():
            context = {
                "pipeline_results": state.get("pipeline_results", {}),
                "use_real_data": state.get("use_real_data", False),
            }
            state["shadow_comparison"] = shadow.run(context)

        return state

    def _node_report(self, state: dict) -> dict:
        """Generate CEO briefing from all agent outputs."""
        chief = self._agents.get("chief_of_staff")
        if chief:
            state["ceo_briefing"] = chief.get_ceo_briefing()

        return state


def _summarize_pipeline(pipeline_results: dict) -> dict:
    """Extract key metrics from raw pipeline results."""
    validation = pipeline_results.get("validation", {})
    agent_results = validation.get("agent_results", {})
    passed = validation.get("passed_agents", [])

    best_agent = ""
    best_return = 0.0
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
