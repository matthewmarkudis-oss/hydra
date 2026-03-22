"""Node functions for the corporation graph.

Each function takes a CorpGraphState dict and returns the modified state.
These are standalone functions that can be used with LangGraph's StateGraph
or with the built-in CorpGraph executor.

Backtesting and training only.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("corp.graph.nodes")


def node_pre_flight(state: dict, agents: dict[str, Any]) -> dict:
    """Pre-flight checks before pipeline execution.

    Runs Senior Dev config validation and hardware detection.
    Sets state['blacklist_check'] to indicate if config is blocked.
    """
    config_dict = state.get("config_dict", {})
    alerts = state.get("alerts", [])

    # Senior Dev review
    senior_dev = agents.get("senior_dev")
    if senior_dev:
        review = senior_dev.run({"config_dict": config_dict})
        state["senior_dev_review"] = review
        if not review.get("checks_passed", True):
            state["blacklist_check"] = {
                "blocked": True,
                "vetoes": review.get("vetoes", []),
            }
            alerts.append({
                "type": "veto",
                "message": f"Config blocked by Senior Dev",
            })
            state["alerts"] = alerts
            return state

    state["blacklist_check"] = {"blocked": False}

    # Hardware optimizer
    hw = agents.get("hardware_optimizer")
    if hw:
        state["hardware_analysis"] = hw.run({"config_dict": config_dict})

    # Data Quality Monitor — validate data feeds before training
    dqm = agents.get("data_quality_monitor")
    if dqm:
        dq_result = dqm.run({"config_dict": config_dict})
        state["data_quality"] = dq_result
        if dq_result.get("critical_failures", 0) > 0:
            alerts.append({
                "type": "data_quality",
                "message": f"Data quality: {dq_result['critical_failures']} critical failures",
            })

    state["alerts"] = alerts
    return state


def node_pipeline(state: dict, agents: dict[str, Any]) -> dict:
    """Execute the Hydra training pipeline."""
    if state.get("skip_pipeline", False):
        return state

    orchestrator = state.get("orchestrator")
    if not orchestrator:
        return state

    # Auto-approve eligible proposals before training starts
    corp_state = agents.get("_corp_state")
    if corp_state and hasattr(corp_state, "auto_resolve_proposals"):
        try:
            approved = corp_state.auto_resolve_proposals()
            if approved:
                logger.info(f"Auto-approved {len(approved)} proposals before pipeline")
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
        # Summarize
        validation = results.get("validation", {})
        agent_results = validation.get("agent_results", {})
        passed = validation.get("passed_agents", [])

        best_agent = ""
        best_return = 0.0
        for name, metrics in agent_results.items():
            ret = metrics.get("total_return", 0)
            if ret > best_return:
                best_return = ret
                best_agent = name

        benchmark = validation.get("benchmark", {})
        state["pipeline_results"] = {
            "best_agent": best_agent,
            "best_return": best_return,
            "passed_count": len(passed),
            "total_agents": len(agent_results),
            "excess_return": best_return - benchmark.get("total_return", 0),
        }

    except Exception as e:
        state["pipeline_error"] = str(e)
        state.setdefault("alerts", []).append({
            "type": "error",
            "message": f"Pipeline failed: {e}",
        })

    return state


def node_strategy(state: dict, agents: dict[str, Any]) -> dict:
    """Strategy Division: Hedge Fund Director + Contrarian analysis."""
    pipeline_results = state.get("pipeline_results", {})
    config_dict = state.get("config_dict", {})

    hfd = agents.get("hedge_fund_director")
    if hfd:
        # Merge regime with ticker_recommendations from geopolitics
        regime = state.get("geopolitics_regime", {})
        if isinstance(regime, dict) and "ticker_recommendations" not in regime:
            # Pull from corp state if geopolitics stored it there
            corp_state_agent = agents.get("_corp_state")
            if corp_state_agent and hasattr(corp_state_agent, "get_regime"):
                stored_regime = corp_state_agent.get_regime()
                regime = {**regime, **{
                    k: v for k, v in stored_regime.items()
                    if k == "ticker_recommendations" and v
                }}

        state["hedge_fund_memo"] = hfd.run({
            "pipeline_results": pipeline_results,
            "config_dict": config_dict,
            "regime": regime,
        })

    contrarian = agents.get("contrarian")
    if contrarian:
        state["contrarian_review"] = contrarian.run({
            "pipeline_results": pipeline_results,
            "config_dict": config_dict,
        })

    # Performance Analyst — attribution, correlation, efficiency
    pa = agents.get("performance_analyst")
    if pa:
        state["performance_analysis"] = pa.run({
            "pipeline_results": pipeline_results,
            "generation_results": [],  # loaded from training state file
        })

    return state


def node_intelligence(state: dict, agents: dict[str, Any]) -> dict:
    """Intelligence Division: Geopolitics + Innovation Scout."""
    force = state.get("force_all_agents", False)

    geo = agents.get("geopolitics_expert")
    if geo:
        state["geopolitics_regime"] = geo.run({"force": force})

    scout = agents.get("innovation_scout")
    if scout:
        state["innovation_briefs"] = scout.run({"force": force})

    # Strategy Distiller — factor analysis + reward calibration
    distiller = agents.get("strategy_distiller")
    if distiller:
        config_dict = state.get("config_dict", {})
        calibration_mode = state.get("calibration_mode", "factor_mapping")
        state["distillation_result"] = distiller.run({
            "config_dict": config_dict,
            "force": force,
            "calibration_mode": calibration_mode,
        })

    return state


def node_shadow(state: dict, agents: dict[str, Any]) -> dict:
    """Shadow trading comparison."""
    shadow = agents.get("shadow_trader")
    if shadow and shadow.has_shadow():
        state["shadow_comparison"] = shadow.run({
            "pipeline_results": state.get("pipeline_results", {}),
            "use_real_data": state.get("use_real_data", False),
        })
    return state


def node_report(state: dict, agents: dict[str, Any]) -> dict:
    """Generate CEO briefing."""
    chief = agents.get("chief_of_staff")
    if chief:
        state["ceo_briefing"] = chief.get_ceo_briefing()
    state["cycle_complete"] = True
    return state
