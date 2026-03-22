"""Conditional edge functions for the corporation graph.

Each function takes the current graph state and returns a string indicating
which node to route to next. Used by both the built-in CorpGraph and
LangGraph StateGraph conditional edges.

Backtesting and training only.
"""

from __future__ import annotations

from typing import Any


def should_run_pipeline(state: dict[str, Any]) -> str:
    """Determine if pipeline should run or be skipped.

    Returns:
        "pipeline" to run the pipeline, "post_analysis" to skip to analysis.
    """
    if state.get("skip_pipeline", False):
        return "post_analysis"

    if state.get("blacklist_check", {}).get("blocked", False):
        return "report"  # Skip everything, go straight to report

    if state.get("orchestrator") is None:
        return "post_analysis"

    return "pipeline"


def should_fire_contrarian(state: dict[str, Any]) -> bool:
    """Determine if the Contrarian agent should activate.

    Fires when:
    - Any agent passed validation
    - Best return exceeds threshold (0.5 by default)
    - Convergence declared
    """
    pipeline_results = state.get("pipeline_results", {})

    if pipeline_results.get("passed_count", 0) > 0:
        return True

    if pipeline_results.get("best_return", 0) > 0.5:
        return True

    if pipeline_results.get("convergence_declared", False):
        return True

    return False


def should_run_intelligence(state: dict[str, Any]) -> bool:
    """Determine if Intelligence Division agents should run.

    They have time-based schedules but can be forced.
    """
    return state.get("force_all_agents", False) or True  # Always check (agents enforce their own schedule)


def should_run_shadow(state: dict[str, Any]) -> bool:
    """Determine if shadow comparison should run."""
    # Shadow runs if there's a shadow config set
    # The ShadowTrader agent checks this internally
    return True


def route_after_pre_flight(state: dict[str, Any]) -> str:
    """Route after pre-flight checks.

    Returns the next node name.
    """
    if state.get("blacklist_check", {}).get("blocked", False):
        return "report"  # Config blocked, skip to report

    if state.get("skip_pipeline", False):
        return "intelligence"  # Analysis-only, go to intelligence first

    return "pipeline"


def route_after_pipeline(state: dict[str, Any]) -> str:
    """Route after pipeline execution.

    Returns the next node name.
    """
    if state.get("pipeline_error"):
        return "report"  # Pipeline failed, skip to report

    return "post_analysis"


def route_after_analysis(state: dict[str, Any]) -> str:
    """Route after strategy analysis.

    Returns the next node name.
    """
    return "shadow" if should_run_shadow(state) else "report"
