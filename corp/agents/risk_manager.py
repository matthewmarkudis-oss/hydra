"""Risk Manager — circuit breaker enforcement for forward-test / paper trading.

Consumes CircuitBreakerAction objects from CHIMERA diagnostics and applies
weight reductions, position flattening, or agent shutdown.

In backtesting mode, actions are logged but not enforced (training explores
freely). In forward-test/paper mode, actions are enforced to protect capital.

Zero LLM cost — pure Python.

Backtesting and training only.
"""

from __future__ import annotations

import logging
from typing import Any

from corp.agents.base_corp_agent import BaseCorpAgent
from corp.state.corporation_state import CorporationState
from corp.state.decision_log import DecisionLog

logger = logging.getLogger("corp.agents.risk_manager")


class RiskManager(BaseCorpAgent):
    """Enforces circuit breaker actions from CHIMERA diagnostics.

    Mid-training hook: reads circuit_breaker_actions from generation results,
    computes weight overrides, and returns them for PopulationTrainer to apply.

    Standalone mode: post-analysis risk summary for the CEO dashboard.
    """

    def __init__(
        self,
        state: CorporationState,
        decision_log: DecisionLog,
        enforce: bool = False,
    ):
        super().__init__("risk_manager", state, decision_log)
        self.enforce = enforce  # True in forward-test/paper, False in backtesting
        self._intervention_history: list[dict] = []
        self._active_reductions: dict[str, float] = {}  # agent → reduction applied

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Standalone risk analysis for corp graph.

        Summarizes current risk state: active circuit breakers, intervention
        history, agent risk scores.
        """
        result = self.get_risk_summary()
        self._mark_run(result)
        return result

    def on_generation_complete(
        self,
        generation: int,
        generation_results: list[dict],
    ) -> dict[str, Any] | None:
        """Mid-training hook: process circuit breaker actions.

        Called after each generation by the intervention hook chain in
        train_phase.py. Reads circuit_breaker_actions from the latest
        generation's diagnosis and returns weight_overrides.

        Returns:
            Dict with weight_overrides and alerts, or None if no action needed.
        """
        if not generation_results:
            return None

        latest = generation_results[-1]
        diagnosis = latest.get("diagnosis", {})
        cb_actions = diagnosis.get("circuit_breaker_actions", [])

        if not cb_actions:
            return None

        weight_overrides: dict[str, float] = {}
        alerts: list[dict] = []

        for action in cb_actions:
            a_type = action.get("action", "")
            target = action.get("target")
            reduction = action.get("reduction_pct", 0.0)

            if a_type == "alert":
                alerts.append({
                    "generation": generation,
                    "type": "circuit_breaker_alert",
                    "message": f"CHIMERA alert: {action.get('reason', 'unknown')}",
                })

            elif a_type == "reduce_allocation" and target:
                # Compute new weight: current * (1 - reduction)
                # We don't have direct pool access here, so we return
                # the reduction factor and let PopulationTrainer apply it.
                current_reduction = self._active_reductions.get(target, 0.0)
                new_reduction = min(current_reduction + reduction, 0.90)  # cap at 90% cut
                self._active_reductions[target] = new_reduction

                # Weight override as a scale factor (e.g., 0.75 = 25% cut)
                scale = 1.0 - new_reduction
                weight_overrides[target] = round(scale, 4)

                alerts.append({
                    "generation": generation,
                    "type": "reduce_allocation",
                    "target": target,
                    "reduction_pct": new_reduction,
                    "message": f"Reduced '{target}' allocation by {new_reduction:.0%}",
                })

                self.log_decision(
                    "circuit_breaker_reduce",
                    detail={"target": target, "reduction": new_reduction, "generation": generation},
                    outcome="applied" if self.enforce else "logged",
                )

            elif a_type == "flatten_positions" and target:
                alerts.append({
                    "generation": generation,
                    "type": "flatten_positions",
                    "target": target,
                    "message": f"Flatten signal for '{target}' (requires broker hook)",
                })

                self.log_decision(
                    "circuit_breaker_flatten",
                    detail={"target": target, "generation": generation},
                    outcome="signal_sent" if self.enforce else "logged",
                )

            elif a_type == "shutdown_agent" and target:
                # Full shutdown = weight to near-zero
                weight_overrides[target] = 0.01
                self._active_reductions[target] = 0.99

                alerts.append({
                    "generation": generation,
                    "type": "shutdown_agent",
                    "target": target,
                    "message": f"Shutdown signal for '{target}' — weight set to 1%",
                })

                self.log_decision(
                    "circuit_breaker_shutdown",
                    detail={"target": target, "generation": generation},
                    outcome="applied" if self.enforce else "logged",
                )

        if not weight_overrides and not alerts:
            return None

        intervention = {
            "alerts": alerts,
        }

        # Only apply weight overrides in enforce mode (forward-test/paper).
        # In backtesting, log the actions but don't constrain training.
        if self.enforce and weight_overrides:
            intervention["weight_overrides"] = weight_overrides
        elif weight_overrides:
            logger.info(
                f"  RISK MANAGER (backtest mode): would reduce {list(weight_overrides.keys())} "
                f"— logged only, not enforced"
            )

        # Record intervention
        self._intervention_history.append({
            "generation": generation,
            "actions_processed": len(cb_actions),
            "weight_overrides": weight_overrides,
            "alerts_count": len(alerts),
        })

        return intervention

    def get_risk_summary(self) -> dict[str, Any]:
        """Risk state summary for the CEO dashboard."""
        return {
            "enforce_mode": self.enforce,
            "active_reductions": dict(self._active_reductions),
            "total_interventions": len(self._intervention_history),
            "recent_interventions": self._intervention_history[-5:],
            "agents_at_risk": [
                {"agent": name, "reduction_pct": red}
                for name, red in self._active_reductions.items()
                if red > 0
            ],
        }

    def clear_reduction(self, agent_name: str) -> None:
        """CEO can clear an active reduction (restore full allocation)."""
        if agent_name in self._active_reductions:
            del self._active_reductions[agent_name]
            logger.info(f"Risk reduction cleared for '{agent_name}'")
