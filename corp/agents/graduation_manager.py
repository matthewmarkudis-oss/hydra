"""Graduation Manager — bridges backtesting to forward testing.

Reads ATHENA validation results, ranks agents by CHIMERA fitness,
selects top-K for forward testing, and manages the graduation
proposal/approval flow through the CEO.

Backtesting and training research only.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from corp.agents.base_corp_agent import BaseCorpAgent
from corp.state.corporation_state import CorporationState
from corp.state.decision_log import DecisionLog

logger = logging.getLogger("corp.agents.graduation_manager")


class GraduationManager(BaseCorpAgent):
    """Graduation Manager — bridges backtest validation to forward testing.

    Responsibilities:
    1. Read ATHENA validation results from training state
    2. Rank passed agents by CHIMERA fitness
    3. Select top-K agents for forward testing
    4. Submit graduation proposals for CEO approval
    5. Evaluate forward test results and produce graduation report
    """

    def __init__(
        self,
        state: CorporationState,
        decision_log: DecisionLog,
        training_state_path: str = "logs/hydra_training_state.json",
    ):
        super().__init__("graduation_manager", state, decision_log)
        self._training_state_path = Path(training_state_path)

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Evaluate readiness for forward testing.

        Context keys:
        - force: bool — skip readiness checks
        - forward_test_config: dict — forward test configuration
        """
        result = {
            "ready_agents": [],
            "proposal_submitted": False,
            "reason": "",
        }

        # Check if forward testing is enabled
        ft_config = context.get("forward_test_config", {})
        if not ft_config.get("enabled", False):
            result["reason"] = "Forward testing is not enabled in config."
            self._mark_run(result)
            return result

        # Evaluate readiness
        readiness = self.evaluate_readiness()

        if not readiness["ready"]:
            result["reason"] = readiness["reason"]
            self._mark_run(result)
            return result

        # We have agents ready — submit graduation proposal
        top_agents = readiness["top_agents"]
        max_agents = ft_config.get("max_agents", 3)
        candidates = top_agents[:max_agents]

        result["ready_agents"] = candidates
        result["all_passed_agents"] = readiness["all_passed"]

        # Submit proposal
        self._submit_graduation_proposal(candidates, ft_config)
        result["proposal_submitted"] = True

        self.log_decision(
            "graduation_proposal",
            detail={
                "candidates": [a["name"] for a in candidates],
                "all_passed": len(readiness["all_passed"]),
            },
            outcome="proposal_submitted",
        )

        self.send_message(
            "chief_of_staff",
            "report",
            {
                "type": "graduation_readiness",
                "candidates": len(candidates),
                "all_passed": len(readiness["all_passed"]),
            },
            priority=3,
        )

        self._mark_run(result)
        return result

    def evaluate_readiness(self) -> dict[str, Any]:
        """Read ATHENA results and determine which agents are ready.

        Returns:
            {
                "ready": bool,
                "reason": str,
                "all_passed": list of agent names that passed ATHENA,
                "top_agents": list of dicts with name, fitness, backtest metrics,
            }
        """
        # Load training state
        ts = self._load_training_state()
        if not ts:
            return {
                "ready": False,
                "reason": "No training state found.",
                "all_passed": [],
                "top_agents": [],
            }

        # Get latest generation
        gens = ts.get("generations", [])
        if not gens:
            return {
                "ready": False,
                "reason": "No training generations found.",
                "all_passed": [],
                "top_agents": [],
            }

        latest = gens[-1]
        validation = latest.get("validation", {})
        passed_agents = validation.get("passed_agents", [])

        if not passed_agents:
            # Check top-level validation
            top_validation = ts.get("validation", {})
            passed_agents = top_validation.get("passed_agents", [])

        if not passed_agents:
            return {
                "ready": False,
                "reason": (
                    "No agents have passed ATHENA validation. "
                    "Run validation phase first."
                ),
                "all_passed": [],
                "top_agents": [],
            }

        # Rank by CHIMERA fitness
        top_agents = self._rank_by_fitness(passed_agents, latest, validation)

        if not top_agents:
            return {
                "ready": False,
                "reason": "Passed agents could not be ranked by fitness.",
                "all_passed": passed_agents,
                "top_agents": [],
            }

        return {
            "ready": True,
            "reason": f"{len(top_agents)} agents ready for forward testing.",
            "all_passed": passed_agents,
            "top_agents": top_agents,
        }

    def _rank_by_fitness(
        self,
        passed_agents: list[str],
        generation: dict,
        validation: dict,
    ) -> list[dict]:
        """Rank passed agents by CHIMERA fitness score.

        Falls back to eval_scores if fitness module is unavailable.
        """
        eval_scores = generation.get("eval_scores", {})
        agent_results = validation.get("agent_results", {})

        ranked = []
        for name in passed_agents:
            # Get backtest metrics from validation results
            agent_val = agent_results.get(name, {})

            entry = {
                "name": name,
                "eval_score": eval_scores.get(name, 0),
                "backtest_metrics": {
                    "sharpe": agent_val.get("deflated_sharpe", agent_val.get("sharpe", 0)),
                    "max_drawdown": agent_val.get("max_drawdown", 0),
                    "win_rate": agent_val.get("win_rate", 0),
                    "profit_factor": agent_val.get("profit_factor", 0),
                    "total_return": agent_val.get("total_return", 0),
                    "wfe": agent_val.get("wfe", 0),
                },
            }

            # Try CHIMERA fitness computation
            try:
                from hydra.evaluation.fitness import AgentFitness, compute_fitness
                af = AgentFitness(
                    agent_name=name,
                    sharpe=entry["backtest_metrics"]["sharpe"],
                    max_drawdown=entry["backtest_metrics"]["max_drawdown"],
                    profit_factor=entry["backtest_metrics"]["profit_factor"],
                    wfe=entry["backtest_metrics"]["wfe"],
                    total_return=entry["backtest_metrics"]["total_return"],
                    win_rate=entry["backtest_metrics"]["win_rate"],
                )
                fitness, breakdown = compute_fitness(af)
                entry["chimera_fitness"] = fitness
                entry["fitness_breakdown"] = breakdown
            except Exception:
                # Fallback to eval score
                entry["chimera_fitness"] = entry["eval_score"]
                entry["fitness_breakdown"] = {}

            ranked.append(entry)

        # Sort by CHIMERA fitness descending
        ranked.sort(key=lambda x: x["chimera_fitness"], reverse=True)
        return ranked

    def _submit_graduation_proposal(
        self, candidates: list[dict], ft_config: dict
    ) -> None:
        """Submit a graduation proposal for CEO approval."""
        agent_summary = []
        for c in candidates:
            metrics = c.get("backtest_metrics", {})
            agent_summary.append(
                f"{c['name']} (fitness={c['chimera_fitness']:.4f}, "
                f"Sharpe={metrics.get('sharpe', 0):.2f}, "
                f"DD={metrics.get('max_drawdown', 0):.1%})"
            )

        self.state.submit_proposal({
            "type": "graduation",
            "source": "graduation_manager",
            "memo": (
                f"Forward test graduation: {len(candidates)} agent(s) passed ATHENA "
                f"and ranked by CHIMERA fitness. Ready for {ft_config.get('duration_days', 20)}-day "
                f"sandbox simulation with ${ft_config.get('initial_capital', 10000):,.0f} capital.\n"
                f"Candidates: {'; '.join(agent_summary)}"
            ),
            "candidates": [
                {
                    "name": c["name"],
                    "chimera_fitness": c["chimera_fitness"],
                    "backtest_metrics": c["backtest_metrics"],
                }
                for c in candidates
            ],
            "forward_test_config": ft_config,
            "confidence": 0.8,
            "risk": "medium",
        })

    def evaluate_forward_results(self, tracker) -> dict[str, Any]:
        """Read forward test results and produce graduation report.

        Args:
            tracker: ForwardTestTracker instance.

        Returns:
            Graduation report dict.
        """
        state = tracker.load_state()
        backtest_expectations = state.get("backtest_expectations", {})
        agents = state.get("agents", [])

        report = tracker.get_graduation_report(
            agents=agents,
            backtest_expectations=backtest_expectations,
            config=state.get("config", {}),
        )

        # Log results
        self.log_decision(
            "graduation_evaluation",
            detail={
                "graduated": report["graduated"],
                "extended": report["extended"],
                "failed": report["failed"],
            },
            outcome=report["summary"],
        )

        # Send report to CEO
        self.send_message(
            "chief_of_staff",
            "report",
            {
                "type": "graduation_results",
                "graduated": report["graduated"],
                "extended": report["extended"],
                "failed": report["failed"],
                "summary": report["summary"],
            },
            priority=4,
        )

        return report

    def _load_training_state(self) -> dict:
        """Load the training state JSON."""
        if not self._training_state_path.exists():
            return {}
        try:
            with open(self._training_state_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
