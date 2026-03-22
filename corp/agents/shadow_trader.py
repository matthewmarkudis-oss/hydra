"""Shadow Trader — runs dual pipeline instances for A/B config comparison.

Pure Python agent (zero LLM cost). Runs the primary (current best) config
alongside an experimental (shadow) config proposed by the Hedge Fund Director
or Contrarian agent. Compares backtesting results and reports to Chief of Staff.

Backtesting and training only.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from corp.agents.base_corp_agent import BaseCorpAgent
from corp.state.corporation_state import CorporationState
from corp.state.decision_log import DecisionLog

logger = logging.getLogger("corp.agents.shadow_trader")


class ShadowTrader(BaseCorpAgent):
    """Shadow Trader — dual pipeline runner for backtesting comparison.

    Responsibilities:
    1. Run primary config through PipelineOrchestrator
    2. Run shadow (experimental) config through a second orchestrator
    3. Compare backtesting results and report which config won
    4. Track consecutive shadow wins for promotion logic
    """

    def __init__(
        self,
        state: CorporationState,
        decision_log: DecisionLog,
    ):
        super().__init__("shadow_trader", state, decision_log)
        self._shadow_config_dict: dict | None = None
        self._comparison_history: list[dict] = []

    def set_shadow_config(self, config_dict: dict) -> None:
        """Set the experimental config to test against primary."""
        self._shadow_config_dict = config_dict
        logger.info(f"Shadow config set ({len(config_dict)} top-level keys)")

    def has_shadow(self) -> bool:
        """Check if a shadow config is set."""
        return self._shadow_config_dict is not None

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Run shadow comparison.

        Context keys:
        - primary_config: HydraConfig instance for primary
        - orchestrator_factory: Callable(config, use_real) -> PipelineOrchestrator
        - use_real_data: bool
        - pipeline_results: Results from the primary run (if already executed)
        """
        result = {
            "shadow_enabled": self.has_shadow(),
            "primary_return": 0.0,
            "shadow_return": 0.0,
            "shadow_won": False,
            "comparison": {},
        }

        if not self.has_shadow():
            result["status"] = "no_shadow_config"
            self._mark_run(result)
            return result

        pipeline_results = context.get("pipeline_results", {})
        primary_return = pipeline_results.get("best_return", 0)
        result["primary_return"] = primary_return

        # Run shadow config if factory is provided
        orchestrator_factory = context.get("orchestrator_factory")
        if orchestrator_factory and self._shadow_config_dict:
            shadow_return = self._run_shadow(
                orchestrator_factory,
                self._shadow_config_dict,
                context.get("use_real_data", False),
            )
            result["shadow_return"] = shadow_return
        else:
            # If no factory, we can't run the shadow — just compare config diffs
            result["status"] = "no_orchestrator_factory"
            self._mark_run(result)
            return result

        # Compare
        result["shadow_won"] = result["shadow_return"] > result["primary_return"]
        result["comparison"] = self._build_comparison(
            result["primary_return"],
            result["shadow_return"],
        )

        # Record in state
        self.state.record_shadow_result(
            primary_return=result["primary_return"],
            shadow_return=result["shadow_return"],
        )

        # Log and message
        outcome = "shadow_won" if result["shadow_won"] else "primary_won"
        self.log_decision(
            "shadow_comparison",
            detail={
                "primary_return": result["primary_return"],
                "shadow_return": result["shadow_return"],
                "winner": outcome,
            },
            outcome=outcome,
        )

        # Notify Chief of Staff
        should_promote = self.state.should_promote_shadow()
        if should_promote:
            self.send_message(
                "chief_of_staff",
                "proposal",
                {
                    "action": "promote_shadow",
                    "shadow_return": result["shadow_return"],
                    "primary_return": result["primary_return"],
                    "consecutive_wins": self.state.get_full_state().get(
                        "shadow_consecutive_wins", 0
                    ),
                },
                priority=4,
            )
            result["promotion_recommended"] = True
        else:
            self.send_message(
                "chief_of_staff",
                "report",
                {
                    "shadow_won": result["shadow_won"],
                    "primary_return": result["primary_return"],
                    "shadow_return": result["shadow_return"],
                },
                priority=2,
            )

        # Track comparison history
        self._comparison_history.append({
            "primary_return": result["primary_return"],
            "shadow_return": result["shadow_return"],
            "shadow_won": result["shadow_won"],
        })
        self._comparison_history = self._comparison_history[-20:]

        self._mark_run(result)
        return result

    def _run_shadow(
        self,
        orchestrator_factory,
        shadow_config_dict: dict,
        use_real_data: bool,
    ) -> float:
        """Execute the shadow pipeline and return best return."""
        try:
            from hydra.config.schema import HydraConfig

            shadow_config = HydraConfig(**shadow_config_dict)
            shadow_orchestrator = orchestrator_factory(shadow_config, use_real_data)

            logger.info("Running shadow pipeline...")
            t0 = time.time()
            shadow_results = shadow_orchestrator.run()
            elapsed = time.time() - t0
            logger.info(f"Shadow pipeline completed in {elapsed:.1f}s")

            # Extract best return from shadow results
            validation = shadow_results.get("validation", {})
            agent_results = validation.get("agent_results", {})
            best_return = 0.0
            for _name, metrics in agent_results.items():
                total_return = metrics.get("total_return", 0)
                if total_return > best_return:
                    best_return = total_return

            return best_return

        except Exception as e:
            logger.error(f"Shadow pipeline failed: {e}")
            self.log_decision(
                "shadow_pipeline_error",
                detail={"error": str(e)},
                outcome="failed",
            )
            return 0.0

    def _build_comparison(
        self, primary_return: float, shadow_return: float
    ) -> dict[str, Any]:
        """Build a detailed comparison dict."""
        diff = shadow_return - primary_return
        pct_improvement = (diff / max(abs(primary_return), 0.0001)) * 100

        return {
            "primary_return_pct": round(primary_return * 100, 2),
            "shadow_return_pct": round(shadow_return * 100, 2),
            "difference_pct": round(diff * 100, 2),
            "pct_improvement": round(pct_improvement, 1),
            "winner": "shadow" if shadow_return > primary_return else "primary",
            "total_comparisons": len(self._comparison_history) + 1,
            "shadow_win_rate": self._shadow_win_rate(),
        }

    def _shadow_win_rate(self) -> float:
        """Calculate shadow win rate from history."""
        if not self._comparison_history:
            return 0.0
        wins = sum(1 for c in self._comparison_history if c["shadow_won"])
        return round(wins / len(self._comparison_history) * 100, 1)

    def clear_shadow(self) -> None:
        """Clear the shadow config (after promotion or rejection)."""
        self._shadow_config_dict = None
        logger.info("Shadow config cleared")
