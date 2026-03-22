"""Strategy Distiller — derives reward weights from hedge fund factor analysis.

Pure Python, zero LLM cost. Runs as a corp agent in the Intelligence Division.
Downloads academic factor data (Fama-French, Fung-Hsieh), analyzes top fund
return profiles, and proposes calibrated reward weights for the RL training
pipeline.

Backtesting and training only.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from corp.agents.base_corp_agent import BaseCorpAgent
from corp.state.corporation_state import CorporationState
from corp.state.decision_log import DecisionLog

logger = logging.getLogger("corp.agents.strategy_distiller")


class StrategyDistiller(BaseCorpAgent):
    """Strategy Distiller — learns decision frameworks from hedge fund data.

    Responsibilities:
    1. Download academic factor data (FF5, FH7)
    2. Run reward calibration (factor mapping or inverse RL)
    3. Incorporate 13F consensus signal when available
    4. Submit calibrated reward weights as a corp proposal
    5. Track calibration history for trend analysis
    """

    def __init__(
        self,
        state: CorporationState,
        decision_log: DecisionLog,
        interval_hours: int = 168,  # Weekly
    ):
        super().__init__("strategy_distiller", state, decision_log)
        self._interval_hours = interval_hours
        self._last_run_time: datetime | None = None
        self._calibration_history: list[dict] = []

    def should_run(self) -> bool:
        """Check if enough time has passed since last run."""
        if self._last_run_time is None:
            return True
        elapsed = datetime.now() - self._last_run_time
        return elapsed > timedelta(hours=self._interval_hours)

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Run factor analysis and propose calibrated reward weights.

        Context keys:
        - config_dict: Current HydraConfig dict
        - force: bool — override schedule check
        - calibration_mode: "factor_mapping" | "constrained_opt" | "inverse_rl"
        - consensus_signal: dict — 13F consensus data (from Phase 3)
        """
        result: dict[str, Any] = {
            "calibrated": False,
            "proposed_weights": {},
            "factor_loadings": {},
            "calibration_mode": "factor_mapping",
            "report": {},
            "llm_used": False,
        }

        # Schedule check
        if not context.get("force", False) and not self.should_run():
            result["skipped"] = True
            return result

        config_dict = context.get("config_dict", {})
        current_reward = config_dict.get("reward", {})
        mode = context.get("calibration_mode", "factor_mapping")
        result["calibration_mode"] = mode

        # Step 1: Download factor data
        factor_data = self._load_factor_data()
        if factor_data is None:
            result["error"] = "Failed to load factor data"
            self._mark_run(result)
            return result

        ff5 = factor_data.get("ff5")
        fh7 = factor_data.get("fh7")

        # Step 2: Run calibration based on mode
        try:
            if mode == "inverse_rl":
                proposed = self._run_inverse_rl(ff5, context)
            elif mode == "constrained_opt":
                proposed = self._run_constrained_opt(ff5, current_reward)
            else:
                proposed = self._run_factor_mapping(ff5, current_reward)
        except Exception as e:
            logger.warning(f"Calibration failed ({mode}): {e}")
            result["error"] = str(e)
            self._mark_run(result)
            return result

        if not proposed:
            result["error"] = "Calibration produced no results"
            self._mark_run(result)
            return result

        result["proposed_weights"] = proposed.get("weights", {})
        result["factor_loadings"] = proposed.get("loadings", {})
        result["report"] = proposed.get("report", {})
        result["calibrated"] = True

        # Step 3: Submit proposal
        if result["proposed_weights"]:
            self._submit_calibration_proposal(
                result["proposed_weights"],
                result["factor_loadings"],
                result["report"],
                mode,
            )

        # Track history
        self._calibration_history.append({
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "weights": result["proposed_weights"],
            "loadings": result["factor_loadings"],
        })
        self._calibration_history = self._calibration_history[-20:]

        self._last_run_time = datetime.now()

        self.log_decision(
            "reward_calibration",
            detail={
                "mode": mode,
                "proposed_weights": result["proposed_weights"],
                "factor_count": len(result["factor_loadings"]),
            },
            outcome="proposal_submitted",
        )

        self._mark_run(result)
        return result

    def _load_factor_data(self) -> dict | None:
        """Load factor data from FactorDataStore."""
        try:
            from hydra.distillation.factor_data import FactorDataStore

            store = FactorDataStore()
            ff5 = store.get_fama_french_5()
            fh7 = store.get_fung_hsieh_7()

            if ff5 is None:
                logger.warning("Could not load Fama-French 5-factor data")
                return None

            data = {"ff5": ff5, "fh7": fh7}
            logger.info(
                f"Factor data loaded: FF5={len(ff5)} rows"
                + (f", FH7={len(fh7)} rows" if fh7 is not None else "")
            )
            return data

        except Exception as e:
            logger.warning(f"Factor data loading failed: {e}")
            return None

    def _run_factor_mapping(
        self, ff5, current_reward: dict
    ) -> dict | None:
        """Phase 1: Direct factor-to-reward mapping."""
        from hydra.distillation.reward_calibrator import RewardCalibrator

        calibrator = RewardCalibrator()
        loadings = calibrator.compute_target_profile(ff5)
        weights = calibrator.map_to_reward_config(loadings, current_reward)
        report = calibrator.get_calibration_report(
            loadings, weights, current_reward
        )

        return {
            "weights": weights,
            "loadings": loadings,
            "report": report,
        }

    def _run_constrained_opt(
        self, ff5, current_reward: dict
    ) -> dict | None:
        """Phase 1 advanced: Constrained optimization."""
        from hydra.distillation.reward_calibrator import RewardCalibrator

        calibrator = RewardCalibrator()
        weights = calibrator.run_constrained_optimization(ff5)
        loadings = calibrator.compute_target_profile(ff5)
        report = calibrator.get_calibration_report(
            loadings, weights, current_reward
        )

        return {
            "weights": weights,
            "loadings": loadings,
            "report": report,
        }

    def _run_inverse_rl(self, ff5, context: dict) -> dict | None:
        """Phase 4: Inverse RL reward inference."""
        try:
            from hydra.distillation.inverse_rl import InverseRLCalibrator
        except ImportError:
            logger.warning("Inverse RL module not available")
            return None

        # Try to get 13F filing history for expert trajectories
        filings_history = context.get("filings_history", [])

        if not filings_history:
            # Try loading from 13F parser
            try:
                from trading_agents.data.sec_13f_parser import SEC13FParser

                parser = SEC13FParser()
                current = parser.get_all_filings()
                if current:
                    filings_history = [current]
            except Exception as e:
                logger.debug(f"Could not load 13F data for IRL: {e}")

        irl = InverseRLCalibrator()
        irl.fit(filings_history, ff5)
        weights = irl.config
        report = irl.report

        loadings = {}
        if ff5 is not None and len(ff5) > 0:
            from hydra.distillation.reward_calibrator import RewardCalibrator

            calibrator = RewardCalibrator()
            loadings = calibrator.compute_target_profile(ff5)

        return {
            "weights": weights,
            "loadings": loadings,
            "report": report,
        }

    def _submit_calibration_proposal(
        self,
        weights: dict,
        loadings: dict,
        report: dict,
        mode: str,
    ) -> None:
        """Submit calibrated reward weights as a corp proposal."""
        # Build readable description
        changes = []
        for param, value in weights.items():
            changes.append(f"{param}={value}")
        changes_str = ", ".join(changes)

        proposal = {
            "type": "reward_calibration",
            "source": "strategy_distiller",
            "priority": "medium",
            "description": (
                f"Factor-derived reward calibration ({mode}): {changes_str}. "
                f"Based on Fama-French 5-factor analysis of top hedge fund "
                f"return profiles."
            ),
            "patch": {"reward": weights},
            "factor_loadings": loadings,
            "calibration_report": report,
            "calibration_mode": mode,
            "confidence": 0.6 if mode == "factor_mapping" else 0.5,
            "risk": "medium",
        }

        try:
            self.state.add_proposal(proposal)
            logger.info(
                f"Reward calibration proposal submitted ({mode}): "
                f"{changes_str}"
            )
        except Exception as e:
            logger.debug(f"Could not submit calibration proposal: {e}")

        # Notify HedgeFundDirector
        self.send_message(
            "hedge_fund_director",
            "calibration_result",
            {
                "weights": weights,
                "loadings": loadings,
                "mode": mode,
            },
        )

    def get_calibration_summary(self) -> dict[str, Any]:
        """Get a summary for the CEO dashboard."""
        latest = (
            self._calibration_history[-1]
            if self._calibration_history
            else None
        )
        return {
            "total_calibrations": len(self._calibration_history),
            "latest": latest,
            "last_run": self._last_run_time.isoformat()
            if self._last_run_time
            else None,
        }
