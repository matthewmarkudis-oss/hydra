"""Senior Code Developer — config regression prevention and changelog review.

Pure Python agent (zero LLM cost). Maintains a blacklist of failed configs,
validates proposed configs before pipeline runs, and detects regressions
by comparing results against historical best.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from corp.agents.base_corp_agent import BaseCorpAgent
from corp.config.ticker_universe import TICKER_TO_SECTOR, TickerSelector
from corp.state.config_blacklist import ConfigBlacklist
from corp.state.corporation_state import CorporationState
from corp.state.decision_log import DecisionLog

logger = logging.getLogger("corp.agents.senior_dev")


class SeniorDev(BaseCorpAgent):
    """Senior Code Developer — prevents config regressions.

    Responsibilities:
    1. Validate configs against blacklist before pipeline runs
    2. Detect regressions by comparing results to historical best
    3. Track config history to prevent reverting to failed configs
    4. Review proposed patches from Hedge Fund Director
    """

    def __init__(
        self,
        state: CorporationState,
        decision_log: DecisionLog,
        blacklist: ConfigBlacklist,
        best_config_path: str = "logs/meta_optimize_best.yaml",
        meta_optimize_log: str = "logs/meta_optimize.jsonl",
    ):
        super().__init__("senior_dev", state, decision_log)
        self.blacklist = blacklist
        self._best_config_path = Path(best_config_path)
        self._meta_log_path = Path(meta_optimize_log)
        self._best_fitness: float | None = None
        self._config_history: list[dict] = []

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Run senior dev checks.

        Context keys:
        - config_dict: The proposed HydraConfig dict
        - pipeline_results: Results from last pipeline run (if post-run)
        - proposed_patch: Config patch from Hedge Fund Director (if reviewing)
        """
        result = {
            "checks_passed": True,
            "warnings": [],
            "vetoes": [],
            "config_hash": "",
        }

        config_dict = context.get("config_dict", {})

        # 1. Compute config hash
        config_hash = self.blacklist.compute_hash(config_dict)
        result["config_hash"] = config_hash

        # 2. Check blacklist
        is_blocked, reason = self.blacklist.is_blacklisted(config_dict)
        if is_blocked:
            result["checks_passed"] = False
            result["vetoes"].append(f"Config {config_hash} is blacklisted: {reason}")
            self.log_decision("config_veto", detail={"hash": config_hash, "reason": reason}, outcome="blocked")
            self.send_message("chief_of_staff", "veto", {"config_hash": config_hash, "reason": reason}, priority=5)
            self._mark_run(result)
            return result

        # 3. Check for duplicate of recent configs
        duplicate = self._check_recent_duplicate(config_hash)
        if duplicate:
            result["warnings"].append(
                f"Config {config_hash} was run recently (run #{duplicate}). "
                "Consider whether re-running is worthwhile."
            )

        # 4. Review proposed patch (if any)
        proposed_patch = context.get("proposed_patch")
        if proposed_patch:
            # Ticker change proposals get specialized review
            if context.get("proposal_type") == "ticker_change" or "ticker_metadata" in context:
                ticker_issues = self._review_ticker_patch(proposed_patch, config_dict, context)
                if ticker_issues:
                    result["warnings"].extend(ticker_issues)
            else:
                patch_issues = self._review_patch(proposed_patch, config_dict)
                if patch_issues:
                    result["warnings"].extend(patch_issues)

        # 5. Post-run regression check
        pipeline_results = context.get("pipeline_results", {})
        if pipeline_results and pipeline_results.get("best_return") is not None:
            regression = self._check_regression(pipeline_results, config_dict, config_hash)
            if regression:
                result["warnings"].append(regression)

        # 6. Track this config in history
        self._config_history.append({
            "hash": config_hash,
            "run_number": self.state.get_full_state().get("pipeline_run_count", 0),
        })
        # Keep last 50
        self._config_history = self._config_history[-50:]

        self.log_decision(
            "config_approved" if result["checks_passed"] else "config_blocked",
            detail={"hash": config_hash, "warnings": result["warnings"]},
            outcome="approved" if result["checks_passed"] else "blocked",
        )

        self._mark_run(result)
        return result

    def _check_recent_duplicate(self, config_hash: str) -> int | None:
        """Check if this config was run in the last 10 runs."""
        for entry in self._config_history[-10:]:
            if entry["hash"] == config_hash:
                return entry["run_number"]
        return None

    def _review_patch(self, patch: dict, current_config: dict) -> list[str]:
        """Review a proposed config patch for potential issues."""
        issues = []

        # Check for extreme parameter changes
        if "reward" in patch:
            reward_patch = patch["reward"]

            # Transaction penalty going to zero → agents may overtrade
            if reward_patch.get("transaction_penalty", 999) < 0.005:
                issues.append(
                    "Transaction penalty near zero — agents may overtrade. "
                    "Consider minimum 0.01."
                )

            # Drawdown penalty going very low → reckless behavior
            if reward_patch.get("drawdown_penalty", 999) < 0.1:
                issues.append(
                    "Drawdown penalty below 0.1 — agents may take extreme risks. "
                    "Historical data shows penalties below 0.1 produce inconsistent results."
                )

            # Reward scale too high → gradient instability
            if reward_patch.get("reward_scale", 0) > 500:
                issues.append(
                    "Reward scale above 500 risks gradient instability. "
                    "Recommended range: 50-300."
                )

        if "env" in patch:
            env_patch = patch["env"]

            # Max position too concentrated
            if env_patch.get("max_position_pct", 0) > 0.60:
                issues.append(
                    "Max position above 60% allows extreme concentration. "
                    "This violated risk guidelines in previous runs."
                )

            # Max drawdown too permissive
            if env_patch.get("max_drawdown_pct", 0) > 0.40:
                issues.append(
                    "Max drawdown above 40% is extremely permissive. "
                    "Consider the impact on starting capital of $2,500."
                )

        return issues

    def _check_regression(
        self, pipeline_results: dict, config_dict: dict, config_hash: str
    ) -> str | None:
        """Check if results are worse than the historical best.

        If significantly worse, blacklists the config.
        """
        best_return = pipeline_results.get("best_return", 0)

        # Load historical best if not cached
        if self._best_fitness is None:
            self._best_fitness = self._load_best_fitness()

        if self._best_fitness is not None and best_return < self._best_fitness * 0.5:
            # Performance dropped by more than 50% → blacklist
            self.blacklist.add(
                config_dict,
                reason=f"Regression: best_return={best_return:.4f} vs historical best={self._best_fitness:.4f}",
                metrics={"best_return": best_return, "historical_best": self._best_fitness},
            )
            return (
                f"REGRESSION DETECTED: Return {best_return:.4f} is less than half "
                f"of historical best {self._best_fitness:.4f}. Config {config_hash} blacklisted."
            )

        # Update best if this run is better
        if self._best_fitness is None or best_return > self._best_fitness:
            self._best_fitness = best_return

        return None

    def _load_best_fitness(self) -> float | None:
        """Load the best historical fitness from meta-optimizer results."""
        if not self._best_config_path.exists():
            return None

        try:
            import yaml
            with open(self._best_config_path) as f:
                best_config = yaml.safe_load(f)
            return best_config.get("best_fitness", best_config.get("value"))
        except Exception:
            return None

    def _review_ticker_patch(
        self, patch: dict, current_config: dict, context: dict
    ) -> list[str]:
        """Review a ticker change proposal for potential issues.

        Validates:
        1. Observation space compatibility (17*N + 5)
        2. Churn rate warnings
        3. Sector concentration limits
        4. Duplicate tickers
        5. Recent ticker change history
        """
        issues = []
        data_patch = patch.get("data", {})
        proposed_tickers = data_patch.get("tickers", [])

        if not proposed_tickers:
            issues.append("Ticker proposal contains no tickers.")
            return issues

        # 1. Observation space check (17 features per stock + 5 global)
        n = len(proposed_tickers)
        obs_dim = 17 * n + 5
        if obs_dim > 17 * 500 + 5:
            issues.append(
                f"Obs space {obs_dim} exceeds maximum (17*500+5=8505). "
                f"Reduce to <= 500 tickers."
            )

        # 2. Duplicate check
        dupes = [t for t in proposed_tickers if proposed_tickers.count(t) > 1]
        if dupes:
            issues.append(f"Duplicate tickers found: {set(dupes)}")

        # 3. Churn rate warning
        ticker_meta = context.get("ticker_metadata", {})
        churn = ticker_meta.get("churn", {})
        churn_pct = churn.get("churn_pct", 0)
        if churn_pct > 0.6:
            issues.append(
                f"High churn ({churn_pct:.0%}) — over 60% of tickers changing. "
                f"This will require full retraining. Consider a more gradual transition."
            )

        # 4. Sector concentration check
        dist = TickerSelector.get_sector_distribution(proposed_tickers)
        total = len(proposed_tickers)
        for sector, count in dist.items():
            if count / total > 0.40:
                issues.append(
                    f"Sector '{sector}' is {count}/{total} ({count/total:.0%}) of portfolio — "
                    f"over 40% concentration. Consider diversifying."
                )

        # 5. All tickers must be in known universe
        unknown = [t for t in proposed_tickers if t not in TICKER_TO_SECTOR]
        if unknown:
            issues.append(
                f"Unknown tickers not in universe: {unknown}. "
                f"Only tickers from ticker_universe.py are supported."
            )

        # 6. Recent ticker change history check
        corp_state = self.state.get_full_state()
        ticker_history = corp_state.get("ticker_change_history", [])
        from datetime import datetime, timedelta
        recent_changes = [
            h for h in ticker_history
            if h.get("timestamp", "") > (datetime.now() - timedelta(days=7)).isoformat()
        ]
        if len(recent_changes) >= 2:
            issues.append(
                f"WARNING: {len(recent_changes)} ticker changes in the past 7 days. "
                f"Frequent ticker changes destabilize training. Consider waiting."
            )

        # 7. Retraining cost warning (always)
        current_tickers = current_config.get("data", {}).get("tickers", [])
        if len(proposed_tickers) != len(current_tickers) or churn_pct > 0:
            issues.append(
                f"RETRAINING REQUIRED: Ticker change from {len(current_tickers)} to "
                f"{len(proposed_tickers)} tickers. Obs space changes from "
                f"{17*len(current_tickers)+5} to {obs_dim}. "
                f"All agent checkpoints will be invalidated."
            )

        return issues

    def populate_blacklist(self, threshold: float = -0.5) -> int:
        """Pre-populate blacklist from meta-optimizer history."""
        if self._meta_log_path.exists():
            return self.blacklist.populate_from_meta_optimize(
                str(self._meta_log_path), threshold
            )
        return 0
