"""Operations Monitor — real-time training watchdog.

Runs after every generation via mid-training hooks.  Scans for
anti-patterns, applies auto-fixes, and escalates issues to CEO.

Zero LLM cost — pure Python analysis.

Backtesting and training only.
"""

from __future__ import annotations

import logging
from typing import Any

from corp.agents.base_corp_agent import BaseCorpAgent
from corp.config.anti_patterns import scan_all
from corp.state.corporation_state import CorporationState
from corp.state.decision_log import DecisionLog

logger = logging.getLogger("corp.agents.operations_monitor")


class OperationsMonitor(BaseCorpAgent):
    """Watchdog agent that monitors training health every generation.

    Responsibilities:
    1. Run anti-pattern scan after each generation
    2. Auto-apply fixes for patterns marked auto_fix=True
    3. Submit proposals for non-auto patterns requiring CEO approval
    4. Track intervention history to avoid repeating failed fixes
    5. Send alerts to CEO dashboard via corp state
    """

    def __init__(
        self,
        state: CorporationState,
        decision_log: DecisionLog,
    ):
        super().__init__("operations_monitor", state, decision_log)
        self._intervention_history: list[dict] = []
        self._alerts_raised: list[dict] = []
        self._auto_fixes_applied: list[dict] = []
        self._generation_health: list[dict] = []
        self._suppressed_patterns: set[str] = set()

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Run operations monitoring on the current training state.

        Context keys:
        - training_state: Full training state dict with generations list
        - generation: Current generation number
        - config: Current HydraConfig or dict
        """
        training_state = context.get("training_state", {})
        generation = context.get("generation", 0)
        config = context.get("config", {})

        result: dict[str, Any] = {
            "generation": generation,
            "patterns_detected": [],
            "auto_fixes": [],
            "proposals": [],
            "health_score": 1.0,
            "alerts": [],
        }

        # Run anti-pattern scan
        triggered = scan_all(training_state)

        if not triggered:
            result["health_score"] = 1.0
            self._record_health(generation, result)
            self._mark_run(result)
            return result

        result["patterns_detected"] = triggered

        # Process each triggered pattern
        for pattern in triggered:
            pattern_name = pattern.get("pattern", "unknown")

            # Skip suppressed patterns (already fixed or CEO dismissed)
            if pattern_name in self._suppressed_patterns:
                continue

            severity = pattern.get("severity", "info")
            auto_fix = pattern.get("auto_fix", False)
            patch = pattern.get("patch", {})
            message = pattern.get("message", "")

            if auto_fix and patch:
                # Apply auto-fix immediately
                fix_record = self._apply_auto_fix(
                    pattern_name, patch, message, generation, config
                )
                result["auto_fixes"].append(fix_record)
                self._auto_fixes_applied.append(fix_record)
                logger.warning(
                    f"[Gen {generation}] AUTO-FIX applied for '{pattern_name}': "
                    f"{message}"
                )
            else:
                # Submit as proposal for CEO review
                proposal = self._submit_proposal(
                    pattern_name, severity, message, patch, generation
                )
                result["proposals"].append(proposal)
                logger.info(
                    f"[Gen {generation}] Proposal submitted for '{pattern_name}': "
                    f"{message}"
                )

            # Track alert
            alert = {
                "generation": generation,
                "pattern": pattern_name,
                "severity": severity,
                "message": message,
                "auto_fixed": auto_fix,
            }
            result["alerts"].append(alert)
            self._alerts_raised.append(alert)

        # Compute health score: 1.0 = perfect, degrades with issues
        result["health_score"] = self._compute_health_score(triggered)

        # Log the decision
        self.log_decision(
            "operations_scan",
            detail={
                "generation": generation,
                "patterns_found": len(triggered),
                "auto_fixes": len(result["auto_fixes"]),
                "proposals": len(result["proposals"]),
                "health_score": result["health_score"],
            },
            outcome="complete",
        )

        self._record_health(generation, result)
        self._mark_run(result)
        return result

    def on_generation_complete(
        self,
        generation: int,
        generation_results: list[dict],
        config: Any = None,
    ) -> dict[str, Any] | None:
        """Hook called after each generation completes.

        This is the entry point from the mid-training intervention system.
        Builds the training_state dict expected by anti-pattern detectors
        from the raw generation_results list.

        Returns intervention dict if action was taken, None otherwise.
        """
        # Build training_state from generation_results
        training_state = {"generations": generation_results}

        context = {
            "training_state": training_state,
            "generation": generation,
            "config": config,
        }

        result = self.run(context)

        # Return config patches if auto-fixes were applied
        if result.get("auto_fixes"):
            patches = {}
            for fix in result["auto_fixes"]:
                patches.update(fix.get("patch", {}))
            return {
                "type": "config_patch",
                "source": "operations_monitor",
                "generation": generation,
                "patches": patches,
                "alerts": result["alerts"],
            }

        # Return alerts even if no fixes (for dashboard display)
        if result.get("alerts"):
            return {
                "type": "alert_only",
                "source": "operations_monitor",
                "generation": generation,
                "alerts": result["alerts"],
                "health_score": result["health_score"],
            }

        return None

    def _apply_auto_fix(
        self,
        pattern_name: str,
        patch: dict,
        message: str,
        generation: int,
        config: Any,
    ) -> dict:
        """Apply an auto-fix patch and record it."""
        fix_record = {
            "pattern": pattern_name,
            "generation": generation,
            "patch": patch,
            "message": message,
            "applied": True,
        }

        # Try to apply patch to live config if it supports apply_patch
        if hasattr(config, "apply_patch"):
            try:
                config.apply_patch({"training": patch})
                fix_record["config_updated"] = True
            except Exception as e:
                logger.warning(f"Could not apply config patch: {e}")
                fix_record["config_updated"] = False
        else:
            fix_record["config_updated"] = False

        # Suppress this pattern to avoid re-triggering next gen
        self._suppressed_patterns.add(pattern_name)

        # Log to decision log
        self.log_decision(
            f"auto_fix_{pattern_name}",
            detail=fix_record,
            outcome="applied",
        )

        return fix_record

    def _submit_proposal(
        self,
        pattern_name: str,
        severity: str,
        message: str,
        patch: dict,
        generation: int,
    ) -> dict:
        """Submit a proposal for CEO review."""
        proposal = {
            "type": "operations_alert",
            "source": "operations_monitor",
            "priority": "high" if severity == "critical" else "medium",
            "description": f"[Gen {generation}] {message}",
            "pattern": pattern_name,
            "severity": severity,
            "patch": patch if patch else None,
            "generation": generation,
        }

        try:
            self.state.add_proposal(proposal)
        except Exception as e:
            logger.debug(f"Could not submit proposal: {e}")

        return proposal

    def _compute_health_score(self, triggered: list[dict]) -> float:
        """Compute a 0-1 health score from triggered patterns."""
        if not triggered:
            return 1.0

        penalties = {
            "critical": 0.3,
            "warning": 0.15,
            "info": 0.05,
        }

        score = 1.0
        for pattern in triggered:
            severity = pattern.get("severity", "info")
            score -= penalties.get(severity, 0.05)

        return max(0.0, round(score, 2))

    def _record_health(self, generation: int, result: dict) -> None:
        """Record health snapshot for trend analysis."""
        self._generation_health.append({
            "generation": generation,
            "health_score": result.get("health_score", 1.0),
            "patterns_count": len(result.get("patterns_detected", [])),
            "auto_fixes_count": len(result.get("auto_fixes", [])),
        })
        # Keep last 50
        self._generation_health = self._generation_health[-50:]

    def get_health_summary(self) -> dict[str, Any]:
        """Get a summary of operational health for the CEO dashboard."""
        return {
            "current_health": (
                self._generation_health[-1]["health_score"]
                if self._generation_health else 1.0
            ),
            "total_alerts": len(self._alerts_raised),
            "total_auto_fixes": len(self._auto_fixes_applied),
            "suppressed_patterns": list(self._suppressed_patterns),
            "health_trend": [
                h["health_score"] for h in self._generation_health[-10:]
            ],
            "recent_alerts": self._alerts_raised[-5:],
        }

    def suppress_pattern(self, pattern_name: str) -> None:
        """CEO can suppress a pattern (dismiss the alert)."""
        self._suppressed_patterns.add(pattern_name)
        logger.info(f"Pattern '{pattern_name}' suppressed by CEO")

    def unsuppress_pattern(self, pattern_name: str) -> None:
        """Re-enable detection for a suppressed pattern."""
        self._suppressed_patterns.discard(pattern_name)
        logger.info(f"Pattern '{pattern_name}' re-enabled")
