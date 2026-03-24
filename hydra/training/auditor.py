"""Training Auditor — per-generation health checks for the training process.

Monitors the training pipeline itself (not just agent performance) to catch
systemic issues that individual subsystems miss. Runs at the end of each
generation and returns a verdict: CONTINUE, WARN, or HALT.

Checks:
1. Reward stagnation — mean reward not improving over N gens
2. Weight collapse — PROMETHEUS weights converging to uniform/floor
3. Regime feedback loops — risk_off/crisis triggered for consecutive gens
4. Auto-tuner penalty ratcheting — penalties monotonically increasing
5. Episode truncation rate — too many episodes cut short by constraints
6. Return stagnation — best_return_pct stuck below threshold
7. Pool diversity — all agents converging to same behavior
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("hydra.training.auditor")


@dataclass
class AuditAlert:
    """A single finding from the training auditor."""

    check_name: str
    severity: str  # "info", "warn", "critical"
    message: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditResult:
    """Result of a per-generation audit."""

    generation: int
    verdict: str  # "CONTINUE", "WARN", "HALT"
    alerts: list[AuditAlert] = field(default_factory=list)
    summary: str = ""

    @property
    def has_critical(self) -> bool:
        return any(a.severity == "critical" for a in self.alerts)

    @property
    def has_warnings(self) -> bool:
        return any(a.severity == "warn" for a in self.alerts)


class TrainingAuditor:
    """Per-generation health monitor for the training pipeline.

    Accumulates history across generations and runs checks that detect
    systemic issues invisible to individual subsystems (CHIMERA, PROMETHEUS,
    ELEOS). The auditor watches the watchers.

    Args:
        stagnation_window: Number of gens to check for reward stagnation.
        stagnation_threshold: Min improvement in mean_reward over the window.
        weight_collapse_ratio: Min ratio of max/min weight before alerting.
        regime_alert_streak: Consecutive risk_off/crisis gens before alerting.
        penalty_ratchet_window: Gens to check for monotonic penalty increase.
        truncation_rate_warn: Episode truncation rate triggering a warning.
        truncation_rate_critical: Episode truncation rate triggering critical.
        return_floor_pct: Minimum best_return_pct to expect by mid-training.
        return_floor_gen: Generation by which return_floor should be met.
        halt_on_critical: Whether to recommend HALT on critical alerts.
    """

    def __init__(
        self,
        stagnation_window: int = 5,
        stagnation_threshold: float = 50.0,
        weight_collapse_ratio: float = 3.0,
        regime_alert_streak: int = 3,
        penalty_ratchet_window: int = 4,
        truncation_rate_warn: float = 0.20,
        truncation_rate_critical: float = 0.40,
        return_floor_pct: float = 0.5,
        return_floor_gen: int = 15,
        halt_on_critical: bool = False,
    ):
        self.stagnation_window = stagnation_window
        self.stagnation_threshold = stagnation_threshold
        self.weight_collapse_ratio = weight_collapse_ratio
        self.regime_alert_streak = regime_alert_streak
        self.penalty_ratchet_window = penalty_ratchet_window
        self.truncation_rate_warn = truncation_rate_warn
        self.truncation_rate_critical = truncation_rate_critical
        self.return_floor_pct = return_floor_pct
        self.return_floor_gen = return_floor_gen
        self.halt_on_critical = halt_on_critical

        # Accumulated history
        self._reward_history: list[float] = []
        self._return_history: list[float] = []
        self._weight_history: list[dict[str, float]] = []
        self._regime_history: list[str] = []
        self._reward_params_history: list[dict[str, float]] = []
        self._truncation_rates: list[float] = []
        self._audit_history: list[AuditResult] = []

    def audit_generation(
        self,
        generation: int,
        gen_result: dict[str, Any],
        reward_params: dict[str, float] | None = None,
        regime: str | None = None,
        truncation_rate: float = 0.0,
    ) -> AuditResult:
        """Run all checks for a completed generation.

        Args:
            generation: Current generation number.
            gen_result: The gen_result dict from population_trainer.
            reward_params: Current reward function parameters.
            regime: Current regime string.
            truncation_rate: Fraction of episodes truncated by constraints.

        Returns:
            AuditResult with verdict and alerts.
        """
        # Record history
        self._reward_history.append(gen_result.get("train_mean_reward", 0.0))
        self._return_history.append(gen_result.get("best_return_pct", 0.0))

        competition = gen_result.get("competition", {})
        weights = competition.get("weights_after", {})
        if weights:
            self._weight_history.append(dict(weights))

        if regime:
            self._regime_history.append(regime)

        if reward_params:
            self._reward_params_history.append(dict(reward_params))

        self._truncation_rates.append(truncation_rate)

        # Run checks
        alerts: list[AuditAlert] = []

        alerts.extend(self._check_reward_stagnation(generation))
        alerts.extend(self._check_weight_collapse(generation))
        alerts.extend(self._check_regime_feedback_loop(generation))
        alerts.extend(self._check_penalty_ratcheting(generation))
        alerts.extend(self._check_truncation_rate(generation))
        alerts.extend(self._check_return_floor(generation))
        alerts.extend(self._check_pool_diversity(gen_result))

        # Determine verdict
        if any(a.severity == "critical" for a in alerts):
            verdict = "HALT" if self.halt_on_critical else "WARN"
        elif any(a.severity == "warn" for a in alerts):
            verdict = "WARN"
        else:
            verdict = "CONTINUE"

        # Build summary
        if alerts:
            alert_strs = [f"[{a.severity.upper()}] {a.check_name}: {a.message}" for a in alerts]
            summary = f"Gen {generation}: {verdict} -- {len(alerts)} alert(s): " + "; ".join(alert_strs)
        else:
            summary = f"Gen {generation}: CONTINUE -- all checks passed"

        result = AuditResult(
            generation=generation,
            verdict=verdict,
            alerts=alerts,
            summary=summary,
        )
        self._audit_history.append(result)

        # Log
        if verdict == "HALT":
            logger.warning(f"  AUDITOR: {summary}")
        elif verdict == "WARN":
            logger.info(f"  AUDITOR: {summary}")
        else:
            logger.debug(f"  AUDITOR: {summary}")

        return result

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_reward_stagnation(self, generation: int) -> list[AuditAlert]:
        """Check if mean reward has stopped improving."""
        alerts = []
        w = self.stagnation_window
        if len(self._reward_history) < w:
            return alerts

        recent = self._reward_history[-w:]
        improvement = recent[-1] - recent[0]

        if improvement < -self.stagnation_threshold:
            alerts.append(AuditAlert(
                check_name="reward_regression",
                severity="critical",
                message=(
                    f"Mean reward REGRESSED over last {w} gens: "
                    f"{recent[0]:.1f} -> {recent[-1]:.1f} (delta={improvement:.1f})"
                ),
                data={"window": w, "start": recent[0], "end": recent[-1], "delta": improvement},
            ))
        elif abs(improvement) < self.stagnation_threshold * 0.1:
            alerts.append(AuditAlert(
                check_name="reward_stagnation",
                severity="warn",
                message=(
                    f"Mean reward stagnant over last {w} gens: "
                    f"{recent[0]:.1f} -> {recent[-1]:.1f} (delta={improvement:.1f})"
                ),
                data={"window": w, "start": recent[0], "end": recent[-1], "delta": improvement},
            ))

        return alerts

    def _check_weight_collapse(self, generation: int) -> list[AuditAlert]:
        """Check if PROMETHEUS weights have collapsed to near-uniform."""
        alerts = []
        if not self._weight_history:
            return alerts

        weights = self._weight_history[-1]
        if len(weights) < 2:
            return alerts

        values = list(weights.values())
        max_w = max(values)
        min_w = min(values)

        if min_w < 1e-6:
            min_w = 1e-6  # Avoid division by zero

        ratio = max_w / min_w

        if ratio < self.weight_collapse_ratio:
            alerts.append(AuditAlert(
                check_name="weight_collapse",
                severity="warn",
                message=(
                    f"Weight max/min ratio={ratio:.1f} (threshold={self.weight_collapse_ratio:.1f}). "
                    f"PROMETHEUS not differentiating agents. "
                    f"Max={max_w:.4f}, Min={min_w:.4f}"
                ),
                data={"ratio": ratio, "max": max_w, "min": min_w, "weights": weights},
            ))

        # Also check if weights have been collapsing over time
        if len(self._weight_history) >= 3:
            ratios = []
            for wh in self._weight_history[-3:]:
                vals = list(wh.values())
                r = max(vals) / max(min(vals), 1e-6)
                ratios.append(r)
            if all(r < self.weight_collapse_ratio for r in ratios):
                alerts.append(AuditAlert(
                    check_name="weight_collapse_persistent",
                    severity="critical",
                    message=(
                        f"Weight collapse persisted for 3+ gens. "
                        f"Ratios: {[round(r, 1) for r in ratios]}"
                    ),
                    data={"ratios": ratios},
                ))

        return alerts

    def _check_regime_feedback_loop(self, generation: int) -> list[AuditAlert]:
        """Check for consecutive risk_off/crisis regimes (feedback loop)."""
        alerts = []
        n = self.regime_alert_streak
        if len(self._regime_history) < n:
            return alerts

        recent = self._regime_history[-n:]
        suppressive = {"risk_off", "crisis"}

        if all(r in suppressive for r in recent):
            alerts.append(AuditAlert(
                check_name="regime_feedback_loop",
                severity="critical",
                message=(
                    f"Suppressive regime for {n} consecutive gens: {recent}. "
                    f"Possible feedback loop: losses -> risk_off -> dampened signal -> more losses"
                ),
                data={"streak": n, "regimes": recent},
            ))

        return alerts

    def _check_penalty_ratcheting(self, generation: int) -> list[AuditAlert]:
        """Check if auto-tuner is monotonically increasing penalties."""
        alerts = []
        w = self.penalty_ratchet_window
        if len(self._reward_params_history) < w:
            return alerts

        recent = self._reward_params_history[-w:]
        penalty_keys = ["drawdown_penalty", "transaction_penalty", "holding_penalty"]

        for key in penalty_keys:
            values = [p.get(key) for p in recent if key in p]
            if len(values) < w:
                continue

            # Check if monotonically increasing
            increasing = all(values[i] <= values[i + 1] for i in range(len(values) - 1))
            if increasing and values[-1] > values[0] * 1.01:  # At least 1% increase
                total_increase = (values[-1] - values[0]) / max(values[0], 1e-8)
                alerts.append(AuditAlert(
                    check_name="penalty_ratchet",
                    severity="warn",
                    message=(
                        f"{key} has increased monotonically over {w} gens: "
                        f"{values[0]:.4f} -> {values[-1]:.4f} (+{total_increase:.0%})"
                    ),
                    data={"param": key, "values": values, "increase_pct": total_increase},
                ))

        return alerts

    def _check_truncation_rate(self, generation: int) -> list[AuditAlert]:
        """Check if too many episodes are being truncated by constraints."""
        alerts = []
        if not self._truncation_rates:
            return alerts

        rate = self._truncation_rates[-1]

        if rate >= self.truncation_rate_critical:
            alerts.append(AuditAlert(
                check_name="truncation_rate",
                severity="critical",
                message=(
                    f"Episode truncation rate={rate:.0%} (critical threshold={self.truncation_rate_critical:.0%}). "
                    f"Constraints are cutting episodes short before agents can learn."
                ),
                data={"rate": rate},
            ))
        elif rate >= self.truncation_rate_warn:
            alerts.append(AuditAlert(
                check_name="truncation_rate",
                severity="warn",
                message=(
                    f"Episode truncation rate={rate:.0%} (warn threshold={self.truncation_rate_warn:.0%})."
                ),
                data={"rate": rate},
            ))

        return alerts

    def _check_return_floor(self, generation: int) -> list[AuditAlert]:
        """Check if returns are meeting minimum expectations by mid-training."""
        alerts = []
        if generation < self.return_floor_gen:
            return alerts

        best_ever = max(self._return_history) if self._return_history else 0.0

        if best_ever < self.return_floor_pct:
            alerts.append(AuditAlert(
                check_name="return_floor",
                severity="critical",
                message=(
                    f"Best return ever={best_ever:+.3f}% after {generation} gens "
                    f"(floor={self.return_floor_pct:+.1f}% by gen {self.return_floor_gen}). "
                    f"Training may not be converging toward profitability."
                ),
                data={"best_ever": best_ever, "floor": self.return_floor_pct, "gen": generation},
            ))

        return alerts

    def _check_pool_diversity(self, gen_result: dict) -> list[AuditAlert]:
        """Check if all agents are converging to same behavior (no diversity)."""
        alerts = []
        eval_scores = gen_result.get("eval_scores", {})
        if len(eval_scores) < 3:
            return alerts

        scores = list(eval_scores.values())
        spread = max(scores) - min(scores)
        mean_abs = sum(abs(s) for s in scores) / len(scores)

        if mean_abs > 0 and spread / mean_abs < 0.1:
            alerts.append(AuditAlert(
                check_name="pool_homogeneity",
                severity="warn",
                message=(
                    f"Agents converging: score spread={spread:.1f}, mean_abs={mean_abs:.1f}, "
                    f"ratio={spread / mean_abs:.2f}. Pool lacks diversity."
                ),
                data={"spread": spread, "mean_abs": mean_abs},
            ))

        return alerts

    # ------------------------------------------------------------------
    # History access
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[AuditResult]:
        """Full audit history."""
        return list(self._audit_history)

    @property
    def latest(self) -> AuditResult | None:
        """Most recent audit result."""
        return self._audit_history[-1] if self._audit_history else None

    def get_alert_counts(self) -> dict[str, int]:
        """Count alerts by severity across all generations."""
        counts: dict[str, int] = {"info": 0, "warn": 0, "critical": 0}
        for result in self._audit_history:
            for alert in result.alerts:
                counts[alert.severity] = counts.get(alert.severity, 0) + 1
        return counts

    def get_summary(self) -> dict[str, Any]:
        """Get overall training health summary."""
        counts = self.get_alert_counts()
        verdicts = [r.verdict for r in self._audit_history]
        return {
            "total_gens_audited": len(self._audit_history),
            "alert_counts": counts,
            "halt_recommendations": verdicts.count("HALT"),
            "warn_count": verdicts.count("WARN"),
            "clean_gens": verdicts.count("CONTINUE"),
            "latest_verdict": verdicts[-1] if verdicts else "N/A",
        }
