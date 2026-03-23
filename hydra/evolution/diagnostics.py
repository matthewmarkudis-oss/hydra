"""Diagnostic engine ported from CHIMERA — analyzes generation results and recommends mutations.

Examines agent performance, detects stagnation/overfitting/drawdown issues,
and prescribes targeted mutations to fix them.

Circuit breakers (production-readiness): when running in forward-test or paper
mode, diagnostic severity maps to actionable interventions — alerts, allocation
cuts, position flattening, or full agent shutdown.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from hydra.evolution.mutation_engine import MutationRecord

logger = logging.getLogger("hydra.evolution.diagnostics")


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _max_severity(*severities: str) -> str:
    order = {"minor": 0, "moderate": 1, "severe": 2, "critical": 3}
    if not severities:
        return "minor"
    return max(severities, key=lambda s: order.get(s, 0))


# ── Circuit Breaker ──────────────────────────────────────────────────────────

@dataclass
class CircuitBreakerAction:
    """An actionable intervention triggered by diagnostic severity.

    Used in forward-test / paper trading to protect capital when CHIMERA
    detects critical issues.

    Actions:
        alert            — Log warning, no weight change.
        reduce_allocation — Cut agent weight by `reduction_pct`.
        flatten_positions — Zero out agent's positions (requires broker hook).
        shutdown_agent    — Remove agent from active pool entirely.
    """

    action: str                      # "alert" | "reduce_allocation" | "flatten_positions" | "shutdown_agent"
    severity: str                    # diagnostic severity that triggered it
    target_agent: str | None = None  # None = applies to whole portfolio
    reason: str = ""
    reduction_pct: float = 0.0       # for reduce_allocation: 0.0–1.0
    timestamp: str = ""


# Severity → default circuit breaker responses.
# In paper/forward-test mode, these fire automatically.
# In backtesting, they are logged but not acted upon.
_CIRCUIT_BREAKER_MAP: dict[str, list[dict]] = {
    "minor": [],
    "moderate": [
        {"action": "alert", "reduction_pct": 0.0},
    ],
    "severe": [
        {"action": "alert", "reduction_pct": 0.0},
        {"action": "reduce_allocation", "reduction_pct": 0.25},
    ],
    "critical": [
        {"action": "alert", "reduction_pct": 0.0},
        {"action": "reduce_allocation", "reduction_pct": 0.50},
        {"action": "flatten_positions", "reduction_pct": 0.0},
    ],
}


@dataclass
class GenerationMetrics:
    """Aggregated metrics for one generation — input to diagnostics."""

    generation: int = 0
    mean_reward: float = 0.0
    best_reward: float = 0.0
    mean_sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    wfe: float = 0.0
    total_trades: int = 0
    fitness_score: float = 0.0
    consistency: float = 0.0
    agent_scores: dict[str, float] = field(default_factory=dict)
    window_sharpes: list[float] = field(default_factory=list)
    mean_cash_ratio: float = 1.0  # Average fraction of portfolio in cash
    benchmark_return: float = 0.0  # Benchmark return for this generation
    mean_return: float = 0.0  # Mean portfolio return for this generation


class DiagnosticEngine:
    """Examines generation results, identifies issues, and recommends mutations.

    Ported from CHIMERA's ChimeraDiagnostics. Adapted for Hydra's RL context.
    """

    def __init__(
        self,
        history: list[GenerationMetrics] | None = None,
        blacklisted: list[dict] | None = None,
    ) -> None:
        self.history: list[GenerationMetrics] = history or []
        self.blacklisted: list[dict] = blacklisted or []

    def diagnose(self, result: GenerationMetrics) -> dict[str, Any]:
        """Run all diagnostic checks and return diagnosis with recommended mutations.

        Returns:
            Dict with keys:
                primary_issue          — one-line summary
                issues                 — list[str] of all detected issues
                recommended_mutations  — list[MutationRecord] ordered by priority
                severity               — "minor" | "moderate" | "severe" | "critical"
                plain_english          — layperson explanation
        """
        issues: list[str] = []
        mutations: list[MutationRecord] = []
        severities: list[str] = []
        explanations: list[str] = []

        for checker in (
            self._check_overfitting,
            self._check_drawdown,
            self._check_insufficient_trades,
            self._check_agent_performance,
            self._check_negative_sharpe,
            self._check_low_win_rate,
            self._check_high_cash_ratio,
            self._check_benchmark_underperformance,
        ):
            info = checker(result)
            if info:
                issues.extend(info.get("issues", []))
                mutations.extend(info.get("mutations", []))
                if info.get("severity"):
                    severities.append(info["severity"])
                if info.get("plain_english"):
                    explanations.append(info["plain_english"])

        # Stagnation uses history
        stag = self._check_stagnation()
        if stag:
            issues.extend(stag.get("issues", []))
            mutations.extend(stag.get("mutations", []))
            if stag.get("severity"):
                severities.append(stag["severity"])
            if stag.get("plain_english"):
                explanations.append(stag["plain_english"])

        # Deduplicate mutations by type while preserving order
        seen_types: set[str] = set()
        unique_mutations: list[MutationRecord] = []
        for m in mutations:
            if m.mutation_type not in seen_types:
                seen_types.add(m.mutation_type)
                unique_mutations.append(m)

        severity = _max_severity(*severities) if severities else "minor"
        primary = issues[0] if issues else "No significant issues detected"
        plain = "\n".join(f"- {e}" for e in explanations) if explanations else (
            "System is performing within acceptable parameters."
        )

        return {
            "primary_issue": primary,
            "issues": issues,
            "recommended_mutations": unique_mutations,
            "severity": severity,
            "plain_english": plain,
        }

    def get_circuit_breaker_actions(
        self, diagnosis: dict[str, Any]
    ) -> list[CircuitBreakerAction]:
        """Map a diagnosis to circuit breaker actions for production use.

        In forward-test / paper trading, the caller should act on these.
        In backtesting, the caller should log but not act.

        Args:
            diagnosis: Output of diagnose().

        Returns:
            List of CircuitBreakerAction to execute.
        """
        severity = diagnosis.get("severity", "minor")
        templates = _CIRCUIT_BREAKER_MAP.get(severity, [])
        if not templates:
            return []

        actions: list[CircuitBreakerAction] = []
        ts = _ts()

        # Identify weak agents from mutations for targeted actions
        weak_agents: list[str] = []
        for mut in diagnosis.get("recommended_mutations", []):
            if mut.mutation_type == "bench_agent":
                agent = mut.params.get("agent")
                if agent:
                    weak_agents.append(agent)

        for tmpl in templates:
            if tmpl["action"] in ("reduce_allocation", "flatten_positions", "shutdown_agent"):
                # Targeted: apply to each weak agent individually
                if weak_agents:
                    for agent in weak_agents:
                        actions.append(CircuitBreakerAction(
                            action=tmpl["action"],
                            severity=severity,
                            target_agent=agent,
                            reason=diagnosis.get("primary_issue", ""),
                            reduction_pct=tmpl.get("reduction_pct", 0.0),
                            timestamp=ts,
                        ))
                else:
                    # Portfolio-wide action (no specific weak agent identified)
                    actions.append(CircuitBreakerAction(
                        action=tmpl["action"],
                        severity=severity,
                        target_agent=None,
                        reason=diagnosis.get("primary_issue", ""),
                        reduction_pct=tmpl.get("reduction_pct", 0.0),
                        timestamp=ts,
                    ))
            else:
                # Alert — always portfolio-wide
                actions.append(CircuitBreakerAction(
                    action=tmpl["action"],
                    severity=severity,
                    target_agent=None,
                    reason=diagnosis.get("primary_issue", ""),
                    reduction_pct=0.0,
                    timestamp=ts,
                ))

        return actions

    def add_generation(self, metrics: GenerationMetrics) -> None:
        """Record a generation's metrics for stagnation tracking."""
        self.history.append(metrics)

    # -------------------------------------------------------------------
    # Diagnostic checks
    # -------------------------------------------------------------------

    def _check_overfitting(self, result: GenerationMetrics) -> dict | None:
        """Detect overfitting via WFE."""
        wfe = result.wfe
        if wfe >= 0.40 or wfe == 0.0:
            return None

        if wfe < 0.25:
            return {
                "issues": [f"Severe overfitting: WFE {wfe:.2f} — IS edge evaporates OOS"],
                "mutations": [
                    MutationRecord(
                        mutation_type="equalize_weights", category="weight",
                        description="Equalize weights to reduce overfitted bias",
                        params={}, timestamp=_ts(),
                    ),
                    MutationRecord(
                        mutation_type="reduce_complexity", category="parameter",
                        description="Reduce parameter complexity toward defaults",
                        params={}, timestamp=_ts(),
                    ),
                ],
                "severity": "critical",
                "plain_english": "The system memorized training patterns instead of learning real ones",
            }
        else:
            return {
                "issues": [f"Moderate overfitting: WFE {wfe:.2f} below target (0.40)"],
                "mutations": [
                    MutationRecord(
                        mutation_type="equalize_weights", category="weight",
                        description="Equalize weights to reduce overfitting risk",
                        params={}, timestamp=_ts(),
                    ),
                ],
                "severity": "moderate",
                "plain_english": "Some signs the system is fitting to historical quirks",
            }

    def _check_drawdown(self, result: GenerationMetrics) -> dict | None:
        """Check maximum drawdown severity."""
        dd = abs(result.max_drawdown)
        if dd <= 0.15:
            return None

        if dd > 0.25:
            return {
                "issues": [f"Excessive drawdown: {dd:.1%} exceeds 25% limit"],
                "mutations": [
                    MutationRecord(
                        mutation_type="tighten_risk", category="parameter",
                        description=f"Tighten risk controls (drawdown={dd:.1%})",
                        params={}, timestamp=_ts(),
                    ),
                    MutationRecord(
                        mutation_type="increase_drawdown_penalty", category="parameter",
                        description="Increase drawdown penalty in reward function",
                        params={}, timestamp=_ts(),
                    ),
                ],
                "severity": "severe",
                "plain_english": "The system lost too much during its worst period",
            }
        return {
            "issues": [f"Elevated drawdown: {dd:.1%} above comfort zone (15%)"],
            "mutations": [
                MutationRecord(
                    mutation_type="tighten_risk", category="parameter",
                    description=f"Slightly tighten risk (drawdown={dd:.1%})",
                    params={}, timestamp=_ts(),
                ),
            ],
            "severity": "moderate",
            "plain_english": "Drawdowns are higher than ideal",
        }

    def _check_insufficient_trades(self, result: GenerationMetrics) -> dict | None:
        """Check if agents are generating enough trades."""
        if result.total_trades >= 10:
            return None

        if result.total_trades == 0:
            return {
                "issues": ["Zero trades — agents are not taking any positions"],
                "mutations": [
                    MutationRecord(
                        mutation_type="loosen_risk", category="parameter",
                        description="Loosen risk limits to allow more trading",
                        params={}, timestamp=_ts(),
                    ),
                    MutationRecord(
                        mutation_type="decrease_drawdown_penalty", category="parameter",
                        description="Reduce drawdown penalty so agents are less conservative",
                        params={}, timestamp=_ts(),
                    ),
                ],
                "severity": "critical",
                "plain_english": "Agents have learned to do nothing — need to encourage action",
            }
        return {
            "issues": [f"Only {result.total_trades} trades — too few for statistical significance"],
            "mutations": [
                MutationRecord(
                    mutation_type="decrease_drawdown_penalty", category="parameter",
                    description="Reduce penalties to encourage more trades",
                    params={}, timestamp=_ts(),
                ),
            ],
            "severity": "moderate",
            "plain_english": "Not enough trades to judge whether the strategy works",
        }

    def _check_agent_performance(self, result: GenerationMetrics) -> dict | None:
        """Identify agents dragging down or boosting the ensemble."""
        scores = result.agent_scores
        if not scores or len(scores) < 2:
            return None

        by_score = sorted(scores.items(), key=lambda x: x[1])
        worst_name, worst_score = by_score[0]
        best_name, best_score = by_score[-1]

        issues = []
        mutations = []

        # Bench worst if significantly negative
        if worst_score < -50:
            issues.append(f"Weak agent: {worst_name} score={worst_score:.1f}")
            mutations.append(MutationRecord(
                mutation_type="bench_agent", category="inclusion",
                description=f"Bench '{worst_name}' (score={worst_score:.1f})",
                params={"agent": worst_name}, timestamp=_ts(),
            ))

        # Boost best if strong
        if best_score > 50:
            issues.append(f"Strong agent: {best_name} score={best_score:.1f} — increase weight")
            mutations.append(MutationRecord(
                mutation_type="reweight_up", category="weight",
                description=f"Increase weight for '{best_name}'",
                params={"agent": best_name, "amount": 0.05}, timestamp=_ts(),
            ))

        if not issues:
            return None

        return {
            "issues": issues,
            "mutations": mutations,
            "severity": "moderate",
            "plain_english": f"Agent gap: best={best_name} ({best_score:.1f}), worst={worst_name} ({worst_score:.1f})",
        }

    def _check_negative_sharpe(self, result: GenerationMetrics) -> dict | None:
        """Check for consistently negative Sharpe ratio."""
        if result.mean_sharpe >= 0:
            return None

        if result.mean_sharpe < -1.0:
            return {
                "issues": [f"Deeply negative Sharpe: {result.mean_sharpe:.3f}"],
                "mutations": [
                    MutationRecord(
                        mutation_type="restore_defaults", category="parameter",
                        description="Restore parameters to defaults (negative Sharpe)",
                        params={}, timestamp=_ts(),
                    ),
                    MutationRecord(
                        mutation_type="equalize_weights", category="weight",
                        description="Equalize weights to reset biased allocation",
                        params={}, timestamp=_ts(),
                    ),
                ],
                "severity": "severe",
                "plain_english": "The system is consistently losing money — resetting to defaults",
            }
        return {
            "issues": [f"Negative Sharpe: {result.mean_sharpe:.3f}"],
            "mutations": [],
            "severity": "moderate",
            "plain_english": "Returns are slightly negative — agents still learning",
        }

    def _check_low_win_rate(self, result: GenerationMetrics) -> dict | None:
        """Check for very low win rate."""
        if result.win_rate >= 0.35 or result.total_trades == 0:
            return None

        return {
            "issues": [f"Low win rate: {result.win_rate:.1%}"],
            "mutations": [
                MutationRecord(
                    mutation_type="prioritize_consistency", category="objective",
                    description="Shift fitness toward consistency to improve win rate",
                    params={}, timestamp=_ts(),
                ),
            ],
            "severity": "moderate",
            "plain_english": f"Only winning {result.win_rate:.0%} of trades",
        }

    def _check_high_cash_ratio(self, result: GenerationMetrics) -> dict | None:
        """Detect agents hoarding cash instead of deploying capital."""
        if result.mean_cash_ratio <= 0.6:
            return None

        if result.mean_cash_ratio > 0.8:
            return {
                "issues": [f"Excessive cash hoarding: {result.mean_cash_ratio:.0%} portfolio in cash"],
                "mutations": [
                    MutationRecord(
                        mutation_type="increase_deployment", category="parameter",
                        description=f"Increase deployment pressure (cash ratio={result.mean_cash_ratio:.0%})",
                        params={}, timestamp=_ts(),
                    ),
                    MutationRecord(
                        mutation_type="decrease_drawdown_penalty", category="parameter",
                        description="Reduce drawdown penalty to encourage risk-taking",
                        params={}, timestamp=_ts(),
                    ),
                ],
                "severity": "severe",
                "plain_english": "Agents are sitting in cash instead of trading — need stronger deployment incentive",
            }
        return {
            "issues": [f"High cash ratio: {result.mean_cash_ratio:.0%} portfolio in cash"],
            "mutations": [
                MutationRecord(
                    mutation_type="increase_deployment", category="parameter",
                    description=f"Nudge deployment higher (cash ratio={result.mean_cash_ratio:.0%})",
                    params={}, timestamp=_ts(),
                ),
            ],
            "severity": "moderate",
            "plain_english": f"Agents holding {result.mean_cash_ratio:.0%} in cash — should be deploying more",
        }

    def _check_benchmark_underperformance(self, result: GenerationMetrics) -> dict | None:
        """Detect consistent underperformance vs benchmark across generations."""
        # Need at least 3 generations of history to detect a pattern
        if len(self.history) < 3:
            return None

        recent = self.history[-3:]
        underperforming_count = sum(
            1 for r in recent
            if r.mean_return < r.benchmark_return and r.benchmark_return != 0.0
        )

        if underperforming_count < 3:
            return None

        avg_shortfall = sum(
            r.benchmark_return - r.mean_return for r in recent
        ) / len(recent)

        return {
            "issues": [f"Benchmark underperformance: trailing benchmark by {avg_shortfall:.4f}/bar for 3+ generations"],
            "mutations": [
                MutationRecord(
                    mutation_type="reward_outperformance", category="parameter",
                    description="Increase benchmark bonus to reward outperformance",
                    params={}, timestamp=_ts(),
                ),
            ],
            "severity": "moderate",
            "plain_english": "Portfolio has consistently trailed the benchmark — increasing outperformance incentive",
        }

    def _check_stagnation(self) -> dict | None:
        """Detect when evolution has plateaued across recent generations."""
        if len(self.history) < 3:
            return None

        recent = self.history[-3:]
        rewards = [r.mean_reward for r in recent]
        spread = max(rewards) - min(rewards)

        if spread >= 5.0:
            return None

        avg = sum(rewards) / len(rewards)
        return {
            "issues": [f"Stagnation: last 3 generations stuck at reward ~{avg:.2f} (spread {spread:.2f})"],
            "mutations": [
                MutationRecord(
                    mutation_type="equalize_weights", category="weight",
                    description="Equalize weights to break stagnation",
                    params={}, timestamp=_ts(),
                ),
                MutationRecord(
                    mutation_type="adjust_sharpe_window", category="parameter",
                    description="Shorten Sharpe window for faster adaptation",
                    params={"direction": "shorter"}, timestamp=_ts(),
                ),
            ],
            "severity": "moderate",
            "plain_english": "Evolution has plateaued — applying mutations to break out of local optimum",
        }
