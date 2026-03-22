"""Performance Analyst — post-pipeline attribution and correlation analysis.

Explains WHY agents succeed or fail through:
- Agent-to-agent correlation (redundancy detection)
- Score decomposition (trend, volatility, best/worst)
- Regime attribution (which agents work in which conditions)
- Pool efficiency (Gini coefficient, positive fraction)

Zero LLM cost — pure Python + numpy.

Backtesting and training only.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from corp.agents.base_corp_agent import BaseCorpAgent
from corp.state.corporation_state import CorporationState
from corp.state.decision_log import DecisionLog

logger = logging.getLogger("corp.agents.performance_analyst")


class PerformanceAnalyst(BaseCorpAgent):
    """Decomposes agent performance into interpretable factors.

    Runs post-analysis in the corp graph. Requires generation_results
    (training state) with eval_scores per generation.
    """

    def __init__(
        self,
        state: CorporationState,
        decision_log: DecisionLog,
    ):
        super().__init__("performance_analyst", state, decision_log)

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Run full performance analysis.

        Context keys:
            pipeline_results: Summary from pipeline phase.
            generation_results: List of per-generation dicts from training state.
                Each should have 'eval_scores': {agent_name: float}.
        """
        generations = context.get("generation_results", [])

        # Try to load from training state file if not in context
        if not generations:
            generations = self._load_training_state()

        if len(generations) < 3:
            result = {
                "skipped": True,
                "reason": f"Need >= 3 generations, have {len(generations)}",
            }
            self._mark_run(result)
            return result

        correlations = self._compute_correlations(generations)
        profiles = self._compute_agent_profiles(generations)
        regime_attr = self._compute_regime_attribution(generations)
        efficiency = self._compute_pool_efficiency(generations)
        recommendations = self._generate_recommendations(
            correlations, profiles, efficiency
        )

        result = {
            "correlations": correlations,
            "agent_profiles": profiles,
            "regime_attribution": regime_attr,
            "pool_efficiency": efficiency,
            "recommendations": recommendations,
            "generations_analyzed": len(generations),
        }

        self.log_decision(
            "performance_analysis",
            detail={
                "generations_analyzed": len(generations),
                "redundant_pairs": sum(1 for c in correlations if c.get("redundant")),
                "pool_gini": efficiency.get("gini", 0),
                "num_recommendations": len(recommendations),
            },
            outcome="complete",
        )

        self._mark_run(result)
        return result

    def _compute_correlations(
        self, generations: list[dict]
    ) -> list[dict]:
        """Compute pairwise Pearson correlation of eval scores across generations.

        Flag pairs with r > 0.85 as redundant (making the same bets).
        """
        # Build score matrix: agents × generations
        all_agents: set[str] = set()
        for gen in generations:
            all_agents.update(gen.get("eval_scores", {}).keys())

        agents = sorted(all_agents)
        if len(agents) < 2:
            return []

        n_gens = len(generations)
        score_matrix = np.full((len(agents), n_gens), np.nan)

        for g_idx, gen in enumerate(generations):
            scores = gen.get("eval_scores", {})
            for a_idx, agent in enumerate(agents):
                if agent in scores:
                    score_matrix[a_idx, g_idx] = scores[agent]

        results = []
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                # Only compute correlation where both agents have scores
                mask = ~np.isnan(score_matrix[i]) & ~np.isnan(score_matrix[j])
                if mask.sum() < 3:
                    continue

                r = np.corrcoef(score_matrix[i, mask], score_matrix[j, mask])[0, 1]
                if np.isnan(r):
                    continue

                results.append({
                    "agent_a": agents[i],
                    "agent_b": agents[j],
                    "r": round(float(r), 3),
                    "redundant": abs(r) > 0.85,
                    "shared_gens": int(mask.sum()),
                })

        # Sort by absolute correlation descending
        results.sort(key=lambda x: abs(x["r"]), reverse=True)
        return results

    def _compute_agent_profiles(
        self, generations: list[dict]
    ) -> dict[str, dict]:
        """Compute per-agent performance profiles."""
        agent_scores: dict[str, list[float]] = {}
        agent_gens: dict[str, list[int]] = {}

        for gen in generations:
            gen_num = gen.get("generation", 0)
            scores = gen.get("eval_scores", {})
            for agent, score in scores.items():
                agent_scores.setdefault(agent, []).append(score)
                agent_gens.setdefault(agent, []).append(gen_num)

        profiles = {}
        for agent, scores in agent_scores.items():
            arr = np.array(scores)
            gens = np.array(agent_gens[agent])

            # Linear regression slope (score trend over time)
            trend = 0.0
            if len(arr) >= 3:
                x = gens - gens.mean()
                y = arr - arr.mean()
                denom = (x * x).sum()
                if denom > 0:
                    trend = float((x * y).sum() / denom)

            profiles[agent] = {
                "mean": round(float(arr.mean()), 2),
                "std": round(float(arr.std()), 2),
                "trend": round(trend, 4),
                "best_score": round(float(arr.max()), 2),
                "worst_score": round(float(arr.min()), 2),
                "best_gen": int(gens[arr.argmax()]),
                "worst_gen": int(gens[arr.argmin()]),
                "appearances": len(scores),
            }

        return profiles

    def _compute_regime_attribution(
        self, generations: list[dict]
    ) -> dict[str, dict[str, float]]:
        """Group eval scores by regime and compute per-agent mean per regime."""
        # regime → agent → list[score]
        regime_scores: dict[str, dict[str, list[float]]] = {}

        for gen in generations:
            # Try to find regime from diagnosis or competition metadata
            diagnosis = gen.get("diagnosis", {})
            regime = None

            # Check if regime was recorded in generation data
            if "regime" in gen:
                regime = gen["regime"]
            elif "competition" in gen:
                # Some generations may have regime in competition metadata
                pass

            if not regime:
                regime = "unknown"

            scores = gen.get("eval_scores", {})
            for agent, score in scores.items():
                regime_scores.setdefault(regime, {}).setdefault(agent, []).append(score)

        # Compute means
        result: dict[str, dict[str, float]] = {}
        for regime, agent_scores in regime_scores.items():
            result[regime] = {
                agent: round(float(np.mean(scores)), 2)
                for agent, scores in agent_scores.items()
            }

        return result

    def _compute_pool_efficiency(
        self, generations: list[dict]
    ) -> dict[str, Any]:
        """Compute pool efficiency metrics from latest generation."""
        latest = generations[-1]
        scores = latest.get("eval_scores", {})

        if not scores:
            return {"gini": 0.0, "positive_fraction": 0.0, "top_contributor": "N/A"}

        values = np.array(list(scores.values()))
        names = list(scores.keys())

        # Positive fraction
        positive = sum(1 for v in values if v > 0)
        positive_fraction = positive / len(values) if values.size > 0 else 0.0

        # Gini coefficient of absolute scores
        abs_values = np.abs(values)
        if abs_values.sum() > 0:
            sorted_vals = np.sort(abs_values)
            n = len(sorted_vals)
            cumulative = np.cumsum(sorted_vals)
            gini = (2.0 * np.sum((np.arange(1, n + 1) * sorted_vals)) / (n * sorted_vals.sum())) - (n + 1) / n
            gini = max(0.0, min(1.0, float(gini)))
        else:
            gini = 0.0

        # Top contributor
        top_idx = int(np.argmax(values))
        top_contributor = names[top_idx]

        return {
            "gini": round(gini, 3),
            "positive_fraction": round(positive_fraction, 3),
            "top_contributor": top_contributor,
            "top_score": round(float(values[top_idx]), 2),
            "total_agents": len(values),
            "positive_agents": positive,
        }

    def _generate_recommendations(
        self,
        correlations: list[dict],
        profiles: dict[str, dict],
        efficiency: dict,
    ) -> list[str]:
        """Generate actionable recommendations from the analysis."""
        recs = []

        # Redundant agent pairs
        redundant = [c for c in correlations if c.get("redundant")]
        for pair in redundant[:3]:
            a, b = pair["agent_a"], pair["agent_b"]
            r = pair["r"]
            # Recommend removing the weaker one
            score_a = profiles.get(a, {}).get("mean", 0)
            score_b = profiles.get(b, {}).get("mean", 0)
            weaker = b if score_a >= score_b else a
            recs.append(
                f"Redundant pair: {a} and {b} (r={r:.2f}). "
                f"Consider removing '{weaker}' (lower mean score)."
            )

        # Negative trend agents
        for agent, profile in profiles.items():
            if profile.get("trend", 0) < -1.0 and profile.get("appearances", 0) >= 5:
                recs.append(
                    f"'{agent}' has declining performance (trend={profile['trend']:.2f}/gen). "
                    f"Consider resetting weights or replacing architecture."
                )

        # Low pool efficiency
        gini = efficiency.get("gini", 0)
        if gini > 0.6:
            top = efficiency.get("top_contributor", "?")
            recs.append(
                f"High concentration: Gini={gini:.2f}. "
                f"'{top}' dominates — pool diversity may be collapsing."
            )

        # No positive agents
        pos_frac = efficiency.get("positive_fraction", 0)
        if pos_frac < 0.2:
            recs.append(
                f"Only {pos_frac:.0%} of agents score positive. "
                f"Consider reward function tuning or simpler ticker universe."
            )

        return recs

    def _load_training_state(self) -> list[dict]:
        """Try to load generation results from the training state file."""
        try:
            import json
            from pathlib import Path
            state_file = Path("logs/hydra_training_state.json")
            if state_file.exists():
                with open(state_file) as f:
                    data = json.load(f)
                return data.get("generations", [])
        except Exception:
            pass
        return []
