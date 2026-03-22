"""Generation Scorecard — structured evaluation of training generations.

Scores each generation across multiple dimensions and produces a verdict
(CONTINUE / RETUNE / HALT) for the CEO dashboard and Hedge Fund Director.
Pure Python, zero LLM cost.
"""

from __future__ import annotations

import statistics
from typing import Any


def score_generation(
    gen: dict,
    prev_gens: list[dict] | None = None,
) -> dict[str, Any]:
    """Score a training generation across structured dimensions.

    Args:
        gen: A single generation dict from hydra_training_state.json.
        prev_gens: Previous generations (most recent last) for trend analysis.

    Returns:
        {
            "generation": int,
            "dimension_scores": {dim: 1-5},
            "overall": 1-10,
            "verdict": "CONTINUE" | "RETUNE" | "HALT",
            "critical_gaps": [{"title": str, "priority": "HIGH"|"MEDIUM"|"LOW"}],
            "summary": str,
        }
    """
    prev_gens = prev_gens or []
    eval_scores = gen.get("eval_scores", {})
    train_mean = gen.get("train_mean_reward", 0)
    pool_size = gen.get("pool_size", 0)
    conviction = gen.get("conviction", {})
    competition = gen.get("competition", {})
    validation = gen.get("validation", {})

    scores = {}
    gaps = []

    # ── 1. Reward Trend (1-5) ────────────────────────────────────────────
    if prev_gens:
        recent_means = [g.get("train_mean_reward", 0) for g in prev_gens[-5:]]
        recent_means.append(train_mean)
        if len(recent_means) >= 2:
            deltas = [
                recent_means[i] - recent_means[i - 1]
                for i in range(1, len(recent_means))
            ]
            avg_delta = statistics.mean(deltas)
            if avg_delta > 50:
                scores["reward_trend"] = 5
            elif avg_delta > 10:
                scores["reward_trend"] = 4
            elif avg_delta > -10:
                scores["reward_trend"] = 3
            elif avg_delta > -50:
                scores["reward_trend"] = 2
            else:
                scores["reward_trend"] = 1
                gaps.append({"title": "Mean reward declining significantly", "priority": "HIGH"})
        else:
            scores["reward_trend"] = 3
    else:
        scores["reward_trend"] = 3  # No trend data

    # ── 2. Top Agent Quality (1-5) ───────────────────────────────────────
    if eval_scores:
        best_score = max(eval_scores.values())
        if best_score > 200:
            scores["top_agent_quality"] = 5
        elif best_score > 100:
            scores["top_agent_quality"] = 4
        elif best_score > 0:
            scores["top_agent_quality"] = 3
        elif best_score > -200:
            scores["top_agent_quality"] = 2
        else:
            scores["top_agent_quality"] = 1
            gaps.append({"title": "Best agent deeply negative", "priority": "HIGH"})
    else:
        scores["top_agent_quality"] = 1
        gaps.append({"title": "No eval scores available", "priority": "HIGH"})

    # ── 3. Pool Diversity (1-5) ──────────────────────────────────────────
    if eval_scores:
        type_counts: dict[str, int] = {}
        for name in eval_scores:
            if "rppo" in name:
                atype = "rppo"
            elif "ppo" in name:
                atype = "ppo"
            elif "sac" in name:
                atype = "sac"
            elif "rule" in name:
                atype = "rule"
            else:
                atype = "other"
            type_counts[atype] = type_counts.get(atype, 0) + 1

        n_types = len(type_counts)
        total = sum(type_counts.values())
        max_share = max(type_counts.values()) / total if total else 1.0

        if n_types >= 4 and max_share < 0.5:
            scores["pool_diversity"] = 5
        elif n_types >= 3 and max_share < 0.6:
            scores["pool_diversity"] = 4
        elif n_types >= 2 and max_share < 0.75:
            scores["pool_diversity"] = 3
        elif n_types >= 2:
            scores["pool_diversity"] = 2
            gaps.append({
                "title": f"Pool dominated by one architecture ({max(type_counts, key=type_counts.get)}: {max_share:.0%})",
                "priority": "MEDIUM",
            })
        else:
            scores["pool_diversity"] = 1
            gaps.append({"title": "Only one agent architecture in pool", "priority": "HIGH"})
    else:
        scores["pool_diversity"] = 1

    # ── 4. Positive Rate (1-5) ───────────────────────────────────────────
    if eval_scores:
        positive = sum(1 for v in eval_scores.values() if v > 0)
        total = len(eval_scores)
        rate = positive / total if total else 0

        if rate > 0.75:
            scores["positive_rate"] = 5
        elif rate > 0.50:
            scores["positive_rate"] = 4
        elif rate > 0.25:
            scores["positive_rate"] = 3
        elif rate > 0.10:
            scores["positive_rate"] = 2
        else:
            scores["positive_rate"] = 1
            gaps.append({"title": f"Only {positive}/{total} agents positive ({rate:.0%})", "priority": "HIGH"})
    else:
        scores["positive_rate"] = 1

    # ── 5. Conviction Strength (1-5) ─────────────────────────────────────
    if conviction:
        win_rates = []
        for agent_name, conv_data in conviction.items():
            if isinstance(conv_data, dict) and conv_data.get("total_trades", 0) >= 3:
                win_rates.append(conv_data.get("overall_win_rate", 0))

        if win_rates:
            top_wr = sorted(win_rates, reverse=True)[:3]
            avg_top = statistics.mean(top_wr)
            if avg_top > 0.70:
                scores["conviction_strength"] = 5
            elif avg_top > 0.55:
                scores["conviction_strength"] = 4
            elif avg_top > 0.45:
                scores["conviction_strength"] = 3
            elif avg_top > 0.30:
                scores["conviction_strength"] = 2
            else:
                scores["conviction_strength"] = 1
                gaps.append({"title": "Top agents have weak win rates", "priority": "MEDIUM"})
        else:
            scores["conviction_strength"] = 2
            gaps.append({"title": "Insufficient trade data for conviction scoring", "priority": "LOW"})
    else:
        scores["conviction_strength"] = 2

    # ── 6. Stability (1-5) ───────────────────────────────────────────────
    if prev_gens and len(prev_gens) >= 3:
        recent_bests = []
        for g in prev_gens[-5:]:
            es = g.get("eval_scores", {})
            if es:
                recent_bests.append(max(es.values()))
        if eval_scores:
            recent_bests.append(max(eval_scores.values()))

        if len(recent_bests) >= 3:
            std = statistics.stdev(recent_bests)
            mean_best = statistics.mean(recent_bests)
            cv = std / abs(mean_best) if mean_best != 0 else 999

            if cv < 0.10:
                scores["stability"] = 5
            elif cv < 0.25:
                scores["stability"] = 4
            elif cv < 0.50:
                scores["stability"] = 3
            elif cv < 1.0:
                scores["stability"] = 2
            else:
                scores["stability"] = 1
                gaps.append({"title": "Top agent scores highly volatile across generations", "priority": "MEDIUM"})
        else:
            scores["stability"] = 3
    else:
        scores["stability"] = 3

    # ── 7. Validation (bonus/penalty) ────────────────────────────────────
    has_validation = bool(validation)
    if not has_validation:
        gaps.append({"title": "No ATHENA validation run", "priority": "HIGH"})

    # ── Overall score (1-10) ─────────────────────────────────────────────
    dim_values = list(scores.values())
    avg_dim = statistics.mean(dim_values) if dim_values else 1.0
    overall = round(avg_dim * 2)  # Scale 1-5 -> 2-10
    overall = max(1, min(10, overall))

    # Penalty for missing validation
    if not has_validation and overall > 6:
        overall = 6

    # ── Verdict ──────────────────────────────────────────────────────────
    high_gaps = sum(1 for g in gaps if g["priority"] == "HIGH")
    if overall >= 7 and high_gaps == 0:
        verdict = "CONTINUE"
    elif overall <= 3 or high_gaps >= 3:
        verdict = "HALT"
    else:
        verdict = "RETUNE"

    # ── Summary ──────────────────────────────────────────────────────────
    best_agent = max(eval_scores, key=eval_scores.get) if eval_scores else "N/A"
    best_val = max(eval_scores.values()) if eval_scores else 0

    summary = (
        f"Gen {gen.get('generation', '?')}: overall {overall}/10 ({verdict}). "
        f"Best: {best_agent} ({best_val:+.1f}). "
        f"Pool: {pool_size} agents, "
        f"{sum(1 for v in eval_scores.values() if v > 0)}/{len(eval_scores)} positive. "
        f"{len(gaps)} gap(s) identified."
    )

    return {
        "generation": gen.get("generation"),
        "dimension_scores": scores,
        "overall": overall,
        "verdict": verdict,
        "critical_gaps": gaps,
        "summary": summary,
        "best_agent": best_agent,
        "best_score": best_val,
    }


def format_scorecard(sc: dict) -> str:
    """Format a scorecard dict as a human-readable string."""
    lines = [
        f"Generation {sc['generation']} Scorecard",
        f"  Overall: {sc['overall']}/10  Verdict: {sc['verdict']}",
        "",
        "  Dimensions:",
    ]
    dim_labels = {
        "reward_trend": "Reward Trend",
        "top_agent_quality": "Top Agent Quality",
        "pool_diversity": "Pool Diversity",
        "positive_rate": "Positive Rate",
        "conviction_strength": "Conviction Strength",
        "stability": "Stability",
    }
    pips = {1: "X____", 2: "XX___", 3: "XXX__", 4: "XXXX_", 5: "XXXXX"}
    for dim, val in sc.get("dimension_scores", {}).items():
        label = dim_labels.get(dim, dim)
        lines.append(f"    {label:<22} [{pips.get(val, '?')}] {val}/5")

    lines.append(f"\n  Best Agent: {sc.get('best_agent', 'N/A')} ({sc.get('best_score', 0):+.1f})")

    gaps = sc.get("critical_gaps", [])
    if gaps:
        lines.append(f"\n  Gaps ({len(gaps)}):")
        for g in gaps:
            lines.append(f"    [{g['priority']}] {g['title']}")

    lines.append(f"\n  {sc.get('summary', '')}")
    return "\n".join(lines)
