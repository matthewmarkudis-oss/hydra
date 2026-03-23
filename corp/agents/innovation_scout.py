"""Innovation Scout — discovers new tools, libraries, and techniques.

LLM-powered agent that searches for improvements to the Hydra system.
Produces structured innovation briefs for CEO review. Does NOT auto-install
anything — proposals only.

Backtesting and training only. Runs on a weekly schedule.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any

from corp.agents.base_corp_agent import BaseCorpAgent
from corp.state.corporation_state import CorporationState
from corp.state.decision_log import DecisionLog

logger = logging.getLogger("corp.agents.innovation_scout")

SYSTEM_PROMPT = """You are the Innovation Scout of HydraCorp, an AI-managed backtesting
research corporation that trains RL agents (PPO, SAC, RecurrentPPO) for
stock trading simulation using Stable-Baselines3, VectorBT, and custom
reward engineering.

Your job is to find NEW tools, libraries, techniques, or research papers
that could improve the system. You search the web and evaluate relevance.

OUTPUT FORMAT — respond with valid JSON only:
{
  "briefs": [
    {
      "tool_name": "Name of tool/technique",
      "category": "library | technique | research | data_source",
      "url": "https://...",
      "relevance_score": 0.0 to 1.0,
      "summary": "What it does and why it's relevant (2-3 sentences)",
      "integration_effort": "trivial | moderate | significant",
      "priority": "low | medium | high",
      "tags": ["rl", "data", "backtesting", "optimization", ...]
    }
  ],
  "top_recommendation": "Name of the most impactful tool to investigate"
}

EVALUATION CRITERIA:
- Directly applicable to RL-based trading simulation systems
- Compatible with Python 3.10+, PyTorch, Stable-Baselines3
- Free or very low cost
- Minimal integration effort preferred
- Must work on Windows with AMD GPU (DirectML)
"""


class InnovationScout(BaseCorpAgent):
    """Innovation Scout — tool and technique discovery.

    Responsibilities:
    1. Search for new relevant tools and libraries
    2. Evaluate relevance and integration effort
    3. Produce structured briefs for CEO review
    4. Track previously discovered tools to avoid duplicates
    """

    def __init__(
        self,
        state: CorporationState,
        decision_log: DecisionLog,
        interval_hours: int = 168,  # weekly
    ):
        super().__init__("innovation_scout", state, decision_log)
        self._interval_hours = interval_hours
        self._last_scout: str | None = None
        self._known_tools: set[str] = set()

    def should_run(self) -> bool:
        """Check if enough time has passed since last run."""
        if self._last_scout is None:
            return True
        try:
            last = datetime.fromisoformat(self._last_scout)
            return datetime.now() - last > timedelta(hours=self._interval_hours)
        except ValueError:
            return True

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Scout for new tools and techniques.

        Context keys:
        - force: bool — override schedule check
        - focus_areas: list[str] — specific areas to search
        """
        result = {
            "briefs": [],
            "top_recommendation": None,
            "llm_used": False,
            "new_discoveries": 0,
        }

        # Check schedule unless forced
        if not context.get("force", False) and not self.should_run():
            result["skipped"] = True
            result["reason"] = "Not yet due (weekly schedule)"
            self._mark_run(result)
            return result

        focus_areas = context.get("focus_areas", [
            "reinforcement learning trading",
            "stable-baselines3 improvements",
            "vectorbt alternatives",
            "reward engineering RL",
            "market simulation backtesting",
        ])

        # Try LLM-based scouting
        llm_response = self._llm_scout(focus_areas)

        if llm_response:
            result.update(llm_response)
            result["llm_used"] = True
        else:
            # Fallback: static list of known useful tools
            result["briefs"] = self._static_recommendations()

        # Filter out previously seen tools
        new_briefs = []
        for brief in result["briefs"]:
            tool_name = brief.get("tool_name", "").lower()
            if tool_name and tool_name not in self._known_tools:
                new_briefs.append(brief)
                self._known_tools.add(tool_name)

        result["briefs"] = new_briefs
        result["new_discoveries"] = len(new_briefs)

        # Store in corp state
        for brief in new_briefs:
            self.state.add_innovation_brief(brief)

        if new_briefs:
            result["top_recommendation"] = new_briefs[0].get("tool_name")

        self._last_scout = datetime.now().isoformat()

        self.log_decision(
            "innovation_scout",
            detail={
                "discoveries": len(new_briefs),
                "top": result.get("top_recommendation"),
            },
            outcome="discoveries" if new_briefs else "no_new",
        )

        self.send_message(
            "chief_of_staff",
            "report",
            {
                "new_discoveries": len(new_briefs),
                "top_recommendation": result.get("top_recommendation"),
                "briefs": [b.get("tool_name") for b in new_briefs],
            },
            priority=1,
        )

        self._mark_run(result)
        return result

    def _llm_scout(self, focus_areas: list[str]) -> dict | None:
        """Use LLM to generate tool recommendations (Groq free tier or Anthropic)."""
        from corp.llm_client import call_llm_json

        focus_text = "\n".join(f"- {area}" for area in focus_areas)
        user_prompt = (
            "Search your knowledge for the latest tools, libraries, and "
            "techniques relevant to these areas:\n\n"
            f"{focus_text}\n\n"
            "Focus on Python tools compatible with:\n"
            "- PyTorch 2.x\n"
            "- Stable-Baselines3\n"
            "- Windows + AMD GPU (DirectML)\n"
            "- VectorBT for backtesting validation\n\n"
            "Provide 3-5 recommendations, prioritized by impact."
        )

        parsed = call_llm_json(SYSTEM_PROMPT, user_prompt, max_tokens=1500, temperature=0.5)
        if parsed is None:
            return None

        return {
            "briefs": parsed.get("briefs", []),
            "top_recommendation": parsed.get("top_recommendation"),
        }

    def _static_recommendations(self) -> list[dict]:
        """Fallback static list of known useful tools."""
        return [
            {
                "tool_name": "FinRL",
                "category": "library",
                "url": "https://github.com/AI4Finance-Foundation/FinRL",
                "relevance_score": 0.8,
                "summary": (
                    "Open-source framework for financial RL. Provides "
                    "pre-built environments and agents for stock simulation. "
                    "Could provide additional baseline comparisons."
                ),
                "integration_effort": "moderate",
                "priority": "medium",
                "tags": ["rl", "finance", "backtesting"],
            },
            {
                "tool_name": "SB3-Contrib",
                "category": "library",
                "url": "https://github.com/Stable-Baselines3-Contrib/stable-baselines3-contrib",
                "relevance_score": 0.9,
                "summary": (
                    "Community extensions for Stable-Baselines3 including "
                    "TQC, QRDQN, and other advanced algorithms. "
                    "Direct drop-in for existing SB3 training pipeline."
                ),
                "integration_effort": "trivial",
                "priority": "high",
                "tags": ["rl", "algorithms", "sb3"],
            },
            {
                "tool_name": "Optuna",
                "category": "library",
                "url": "https://optuna.org/",
                "relevance_score": 0.85,
                "summary": (
                    "Hyperparameter optimization framework. Already partially "
                    "used in meta_optimize.py but could be expanded with "
                    "pruning and multi-objective optimization."
                ),
                "integration_effort": "trivial",
                "priority": "medium",
                "tags": ["optimization", "hyperparameter"],
            },
        ]
