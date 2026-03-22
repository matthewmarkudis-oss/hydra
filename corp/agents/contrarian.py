"""Contrarian Agent (Taleb-style) — adversarial reviewer of positive results.

LLM-powered agent that challenges every positive result. Fires conditionally
when: (a) an agent passes ATHENA validation, (b) PROMETHEUS declares
convergence, or (c) best fitness > threshold.

Backtesting and training only.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from corp.agents.base_corp_agent import BaseCorpAgent
from corp.state.corporation_state import CorporationState
from corp.state.decision_log import DecisionLog

logger = logging.getLogger("corp.agents.contrarian")

SYSTEM_PROMPT = """You are the Contrarian Analyst of HydraCorp, inspired by Nassim Taleb's
philosophy. Your SOLE PURPOSE is to find flaws in seemingly good results.
You are skeptical of every positive outcome and search for hidden fragility.

YOUR MANDATE:
1. Challenge inflated Sharpe ratios — is it driven by one lucky episode?
2. Question walk-forward efficiency — is the out-of-sample window too short?
3. Check for overfitting — does the agent exploit a quirk in the training data?
4. Propose stress tests — higher transaction costs, tighter drawdown limits
5. Advocate barbell strategy — split between ultra-conservative and aggressive

YOU MUST find at least ONE concern with any positive result.
Good results that survive your scrutiny become STRONGER.

OUTPUT FORMAT — respond with valid JSON only:
{
  "verdict": "fragile | antifragile | inconclusive",
  "concerns": ["concern 1", "concern 2", ...],
  "stress_test_config": {
    "reward": { ... },
    "env": { ... }
  },
  "recommendation": "Brief recommendation (1-2 sentences)",
  "fragility_score": 0.0 to 1.0
}

FRAGILITY SCORING:
- 0.0-0.3: Antifragile (robust to stress, multiple scenarios)
- 0.3-0.6: Moderate (some hidden risks but generally sound)
- 0.6-1.0: Fragile (likely to break under stress, overfit, or lucky)
"""


class Contrarian(BaseCorpAgent):
    """Contrarian Agent — Taleb-style adversarial analysis.

    Responsibilities:
    1. Challenge positive backtesting results for hidden fragility
    2. Propose stress-test configurations
    3. Detect overfitting and inflated metrics
    4. Advocate for robustness over raw performance
    """

    def __init__(
        self,
        state: CorporationState,
        decision_log: DecisionLog,
        trigger_fitness: float = 0.5,
    ):
        super().__init__("contrarian", state, decision_log)
        self._trigger_fitness = trigger_fitness

    def should_fire(self, context: dict[str, Any]) -> bool:
        """Determine if the contrarian should activate.

        Fires when:
        - Any agent passed ATHENA validation
        - Best return exceeds trigger threshold
        - Pipeline declares convergence
        """
        pipeline_results = context.get("pipeline_results", {})

        # Trigger on passed agents
        if pipeline_results.get("passed_count", 0) > 0:
            return True

        # Trigger on high returns
        if pipeline_results.get("best_return", 0) > self._trigger_fitness:
            return True

        # Trigger on convergence
        if pipeline_results.get("convergence_declared", False):
            return True

        return False

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Run adversarial analysis on pipeline results.

        Context keys:
        - pipeline_results: Dict with best_return, best_agent, passed_count, etc.
        - config_dict: Current HydraConfig dict
        """
        result = {
            "fired": False,
            "verdict": "inconclusive",
            "concerns": [],
            "stress_test_config": None,
            "fragility_score": 0.5,
            "llm_used": False,
        }

        # Check trigger conditions
        if not self.should_fire(context):
            result["reason"] = "Trigger conditions not met"
            self._mark_run(result)
            return result

        result["fired"] = True
        pipeline_results = context.get("pipeline_results", {})
        config_dict = context.get("config_dict", {})

        # Try LLM analysis
        user_prompt = self._build_prompt(pipeline_results, config_dict)
        llm_response = self._call_llm(user_prompt)

        if llm_response:
            result.update(llm_response)
            result["llm_used"] = True
        else:
            result.update(self._rule_based_scrutiny(pipeline_results, config_dict))

        # Alert if fragile
        if result["fragility_score"] > 0.6:
            self.send_message(
                "broadcast",
                "alert",
                {
                    "source": "contrarian",
                    "verdict": result["verdict"],
                    "concerns": result["concerns"],
                    "fragility_score": result["fragility_score"],
                },
                priority=4,
            )

        # Submit stress test config as proposal if available
        if result.get("stress_test_config"):
            self.state.submit_proposal({
                "source": "contrarian",
                "type": "stress_test",
                "patch": result["stress_test_config"],
                "concerns": result["concerns"],
                "fragility_score": result["fragility_score"],
            })

        self.log_decision(
            "contrarian_review",
            detail={
                "verdict": result["verdict"],
                "fragility_score": result["fragility_score"],
                "num_concerns": len(result["concerns"]),
                "llm_used": result["llm_used"],
            },
            outcome=result["verdict"],
        )

        self._mark_run(result)
        return result

    def _build_prompt(self, pipeline_results: dict, config_dict: dict) -> str:
        """Build the adversarial analysis prompt."""
        lines = [
            "Scrutinize these backtesting results for hidden fragility:\n",
            "## Results Being Claimed As Positive",
            f"- Best agent: {pipeline_results.get('best_agent', 'unknown')}",
            f"- Best return: {pipeline_results.get('best_return', 0):.4f} ({pipeline_results.get('best_return', 0) * 100:.2f}%)",
            f"- Agents passed validation: {pipeline_results.get('passed_count', 0)}/{pipeline_results.get('total_agents', 0)}",
            f"- Excess return vs benchmark: {pipeline_results.get('excess_return', 0):.4f}",
        ]

        # Current config
        lines.append("\n## Config Used")
        reward = config_dict.get("reward", {})
        env = config_dict.get("env", {})
        for k, v in reward.items():
            lines.append(f"  reward.{k}: {v}")
        for k, v in env.items():
            lines.append(f"  env.{k}: {v}")

        # Performance history for trend analysis
        corp_state = self.state.get_full_state()
        history = corp_state.get("agent_performance_history", [])[-10:]
        if history:
            lines.append("\n## Performance History (last 10 runs)")
            for entry in history:
                lines.append(
                    f"  Run #{entry.get('run', '?')}: "
                    f"return={entry.get('best_return', 0):.4f}, "
                    f"passed={entry.get('passed_count', 0)}"
                )

        lines.append(
            "\n## Your Task"
            "\nFind AT LEAST one flaw. Propose a stress-test config that would "
            "expose fragility. Use higher costs, tighter drawdowns, or different "
            "market conditions."
        )

        return "\n".join(lines)

    def _call_llm(self, user_prompt: str) -> dict | None:
        """Call the LLM for adversarial analysis."""
        try:
            import anthropic
        except ImportError:
            logger.debug("anthropic not installed, using rule-based fallback")
            return None

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            logger.debug("No ANTHROPIC_API_KEY, using rule-based fallback")
            return None

        try:
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1200,
                temperature=0.4,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            parsed = json.loads(text)

            return {
                "verdict": parsed.get("verdict", "inconclusive"),
                "concerns": parsed.get("concerns", []),
                "stress_test_config": parsed.get("stress_test_config"),
                "recommendation": parsed.get("recommendation", ""),
                "fragility_score": min(max(parsed.get("fragility_score", 0.5), 0.0), 1.0),
            }

        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return None

    def _rule_based_scrutiny(
        self, pipeline_results: dict, config_dict: dict
    ) -> dict[str, Any]:
        """Fallback rule-based adversarial analysis."""
        concerns = []
        fragility = 0.5
        stress_test = {}

        best_return = pipeline_results.get("best_return", 0)
        passed_count = pipeline_results.get("passed_count", 0)
        total_agents = pipeline_results.get("total_agents", 0)
        excess = pipeline_results.get("excess_return", 0)

        reward = config_dict.get("reward", {})
        env = config_dict.get("env", {})

        # Concern: only one agent passed → fragile
        if passed_count == 1 and total_agents > 3:
            concerns.append(
                f"Only 1 of {total_agents} agents passed — the 'best' result "
                "may be an outlier rather than a robust strategy."
            )
            fragility += 0.15

        # Concern: very high return might be overfitting
        if best_return > 0.20:
            concerns.append(
                f"Return of {best_return:.1%} is suspiciously high. "
                "Check for overfitting to training data or look-ahead bias."
            )
            fragility += 0.1

        # Concern: low transaction penalty → unrealistic
        tp = reward.get("transaction_penalty", 0.01)
        if tp < 0.01:
            concerns.append(
                f"Transaction penalty of {tp} is unrealistically low. "
                "Real-world slippage and commissions would reduce returns."
            )
            stress_test.setdefault("reward", {})["transaction_penalty"] = 0.02
            fragility += 0.1

        # Concern: high max drawdown allowance
        dd = env.get("max_drawdown_pct", 0.25)
        if dd > 0.20:
            concerns.append(
                f"Max drawdown of {dd:.0%} is permissive. "
                f"On $2,500, that's ${2500 * dd:,.0f} at risk."
            )
            stress_test.setdefault("env", {})["max_drawdown_pct"] = 0.15
            fragility += 0.05

        # Concern: low position limit might not be binding
        mp = env.get("max_position_pct", 0.40)
        if mp >= 0.50:
            concerns.append(
                f"Max position of {mp:.0%} allows heavy concentration. "
                "A single stock collapse could be devastating."
            )
            stress_test.setdefault("env", {})["max_position_pct"] = 0.30

        # Concern: excess return could be period-specific
        if excess > 0.10:
            concerns.append(
                f"Excess return of {excess:.1%} over benchmark is unusually high. "
                "Test across different market periods to confirm robustness."
            )

        if not concerns:
            concerns.append(
                "Results appear reasonable but should be validated "
                "across multiple market regimes."
            )

        fragility = min(max(fragility, 0.0), 1.0)
        verdict = (
            "fragile" if fragility > 0.6
            else "inconclusive" if fragility > 0.3
            else "antifragile"
        )

        return {
            "verdict": verdict,
            "concerns": concerns,
            "stress_test_config": stress_test if stress_test else None,
            "fragility_score": round(fragility, 2),
            "recommendation": (
                "Run stress tests with tighter constraints before deploying."
                if fragility > 0.5
                else "Results appear reasonably robust. Continue monitoring."
            ),
        }
