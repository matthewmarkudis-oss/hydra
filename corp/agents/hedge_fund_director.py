"""Hedge Fund Director — LLM-powered strategic advisor for config optimization.

Receives structured JSON of all protocol outputs (CHIMERA fitness, PROMETHEUS
weights, ELEOS conviction, ATHENA validation) after each pipeline run.
Generates a strategy memo recommending specific parameter changes to
RewardConfig and PoolConfig.

Backtesting and training only.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from corp.agents.base_corp_agent import BaseCorpAgent
from corp.config.ticker_universe import TickerSelector
from corp.state.corporation_state import CorporationState
from corp.state.decision_log import DecisionLog

logger = logging.getLogger("corp.agents.hedge_fund_director")

SYSTEM_PROMPT = """You are the Hedge Fund Director of HydraCorp, an AI-managed trading research
corporation. Your role is to analyze backtesting pipeline results and propose
specific configuration changes to improve training outcomes.

CONTEXT:
- Starting capital: $2,500 CAD (backtesting simulation)
- The RL pipeline trains PPO, SAC, and RecurrentPPO agents on stock data
- Results include: fitness scores, Sharpe ratios, drawdowns, win rates
- You propose changes to reward parameters, environment settings, and pool config
- Your proposals are validated by the Senior Dev before application

OUTPUT FORMAT — you MUST respond with valid JSON only:
{
  "memo": "Brief strategic rationale (2-3 sentences)",
  "confidence": 0.0 to 1.0,
  "proposed_patch": {
    "reward": { ... },
    "env": { ... },
    "pool": { ... }
  },
  "risk_assessment": "low | medium | high",
  "expected_improvement": "Brief description of expected outcome"
}

TICKER UNIVERSE AWARENESS:
- You may recommend ticker list changes when geopolitics data shows strong sector signals
- Available tiers: 10, 20, 50, or 100 tickers
- Ticker changes trigger retraining — only propose when macro conditions justify it
- Include a "ticker_change" key in proposed_patch if recommending ticker changes:
  "ticker_change": {
    "target_tier": 10|20|50|100,
    "sector_bias": {"tech": -1 to 1, "energy": -1 to 1, ...},
    "reasoning": "Why this change is warranted"
  }

CONSTRAINTS:
- transaction_penalty must be >= 0.005
- drawdown_penalty must be >= 0.1
- reward_scale must be between 10 and 500
- max_position_pct must be <= 0.60
- max_drawdown_pct must be <= 0.40
- Only propose changes you have evidence to support
- If results are already good (Sharpe > 1.5, positive returns), propose conservative changes
"""


class HedgeFundDirector(BaseCorpAgent):
    """Hedge Fund Director — LLM-powered strategic config advisor.

    Responsibilities:
    1. Analyze pipeline results (fitness, Sharpe, drawdown, win rate)
    2. Generate strategy memos with specific config change proposals
    3. Submit proposals for Senior Dev review
    4. Track which proposals improved or worsened performance
    """

    def __init__(
        self,
        state: CorporationState,
        decision_log: DecisionLog,
        model: str = "routine",
    ):
        super().__init__("hedge_fund_director", state, decision_log)
        self._model = model  # "routine" (Haiku) or "strategic" (Sonnet)
        self._proposal_history: list[dict] = []

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Analyze pipeline results and propose config changes.

        Context keys:
        - pipeline_results: Dict with best_return, best_agent, passed_count, etc.
        - config_dict: Current HydraConfig dict
        - regime: Current macro regime assessment (optional)
        """
        result = {
            "memo": "",
            "proposed_patch": None,
            "confidence": 0.0,
            "risk_assessment": "medium",
            "llm_used": False,
        }

        pipeline_results = context.get("pipeline_results", {})
        config_dict = context.get("config_dict", {})
        regime = context.get("regime", {})

        if not pipeline_results:
            result["memo"] = "No pipeline results to analyze."
            self._mark_run(result)
            return result

        # Build analysis prompt
        user_prompt = self._build_prompt(pipeline_results, config_dict, regime)

        # Try LLM call
        llm_response = self._call_llm(user_prompt)

        if llm_response:
            result.update(llm_response)
            result["llm_used"] = True
        else:
            # Fallback: rule-based recommendations
            result.update(self._rule_based_analysis(pipeline_results, config_dict))
            result["llm_used"] = False

        # Submit proposal if we have one
        if result.get("proposed_patch"):
            self.state.submit_proposal({
                "source": "hedge_fund_director",
                "memo": result["memo"],
                "patch": result["proposed_patch"],
                "confidence": result["confidence"],
                "risk": result["risk_assessment"],
                "pipeline_run": pipeline_results.get("run_number", 0),
            })
            self.send_message(
                "senior_dev",
                "proposal",
                {
                    "patch": result["proposed_patch"],
                    "memo": result["memo"],
                    "confidence": result["confidence"],
                },
                priority=3,
                requires_response=True,
            )

        # Ticker intelligence: evaluate whether ticker list should change
        ticker_proposal = self._rule_based_ticker_analysis(
            pipeline_results, config_dict, regime,
        )
        if ticker_proposal:
            self._submit_ticker_proposal(ticker_proposal, config_dict)
            result["ticker_proposal"] = ticker_proposal

        self.log_decision(
            "strategy_memo",
            detail={
                "memo": result["memo"],
                "confidence": result["confidence"],
                "has_patch": result.get("proposed_patch") is not None,
                "has_ticker_proposal": ticker_proposal is not None,
                "llm_used": result["llm_used"],
            },
            outcome="proposal_submitted" if result.get("proposed_patch") else "no_action",
        )

        self._mark_run(result)
        return result

    def _build_prompt(
        self,
        pipeline_results: dict,
        config_dict: dict,
        regime: dict,
    ) -> str:
        """Build the analysis prompt for the LLM."""
        lines = ["Analyze these backtesting pipeline results and propose config improvements:\n"]

        # Pipeline results
        lines.append("## Pipeline Results")
        lines.append(f"- Best agent: {pipeline_results.get('best_agent', 'unknown')}")
        lines.append(f"- Best return: {pipeline_results.get('best_return', 0):.4f}")
        lines.append(f"- Passed agents: {pipeline_results.get('passed_count', 0)}/{pipeline_results.get('total_agents', 0)}")
        lines.append(f"- Excess vs benchmark: {pipeline_results.get('excess_return', 0):.4f}")

        # Current config (key sections)
        lines.append("\n## Current Config")
        for section in ("reward", "env", "pool"):
            if section in config_dict:
                lines.append(f"\n### {section}")
                for k, v in config_dict[section].items():
                    lines.append(f"  {k}: {v}")

        # Regime if available
        if regime and regime.get("classification", "unknown") != "unknown":
            lines.append(f"\n## Market Regime: {regime['classification']}")
            lines.append(f"  Volatility outlook: {regime.get('volatility_outlook', 'unknown')}")

            # Ticker recommendations from GeopoliticsExpert
            ticker_recs = regime.get("ticker_recommendations", {})
            if ticker_recs:
                overweight = ticker_recs.get("sectors_to_overweight", [])
                underweight = ticker_recs.get("sectors_to_underweight", [])
                if overweight or underweight:
                    lines.append("\n## Geopolitics Ticker Signals")
                    if overweight:
                        lines.append(f"  Overweight: {', '.join(overweight)}")
                    if underweight:
                        lines.append(f"  Underweight: {', '.join(underweight)}")
                    reasoning = ticker_recs.get("reasoning", "")
                    if reasoning:
                        lines.append(f"  Reasoning: {reasoning}")

        # 13F Hedge Fund Consensus (from SEC filings)
        consensus_13f = self._get_13f_consensus()
        if consensus_13f and consensus_13f.get("consensus"):
            lines.append("\n## 13F Hedge Fund Consensus")
            lines.append(f"  Funds tracked: {consensus_13f.get('fund_count', 0)}")
            for sector, score in sorted(
                consensus_13f["consensus"].items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            ):
                direction = "overweight" if score > 0 else "underweight"
                lines.append(f"  {sector}: {score:+.2f} ({direction})")

        # Current ticker list
        current_tickers = config_dict.get("data", {}).get("tickers", [])
        if current_tickers:
            lines.append(f"\n## Current Tickers ({len(current_tickers)})")
            lines.append(f"  {', '.join(current_tickers)}")

        # Performance history
        corp_state = self.state.get_full_state()
        history = corp_state.get("agent_performance_history", [])[-5:]
        if history:
            lines.append("\n## Recent Performance Trend")
            for entry in history:
                lines.append(
                    f"  Run #{entry.get('run', '?')}: "
                    f"best_return={entry.get('best_return', 0):.4f}, "
                    f"passed={entry.get('passed_count', 0)}"
                )

        return "\n".join(lines)

    def _call_llm(self, user_prompt: str) -> dict | None:
        """Call the LLM for analysis. Returns parsed JSON or None on failure."""
        try:
            import anthropic
        except ImportError:
            logger.debug("anthropic package not installed, using rule-based fallback")
            return None

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            logger.debug("No ANTHROPIC_API_KEY set, using rule-based fallback")
            return None

        try:
            client = anthropic.Anthropic(api_key=api_key)

            # Select model
            if self._model == "strategic":
                model = "claude-sonnet-4-20250514"
            else:
                model = "claude-3-haiku-20240307"

            response = client.messages.create(
                model=model,
                max_tokens=1500,
                temperature=0.3,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            # Parse JSON from response
            text = response.content[0].text.strip()

            # Handle markdown code blocks
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            parsed = json.loads(text)

            # Validate required fields
            if "proposed_patch" not in parsed:
                logger.warning("LLM response missing proposed_patch")
                return None

            return {
                "memo": parsed.get("memo", ""),
                "proposed_patch": parsed["proposed_patch"],
                "confidence": min(max(parsed.get("confidence", 0.5), 0.0), 1.0),
                "risk_assessment": parsed.get("risk_assessment", "medium"),
                "expected_improvement": parsed.get("expected_improvement", ""),
            }

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return None
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return None

    def _rule_based_analysis(
        self, pipeline_results: dict, config_dict: dict
    ) -> dict[str, Any]:
        """Fallback rule-based analysis when LLM is unavailable."""
        best_return = pipeline_results.get("best_return", 0)
        passed_count = pipeline_results.get("passed_count", 0)
        total_agents = pipeline_results.get("total_agents", 0)
        excess = pipeline_results.get("excess_return", 0)

        patch = {}
        memo_parts = []

        reward = config_dict.get("reward", {})
        env = config_dict.get("env", {})

        # If no agents passed, try adjusting reward parameters
        if passed_count == 0 and total_agents > 0:
            memo_parts.append("No agents passed validation.")

            # Reduce transaction penalty if high
            tp = reward.get("transaction_penalty", 0.01)
            if tp > 0.02:
                patch.setdefault("reward", {})["transaction_penalty"] = round(tp * 0.7, 4)
                memo_parts.append(f"Reducing transaction penalty from {tp} to {tp * 0.7:.4f}.")

            # Increase reward scale if low
            rs = reward.get("reward_scale", 100)
            if rs < 200:
                patch.setdefault("reward", {})["reward_scale"] = min(rs * 1.5, 300)
                memo_parts.append("Increasing reward scale for stronger learning signal.")

        # If negative returns, tighten risk controls
        elif best_return < -0.05:
            memo_parts.append(f"Negative returns ({best_return:.2%}). Tightening risk controls.")

            dd = env.get("max_drawdown_pct", 0.25)
            if dd > 0.15:
                patch.setdefault("env", {})["max_drawdown_pct"] = round(dd * 0.8, 3)
                memo_parts.append(f"Reducing max drawdown from {dd:.1%} to {dd * 0.8:.1%}.")

            dp = reward.get("drawdown_penalty", 0.5)
            if dp < 1.0:
                patch.setdefault("reward", {})["drawdown_penalty"] = round(dp * 1.5, 3)
                memo_parts.append("Increasing drawdown penalty.")

        # If positive but underperforming benchmark
        elif excess < 0 and best_return > 0:
            memo_parts.append(
                f"Positive returns ({best_return:.2%}) but underperforming benchmark. "
                "Adjusting for alpha generation."
            )

            # Slightly reduce holding penalty to encourage longer-term positions
            hp = reward.get("hold_penalty", 0.001)
            if hp > 0.0005:
                patch.setdefault("reward", {})["hold_penalty"] = round(hp * 0.5, 5)
                memo_parts.append("Reducing hold penalty to encourage longer positions.")

        # If results are good, propose minor tweaks
        elif best_return > 0.05:
            memo_parts.append(
                f"Good results ({best_return:.2%}). Proposing minor optimizations."
            )
            # No major changes — don't fix what isn't broken
            patch = {}

        if not memo_parts:
            memo_parts.append("Insufficient data for recommendations.")

        return {
            "memo": " ".join(memo_parts),
            "proposed_patch": patch if patch else None,
            "confidence": 0.4,  # Lower confidence for rule-based
            "risk_assessment": "low",
        }

    def _get_13f_consensus(self) -> dict:
        """Fetch 13F consensus data from HedgeFundTracker."""
        try:
            from trading_agents.data.hedge_fund_tracker import HedgeFundTracker
            tracker = HedgeFundTracker()
            return tracker.get_sector_rotation()
        except Exception as e:
            logger.debug(f"Could not fetch 13F consensus: {e}")
            return {}

    # ── Ticker Intelligence ────────────────────────────────────────────────

    def _rule_based_ticker_analysis(
        self,
        pipeline_results: dict,
        config_dict: dict,
        regime: dict,
    ) -> dict | None:
        """Evaluate whether the ticker list should change.

        Four gates must be passed before proposing a change:
        1. Geopolitics confidence > 0.5
        2. Max sector bias > 0.3
        3. Pipeline not already performing well
        4. Computed churn > 20%

        Returns a ticker change proposal dict, or None if no change warranted.
        """
        if not regime:
            return None

        # Gate 1: Confidence threshold
        confidence = regime.get("confidence", 0.0)
        if confidence < 0.5:
            logger.debug("Ticker analysis: confidence %.2f below 0.5 threshold", confidence)
            return None

        # Gate 2: Must have meaningful sector signals
        sector_bias = regime.get("sector_bias", {})
        ticker_recs = regime.get("ticker_recommendations", {})

        # Derive sector bias from ticker_recommendations if sector_bias is empty
        if not sector_bias and ticker_recs:
            for s in ticker_recs.get("sectors_to_overweight", []):
                sector_bias[s] = sector_bias.get(s, 0) + 0.5
            for s in ticker_recs.get("sectors_to_underweight", []):
                sector_bias[s] = sector_bias.get(s, 0) - 0.5

        # Boost sector bias with 13F consensus when 5+ funds agree
        consensus_13f = self._get_13f_consensus()
        if consensus_13f and consensus_13f.get("consensus"):
            fund_count = consensus_13f.get("fund_count", 0)
            if fund_count >= 5:
                for sector, score in consensus_13f["consensus"].items():
                    if abs(score) > 0.2:  # Meaningful consensus
                        sector_bias[sector] = sector_bias.get(sector, 0) + score * 0.3

        max_bias = max((abs(v) for v in sector_bias.values()), default=0.0)
        if max_bias < 0.3:
            logger.debug("Ticker analysis: max bias %.2f below 0.3 threshold", max_bias)
            return None

        # Gate 3: Don't change if pipeline is performing well
        best_return = pipeline_results.get("best_return", 0)
        passed_count = pipeline_results.get("passed_count", 0)
        if best_return > 0.05 and passed_count >= 2:
            logger.debug("Ticker analysis: pipeline performing well, no change needed")
            return None

        # Compute recommended tier and tickers
        classification = regime.get("classification", "risk_on")
        vol_outlook = regime.get("volatility_outlook", "stable")
        current_tickers = config_dict.get("data", {}).get("tickers", [])

        target_tier = TickerSelector.recommend_tier(
            classification, vol_outlook, sector_bias, len(current_tickers),
        )
        proposed_tickers = TickerSelector.select_tickers(
            target_tier, sector_bias, current_tickers, classification,
        )

        # Gate 4: Churn must be significant enough to justify retraining
        churn = TickerSelector.compute_churn(current_tickers, proposed_tickers)
        if churn["churn_pct"] < 0.20:
            logger.debug("Ticker analysis: churn %.1f%% below 20%% threshold", churn["churn_pct"] * 100)
            return None

        return {
            "target_tier": target_tier,
            "proposed_tickers": proposed_tickers,
            "current_tickers": current_tickers,
            "churn": churn,
            "sector_bias": sector_bias,
            "regime": classification,
            "confidence": confidence,
            "reasoning": ticker_recs.get("reasoning", f"Regime: {classification}, sectors shifting"),
        }

    def _submit_ticker_proposal(self, proposal: dict, config_dict: dict) -> None:
        """Submit a ticker change proposal through the corp proposals system."""
        self.state.submit_proposal({
            "type": "ticker_change",
            "source": "hedge_fund_director",
            "memo": (
                f"Ticker change: {len(proposal['current_tickers'])} -> "
                f"{len(proposal['proposed_tickers'])} tickers "
                f"(tier {proposal['target_tier']}). "
                f"{proposal['reasoning']}"
            ),
            "patch": {
                "data": {"tickers": proposal["proposed_tickers"]},
                "env": {"num_stocks": len(proposal["proposed_tickers"])},
            },
            "ticker_metadata": {
                "target_tier": proposal["target_tier"],
                "churn": proposal["churn"],
                "sector_bias": proposal["sector_bias"],
                "regime": proposal["regime"],
                "sector_distribution": TickerSelector.get_sector_distribution(
                    proposal["proposed_tickers"]
                ),
            },
            "confidence": proposal["confidence"],
            "risk": "high",  # Ticker changes always high risk (retraining needed)
        })

        self.send_message(
            "senior_dev",
            "proposal",
            {
                "type": "ticker_change",
                "proposed_tickers": proposal["proposed_tickers"],
                "churn": proposal["churn"],
                "memo": proposal["reasoning"],
            },
            priority=4,
            requires_response=True,
        )

        logger.info(
            "Ticker proposal submitted: %d -> %d tickers, churn %.1f%%",
            len(proposal["current_tickers"]),
            len(proposal["proposed_tickers"]),
            proposal["churn"]["churn_pct"] * 100,
        )
