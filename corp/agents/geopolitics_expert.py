"""Geopolitics Expert — news-driven regime classification for config adjustments.

LLM-powered agent that fetches headlines from free APIs (NewsAPI, RSS feeds),
classifies the current macro regime (risk_on / risk_off / crisis), and produces
config adjustment recommendations. Does NOT generate trade signals — only
influences how RL agents are configured for the next training cycle.

Backtesting and training only. Runs on a daily schedule.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any

from corp.agents.base_corp_agent import BaseCorpAgent
from corp.state.corporation_state import CorporationState
from corp.state.decision_log import DecisionLog

logger = logging.getLogger("corp.agents.geopolitics_expert")

SYSTEM_PROMPT = """You are the Geopolitics Expert of HydraCorp, an AI-managed backtesting
research corporation. You analyze news headlines to classify the current
macro-economic regime and suggest training configuration adjustments.

YOU DO NOT GENERATE TRADE SIGNALS. You only influence training config parameters.

OUTPUT FORMAT — respond with valid JSON only:
{
  "regime": "risk_on | risk_off | crisis | antifragile",
  "volatility_outlook": "low | stable | elevated | extreme",
  "sector_bias": {
    "tech": -1.0 to 1.0,
    "energy": -1.0 to 1.0,
    "finance": -1.0 to 1.0,
    "healthcare": -1.0 to 1.0,
    "consumer": -1.0 to 1.0
  },
  "confidence": 0.0 to 1.0,
  "summary": "2-3 sentence summary of current conditions",
  "ticker_recommendations": {
    "sectors_to_overweight": ["list of sector names to increase exposure"],
    "sectors_to_underweight": ["list of sector names to decrease exposure"],
    "specific_tickers_to_add": ["optional: specific tickers relevant to current events"],
    "specific_tickers_to_remove": ["optional: tickers that face outsized risk"],
    "reasoning": "Brief explanation of why these changes are warranted"
  },
  "config_suggestions": {
    "reason": "Why these adjustments",
    "reward": { ... },
    "env": { ... }
  }
}

TICKER RECOMMENDATION GUIDELINES:
- Only recommend ticker changes when there is a STRONG macro justification
- Geopolitical conflicts in oil-producing regions -> overweight energy
- Tech regulatory crackdowns -> underweight tech, add defensive sectors
- Financial contagion risk -> underweight finance, overweight utilities/healthcare
- Crisis regime -> recommend concentrating to fewer, high-liquidity tickers
- Do NOT recommend changes if the regime is stable risk_on with no sector signals

REGIME DEFINITIONS:
- risk_on: Markets bullish, low VIX, positive economic data. Train with normal parameters.
- risk_off: Uncertainty rising, moderate VIX, mixed data. Tighten drawdown limits, increase penalties.
- crisis: Black swan events, high VIX, market panic. Maximum conservatism, minimal position sizes.
- antifragile: High volatility with identifiable asymmetric opportunities. Wars, oil shocks, sector
  disruptions, tariff escalation — situations where specific sectors surge while others crash.
  Use this when there is chaos BUT clear directional signals (e.g., energy up during oil crisis,
  defense up during conflict). Trains agents to protect capital tightly while swinging hard on
  high-conviction asymmetric bets. Inspired by Taleb's barbell strategy.
"""


class GeopoliticsExpert(BaseCorpAgent):
    """Geopolitics Expert — macro regime classification.

    Responsibilities:
    1. Fetch recent news headlines (free APIs)
    2. Classify macro regime (risk_on / risk_off / crisis)
    3. Produce config adjustment recommendations
    4. Update regime in corporation state
    """

    def __init__(
        self,
        state: CorporationState,
        decision_log: DecisionLog,
        interval_hours: int = 24,
    ):
        super().__init__("geopolitics_expert", state, decision_log)
        self._interval_hours = interval_hours
        self._last_fetch: str | None = None

    def should_run(self) -> bool:
        """Check if enough time has passed since last run."""
        if self._last_fetch is None:
            return True
        try:
            last = datetime.fromisoformat(self._last_fetch)
            return datetime.now() - last > timedelta(hours=self._interval_hours)
        except ValueError:
            return True

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Fetch news and classify regime.

        Context keys:
        - force: bool — override schedule check
        """
        result = {
            "regime": "unknown",
            "volatility_outlook": "stable",
            "sector_bias": {},
            "confidence": 0.0,
            "headlines_fetched": 0,
            "llm_used": False,
        }

        # Check schedule unless forced
        if not context.get("force", False) and not self.should_run():
            result["skipped"] = True
            result["reason"] = "Not yet due (schedule)"
            self._mark_run(result)
            return result

        # Fetch headlines
        headlines = self._fetch_headlines()
        result["headlines_fetched"] = len(headlines)

        if not headlines:
            # No headlines available — use default regime
            result["regime"] = "risk_on"
            result["confidence"] = 0.2
            result["summary"] = "No news data available. Defaulting to risk_on."
            self._update_state(result)
            self._mark_run(result)
            return result

        # Analyze with LLM
        llm_response = self._analyze_headlines(headlines)

        if llm_response:
            result.update(llm_response)
            result["llm_used"] = True
        else:
            # Rule-based fallback
            result.update(self._rule_based_classification(headlines))

        # Update corp state
        self._update_state(result)
        self._last_fetch = datetime.now().isoformat()

        self.log_decision(
            "regime_classification",
            detail={
                "regime": result["regime"],
                "confidence": result["confidence"],
                "headlines": len(headlines),
            },
            outcome=result["regime"],
        )

        self.send_message(
            "chief_of_staff",
            "report",
            {
                "regime": result["regime"],
                "volatility_outlook": result["volatility_outlook"],
                "confidence": result["confidence"],
                "ticker_recommendations": result.get("ticker_recommendations", {}),
            },
            priority=2,
        )

        self._mark_run(result)
        return result

    def _fetch_headlines(self) -> list[dict]:
        """Fetch recent headlines from free news APIs."""
        headlines = []

        # Try NewsAPI (free tier: 100 requests/day)
        newsapi_key = os.environ.get("NEWSAPI_KEY", "")
        if newsapi_key:
            headlines.extend(self._fetch_newsapi(newsapi_key))

        # Try Finnhub (free tier: 60 requests/min)
        finnhub_key = os.environ.get("FINNHUB_API_KEY", "")
        if finnhub_key:
            headlines.extend(self._fetch_finnhub(finnhub_key))

        # Fallback: hardcoded RSS feeds (no API key needed)
        if not headlines:
            headlines.extend(self._fetch_rss())

        return headlines[:50]  # Cap at 50 headlines

    def _fetch_newsapi(self, api_key: str) -> list[dict]:
        """Fetch from NewsAPI free tier."""
        try:
            import urllib.request
            import urllib.parse

            params = urllib.parse.urlencode({
                "q": "stock market economy finance",
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 20,
                "apiKey": api_key,
            })
            url = f"https://newsapi.org/v2/everything?{params}"

            req = urllib.request.Request(url, headers={"User-Agent": "HydraCorp/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())

            articles = data.get("articles", [])
            return [
                {
                    "title": a.get("title", ""),
                    "source": a.get("source", {}).get("name", "unknown"),
                    "published": a.get("publishedAt", ""),
                }
                for a in articles
                if a.get("title")
            ]

        except Exception as e:
            logger.debug(f"NewsAPI fetch failed: {e}")
            return []

    def _fetch_finnhub(self, api_key: str) -> list[dict]:
        """Fetch from Finnhub general news."""
        try:
            import urllib.request

            url = f"https://finnhub.io/api/v1/news?category=general&token={api_key}"
            req = urllib.request.Request(url, headers={"User-Agent": "HydraCorp/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())

            return [
                {
                    "title": item.get("headline", ""),
                    "source": item.get("source", "unknown"),
                    "published": datetime.fromtimestamp(
                        item.get("datetime", 0)
                    ).isoformat() if item.get("datetime") else "",
                }
                for item in data[:20]
                if item.get("headline")
            ]

        except Exception as e:
            logger.debug(f"Finnhub fetch failed: {e}")
            return []

    def _fetch_rss(self) -> list[dict]:
        """Fetch from public RSS feeds (no API key needed)."""
        feeds = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
        ]

        headlines = []
        for feed_url in feeds:
            try:
                import urllib.request
                import xml.etree.ElementTree as ET

                req = urllib.request.Request(
                    feed_url, headers={"User-Agent": "HydraCorp/1.0"}
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    xml_data = resp.read().decode()

                root = ET.fromstring(xml_data)
                for item in root.iter("item"):
                    title_el = item.find("title")
                    if title_el is not None and title_el.text:
                        headlines.append({
                            "title": title_el.text,
                            "source": "RSS",
                            "published": "",
                        })
            except Exception as e:
                logger.debug(f"RSS fetch failed for {feed_url}: {e}")

        return headlines

    def _analyze_headlines(self, headlines: list[dict]) -> dict | None:
        """Analyze headlines with LLM (Groq free tier or Anthropic)."""
        from corp.llm_client import call_llm_json

        headline_text = "\n".join(
            f"- [{h['source']}] {h['title']}" for h in headlines[:30]
        )
        user_prompt = (
            "Analyze these recent financial headlines and classify the "
            "current macro regime:\n\n" + headline_text
        )

        parsed = call_llm_json(SYSTEM_PROMPT, user_prompt, max_tokens=800, temperature=0.2)
        if parsed is None:
            return None

        return {
            "regime": parsed.get("regime", "risk_on"),
            "volatility_outlook": parsed.get("volatility_outlook", "stable"),
            "sector_bias": parsed.get("sector_bias", {}),
            "confidence": min(max(parsed.get("confidence", 0.5), 0.0), 1.0),
            "summary": parsed.get("summary", ""),
            "config_suggestions": parsed.get("config_suggestions"),
            "ticker_recommendations": parsed.get("ticker_recommendations", {}),
        }

    def _rule_based_classification(self, headlines: list[dict]) -> dict[str, Any]:
        """Simple keyword-based regime classification."""
        text = " ".join(h["title"].lower() for h in headlines)

        crisis_words = [
            "crash", "crisis", "panic", "recession", "collapse",
            "bankruptcy", "default", "contagion", "black swan",
        ]
        risk_off_words = [
            "fear", "uncertainty", "decline", "sell-off", "correction",
            "inflation", "rate hike", "hawkish", "volatility", "warning",
            "tariff", "war", "sanctions", "debt ceiling",
        ]
        risk_on_words = [
            "rally", "surge", "record high", "bull", "growth",
            "earnings beat", "dovish", "rate cut", "optimism",
            "recovery", "expansion",
        ]

        crisis_count = sum(1 for w in crisis_words if w in text)
        risk_off_count = sum(1 for w in risk_off_words if w in text)
        risk_on_count = sum(1 for w in risk_on_words if w in text)

        total = crisis_count + risk_off_count + risk_on_count
        if total == 0:
            return {
                "regime": "risk_on",
                "volatility_outlook": "stable",
                "sector_bias": {},
                "confidence": 0.2,
                "summary": "No strong signals detected. Defaulting to risk_on.",
            }

        if crisis_count >= 3 or crisis_count > risk_off_count:
            regime = "crisis"
            vol = "extreme"
            confidence = min(0.3 + crisis_count * 0.1, 0.8)
        elif risk_off_count > risk_on_count:
            regime = "risk_off"
            vol = "elevated"
            confidence = min(0.3 + risk_off_count * 0.05, 0.7)
        else:
            regime = "risk_on"
            vol = "low" if risk_on_count > 5 else "stable"
            confidence = min(0.3 + risk_on_count * 0.05, 0.7)

        # Upgrade to antifragile when there's volatility WITH clear sector signals
        # (chaos + identifiable asymmetric opportunities = antifragile, not just crisis)
        if total >= 3:
            sector_words = {
                "energy": ["oil", "opec", "pipeline", "drilling", "energy", "crude"],
                "defense": ["war", "military", "defense", "missile", "conflict", "nato"],
            }
            sector_signal_count = sum(
                1 for words in sector_words.values()
                for w in words if w in text
            )
            if regime in ("crisis", "risk_off") and sector_signal_count >= 3:
                regime = "antifragile"
                vol = "elevated"
                confidence = min(confidence + 0.1, 0.8)

        # Rule-based ticker recommendations based on keyword signals
        ticker_recs: dict[str, Any] = {
            "sectors_to_overweight": [],
            "sectors_to_underweight": [],
            "specific_tickers_to_add": [],
            "specific_tickers_to_remove": [],
            "reasoning": "",
        }

        # Only recommend changes when signals are strong enough
        if total >= 3:
            energy_words = ["oil", "opec", "pipeline", "drilling", "energy"]
            defense_words = ["war", "military", "defense", "missile", "conflict", "nato"]
            tech_reg_words = ["antitrust", "regulation", "ban", "fine", "scrutiny"]
            financial_words = ["bank failure", "contagion", "credit", "default", "liquidity"]

            energy_signal = sum(1 for w in energy_words if w in text)
            defense_signal = sum(1 for w in defense_words if w in text)
            tech_reg_signal = sum(1 for w in tech_reg_words if w in text)
            financial_signal = sum(1 for w in financial_words if w in text)

            reasons = []
            if energy_signal >= 2:
                ticker_recs["sectors_to_overweight"].append("energy")
                reasons.append(f"Energy signals strong ({energy_signal} keywords)")
            if defense_signal >= 2:
                ticker_recs["sectors_to_overweight"].append("industrial")
                reasons.append(f"Defense/conflict signals ({defense_signal} keywords)")
            if tech_reg_signal >= 2:
                ticker_recs["sectors_to_underweight"].append("tech")
                ticker_recs["sectors_to_overweight"].append("healthcare")
                reasons.append(f"Tech regulatory pressure ({tech_reg_signal} keywords)")
            if financial_signal >= 2:
                ticker_recs["sectors_to_underweight"].append("finance")
                ticker_recs["sectors_to_overweight"].append("utilities")
                reasons.append(f"Financial sector stress ({financial_signal} keywords)")
            if regime == "crisis":
                ticker_recs["sectors_to_underweight"].append("crypto")
                reasons.append("Crisis regime: reduce speculative exposure")

            ticker_recs["reasoning"] = "; ".join(reasons) if reasons else "No strong sector signals"

        return {
            "regime": regime,
            "volatility_outlook": vol,
            "sector_bias": {},
            "confidence": round(confidence, 2),
            "summary": (
                f"Keyword analysis: {crisis_count} crisis, "
                f"{risk_off_count} risk-off, {risk_on_count} risk-on signals."
            ),
            "ticker_recommendations": ticker_recs,
        }

    def _update_state(self, result: dict) -> None:
        """Update the regime in corporation state."""
        self.state.update_regime({
            "classification": result.get("regime", "unknown"),
            "volatility_outlook": result.get("volatility_outlook", "stable"),
            "sector_bias": result.get("sector_bias", {}),
            "confidence": result.get("confidence", 0.0),
            "summary": result.get("summary", ""),
            "ticker_recommendations": result.get("ticker_recommendations", {}),
        })
