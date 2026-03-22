"""Ticker Universe — sector-mapped ticker pool for intelligent selection.

Provides a curated universe of tradeable tickers organized by sector,
with liquidity tiers and selection logic driven by sector bias scores
from the GeopoliticsExpert.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TickerInfo:
    """Metadata for a single ticker."""

    symbol: str
    sector: str  # tech|energy|finance|healthcare|consumer|crypto|industrial|materials|utilities|real_estate
    liquidity_tier: int  # 1=mega-cap, 2=large-cap, 3=mid-cap
    description: str = ""


# ── Master Universe ───────────────────────────────────────────────────────────
# Every ticker the system knows about, organized by sector.

TICKER_UNIVERSE: dict[str, list[TickerInfo]] = {
    "tech": [
        TickerInfo("NVDA", "tech", 1, "NVIDIA"),
        TickerInfo("AAPL", "tech", 1, "Apple"),
        TickerInfo("MSFT", "tech", 1, "Microsoft"),
        TickerInfo("GOOGL", "tech", 1, "Alphabet"),
        TickerInfo("META", "tech", 1, "Meta Platforms"),
        TickerInfo("AMD", "tech", 1, "AMD"),
        TickerInfo("AMZN", "tech", 1, "Amazon"),
        TickerInfo("CRM", "tech", 1, "Salesforce"),
        TickerInfo("ORCL", "tech", 1, "Oracle"),
        TickerInfo("INTC", "tech", 2, "Intel"),
        TickerInfo("NFLX", "tech", 1, "Netflix"),
        TickerInfo("ADBE", "tech", 1, "Adobe"),
        TickerInfo("AVGO", "tech", 1, "Broadcom"),
        TickerInfo("QCOM", "tech", 1, "Qualcomm"),
        TickerInfo("CSCO", "tech", 1, "Cisco"),
    ],
    "energy": [
        TickerInfo("XOM", "energy", 1, "Exxon Mobil"),
        TickerInfo("CVX", "energy", 1, "Chevron"),
        TickerInfo("COP", "energy", 1, "ConocoPhillips"),
        TickerInfo("SLB", "energy", 1, "Schlumberger"),
        TickerInfo("EOG", "energy", 1, "EOG Resources"),
        TickerInfo("OXY", "energy", 2, "Occidental"),
        TickerInfo("VLO", "energy", 2, "Valero"),
        TickerInfo("MPC", "energy", 2, "Marathon Petroleum"),
        TickerInfo("PSX", "energy", 2, "Phillips 66"),
        TickerInfo("HAL", "energy", 2, "Halliburton"),
    ],
    "finance": [
        TickerInfo("JPM", "finance", 1, "JPMorgan"),
        TickerInfo("BAC", "finance", 1, "Bank of America"),
        TickerInfo("GS", "finance", 1, "Goldman Sachs"),
        TickerInfo("MS", "finance", 1, "Morgan Stanley"),
        TickerInfo("WFC", "finance", 1, "Wells Fargo"),
        TickerInfo("V", "finance", 1, "Visa"),
        TickerInfo("MA", "finance", 1, "Mastercard"),
        TickerInfo("BLK", "finance", 1, "BlackRock"),
        TickerInfo("SCHW", "finance", 1, "Schwab"),
        TickerInfo("SQ", "finance", 2, "Block Inc"),
    ],
    "healthcare": [
        TickerInfo("JNJ", "healthcare", 1, "Johnson & Johnson"),
        TickerInfo("UNH", "healthcare", 1, "UnitedHealth"),
        TickerInfo("PFE", "healthcare", 1, "Pfizer"),
        TickerInfo("ABBV", "healthcare", 1, "AbbVie"),
        TickerInfo("MRK", "healthcare", 1, "Merck"),
        TickerInfo("LLY", "healthcare", 1, "Eli Lilly"),
        TickerInfo("TMO", "healthcare", 1, "Thermo Fisher"),
        TickerInfo("ABT", "healthcare", 1, "Abbott"),
        TickerInfo("BMY", "healthcare", 1, "Bristol-Myers"),
        TickerInfo("AMGN", "healthcare", 1, "Amgen"),
    ],
    "consumer": [
        TickerInfo("TSLA", "consumer", 1, "Tesla"),
        TickerInfo("WMT", "consumer", 1, "Walmart"),
        TickerInfo("HD", "consumer", 1, "Home Depot"),
        TickerInfo("COST", "consumer", 1, "Costco"),
        TickerInfo("NKE", "consumer", 1, "Nike"),
        TickerInfo("MCD", "consumer", 1, "McDonald's"),
        TickerInfo("SBUX", "consumer", 1, "Starbucks"),
        TickerInfo("TGT", "consumer", 2, "Target"),
        TickerInfo("LOW", "consumer", 1, "Lowe's"),
        TickerInfo("DIS", "consumer", 1, "Disney"),
    ],
    "crypto": [
        TickerInfo("COIN", "crypto", 2, "Coinbase"),
        TickerInfo("MARA", "crypto", 3, "Marathon Digital"),
        TickerInfo("RIOT", "crypto", 3, "Riot Platforms"),
        TickerInfo("MSTR", "crypto", 2, "MicroStrategy"),
        TickerInfo("CLSK", "crypto", 3, "CleanSpark"),
    ],
    "industrial": [
        TickerInfo("CAT", "industrial", 1, "Caterpillar"),
        TickerInfo("GE", "industrial", 1, "GE Aerospace"),
        TickerInfo("HON", "industrial", 1, "Honeywell"),
        TickerInfo("UPS", "industrial", 1, "UPS"),
        TickerInfo("RTX", "industrial", 1, "Raytheon"),
        TickerInfo("BA", "industrial", 1, "Boeing"),
        TickerInfo("LMT", "industrial", 1, "Lockheed Martin"),
        TickerInfo("DE", "industrial", 1, "Deere & Co"),
    ],
    "materials": [
        TickerInfo("LIN", "materials", 1, "Linde"),
        TickerInfo("APD", "materials", 1, "Air Products"),
        TickerInfo("SHW", "materials", 1, "Sherwin-Williams"),
        TickerInfo("FCX", "materials", 1, "Freeport-McMoRan"),
        TickerInfo("NEM", "materials", 1, "Newmont"),
    ],
    "utilities": [
        TickerInfo("NEE", "utilities", 1, "NextEra Energy"),
        TickerInfo("DUK", "utilities", 1, "Duke Energy"),
        TickerInfo("SO", "utilities", 1, "Southern Company"),
        TickerInfo("D", "utilities", 1, "Dominion Energy"),
    ],
    "real_estate": [
        TickerInfo("PLD", "real_estate", 1, "Prologis"),
        TickerInfo("AMT", "real_estate", 1, "American Tower"),
        TickerInfo("SPG", "real_estate", 1, "Simon Property"),
        TickerInfo("EQIX", "real_estate", 1, "Equinix"),
    ],
}

# Reverse lookup: symbol -> sector
TICKER_TO_SECTOR: dict[str, str] = {}
for _sector, _tickers in TICKER_UNIVERSE.items():
    for _t in _tickers:
        TICKER_TO_SECTOR[_t.symbol] = _sector

# Sector name mapping from geopolitics output to universe keys
GEOPOLITICS_SECTOR_MAP = {
    "tech": "tech",
    "energy": "energy",
    "finance": "finance",
    "healthcare": "healthcare",
    "consumer": "consumer",
}


# ── Ticker Selector ──────────────────────────────────────────────────────────


class TickerSelector:
    """Selects optimal ticker lists based on sector bias and constraints.

    Pure-function class with no state — takes inputs, returns a recommended
    ticker list. Used by HedgeFundDirector for ticker proposals.
    """

    TIER_TARGETS = {
        10: {"per_sector_min": 1, "per_sector_max": 3, "liquidity_tier_max": 2},
        20: {"per_sector_min": 1, "per_sector_max": 5, "liquidity_tier_max": 2},
        50: {"per_sector_min": 3, "per_sector_max": 10, "liquidity_tier_max": 3},
        100: {"per_sector_min": 5, "per_sector_max": 20, "liquidity_tier_max": 3},
    }

    @classmethod
    def recommend_tier(
        cls,
        regime: str,
        volatility_outlook: str,
        sector_bias: dict[str, float],
        current_ticker_count: int,
    ) -> int:
        """Recommend optimal ticker count tier based on macro conditions.

        Returns one of: 10, 20, 50, 100.
        """
        if regime == "crisis":
            return 10

        if regime == "risk_off" and volatility_outlook in ("elevated", "extreme"):
            return 10

        active_sectors = sum(1 for v in sector_bias.values() if abs(v) > 0.2)

        if active_sectors >= 4:
            return 50 if regime == "risk_on" else 20

        if active_sectors >= 2:
            return 20

        return min(max(current_ticker_count, 10), 20)

    @classmethod
    def select_tickers(
        cls,
        target_count: int,
        sector_bias: dict[str, float],
        current_tickers: list[str],
        regime: str = "risk_on",
    ) -> list[str]:
        """Select tickers from the universe based on sector bias.

        Algorithm:
        1. Map geopolitics sector names to universe sector names
        2. Compute per-sector allocation proportional to (1 + bias)
        3. Fill from highest liquidity tier first
        4. Prefer keeping current tickers to minimize churn
        """
        all_sectors = list(TICKER_UNIVERSE.keys())
        sector_weights: dict[str, float] = {}
        for sector in all_sectors:
            geo_key = next(
                (k for k, v in GEOPOLITICS_SECTOR_MAP.items() if v == sector),
                None,
            )
            bias = sector_bias.get(geo_key, 0.0) if geo_key else 0.0
            sector_weights[sector] = max(0.1, min(2.0, 1.0 + bias))

        # In crisis, zero out crypto and boost defensive sectors
        if regime == "crisis":
            sector_weights["crypto"] = 0.0
            for s in ("healthcare", "utilities", "consumer"):
                sector_weights[s] = sector_weights.get(s, 1.0) * 1.5

        # Find the closest tier config
        tier_config = cls.TIER_TARGETS.get(
            target_count,
            cls.TIER_TARGETS[
                min(cls.TIER_TARGETS.keys(), key=lambda k: abs(k - target_count))
            ],
        )

        # Compute per-sector allocations
        total_weight = sum(sector_weights.values())
        allocations: dict[str, int] = {}
        for sector, weight in sector_weights.items():
            raw_alloc = (weight / total_weight) * target_count
            allocations[sector] = max(
                tier_config["per_sector_min"],
                min(tier_config["per_sector_max"], round(raw_alloc)),
            )

        # Adjust to hit target_count exactly
        current_total = sum(allocations.values())
        diff = target_count - current_total
        if diff != 0:
            sorted_sectors = sorted(
                allocations.keys(),
                key=lambda s: sector_weights[s],
                reverse=(diff > 0),
            )
            for sector in sorted_sectors:
                if diff == 0:
                    break
                if diff > 0 and allocations[sector] < tier_config["per_sector_max"]:
                    add = min(diff, tier_config["per_sector_max"] - allocations[sector])
                    allocations[sector] += add
                    diff -= add
                elif diff < 0 and allocations[sector] > tier_config["per_sector_min"]:
                    remove = min(
                        -diff, allocations[sector] - tier_config["per_sector_min"]
                    )
                    allocations[sector] -= remove
                    diff += remove

        # Fill tickers per sector, preferring current tickers
        current_set = set(current_tickers)
        selected: list[str] = []
        for sector, count in allocations.items():
            if count <= 0:
                continue
            candidates = [
                t
                for t in TICKER_UNIVERSE.get(sector, [])
                if t.liquidity_tier <= tier_config["liquidity_tier_max"]
            ]
            # Sort: current tickers first, then by liquidity tier
            candidates.sort(
                key=lambda t: (0 if t.symbol in current_set else 1, t.liquidity_tier)
            )
            selected.extend(c.symbol for c in candidates[:count])

        return selected

    @classmethod
    def compute_churn(
        cls, current: list[str], proposed: list[str]
    ) -> dict[str, Any]:
        """Compute churn metrics between current and proposed ticker lists."""
        current_set = set(current)
        proposed_set = set(proposed)
        added = proposed_set - current_set
        removed = current_set - proposed_set
        retained = current_set & proposed_set
        total = max(len(current_set | proposed_set), 1)
        return {
            "added": sorted(added),
            "removed": sorted(removed),
            "retained": sorted(retained),
            "churn_pct": len(added | removed) / total,
            "size_change": len(proposed) - len(current),
            "retraining_required": len(proposed) != len(current) or len(added) > 0,
        }

    @classmethod
    def get_sector_distribution(cls, tickers: list[str]) -> dict[str, int]:
        """Get sector distribution for a list of tickers."""
        dist: dict[str, int] = {}
        for t in tickers:
            s = TICKER_TO_SECTOR.get(t, "unknown")
            dist[s] = dist.get(s, 0) + 1
        return dict(sorted(dist.items()))
