"""Thesis Library — curated strategic intelligence from macro thinkers.

Loads YAML thesis entries and cross-references them against news headlines
to produce agreement matrices, thesis confirmations, and sector signals.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("corp.data.thesis_library")


class ThesisLibrary:
    """Loads and queries the curated thesis library."""

    def __init__(self, theses_dir: str | Path | None = None):
        if theses_dir is None:
            theses_dir = Path(__file__).parent / "theses"
        self._dir = Path(theses_dir)
        self._theses: list[dict] = []
        self._load()

    def _load(self) -> None:
        """Load all YAML thesis files from the theses directory."""
        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML not installed — thesis library disabled")
            return

        if not self._dir.exists():
            logger.warning(f"Theses directory not found: {self._dir}")
            return

        for path in sorted(self._dir.glob("*.yaml")):
            try:
                with open(path, encoding="utf-8") as f:
                    entries = yaml.safe_load(f)
                if isinstance(entries, list):
                    self._theses.extend(entries)
                    logger.info(f"Loaded {len(entries)} theses from {path.name}")
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

    @property
    def count(self) -> int:
        return len(self._theses)

    @property
    def theses(self) -> list[dict]:
        return list(self._theses)

    def get_by_category(self, category: str) -> list[dict]:
        return [t for t in self._theses if t.get("category") == category]

    def get_by_thinker(self, thinker: str) -> list[dict]:
        thinker_lower = thinker.lower()
        return [t for t in self._theses if thinker_lower in t.get("thinker", "").lower()]

    def cross_reference(self, headlines: list[dict]) -> dict[str, Any]:
        """Cross-reference headlines against standing theses.

        Returns a dict with:
        - confirmations: list of {thesis_id, thinker, headline, keywords_matched}
        - agreement_matrix: {category: {thinkers: [...], agreement_score}}
        - sector_signals: {sector: score} aggregated from confirmed theses
        - confidence_adjustment: float (-0.3 to +0.3) based on agreement/disagreement
        """
        if not self._theses:
            return {
                "confirmations": [],
                "agreement_matrix": {},
                "sector_signals": {},
                "confidence_adjustment": 0.0,
            }

        headline_text = " ".join(h.get("title", "").lower() for h in headlines)

        confirmations = []
        confirmed_ids = set()

        for thesis in self._theses:
            keywords = thesis.get("keywords", [])
            if not keywords:
                continue

            matched = [kw for kw in keywords if kw.lower() in headline_text]
            # Require at least 2 keyword matches for a confirmation
            if len(matched) >= 2:
                confirmations.append({
                    "thesis_id": thesis["id"],
                    "thinker": thesis["thinker"],
                    "category": thesis.get("category", ""),
                    "keywords_matched": matched,
                    "match_strength": len(matched) / len(keywords),
                    "sector_implications": thesis.get("sector_implications", {}),
                })
                confirmed_ids.add(thesis["id"])

        # Build agreement matrix by category
        agreement_matrix = self._build_agreement_matrix(confirmations)

        # Aggregate sector signals from confirmed theses
        sector_signals = self._aggregate_sector_signals(confirmations)

        # Compute confidence adjustment from agreement/disagreement
        confidence_adj = self._compute_confidence_adjustment(confirmations)

        return {
            "confirmations": confirmations,
            "agreement_matrix": agreement_matrix,
            "sector_signals": sector_signals,
            "confidence_adjustment": confidence_adj,
        }

    def _build_agreement_matrix(self, confirmations: list[dict]) -> dict[str, Any]:
        """Group confirmed theses by category and check for agreement."""
        by_category: dict[str, list[str]] = {}
        for c in confirmations:
            cat = c.get("category", "other")
            by_category.setdefault(cat, []).append(c["thinker"])

        matrix = {}
        for cat, thinkers in by_category.items():
            unique = list(set(thinkers))
            matrix[cat] = {
                "thinkers": unique,
                "count": len(unique),
                "agreement_score": min(len(unique) / 3.0, 1.0),  # 3+ thinkers = full agreement
            }
        return matrix

    def _aggregate_sector_signals(self, confirmations: list[dict]) -> dict[str, float]:
        """Aggregate overweight/underweight signals across confirmed theses."""
        scores: dict[str, float] = {}
        for c in confirmations:
            impl = c.get("sector_implications", {})
            strength = c.get("match_strength", 0.5)
            for sector in impl.get("overweight", []):
                scores[sector] = scores.get(sector, 0) + strength
            for sector in impl.get("underweight", []):
                scores[sector] = scores.get(sector, 0) - strength
        return scores

    def _compute_confidence_adjustment(self, confirmations: list[dict]) -> float:
        """Compute regime confidence adjustment based on thesis agreement.

        More confirmed theses = higher confidence.
        Disagreement among confirmed thinkers = lower confidence.
        """
        if not confirmations:
            return 0.0

        # Base boost from number of confirmations
        n = len(confirmations)
        boost = min(n * 0.05, 0.2)  # Up to +0.2 from confirmations

        # Check for disagreements among confirmed thinkers
        confirmed_thinkers = {c["thinker"] for c in confirmations}
        disagree_count = 0
        for thesis in self._theses:
            if thesis["id"] in {c["thesis_id"] for c in confirmations}:
                for other in thesis.get("disagrees_with", []):
                    if any(other.lower() in t.lower() for t in confirmed_thinkers):
                        disagree_count += 1

        # Disagreement penalizes confidence
        penalty = min(disagree_count * 0.1, 0.3)

        return round(boost - penalty, 3)

    def get_summary(self) -> dict[str, Any]:
        """Summary for dashboard display."""
        categories = {}
        for t in self._theses:
            cat = t.get("category", "other")
            categories.setdefault(cat, []).append(t["thinker"])

        return {
            "total_theses": self.count,
            "categories": {k: list(set(v)) for k, v in categories.items()},
            "thinkers": list(set(t["thinker"] for t in self._theses)),
        }
