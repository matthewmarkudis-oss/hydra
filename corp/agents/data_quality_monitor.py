"""Data Quality Monitor — pre-flight validation of all data feeds.

Checks factor data, news APIs, corp state freshness, and checkpoint
integrity before training starts. Blocks pipeline if critical feeds
are broken.

Zero LLM cost — pure Python validation.

Backtesting and training only.
"""

from __future__ import annotations

import logging
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from corp.agents.base_corp_agent import BaseCorpAgent
from corp.state.corporation_state import CorporationState
from corp.state.decision_log import DecisionLog

logger = logging.getLogger("corp.agents.data_quality_monitor")

# News feed endpoints to check reachability (HEAD requests only)
_NEWS_ENDPOINTS = [
    ("NewsAPI", "https://newsapi.org/v2/top-headlines?country=us&pageSize=1"),
    ("Finnhub", "https://finnhub.io/api/v1/news?category=general"),
    ("Yahoo RSS", "https://finance.yahoo.com/news/rssindex"),
]

_CHECK_TIMEOUT = 10  # seconds per HTTP check


class DataQualityMonitor(BaseCorpAgent):
    """Pre-flight data feed validator.

    Runs before the pipeline to verify that all data sources are
    reachable and returning fresh data. Prevents wasted training
    time on stale or broken feeds.

    Checks:
    1. Factor data (FF5, FH7) — downloadable and recent
    2. News feeds — reachable (HTTP HEAD)
    3. Corp state — exists and regime is not stale
    4. Checkpoint integrity — latest.json valid if warm-starting
    """

    def __init__(
        self,
        state: CorporationState,
        decision_log: DecisionLog,
        checkpoint_dir: str = "checkpoints",
    ):
        super().__init__("data_quality_monitor", state, decision_log)
        self.checkpoint_dir = checkpoint_dir

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Run all data quality checks.

        Returns:
            Dict with checks list, overall status, and failure counts.
        """
        checks: list[dict] = []

        checks.append(self._check_factor_data())
        checks.append(self._check_news_feeds())
        checks.append(self._check_corp_state_freshness())
        checks.append(self._check_checkpoint_integrity())

        critical_failures = sum(1 for c in checks if c["status"] == "error")
        warnings = sum(1 for c in checks if c["status"] == "warning")
        all_passed = critical_failures == 0

        result = {
            "checks": checks,
            "all_passed": all_passed,
            "critical_failures": critical_failures,
            "warnings": warnings,
        }

        # Alert Chief of Staff on critical failures
        if critical_failures > 0:
            failed_names = [c["name"] for c in checks if c["status"] == "error"]
            self.send_message(
                recipient="chief_of_staff",
                msg_type="alert",
                payload={
                    "severity": "critical",
                    "message": f"Data quality check failed: {', '.join(failed_names)}",
                    "checks": checks,
                },
                priority=5,
            )

        self.log_decision(
            "data_quality_check",
            detail={
                "critical_failures": critical_failures,
                "warnings": warnings,
                "all_passed": all_passed,
                "checks": [{"name": c["name"], "status": c["status"]} for c in checks],
            },
            outcome="passed" if all_passed else "failed",
        )

        self._mark_run(result)
        return result

    def _check_factor_data(self) -> dict:
        """Check that academic factor data is downloadable and recent."""
        try:
            from hydra.distillation.factor_data import FactorDataStore

            store = FactorDataStore()

            # Check FF5
            ff5 = store.get_fama_french_5()
            if ff5 is None or ff5.empty:
                return {
                    "name": "factor_data",
                    "status": "error",
                    "message": "Fama-French 5-factor data is empty or unavailable",
                    "details": {},
                }

            latest_date = ff5.index.max()
            age_days = (datetime.now() - latest_date.to_pydatetime().replace(tzinfo=None)).days

            if age_days > 90:
                return {
                    "name": "factor_data",
                    "status": "warning",
                    "message": f"FF5 data is {age_days} days old (latest: {latest_date.date()})",
                    "details": {"latest_date": str(latest_date.date()), "age_days": age_days},
                }

            return {
                "name": "factor_data",
                "status": "ok",
                "message": f"FF5: {len(ff5)} rows, latest {latest_date.date()} ({age_days}d old)",
                "details": {"rows": len(ff5), "latest_date": str(latest_date.date())},
            }

        except Exception as e:
            return {
                "name": "factor_data",
                "status": "warning",
                "message": f"Factor data check failed: {e}",
                "details": {"error": str(e)},
            }

    def _check_news_feeds(self) -> dict:
        """Check that at least one news feed is reachable."""
        reachable = 0
        failed = []

        for name, url in _NEWS_ENDPOINTS:
            try:
                req = urllib.request.Request(url, method="HEAD")
                req.add_header("User-Agent", "HydraCorp/1.0 DataQualityMonitor")
                with urllib.request.urlopen(req, timeout=_CHECK_TIMEOUT):
                    reachable += 1
            except Exception:
                failed.append(name)

        total = len(_NEWS_ENDPOINTS)
        if reachable == 0:
            return {
                "name": "news_feeds",
                "status": "warning",
                "message": f"No news feeds reachable (0/{total}). Geopolitics will use fallback.",
                "details": {"reachable": 0, "failed": failed},
            }

        return {
            "name": "news_feeds",
            "status": "ok",
            "message": f"{reachable}/{total} news feeds reachable",
            "details": {"reachable": reachable, "failed": failed},
        }

    def _check_corp_state_freshness(self) -> dict:
        """Check that corporation state exists and regime is not stale."""
        try:
            regime = self.state.get_regime()
            updated = regime.get("updated")
            classification = regime.get("classification", "unknown")

            if updated is None or classification == "unknown":
                return {
                    "name": "corp_state",
                    "status": "warning",
                    "message": "No regime classification yet — Geopolitics Expert hasn't run",
                    "details": {"classification": classification},
                }

            # Check staleness (>48h = warning)
            try:
                updated_dt = datetime.fromisoformat(updated)
                age = datetime.now() - updated_dt
                if age > timedelta(hours=48):
                    return {
                        "name": "corp_state",
                        "status": "warning",
                        "message": f"Regime data is {age.total_seconds() / 3600:.0f}h old ({classification})",
                        "details": {"classification": classification, "age_hours": age.total_seconds() / 3600},
                    }
            except (ValueError, TypeError):
                pass

            return {
                "name": "corp_state",
                "status": "ok",
                "message": f"Regime: {classification}, updated: {updated}",
                "details": {"classification": classification, "updated": updated},
            }

        except Exception as e:
            return {
                "name": "corp_state",
                "status": "warning",
                "message": f"Corp state check failed: {e}",
                "details": {"error": str(e)},
            }

    def _check_checkpoint_integrity(self) -> dict:
        """Check that warm-start checkpoint is valid if present."""
        ckpt_root = Path(self.checkpoint_dir)
        latest_file = ckpt_root / "latest.json"

        if not latest_file.exists():
            return {
                "name": "checkpoint",
                "status": "ok",
                "message": "No previous checkpoint — will start fresh",
                "details": {"warm_start": False},
            }

        try:
            import json
            with open(latest_file) as f:
                pointer = json.load(f)

            ckpt_path = Path(pointer.get("checkpoint_path", ""))
            if not ckpt_path.exists():
                return {
                    "name": "checkpoint",
                    "status": "warning",
                    "message": f"Checkpoint path missing: {ckpt_path}",
                    "details": {"warm_start": True, "path": str(ckpt_path), "exists": False},
                }

            metadata_file = ckpt_path / "pool_metadata.json"
            if not metadata_file.exists():
                return {
                    "name": "checkpoint",
                    "status": "warning",
                    "message": f"Checkpoint missing pool_metadata.json: {ckpt_path}",
                    "details": {"warm_start": True, "path": str(ckpt_path), "metadata": False},
                }

            return {
                "name": "checkpoint",
                "status": "ok",
                "message": f"Valid checkpoint from gen {pointer.get('generation', '?')}",
                "details": {
                    "warm_start": True,
                    "path": str(ckpt_path),
                    "generation": pointer.get("generation"),
                },
            }

        except Exception as e:
            return {
                "name": "checkpoint",
                "status": "warning",
                "message": f"Checkpoint check failed: {e}",
                "details": {"error": str(e)},
            }
