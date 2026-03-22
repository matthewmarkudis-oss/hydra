"""Shared corporation state — JSON-backed persistence.

Tracks agent decisions, proposals, regime assessments, and pipeline results
across runs. All corp agents read/write through this single state manager.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("corp.state")


@dataclass
class CorpMessage:
    """Typed message between corp agents."""

    sender: str
    recipient: str  # Agent name or "broadcast"
    msg_type: str  # "report" | "proposal" | "alert" | "veto" | "approval"
    priority: int  # 1-5 (5 = urgent)
    payload: dict = field(default_factory=dict)
    timestamp: str = ""
    requires_response: bool = False

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class CorporationState:
    """Shared state manager for HydraCorp.

    Persists to JSON on every write. Reads from JSON on every access
    (no in-memory caching — multiple processes may read/write).
    """

    def __init__(self, state_file: str = "logs/corporation_state.json"):
        self._path = Path(state_file)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._write_state(self._default_state())

    @staticmethod
    def _default_state() -> dict:
        return {
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat(),
            "messages": [],
            "proposals": [],
            "active_config_hash": None,
            "regime": {
                "classification": "unknown",
                "volatility_outlook": "stable",
                "sector_bias": {},
                "updated": None,
            },
            "innovation_briefs": [],
            "shadow_results": [],
            "shadow_consecutive_wins": 0,
            "pipeline_run_count": 0,
            "last_pipeline_result": None,
            "agent_performance_history": [],
            "ticker_change_history": [],
        }

    def _read_state(self) -> dict:
        try:
            with open(self._path) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning(f"Corrupt or missing state file, resetting: {self._path}")
            state = self._default_state()
            self._write_state(state)
            return state

    def _write_state(self, state: dict) -> None:
        state["updated"] = datetime.now().isoformat()
        with open(self._path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def post_message(self, msg: CorpMessage) -> None:
        """Post a message to the corp message bus."""
        state = self._read_state()
        state["messages"].append(asdict(msg))
        # Keep last 500 messages
        if len(state["messages"]) > 500:
            state["messages"] = state["messages"][-500:]
        self._write_state(state)
        logger.info(
            f"[{msg.sender} -> {msg.recipient}] {msg.msg_type}: "
            f"{list(msg.payload.keys()) if msg.payload else 'empty'}"
        )

    def get_messages(
        self,
        recipient: str | None = None,
        msg_type: str | None = None,
        since: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Retrieve messages, optionally filtered."""
        state = self._read_state()
        msgs = state["messages"]
        if recipient:
            msgs = [m for m in msgs if m["recipient"] in (recipient, "broadcast")]
        if msg_type:
            msgs = [m for m in msgs if m["msg_type"] == msg_type]
        if since:
            msgs = [m for m in msgs if m["timestamp"] > since]
        return msgs[-limit:]

    def submit_proposal(self, proposal: dict) -> None:
        """Submit a config change proposal for review."""
        state = self._read_state()
        proposal["submitted"] = datetime.now().isoformat()
        proposal["status"] = "pending"
        state["proposals"].append(proposal)
        self._write_state(state)

    def get_pending_proposals(self) -> list[dict]:
        """Get proposals that haven't been approved/vetoed."""
        state = self._read_state()
        return [p for p in state["proposals"] if p.get("status") == "pending"]

    def resolve_proposal(self, index: int, status: str, reason: str = "") -> None:
        """Approve or veto a proposal by index."""
        state = self._read_state()
        if 0 <= index < len(state["proposals"]):
            state["proposals"][index]["status"] = status
            state["proposals"][index]["resolved"] = datetime.now().isoformat()
            state["proposals"][index]["reason"] = reason
            self._write_state(state)

    def update_regime(self, regime: dict) -> None:
        """Update the current macro regime assessment."""
        state = self._read_state()
        regime["updated"] = datetime.now().isoformat()
        state["regime"] = regime
        self._write_state(state)

    def get_regime(self) -> dict:
        """Get current macro regime."""
        return self._read_state()["regime"]

    def record_pipeline_result(self, result_summary: dict) -> None:
        """Record a pipeline run result."""
        state = self._read_state()
        state["pipeline_run_count"] += 1
        state["last_pipeline_result"] = {
            **result_summary,
            "run_number": state["pipeline_run_count"],
            "timestamp": datetime.now().isoformat(),
        }
        # Keep history of performance for trending
        state["agent_performance_history"].append({
            "run": state["pipeline_run_count"],
            "timestamp": datetime.now().isoformat(),
            "best_return": result_summary.get("best_return", 0),
            "best_agent": result_summary.get("best_agent", ""),
            "passed_count": result_summary.get("passed_count", 0),
        })
        # Keep last 100 entries
        if len(state["agent_performance_history"]) > 100:
            state["agent_performance_history"] = state["agent_performance_history"][-100:]
        self._write_state(state)

    def record_shadow_result(self, primary_return: float, shadow_return: float) -> None:
        """Record a shadow trading comparison result."""
        state = self._read_state()
        shadow_won = shadow_return > primary_return
        state["shadow_results"].append({
            "timestamp": datetime.now().isoformat(),
            "primary_return": primary_return,
            "shadow_return": shadow_return,
            "shadow_won": shadow_won,
        })
        if shadow_won:
            state["shadow_consecutive_wins"] += 1
        else:
            state["shadow_consecutive_wins"] = 0
        # Keep last 50
        if len(state["shadow_results"]) > 50:
            state["shadow_results"] = state["shadow_results"][-50:]
        self._write_state(state)

    def should_promote_shadow(self, required_wins: int = 3) -> bool:
        """Check if shadow config should be promoted."""
        state = self._read_state()
        return state["shadow_consecutive_wins"] >= required_wins

    def set_active_config_hash(self, config_hash: str) -> None:
        """Set the hash of the currently active config."""
        state = self._read_state()
        state["active_config_hash"] = config_hash
        self._write_state(state)

    def get_full_state(self) -> dict:
        """Return the full state dict (for dashboard consumption)."""
        return self._read_state()

    def record_ticker_change(self, change: dict) -> None:
        """Record a ticker list change for history tracking."""
        state = self._read_state()
        change["timestamp"] = datetime.now().isoformat()
        if "ticker_change_history" not in state:
            state["ticker_change_history"] = []
        state["ticker_change_history"].append(change)
        # Keep last 50
        if len(state["ticker_change_history"]) > 50:
            state["ticker_change_history"] = state["ticker_change_history"][-50:]
        self._write_state(state)

    def auto_resolve_proposals(
        self,
        confidence_threshold: float = 0.6,
        max_risk: str = "medium",
    ) -> list[dict]:
        """Auto-approve eligible pending proposals for backtesting mode.

        Approves proposals that meet ALL criteria:
        1. confidence >= threshold (or confidence not set)
        2. risk_assessment <= max_risk or not set
        3. type is config_patch or reward_calibration (not stress_test/graduation)
        4. No existing veto

        Returns list of approved proposal patches.
        """
        safe_types = {"config_patch", "reward_calibration"}
        # Build risk set: "medium" includes "low"; "low" is just "low"
        _risk_levels = {"low": {"low"}, "medium": {"low", "medium"}}
        safe_risks = _risk_levels.get(max_risk, {max_risk})
        safe_risks = safe_risks | {None, ""}

        state = self._read_state()
        approved_patches: list[dict] = []

        for i, proposal in enumerate(state["proposals"]):
            if proposal.get("status") != "pending":
                continue

            p_type = proposal.get("type", "")
            p_conf = proposal.get("confidence", 1.0)
            p_risk = proposal.get("risk_assessment", None)

            if p_type not in safe_types:
                continue
            if p_conf < confidence_threshold:
                continue
            if p_risk and p_risk not in safe_risks:
                continue

            # Auto-approve
            proposal["status"] = "approved"
            proposal["resolved"] = datetime.now().isoformat()
            proposal["reason"] = "auto-approved (backtesting mode)"

            patch = proposal.get("patch") or proposal.get("proposed_patch", {})
            if patch:
                approved_patches.append(patch)
                logger.info(
                    f"Auto-approved proposal: {proposal.get('description', p_type)[:80]}"
                )

        if approved_patches:
            self._write_state(state)

        return approved_patches

    def add_innovation_brief(self, brief: dict) -> None:
        """Add an innovation scout brief."""
        state = self._read_state()
        brief["submitted"] = datetime.now().isoformat()
        state["innovation_briefs"].append(brief)
        if len(state["innovation_briefs"]) > 50:
            state["innovation_briefs"] = state["innovation_briefs"][-50:]
        self._write_state(state)
