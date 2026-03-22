"""Decision audit log — append-only JSONL log of all corp agent decisions.

Every decision (proposal, approval, veto, config change) is logged here
for the Senior Dev to review and the CEO Dashboard to display.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("corp.state.decisions")


class DecisionLog:
    """Append-only JSONL log of corp agent decisions."""

    def __init__(self, log_file: str = "logs/corporation_decisions.jsonl"):
        self._path = Path(log_file)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        agent: str,
        action: str,
        detail: dict[str, Any] | None = None,
        outcome: str = "pending",
    ) -> None:
        """Append a decision entry."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "action": action,
            "detail": detail or {},
            "outcome": outcome,
        }
        with open(self._path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        logger.debug(f"Decision logged: [{agent}] {action}")

    def get_recent(self, limit: int = 50) -> list[dict]:
        """Read the most recent N decisions."""
        if not self._path.exists():
            return []

        entries = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        return entries[-limit:]

    def get_by_agent(self, agent: str, limit: int = 20) -> list[dict]:
        """Get recent decisions by a specific agent."""
        all_entries = self.get_recent(limit=500)
        return [e for e in all_entries if e.get("agent") == agent][-limit:]

    def count_by_action(self, action: str) -> int:
        """Count how many times an action has been taken."""
        if not self._path.exists():
            return 0

        count = 0
        with open(self._path) as f:
            for line in f:
                if action in line:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("action") == action:
                            count += 1
                    except json.JSONDecodeError:
                        continue
        return count
