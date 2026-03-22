"""Abstract base class for all corporation agents."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from corp.state.corporation_state import CorporationState, CorpMessage
from corp.state.decision_log import DecisionLog


class BaseCorpAgent(ABC):
    """Base class for HydraCorp agents.

    Every corp agent has:
    - A name (used in messages and logs)
    - Access to shared state and decision log
    - A run() method that executes its primary function
    - A report() method that summarizes its last run for the dashboard
    """

    def __init__(
        self,
        name: str,
        state: CorporationState,
        decision_log: DecisionLog,
    ):
        self.name = name
        self.state = state
        self.decision_log = decision_log
        self.logger = logging.getLogger(f"corp.agents.{name}")
        self._last_run: str | None = None
        self._last_result: dict[str, Any] = {}

    @abstractmethod
    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent's primary function.

        Args:
            context: Dict containing pipeline results, config, and any
                upstream agent outputs needed for this agent's work.

        Returns:
            Dict with the agent's output (recommendations, alerts, etc.).
        """

    def report(self) -> dict[str, Any]:
        """Summarize the last run for the CEO dashboard."""
        return {
            "agent": self.name,
            "last_run": self._last_run,
            "result": self._last_result,
        }

    def send_message(
        self,
        recipient: str,
        msg_type: str,
        payload: dict,
        priority: int = 3,
        requires_response: bool = False,
    ) -> None:
        """Send a message through the corp message bus."""
        msg = CorpMessage(
            sender=self.name,
            recipient=recipient,
            msg_type=msg_type,
            priority=priority,
            payload=payload,
            requires_response=requires_response,
        )
        self.state.post_message(msg)

    def log_decision(
        self,
        action: str,
        detail: dict[str, Any] | None = None,
        outcome: str = "pending",
    ) -> None:
        """Log a decision to the audit trail."""
        self.decision_log.log(
            agent=self.name,
            action=action,
            detail=detail,
            outcome=outcome,
        )

    def _mark_run(self, result: dict[str, Any]) -> None:
        """Record that a run completed."""
        self._last_run = datetime.now().isoformat()
        self._last_result = result
