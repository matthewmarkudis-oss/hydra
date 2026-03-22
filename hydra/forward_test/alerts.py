"""Forward-test alert system — webhook + corp bus notifications.

Monitors forward-test metrics at end of each trading day and sends
alerts when thresholds are breached. Supports Discord/Slack webhooks
and the internal corp state message bus.

Backtesting and training research only.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger("hydra.forward_test.alerts")


class ForwardTestAlertManager:
    """Sends alerts when forward-test metrics breach thresholds.

    Integrates with:
    - External webhooks (Discord, Slack, or generic HTTP POST)
    - Corp state message bus (CorporationState.post_message)

    Each alert type has a 1-per-day cooldown to avoid spamming.
    """

    def __init__(
        self,
        webhook_url: str = "",
        corp_state=None,
    ):
        self._webhook_url = webhook_url
        self._corp_state = corp_state
        # Track which alerts fired today to enforce cooldown
        # {(agent, alert_type, date): True}
        self._cooldowns: dict[tuple[str, str, str], bool] = {}

    def check_and_alert(
        self,
        agent_name: str,
        metrics: dict[str, Any],
        thresholds: dict[str, float],
    ) -> list[dict]:
        """Check metrics against thresholds and send alerts if breached.

        Args:
            agent_name: Name of the agent being monitored.
            metrics: Current forward-test metrics (from tracker.get_metrics).
            thresholds: Alert thresholds:
                - daily_loss_pct: max acceptable daily loss (default 0.03)
                - max_drawdown_pct: max acceptable drawdown (default 0.10)

        Returns:
            List of alert dicts that were fired.
        """
        if "error" in metrics:
            return []

        today = datetime.now().strftime("%Y-%m-%d")
        alerts_fired = []

        # Check daily loss
        daily_loss_threshold = thresholds.get("daily_loss_pct", 0.03)
        total_return = metrics.get("total_return", 0)
        if total_return < -daily_loss_threshold:
            alert = self._fire_alert(
                agent_name, "daily_loss", today,
                f"Daily loss alert: {agent_name} return is {total_return:+.2%} "
                f"(threshold: -{daily_loss_threshold:.0%})",
                priority=5,
            )
            if alert:
                alerts_fired.append(alert)

        # Check drawdown
        dd_threshold = thresholds.get("max_drawdown_pct", 0.10)
        max_dd = metrics.get("max_drawdown", 0)
        if max_dd > dd_threshold:
            alert = self._fire_alert(
                agent_name, "drawdown", today,
                f"Drawdown alert: {agent_name} drawdown is {max_dd:.2%} "
                f"(threshold: {dd_threshold:.0%})",
                priority=4,
            )
            if alert:
                alerts_fired.append(alert)

        # Check Sharpe going negative
        sharpe = metrics.get("sharpe", 0)
        if sharpe < 0 and metrics.get("trading_days", 0) >= 5:
            alert = self._fire_alert(
                agent_name, "negative_sharpe", today,
                f"Sharpe alert: {agent_name} Sharpe ratio is {sharpe:.2f} "
                f"(negative after {metrics.get('trading_days', 0)} days)",
                priority=3,
            )
            if alert:
                alerts_fired.append(alert)

        return alerts_fired

    def _fire_alert(
        self,
        agent_name: str,
        alert_type: str,
        date: str,
        message: str,
        priority: int = 3,
    ) -> dict | None:
        """Fire an alert if not already fired today (cooldown)."""
        key = (agent_name, alert_type, date)
        if key in self._cooldowns:
            return None

        self._cooldowns[key] = True
        logger.warning("ALERT [%s] %s: %s", alert_type, agent_name, message)

        alert = {
            "agent": agent_name,
            "type": alert_type,
            "message": message,
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
        }

        # Send to webhook
        if self._webhook_url:
            self._send_webhook(alert)

        # Send to corp state
        if self._corp_state:
            self._send_corp_alert(message, priority)

        return alert

    def _send_webhook(self, alert: dict) -> None:
        """Send alert to configured webhook URL (Discord/Slack/generic)."""
        try:
            import urllib.request
            import json

            url = self._webhook_url
            # Auto-detect Discord vs Slack vs generic
            if "discord" in url:
                payload = {
                    "embeds": [{
                        "title": f"Forward Test Alert: {alert['type']}",
                        "description": alert["message"],
                        "color": 0xFF0000 if alert["priority"] >= 4 else 0xFFAA00,
                        "timestamp": alert["timestamp"],
                    }],
                }
            elif "slack" in url:
                payload = {
                    "text": f"*{alert['type'].upper()}*: {alert['message']}",
                }
            else:
                payload = alert

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
            logger.info("Webhook alert sent to %s", url[:50])
        except Exception as e:
            logger.debug("Webhook send failed: %s", e)

    def _send_corp_alert(self, message: str, priority: int) -> None:
        """Post alert to the corp state message bus."""
        try:
            from corp.state.corporation_state import CorpMessage
            self._corp_state.post_message(CorpMessage(
                sender="forward_test",
                recipient="chief_of_staff",
                msg_type="alert",
                priority=priority,
                payload={"message": message},
            ))
        except Exception as e:
            logger.debug("Corp alert failed: %s", e)
