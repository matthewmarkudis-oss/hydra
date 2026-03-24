"""Forward-test configuration schema."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ForwardTestConfig(BaseModel):
    """Configuration for the forward-testing graduation pipeline."""

    enabled: bool = Field(
        default=False,
        description="Must be explicitly enabled by CEO. Disabled by default.",
    )
    duration_days: int = Field(
        default=60,
        ge=1,
        le=90,
        description="Number of trading days to run forward test.",
    )
    max_agents: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum simultaneous agents in forward test.",
    )
    initial_capital: float = Field(
        default=10000.0,
        gt=0,
        description="Sandbox starting capital (USD).",
    )
    max_position_pct: float = Field(
        default=0.35,
        gt=0,
        le=0.50,
        description="Position limit for forward testing (matches training regime).",
    )
    sharpe_retention_min: float = Field(
        default=0.50,
        ge=0,
        le=1.0,
        description="Minimum fraction of backtest Sharpe to retain for GRADUATE verdict.",
    )
    drawdown_tolerance: float = Field(
        default=1.5,
        ge=1.0,
        description="Max drawdown as multiple of backtest DD.",
    )
    win_rate_tolerance: float = Field(
        default=0.80,
        ge=0,
        le=1.0,
        description="Min win rate as fraction of backtest win rate.",
    )
    poll_interval_minutes: int = Field(
        default=5,
        ge=1,
        description="Bar polling interval in minutes.",
    )
    alert_webhook_url: str = Field(
        default="",
        description="Discord/Slack webhook URL for forward-test alerts. Empty = disabled.",
    )
    alert_daily_loss_pct: float = Field(
        default=0.03,
        ge=0,
        le=1.0,
        description="Daily loss threshold that triggers an alert.",
    )
    allocation_method: str = Field(
        default="sharpe_weighted",
        description="Capital allocation method: sharpe_weighted | equal.",
    )
    min_allocation_pct: float = Field(
        default=0.05,
        ge=0,
        le=1.0,
        description="Minimum allocation percentage. Agents below this get zero.",
    )
    route_to_broker: bool = Field(
        default=False,
        description="Also route orders through real broker for paper trading.",
    )
