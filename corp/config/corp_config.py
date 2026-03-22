"""Corporation configuration schema."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = Field(default="anthropic", description="anthropic | ollama")
    routine_model: str = Field(default="claude-3-haiku-20240307")
    strategic_model: str = Field(default="claude-sonnet-4-20250514")
    api_key_env: str = Field(default="ANTHROPIC_API_KEY")
    max_tokens_per_call: int = Field(default=2000, ge=100)
    monthly_budget_usd: float = Field(default=10.0, gt=0)
    temperature: float = Field(default=0.3, ge=0, le=1)


class ScheduleConfig(BaseModel):
    """Agent run schedules."""

    geopolitics_interval_hours: int = Field(default=24, ge=1)
    innovation_scout_interval_hours: int = Field(default=168, ge=1)  # weekly
    shadow_trader_enabled: bool = Field(default=True)
    contrarian_trigger_fitness: float = Field(
        default=0.5, description="Trigger contrarian when best fitness exceeds this"
    )
    shadow_promote_after_wins: int = Field(
        default=3, ge=1, description="Promote shadow config after N consecutive wins"
    )
    auto_approve_enabled: bool = Field(
        default=True, description="Auto-approve eligible proposals in backtesting mode"
    )
    auto_approve_confidence: float = Field(
        default=0.6, ge=0, le=1, description="Min confidence for auto-approval"
    )
    auto_approve_max_risk: str = Field(
        default="medium", description="Max risk level for auto-approval (low | medium)"
    )


class CorporationConfig(BaseModel):
    """Root configuration for HydraCorp."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    starting_capital_cad: float = Field(default=2500.0, gt=0)
    state_file: str = Field(default="logs/corporation_state.json")
    decision_log_file: str = Field(default="logs/corporation_decisions.jsonl")
    dashboard_port: int = Field(default=5050)
