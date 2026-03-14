"""Pydantic configuration schema for Hydra system."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class EnvConfig(BaseModel):
    """Trading environment configuration."""

    num_stocks: int = Field(default=10, ge=1, le=500)
    episode_bars: int = Field(default=78, description="Bars per episode (5-min, 9:30-16:00)")
    bar_interval_minutes: int = Field(default=5)
    initial_cash: float = Field(default=100_000.0, gt=0)
    transaction_cost_bps: float = Field(default=5.0, ge=0, description="Transaction cost in basis points")
    slippage_bps: float = Field(default=2.0, ge=0, description="Slippage in basis points")
    spread_bps: float = Field(default=1.0, ge=0, description="Bid-ask spread in basis points")
    max_position_pct: float = Field(default=0.10, gt=0, le=1.0)
    max_drawdown_pct: float = Field(default=0.10, gt=0, le=1.0)
    max_daily_loss_pct: float = Field(default=0.03, gt=0, le=1.0)
    normalize_obs: bool = Field(default=True)


class RewardConfig(BaseModel):
    """Reward function configuration."""

    sharpe_window: int = Field(default=20, ge=2)
    sharpe_eta: float = Field(default=0.05, gt=0, description="Differential Sharpe EMA decay")
    drawdown_penalty: float = Field(default=2.0, ge=0)
    transaction_penalty: float = Field(default=0.5, ge=0)
    holding_penalty: float = Field(default=0.0, ge=0, description="Penalty for large positions")
    reward_scale: float = Field(default=100.0, gt=0, description="Multiplier for reward signal magnitude")


class AgentConfig(BaseModel):
    """Individual agent configuration."""

    agent_type: str = Field(description="One of: ppo, sac, a2c, static, rule_based")
    learning_rate: float = Field(default=3e-4, gt=0)
    batch_size: int = Field(default=64, ge=1)
    gamma: float = Field(default=0.99, ge=0, le=1)
    n_steps: int = Field(default=2048, ge=1, description="Rollout steps before update (PPO/A2C)")
    ent_coef: float = Field(default=0.01, ge=0, description="Entropy coefficient")
    clip_range: float = Field(default=0.2, gt=0, le=1, description="PPO clip range")
    buffer_size: int = Field(default=100_000, ge=1, description="SAC replay buffer size")
    tau: float = Field(default=0.005, gt=0, le=1, description="SAC soft update coefficient")
    frozen: bool = Field(default=False, description="If True, inference only")
    checkpoint_path: Optional[str] = Field(default=None)
    rule_agent_class: Optional[str] = Field(default=None, description="e.g. alpha_momentum.AlphaMomentum")


class PoolConfig(BaseModel):
    """Agent pool configuration."""

    agents: list[AgentConfig] = Field(default_factory=lambda: [
        AgentConfig(agent_type="ppo"),
        AgentConfig(agent_type="sac"),
        AgentConfig(agent_type="static", frozen=True),
        AgentConfig(agent_type="static", frozen=True),
        AgentConfig(agent_type="rule_based", rule_agent_class="alpha_momentum.AlphaMomentum"),
        AgentConfig(agent_type="rule_based", rule_agent_class="beta_mean_reversion.BetaMeanReversion"),
    ])
    action_aggregation: str = Field(default="weighted_mean", description="weighted_mean | majority_vote")
    equal_weights: bool = Field(default=True)


class TrainingConfig(BaseModel):
    """Training pipeline configuration."""

    total_timesteps: int = Field(default=500_000, ge=1)
    eval_interval: int = Field(default=10_000, ge=1)
    checkpoint_interval: int = Field(default=50_000, ge=1)
    num_generations: int = Field(default=10, ge=1, description="Population-based training generations")
    episodes_per_generation: int = Field(default=100, ge=1)
    top_k_promote: int = Field(default=2, ge=1)
    bottom_k_demote: int = Field(default=1, ge=0)
    tensorboard_log_dir: str = Field(default="logs/tensorboard")
    checkpoint_dir: str = Field(default="checkpoints")


class ComputeConfig(BaseModel):
    """Compute orchestration configuration."""

    prefer_gpu: bool = Field(default=True)
    cpu_workers: int = Field(default=6, ge=1, le=16)
    gpu_memory_limit_gb: float = Field(default=12.0, gt=0)
    fallback_to_cpu: bool = Field(default=True)


class DataConfig(BaseModel):
    """Data pipeline configuration."""

    tickers: list[str] = Field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "TSLA", "JPM", "V", "UNH",
    ])
    cache_dir: str = Field(default="data/cache")
    feature_cache_format: str = Field(default="npy", description="npy | parquet")
    lookback_days: int = Field(default=60, ge=1)


class ValidationConfig(BaseModel):
    """ATHENA-style validation configuration."""

    bootstrap_samples: int = Field(default=2000, ge=100)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    min_sharpe: float = Field(default=0.3)
    max_drawdown_pct: float = Field(default=0.25)
    min_win_rate: float = Field(default=0.40)
    min_profit_factor: float = Field(default=1.1)
    walk_forward_windows: int = Field(default=4, ge=2)
    min_wfe: float = Field(default=0.40, description="Walk-forward efficiency")


class HydraConfig(BaseModel):
    """Root configuration for the Hydra system."""

    env: EnvConfig = Field(default_factory=EnvConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    pool: PoolConfig = Field(default_factory=PoolConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    seed: int = Field(default=42)
    log_level: str = Field(default="INFO")

    @classmethod
    def from_yaml(cls, path: str | Path) -> HydraConfig:
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw or {})

    def to_yaml(self, path: str | Path) -> None:
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
