"""Pydantic configuration schema for Hydra system."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class EnvConfig(BaseModel):
    """Trading environment configuration."""

    num_stocks: int = Field(default=10, ge=1, le=500)
    episode_bars: int = Field(default=390, description="Bars per episode (5-min); 390 = 1 trading week")
    bar_interval_minutes: int = Field(default=5)
    initial_cash: float = Field(default=2_500.0, gt=0)
    transaction_cost_bps: float = Field(default=5.0, ge=0, description="Transaction cost in basis points")
    slippage_bps: float = Field(default=2.0, ge=0, description="Slippage in basis points")
    spread_bps: float = Field(default=1.0, ge=0, description="Bid-ask spread in basis points")
    max_position_pct: float = Field(default=0.30, gt=0, le=1.0)
    max_drawdown_pct: float = Field(default=0.20, gt=0, le=1.0)
    max_daily_loss_pct: float = Field(default=0.05, gt=0, le=1.0)
    normalize_obs: bool = Field(default=True)


class RewardConfig(BaseModel):
    """Reward function configuration."""

    sharpe_window: int = Field(default=20, ge=2)
    sharpe_eta: float = Field(default=0.05, gt=0, description="Differential Sharpe EMA decay")
    drawdown_penalty: float = Field(default=0.5, ge=0)
    transaction_penalty: float = Field(default=0.1, ge=0)
    holding_penalty: float = Field(default=0.1, ge=0, description="Penalty for large/idle positions")
    pnl_bonus_weight: float = Field(default=1.0, ge=0, description="Weight for direct P&L return bonus")
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
    num_generations: int = Field(default=30, ge=1, description="Population-based training generations")
    episodes_per_generation: int = Field(default=100, ge=1)
    top_k_promote: int = Field(default=2, ge=1)
    bottom_k_demote: int = Field(default=3, ge=0)
    max_pool_size: int = Field(default=20, ge=5, description="Max agents in pool; excess demoted after promotion")
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
        "NVDA", "TSLA", "AMD", "MARA", "COIN",
        "META", "AMZN", "GOOGL", "NFLX", "SQ",
    ])
    cache_dir: str = Field(default="data/cache")
    feature_cache_format: str = Field(default="npy", description="npy | parquet")
    lookback_days: int = Field(default=252, ge=1)


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


class ForwardTestConfig(BaseModel):
    """Forward-testing graduation pipeline configuration."""

    enabled: bool = Field(default=False, description="Must be explicitly enabled by CEO")
    duration_days: int = Field(default=20, ge=1, le=90)
    max_agents: int = Field(default=3, ge=1, le=10)
    initial_capital: float = Field(default=10000.0, gt=0)
    max_position_pct: float = Field(default=0.20, gt=0, le=0.50)
    sharpe_retention_min: float = Field(default=0.50, ge=0, le=1.0)
    drawdown_tolerance: float = Field(default=1.5, ge=1.0)
    win_rate_tolerance: float = Field(default=0.80, ge=0, le=1.0)
    poll_interval_minutes: int = Field(default=5, ge=1)


class HydraConfig(BaseModel):
    """Root configuration for the Hydra system."""

    env: EnvConfig = Field(default_factory=EnvConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    pool: PoolConfig = Field(default_factory=PoolConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    forward_test: ForwardTestConfig = Field(default_factory=ForwardTestConfig)
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

    def config_hash(self) -> str:
        """Compute a short SHA-256 hash of the tunable config parameters.

        Useful for tracking which configs have been tried and detecting
        regressions. Ignores non-functional fields (paths, log levels).
        """
        import hashlib
        import json

        hashable = {
            "env": self.env.model_dump(),
            "reward": self.reward.model_dump(),
            "pool": self.pool.model_dump(),
            "training": self.training.model_dump(),
            "data": self.data.model_dump(),
        }
        serialized = json.dumps(hashable, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def apply_patch(self, patch: dict) -> "HydraConfig":
        """Apply a partial config update and return a new validated config.

        Args:
            patch: Dict with section keys (env, reward, etc.) containing
                only the fields to change. Example:
                {"reward": {"drawdown_penalty": 0.3}, "env": {"max_position_pct": 0.4}}

        Returns:
            New HydraConfig instance with the patch applied.
        """
        current = self.model_dump()
        for section, values in patch.items():
            if section in current and isinstance(values, dict):
                current[section].update(values)
            else:
                current[section] = values

        # Auto-sync num_stocks when tickers change
        if "data" in patch and "tickers" in patch["data"]:
            current["env"]["num_stocks"] = len(current["data"]["tickers"])

        return HydraConfig.model_validate(current)
