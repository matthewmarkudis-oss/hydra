"""Default configuration factory."""

from hydra.config.schema import HydraConfig


def get_default_config() -> HydraConfig:
    """Return the default Hydra configuration."""
    return HydraConfig()


def get_test_config() -> HydraConfig:
    """Return a lightweight config for testing."""
    return HydraConfig(
        env={"num_stocks": 3, "initial_cash": 10_000.0},
        training={"total_timesteps": 1000, "eval_interval": 500, "checkpoint_interval": 500},
        data={"tickers": ["AAPL", "MSFT", "GOOGL"], "lookback_days": 10},
        compute={"cpu_workers": 2, "prefer_gpu": False},
    )
