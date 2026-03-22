"""Strategy Distillation — learn decision frameworks from hedge fund factor analysis.

Backtesting and training only.
"""

from hydra.distillation.factor_data import FactorDataStore
from hydra.distillation.reward_calibrator import RewardCalibrator
from hydra.distillation.regime_rewards import REGIME_MULTIPLIERS, get_multipliers
from hydra.distillation.inverse_rl import InverseRLCalibrator

__all__ = [
    "FactorDataStore",
    "RewardCalibrator",
    "InverseRLCalibrator",
    "REGIME_MULTIPLIERS",
    "get_multipliers",
]
