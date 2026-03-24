"""Tests for the Strategy Distillation system (Phases 1-4).

Covers:
- Phase 1: FactorDataStore, RewardCalibrator, StrategyDistiller
- Phase 2: Regime-conditional rewards
- Phase 3: SEC13FParser integration
- Phase 4: Inverse RL reward inference
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure project root and parent are on sys.path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Factor Data
# ═══════════════════════════════════════════════════════════════════════════


class TestFactorDataStore:
    """Test FactorDataStore caching and parsing logic."""

    def test_import(self):
        from hydra.distillation.factor_data import FactorDataStore
        store = FactorDataStore()
        assert store is not None

    def test_ff5_returns_dataframe_or_none(self):
        """get_fama_french_5 should return a DataFrame or None (no crash)."""
        from hydra.distillation.factor_data import FactorDataStore
        store = FactorDataStore(cache_dir="data/test_factor_cache")
        result = store.get_fama_french_5()
        # It may be None if network is down — that's OK
        if result is not None:
            assert isinstance(result, pd.DataFrame)
            assert "Mkt-RF" in result.columns
            assert "SMB" in result.columns
            assert len(result) > 100  # Should have years of daily data

    def test_fh7_returns_dataframe_or_none(self):
        """get_fung_hsieh_7 should return a DataFrame or None."""
        from hydra.distillation.factor_data import FactorDataStore
        store = FactorDataStore(cache_dir="data/test_factor_cache")
        result = store.get_fung_hsieh_7()
        if result is not None:
            assert isinstance(result, pd.DataFrame)

    def test_hfri_returns_none(self):
        """get_hfri_composite should return None (requires paid subscription)."""
        from hydra.distillation.factor_data import FactorDataStore
        store = FactorDataStore()
        assert store.get_hfri_composite() is None

    def test_clear_memory_cache(self):
        from hydra.distillation.factor_data import FactorDataStore
        store = FactorDataStore()
        store.clear_memory_cache()
        # Should not raise


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Reward Calibrator
# ═══════════════════════════════════════════════════════════════════════════


class TestRewardCalibrator:
    """Test factor-to-reward mapping."""

    @pytest.fixture
    def synthetic_ff5(self):
        """Create synthetic FF5 factor data."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        data = {
            "Mkt-RF": np.random.normal(0.0003, 0.01, 500),
            "SMB": np.random.normal(0.0001, 0.005, 500),
            "HML": np.random.normal(0.0001, 0.005, 500),
            "RMW": np.random.normal(0.0001, 0.003, 500),
            "CMA": np.random.normal(0.0000, 0.003, 500),
            "RF": np.full(500, 0.0001),
        }
        return pd.DataFrame(data, index=dates)

    def test_compute_target_profile_no_target(self, synthetic_ff5):
        """Without target returns, should return mean factor returns."""
        from hydra.distillation.reward_calibrator import RewardCalibrator
        cal = RewardCalibrator()
        loadings = cal.compute_target_profile(synthetic_ff5)
        assert isinstance(loadings, dict)
        assert "Mkt-RF" in loadings

    def test_compute_target_profile_with_target(self, synthetic_ff5):
        """With target returns, should return OLS betas."""
        from hydra.distillation.reward_calibrator import RewardCalibrator
        cal = RewardCalibrator()
        # Synthetic target: correlated with Mkt-RF
        target = synthetic_ff5["Mkt-RF"] * 1.2 + np.random.normal(0, 0.001, 500)
        loadings = cal.compute_target_profile(synthetic_ff5, target)
        assert isinstance(loadings, dict)
        # Should have positive Mkt-RF loading
        assert loadings.get("Mkt-RF", 0) > 0.5

    def test_map_to_reward_config(self, synthetic_ff5):
        """Factor loadings should map to valid reward parameters."""
        from hydra.distillation.reward_calibrator import RewardCalibrator
        cal = RewardCalibrator()
        loadings = {"Mkt-RF": 1.0, "SMB": -0.3, "HML": 0.5, "RMW": 0.2, "CMA": -0.1}
        current = {
            "sharpe_eta": 0.05,
            "drawdown_penalty": 0.5,
            "transaction_penalty": 0.1,
            "holding_penalty": 0.1,
            "pnl_bonus_weight": 1.0,
            "reward_scale": 100.0,
        }
        proposed = cal.map_to_reward_config(loadings, current)
        assert isinstance(proposed, dict)
        # Positive Mkt-RF should increase pnl_bonus_weight
        assert proposed["pnl_bonus_weight"] > current["pnl_bonus_weight"]
        # All values should be within valid bounds
        assert 0.001 <= proposed["sharpe_eta"] <= 0.5
        assert 0.1 <= proposed["drawdown_penalty"] <= 5.0
        assert 0.005 <= proposed["transaction_penalty"] <= 1.0

    def test_get_calibration_report(self):
        from hydra.distillation.reward_calibrator import RewardCalibrator
        cal = RewardCalibrator()
        loadings = {"Mkt-RF": 0.8, "SMB": -0.2}
        proposed = {"sharpe_eta": 0.06, "pnl_bonus_weight": 1.4}
        current = {"sharpe_eta": 0.05, "pnl_bonus_weight": 1.0}
        report = cal.get_calibration_report(loadings, proposed, current)
        assert isinstance(report, dict)
        assert "factor_loadings" in report or "current_vs_proposed" in report

    def test_constrained_optimization(self, synthetic_ff5):
        """Constrained optimization should return valid reward weights."""
        from hydra.distillation.reward_calibrator import RewardCalibrator
        cal = RewardCalibrator()
        weights = cal.run_constrained_optimization(synthetic_ff5)
        assert isinstance(weights, dict)
        # Should contain reward parameters
        assert len(weights) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Strategy Distiller Corp Agent
# ═══════════════════════════════════════════════════════════════════════════


class TestStrategyDistiller:
    """Test the StrategyDistiller corp agent."""

    @pytest.fixture
    def distiller(self):
        from corp.agents.strategy_distiller import StrategyDistiller
        state = MagicMock()
        state.add_proposal = MagicMock()
        decision_log = MagicMock()
        return StrategyDistiller(state, decision_log, interval_hours=0)

    def test_should_run_first_time(self, distiller):
        """Should run on first invocation."""
        assert distiller.should_run() is True

    def test_should_not_run_after_recent_run(self, distiller):
        """Should not run if recently run."""
        from datetime import datetime
        distiller._last_run_time = datetime.now()
        distiller._interval_hours = 168  # Weekly
        assert distiller.should_run() is False

    def test_run_skips_when_not_scheduled(self, distiller):
        """Should skip when not scheduled and force=False."""
        from datetime import datetime
        distiller._last_run_time = datetime.now()
        distiller._interval_hours = 168
        result = distiller.run({"config_dict": {}})
        assert result.get("skipped") is True

    def test_run_with_force(self, distiller):
        """Should run when force=True regardless of schedule."""
        from datetime import datetime
        distiller._last_run_time = datetime.now()
        distiller._interval_hours = 168

        # Mock factor data loading to avoid network calls
        with patch.object(distiller, "_load_factor_data", return_value=None):
            result = distiller.run({"config_dict": {}, "force": True})
            # Should not be skipped (but may error on no factor data)
            assert "skipped" not in result

    def test_run_handles_factor_load_failure(self, distiller):
        """Should handle factor data loading failure gracefully."""
        with patch.object(distiller, "_load_factor_data", return_value=None):
            result = distiller.run({"config_dict": {}, "force": True})
            assert result.get("error") == "Failed to load factor data"
            assert result["calibrated"] is False

    def test_get_calibration_summary(self, distiller):
        summary = distiller.get_calibration_summary()
        assert summary["total_calibrations"] == 0
        assert summary["latest"] is None


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Regime-Conditional Rewards
# ═══════════════════════════════════════════════════════════════════════════


class TestRegimeRewards:
    """Test regime multiplier configuration."""

    def test_regime_multipliers_structure(self):
        from hydra.distillation.regime_rewards import REGIME_MULTIPLIERS
        assert "risk_on" in REGIME_MULTIPLIERS
        assert "risk_off" in REGIME_MULTIPLIERS
        assert "crisis" in REGIME_MULTIPLIERS

    def test_risk_on_is_aggressive(self):
        """Risk-on multipliers should lean aggressive (higher P&L, lower DD penalty)."""
        from hydra.distillation.regime_rewards import get_multipliers
        mult = get_multipliers("risk_on")
        assert mult["pnl_bonus_weight"] >= 1.5, "risk_on should amplify P&L bonus"
        assert mult["drawdown_penalty"] <= 0.8, "risk_on should ease drawdown penalty"
        for key, val in mult.items():
            assert 0.3 <= val <= 2.0, f"risk_on {key}={val} out of expected range"

    def test_crisis_amplifies_drawdown(self):
        """Crisis should significantly increase drawdown penalty."""
        from hydra.distillation.regime_rewards import get_multipliers
        mult = get_multipliers("crisis")
        assert mult["drawdown_penalty"] >= 2.0
        assert mult["pnl_bonus_weight"] <= 0.5

    def test_unknown_regime_falls_back(self):
        """Unknown regime should fall back to risk_on."""
        from hydra.distillation.regime_rewards import get_multipliers
        unknown = get_multipliers("unknown_regime")
        risk_on = get_multipliers("risk_on")
        assert unknown == risk_on

    def test_reward_function_regime_integration(self):
        """Reward function should apply regime multipliers."""
        from hydra.envs.reward import DifferentialSharpeReward
        reward_fn = DifferentialSharpeReward(
            drawdown_penalty=0.5,
            pnl_bonus_weight=1.0,
        )
        reward_fn.reset(2500.0)

        # Default regime should be risk_on
        assert reward_fn.regime == "risk_on"

        # Set to crisis and compute reward
        reward_fn.set_regime("crisis")
        assert reward_fn.regime == "crisis"

        # Compute a step with positive return
        holdings = np.array([10.0, 5.0], dtype=np.float32)
        prices = np.array([50.0, 100.0], dtype=np.float32)
        reward_val, info = reward_fn.compute(2600.0, 0.0, holdings, prices)
        assert info["regime"] == "crisis"

    def test_reward_regime_changes_output(self):
        """Different regimes should produce different reward values."""
        from hydra.envs.reward import DifferentialSharpeReward

        holdings = np.array([10.0, 5.0], dtype=np.float32)
        prices = np.array([50.0, 100.0], dtype=np.float32)

        infos = {}
        for regime in ("risk_on", "risk_off", "crisis"):
            fn = DifferentialSharpeReward(
                drawdown_penalty=0.5,
                pnl_bonus_weight=1.0,
                reward_scale=100.0,
            )
            fn.reset(2500.0)
            fn.set_regime(regime)
            # Simulate drawdown
            reward_val, info = fn.compute(2400.0, 5.0, holdings, prices)
            infos[regime] = info

        # Crisis should penalize drawdown component more than risk_on
        assert infos["crisis"]["drawdown_penalty"] < infos["risk_on"]["drawdown_penalty"]
        # Different regimes should produce different total rewards
        assert infos["crisis"]["total_reward"] != infos["risk_on"]["total_reward"]


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Curriculum Regime
# ═══════════════════════════════════════════════════════════════════════════


class TestCurriculumRegime:
    """Test regime property in Curriculum."""

    def test_default_regime(self):
        from hydra.training.curriculum import Curriculum
        c = Curriculum()
        assert c.regime == "risk_on"

    def test_set_regime(self):
        from hydra.training.curriculum import Curriculum
        c = Curriculum()
        c.set_regime("crisis")
        assert c.regime == "crisis"

    def test_invalid_regime_rejected(self):
        from hydra.training.curriculum import Curriculum
        c = Curriculum()
        c.set_regime("invalid_regime")
        assert c.regime == "risk_on"  # Should keep default

    def test_regime_in_adjustments(self):
        from hydra.training.curriculum import Curriculum
        c = Curriculum()
        c.set_regime("risk_off")
        adjustments = c.on_generation(1, {"agent1": 10.0})
        assert adjustments["regime"] == "risk_off"


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: SEC 13F Parser
# ═══════════════════════════════════════════════════════════════════════════


class TestSEC13FParser:
    """Test SEC13FParser structure and offline capabilities."""

    def test_import(self):
        from trading_agents.data.sec_13f_parser import SEC13FParser
        parser = SEC13FParser()
        assert parser is not None

    def test_top_funds_dict(self):
        from trading_agents.data.sec_13f_parser import TOP_FUNDS
        assert len(TOP_FUNDS) >= 10
        # Bridgewater should be tracked
        assert any("bridgewater" in k.lower() for k in TOP_FUNDS)

    def test_sector_map_coverage(self):
        from trading_agents.data.sec_13f_parser import SECTOR_MAP
        assert isinstance(SECTOR_MAP, dict)
        assert len(SECTOR_MAP) > 50
        # AAPL should be mapped to tech
        assert SECTOR_MAP.get("AAPL") == "tech"
        assert SECTOR_MAP.get("XOM") == "energy"
        assert SECTOR_MAP.get("JPM") == "finance"

    def test_cusip_to_ticker_mapping(self):
        from trading_agents.data.sec_13f_parser import CUSIP_TO_TICKER
        assert isinstance(CUSIP_TO_TICKER, dict)
        assert len(CUSIP_TO_TICKER) > 50

    def test_rate_limiter(self):
        from trading_agents.data.sec_13f_parser import RateLimiter
        limiter = RateLimiter(max_per_second=100.0)  # Fast for testing
        limiter.wait()
        limiter.wait()
        # Should not crash

    def test_compute_sector_consensus_empty(self):
        """Consensus with empty filings should return empty dict."""
        from trading_agents.data.sec_13f_parser import SEC13FParser
        parser = SEC13FParser()
        result = parser.compute_sector_consensus({})
        assert isinstance(result, dict)

    def test_compute_sector_consensus_synthetic(self):
        """Test consensus with synthetic filing data."""
        from trading_agents.data.sec_13f_parser import SEC13FParser
        parser = SEC13FParser()

        # Synthetic filings: two funds heavily in tech
        filings = {
            "FundA": {
                "AAPL": {"ticker": "AAPL", "shares": 1000, "value": 150000},
                "MSFT": {"ticker": "MSFT", "shares": 500, "value": 200000},
            },
            "FundB": {
                "AAPL": {"ticker": "AAPL", "shares": 2000, "value": 300000},
                "GOOGL": {"ticker": "GOOGL", "shares": 300, "value": 45000},
                "XOM": {"ticker": "XOM", "shares": 100, "value": 10000},
            },
        }
        result = parser.compute_sector_consensus(filings)
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: HedgeFundTracker Integration
# ═══════════════════════════════════════════════════════════════════════════


class TestHedgeFundTrackerIntegration:
    """Test that HedgeFundTracker wires to SEC13FParser."""

    def test_get_sector_rotation_exists(self):
        from trading_agents.data.hedge_fund_tracker import HedgeFundTracker
        tracker = HedgeFundTracker()
        assert hasattr(tracker, "get_sector_rotation")

    def test_get_sector_rotation_returns_dict(self):
        """Should return a dict (possibly empty) without crashing."""
        from trading_agents.data.hedge_fund_tracker import HedgeFundTracker
        tracker = HedgeFundTracker()
        result = tracker.get_sector_rotation()
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: Inverse RL
# ═══════════════════════════════════════════════════════════════════════════


class TestInverseRL:
    """Test MaxEnt IRL reward inference."""

    def test_import(self):
        from hydra.distillation.inverse_rl import InverseRLCalibrator
        irl = InverseRLCalibrator()
        assert irl is not None

    def test_extract_trajectories_empty(self):
        """Empty filings history should return zero-padded array."""
        from hydra.distillation.inverse_rl import extract_expert_trajectories
        result = extract_expert_trajectories([], None)
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 12

    def test_extract_trajectories_synthetic(self):
        """Synthetic filings should produce feature vectors."""
        from hydra.distillation.inverse_rl import extract_expert_trajectories

        # Two quarters of synthetic filings
        q1 = {
            "FundA": {
                "AAPL": {"ticker": "AAPL", "shares": 1000, "value": 150000},
                "XOM": {"ticker": "XOM", "shares": 500, "value": 50000},
            }
        }
        q2 = {
            "FundA": {
                "AAPL": {"ticker": "AAPL", "shares": 1200, "value": 180000},
                "MSFT": {"ticker": "MSFT", "shares": 300, "value": 120000},
            }
        }

        result = extract_expert_trajectories([q1, q2], None)
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 12
        assert result.shape[0] >= 1

    def test_compute_expert_features(self):
        from hydra.distillation.inverse_rl import compute_expert_feature_expectations
        # Synthetic trajectory features
        trajectories = np.random.randn(5, 12).astype(np.float32)
        result = compute_expert_feature_expectations(trajectories)
        assert result.shape == (12,)
        # Should be the mean
        np.testing.assert_allclose(result, trajectories.mean(axis=0), atol=1e-5)

    def test_infer_reward_weights(self):
        from hydra.distillation.inverse_rl import infer_reward_weights
        expert_features = np.array([0.5, -0.3, 0.2, 0.1, -0.1,
                                     0.4, -0.2, 0.1, -0.3,
                                     0.2, 0.6, -0.4], dtype=np.float64)
        weights = infer_reward_weights(expert_features)
        assert weights.shape == (12,)
        # Should be normalized
        assert np.linalg.norm(weights) > 0

    def test_map_weights_to_reward_config(self):
        from hydra.distillation.inverse_rl import map_weights_to_reward_config
        weights = np.array([0.5, -0.3, 0.2, 0.1, -0.1,
                            0.4, -0.2, 0.1, -0.3,
                            0.2, 0.6, -0.4], dtype=np.float64)
        config = map_weights_to_reward_config(weights)
        assert isinstance(config, dict)
        assert "pnl_bonus_weight" in config
        assert "drawdown_penalty" in config
        assert "transaction_penalty" in config
        assert "sharpe_eta" in config
        # All values should be positive
        for key, val in config.items():
            assert val >= 0, f"{key}={val} should be >= 0"

    def test_map_weights_wrong_size(self):
        from hydra.distillation.inverse_rl import map_weights_to_reward_config
        with pytest.raises(ValueError):
            map_weights_to_reward_config(np.array([1.0, 2.0, 3.0]))

    def test_get_inference_report(self):
        from hydra.distillation.inverse_rl import (
            get_inference_report,
            map_weights_to_reward_config,
        )
        weights = np.random.randn(12)
        config = map_weights_to_reward_config(weights)
        report = get_inference_report(weights, config, num_transitions=10)
        assert isinstance(report, dict)
        assert "proposed_config" in report or "raw_weights" in report

    def test_irl_calibrator_class(self):
        """Test the InverseRLCalibrator orchestrator class."""
        from hydra.distillation.inverse_rl import InverseRLCalibrator
        irl = InverseRLCalibrator()
        assert hasattr(irl, "fit")
        assert hasattr(irl, "reward_patch")

    def test_irl_recovers_known_weights(self):
        """IRL should recover directionally consistent weights from synthetic data.

        Generate synthetic expert trajectories with known preferences:
        - High market exposure (positive Mkt-RF weight)
        - High drawdown avoidance (negative drawdown weight)
        Then verify the inferred weights match the direction.
        """
        from hydra.distillation.inverse_rl import (
            compute_expert_feature_expectations,
            infer_reward_weights,
        )

        np.random.seed(42)
        # Expert with strong market exposure (dim 0) and drawdown avoidance (dim 11)
        expert_trajectories = np.random.randn(20, 12).astype(np.float64) * 0.1
        expert_trajectories[:, 0] += 0.8   # Strong positive Mkt-RF
        expert_trajectories[:, 11] -= 0.5  # Strong negative drawdown

        features = compute_expert_feature_expectations(expert_trajectories)
        weights = infer_reward_weights(features)

        # Directions should be consistent
        assert weights[0] > 0, "Mkt-RF weight should be positive"
        assert weights[11] < 0, "Drawdown weight should be negative"


# ═══════════════════════════════════════════════════════════════════════════
# Integration: StrategyDistiller in Corp Graph
# ═══════════════════════════════════════════════════════════════════════════


class TestCorpGraphIntegration:
    """Test that distiller is registered in the corp graph."""

    def test_distiller_registered_in_build_all_agents(self):
        from corp.state.corporation_state import CorporationState
        from corp.state.decision_log import DecisionLog
        from corp.state.config_blacklist import ConfigBlacklist
        from corp.scripts.run_corporation import build_all_agents

        state = CorporationState()
        decision_log = DecisionLog()
        blacklist = ConfigBlacklist()

        agents = build_all_agents(state, decision_log, blacklist)
        assert "strategy_distiller" in agents

    def test_node_intelligence_includes_distiller(self):
        """node_intelligence should call strategy_distiller if present."""
        from corp.graph.nodes import node_intelligence

        mock_distiller = MagicMock()
        mock_distiller.run.return_value = {"calibrated": True, "proposed_weights": {}}

        agents = {
            "geopolitics_expert": MagicMock(),
            "innovation_scout": MagicMock(),
            "strategy_distiller": mock_distiller,
        }
        agents["geopolitics_expert"].run.return_value = {"regime": "risk_on"}
        agents["innovation_scout"].run.return_value = {}

        state = {"config_dict": {}, "force_all_agents": True}
        result = node_intelligence(state, agents)

        mock_distiller.run.assert_called_once()
        assert "distillation_result" in result


# ═══════════════════════════════════════════════════════════════════════════
# Integration: __init__.py Exports
# ═══════════════════════════════════════════════════════════════════════════


class TestDistillationPackageExports:
    """Test that the distillation package exports are correct."""

    def test_all_exports(self):
        from hydra.distillation import (
            FactorDataStore,
            RewardCalibrator,
            InverseRLCalibrator,
            REGIME_MULTIPLIERS,
            get_multipliers,
        )
        assert FactorDataStore is not None
        assert RewardCalibrator is not None
        assert InverseRLCalibrator is not None
        assert isinstance(REGIME_MULTIPLIERS, dict)
        assert callable(get_multipliers)
