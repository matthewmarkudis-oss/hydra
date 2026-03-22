"""Tests for the new corp agents: RiskManager, DataQualityMonitor, PerformanceAnalyst.

Also tests regime wiring, auto-approve pipeline, and graph reordering.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from corp.state.corporation_state import CorporationState
from corp.state.decision_log import DecisionLog


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_state(tmp_path):
    """Create a temporary CorporationState."""
    state_file = tmp_path / "state.json"
    return CorporationState(state_file=str(state_file))


@pytest.fixture
def decision_log(tmp_path):
    """Create a temporary DecisionLog."""
    log_file = tmp_path / "decisions.jsonl"
    return DecisionLog(log_file=str(log_file))


# ── RiskManager ──────────────────────────────────────────────────────────────


class TestRiskManager:
    def test_no_circuit_breakers(self, tmp_state, decision_log):
        from corp.agents.risk_manager import RiskManager

        rm = RiskManager(tmp_state, decision_log)
        result = rm.on_generation_complete(1, [{"diagnosis": {}}])
        assert result is None

    def test_alert_action(self, tmp_state, decision_log):
        from corp.agents.risk_manager import RiskManager

        rm = RiskManager(tmp_state, decision_log)
        gen_results = [{
            "diagnosis": {
                "circuit_breaker_actions": [
                    {"action": "alert", "target": None, "reduction_pct": 0.0, "reason": "test"},
                ],
            },
        }]
        result = rm.on_generation_complete(1, gen_results)
        assert result is not None
        assert len(result["alerts"]) == 1
        assert result["alerts"][0]["type"] == "circuit_breaker_alert"

    def test_reduce_allocation_enforce(self, tmp_state, decision_log):
        from corp.agents.risk_manager import RiskManager

        rm = RiskManager(tmp_state, decision_log, enforce=True)
        gen_results = [{
            "diagnosis": {
                "circuit_breaker_actions": [
                    {"action": "reduce_allocation", "target": "ppo_1", "reduction_pct": 0.25},
                ],
            },
        }]
        result = rm.on_generation_complete(1, gen_results)
        assert result is not None
        assert "weight_overrides" in result
        assert result["weight_overrides"]["ppo_1"] == 0.75

    def test_reduce_allocation_backtest_mode(self, tmp_state, decision_log):
        from corp.agents.risk_manager import RiskManager

        rm = RiskManager(tmp_state, decision_log, enforce=False)
        gen_results = [{
            "diagnosis": {
                "circuit_breaker_actions": [
                    {"action": "reduce_allocation", "target": "ppo_1", "reduction_pct": 0.25},
                ],
            },
        }]
        result = rm.on_generation_complete(1, gen_results)
        assert result is not None
        # In backtest mode, weight_overrides should NOT be in the result
        assert "weight_overrides" not in result

    def test_cumulative_reductions(self, tmp_state, decision_log):
        from corp.agents.risk_manager import RiskManager

        rm = RiskManager(tmp_state, decision_log, enforce=True)
        action = {"action": "reduce_allocation", "target": "ppo_1", "reduction_pct": 0.25}

        # First reduction: 25%
        rm.on_generation_complete(1, [{"diagnosis": {"circuit_breaker_actions": [action]}}])
        assert rm._active_reductions["ppo_1"] == 0.25

        # Second reduction: cumulative 50%
        rm.on_generation_complete(2, [{"diagnosis": {"circuit_breaker_actions": [action]}}])
        assert rm._active_reductions["ppo_1"] == 0.50

    def test_shutdown_agent(self, tmp_state, decision_log):
        from corp.agents.risk_manager import RiskManager

        rm = RiskManager(tmp_state, decision_log, enforce=True)
        gen_results = [{
            "diagnosis": {
                "circuit_breaker_actions": [
                    {"action": "shutdown_agent", "target": "sac_1", "reduction_pct": 0.0},
                ],
            },
        }]
        result = rm.on_generation_complete(1, gen_results)
        assert result["weight_overrides"]["sac_1"] == 0.01

    def test_risk_summary(self, tmp_state, decision_log):
        from corp.agents.risk_manager import RiskManager

        rm = RiskManager(tmp_state, decision_log)
        summary = rm.get_risk_summary()
        assert "enforce_mode" in summary
        assert "active_reductions" in summary
        assert "total_interventions" in summary

    def test_clear_reduction(self, tmp_state, decision_log):
        from corp.agents.risk_manager import RiskManager

        rm = RiskManager(tmp_state, decision_log, enforce=True)
        rm._active_reductions["ppo_1"] = 0.5
        rm.clear_reduction("ppo_1")
        assert "ppo_1" not in rm._active_reductions


# ── DataQualityMonitor ───────────────────────────────────────────────────────


class TestDataQualityMonitor:
    def test_runs_all_checks(self, tmp_state, decision_log):
        from corp.agents.data_quality_monitor import DataQualityMonitor

        dqm = DataQualityMonitor(tmp_state, decision_log, checkpoint_dir="/nonexistent")
        result = dqm.run({})
        assert "checks" in result
        assert "all_passed" in result
        assert "critical_failures" in result
        assert "warnings" in result
        # Should have 4 checks
        assert len(result["checks"]) == 4

    def test_check_names(self, tmp_state, decision_log):
        from corp.agents.data_quality_monitor import DataQualityMonitor

        dqm = DataQualityMonitor(tmp_state, decision_log, checkpoint_dir="/nonexistent")
        result = dqm.run({})
        names = {c["name"] for c in result["checks"]}
        assert names == {"factor_data", "news_feeds", "corp_state", "checkpoint"}

    def test_checkpoint_no_dir(self, tmp_state, decision_log):
        from corp.agents.data_quality_monitor import DataQualityMonitor

        dqm = DataQualityMonitor(tmp_state, decision_log, checkpoint_dir="/nonexistent")
        check = dqm._check_checkpoint_integrity()
        assert check["status"] == "ok"
        assert check["details"]["warm_start"] is False

    def test_checkpoint_valid(self, tmp_state, decision_log, tmp_path):
        from corp.agents.data_quality_monitor import DataQualityMonitor

        # Create a valid checkpoint structure
        ckpt_dir = tmp_path / "ckpts"
        ckpt_dir.mkdir()
        ckpt_path = ckpt_dir / "gen_10" / "episode_50"
        ckpt_path.mkdir(parents=True)
        (ckpt_path / "pool_metadata.json").write_text("{}")
        (ckpt_dir / "latest.json").write_text(json.dumps({
            "checkpoint_path": str(ckpt_path),
            "generation": 10,
        }))

        dqm = DataQualityMonitor(tmp_state, decision_log, checkpoint_dir=str(ckpt_dir))
        check = dqm._check_checkpoint_integrity()
        assert check["status"] == "ok"
        assert check["details"]["warm_start"] is True

    def test_corp_state_no_regime(self, tmp_state, decision_log):
        from corp.agents.data_quality_monitor import DataQualityMonitor

        dqm = DataQualityMonitor(tmp_state, decision_log)
        check = dqm._check_corp_state_freshness()
        assert check["status"] == "warning"
        assert "hasn't run" in check["message"]


# ── PerformanceAnalyst ───────────────────────────────────────────────────────


class TestPerformanceAnalyst:
    def _make_generations(self, n_gens=10, agents=("ppo_1", "td3_1", "rule_1")):
        """Create synthetic generation results for testing."""
        gens = []
        for i in range(n_gens):
            scores = {}
            for a_idx, agent in enumerate(agents):
                # Give each agent a distinctive pattern
                base = (a_idx + 1) * 10
                noise = np.random.normal(0, 5)
                scores[agent] = base + i * 0.5 + noise  # slight upward trend
            gens.append({
                "generation": i + 1,
                "eval_scores": scores,
                "diagnosis": {},
            })
        return gens

    def test_skips_with_few_generations(self, tmp_state, decision_log):
        from corp.agents.performance_analyst import PerformanceAnalyst

        pa = PerformanceAnalyst(tmp_state, decision_log)
        result = pa.run({"generation_results": [{"eval_scores": {}}]})
        assert result.get("skipped") is True

    def test_full_analysis(self, tmp_state, decision_log):
        from corp.agents.performance_analyst import PerformanceAnalyst

        pa = PerformanceAnalyst(tmp_state, decision_log)
        gens = self._make_generations(10)
        result = pa.run({"generation_results": gens})

        assert "correlations" in result
        assert "agent_profiles" in result
        assert "regime_attribution" in result
        assert "pool_efficiency" in result
        assert "recommendations" in result
        assert result["generations_analyzed"] == 10

    def test_correlation_detection(self, tmp_state, decision_log):
        from corp.agents.performance_analyst import PerformanceAnalyst

        pa = PerformanceAnalyst(tmp_state, decision_log)

        # Create two perfectly correlated agents
        gens = []
        for i in range(10):
            score = i * 10
            gens.append({
                "generation": i + 1,
                "eval_scores": {
                    "agent_a": score,
                    "agent_b": score * 1.1,  # perfectly correlated
                    "agent_c": (-1) ** i * 50,  # uncorrelated
                },
            })

        result = pa.run({"generation_results": gens})
        corrs = result["correlations"]

        # Find the a-b pair
        ab = next((c for c in corrs if set([c["agent_a"], c["agent_b"]]) == {"agent_a", "agent_b"}), None)
        assert ab is not None
        assert ab["r"] > 0.99
        assert ab["redundant"] == True

    def test_agent_profiles(self, tmp_state, decision_log):
        from corp.agents.performance_analyst import PerformanceAnalyst

        pa = PerformanceAnalyst(tmp_state, decision_log)
        gens = self._make_generations(10, agents=("ppo_1",))
        result = pa.run({"generation_results": gens})

        profile = result["agent_profiles"]["ppo_1"]
        assert "mean" in profile
        assert "std" in profile
        assert "trend" in profile
        assert "best_score" in profile
        assert "appearances" in profile
        assert profile["appearances"] == 10

    def test_pool_efficiency(self, tmp_state, decision_log):
        from corp.agents.performance_analyst import PerformanceAnalyst

        pa = PerformanceAnalyst(tmp_state, decision_log)
        # Create a scenario where one agent dominates
        gens = [{
            "generation": i + 1,
            "eval_scores": {"star": 100, "weak_1": -5, "weak_2": -10},
        } for i in range(5)]

        result = pa.run({"generation_results": gens})
        eff = result["pool_efficiency"]
        assert eff["top_contributor"] == "star"
        assert eff["positive_fraction"] < 0.5

    def test_recommendations_generated(self, tmp_state, decision_log):
        from corp.agents.performance_analyst import PerformanceAnalyst

        pa = PerformanceAnalyst(tmp_state, decision_log)

        # Create two redundant agents
        gens = []
        for i in range(10):
            score = i * 10
            gens.append({
                "generation": i + 1,
                "eval_scores": {"a": score, "b": score * 1.05},
            })

        result = pa.run({"generation_results": gens})
        recs = result["recommendations"]
        # Should recommend removing the redundant pair
        assert any("Redundant" in r for r in recs)


# ── Auto-Approve ─────────────────────────────────────────────────────────────


class TestAutoApprove:
    def test_auto_approve_eligible(self, tmp_state):
        tmp_state.submit_proposal({
            "type": "config_patch",
            "confidence": 0.9,
            "risk_assessment": "low",
            "patch": {"reward.drawdown_penalty": 0.6},
            "description": "Test patch",
        })

        approved = tmp_state.auto_resolve_proposals()
        assert len(approved) == 1
        assert approved[0]["reward.drawdown_penalty"] == 0.6

    def test_auto_approve_low_confidence_rejected(self, tmp_state):
        tmp_state.submit_proposal({
            "type": "config_patch",
            "confidence": 0.5,
            "risk_assessment": "low",
            "patch": {"reward.drawdown_penalty": 0.6},
        })

        approved = tmp_state.auto_resolve_proposals()
        assert len(approved) == 0

    def test_auto_approve_high_risk_rejected(self, tmp_state):
        tmp_state.submit_proposal({
            "type": "config_patch",
            "confidence": 0.9,
            "risk_assessment": "high",
            "patch": {"reward.drawdown_penalty": 0.6},
        })

        approved = tmp_state.auto_resolve_proposals()
        assert len(approved) == 0

    def test_auto_approve_stress_test_rejected(self, tmp_state):
        tmp_state.submit_proposal({
            "type": "stress_test",
            "confidence": 0.95,
            "risk_assessment": "low",
            "patch": {"env.max_position_pct": 0.1},
        })

        approved = tmp_state.auto_resolve_proposals()
        assert len(approved) == 0

    def test_auto_approve_medium_risk_accepted(self, tmp_state):
        tmp_state.submit_proposal({
            "type": "config_patch",
            "confidence": 0.7,
            "risk_assessment": "medium",
            "patch": {"reward.sharpe_eta": 0.1},
            "description": "Medium risk patch",
        })

        approved = tmp_state.auto_resolve_proposals()
        assert len(approved) == 1
        assert approved[0]["reward.sharpe_eta"] == 0.1

    def test_auto_approve_confidence_at_threshold(self, tmp_state):
        tmp_state.submit_proposal({
            "type": "config_patch",
            "confidence": 0.6,
            "risk_assessment": "low",
            "patch": {"reward.drawdown_penalty": 0.5},
        })

        approved = tmp_state.auto_resolve_proposals()
        assert len(approved) == 1

    def test_auto_approve_medium_rejected_when_max_low(self, tmp_state):
        tmp_state.submit_proposal({
            "type": "config_patch",
            "confidence": 0.9,
            "risk_assessment": "medium",
            "patch": {"reward.drawdown_penalty": 0.6},
        })

        # Explicitly pass max_risk="low" to override the default
        approved = tmp_state.auto_resolve_proposals(max_risk="low")
        assert len(approved) == 0

    def test_auto_approve_marks_status(self, tmp_state):
        tmp_state.submit_proposal({
            "type": "reward_calibration",
            "confidence": 0.85,
            "patch": {"reward.sharpe_eta": 0.08},
        })

        tmp_state.auto_resolve_proposals()
        pending = tmp_state.get_pending_proposals()
        assert len(pending) == 0


# ── Graph Reordering ─────────────────────────────────────────────────────────


class TestGraphReorder:
    def test_intelligence_before_pipeline(self):
        from corp.graph.corporation_graph import CorpGraph

        graph = CorpGraph({})
        intel_idx = graph._node_order.index("intelligence")
        pipe_idx = graph._node_order.index("pipeline")
        assert intel_idx < pipe_idx, "Intelligence must run before pipeline"


# ── Circuit Breaker Actions ──────────────────────────────────────────────────


class TestCircuitBreakerActions:
    def test_minor_severity_no_actions(self):
        from hydra.evolution.diagnostics import DiagnosticEngine

        engine = DiagnosticEngine()
        diagnosis = {"severity": "minor", "primary_issue": "none", "recommended_mutations": []}
        actions = engine.get_circuit_breaker_actions(diagnosis)
        assert actions == []

    def test_severe_generates_reduce_allocation(self):
        from hydra.evolution.diagnostics import DiagnosticEngine
        from hydra.evolution.mutation_engine import MutationRecord

        engine = DiagnosticEngine()
        diagnosis = {
            "severity": "severe",
            "primary_issue": "High drawdown",
            "recommended_mutations": [
                MutationRecord(
                    mutation_type="bench_agent",
                    category="inclusion",
                    description="Bench ppo_1",
                    params={"agent": "ppo_1"},
                ),
            ],
        }
        actions = engine.get_circuit_breaker_actions(diagnosis)
        action_types = [a.action for a in actions]
        assert "alert" in action_types
        assert "reduce_allocation" in action_types

        # reduce_allocation should target ppo_1
        reduce = [a for a in actions if a.action == "reduce_allocation"]
        assert reduce[0].target_agent == "ppo_1"
        assert reduce[0].reduction_pct == 0.25
