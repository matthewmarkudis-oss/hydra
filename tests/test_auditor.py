"""Tests for the Training Auditor module."""

import pytest

from hydra.training.auditor import TrainingAuditor, AuditResult


def _make_gen_result(
    generation=1,
    train_mean_reward=-500.0,
    best_return_pct=0.5,
    mean_return_pct=-0.1,
    weights_after=None,
    eval_scores=None,
):
    """Helper to build a minimal gen_result dict."""
    result = {
        "generation": generation,
        "train_mean_reward": train_mean_reward,
        "best_return_pct": best_return_pct,
        "mean_return_pct": mean_return_pct,
        "eval_scores": eval_scores or {"ppo_1": -50, "td3_1": 100, "rppo_1": -200},
    }
    if weights_after is not None:
        result["competition"] = {"weights_after": weights_after, "converged": False}
    return result


class TestAuditorBasics:
    def test_clean_generation_continues(self):
        auditor = TrainingAuditor()
        result = auditor.audit_generation(
            generation=1,
            gen_result=_make_gen_result(),
            reward_params={"drawdown_penalty": 0.15, "transaction_penalty": 0.01},
            regime="risk_on",
        )
        assert result.verdict == "CONTINUE"
        assert len(result.alerts) == 0

    def test_history_accumulates(self):
        auditor = TrainingAuditor()
        for i in range(5):
            auditor.audit_generation(
                generation=i + 1,
                gen_result=_make_gen_result(generation=i + 1, train_mean_reward=-500 + i * 50),
            )
        assert len(auditor.history) == 5

    def test_summary(self):
        auditor = TrainingAuditor()
        auditor.audit_generation(1, _make_gen_result())
        summary = auditor.get_summary()
        assert summary["total_gens_audited"] == 1
        assert summary["latest_verdict"] == "CONTINUE"


class TestRewardStagnation:
    def test_no_alert_when_improving(self):
        auditor = TrainingAuditor(stagnation_window=3, stagnation_threshold=20.0)
        rewards = [-500, -400, -300, -200, -100]
        for i, r in enumerate(rewards):
            result = auditor.audit_generation(i + 1, _make_gen_result(train_mean_reward=r))
        assert result.verdict == "CONTINUE"

    def test_regression_alert(self):
        auditor = TrainingAuditor(stagnation_window=3, stagnation_threshold=20.0)
        # Reward gets worse
        rewards = [-100, -200, -400]
        for i, r in enumerate(rewards):
            result = auditor.audit_generation(i + 1, _make_gen_result(train_mean_reward=r))

        assert result.verdict != "CONTINUE"
        checks = [a.check_name for a in result.alerts]
        assert "reward_regression" in checks

    def test_stagnation_alert(self):
        auditor = TrainingAuditor(stagnation_window=3, stagnation_threshold=20.0)
        # Reward flat
        rewards = [-500, -500, -499]
        for i, r in enumerate(rewards):
            result = auditor.audit_generation(i + 1, _make_gen_result(train_mean_reward=r))

        checks = [a.check_name for a in result.alerts]
        assert "reward_stagnation" in checks


class TestWeightCollapse:
    def test_healthy_weights_no_alert(self):
        auditor = TrainingAuditor(weight_collapse_ratio=3.0)
        weights = {"ppo_1": 0.30, "td3_1": 0.25, "rppo_1": 0.10, "cmaes_1": 0.35}
        result = auditor.audit_generation(
            1, _make_gen_result(weights_after=weights)
        )
        # ratio = 0.35 / 0.10 = 3.5 > threshold 3.0, so no collapse
        collapse_alerts = [a for a in result.alerts if "weight_collapse" in a.check_name]
        assert len(collapse_alerts) == 0

    def test_collapsed_weights_alert(self):
        auditor = TrainingAuditor(weight_collapse_ratio=3.0)
        # All weights nearly equal = collapsed
        weights = {"ppo_1": 0.14, "td3_1": 0.15, "rppo_1": 0.14, "cmaes_1": 0.14}
        result = auditor.audit_generation(
            1, _make_gen_result(weights_after=weights)
        )
        collapse_alerts = [a for a in result.alerts if a.check_name == "weight_collapse"]
        assert len(collapse_alerts) > 0

    def test_persistent_collapse_critical(self):
        auditor = TrainingAuditor(weight_collapse_ratio=3.0)
        # 3 gens of collapsed weights
        for i in range(3):
            weights = {"ppo_1": 0.14, "td3_1": 0.15, "rppo_1": 0.14, "cmaes_1": 0.14}
            result = auditor.audit_generation(
                i + 1, _make_gen_result(weights_after=weights)
            )
        critical = [a for a in result.alerts if a.severity == "critical" and "weight_collapse" in a.check_name]
        assert len(critical) > 0


class TestRegimeFeedbackLoop:
    def test_risk_on_no_alert(self):
        auditor = TrainingAuditor(regime_alert_streak=3)
        for i in range(5):
            result = auditor.audit_generation(
                i + 1, _make_gen_result(), regime="risk_on"
            )
        regime_alerts = [a for a in result.alerts if a.check_name == "regime_feedback_loop"]
        assert len(regime_alerts) == 0

    def test_consecutive_risk_off_alert(self):
        auditor = TrainingAuditor(regime_alert_streak=3)
        for i in range(3):
            result = auditor.audit_generation(
                i + 1, _make_gen_result(), regime="risk_off"
            )
        regime_alerts = [a for a in result.alerts if a.check_name == "regime_feedback_loop"]
        assert len(regime_alerts) > 0
        assert regime_alerts[0].severity == "critical"

    def test_mixed_regimes_no_alert(self):
        auditor = TrainingAuditor(regime_alert_streak=3)
        regimes = ["risk_off", "risk_on", "risk_off"]
        for i, regime in enumerate(regimes):
            result = auditor.audit_generation(
                i + 1, _make_gen_result(), regime=regime
            )
        regime_alerts = [a for a in result.alerts if a.check_name == "regime_feedback_loop"]
        assert len(regime_alerts) == 0


class TestPenaltyRatcheting:
    def test_stable_params_no_alert(self):
        auditor = TrainingAuditor(penalty_ratchet_window=3)
        params = {"drawdown_penalty": 0.15, "transaction_penalty": 0.01, "holding_penalty": 0.02}
        for i in range(4):
            result = auditor.audit_generation(
                i + 1, _make_gen_result(), reward_params=params
            )
        ratchet_alerts = [a for a in result.alerts if a.check_name == "penalty_ratchet"]
        assert len(ratchet_alerts) == 0

    def test_increasing_penalty_alert(self):
        auditor = TrainingAuditor(penalty_ratchet_window=4)
        for i in range(4):
            params = {
                "drawdown_penalty": 0.15 + i * 0.05,  # Monotonically increasing
                "transaction_penalty": 0.01,
                "holding_penalty": 0.02,
            }
            result = auditor.audit_generation(
                i + 1, _make_gen_result(), reward_params=params
            )
        ratchet_alerts = [a for a in result.alerts if a.check_name == "penalty_ratchet"]
        assert len(ratchet_alerts) > 0
        assert "drawdown_penalty" in ratchet_alerts[0].message


class TestTruncationRate:
    def test_no_truncation_no_alert(self):
        auditor = TrainingAuditor()
        result = auditor.audit_generation(
            1, _make_gen_result(), truncation_rate=0.0
        )
        trunc_alerts = [a for a in result.alerts if a.check_name == "truncation_rate"]
        assert len(trunc_alerts) == 0

    def test_high_truncation_critical(self):
        auditor = TrainingAuditor(truncation_rate_critical=0.40)
        result = auditor.audit_generation(
            1, _make_gen_result(), truncation_rate=0.50
        )
        trunc_alerts = [a for a in result.alerts if a.check_name == "truncation_rate"]
        assert len(trunc_alerts) > 0
        assert trunc_alerts[0].severity == "critical"

    def test_moderate_truncation_warn(self):
        auditor = TrainingAuditor(truncation_rate_warn=0.20, truncation_rate_critical=0.40)
        result = auditor.audit_generation(
            1, _make_gen_result(), truncation_rate=0.25
        )
        trunc_alerts = [a for a in result.alerts if a.check_name == "truncation_rate"]
        assert len(trunc_alerts) > 0
        assert trunc_alerts[0].severity == "warn"


class TestReturnFloor:
    def test_before_deadline_no_alert(self):
        auditor = TrainingAuditor(return_floor_pct=0.5, return_floor_gen=15)
        result = auditor.audit_generation(
            5, _make_gen_result(best_return_pct=0.1)
        )
        floor_alerts = [a for a in result.alerts if a.check_name == "return_floor"]
        assert len(floor_alerts) == 0

    def test_after_deadline_below_floor_critical(self):
        auditor = TrainingAuditor(return_floor_pct=0.5, return_floor_gen=5)
        for i in range(5):
            result = auditor.audit_generation(
                i + 1, _make_gen_result(best_return_pct=0.1)
            )
        floor_alerts = [a for a in result.alerts if a.check_name == "return_floor"]
        assert len(floor_alerts) > 0
        assert floor_alerts[0].severity == "critical"

    def test_above_floor_no_alert(self):
        auditor = TrainingAuditor(return_floor_pct=0.5, return_floor_gen=5)
        for i in range(5):
            result = auditor.audit_generation(
                i + 1, _make_gen_result(best_return_pct=1.0)
            )
        floor_alerts = [a for a in result.alerts if a.check_name == "return_floor"]
        assert len(floor_alerts) == 0


class TestPoolDiversity:
    def test_diverse_pool_no_alert(self):
        auditor = TrainingAuditor()
        scores = {"ppo_1": 100, "td3_1": -200, "rppo_1": 50}
        result = auditor.audit_generation(
            1, _make_gen_result(eval_scores=scores)
        )
        diversity_alerts = [a for a in result.alerts if a.check_name == "pool_homogeneity"]
        assert len(diversity_alerts) == 0

    def test_homogeneous_pool_alert(self):
        auditor = TrainingAuditor()
        # All agents scoring nearly identically
        scores = {"ppo_1": -500.0, "td3_1": -500.5, "rppo_1": -499.8}
        result = auditor.audit_generation(
            1, _make_gen_result(eval_scores=scores)
        )
        diversity_alerts = [a for a in result.alerts if a.check_name == "pool_homogeneity"]
        assert len(diversity_alerts) > 0


class TestVerdictLogic:
    def test_halt_when_configured(self):
        auditor = TrainingAuditor(
            stagnation_window=3, stagnation_threshold=20.0, halt_on_critical=True
        )
        # Force a critical alert via reward regression
        rewards = [-100, -200, -500]
        for i, r in enumerate(rewards):
            result = auditor.audit_generation(i + 1, _make_gen_result(train_mean_reward=r))
        assert result.verdict == "HALT"

    def test_warn_without_halt_config(self):
        auditor = TrainingAuditor(
            stagnation_window=3, stagnation_threshold=20.0, halt_on_critical=False
        )
        rewards = [-100, -200, -500]
        for i, r in enumerate(rewards):
            result = auditor.audit_generation(i + 1, _make_gen_result(train_mean_reward=r))
        # Critical alert exists but halt_on_critical=False, so verdict is WARN
        assert result.verdict == "WARN"
        assert result.has_critical

    def test_alert_counts(self):
        auditor = TrainingAuditor(stagnation_window=2, stagnation_threshold=10.0)
        auditor.audit_generation(1, _make_gen_result(train_mean_reward=-100))
        auditor.audit_generation(2, _make_gen_result(train_mean_reward=-500))
        counts = auditor.get_alert_counts()
        assert counts["critical"] >= 1
