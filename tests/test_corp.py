"""Tests for the HydraCorp corporation layer.

Validates that all agents instantiate, run, and communicate correctly
using in-memory state (no file persistence needed for tests).

Backtesting and training only.
"""

import json
import tempfile
from pathlib import Path

import pytest

# --- State layer tests ---

class TestCorporationState:
    def test_create_and_read(self, tmp_path):
        from corp.state.corporation_state import CorporationState
        state_file = str(tmp_path / "test_state.json")
        state = CorporationState(state_file=state_file)
        full = state.get_full_state()
        assert "messages" in full
        assert "proposals" in full
        assert full["pipeline_run_count"] == 0

    def test_post_message(self, tmp_path):
        from corp.state.corporation_state import CorporationState, CorpMessage
        state = CorporationState(state_file=str(tmp_path / "s.json"))
        msg = CorpMessage(sender="test", recipient="chief", msg_type="report", priority=3)
        state.post_message(msg)
        msgs = state.get_messages()
        assert len(msgs) == 1
        assert msgs[0]["sender"] == "test"

    def test_record_pipeline_result(self, tmp_path):
        from corp.state.corporation_state import CorporationState
        state = CorporationState(state_file=str(tmp_path / "s.json"))
        state.record_pipeline_result({"best_return": 0.05, "best_agent": "ppo"})
        full = state.get_full_state()
        assert full["pipeline_run_count"] == 1
        assert full["last_pipeline_result"]["best_return"] == 0.05

    def test_shadow_tracking(self, tmp_path):
        from corp.state.corporation_state import CorporationState
        state = CorporationState(state_file=str(tmp_path / "s.json"))
        # 3 consecutive shadow wins
        state.record_shadow_result(0.05, 0.08)
        state.record_shadow_result(0.05, 0.07)
        state.record_shadow_result(0.05, 0.06)
        assert state.should_promote_shadow(required_wins=3)

    def test_shadow_reset_on_loss(self, tmp_path):
        from corp.state.corporation_state import CorporationState
        state = CorporationState(state_file=str(tmp_path / "s.json"))
        state.record_shadow_result(0.05, 0.08)
        state.record_shadow_result(0.05, 0.07)
        state.record_shadow_result(0.08, 0.03)  # shadow loses
        assert not state.should_promote_shadow(required_wins=3)


class TestConfigBlacklist:
    def test_add_and_check(self, tmp_path):
        from corp.state.config_blacklist import ConfigBlacklist
        bl = ConfigBlacklist(str(tmp_path / "bl.json"))
        config = {"env": {"num_stocks": 10}, "reward": {"reward_scale": 100}}
        bl.add(config, reason="test failure", metrics={"fitness": -0.5})

        is_blocked, reason = bl.is_blacklisted(config)
        assert is_blocked
        assert "test failure" in reason

    def test_different_config_not_blocked(self, tmp_path):
        from corp.state.config_blacklist import ConfigBlacklist
        bl = ConfigBlacklist(str(tmp_path / "bl.json"))
        config1 = {"env": {"num_stocks": 10}}
        config2 = {"env": {"num_stocks": 20}}
        bl.add(config1, reason="bad")

        is_blocked, _ = bl.is_blacklisted(config2)
        assert not is_blocked

    def test_remove(self, tmp_path):
        from corp.state.config_blacklist import ConfigBlacklist
        bl = ConfigBlacklist(str(tmp_path / "bl.json"))
        config = {"env": {"num_stocks": 5}}
        h = bl.add(config, reason="test")
        assert bl.is_blacklisted(config)[0]
        bl.remove(h)
        assert not bl.is_blacklisted(config)[0]


class TestDecisionLog:
    def test_log_and_retrieve(self, tmp_path):
        from corp.state.decision_log import DecisionLog
        dl = DecisionLog(str(tmp_path / "log.jsonl"))
        dl.log(agent="test_agent", action="test_action", detail={"key": "val"})
        recent = dl.get_recent(limit=10)
        assert len(recent) == 1
        assert recent[0]["agent"] == "test_agent"


# --- Agent tests ---

class TestSeniorDev:
    def test_passes_clean_config(self, tmp_path):
        from corp.state.corporation_state import CorporationState
        from corp.state.config_blacklist import ConfigBlacklist
        from corp.state.decision_log import DecisionLog
        from corp.agents.senior_dev import SeniorDev

        state = CorporationState(str(tmp_path / "s.json"))
        bl = ConfigBlacklist(str(tmp_path / "bl.json"))
        dl = DecisionLog(str(tmp_path / "dl.jsonl"))

        agent = SeniorDev(state=state, decision_log=dl, blacklist=bl)
        result = agent.run({"config_dict": {"env": {"num_stocks": 10}}})
        assert result["checks_passed"] is True

    def test_blocks_blacklisted_config(self, tmp_path):
        from corp.state.corporation_state import CorporationState
        from corp.state.config_blacklist import ConfigBlacklist
        from corp.state.decision_log import DecisionLog
        from corp.agents.senior_dev import SeniorDev

        state = CorporationState(str(tmp_path / "s.json"))
        bl = ConfigBlacklist(str(tmp_path / "bl.json"))
        dl = DecisionLog(str(tmp_path / "dl.jsonl"))

        config = {"env": {"num_stocks": 10}, "reward": {"reward_scale": 100}}
        bl.add(config, reason="known bad config")

        agent = SeniorDev(state=state, decision_log=dl, blacklist=bl)
        result = agent.run({"config_dict": config})
        assert result["checks_passed"] is False
        assert len(result["vetoes"]) > 0

    def test_reviews_patch_extreme_values(self, tmp_path):
        from corp.state.corporation_state import CorporationState
        from corp.state.config_blacklist import ConfigBlacklist
        from corp.state.decision_log import DecisionLog
        from corp.agents.senior_dev import SeniorDev

        state = CorporationState(str(tmp_path / "s.json"))
        bl = ConfigBlacklist(str(tmp_path / "bl.json"))
        dl = DecisionLog(str(tmp_path / "dl.jsonl"))

        agent = SeniorDev(state=state, decision_log=dl, blacklist=bl)
        result = agent.run({
            "config_dict": {"env": {}, "reward": {}},
            "proposed_patch": {
                "reward": {"transaction_penalty": 0.001, "reward_scale": 600},
                "env": {"max_position_pct": 0.80},
            },
        })
        assert result["checks_passed"] is True  # Passes but with warnings
        assert len(result["warnings"]) >= 2  # Should flag at least 2 issues


class TestHardwareOptimizer:
    def test_runs_without_error(self, tmp_path):
        from corp.state.corporation_state import CorporationState
        from corp.state.decision_log import DecisionLog
        from corp.agents.hardware_optimizer import HardwareOptimizer

        state = CorporationState(str(tmp_path / "s.json"))
        dl = DecisionLog(str(tmp_path / "dl.jsonl"))

        agent = HardwareOptimizer(state=state, decision_log=dl)
        result = agent.run({})
        assert "hardware" in result
        assert "recommendations" in result
        assert result["hardware"]["cpu_count"] >= 1


class TestShadowTrader:
    def test_no_shadow_config(self, tmp_path):
        from corp.state.corporation_state import CorporationState
        from corp.state.decision_log import DecisionLog
        from corp.agents.shadow_trader import ShadowTrader

        state = CorporationState(str(tmp_path / "s.json"))
        dl = DecisionLog(str(tmp_path / "dl.jsonl"))

        agent = ShadowTrader(state=state, decision_log=dl)
        assert not agent.has_shadow()
        result = agent.run({})
        assert result["status"] == "no_shadow_config"

    def test_set_shadow(self, tmp_path):
        from corp.state.corporation_state import CorporationState
        from corp.state.decision_log import DecisionLog
        from corp.agents.shadow_trader import ShadowTrader

        state = CorporationState(str(tmp_path / "s.json"))
        dl = DecisionLog(str(tmp_path / "dl.jsonl"))

        agent = ShadowTrader(state=state, decision_log=dl)
        agent.set_shadow_config({"env": {"num_stocks": 10}})
        assert agent.has_shadow()
        agent.clear_shadow()
        assert not agent.has_shadow()


class TestHedgeFundDirector:
    def test_rule_based_fallback(self, tmp_path):
        """Without API key, should fall back to rule-based analysis."""
        from corp.state.corporation_state import CorporationState
        from corp.state.decision_log import DecisionLog
        from corp.agents.hedge_fund_director import HedgeFundDirector

        state = CorporationState(str(tmp_path / "s.json"))
        dl = DecisionLog(str(tmp_path / "dl.jsonl"))

        agent = HedgeFundDirector(state=state, decision_log=dl)
        result = agent.run({
            "pipeline_results": {
                "best_return": -0.03,
                "passed_count": 0,
                "total_agents": 6,
                "excess_return": -0.05,
            },
            "config_dict": {
                "reward": {"transaction_penalty": 0.03, "reward_scale": 100},
                "env": {"max_drawdown_pct": 0.25},
            },
        })
        assert result["llm_used"] is False
        assert result["memo"] != ""

    def test_no_pipeline_results(self, tmp_path):
        from corp.state.corporation_state import CorporationState
        from corp.state.decision_log import DecisionLog
        from corp.agents.hedge_fund_director import HedgeFundDirector

        state = CorporationState(str(tmp_path / "s.json"))
        dl = DecisionLog(str(tmp_path / "dl.jsonl"))

        agent = HedgeFundDirector(state=state, decision_log=dl)
        result = agent.run({})
        assert "No pipeline results" in result["memo"]


class TestContrarian:
    def test_does_not_fire_on_bad_results(self, tmp_path):
        from corp.state.corporation_state import CorporationState
        from corp.state.decision_log import DecisionLog
        from corp.agents.contrarian import Contrarian

        state = CorporationState(str(tmp_path / "s.json"))
        dl = DecisionLog(str(tmp_path / "dl.jsonl"))

        agent = Contrarian(state=state, decision_log=dl)
        result = agent.run({
            "pipeline_results": {"best_return": 0.01, "passed_count": 0},
        })
        assert result["fired"] is False

    def test_fires_on_good_results(self, tmp_path):
        from corp.state.corporation_state import CorporationState
        from corp.state.decision_log import DecisionLog
        from corp.agents.contrarian import Contrarian

        state = CorporationState(str(tmp_path / "s.json"))
        dl = DecisionLog(str(tmp_path / "dl.jsonl"))

        agent = Contrarian(state=state, decision_log=dl)
        result = agent.run({
            "pipeline_results": {"best_return": 0.15, "passed_count": 2, "total_agents": 6},
            "config_dict": {"reward": {"transaction_penalty": 0.005}, "env": {"max_drawdown_pct": 0.25}},
        })
        assert result["fired"] is True
        assert len(result["concerns"]) > 0
        assert result["verdict"] in ("fragile", "antifragile", "inconclusive")


class TestGeopoliticsExpert:
    def test_runs_with_no_api_keys(self, tmp_path):
        from corp.state.corporation_state import CorporationState
        from corp.state.decision_log import DecisionLog
        from corp.agents.geopolitics_expert import GeopoliticsExpert

        state = CorporationState(str(tmp_path / "s.json"))
        dl = DecisionLog(str(tmp_path / "dl.jsonl"))

        agent = GeopoliticsExpert(state=state, decision_log=dl)
        result = agent.run({"force": True})
        # Without API keys, should still return a valid result
        assert result["regime"] in ("risk_on", "risk_off", "crisis", "unknown")


class TestInnovationScout:
    def test_static_recommendations(self, tmp_path):
        from corp.state.corporation_state import CorporationState
        from corp.state.decision_log import DecisionLog
        from corp.agents.innovation_scout import InnovationScout

        state = CorporationState(str(tmp_path / "s.json"))
        dl = DecisionLog(str(tmp_path / "dl.jsonl"))

        agent = InnovationScout(state=state, decision_log=dl)
        result = agent.run({"force": True})
        # Without API key, should return static recommendations
        assert result["new_discoveries"] > 0
        assert result["briefs"][0]["tool_name"] != ""


# --- Graph tests ---

class TestCorpGraph:
    def test_graph_executes(self, tmp_path):
        from corp.state.corporation_state import CorporationState
        from corp.state.config_blacklist import ConfigBlacklist
        from corp.state.decision_log import DecisionLog
        from corp.scripts.run_corporation import build_all_agents
        from corp.graph.corporation_graph import build_corporation_graph

        state = CorporationState(str(tmp_path / "s.json"))
        bl = ConfigBlacklist(str(tmp_path / "bl.json"))
        dl = DecisionLog(str(tmp_path / "dl.jsonl"))

        agents = build_all_agents(state, dl, bl)
        graph = build_corporation_graph(agents)

        result = graph.execute({
            "config_dict": {"env": {"num_stocks": 3}, "reward": {"reward_scale": 100}},
            "skip_pipeline": True,
            "force_all_agents": True,
            "alerts": [],
        })

        assert result["cycle_complete"] is True
        assert "blacklist_check" in result

    def test_graph_blocks_blacklisted(self, tmp_path):
        from corp.state.corporation_state import CorporationState
        from corp.state.config_blacklist import ConfigBlacklist
        from corp.state.decision_log import DecisionLog
        from corp.scripts.run_corporation import build_all_agents
        from corp.graph.corporation_graph import build_corporation_graph

        state = CorporationState(str(tmp_path / "s.json"))
        bl = ConfigBlacklist(str(tmp_path / "bl.json"))
        dl = DecisionLog(str(tmp_path / "dl.jsonl"))

        config = {"env": {"num_stocks": 3}, "reward": {"reward_scale": 100}}
        bl.add(config, reason="test block")

        agents = build_all_agents(state, dl, bl)
        graph = build_corporation_graph(agents)

        result = graph.execute({
            "config_dict": config,
            "skip_pipeline": True,
            "alerts": [],
        })

        assert result["blacklist_check"]["blocked"] is True
