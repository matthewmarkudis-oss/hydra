"""CEO Interface Agent — natural language command parser for HydraCorp.

Accepts plain English commands from the CEO, interprets them into structured
intents (config changes, status queries, proposal actions), routes proposed
changes through Senior Dev review, and returns results for CLI presentation.

This agent is NOT part of the automated pipeline graph. It is used
interactively via corp/scripts/ceo_cli.py.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from corp.agents.base_corp_agent import BaseCorpAgent
from corp.agents.senior_dev import SeniorDev
from corp.state.corporation_state import CorporationState
from corp.state.decision_log import DecisionLog

logger = logging.getLogger("corp.agents.ceo_interface")

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "ceo_interface.txt"

SYSTEM_PROMPT = ""
if _PROMPT_PATH.exists():
    SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")


class CEOInterface(BaseCorpAgent):
    """CEO Interface — parses natural language commands into structured actions.

    Uses Claude Sonnet for command parsing with a rule-based regex fallback
    when the LLM is unavailable. Every config change is routed through
    SeniorDev._review_patch() before being presented to the CEO.
    """

    def __init__(
        self,
        state: CorporationState,
        decision_log: DecisionLog,
        senior_dev: SeniorDev | None = None,
    ):
        super().__init__("ceo_interface", state, decision_log)
        self.senior_dev = senior_dev

    # ------------------------------------------------------------------
    # BaseCorpAgent interface
    # ------------------------------------------------------------------

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Not used in the automated pipeline. See process_command()."""
        return {"info": "CEOInterface is interactive-only. Use process_command()."}

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def process_command(
        self,
        command: str,
        config: Any,  # HydraConfig
    ) -> dict[str, Any]:
        """Parse a CEO command and return a structured result.

        Returns:
            {
                "intent": str,
                "patch": dict | None,
                "warnings": list[str],
                "needs_confirmation": bool,
                "response_text": str,
                "senior_dev_verdict": str,
            }
        """
        command = command.strip()
        if not command:
            return self._result("unknown", response_text="Empty command.")

        config_dict = config.model_dump()

        # --- Status queries (fast path, no LLM needed) ---
        if self._is_status_query(command):
            return self._handle_status_query(command, config, config_dict)

        # --- Proposal actions (fast path) ---
        proposal_match = self._is_proposal_action(command)
        if proposal_match:
            return self._handle_proposal_action(proposal_match, config)

        # --- Config / ticker changes: LLM parse, then regex fallback ---
        parsed = self._llm_parse(command, config_dict)
        if parsed is None:
            parsed = self._regex_parse(command, config_dict)

        if parsed["intent"] == "unknown":
            return self._result(
                "unknown",
                response_text=(
                    f"I didn't understand: \"{command}\"\n"
                    "Try: add AAPL, tighten risk, show status, show proposals, "
                    "set max_position_pct to 0.25"
                ),
            )

        patch = parsed.get("patch")
        if not patch:
            return self._result(
                parsed["intent"],
                response_text=parsed.get("explanation", "No changes needed."),
            )

        # --- Validate patch against Pydantic schema ---
        try:
            new_config = config.apply_patch(patch)
        except Exception as e:
            return self._result(
                parsed["intent"],
                response_text=f"Invalid config change: {e}",
                warnings=[str(e)],
            )

        # --- Build diff text ---
        diff_lines = self._build_diff(config_dict, patch)

        # --- Senior Dev review ---
        warnings = list(parsed.get("risks", []))
        senior_verdict = "No Senior Dev available"
        if self.senior_dev:
            patch_issues = self.senior_dev._review_patch(patch, config_dict)
            warnings.extend(patch_issues)
            senior_verdict = (
                "Approved (no safety violations)"
                if not patch_issues
                else f"Concerns: {'; '.join(patch_issues)}"
            )

        # --- Ticker change warning ---
        if parsed["intent"] == "ticker_change":
            warnings.append(
                "Adding/removing a ticker changes observation dimensions (17N+5). "
                "All RL agents must be retrained. Rule-based agents adapt automatically."
            )

        # --- Log the command ---
        self.log_decision(
            "ceo_command",
            detail={
                "command": command,
                "intent": parsed["intent"],
                "patch": patch,
                "warnings": warnings,
            },
            outcome="pending_approval",
        )

        return self._result(
            intent=parsed["intent"],
            patch=patch,
            warnings=warnings,
            needs_confirmation=True,
            response_text=parsed.get("explanation", ""),
            diff_lines=diff_lines,
            senior_dev_verdict=senior_verdict,
        )

    # ------------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------------

    @staticmethod
    def _is_status_query(cmd: str) -> bool:
        lower = cmd.lower()
        patterns = [
            "status", "how are we", "how's it going", "what's happening",
            "show best", "best agent", "market regime", "regime",
            "show config", "current config", "what would happen",
            "portfolio", "performance",
        ]
        return any(p in lower for p in patterns)

    def _handle_status_query(
        self,
        command: str,
        config: Any,
        config_dict: dict,
    ) -> dict[str, Any]:
        corp_state = self.state.get_full_state()
        lower = command.lower()

        # Show current config
        if "config" in lower:
            return self._handle_config_query(config_dict, lower)

        # General status
        last_result = corp_state.get("last_pipeline_result") or {}
        regime = corp_state.get("regime", {})
        pending = self.state.get_pending_proposals()
        history = corp_state.get("agent_performance_history", [])

        lines = []
        lines.append(f"Pipeline Runs: {corp_state.get('pipeline_run_count', 0)}")

        if last_result:
            lines.append(f"Best Agent: {last_result.get('best_agent', 'N/A')}")
            best_ret = last_result.get("best_return")
            if best_ret is not None:
                lines.append(f"Best Return: {best_ret:+.4f}")
            passed = last_result.get("passed_count", 0)
            total = last_result.get("total_agents", 0)
            lines.append(f"Passed Validation: {passed}/{total}")
        else:
            lines.append("No pipeline results yet.")

        regime_class = regime.get("classification", "unknown")
        confidence = regime.get("confidence")
        regime_str = regime_class
        if confidence is not None:
            regime_str += f" ({confidence:.0%} confidence)"
        lines.append(f"Market Regime: {regime_str}")

        lines.append(f"Pending Proposals: {len(pending)}")

        tickers = config_dict.get("data", {}).get("tickers", [])
        lines.append(f"Tickers ({len(tickers)}): {', '.join(tickers)}")

        if history:
            recent = history[-1]
            lines.append(
                f"Last Run (#{recent.get('run', '?')}): "
                f"return={recent.get('best_return', 0):+.4f}, "
                f"passed={recent.get('passed_count', 0)}"
            )

        self.log_decision("ceo_command", detail={"command": command, "intent": "status_query"}, outcome="displayed")
        return self._result("status_query", response_text="\n".join(lines))

    def _handle_config_query(self, config_dict: dict, lower: str) -> dict[str, Any]:
        """Show current config sections."""
        lines = []
        # Determine which sections to show
        sections = ["env", "reward", "training", "data", "pool", "validation"]
        for section_name in ("env", "reward", "training", "data", "pool", "compute", "validation"):
            if section_name in lower:
                sections = [section_name]
                break

        for section_name in sections:
            section = config_dict.get(section_name, {})
            if section:
                lines.append(f"\n  [{section_name}]")
                for k, v in section.items():
                    if isinstance(v, list) and len(v) > 5:
                        lines.append(f"    {k}: [{len(v)} items]")
                    else:
                        lines.append(f"    {k}: {v}")
        return self._result("status_query", response_text="\n".join(lines))

    # ------------------------------------------------------------------
    # Proposal actions
    # ------------------------------------------------------------------

    @staticmethod
    def _is_proposal_action(cmd: str) -> dict | None:
        lower = cmd.lower().strip()
        if re.match(r"(show|list)\s+proposals?", lower):
            return {"action": "list"}
        m = re.match(r"approve\s+(?:proposal\s+)?#?(\d+)", lower)
        if m:
            return {"action": "approve", "index": int(m.group(1))}
        m = re.match(r"reject\s+(?:proposal\s+)?#?(\d+)", lower)
        if m:
            return {"action": "reject", "index": int(m.group(1))}
        if "reject all" in lower:
            return {"action": "reject_all"}
        return None

    def _handle_proposal_action(
        self,
        match: dict,
        config: Any,
    ) -> dict[str, Any]:
        action = match["action"]
        pending = self.state.get_pending_proposals()

        if action == "list":
            if not pending:
                return self._result("proposal_action", response_text="No pending proposals.")
            lines = []
            for i, p in enumerate(pending):
                lines.append(
                    f"  #{i + 1} [{p.get('status', 'pending').upper()}] "
                    f"From: {p.get('source', 'unknown')}"
                )
                if p.get("memo"):
                    lines.append(f"     Memo: \"{p['memo']}\"")
                if p.get("patch"):
                    for section, values in p["patch"].items():
                        if isinstance(values, dict):
                            for k, v in values.items():
                                lines.append(f"     Patch: {section}.{k}: {v}")
                        else:
                            lines.append(f"     Patch: {section}: {values}")
                conf = p.get("confidence")
                risk = p.get("risk")
                if conf is not None or risk:
                    parts = []
                    if conf is not None:
                        parts.append(f"Confidence: {conf:.0%}")
                    if risk:
                        parts.append(f"Risk: {risk}")
                    lines.append(f"     {' | '.join(parts)}")
            return self._result("proposal_action", response_text="\n".join(lines))

        if action == "approve":
            idx = match["index"] - 1  # 1-based to 0-based
            all_proposals = self.state._read_state()["proposals"]
            # Map pending index to absolute index
            pending_indices = [
                i for i, p in enumerate(all_proposals) if p.get("status") == "pending"
            ]
            if idx < 0 or idx >= len(pending_indices):
                return self._result(
                    "proposal_action",
                    response_text=f"Proposal #{match['index']} not found. Use 'show proposals' to list.",
                )
            abs_idx = pending_indices[idx]
            proposal = all_proposals[abs_idx]
            patch = proposal.get("patch")

            # Apply patch to config
            if patch:
                try:
                    new_config = config.apply_patch(patch)
                except Exception as e:
                    return self._result(
                        "proposal_action",
                        response_text=f"Cannot apply proposal: {e}",
                        warnings=[str(e)],
                    )

            self.state.resolve_proposal(abs_idx, "approved", reason="CEO approved")
            self.log_decision(
                "ceo_proposal_approval",
                detail={"proposal_index": abs_idx, "patch": patch},
                outcome="approved",
            )
            return self._result(
                "proposal_action",
                patch=patch,
                needs_confirmation=False,
                response_text=f"Proposal #{match['index']} approved and applied to config.",
            )

        if action == "reject":
            idx = match["index"] - 1
            all_proposals = self.state._read_state()["proposals"]
            pending_indices = [
                i for i, p in enumerate(all_proposals) if p.get("status") == "pending"
            ]
            if idx < 0 or idx >= len(pending_indices):
                return self._result(
                    "proposal_action",
                    response_text=f"Proposal #{match['index']} not found.",
                )
            abs_idx = pending_indices[idx]
            self.state.resolve_proposal(abs_idx, "rejected", reason="CEO rejected")
            self.log_decision(
                "ceo_proposal_rejection",
                detail={"proposal_index": abs_idx},
                outcome="rejected",
            )
            return self._result(
                "proposal_action",
                response_text=f"Proposal #{match['index']} rejected.",
            )

        if action == "reject_all":
            all_proposals = self.state._read_state()["proposals"]
            count = 0
            for i, p in enumerate(all_proposals):
                if p.get("status") == "pending":
                    self.state.resolve_proposal(i, "rejected", reason="CEO rejected all")
                    count += 1
            self.log_decision("ceo_reject_all", detail={"count": count}, outcome="rejected")
            return self._result(
                "proposal_action",
                response_text=f"Rejected {count} pending proposal(s).",
            )

        return self._result("unknown", response_text="Unknown proposal action.")

    # ------------------------------------------------------------------
    # LLM-based command parsing
    # ------------------------------------------------------------------

    def _llm_parse(self, command: str, config_dict: dict) -> dict | None:
        """Use LLM to parse the command into structured intent (Groq free tier or Anthropic)."""
        if not SYSTEM_PROMPT:
            logger.debug("No system prompt loaded, using regex fallback")
            return None

        from corp.llm_client import call_llm_json

        # Include current tickers in the prompt so LLM can build correct list
        tickers = config_dict.get("data", {}).get("tickers", [])
        user_content = (
            f"Current tickers: {json.dumps(tickers)}\n"
            f"Current config summary:\n"
            f"  env.max_position_pct: {config_dict.get('env', {}).get('max_position_pct')}\n"
            f"  env.max_drawdown_pct: {config_dict.get('env', {}).get('max_drawdown_pct')}\n"
            f"  reward.transaction_penalty: {config_dict.get('reward', {}).get('transaction_penalty')}\n"
            f"  reward.drawdown_penalty: {config_dict.get('reward', {}).get('drawdown_penalty')}\n"
            f"  reward.reward_scale: {config_dict.get('reward', {}).get('reward_scale')}\n"
            f"  training.num_generations: {config_dict.get('training', {}).get('num_generations')}\n"
            f"\nCEO command: {command}"
        )

        parsed = call_llm_json(SYSTEM_PROMPT, user_content, max_tokens=1500, temperature=0.2)
        if parsed is None:
            return None

        return {
            "intent": parsed.get("intent", "unknown"),
            "patch": parsed.get("patch"),
            "explanation": parsed.get("explanation", ""),
            "risks": parsed.get("risks", []),
        }

    # ------------------------------------------------------------------
    # Regex-based fallback parsing
    # ------------------------------------------------------------------

    def _regex_parse(self, command: str, config_dict: dict) -> dict:
        """Rule-based pattern matching when LLM is unavailable."""
        lower = command.lower().strip()
        tickers = list(config_dict.get("data", {}).get("tickers", []))

        # --- add TICKER ---
        m = re.match(r"add\s+([A-Za-z]{1,5})(?:\s+to\s+.*)?$", command.strip(), re.IGNORECASE)
        if m:
            ticker = m.group(1).upper()
            if ticker in tickers:
                return {
                    "intent": "ticker_change",
                    "patch": None,
                    "explanation": f"{ticker} is already in the ticker list.",
                    "risks": [],
                }
            new_tickers = tickers + [ticker]
            return {
                "intent": "ticker_change",
                "patch": {"data": {"tickers": new_tickers}},
                "explanation": f"Adding {ticker} to ticker list ({len(new_tickers)} stocks, was {len(tickers)}).",
                "risks": [],
            }

        # --- remove/drop TICKER ---
        m = re.match(r"(?:remove|drop)\s+([A-Za-z]{1,5})(?:\s+from\s+.*)?$", command.strip(), re.IGNORECASE)
        if m:
            ticker = m.group(1).upper()
            if ticker not in tickers:
                return {
                    "intent": "ticker_change",
                    "patch": None,
                    "explanation": f"{ticker} is not in the ticker list.",
                    "risks": [],
                }
            new_tickers = [t for t in tickers if t != ticker]
            return {
                "intent": "ticker_change",
                "patch": {"data": {"tickers": new_tickers}},
                "explanation": f"Removing {ticker} from ticker list ({len(new_tickers)} stocks, was {len(tickers)}).",
                "risks": [],
            }

        # --- set FIELD to VALUE ---
        m = re.match(r"set\s+([\w.]+)\s+to\s+(.+)$", lower)
        if m:
            field_path = m.group(1)
            raw_value = m.group(2).strip()
            return self._parse_set_command(field_path, raw_value, config_dict)

        # --- tighten risk ---
        if "tighten" in lower and "risk" in lower:
            env = config_dict.get("env", {})
            cur_pos = env.get("max_position_pct", 0.30)
            cur_dd = env.get("max_drawdown_pct", 0.20)
            new_pos = round(max(cur_pos * 0.75, 0.05), 3)
            new_dd = round(max(cur_dd * 0.75, 0.05), 3)
            return {
                "intent": "config_change",
                "patch": {"env": {"max_position_pct": new_pos, "max_drawdown_pct": new_dd}},
                "explanation": (
                    f"Tightening risk: max_position_pct {cur_pos} -> {new_pos}, "
                    f"max_drawdown_pct {cur_dd} -> {new_dd}."
                ),
                "risks": ["Tighter limits may reduce returns."],
            }

        # --- more aggressive / be aggressive ---
        if "aggressive" in lower or "looser" in lower:
            env = config_dict.get("env", {})
            reward = config_dict.get("reward", {})
            cur_pos = env.get("max_position_pct", 0.30)
            new_pos = round(min(cur_pos * 1.3, 0.60), 3)
            cur_tp = reward.get("transaction_penalty", 0.1)
            new_tp = round(max(cur_tp * 0.7, 0.005), 4)
            return {
                "intent": "config_change",
                "patch": {
                    "env": {"max_position_pct": new_pos},
                    "reward": {"transaction_penalty": new_tp},
                },
                "explanation": (
                    f"More aggressive: max_position_pct {cur_pos} -> {new_pos}, "
                    f"transaction_penalty {cur_tp} -> {new_tp}."
                ),
                "risks": [
                    "Higher position concentration increases single-stock risk.",
                    "Lower transaction penalty may lead to overtrading.",
                ],
            }

        # --- run N generations ---
        m = re.match(r"(?:run|train)\s+(\d+)\s+generations?", lower)
        if m:
            n = int(m.group(1))
            return {
                "intent": "config_change",
                "patch": {"training": {"num_generations": n}},
                "explanation": f"Setting training to {n} generations.",
                "risks": [] if n <= 100 else ["Large generation counts increase compute time."],
            }

        # --- increase/decrease learning rate ---
        m = re.match(r"(increase|decrease|raise|lower)\s+learning\s*rate", lower)
        if m:
            direction = m.group(1)
            agents = config_dict.get("pool", {}).get("agents", [])
            if agents:
                cur_lr = agents[0].get("learning_rate", 3e-4)
                if direction in ("increase", "raise"):
                    new_lr = cur_lr * 2
                else:
                    new_lr = cur_lr / 2
                return {
                    "intent": "config_change",
                    "patch": {"pool": {"agents": [{"learning_rate": new_lr}]}},
                    "explanation": f"{'Increasing' if direction in ('increase', 'raise') else 'Decreasing'} learning rate from {cur_lr:.1e} to {new_lr:.1e} for the first agent.",
                    "risks": ["Learning rate changes affect all future training runs."],
                }

        # --- switch to SECTOR stocks ---
        m = re.match(r"switch\s+to\s+(\w+)\s+stocks?", lower)
        if m:
            sector = m.group(1).lower()
            sector_map = {
                "tech": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "AMZN", "CRM", "ORCL", "INTC"],
                "finance": ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW", "AXP", "V"],
                "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "OXY", "VLO", "MPC", "PSX", "HAL"],
                "crypto": ["COIN", "MARA", "RIOT", "MSTR", "CLSK", "BITF", "HUT", "BTBT", "SOS", "EBON"],
                "healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "BMY", "AMGN"],
            }
            if sector in sector_map:
                new_tickers = sector_map[sector]
                return {
                    "intent": "ticker_change",
                    "patch": {"data": {"tickers": new_tickers}},
                    "explanation": f"Switching to {sector} stocks: {', '.join(new_tickers)}.",
                    "risks": ["Complete ticker list change requires full retraining."],
                }

        return {"intent": "unknown", "patch": None, "explanation": "", "risks": []}

    def _parse_set_command(
        self,
        field_path: str,
        raw_value: str,
        config_dict: dict,
    ) -> dict:
        """Parse 'set field.name to value' into a patch."""
        # Supported dotted paths
        parts = field_path.split(".")
        if len(parts) == 2:
            section, field = parts
        elif len(parts) == 1:
            # Try to guess the section
            field = parts[0]
            section = self._guess_section(field, config_dict)
            if section is None:
                return {
                    "intent": "unknown",
                    "patch": None,
                    "explanation": f"Unknown field: {field_path}",
                    "risks": [],
                }
        else:
            return {
                "intent": "unknown",
                "patch": None,
                "explanation": f"Unsupported field path: {field_path}",
                "risks": [],
            }

        # Coerce value
        value = self._coerce_value(raw_value)

        return {
            "intent": "config_change",
            "patch": {section: {field: value}},
            "explanation": f"Setting {section}.{field} to {value}.",
            "risks": [],
        }

    @staticmethod
    def _guess_section(field: str, config_dict: dict) -> str | None:
        """Guess which config section a field belongs to."""
        for section in ("env", "reward", "training", "data", "pool", "compute", "validation"):
            if field in config_dict.get(section, {}):
                return section
        return None

    @staticmethod
    def _coerce_value(raw: str) -> Any:
        """Coerce a string value to the appropriate Python type."""
        raw = raw.strip().strip('"').strip("'")
        if raw.lower() in ("true", "yes"):
            return True
        if raw.lower() in ("false", "no"):
            return False
        try:
            if "." in raw:
                return float(raw)
            return int(raw)
        except ValueError:
            return raw

    # ------------------------------------------------------------------
    # Diff builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_diff(config_dict: dict, patch: dict) -> list[str]:
        """Build human-readable diff lines for a patch."""
        lines = []
        for section, values in patch.items():
            if not isinstance(values, dict):
                old = config_dict.get(section)
                lines.append(f"  {section}: {old} -> {values}")
                continue
            for field, new_val in values.items():
                old_section = config_dict.get(section, {})
                old_val = old_section.get(field, "<unset>")
                if isinstance(new_val, list) and isinstance(old_val, list):
                    added = set(new_val) - set(old_val)
                    removed = set(old_val) - set(new_val)
                    desc = f"{len(new_val)} items (was {len(old_val)})"
                    if added:
                        desc += f", +{added}"
                    if removed:
                        desc += f", -{removed}"
                    lines.append(f"  {section}.{field}: {desc}")
                else:
                    lines.append(f"  {section}.{field}: {old_val} -> {new_val}")
        return lines

    # ------------------------------------------------------------------
    # Result builder
    # ------------------------------------------------------------------

    @staticmethod
    def _result(
        intent: str,
        patch: dict | None = None,
        warnings: list[str] | None = None,
        needs_confirmation: bool = False,
        response_text: str = "",
        diff_lines: list[str] | None = None,
        senior_dev_verdict: str = "",
    ) -> dict[str, Any]:
        return {
            "intent": intent,
            "patch": patch,
            "warnings": warnings or [],
            "needs_confirmation": needs_confirmation,
            "response_text": response_text,
            "diff_lines": diff_lines or [],
            "senior_dev_verdict": senior_dev_verdict,
        }
