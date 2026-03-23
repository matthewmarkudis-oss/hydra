"""CEO CLI — Agentic interactive terminal for HydraCorp.

Two modes:
  Default  — full agentic interface with Claude Sonnet, streaming, tool use,
             web search, training analysis, and conversational interaction.
  --simple — lightweight regex-based REPL (no API key needed).

Usage:
    python corp/scripts/ceo_cli.py [--config path/to/config.yaml] [--simple]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
from pathlib import Path

# Add project root to path
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))

# Load API keys from .env
_env_path = _ROOT.parent / "trading_agents" / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _, _v = _line.partition("=")
            _k, _v = _k.strip(), _v.strip()
            if _k and _v and _k not in os.environ:
                os.environ[_k] = _v

from hydra.config.schema import HydraConfig
from corp.state.corporation_state import CorporationState
from corp.state.config_blacklist import ConfigBlacklist
from corp.state.decision_log import DecisionLog
from corp.agents.senior_dev import SeniorDev
from corp.agents.ceo_interface import CEOInterface
from corp.agents.generation_scorer import score_generation, format_scorecard

# ═══════════════════════════════════════════════════════════════════════════
# ANSI helpers
# ═══════════════════════════════════════════════════════════════════════════

BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def _c(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


# ═══════════════════════════════════════════════════════════════════════════
# Shared state — initialized in main(), used by tool implementations
# ═══════════════════════════════════════════════════════════════════════════

_config: HydraConfig = None  # type: ignore
_config_path: Path = None  # type: ignore
_state: CorporationState = None  # type: ignore
_decision_log: DecisionLog = None  # type: ignore
_senior_dev: SeniorDev = None  # type: ignore
_ceo_agent: CEOInterface = None  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════
# Tool definitions for the agentic loop
# ═══════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "name": "get_system_status",
        "description": (
            "Get current HydraCorp system status: pipeline runs, best agent, "
            "market regime, training progress, pending proposals, ticker list."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_training_detail",
        "description": (
            "Get detailed training data for recent generations: per-agent "
            "eval scores, promotions, demotions, conviction win rates, "
            "weight convergence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "last_n": {
                    "type": "integer",
                    "description": "Number of recent generations to return (default 5).",
                }
            },
        },
    },
    {
        "name": "get_generation_scorecard",
        "description": (
            "Get a structured scorecard evaluating a training generation "
            "across 6 dimensions (reward_trend, top_agent_quality, "
            "pool_diversity, positive_rate, conviction_strength, stability). "
            "Returns overall 1-10 score and CONTINUE/RETUNE/HALT verdict."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "generation": {
                    "type": "integer",
                    "description": "Generation number. 0 or omit for latest.",
                }
            },
        },
    },
    {
        "name": "get_config",
        "description": (
            "Get the current HydraConfig. Optionally filter to one section: "
            "env, reward, training, data, pool, compute, validation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "description": "Config section name, or omit for all.",
                }
            },
        },
    },
    {
        "name": "list_proposals",
        "description": "List all pending config change proposals from HydraCorp agents.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "propose_config_change",
        "description": (
            "Propose a config change. Validates against Pydantic schema and "
            "runs Senior Dev safety review. Does NOT apply — CEO must approve."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patch": {
                    "type": "object",
                    "description": 'Config patch, e.g. {"env": {"max_position_pct": 0.25}}',
                },
                "explanation": {
                    "type": "string",
                    "description": "Why this change is being proposed.",
                },
            },
            "required": ["patch", "explanation"],
        },
    },
    {
        "name": "apply_config_change",
        "description": (
            "Apply a previously proposed config change and save to YAML. "
            "ONLY call this after the CEO has explicitly approved."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patch": {
                    "type": "object",
                    "description": "The config patch to apply.",
                }
            },
            "required": ["patch"],
        },
    },
    {
        "name": "resolve_proposal",
        "description": "Approve or reject a pending proposal by its 1-based number.",
        "input_schema": {
            "type": "object",
            "properties": {
                "proposal_number": {
                    "type": "integer",
                    "description": "1-based proposal number.",
                },
                "action": {
                    "type": "string",
                    "enum": ["approve", "reject"],
                    "description": "Whether to approve or reject.",
                },
            },
            "required": ["proposal_number", "action"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web for current market data, stock news, competitor "
            "info, or any other real-time information."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a file from the HydraCorp project. Use relative paths "
            "from project root, e.g. 'logs/hydra_training_state.json'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path from project root.",
                }
            },
            "required": ["path"],
        },
    },
]

# ═══════════════════════════════════════════════════════════════════════════
# Tool implementations
# ═══════════════════════════════════════════════════════════════════════════


def _load_training_state() -> dict:
    ts_path = _ROOT / "logs" / "hydra_training_state.json"
    if not ts_path.exists():
        return {}
    with open(ts_path, encoding="utf-8") as f:
        return json.load(f)


def _check_roadmap_triggers(ts: dict, corp_state: dict) -> list[str]:
    """Check if roadmap items should be surfaced based on system state."""
    items = []
    summary = ts.get("summary", {})
    total_gens = summary.get("total_generations", 0)
    regime = corp_state.get("regime", {})
    regime_class = regime.get("classification", "unknown")

    # Strategic Intelligence Layer — trigger after curriculum training completes
    # or after sufficient geopolitics expert runs
    roadmap_file = _ROOT / "docs" / "ROADMAP_strategic_intelligence.md"
    if roadmap_file.exists():
        geo_runs = sum(
            1 for m in corp_state.get("messages", [])
            if m.get("sender") == "geopolitics_expert"
        )
        curriculum_done = (_ROOT / "logs" / "curriculum_results.json").exists()

        if curriculum_done:
            items.append(
                "Strategic Intelligence Layer ready to build — curriculum "
                "training complete. See docs/ROADMAP_strategic_intelligence.md "
                "(thesis library: 12 priority thinkers, 15+ tier-2)"
            )
        elif total_gens >= 80:
            items.append(
                "Approaching readiness for Strategic Intelligence Layer "
                f"({total_gens} gens complete). Review roadmap: "
                "docs/ROADMAP_strategic_intelligence.md"
            )

    return items


def _tool_get_system_status(_input: dict) -> str:
    global _config
    config_dict = _config.model_dump()
    corp_state = _state.get_full_state()
    ts = _load_training_state()

    lines = []
    # Training state
    summary = ts.get("summary", {})
    metrics = ts.get("metrics", {})
    ts_config = ts.get("config", {})
    lines.append(f"Training: {summary.get('status', 'unknown')}")
    lines.append(f"  Generations: {summary.get('total_generations', 0)}/{ts_config.get('num_generations', '?')}")
    lines.append(f"  Real data: {ts_config.get('real_data', False)}")
    lines.append(f"  Mean reward: {metrics.get('mean_reward', 0):.1f}")
    lines.append(f"  Reward delta (total improvement): {metrics.get('reward_delta', 0):.1f}")

    # Best agent from latest gen
    gens = ts.get("generations", [])
    if gens:
        last = gens[-1]
        scores = last.get("eval_scores", {})
        if scores:
            best = max(scores, key=scores.get)
            lines.append(f"  Best agent: {best} ({scores[best]:+.1f})")
            positive = sum(1 for v in scores.values() if v > 0)
            lines.append(f"  Pool: {last.get('pool_size', len(scores))} agents, {positive}/{len(scores)} positive")

    # Corp state
    lines.append("")
    lines.append(f"Pipeline runs (corp): {corp_state.get('pipeline_run_count', 0)}")

    regime = corp_state.get("regime", {})
    r_class = regime.get("classification", "unknown")
    r_conf = regime.get("confidence")
    lines.append(f"Market regime: {r_class}" + (f" ({r_conf:.0%})" if r_conf else ""))

    pending = _state.get_pending_proposals()
    lines.append(f"Pending proposals: {len(pending)}")

    tickers = config_dict.get("data", {}).get("tickers", [])
    lines.append(f"Tickers ({len(tickers)}): {', '.join(tickers)}")

    # Roadmap reminders — surface when trigger conditions are met
    roadmap_items = _check_roadmap_triggers(ts, corp_state)
    if roadmap_items:
        lines.append("")
        lines.append("ROADMAP REMINDERS:")
        for item in roadmap_items:
            lines.append(f"  >> {item}")

    return "\n".join(lines)


def _tool_get_training_detail(input_: dict) -> str:
    last_n = input_.get("last_n", 5)
    ts = _load_training_state()
    gens = ts.get("generations", [])
    if not gens:
        return "No training generations found."

    recent = gens[-last_n:]
    lines = []
    for g in recent:
        gen_num = g["generation"]
        scores = g.get("eval_scores", {})
        lines.append(f"--- Generation {gen_num} ---")
        lines.append(f"  Train mean: {g.get('train_mean_reward', 0):.1f}")
        lines.append(f"  Pool size: {g.get('pool_size', 0)}")
        diag = g.get("diagnosis", {})
        lines.append(f"  Diagnosis: {diag.get('severity', '?')} - {diag.get('primary_issue', 'N/A')}")

        # Top 5 agents
        if scores:
            top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
            lines.append("  Top 5:")
            for name, score in top5:
                lines.append(f"    {name}: {score:+.1f}")

        # Conviction for agents with enough trades
        conv = g.get("conviction", {})
        good_conv = [(n, c) for n, c in conv.items()
                     if isinstance(c, dict) and c.get("total_trades", 0) >= 3]
        if good_conv:
            good_conv.sort(key=lambda x: x[1].get("overall_win_rate", 0), reverse=True)
            lines.append("  Conviction (3+ trades):")
            for name, c in good_conv[:3]:
                lines.append(
                    f"    {name}: WR={c['overall_win_rate']:.0%} "
                    f"({c['total_wins']}/{c['total_trades']})"
                )

        # Weight convergence
        comp = g.get("competition", {})
        if comp:
            lines.append(f"  Converged: {comp.get('converged', False)}")

        promoted = g.get("promoted", [])
        demoted = g.get("demoted", [])
        if promoted:
            lines.append(f"  Promoted: {', '.join(promoted)}")
        if demoted:
            lines.append(f"  Demoted: {', '.join(demoted)}")

        lines.append("")

    return "\n".join(lines)


def _tool_get_generation_scorecard(input_: dict) -> str:
    gen_num = input_.get("generation", 0)
    ts = _load_training_state()
    gens = ts.get("generations", [])
    if not gens:
        return "No training generations found."

    if gen_num <= 0:
        target = gens[-1]
        prev = gens[:-1]
    else:
        target = None
        prev = []
        for i, g in enumerate(gens):
            if g["generation"] == gen_num:
                target = g
                prev = gens[:i]
                break
        if target is None:
            return f"Generation {gen_num} not found."

    sc = score_generation(target, prev)
    return format_scorecard(sc)


def _tool_get_config(input_: dict) -> str:
    section = input_.get("section")
    config_dict = _config.model_dump()
    if section:
        data = config_dict.get(section)
        if data is None:
            return f"Unknown section: {section}. Valid: env, reward, training, data, pool, compute, validation"
        lines = [f"[{section}]"]
        for k, v in data.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)
    else:
        lines = []
        for sec_name in ("env", "reward", "training", "data", "pool", "compute", "validation"):
            sec = config_dict.get(sec_name, {})
            if sec:
                lines.append(f"\n[{sec_name}]")
                for k, v in sec.items():
                    if isinstance(v, list) and len(v) > 5:
                        lines.append(f"  {k}: [{len(v)} items]")
                    else:
                        lines.append(f"  {k}: {v}")
        return "\n".join(lines)


def _tool_list_proposals(_input: dict) -> str:
    pending = _state.get_pending_proposals()
    if not pending:
        return "No pending proposals."
    lines = []
    for i, p in enumerate(pending):
        prop_type = p.get("type", "config_change")
        lines.append(
            f"#{i + 1} [{p.get('status', 'pending').upper()}] "
            f"Type: {prop_type} | From: {p.get('source', 'unknown')}"
        )
        if p.get("memo"):
            lines.append(f"   Memo: \"{p['memo']}\"")

        # Ticker change proposals get enhanced display
        if prop_type == "ticker_change":
            meta = p.get("ticker_metadata", {})
            churn = meta.get("churn", {})
            lines.append(f"   Target tier: {meta.get('target_tier', '?')} tickers")
            if churn:
                lines.append(
                    f"   Churn: {churn.get('churn_pct', 0):.0%} "
                    f"(+{len(churn.get('added', []))} / "
                    f"-{len(churn.get('removed', []))} / "
                    f"={len(churn.get('retained', []))})"
                )
            dist = meta.get("sector_distribution", {})
            if dist:
                dist_str = ", ".join(f"{s}:{n}" for s, n in dist.items())
                lines.append(f"   Sectors: {dist_str}")
            patch = p.get("patch", {})
            tickers = patch.get("data", {}).get("tickers", [])
            if tickers:
                lines.append(f"   Tickers ({len(tickers)}): {', '.join(tickers[:15])}")
                if len(tickers) > 15:
                    lines.append(f"     ... and {len(tickers) - 15} more")
            lines.append("   WARNING: Ticker changes require full retraining.")
        elif p.get("patch"):
            for sec, vals in p["patch"].items():
                if isinstance(vals, dict):
                    for k, v in vals.items():
                        lines.append(f"   Patch: {sec}.{k} = {v}")
                else:
                    lines.append(f"   Patch: {sec} = {vals}")

        conf = p.get("confidence")
        risk = p.get("risk")
        parts = []
        if conf is not None:
            parts.append(f"Confidence: {conf:.0%}")
        if risk:
            parts.append(f"Risk: {risk}")
        if parts:
            lines.append(f"   {' | '.join(parts)}")
        lines.append("")
    return "\n".join(lines)


def _tool_propose_config_change(input_: dict) -> str:
    global _config
    patch = input_["patch"]
    explanation = input_.get("explanation", "")

    # Validate against Pydantic
    try:
        _config.apply_patch(patch)
    except Exception as e:
        return f"INVALID: {e}"

    # Senior Dev review
    config_dict = _config.model_dump()
    issues = _senior_dev._review_patch(patch, config_dict)

    # Build diff
    diff_lines = CEOInterface._build_diff(config_dict, patch)

    lines = ["PROPOSED CHANGE:"]
    for d in diff_lines:
        lines.append(f"  {d}")
    if explanation:
        lines.append(f"\nExplanation: {explanation}")

    if issues:
        lines.append(f"\nSENIOR DEV CONCERNS:")
        for issue in issues:
            lines.append(f"  - {issue}")
    else:
        lines.append("\nSENIOR DEV: Approved (no safety violations)")

    # Check if ticker change
    if "data" in patch and "tickers" in patch.get("data", {}):
        lines.append(
            "\nWARNING: Ticker changes alter observation dimensions (17N+5). "
            "All RL agents must be retrained."
        )

    lines.append("\nAwaiting CEO approval. Ask the CEO if they want to apply this change.")

    _decision_log.log(
        agent="ceo_interface",
        action="ceo_config_proposed",
        detail={"patch": patch, "explanation": explanation, "issues": issues},
        outcome="pending_approval",
    )

    return "\n".join(lines)


def _tool_apply_config_change(input_: dict) -> str:
    global _config
    patch = input_["patch"]

    try:
        _config = _config.apply_patch(patch)
        _config.to_yaml(str(_config_path))
    except Exception as e:
        return f"ERROR applying change: {e}"

    _decision_log.log(
        agent="ceo_interface",
        action="ceo_config_applied",
        detail={"patch": patch, "saved_to": str(_config_path)},
        outcome="applied",
    )

    return f"Config saved to {_config_path}. Will take effect on next pipeline run."


def _tool_resolve_proposal(input_: dict) -> str:
    global _config
    idx = input_["proposal_number"] - 1  # 1-based to 0-based
    action = input_["action"]

    all_proposals = _state._read_state()["proposals"]
    pending_indices = [i for i, p in enumerate(all_proposals) if p.get("status") == "pending"]

    if idx < 0 or idx >= len(pending_indices):
        return f"Proposal #{input_['proposal_number']} not found. Use list_proposals to see pending."

    abs_idx = pending_indices[idx]
    proposal = all_proposals[abs_idx]
    patch = proposal.get("patch")

    if action == "approve":
        if patch:
            try:
                _config = _config.apply_patch(patch)
                _config.to_yaml(str(_config_path))
            except Exception as e:
                return f"Cannot apply proposal: {e}"
        _state.resolve_proposal(abs_idx, "approved", reason="CEO approved")

        # Record ticker changes in history for anti-churn tracking
        if proposal.get("type") == "ticker_change":
            _state.record_ticker_change({
                "old_tickers": proposal.get("ticker_metadata", {}).get("churn", {}).get("retained", []),
                "new_tickers": patch.get("data", {}).get("tickers", []) if patch else [],
                "churn": proposal.get("ticker_metadata", {}).get("churn", {}),
                "regime": proposal.get("ticker_metadata", {}).get("regime", ""),
            })

        _decision_log.log(
            agent="ceo_interface",
            action="ceo_proposal_approval",
            detail={"proposal_index": abs_idx, "patch": patch, "type": proposal.get("type")},
            outcome="approved",
        )
        result = f"Proposal #{input_['proposal_number']} approved."
        if patch:
            result += f" Config saved to {_config_path}."
        if proposal.get("type") == "ticker_change":
            result += " WARNING: All agent checkpoints are now invalidated. Retraining required."
        return result
    else:
        _state.resolve_proposal(abs_idx, "rejected", reason="CEO rejected")
        _decision_log.log(
            agent="ceo_interface",
            action="ceo_proposal_rejection",
            detail={"proposal_index": abs_idx},
            outcome="rejected",
        )
        return f"Proposal #{input_['proposal_number']} rejected."


def _tool_web_search(input_: dict) -> str:
    query = input_["query"]
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        return "Web search unavailable (install requests + beautifulsoup4)."

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/132.0.0.0 Safari/537.36"
            )
        }
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers=headers,
            timeout=12,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        results = []
        for result in soup.select(".result")[:8]:
            title_el = result.select_one(".result__title")
            snippet_el = result.select_one(".result__snippet")
            url_el = result.select_one(".result__url")

            title = title_el.get_text(strip=True) if title_el else ""
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            link = url_el.get_text(strip=True) if url_el else ""

            if title:
                results.append(f"TITLE: {title}")
                if snippet:
                    results.append(f"SNIPPET: {snippet}")
                if link:
                    results.append(f"URL: https://{link}" if not link.startswith("http") else f"URL: {link}")
                results.append("")

        return "\n".join(results).strip() if results else f"No results for: {query}"

    except Exception as e:
        return f"Search error: {e}"


def _tool_read_file(input_: dict) -> str:
    rel_path = input_["path"]
    full_path = _ROOT / rel_path
    if not full_path.exists():
        return f"File not found: {rel_path}"
    try:
        text = full_path.read_text(encoding="utf-8")
        limit = 15_000
        if len(text) > limit:
            text = text[:limit] + f"\n\n[...truncated at {limit} chars]"
        return text
    except Exception as e:
        return f"Error reading {rel_path}: {e}"


# Dispatch table
_TOOL_DISPATCH = {
    "get_system_status": _tool_get_system_status,
    "get_training_detail": _tool_get_training_detail,
    "get_generation_scorecard": _tool_get_generation_scorecard,
    "get_config": _tool_get_config,
    "list_proposals": _tool_list_proposals,
    "propose_config_change": _tool_propose_config_change,
    "apply_config_change": _tool_apply_config_change,
    "resolve_proposal": _tool_resolve_proposal,
    "web_search": _tool_web_search,
    "read_file": _tool_read_file,
}


# ═══════════════════════════════════════════════════════════════════════════
# Agentic loop (streaming + tool use)
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are the CEO Interface for HydraCorp, an AI-managed trading research corporation.
You help the CEO manage the system through natural conversation.

HYDRACORP CONTEXT:
- Starting capital: $2,500 CAD (backtesting simulation)
- Default 10 stock tickers (tech/crypto/growth) — system supports up to 500
- RL agents (PPO, SAC, RecurrentPPO) + rule-based agents trained via population-based training
- 8 corp agents: Senior Dev, Hedge Fund Director, Contrarian, Shadow Trader, Hardware Optimizer, Geopolitics Expert, Innovation Scout, Chief of Staff
- Config is Pydantic-validated (HydraConfig), saved to YAML
- Observation space: 17*N+5 where N = number of tickers

RULES:
- Always use tools to get current data. NEVER guess numbers or make up status.
- For config changes: ALWAYS use propose_config_change first, explain the change and any warnings to the CEO, and ONLY call apply_config_change after the CEO explicitly says yes/approve/do it.
- Keep responses concise — this is a terminal, not a report.
- When discussing training, reference specific generation numbers and agent names.
- If the CEO asks about market conditions, use web_search to get current data.

AVAILABLE TOOLS: get_system_status, get_training_detail, get_generation_scorecard, get_config, list_proposals, propose_config_change, apply_config_change, resolve_proposal, web_search, read_file"""


def _run_agentic_turn(client, messages: list) -> list:
    """Run one agentic turn with streaming and tool calls. Returns updated messages."""
    while True:
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            temperature=0.3,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        ) as stream:
            tool_calls = []
            for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        tool_calls.append({"id": block.id, "name": block.name, "input_json": ""})
                    elif block.type == "text":
                        pass  # Will stream in delta

                elif event.type == "content_block_delta":
                    delta = event.delta
                    if hasattr(delta, "text") and delta.text:
                        print(delta.text, end="", flush=True)
                    elif hasattr(delta, "partial_json") and tool_calls:
                        tool_calls[-1]["input_json"] += delta.partial_json

            response = stream.get_final_message()

        messages.append({"role": "assistant", "content": response.content})

        # If no tool calls, we're done
        if response.stop_reason == "end_turn" or not tool_calls:
            print()  # Final newline
            return messages

        # Execute tool calls
        tool_results = []
        for tc in tool_calls:
            try:
                tool_input = json.loads(tc["input_json"]) if tc["input_json"] else {}
            except json.JSONDecodeError:
                tool_input = {}

            # Show tool invocation
            tool_label = tc["name"]
            if tool_input:
                first_val = list(tool_input.values())[0]
                if isinstance(first_val, str) and len(first_val) < 60:
                    tool_label += f"({first_val})"
            print(f"\n  {_c(f'[{tool_label}]', DIM)}", flush=True)

            handler = _TOOL_DISPATCH.get(tc["name"])
            if handler:
                result = handler(tool_input)
            else:
                result = f"Unknown tool: {tc['name']}"

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc["id"],
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})


def _trim_history(messages: list, max_pairs: int = 30) -> list:
    """Trim conversation history, preserving tool_use/tool_result pairs."""
    if len(messages) <= max_pairs:
        return messages
    start = len(messages) - max_pairs
    while start < len(messages):
        msg = messages[start]
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            break
        start += 1
    if start >= len(messages):
        return messages[-2:]
    return messages[start:]


# ═══════════════════════════════════════════════════════════════════════════
# Simple mode (fallback — no API key needed)
# ═══════════════════════════════════════════════════════════════════════════

def _simple_print_result(result: dict) -> None:
    intent = result.get("intent", "unknown")
    if intent in ("status_query", "proposal_action") and not result.get("needs_confirmation"):
        text = result.get("response_text", "")
        if text:
            print()
            for line in text.split("\n"):
                print(f"  {line}")
            print()
        if result.get("patch"):
            print(_c("  Config updated.", GREEN))
            print()
        return

    if intent == "unknown":
        print(f"\n  {result.get('response_text', 'Unknown command.')}\n")
        return

    print(f"\n{_c('  PROPOSED CHANGE:', BOLD)}")
    for line in result.get("diff_lines", []):
        print(f"  {line}")
    if result.get("response_text"):
        print(f"  {result['response_text']}")

    for w in result.get("warnings", []):
        print(f"  {_c('WARNING:', YELLOW)} {w}")

    verdict = result.get("senior_dev_verdict", "")
    if verdict:
        color = YELLOW if "concern" in verdict.lower() else GREEN
        print(f"  {_c('SENIOR DEV:', color)} {verdict}")
    print()


def run_simple_mode() -> None:
    """Regex-based REPL — no API key needed."""
    global _config

    print()
    print(_c("=" * 56, CYAN))
    print(_c("  HYDRACORP CEO TERMINAL (simple mode)", BOLD))
    print(_c("  Type commands. 'help' for examples, 'quit' to exit.", DIM))
    print(_c("=" * 56, CYAN))
    print()

    try:
        import readline  # noqa: F401
    except ImportError:
        pass

    while True:
        try:
            raw = input(_c("  CEO > ", BOLD))
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break

        cmd = raw.strip()
        if not cmd:
            continue
        if cmd.lower() in ("quit", "exit", "q"):
            break
        if cmd.lower() == "help":
            _print_simple_help()
            continue
        if cmd.lower() == "scorecard":
            ts = _load_training_state()
            gens = ts.get("generations", [])
            if gens:
                sc = score_generation(gens[-1], gens[:-1])
                print(f"\n  {format_scorecard(sc)}\n")
            else:
                print("\n  No training generations found.\n")
            continue

        result = _ceo_agent.process_command(cmd, _config)
        _simple_print_result(result)

        if result.get("needs_confirmation") and result.get("patch"):
            try:
                answer = input(_c("  Apply this change? [y/n] > ", BOLD)).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  Cancelled.\n")
                continue
            if answer in ("y", "yes"):
                try:
                    _config = _config.apply_patch(result["patch"])
                    _config.to_yaml(str(_config_path))
                    print(_c(f"\n  Config saved to {_config_path}", GREEN))
                    _decision_log.log("ceo_interface", "ceo_config_applied",
                                      {"patch": result["patch"]}, "applied")
                except Exception as e:
                    print(_c(f"\n  Error: {e}", RED))
                print()
            else:
                print("  Discarded.\n")

        elif result.get("intent") == "proposal_action" and result.get("patch"):
            try:
                _config = _config.apply_patch(result["patch"])
                _config.to_yaml(str(_config_path))
                print(_c(f"  Config saved to {_config_path}", GREEN))
            except Exception as e:
                print(_c(f"  Error: {e}", RED))
            print()


def _print_simple_help() -> None:
    print(f"""
  {_c('COMMANDS:', BOLD)}

  add AAPL / drop MARA / switch to tech stocks
  tighten risk / more aggressive / set max_position_pct to 0.25
  run 50 generations / increase learning rate
  show status / show config / show config env
  show proposals / approve proposal 1 / reject proposal 1
  scorecard                   Show generation scorecard
""")


# ═══════════════════════════════════════════════════════════════════════════
# Agentic mode
# ═══════════════════════════════════════════════════════════════════════════

def run_agentic_mode() -> None:
    """Full agentic interface with Claude Sonnet, streaming, and tools."""
    import anthropic
    client = anthropic.Anthropic()

    print()
    print(_c("=" * 56, CYAN))
    print(_c("  HYDRACORP CEO TERMINAL", BOLD))
    print(_c("  Agentic mode — conversational, with tools.", DIM))
    print(_c("  Type naturally. 'quit' to exit.", DIM))
    print(_c("=" * 56, CYAN))
    print()

    messages: list = []

    # Opening turn: auto-load status
    messages.append({
        "role": "user",
        "content": (
            "Give me a brief status update. Use get_system_status and "
            "get_generation_scorecard to pull current data. Keep it to 5-6 lines."
        ),
    })
    print(f"  {_c('CEO >', BOLD)} show status\n")
    try:
        messages = _run_agentic_turn(client, messages)
    except Exception as e:
        print(f"\n  {_c(f'Error: {e}', RED)}")
        messages = []
    print()

    # readline support
    try:
        import readline  # noqa: F401
    except ImportError:
        pass

    while True:
        try:
            raw = input(_c("  CEO > ", BOLD))
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break

        cmd = raw.strip()
        if not cmd:
            continue
        if cmd.lower() in ("quit", "exit", "q"):
            break

        messages = _trim_history(messages)
        messages.append({"role": "user", "content": cmd})

        print()
        try:
            messages = _run_agentic_turn(client, messages)
        except Exception as e:
            print(f"\n  {_c(f'Error: {e}', RED)}")
            messages = messages[:-1]  # Remove failed user msg
        print()


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    global _config, _config_path, _state, _decision_log, _senior_dev, _ceo_agent

    parser = argparse.ArgumentParser(description="HydraCorp CEO Terminal")
    parser.add_argument("--config", type=str, default=None, help="Path to HydraConfig YAML")
    parser.add_argument("--simple", action="store_true", help="Use simple regex mode (no API key)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    # Load config
    _config_path = Path(args.config) if args.config else Path("hydra_config.yaml")
    if _config_path.exists():
        _config = HydraConfig.from_yaml(_config_path)
    else:
        _config = HydraConfig()

    # Init state
    _state = CorporationState()
    _decision_log = DecisionLog()
    blacklist = ConfigBlacklist()

    _senior_dev = SeniorDev(state=_state, decision_log=_decision_log, blacklist=blacklist)
    _ceo_agent = CEOInterface(state=_state, decision_log=_decision_log, senior_dev=_senior_dev)

    # Choose mode
    if args.simple:
        run_simple_mode()
        return

    # Try agentic mode (requires anthropic SDK for streaming/tool use)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if groq_key:
            print(f"\n  {_c('Groq API key found — corp agents will use Llama 3.3.', DIM)}")
        print(f"\n  {_c('No ANTHROPIC_API_KEY — using simple mode for CEO CLI.', YELLOW)}")
        print(f"  {_c('(Corp agents still use Groq/rule-based for analysis.)', DIM)}\n")
        run_simple_mode()
        return

    try:
        import anthropic  # noqa: F401
    except ImportError:
        print(f"\n  {_c('anthropic package not installed. Falling back to simple mode.', YELLOW)}\n")
        run_simple_mode()
        return

    run_agentic_mode()


if __name__ == "__main__":
    main()
