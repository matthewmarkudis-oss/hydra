"""Shared LLM client for HydraCorp agents.

Supports multiple providers with automatic fallback:
1. Groq (free tier, Llama 3.3 70B — no credit card required)
2. Anthropic (Claude Haiku — requires API key + billing)
3. Returns None if no LLM is available (agents use rule-based fallback)

All calls use raw urllib — no pip install needed.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error
from typing import Any

logger = logging.getLogger("corp.llm_client")


def call_llm(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1200,
    temperature: float = 0.3,
) -> str | None:
    """Call an LLM and return the raw text response.

    Tries providers in order: Groq → Anthropic → None.

    Returns:
        Response text, or None if no provider is available.
    """
    # Try Groq first (free, no credit card)
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        result = _call_groq(groq_key, system_prompt, user_prompt, max_tokens, temperature)
        if result is not None:
            return result

    # Try Anthropic
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        result = _call_anthropic(anthropic_key, system_prompt, user_prompt, max_tokens, temperature)
        if result is not None:
            return result

    return None


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1200,
    temperature: float = 0.3,
) -> dict | None:
    """Call an LLM and parse the response as JSON.

    Returns:
        Parsed dict, or None if no provider is available or parsing fails.
    """
    text = call_llm(system_prompt, user_prompt, max_tokens, temperature)
    if text is None:
        return None

    return _parse_json_response(text)


def _call_groq(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str | None:
    """Call Groq API (OpenAI-compatible) via urllib."""
    try:
        body = json.dumps({
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }).encode()

        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "HydraCorp/1.0",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())

        text = data["choices"][0]["message"]["content"]
        logger.debug("Groq LLM call succeeded (%d chars)", len(text))
        return text

    except urllib.error.HTTPError as e:
        body_text = e.read().decode() if hasattr(e, "read") else ""
        logger.warning("Groq API error %d: %s", e.code, body_text[:200])
        return None
    except Exception as e:
        logger.warning("Groq call failed: %s", e)
        return None


def _call_anthropic(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str | None:
    """Call Anthropic Messages API via urllib (no SDK needed)."""
    try:
        body = json.dumps({
            "model": "claude-3-haiku-20240307",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }).encode()

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())

        text = data["content"][0]["text"]
        logger.debug("Anthropic LLM call succeeded (%d chars)", len(text))
        return text

    except urllib.error.HTTPError as e:
        body_text = e.read().decode() if hasattr(e, "read") else ""
        logger.warning("Anthropic API error %d: %s", e.code, body_text[:200])
        return None
    except Exception as e:
        logger.warning("Anthropic call failed: %s", e)
        return None


def _parse_json_response(text: str) -> dict | None:
    """Extract JSON from an LLM response that might have markdown fences."""
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

    logger.warning("Failed to parse LLM JSON response: %s...", text[:100])
    return None
