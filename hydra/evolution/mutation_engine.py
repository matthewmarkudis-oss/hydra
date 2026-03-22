"""Mutation record dataclass — used by diagnostic engine to tag recommended actions.

The full MutationEngine (weight/param/risk/reward mutations, random variant
generator) was removed as dead code: mutations were applied to an internal
overlay dict that PROMETHEUS immediately overwrote every generation.

Only MutationRecord is retained as a lightweight label for diagnostic output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MutationRecord:
    """A single mutation recommended by the diagnostic engine."""

    mutation_type: str          # e.g. "bench_agent", "tighten_risk"
    category: str               # "weight", "inclusion", "parameter", "objective"
    description: str            # Human-readable
    params: dict = field(default_factory=dict)
    timestamp: str = ""
