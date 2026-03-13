"""Model checkpoint save/load and training resume support."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from hydra.utils.serialization import save_json, load_json

logger = logging.getLogger("hydra.training.checkpointer")


class Checkpointer:
    """Manages training checkpoints for save/resume capability."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        pool,
        generation: int,
        episode: int,
        metrics: dict[str, Any],
    ) -> Path:
        """Save a full training checkpoint."""
        ckpt_path = self.checkpoint_dir / f"gen{generation:04d}_ep{episode:06d}"
        ckpt_path.mkdir(parents=True, exist_ok=True)

        # Save pool (all agents)
        pool.save(ckpt_path / "pool")

        # Save training state
        save_json({
            "generation": generation,
            "episode": episode,
            "metrics": metrics,
        }, ckpt_path / "training_state.json")

        logger.info(f"Checkpoint saved: {ckpt_path}")
        return ckpt_path

    def load_latest(self) -> tuple[Path | None, dict[str, Any]]:
        """Find and return the latest checkpoint."""
        checkpoints = sorted(self.checkpoint_dir.glob("gen*_ep*"))
        if not checkpoints:
            return None, {}

        latest = checkpoints[-1]
        state = load_json(latest / "training_state.json")
        logger.info(f"Found latest checkpoint: {latest}")
        return latest, state

    def load(self, path: Path, pool) -> dict[str, Any]:
        """Load a specific checkpoint into the pool."""
        pool.load(path / "pool")
        state = load_json(path / "training_state.json")
        logger.info(f"Loaded checkpoint from {path}")
        return state

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all available checkpoints."""
        checkpoints = []
        for ckpt_dir in sorted(self.checkpoint_dir.glob("gen*_ep*")):
            try:
                state = load_json(ckpt_dir / "training_state.json")
                checkpoints.append({
                    "path": str(ckpt_dir),
                    "generation": state.get("generation", 0),
                    "episode": state.get("episode", 0),
                })
            except Exception:
                pass
        return checkpoints
