"""Model serialization helpers."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def save_numpy(data: dict[str, np.ndarray], path: str | Path) -> None:
    """Save dict of numpy arrays to .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **data)


def load_numpy(path: str | Path) -> dict[str, np.ndarray]:
    """Load dict of numpy arrays from .npz file."""
    with np.load(str(path)) as data:
        return {k: data[k] for k in data.files}


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    """Save a training checkpoint (model state + metadata)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a training checkpoint."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(data: dict, path: str | Path) -> None:
    """Save JSON metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str | Path) -> dict:
    """Load JSON metadata."""
    with open(path) as f:
        return json.load(f)
