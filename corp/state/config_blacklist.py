"""Config blacklist — tracks failed configurations to prevent regressions.

Hashes HydraConfig dicts and maintains a set of known-bad config hashes
with their failure reasons and metrics.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("corp.state.blacklist")


class ConfigBlacklist:
    """Tracks configurations that produced poor results.

    Uses SHA-256 of the sorted config dict to identify configs.
    Stores blacklist entries with failure metrics and reasons.
    """

    def __init__(self, blacklist_file: str = "logs/config_blacklist.json"):
        self._path = Path(blacklist_file)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._write({"entries": {}, "created": datetime.now().isoformat()})

    def _read(self) -> dict:
        try:
            with open(self._path) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = {"entries": {}, "created": datetime.now().isoformat()}
            self._write(data)
            return data

    def _write(self, data: dict) -> None:
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def compute_hash(config_dict: dict) -> str:
        """Compute SHA-256 hash of a config dict.

        Only hashes the tunable parameters (env, reward, pool structure)
        — ignores paths, log levels, and other non-functional fields.
        """
        # Extract only the parameters that affect training behavior
        hashable = {}
        for section in ("env", "reward", "pool", "training", "data"):
            if section in config_dict:
                hashable[section] = config_dict[section]

        serialized = json.dumps(hashable, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def is_blacklisted(self, config_dict: dict) -> tuple[bool, str]:
        """Check if a config is blacklisted.

        Returns:
            (is_blacklisted, reason) tuple.
        """
        config_hash = self.compute_hash(config_dict)
        data = self._read()
        entry = data["entries"].get(config_hash)
        if entry:
            return True, entry.get("reason", "Previously failed")
        return False, ""

    def add(
        self,
        config_dict: dict,
        reason: str,
        metrics: dict[str, float] | None = None,
    ) -> str:
        """Add a config to the blacklist.

        Returns:
            The config hash.
        """
        config_hash = self.compute_hash(config_dict)
        data = self._read()
        data["entries"][config_hash] = {
            "reason": reason,
            "metrics": metrics or {},
            "added": datetime.now().isoformat(),
        }
        self._write(data)
        logger.info(f"Blacklisted config {config_hash}: {reason}")
        return config_hash

    def remove(self, config_hash: str) -> bool:
        """Remove a config from the blacklist (CEO override)."""
        data = self._read()
        if config_hash in data["entries"]:
            del data["entries"][config_hash]
            self._write(data)
            logger.info(f"Removed config {config_hash} from blacklist")
            return True
        return False

    def list_entries(self) -> dict[str, dict]:
        """List all blacklisted configs."""
        return self._read()["entries"]

    def populate_from_meta_optimize(self, jsonl_path: str, threshold_fitness: float = -0.5) -> int:
        """Pre-populate blacklist from meta-optimizer trial results.

        Blacklists any trial with fitness below the threshold.

        Args:
            jsonl_path: Path to meta_optimize.jsonl.
            threshold_fitness: Trials below this fitness are blacklisted.

        Returns:
            Number of entries added.
        """
        path = Path(jsonl_path)
        if not path.exists():
            logger.warning(f"Meta-optimize log not found: {path}")
            return 0

        count = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    trial = json.loads(line)
                except json.JSONDecodeError:
                    continue

                fitness = trial.get("fitness", trial.get("value", 0))
                if fitness < threshold_fitness:
                    config = trial.get("params", trial.get("config", {}))
                    if config:
                        self.add(
                            config,
                            reason=f"Meta-optimizer trial with fitness={fitness:.4f}",
                            metrics={"fitness": fitness},
                        )
                        count += 1

        logger.info(f"Populated blacklist with {count} entries from {path}")
        return count
