"""Agent pool manager.

Manages the diverse population of agents (learning, static, rule-based).
Handles add/remove/freeze/promote/demote operations and serialization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from hydra.agents.base_rl_agent import BaseRLAgent
from hydra.agents.static_agent import StaticAgent

logger = logging.getLogger("hydra.agents.pool")


class AgentPool:
    """Population manager for the multi-agent system.

    Maintains a diverse pool of agents and handles lifecycle operations:
    promotion (learning → static), demotion (remove worst), and freezing.
    """

    def __init__(self):
        self._agents: dict[str, BaseRLAgent] = {}
        self._weights: dict[str, float] = {}
        self._rankings: dict[str, float] = {}  # agent_name → performance score

    def add(self, agent: BaseRLAgent, weight: float = 1.0) -> None:
        """Add an agent to the pool."""
        if agent.name in self._agents:
            logger.warning(f"Agent '{agent.name}' already in pool, replacing")
        self._agents[agent.name] = agent
        self._weights[agent.name] = weight
        logger.info(f"Added agent '{agent.name}' ({agent.__class__.__name__}) to pool")

    def remove(self, name: str) -> BaseRLAgent | None:
        """Remove an agent from the pool."""
        agent = self._agents.pop(name, None)
        self._weights.pop(name, None)
        self._rankings.pop(name, None)
        if agent:
            logger.info(f"Removed agent '{name}' from pool")
        return agent

    def get(self, name: str) -> BaseRLAgent | None:
        """Get an agent by name."""
        return self._agents.get(name)

    def get_all(self) -> list[BaseRLAgent]:
        """Get all agents in the pool."""
        return list(self._agents.values())

    def get_learning_agents(self) -> list[BaseRLAgent]:
        """Get all agents that actively learn."""
        return [a for a in self._agents.values() if a.is_learning]

    def get_frozen_agents(self) -> list[BaseRLAgent]:
        """Get all frozen/static agents."""
        return [a for a in self._agents.values() if a.is_frozen]

    @property
    def size(self) -> int:
        return len(self._agents)

    @property
    def agent_names(self) -> list[str]:
        return list(self._agents.keys())

    def get_weights(self) -> np.ndarray:
        """Get normalized action aggregation weights for all agents."""
        names = list(self._agents.keys())
        raw = np.array([self._weights.get(n, 1.0) for n in names], dtype=np.float32)
        total = np.sum(raw)
        if total > 0:
            return raw / total
        return np.ones(len(names), dtype=np.float32) / max(len(names), 1)

    def set_weight(self, name: str, weight: float) -> None:
        """Set the action aggregation weight for an agent."""
        if name in self._agents:
            self._weights[name] = weight

    def collect_actions(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> dict[str, np.ndarray]:
        """Collect actions from all agents for the same observation.

        Returns dict of agent_name → action_array.
        """
        actions = {}
        for name, agent in self._agents.items():
            actions[name] = agent.select_action(observation, deterministic=deterministic)
        return actions

    def aggregate_actions(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Get weighted aggregate action from all agents.

        Returns a single action array (weighted mean of all agent actions).
        """
        actions = self.collect_actions(observation, deterministic)
        if not actions:
            return np.zeros(0, dtype=np.float32)

        names = list(actions.keys())
        weights = self.get_weights()
        action_matrix = np.stack([actions[n] for n in names])

        # Weighted mean
        aggregated = np.average(action_matrix, axis=0, weights=weights)
        return np.clip(aggregated, -1.0, 1.0).astype(np.float32)

    def update_rankings(self, scores: dict[str, float]) -> None:
        """Update performance rankings for agents.

        Args:
            scores: Dict of agent_name → performance score (higher is better).
        """
        self._rankings.update(scores)

    def get_ranked_agents(self) -> list[tuple[str, float]]:
        """Get agents sorted by performance (best first)."""
        return sorted(self._rankings.items(), key=lambda x: x[1], reverse=True)

    def promote_top(self, k: int = 2) -> list[str]:
        """Freeze top-K learning agents as static snapshots.

        Creates static copies of the best learning agents and adds them to pool.
        Returns names of promoted agents.
        """
        ranked = self.get_ranked_agents()
        learning = {a.name for a in self.get_learning_agents()}
        promoted = []

        for name, score in ranked:
            if name in learning and len(promoted) < k:
                agent = self._agents[name]
                # Log source agent weight checksum before snapshot
                src_model = getattr(agent, '_model', None)
                if src_model is not None:
                    from hydra.agents.static_agent import _log_weight_checksum
                    _log_weight_checksum(f"{name} (live)", src_model)
                static = StaticAgent.from_agent(agent, name=f"{name}_gen{agent.episode_count}")
                self.add(static)
                promoted.append(static.name)
                logger.info(f"Promoted '{name}' -> '{static.name}' (score={score:.4f})")

        return promoted

    def demote_bottom(self, k: int = 1) -> list[str]:
        """Remove bottom-K static agents from the pool.

        Only removes static/frozen agents (never removes learning agents).
        Returns names of removed agents.
        """
        ranked = self.get_ranked_agents()
        frozen_names = {a.name for a in self.get_frozen_agents()}
        demoted = []

        for name, score in reversed(ranked):
            if name in frozen_names and len(demoted) < k:
                self.remove(name)
                demoted.append(name)
                logger.info(f"Demoted '{name}' (score={score:.4f})")

        return demoted

    def save(self, directory: str | Path) -> None:
        """Save all agents and pool metadata to directory."""
        from hydra.utils.serialization import save_json

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Determine obs/action dims from first agent for feature_config
        first_agent = next(iter(self._agents.values()), None)
        feature_config = {}
        if first_agent:
            feature_config = {
                "num_tickers": first_agent.action_dim,
                "obs_dim": first_agent.obs_dim,
                "action_dim": first_agent.action_dim,
                "obs_layout": "17N+5",
            }

        metadata = {
            "agents": {},
            "weights": self._weights,
            "rankings": self._rankings,
            "feature_config": feature_config,
        }

        for name, agent in self._agents.items():
            agent_dir = directory / name
            agent_dir.mkdir(exist_ok=True)
            try:
                agent.save(agent_dir / "model")
            except Exception as e:
                logger.warning(f"Failed to save agent '{name}': {e}")

            agent_meta = {
                "type": agent.__class__.__name__,
                "obs_dim": agent.obs_dim,
                "action_dim": agent.action_dim,
                "frozen": agent.is_frozen,
                "total_steps": agent.total_steps,
            }
            # Persist source_type for StaticAgent so SAC snapshots
            # reload with the correct SB3 model class.
            if hasattr(agent, "_source_type"):
                agent_meta["source_type"] = agent._source_type
            metadata["agents"][name] = agent_meta

        save_json(metadata, directory / "pool_metadata.json")
        logger.info(f"Saved agent pool ({self.size} agents) to {directory}")

    def load(self, directory: str | Path) -> None:
        """Load agents and pool metadata from directory."""
        from hydra.utils.serialization import load_json

        directory = Path(directory)
        metadata = load_json(directory / "pool_metadata.json")

        self._weights = metadata.get("weights", {})
        self._rankings = metadata.get("rankings", {})

        failed_agents = []
        for name, info in metadata.get("agents", {}).items():
            agent_dir = directory / name
            agent_type = info["type"]
            obs_dim = info["obs_dim"]
            action_dim = info["action_dim"]
            extra = {}
            if "source_type" in info:
                extra["source_type"] = info["source_type"]

            try:
                agent = self._create_agent(agent_type, name, obs_dim, action_dim, **extra)
                agent.load(agent_dir / "model")
                if info.get("frozen", False):
                    agent.freeze()
                self._agents[name] = agent
            except Exception as e:
                logger.warning(
                    f"Primary load failed for agent '{name}' ({agent_type}): {e}. "
                    f"Attempting CPU-only fallback..."
                )
                # CPU fallback retry — handles DirectML/device comparison bugs
                try:
                    agent = self._create_agent(agent_type, name, obs_dim, action_dim, **extra)
                    agent.load(agent_dir / "model")  # agents already use device="cpu" internally
                    if info.get("frozen", False):
                        agent.freeze()
                    self._agents[name] = agent
                    logger.info(f"CPU fallback succeeded for agent '{name}'")
                except Exception as e2:
                    logger.error(
                        f"CHECKPOINT LOAD FAILED for agent '{name}' ({agent_type}): {e2}. "
                        f"Learned weights from this agent will be LOST."
                    )
                    failed_agents.append(name)

        if failed_agents:
            logger.error(
                f"Failed to load {len(failed_agents)}/{len(metadata.get('agents', {}))} "
                f"agents: {failed_agents}. Training may restart from scratch."
            )
        logger.info(f"Loaded agent pool ({self.size} agents) from {directory}")

    @staticmethod
    def _create_agent(
        agent_type: str,
        name: str,
        obs_dim: int,
        action_dim: int,
        **kwargs,
    ) -> BaseRLAgent:
        """Factory method to create agent by type name.

        Extra kwargs (e.g. source_type for StaticAgent) are forwarded
        to the constructor when the class accepts them.
        """
        from hydra.agents.ppo_agent import PPOAgent
        from hydra.agents.sac_agent import SACAgent
        from hydra.agents.a2c_agent import A2CAgent
        from hydra.agents.td3_agent import TD3Agent
        from hydra.agents.recurrent_ppo_agent import RecurrentPPOAgent
        from hydra.agents.rule_based_agent import RuleBasedAgent
        from hydra.agents.cmaes_agent import CMAESAgent

        type_map = {
            "PPOAgent": PPOAgent,
            "SACAgent": SACAgent,
            "A2CAgent": A2CAgent,
            "TD3Agent": TD3Agent,
            "RecurrentPPOAgent": RecurrentPPOAgent,
            "CMAESAgent": CMAESAgent,
            "StaticAgent": StaticAgent,
            "RuleBasedAgent": RuleBasedAgent,
        }

        cls = type_map.get(agent_type)
        if cls is None:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Only pass kwargs the constructor accepts (e.g. source_type for StaticAgent)
        import inspect
        sig = inspect.signature(cls.__init__)
        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

        return cls(name=name, obs_dim=obs_dim, action_dim=action_dim, **valid_kwargs)

    def get_summary(self) -> dict[str, Any]:
        """Get pool summary for logging."""
        return {
            "total_agents": self.size,
            "learning_agents": len(self.get_learning_agents()),
            "frozen_agents": len(self.get_frozen_agents()),
            "agents": {n: a.get_info() for n, a in self._agents.items()},
            "rankings": dict(self.get_ranked_agents()),
        }
