"""Master pipeline controller — chains phases 1-6 via workflow DAG.

Supports checkpoint/resume for long-running training sessions.
Backtesting and training only.
"""

from __future__ import annotations

import logging
from typing import Any

from hydra.compute.workflow import Workflow
from hydra.config.schema import HydraConfig
from hydra.pipeline.data_prep import prepare_data
from hydra.pipeline.env_builder import build_environments
from hydra.pipeline.train_phase import run_training
from hydra.pipeline.eval_phase import run_evaluation
from hydra.pipeline.pool_update import update_pool
from hydra.pipeline.validation_phase import run_validation

logger = logging.getLogger("hydra.pipeline.orchestrator")


class PipelineOrchestrator:
    """Orchestrates the full Hydra training pipeline.

    Phases:
    1. Data preparation (fetch + cache + features)
    2. Environment construction (train/val/test splits)
    3. RL training (population-based)
    4. Evaluation (RL env + VectorBT backtest)
    5. Pool update (promote/demote agents)
    6. Validation (ATHENA walk-forward)
    """

    def __init__(self, config: HydraConfig | None = None):
        self.config = config or HydraConfig()
        self._workflow: Workflow | None = None
        self._results: dict[str, Any] = {}

    def build_workflow(self) -> Workflow:
        """Construct the pipeline DAG."""
        cfg = self.config

        wf = Workflow(name="hydra_training_pipeline")

        # Phase 1: Data preparation
        wf.add_task(
            "data_prep",
            prepare_data,
            tickers=cfg.data.tickers,
            episode_bars=cfg.env.episode_bars,
            cache_dir=cfg.data.cache_dir,
            use_synthetic=True,  # Default to synthetic for initial runs
            seed=cfg.seed,
        )

        # Phase 2: Environment construction
        wf.add_task(
            "env_builder",
            build_environments,
            dependencies=["data_prep"],
            num_stocks=cfg.env.num_stocks,
            episode_bars=cfg.env.episode_bars,
            initial_cash=cfg.env.initial_cash,
            seed=cfg.seed,
        )

        # Phase 3: Training
        wf.add_task(
            "train_phase",
            run_training,
            dependencies=["env_builder"],
            num_generations=cfg.training.num_generations,
            episodes_per_generation=cfg.training.episodes_per_generation,
            top_k_promote=cfg.training.top_k_promote,
            bottom_k_demote=cfg.training.bottom_k_demote,
            checkpoint_dir=cfg.training.checkpoint_dir,
            tensorboard_dir=cfg.training.tensorboard_log_dir,
            prefer_gpu=cfg.compute.prefer_gpu,
        )

        # Phase 4: Evaluation
        wf.add_task(
            "eval_phase",
            run_evaluation,
            dependencies=["train_phase", "env_builder"],
        )

        # Phase 5: Pool update
        wf.add_task(
            "pool_update",
            update_pool,
            dependencies=["eval_phase", "train_phase"],
            top_k_promote=cfg.training.top_k_promote,
            bottom_k_demote=cfg.training.bottom_k_demote,
        )

        # Phase 6: Validation
        wf.add_task(
            "validation",
            run_validation,
            dependencies=["pool_update", "env_builder"],
            bootstrap_samples=cfg.validation.bootstrap_samples,
            confidence_level=cfg.validation.confidence_level,
            min_sharpe=cfg.validation.min_sharpe,
            max_drawdown_pct=cfg.validation.max_drawdown_pct,
            min_win_rate=cfg.validation.min_win_rate,
            min_profit_factor=cfg.validation.min_profit_factor,
            walk_forward_windows=cfg.validation.walk_forward_windows,
            min_wfe=cfg.validation.min_wfe,
        )

        self._workflow = wf
        return wf

    def run(self) -> dict[str, Any]:
        """Execute the full pipeline."""
        if self._workflow is None:
            self.build_workflow()

        logger.info("Starting Hydra training pipeline")
        self._results = self._workflow.execute()
        logger.info("Pipeline completed")

        return self._results

    def get_summary(self) -> dict[str, Any]:
        """Get pipeline execution summary."""
        if self._workflow is None:
            return {"status": "not_built"}

        summary = self._workflow.get_summary()

        # Add validation results if available
        validation = self._results.get("validation", {})
        if validation:
            summary["passed_agents"] = validation.get("passed_agents", [])

        return summary
