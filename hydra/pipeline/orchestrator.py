"""Master pipeline controller — chains phases 1-6 via workflow DAG.

Supports checkpoint/resume for long-running training sessions.
Backtesting and training only.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

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

    def __init__(
        self,
        config: HydraConfig | None = None,
        alpaca_config: dict | None = None,
        use_real_data: bool = False,
        resume_checkpoint: str | None = None,
        pre_run_hook: Callable[[HydraConfig], bool] | None = None,
        post_run_hook: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.config = config or HydraConfig()
        self.alpaca_config = alpaca_config
        self.use_real_data = use_real_data
        self.resume_checkpoint = resume_checkpoint
        self._pre_run_hook = pre_run_hook
        self._post_run_hook = post_run_hook
        self._workflow: Workflow | None = None
        self._results: dict[str, Any] = {}

    def build_workflow(self) -> Workflow:
        """Construct the pipeline DAG."""
        cfg = self.config

        wf = Workflow(name="hydra_training_pipeline")

        # Phase 1: Data preparation
        use_synthetic = not self.use_real_data

        data_prep_kwargs: dict[str, Any] = dict(
            tickers=cfg.data.tickers,
            episode_bars=cfg.env.episode_bars,
            cache_dir=cfg.data.cache_dir,
            use_synthetic=use_synthetic,
            seed=cfg.seed,
        )

        if self.use_real_data:
            from datetime import date, timedelta
            end_date = date.today()
            # ~60 trading days = ~84 calendar days
            start_date = end_date - timedelta(days=int(cfg.data.lookback_days * 1.4))
            data_prep_kwargs["start_date"] = start_date
            data_prep_kwargs["end_date"] = end_date
            data_prep_kwargs["adapter_config"] = self.alpaca_config
            logger.info(
                f"Real data mode: {len(cfg.data.tickers)} tickers, "
                f"{start_date} to {end_date}"
            )

        wf.add_task(
            "data_prep",
            prepare_data,
            **data_prep_kwargs,
        )

        # Phase 2: Environment construction
        wf.add_task(
            "env_builder",
            build_environments,
            dependencies=["data_prep"],
            num_stocks=cfg.env.num_stocks,
            episode_bars=cfg.env.episode_bars,
            initial_cash=cfg.env.initial_cash,
            transaction_cost_bps=cfg.env.transaction_cost_bps,
            slippage_bps=cfg.env.slippage_bps,
            spread_bps=cfg.env.spread_bps,
            max_position_pct=cfg.env.max_position_pct,
            max_drawdown_pct=cfg.env.max_drawdown_pct,
            max_daily_loss_pct=cfg.env.max_daily_loss_pct,
            normalize_obs=cfg.env.normalize_obs,
            sharpe_eta=cfg.reward.sharpe_eta,
            drawdown_penalty=cfg.reward.drawdown_penalty,
            transaction_penalty=cfg.reward.transaction_penalty,
            holding_penalty=cfg.reward.holding_penalty,
            pnl_bonus_weight=cfg.reward.pnl_bonus_weight,
            reward_scale=cfg.reward.reward_scale,
            seed=cfg.seed,
        )

        # Phase 3: Training
        train_kwargs: dict[str, Any] = dict(
            num_generations=cfg.training.num_generations,
            episodes_per_generation=cfg.training.episodes_per_generation,
            top_k_promote=cfg.training.top_k_promote,
            bottom_k_demote=cfg.training.bottom_k_demote,
            max_pool_size=cfg.training.max_pool_size,
            checkpoint_dir=cfg.training.checkpoint_dir,
            tensorboard_dir=cfg.training.tensorboard_log_dir,
            prefer_gpu=cfg.compute.prefer_gpu,
        )
        if self.resume_checkpoint:
            train_kwargs["resume_from"] = self.resume_checkpoint

        wf.add_task(
            "train_phase",
            run_training,
            dependencies=["env_builder", "data_prep"],
            **train_kwargs,
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

        # Phase 6: Validation (also needs data_prep for benchmark data)
        wf.add_task(
            "validation",
            run_validation,
            dependencies=["pool_update", "env_builder", "data_prep"],
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
        """Execute the full pipeline.

        If a pre_run_hook is set, it is called with the config before
        execution. If it returns False, the pipeline is aborted.
        If a post_run_hook is set, it is called with the results after
        execution completes.
        """
        if self._workflow is None:
            self.build_workflow()

        # Pre-run hook (corp layer uses this for config validation)
        if self._pre_run_hook is not None:
            should_run = self._pre_run_hook(self.config)
            if not should_run:
                logger.warning("Pipeline aborted by pre-run hook")
                return {"aborted": True, "reason": "pre_run_hook returned False"}

        logger.info("Starting Hydra training pipeline")
        self._results = self._workflow.execute()
        logger.info("Pipeline completed")

        # Post-run hook (corp layer uses this for result analysis)
        if self._post_run_hook is not None:
            try:
                self._post_run_hook(self._results)
            except Exception as e:
                logger.error(f"Post-run hook failed: {e}")

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
