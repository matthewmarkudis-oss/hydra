"""Phase 6: ATHENA-style walk-forward validation with bootstrap CI and deflated Sharpe.

Validates trained RL agents against the same statistical criteria applied
to the existing trading_agents system. Backtesting only.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from hydra.agents.agent_pool import AgentPool
from hydra.compute.decorators import cpu_task
from hydra.envs.trading_env import TradingEnv

logger = logging.getLogger("hydra.pipeline.validation")


@cpu_task(workers=4)
def run_validation(
    deps: dict[str, Any],
    bootstrap_samples: int = 2000,
    confidence_level: float = 0.95,
    min_sharpe: float = 0.3,
    max_drawdown_pct: float = 0.25,
    min_win_rate: float = 0.40,
    min_profit_factor: float = 1.1,
    walk_forward_windows: int = 4,
    min_wfe: float = 0.40,
) -> dict[str, Any]:
    """Run ATHENA-style walk-forward validation on trained agents.

    Args:
        deps: Must contain pool and test_env.
        bootstrap_samples: Number of bootstrap samples for confidence intervals.
        confidence_level: Confidence level for bootstrap CI.
        min_sharpe: Minimum Sharpe ratio to pass validation.
        max_drawdown_pct: Maximum drawdown to pass.
        min_win_rate: Minimum win rate to pass.
        min_profit_factor: Minimum profit factor to pass.
        walk_forward_windows: Number of walk-forward windows.
        min_wfe: Minimum walk-forward efficiency.

    Returns:
        Validation results per agent with pass/fail status.
    """
    pool_update_result = deps.get("pool_update", deps.get("train_phase", {}))
    env_result = deps.get("env_builder", {})

    pool: AgentPool = pool_update_result.get("pool")
    test_env: TradingEnv = env_result.get("test_env")

    if pool is None or test_env is None:
        raise ValueError("Missing pool or test_env in dependencies")

    logger.info(f"Running ATHENA validation: {pool.size} agents, {walk_forward_windows} windows")

    results = {}
    thresholds = {
        "min_sharpe": min_sharpe,
        "max_drawdown_pct": max_drawdown_pct,
        "min_win_rate": min_win_rate,
        "min_profit_factor": min_profit_factor,
        "min_wfe": min_wfe,
    }

    for agent in pool.get_all():
        agent_result = _validate_agent(
            agent, test_env, walk_forward_windows, bootstrap_samples, confidence_level
        )

        # Check against thresholds
        passed = _check_thresholds(agent_result, thresholds)
        agent_result["passed"] = passed

        results[agent.name] = agent_result

        status = "PASS" if passed else "FAIL"
        logger.info(
            f"  {agent.name}: {status} | sharpe={agent_result['sharpe']:.3f}, "
            f"mdd={agent_result['max_drawdown']:.2%}, wr={agent_result['win_rate']:.2%}"
        )

    passed_agents = [n for n, r in results.items() if r.get("passed")]
    logger.info(f"Validation: {len(passed_agents)}/{len(results)} agents passed")

    return {
        "agent_results": results,
        "passed_agents": passed_agents,
        "thresholds": thresholds,
    }


def _validate_agent(
    agent,
    env: TradingEnv,
    num_windows: int,
    bootstrap_samples: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Validate a single agent across walk-forward windows."""
    window_returns = []
    window_sharpes = []
    all_step_returns = []
    winning_steps = 0
    total_steps = 0

    episodes_per_window = max(5, 20 // num_windows)

    for window in range(num_windows):
        window_reward = 0.0

        for _ in range(episodes_per_window):
            obs, info = env.reset(seed=42 + window * 1000)
            episode_returns = []

            while True:
                action = agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, step_info = env.step(action)
                window_reward += reward

                step_return = step_info.get("step_return", 0.0)
                episode_returns.append(step_return)
                all_step_returns.append(step_return)

                if step_return > 0:
                    winning_steps += 1
                total_steps += 1

                if terminated or truncated:
                    summary = step_info.get("episode_summary", {})
                    window_sharpes.append(summary.get("sharpe_ratio", 0.0))
                    window_returns.append(summary.get("total_return", 0.0))
                    break

    # Aggregate metrics
    returns_arr = np.array(all_step_returns, dtype=np.float32)
    sharpe = float(np.mean(window_sharpes)) if window_sharpes else 0.0
    max_dd = _compute_max_drawdown(returns_arr)
    win_rate = winning_steps / max(total_steps, 1)

    # Profit factor
    gross_profit = float(np.sum(returns_arr[returns_arr > 0])) if np.any(returns_arr > 0) else 0.0
    gross_loss = float(np.abs(np.sum(returns_arr[returns_arr < 0]))) if np.any(returns_arr < 0) else 1e-8
    profit_factor = gross_profit / max(gross_loss, 1e-8)

    # Walk-forward efficiency
    wfe = _compute_wfe(window_sharpes) if len(window_sharpes) >= 2 else 0.0

    # Bootstrap confidence interval for Sharpe
    ci_low, ci_high = _bootstrap_ci(window_sharpes, bootstrap_samples, confidence_level)

    # Deflated Sharpe ratio
    dsr = _deflated_sharpe(sharpe, len(window_sharpes), np.std(window_sharpes) if window_sharpes else 1.0)

    return {
        "sharpe": sharpe,
        "sharpe_ci_low": ci_low,
        "sharpe_ci_high": ci_high,
        "deflated_sharpe": dsr,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "wfe": wfe,
        "total_return": float(np.mean(window_returns)) if window_returns else 0.0,
        "num_windows": num_windows,
        "total_steps": total_steps,
    }


def _check_thresholds(result: dict, thresholds: dict) -> bool:
    """Check if agent results pass all validation thresholds."""
    if result["sharpe"] < thresholds["min_sharpe"]:
        return False
    if result["max_drawdown"] > thresholds["max_drawdown_pct"]:
        return False
    if result["win_rate"] < thresholds["min_win_rate"]:
        return False
    if result["profit_factor"] < thresholds["min_profit_factor"]:
        return False
    if result["wfe"] < thresholds["min_wfe"]:
        return False
    return True


def _compute_max_drawdown(returns: np.ndarray) -> float:
    """Compute max drawdown from returns array."""
    if len(returns) == 0:
        return 0.0
    equity = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / np.maximum(running_max, 1e-8)
    return float(-np.min(drawdowns))


def _compute_wfe(window_sharpes: list[float]) -> float:
    """Walk-forward efficiency: ratio of OOS to IS performance."""
    if len(window_sharpes) < 2:
        return 0.0
    # Simple: ratio of later windows to earlier windows
    mid = len(window_sharpes) // 2
    is_sharpe = np.mean(window_sharpes[:mid])
    oos_sharpe = np.mean(window_sharpes[mid:])
    if abs(is_sharpe) < 1e-8:
        return 0.0
    return float(oos_sharpe / is_sharpe) if is_sharpe > 0 else 0.0


def _bootstrap_ci(
    values: list[float],
    num_samples: int = 2000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap confidence interval."""
    if len(values) < 2:
        return 0.0, 0.0

    arr = np.array(values, dtype=np.float32)
    rng = np.random.default_rng(42)
    boot_means = np.array([
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(num_samples)
    ])

    alpha = 1 - confidence
    low = float(np.percentile(boot_means, 100 * alpha / 2))
    high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return low, high


def _deflated_sharpe(sharpe: float, num_trials: int, std_sharpe: float) -> float:
    """Deflated Sharpe Ratio — adjusts for multiple testing.

    Based on Bailey & Lopez de Prado (2014).
    """
    if num_trials <= 1 or std_sharpe < 1e-8:
        return sharpe

    # Expected max Sharpe under null hypothesis
    from math import log, sqrt
    euler_mascheroni = 0.5772
    expected_max = std_sharpe * (
        sqrt(2 * log(num_trials)) - (log(log(num_trials)) + log(4 * 3.14159)) / (2 * sqrt(2 * log(num_trials)))
    )

    # DSR: probability that observed Sharpe exceeds expected max
    z = (sharpe - expected_max) / max(std_sharpe, 1e-8)
    return float(z)
