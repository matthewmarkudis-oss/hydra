"""Phase 6: ATHENA-style walk-forward validation with bootstrap CI and deflated Sharpe.

Validates trained RL agents against the same statistical criteria applied
to the existing trading_agents system. Uses ported ATHENA DSR/PSR and
KRONOS WFE overfitting detection. Backtesting only.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from hydra.agents.agent_pool import AgentPool
from hydra.compute.decorators import cpu_task
from hydra.envs.trading_env import TradingEnv
from hydra.evaluation.statistical_tests import (
    bootstrap_sharpe_ci,
    compute_wfe,
    deflated_sharpe_ratio,
    diagnose_wfe,
    probabilistic_sharpe_ratio,
    return_statistics,
    run_full_calibration,
)
from hydra.evaluation.fitness import AgentFitness, compute_fitness, rank_agents

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
    data_result = deps.get("data_prep", env_result.get("data_prep", {}))

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

    best_fitness = -1.0
    best_price_history = None
    best_trade_signals = None

    for agent in pool.get_all():
        agent_result = _validate_agent(
            agent, test_env, walk_forward_windows, bootstrap_samples, confidence_level
        )

        # Check against thresholds
        passed = _check_thresholds(agent_result, thresholds)
        agent_result["passed"] = passed

        # Track best agent's price history for dashboard
        fitness = agent_result.get("fitness_score", 0)
        if fitness > best_fitness:
            best_fitness = fitness
            best_price_history = agent_result.pop("_price_history", None)
            best_trade_signals = agent_result.pop("_trade_signals", None)
        else:
            agent_result.pop("_price_history", None)
            agent_result.pop("_trade_signals", None)

        results[agent.name] = agent_result

        status = "PASS" if passed else "FAIL"
        logger.info(
            f"  {agent.name}: {status} | sharpe={agent_result['sharpe']:.3f}, "
            f"mdd={agent_result['max_drawdown']:.2%}, wr={agent_result['win_rate']:.2%}, "
            f"WFE={agent_result['wfe']:.2f}, PSR={agent_result.get('psr', 0):.3f}, "
            f"DSR={agent_result.get('dsr', 0):.3f}, "
            f"fitness={agent_result.get('fitness_score', 0):.4f}"
        )

    passed_agents = [n for n, r in results.items() if r.get("passed")]
    logger.info(f"Validation: {len(passed_agents)}/{len(results)} agents passed")

    # Compute benchmark metrics from SPY data
    benchmark_data = data_result.get("benchmark_data", {})
    benchmark_result = _compute_benchmark(benchmark_data)

    return {
        "agent_results": results,
        "passed_agents": passed_agents,
        "thresholds": thresholds,
        "price_history": best_price_history,
        "trade_signals": best_trade_signals,
        "benchmark": benchmark_result,
    }


def _validate_agent(
    agent,
    env: TradingEnv,
    num_windows: int,
    bootstrap_samples: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Validate a single agent across walk-forward windows.

    Uses ported ATHENA DSR/PSR, KRONOS WFE, and CHIMERA fitness decomposition.
    """
    window_returns = []
    window_sharpes = []
    all_step_returns = []
    winning_steps = 0
    total_steps = 0
    total_trades = 0

    # Capture price history + trade signals from first episode of first window
    price_history_raw = []  # [{step, close_prices, actions}]
    capture_done = False

    episodes_per_window = max(5, 20 // num_windows)

    for window in range(num_windows):
        window_reward = 0.0

        for ep_idx in range(episodes_per_window):
            # Per-agent seed variation ensures even similar agents explore
            # different state trajectories, making validation more robust
            # and preventing identical metrics from near-identical weights.
            agent_seed = 42 + window * 1000 + hash(agent.name) % 100
            obs, info = env.reset(seed=agent_seed)
            episode_returns = []
            # Capture first episode of first window
            capturing = (window == 0 and ep_idx == 0 and not capture_done)

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

                # Record price and action for dashboard chart
                if capturing:
                    step_num = step_info.get("step", total_steps)
                    # Get close prices from state builder if available
                    try:
                        close_prices = env.state_builder.get_current_prices(
                            min(step_num, env.episode_bars - 1)
                        ).tolist()
                    except Exception:
                        close_prices = []
                    price_history_raw.append({
                        "step": step_num,
                        "close_prices": close_prices,
                        "actions": action.tolist() if hasattr(action, 'tolist') else list(action),
                    })

                if terminated or truncated:
                    summary = step_info.get("episode_summary", {})
                    window_sharpes.append(summary.get("sharpe_ratio", 0.0))
                    window_returns.append(summary.get("total_return", 0.0))
                    # Use actual executed trade count from MarketSimulator
                    total_trades += step_info.get("num_trades", 0)
                    if capturing:
                        capture_done = True
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

    # --- KRONOS WFE overfitting detection ---
    wfe_val = 0.0
    wfe_diagnosis = {"severity": "no_data", "diagnosis": "N/A", "recommendation": "N/A"}
    if len(window_sharpes) >= 2:
        mid = len(window_sharpes) // 2
        is_sharpe = float(np.mean(window_sharpes[:mid]))
        oos_sharpe = float(np.mean(window_sharpes[mid:]))
        wfe_val = compute_wfe(oos_sharpe, is_sharpe)
        wfe_diagnosis = diagnose_wfe(wfe_val)

    # --- ATHENA statistical calibration ---
    calibration = None
    if len(returns_arr) >= 10:
        calibration = run_full_calibration(
            oos_returns=returns_arr,
            n_trials=max(num_windows, 2),
            confidence=confidence_level,
            n_bootstrap=bootstrap_samples,
            is_sharpe=is_sharpe if len(window_sharpes) >= 2 else None,
        )

    # Extract PSR/DSR from calibration
    psr_val = calibration["probabilistic_sharpe"]["psr"] if calibration else 0.0
    dsr_val = calibration["deflated_sharpe"]["dsr"] if calibration else 0.0
    ci_low = calibration["bootstrap"]["ci_lower"] if calibration else 0.0
    ci_high = calibration["bootstrap"]["ci_upper"] if calibration else 0.0

    # --- CHIMERA fitness decomposition ---
    consistency = sum(1 for s in window_sharpes if s > 0) / max(len(window_sharpes), 1)
    agent_fitness = AgentFitness(
        agent_name=agent.name,
        sharpe=sharpe,
        max_drawdown=max_dd,
        profit_factor=profit_factor,
        wfe=wfe_val,
        consistency=consistency,
        window_sharpes=window_sharpes,
        total_trades=total_trades,
        total_return=float(np.mean(window_returns)) if window_returns else 0.0,
        win_rate=win_rate,
    )
    fitness_score, fitness_breakdown = compute_fitness(agent_fitness)

    # Downsample price history (every Nth bar to keep JSON small, target ~200 points)
    downsample_n = max(1, len(price_history_raw) // 200)
    price_history = price_history_raw[::downsample_n]

    # Extract trade signals: steps where any action magnitude > 0.01
    trade_signals = []
    for entry in price_history:
        actions = entry.get("actions", [])
        for ticker_idx, act in enumerate(actions):
            if abs(act) > 0.01:
                trade_signals.append({
                    "step": entry["step"],
                    "ticker_idx": ticker_idx,
                    "action": act,
                    "type": "buy" if act > 0 else "sell",
                })

    return {
        "sharpe": sharpe,
        "sharpe_ci_low": ci_low,
        "sharpe_ci_high": ci_high,
        "psr": psr_val,
        "dsr": dsr_val,
        "deflated_sharpe": dsr_val,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "wfe": wfe_val,
        "wfe_diagnosis": wfe_diagnosis,
        "fitness_score": fitness_score,
        "fitness_breakdown": fitness_breakdown,
        "calibration_verdict": calibration["verdict"] if calibration else "N/A",
        "total_return": float(np.mean(window_returns)) if window_returns else 0.0,
        "num_windows": num_windows,
        "total_steps": total_steps,
        "total_trades": total_trades,
        "consistency": consistency,
        # Private fields — extracted by run_validation(), not stored per-agent
        "_price_history": price_history,
        "_trade_signals": trade_signals,
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
    """Walk-forward efficiency — legacy wrapper, uses hydra.evaluation.statistical_tests."""
    if len(window_sharpes) < 2:
        return 0.0
    mid = len(window_sharpes) // 2
    is_sharpe = float(np.mean(window_sharpes[:mid]))
    oos_sharpe = float(np.mean(window_sharpes[mid:]))
    return compute_wfe(oos_sharpe, is_sharpe)


def _compute_benchmark(benchmark_data: dict) -> dict:
    """Compute buy-and-hold metrics for SPY benchmark.

    Returns a dict matching the agent validation format for side-by-side comparison.
    """
    if not benchmark_data or "close" not in benchmark_data:
        return {
            "ticker": "N/A",
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "equity_curve": [],
        }

    close = np.array(benchmark_data["close"], dtype=np.float32)
    if len(close) < 2:
        return {
            "ticker": benchmark_data.get("ticker", "N/A"),
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "equity_curve": [],
        }

    # Buy-and-hold return
    total_return = float((close[-1] / close[0]) - 1.0)

    # Step returns
    returns = np.diff(close) / close[:-1]

    # Sharpe ratio (annualized, assuming 5-min bars: 78 bars/day, 252 days/year)
    bars_per_year = 78 * 252
    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    sharpe = (mean_ret / max(std_ret, 1e-8)) * np.sqrt(bars_per_year)

    # Max drawdown
    equity = close / close[0]  # Normalize to 1.0
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / np.maximum(running_max, 1e-8)
    max_dd = float(-np.min(drawdowns))

    # Downsample equity curve for dashboard (target ~200 points)
    downsample_n = max(1, len(equity) // 200)
    equity_curve = equity[::downsample_n].tolist()

    return {
        "ticker": benchmark_data.get("ticker", "SPY"),
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "equity_curve": equity_curve,
    }
