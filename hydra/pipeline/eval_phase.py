"""Phase 4: Evaluation — export RL signals and run VectorBT backtest validation.

Uses vectorbt.Portfolio.from_signals() for independent cross-validation of
RL agent trading decisions with proper slippage and cost modelling.
Backtesting only.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from hydra.agents.agent_pool import AgentPool
from hydra.compute.decorators import cpu_task
from hydra.envs.trading_env import TradingEnv

logger = logging.getLogger("hydra.pipeline.eval_phase")


@cpu_task(workers=4)
def run_evaluation(
    deps: dict[str, Any],
    num_eval_episodes: int = 20,
    use_vectorbt: bool = True,
) -> dict[str, Any]:
    """Evaluate trained agents via RL environment and VectorBT backtest.

    Args:
        deps: Must contain 'train_phase' with pool, and 'env_builder' with val_env.
        num_eval_episodes: Number of evaluation episodes per agent.
        use_vectorbt: Whether to also run VectorBT validation.

    Returns:
        Evaluation results per agent.
    """
    train_result = deps.get("train_phase", {})
    env_result = deps.get("env_builder", {})

    pool: AgentPool = train_result.get("pool")
    val_env: TradingEnv = env_result.get("val_env")

    if pool is None or val_env is None:
        raise ValueError("Missing pool or val_env in dependencies")

    logger.info(f"Evaluating {pool.size} agents over {num_eval_episodes} episodes")

    # RL environment evaluation
    rl_results = _evaluate_in_env(pool, val_env, num_eval_episodes)

    # VectorBT cross-validation — backtest top agents with independent sim
    vbt_results = {}
    if use_vectorbt:
        vbt_results = _evaluate_vectorbt(pool, val_env, rl_results)

    return {
        "rl_eval": rl_results,
        "vbt_eval": vbt_results,
        "num_agents": pool.size,
    }


def _evaluate_in_env(
    pool: AgentPool,
    env: TradingEnv,
    num_episodes: int,
) -> dict[str, dict[str, float]]:
    """Evaluate each agent in the RL environment."""
    results = {}

    for agent in pool.get_all():
        episode_rewards = []
        episode_returns = []
        episode_sharpes = []

        for _ in range(num_episodes):
            obs, info = env.reset()
            total_reward = 0.0

            while True:
                action = agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, step_info = env.step(action)
                total_reward += reward

                if terminated or truncated:
                    summary = step_info.get("episode_summary", {})
                    episode_rewards.append(total_reward)
                    episode_returns.append(summary.get("total_return", 0.0))
                    episode_sharpes.append(summary.get("sharpe_ratio", 0.0))
                    break

        results[agent.name] = {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_return": float(np.mean(episode_returns)),
            "mean_sharpe": float(np.mean(episode_sharpes)),
            "num_episodes": num_episodes,
        }

        logger.info(
            f"  {agent.name}: reward={results[agent.name]['mean_reward']:.4f}, "
            f"return={results[agent.name]['mean_return']:.4%}, "
            f"sharpe={results[agent.name]['mean_sharpe']:.3f}"
        )

    return results


def _evaluate_vectorbt(
    pool: AgentPool,
    env: TradingEnv,
    rl_results: dict[str, dict[str, float]],
    top_n: int = 5,
    num_episodes: int = 5,
    threshold: float = 0.3,
) -> dict[str, Any]:
    """Cross-validate top agents with VectorBT portfolio simulation.

    Runs each agent through the env, converts continuous actions to
    entry/exit boolean signals, and feeds them to vbt.Portfolio.from_signals()
    for an independent PnL calculation with slippage + fees.

    Args:
        pool: Agent pool with trained agents.
        env: Validation TradingEnv.
        rl_results: RL eval results (used to rank agents for top-N selection).
        top_n: Number of top agents to backtest.
        num_episodes: Episodes per agent for VBT evaluation.
        threshold: Action threshold for BUY (>thresh) / SELL (<-thresh).
    """
    try:
        import vectorbt as vbt  # noqa: F401
    except ImportError:
        logger.info("VectorBT not available, skipping cross-validation")
        return {"status": "unavailable"}

    # Select top agents by RL mean reward
    ranked = sorted(
        rl_results.items(),
        key=lambda x: x[1].get("mean_reward", 0),
        reverse=True,
    )
    top_names = [name for name, _ in ranked[:top_n]]
    logger.info(f"VectorBT cross-validation: {len(top_names)} agents, {num_episodes} episodes each")

    agent_results = {}
    for name in top_names:
        agent = pool.get(name)
        if agent is None:
            continue
        try:
            result = _vbt_backtest_agent(agent, env, num_episodes, threshold)
            agent_results[name] = result
            logger.info(
                f"  VBT {name}: return={result['vbt_return_pct']:.2f}%, "
                f"sharpe={result['vbt_sharpe']:.3f}, "
                f"trades={result['vbt_total_trades']}, "
                f"dd={result['vbt_max_dd_pct']:.2f}%"
            )
        except Exception as e:
            logger.warning(f"  VBT backtest failed for '{name}': {e}")

    return {"status": "available", "agent_results": agent_results}


def _vbt_backtest_agent(
    agent,
    env: TradingEnv,
    num_episodes: int = 5,
    threshold: float = 0.3,
) -> dict[str, float]:
    """Run one agent through the env, convert actions to VBT portfolio sim.

    Collects close prices + actions at each step across multiple episodes,
    then runs vbt.Portfolio.from_signals() per ticker and aggregates results.

    Returns dict with vbt_return_pct, vbt_sharpe, vbt_max_dd_pct, vbt_total_trades.
    """
    import pandas as pd
    import vectorbt as vbt

    per_ticker_results: list[dict] = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        tickers = info.get("tickers", [])
        actions_list: list[np.ndarray] = []
        closes_list: list[np.ndarray] = []
        step = 0

        while True:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)
            actions_list.append(action.copy())
            closes_list.append(
                env.state_builder.get_current_prices(
                    min(step, env.episode_bars - 1)
                ).copy()
            )
            step += 1
            if terminated or truncated:
                break

        if not actions_list:
            continue

        actions_arr = np.array(actions_list)   # (steps, num_stocks)
        closes_arr = np.array(closes_list)     # (steps, num_stocks)

        for i, ticker in enumerate(tickers):
            if i >= actions_arr.shape[1]:
                break

            close = pd.Series(closes_arr[:, i], dtype=np.float64)
            entries = pd.Series(actions_arr[:, i] > threshold)
            exits = pd.Series(actions_arr[:, i] < -threshold)

            if entries.sum() == 0:
                continue

            try:
                pf = vbt.Portfolio.from_signals(
                    close,
                    entries,
                    exits,
                    init_cash=10_000.0,
                    fees=0.0005,       # 5 bps — matches env transaction cost
                    slippage=0.001,    # 10 bps — matches env slippage
                )

                total_return = float(pf.total_return()) * 100
                try:
                    max_dd = abs(float(pf.max_drawdown())) * 100
                    max_dd = max_dd if np.isfinite(max_dd) else 0.0
                except Exception:
                    max_dd = 0.0
                try:
                    trade_count = int(pf.trades.count())
                except Exception:
                    trade_count = 0

                per_ticker_results.append({
                    "ticker": ticker,
                    "episode": ep,
                    "total_return_pct": total_return,
                    "max_drawdown_pct": max_dd,
                    "total_trades": trade_count,
                })
            except Exception as e:
                logger.debug(f"VBT sim failed for {ticker} ep{ep}: {e}")

    # Aggregate across episodes + tickers
    if not per_ticker_results:
        return {
            "vbt_return_pct": 0.0,
            "vbt_sharpe": 0.0,
            "vbt_max_dd_pct": 0.0,
            "vbt_total_trades": 0,
        }

    returns = [r["total_return_pct"] for r in per_ticker_results]
    std_ret = float(np.std(returns)) if len(returns) > 1 else 1e-8

    return {
        "vbt_return_pct": float(np.mean(returns)),
        "vbt_sharpe": float(np.mean(returns)) / max(std_ret, 1e-8),
        "vbt_max_dd_pct": float(np.max([r["max_drawdown_pct"] for r in per_ticker_results])),
        "vbt_total_trades": sum(r["total_trades"] for r in per_ticker_results),
    }


def export_signals(
    agent,
    env: TradingEnv,
    num_episodes: int = 10,
    threshold: float = 0.3,
) -> dict[str, list[dict]]:
    """Export RL agent actions as entry/exit signals for VectorBT.

    Converts continuous actions into discrete BUY/SELL signals.

    Args:
        agent: Trained RL agent.
        env: Trading environment.
        num_episodes: Episodes to export.
        threshold: Action threshold for signal generation.

    Returns:
        Dict of ticker → list of signal dicts with 'step', 'action', 'strength'.
    """
    signals: dict[str, list[dict]] = {}

    for ep in range(num_episodes):
        obs, info = env.reset()
        tickers = info.get("tickers", [])

        if not signals:
            signals = {t: [] for t in tickers}

        step = 0
        while True:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)

            for i, ticker in enumerate(tickers):
                if i < len(action):
                    if action[i] > threshold:
                        signals[ticker].append({
                            "episode": ep,
                            "step": step,
                            "action": "BUY",
                            "strength": float(action[i]),
                        })
                    elif action[i] < -threshold:
                        signals[ticker].append({
                            "episode": ep,
                            "step": step,
                            "action": "SELL",
                            "strength": float(-action[i]),
                        })

            step += 1
            if terminated or truncated:
                break

    return signals
