# Hydra — Multi-Agent RL Trading System

## Project Overview
Multi-agent reinforcement learning system for intraday US equity trading.
Sibling project to `trading_agents/` — imports data and validation infrastructure from it.

## Architecture
- **Environments**: Gymnasium-compatible, numpy-optimized, 5-min bar intraday episodes
- **Agents**: PPO, SAC, A2C (via stable-baselines3) + static snapshots + rule-based adapters
- **Training**: Population-based training with diverse agent pool (DeepMind cooperative approach)
- **Compute**: DirectML GPU (AMD 6900XT) + CPU orchestration, no CUDA
- **Validation**: VectorBT backtesting + ATHENA walk-forward criteria

## Hardware
- AMD Ryzen 7 5800X (8-core/16-thread) — CPU workers capped at 6
- AMD 6900XT (16GB VRAM) — via `torch-directml`, NOT CUDA
- Use `torch_directml.device()` for GPU tensors

## Key Constraints
- All environment internals use float32 numpy arrays (not float64)
- Episode = 1 trading day, ~78 steps (5-min bars, 9:30–16:00 ET)
- Risk constraints enforced inside `env.step()`, not post-hoc
- No lookahead bias — data indexed by step within episode
- VectorBT used for validation only, not training

## Testing
- Run tests: `pytest tests/ -v`
- Environment must pass `gymnasium.utils.env_checker.check_env()`
- Performance target: >10K env steps/sec on CPU

## Imports from trading_agents
- `trading_agents.data.market_data.MarketDataProvider` — Alpaca bar fetching
- `trading_agents.backtesting.intraday_backtester.IntradayBacktester` — VectorBT validation
- `trading_agents.agents.base_agent.TradeSignal, MarketContext, BaseAgent` — agent interfaces
- `trading_agents.agents.alpha_momentum.AlphaMomentum` — rule-based pool member
- `trading_agents.agents.beta_mean_reversion.BetaMeanReversion` — rule-based pool member
