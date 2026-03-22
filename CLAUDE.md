# Hydra -- Multi-Agent RL Trading System

## What Is This?

Hydra is an **automated trading strategy factory**. It breeds a population of AI trading agents, trains them on real market data, and uses statistical tests to determine which ones are genuinely profitable vs. just lucky.

Think of it like this:

```
  Real Market Data (Alpaca)          Trained Agents
        |                                  ^
        v                                  |
  +------------------+    +----------------------------------+
  | 10 Sector ETFs   | -> | Train -> Evolve -> Validate      |
  | 60 days, 5m bars |    | (repeat for N generations)       |
  +------------------+    +----------------------------------+
                                           |
                                     Dashboard
                                   localhost:5010
```

**In plain English:** Download stock data -> train AI agents -> test if they actually work -> keep the good ones, discard the bad ones -> repeat.

---

## Architecture Overview

```
+------------------------------------------------------------------+
|                        HYDRA SYSTEM                               |
|                                                                   |
|  +-----------+   +----------+   +---------+   +---------------+   |
|  |   DATA    |-->|   ENV    |-->| AGENTS  |-->|   TRAINING    |   |
|  | Alpaca API|   | Trading  |   | PPO,SAC |   | Population-   |   |
|  | Features  |   | Gym Env  |   | Static  |   | based w/      |   |
|  | Indicators|   | Simulator|   | Rules   |   | generations   |   |
|  +-----------+   +----------+   +---------+   +-------+-------+   |
|                                                       |           |
|  +----------------------------------------------------v-------+   |
|  |                    PROTOCOL ENGINES                        |   |
|  |                                                            |   |
|  |  CHIMERA        PROMETHEUS      ELEOS       ATHENA/KRONOS  |   |
|  |  Diagnostics    Competition     Conviction  Validation     |   |
|  |  + Mutations    Weights         Bayesian    PSR/DSR/WFE    |   |
|  |  + Fitness      Rebalancing     Calibration Overfitting    |   |
|  +------------------------------------------------------------+   |
|                           |                                       |
|                    +------v------+                                 |
|                    |  DASHBOARD  |                                 |
|                    | :5010       |                                 |
|                    +-------------+                                 |
+------------------------------------------------------------------+
```

---

## How Training Works (The Loop)

Each "generation" is one cycle of train-evaluate-evolve. The system runs N generations (default 10).

```
  Generation 1          Generation 2              Generation N
  +----------+          +----------+              +----------+
  |1. Train  |    +---> |1. Train  |    +---->    |1. Train  |
  |2. Learn  |    |     |2. Learn  |    |         |2. Learn  |
  |3. Evaluate|   |     |3. Evaluate|   |         |3. Evaluate|
  |4. Diagnose|   |     |4. Diagnose|   |         |4. Diagnose|
  |5. Rebalance|  |     |5. Rebalance|  |         |5. Rebalance|
  |6. Promote |   |     |6. Promote |   |         |6. Promote |
  |7. Demote  +---+     |7. Demote  +---+         |7. Validate|
  +----------+          +----------+              +----------+
                                                       |
                                                  Pass / Fail
```

**Step by step:**

| Step | What Happens | Why |
|------|-------------|-----|
| 1. Train | All agents trade in the environment for 30 episodes | Collect experience |
| 2. Learn | PPO/SAC agents do gradient descent (neural network weight updates) | Actual learning |
| 3. Evaluate | Each agent trades solo, measure average reward | See who's best |
| 4. Diagnose | CHIMERA checks for problems (no trades? overfitting? stagnation?) | Auto-fix issues |
| 5. Rebalance | PROMETHEUS adjusts how much capital each agent gets based on rank | Reward winners |
| 6. Promote | Top agents get frozen as "static" snapshots (preserve what works) | Save progress |
| 7. Demote | Worst agents get removed from the pool | Cut losses |

After all generations, ATHENA validation runs the final statistical tests.

---

## The Five Protocols

Hydra integrates five named protocol engines, each doing one specific job:

```
+------------------------------------------------------------------+
|                                                                    |
|   CHIMERA            "The Doctor"                                  |
|   +---------------------------------------------------------+     |
|   | Diagnoses what's wrong with training each generation     |     |
|   | Prescribes mutations: loosen risk, adjust penalties, etc |     |
|   | Computes multi-objective fitness scores (0 to 1)         |     |
|   +---------------------------------------------------------+     |
|                                                                    |
|   PROMETHEUS          "The Judge"                                  |
|   +---------------------------------------------------------+     |
|   | Ranks agents by composite score (Sharpe + WR + PF)       |     |
|   | Rebalances portfolio weights: winners get more capital    |     |
|   | Detects convergence (weights stabilized = done)           |     |
|   +---------------------------------------------------------+     |
|                                                                    |
|   ELEOS               "The Skeptic"                                |
|   +---------------------------------------------------------+     |
|   | Tracks each agent's win/loss history (Bayesian stats)    |     |
|   | Dampens weights for agents on lucky streaks              |     |
|   | Boosts weights for consistently profitable agents        |     |
|   | Max adjustment: +/-20%, conservative by design           |     |
|   +---------------------------------------------------------+     |
|                                                                    |
|   ATHENA              "The Gatekeeper"                             |
|   +---------------------------------------------------------+     |
|   | Final validation gate -- agents must pass ALL of:        |     |
|   |   Sharpe >= 0.3  (risk-adjusted return)                  |     |
|   |   Drawdown <= 25% (max loss from peak)                   |     |
|   |   Win Rate >= 40% (profitable trades)                    |     |
|   |   Profit Factor >= 1.1 (gross profit / gross loss)       |     |
|   |   WFE >= 0.40 (not overfitting, see KRONOS)              |     |
|   | Also computes PSR and DSR (statistical confidence)       |     |
|   +---------------------------------------------------------+     |
|                                                                    |
|   KRONOS              "The Overfitting Detector"                   |
|   +---------------------------------------------------------+     |
|   | Walk-Forward Efficiency = OOS performance / IS performance|    |
|   |   >= 0.60  "Good"     (real edge)                        |     |
|   |   >= 0.40  "OK"       (acceptable)                       |     |
|   |   >= 0.25  "Warning"  (possible overfitting)             |     |
|   |   < 0.25   "Critical" (strategy is overfit)              |     |
|   +---------------------------------------------------------+     |
|                                                                    |
+------------------------------------------------------------------+
```

---

## Data Flow

```
Alpaca API (IEX feed)
    |
    v
+-------------------+     +------------------+     +------------------+
| data/adapter.py   | --> | data/indicators  | --> | data/feature     |
| Fetch OHLCV bars  |     | RSI, MACD, BB,   |     | store.py         |
| 10 tickers x 60d  |     | ATR, VWAP, etc.  |     | Organized arrays |
+-------------------+     +------------------+     +--------+---------+
                                                            |
                          +------------------+              |
                          | envs/trading_env | <------------+
                          | Gym environment  |
                          | 78 steps/episode |
                          | (one trading day)|
                          +--------+---------+
                                   |
                    +--------------+--------------+
                    |              |              |
               +---------+  +---------+  +-----------+
               |  PPO    |  |  SAC    |  | Rule-Based|
               | Agent 1 |  | Agent 1 |  |  Alpha    |
               +---------+  +---------+  +-----------+
```

**Tickers (Sector ETFs):** XLK, XLF, XLE, XLV, XLI, XLU, XLP, XLY, XLB, XLRE

Each ticker = one sector of the US economy (tech, finance, energy, healthcare, etc.).

---

## Agent Types

```
+-------------------+--------------------------------------------------+
| Type              | Description                                      |
+-------------------+--------------------------------------------------+
| PPO Agent         | Neural network, learns via policy gradients      |
|                   | (Proximal Policy Optimization, via SB3)          |
+-------------------+--------------------------------------------------+
| SAC Agent         | Neural network, learns via entropy-regularized   |
|                   | Q-learning (Soft Actor-Critic, via SB3)          |
+-------------------+--------------------------------------------------+
| Static Agent      | Frozen snapshot of a PPO/SAC at a point in time  |
|                   | Preserves what worked, doesn't learn further     |
+-------------------+--------------------------------------------------+
| Rule-Based Agent  | Hand-coded strategies (momentum, mean-reversion) |
|                   | Imported from trading_agents/ project            |
+-------------------+--------------------------------------------------+
```

The **agent pool** starts with 2 PPO + 2 SAC + 2 rule-based agents. Over generations, good agents get promoted to static snapshots, bad ones get removed, and the pool grows/shrinks organically.

---

## Pipeline Phases

```
train.py
   |
   v
+------------------------------------------------------------------+
| PipelineOrchestrator                                              |
|                                                                   |
|  Phase 1: DATA PREP                                               |
|  [Fetch from Alpaca] --> [Compute indicators] --> [Feature store] |
|                                                                   |
|  Phase 2: ENV BUILDER                                             |
|  [Create train env] + [Create test env]                           |
|                                                                   |
|  Phase 3: TRAIN                                                   |
|  [PopulationTrainer.train()]                                      |
|  Runs N generations with CHIMERA + PROMETHEUS + ELEOS             |
|                                                                   |
|  Phase 4: EVAL                                                    |
|  [Individual agent evaluation on held-out data]                   |
|                                                                   |
|  Phase 5: POOL UPDATE                                             |
|  [Final promotions/demotions based on eval scores]                |
|                                                                   |
|  Phase 6: VALIDATION (ATHENA)                                     |
|  [Walk-forward validation with PSR/DSR/WFE]                      |
|  [Pass/Fail gate for each agent]                                  |
|                                                                   |
+------------------------------------------------------------------+
        |
        v
  logs/hydra_training_state.json  -->  Dashboard (:5010)
```

---

## Module Map

```
hydra/
+-- agents/           Agent implementations (PPO, SAC, static, rules, pool)
+-- compute/          GPU/CPU execution (DirectML, workflow engine)
+-- config/           Settings and defaults
+-- data/             Market data fetching + feature engineering
+-- envs/             Gymnasium trading environments
+-- evaluation/       Protocol engines:
|   +-- competition.py    PROMETHEUS (weight rebalancing)
|   +-- conviction.py     ELEOS (Bayesian calibration)
|   +-- fitness.py        CHIMERA (multi-objective scoring)
|   +-- statistical_tests.py  ATHENA (PSR/DSR) + KRONOS (WFE)
+-- evolution/         Adaptive evolution:
|   +-- diagnostics.py    CHIMERA (problem detection)
|   +-- mutation_engine.py CHIMERA (16 mutation types)
+-- pipeline/          Orchestration (6 phases)
+-- risk/              Position limits, drawdown constraints
+-- training/          Training loop, population trainer, curriculum
+-- utils/             Logging, serialization, numpy optimizations
```

---

## CHIMERA Fitness Score (How Agents Are Scored)

```
Fitness = (weighted sum of components) x stability_multiplier

  Component       Weight    What It Measures
  +-----------+   ------    +------------------------------------------+
  | Sharpe    |   35%       | Return per unit of risk (higher = better) |
  | Drawdown  |   20%       | Worst peak-to-trough loss (lower = better)|
  | Profit Fac|   20%       | Gross profit / gross loss (>1 = profitable)|
  | WFE       |   15%       | Overfitting check (higher = more real)    |
  | Consistency|  10%       | % of windows with positive Sharpe         |
  +-----------+             +------------------------------------------+

  Stability multiplier penalizes inconsistent agents (high variance).
  Score range: 0.0 (terrible) to ~0.8 (excellent)
```

---

## Dashboard (localhost:5010)

```
+------------------------------------------------------------------+
|  HYDRA Training Dashboard                                         |
|------------------------------------------------------------------|
|  [*] Trained 3 generations on 10 real tickers -- improving +86.9  |
|                                                                   |
|  Generations: 3    Best Fitness: 0.000   Best Sharpe: 0.000       |
|  Agents Passed: 0/6                                               |
|------------------------------------------------------------------|
|  [Reward Trend Chart]        |  [Weight Evolution Chart]          |
|   -103 -> -37 -> -16        |   PROMETHEUS allocations           |
|------------------------------------------------------------------|
|  Generation History (CHIMERA + PROMETHEUS + ELEOS)                |
|  Gen | Reward | Diagnosis         | Mutations | Converged         |
|------------------------------------------------------------------|
|  Agent Validation (ATHENA + KRONOS)                               |
|  Agent | Sharpe | PSR | DSR | WFE | Win Rate | Result            |
|------------------------------------------------------------------|
|  Fitness Decomposition (CHIMERA)   |  Pipeline Timing             |
|  [stacked bars per agent]          |  [phase durations]           |
+------------------------------------------------------------------+
```

Auto-refreshes every 15 seconds. Run with: `python scripts/hydra_dashboard.py`

---

## Hardware

- AMD Ryzen 7 5800X (8-core/16-thread) -- CPU workers capped at 6
- AMD 6900XT (16GB VRAM) -- via `torch-directml`, NOT CUDA
- Use `torch_directml.device()` for GPU tensors
- DirectML static agent load has a known fallback to CPU (non-blocking)

## Key Constraints

- All environment internals use float32 numpy arrays (not float64)
- Episode = 1 trading day, ~78 steps (5-min bars, 9:30-16:00 ET)
- Risk constraints enforced inside `env.step()`, not post-hoc
- No lookahead bias -- data indexed by step within episode
- Alpaca IEX free tier has rate limiting (3s retry backoff)

## Running

```bash
# Train with real Alpaca data (10 sector ETFs, 10 generations)
python scripts/train.py --real-data --generations 10

# Train with specific tickers
python scripts/train.py --real-data --tickers XLK,XLF,XLE --generations 5

# Train with synthetic data (for testing)
python scripts/train.py --synthetic --generations 3

# Launch dashboard
python scripts/hydra_dashboard.py

# Run tests
pytest tests/ -v
```

## Imports from trading_agents

- `trading_agents.data.market_data.MarketDataProvider` -- Alpaca bar fetching
- `trading_agents.backtesting.intraday_backtester.IntradayBacktester` -- VectorBT validation
- `trading_agents.agents.base_agent.TradeSignal, MarketContext, BaseAgent` -- agent interfaces
- `trading_agents.agents.alpha_momentum.AlphaMomentum` -- rule-based pool member
- `trading_agents.agents.beta_mean_reversion.BetaMeanReversion` -- rule-based pool member
