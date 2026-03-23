# ROADMAP: Strategic Intelligence Layer (Thesis Library)

Status: PLANNED — not yet implemented
Created: 2026-03-22
Priority: High (after curriculum training completes)

## Summary

Add a curated thesis library to the GeopoliticsExpert that captures
long-arc structural views from leading thinkers across geopolitics,
economics, demographics, energy, and complex systems. Headlines provide
timing signals; theses provide the strategic frame.

The system does NOT try to reconcile contradictory thinkers. Instead it
maintains separate thesis tracks and uses agreement/disagreement as signal:
- Agreement across independent thinkers = high-conviction structural view
- Disagreement = widen confidence interval, don't pick a side

## Architecture Concept

```
THESIS LIBRARY (curated, manually updated)
    │
    ├── structural_risks/     (1-5 year horizon)
    ├── timing_signals/       (weeks to months)
    └── sector_implications/  (overweight/underweight)
         │
         ▼
GEOPOLITICS EXPERT
    │
    ├── Daily headlines (NewsAPI + Finnhub)
    ├── Cross-reference headlines against standing theses
    ├── Agreement matrix (where do thinkers converge?)
    ├── Disagreement flags (where do they diverge?)
    │
    ▼
REGIME CLASSIFICATION (enhanced)
    │
    ├── Regime: risk_on / risk_off / crisis / antifragile
    ├── Structural sector bias (thesis-informed)
    ├── Conviction level (adjusted by agreement/disagreement)
    └── Thesis confirmations/invalidations (when news matches thesis)
```

## Thesis Entry Format

```yaml
id: zeihan_supply_chain
thinker: Peter Zeihan
claim: "Global supply chains are fragmenting permanently. US reshoring
        accelerates. Countries without demographic dividends lose
        manufacturing base."
category: structural_risk
time_horizon: 3-5 years
sector_implications:
  overweight: [XLI, XLE]  # industrials, energy
  underweight: [EEM]       # emerging markets
  specific: []
confidence: high
agreement_with: [Friedman, Brands]
disagrees_with: [Varoufakis]
last_reviewed: 2026-03
sources:
  - "The End of the World Is Just the Beginning (2022)"
  - "Disunited Nations (2020)"
```

---

## Priority Tier 1 — V1 Thesis Library (12 thinkers)

These have the highest signal-to-noise for tradeable macro views.

### 1. Ray Dalio — Debt Cycles
- Framework: Long-term debt cycle positioning, reserve currency transitions
- Key work: "Principles for Dealing with the Changing World Order"
- Tradeable signal: Where are we in the debt cycle? Late-cycle = defensive
- Affects: Regime classification (risk_on vs risk_off), bond vs equity tilt
- Category: monetary/debt

### 2. Peter Zeihan — Supply Chain & Demographics
- Framework: Demographic decline + geography determine economic destiny
- Key work: "The End of the World Is Just the Beginning"
- Tradeable signal: Reshoring winners (US industrials, energy), losers (China export model)
- Affects: Sector bias (XLI, XLE overweight), emerging market exposure
- Category: demographics/structural

### 3. Michael Pettis — China Rebalancing
- Framework: China's investment-driven growth model is structurally broken
- Key work: "Trade Wars Are Class Wars" (with Matthew Klein)
- Tradeable signal: China demand slowdown → commodity impact, global growth revision
- Affects: Global growth outlook, commodity demand, EM exposure
- Category: monetary/structural

### 4. Nassim Taleb — Fragility & Antifragility
- Framework: Fragility detection, fat tails, barbell strategy
- Key work: "Antifragile," "The Black Swan"
- Tradeable signal: Already integrated via THETA agent and antifragile regime
- Affects: Regime classification, tail risk hedging, position sizing
- Category: risk/complex_systems
- Note: Already partially implemented in the system

### 5. Zoltan Pozsar — Bretton Woods III
- Framework: Commodity-backed currencies challenging dollar hegemony
- Key work: Credit Suisse "War and Commodity Encumbrance" series
- Tradeable signal: Dollar weakening → commodity/gold overweight, TIPS
- Affects: Currency regime, commodity exposure, inflation hedge positioning
- Category: monetary/structural

### 6. Howard Marks — Market Cycles
- Framework: "Knowing where we stand" in the cycle
- Key work: "Mastering the Market Cycle"
- Tradeable signal: Cycle positioning → risk appetite calibration
- Affects: Regime classification confidence, overall risk budget
- Category: markets/timing

### 7. Daniel Yergin — Energy Geopolitics
- Framework: Oil/energy as geopolitical weapon, transition timelines
- Key work: "The New Map," "The Prize"
- Tradeable signal: OPEC dynamics, energy transition pace, pipeline politics
- Affects: XLE positioning, energy sector timing
- Category: energy/geopolitics

### 8. Elbridge Colby — Taiwan Tail Risk
- Framework: US-China military competition, Taiwan strait scenarios
- Key work: "The Strategy of Denial"
- Tradeable signal: If Taiwan conflict probability rises → semiconductor tail risk
- Affects: XLK exposure (TSMC dependency), defense sector, supply chain hedges
- Category: military/security

### 9. Peter Turchin — Political Instability Cycles
- Framework: Cliodynamics — mathematical modeling of historical cycles
- Key work: "Ages of Discord," "End Times"
- Tradeable signal: Elite overproduction → populist policy (tariffs, redistribution)
- Affects: Policy risk forecasting, tariff probability, regulatory regime
- Category: history/complex_systems

### 10. Didier Sornette — Bubble Detection
- Framework: Log-periodic power law models, "Dragon Kings"
- Key work: "Why Stock Markets Crash"
- Tradeable signal: Quantitative bubble signatures in price data
- Affects: Timing signals for regime transitions, crash probability
- Category: quantitative/timing

### 11. Claudia Sahm — Recession Timing
- Framework: Sahm Rule (unemployment rate trigger)
- Key work: Sahm Rule recession indicator
- Tradeable signal: Binary recession signal with strong historical accuracy
- Affects: Regime transitions (risk_on → risk_off), defensive rotation timing
- Category: monetary/timing

### 12. Adam Tooze — Financial Crisis Patterns
- Framework: Historical analysis of financial crises, real-time macro commentary
- Key work: "Crashed," "Shutdown," Chartbook newsletter
- Tradeable signal: Pattern recognition across financial crises
- Affects: Crisis regime identification, contagion pathway detection
- Category: history/monetary

---

## Tier 2 — Valuable, Add When Expanding

### Energy & Commodities
- **Anas Alhajji** — Former OPEC advisor. Granular spare capacity and supply/demand
  mechanics. Counterweight to Zeihan's broader strokes on energy.
- **Vaclav Smil** — Energy transitions take decades, not years. Structural bet
  on legacy energy persistence. Skepticism on rapid green transition timelines.

### China-Specific
- **Rush Doshi** — "The Long Game" on China's grand strategy. Useful for
  predicting US policy moves toward China.

### Monetary / Central Banking
- **Lacy Hunt** — Velocity of money, deflationary debt dynamics. Outstanding
  bond market calls. Structural deflation thesis.
- **Perry Mehrling** — "Money view" framework. Understands financial plumbing
  (repo markets, swap lines) better than almost anyone.

### Military / Security
- **Michael Kofman** — Russia/Ukraine military analysis. Most accurate real-time
  assessor of that conflict, which affects energy and European equities.

### Complex Systems / Quantitative
- **Yaneer Bar-Yam** — Food price spikes predicting political instability
  (predicted Arab Spring). Cross-domain signal detection.
- **Ole Peters** — Ergodicity economics. What's rational for an ensemble is not
  rational for an individual over time. Directly relevant to position sizing.
  Endorsed by Taleb.
- **Benoit Mandelbrot** (posthumous) — Fat tails in financial markets. "The
  Misbehavior of Markets." Foundational for why THETA exists.

### Heterodox / Left
- **Yanis Varoufakis** — Eurozone fragility, "techno-feudalism," capital flow
  imbalances. Useful contrarian lens on tech dominance.
- **Michael Hudson** — Debt jubilee cycles, dollar hegemony erosion. Structural
  thesis about unpayable debt loads.
- **Mariana Mazzucato** — State vs private innovation. Where actual growth comes
  from. Useful for assessing which sectors have real vs speculative value.

### Historians
- **Niall Ferguson** — Imperial overstretch, Cold War II framing. Frequent
  market-relevant commentary on US-China rivalry.
- **George Friedman (Stratfor)** — Geopolitical forecasting frameworks.
  "The Next 100 Years" structural analysis.
- **Hal Brands** — Great power competition, Taiwan scenarios. Complements Colby.

### Practitioners
- **Stan Druckenmiller** — Macro trading framework, liquidity-driven analysis.
  "Earnings don't move markets, liquidity does."
- **Michael Burry** — Contrarian deep-value, structural mispricing detection.
  Housing, water, specific sector bets.
- **Pippa Malmgren** — Signal reading from non-obvious data. Called 2022
  invasion months early from satellite/logistics patterns.

---

## Thesis Category Framework

```
STRUCTURAL RISKS (slow-moving, 1-5 year horizon)
├── Demographics    → Zeihan, Acemoglu
├── Debt cycles     → Dalio, Hunt, Hudson
├── Institutions    → Varoufakis, Acemoglu, Mazzucato
└── Power shifts    → Kotkin, Friedman, Brands, Colby

TIMING SIGNALS (fast-moving, weeks to months)
├── Cycle position  → Marks, Druckenmiller, Dalio
├── Bubble detection→ Sornette, Burry
├── Event reading   → Malmgren, Ferguson, Tooze
├── Recession timing→ Sahm, Hunt
└── Tail risk       → Taleb, Sornette

SECTOR IMPLICATIONS (what to overweight/underweight)
├── Energy geography→ Zeihan, Yergin, Alhajji, Smil
├── Tech / chips    → Colby (Taiwan risk), Mazzucato (innovation)
├── Finance / debt  → Hunt, Hudson, Mehrling (plumbing risk)
├── Industrial      → Zeihan (reshoring), Friedman (defense)
├── Commodities     → Pozsar (Bretton Woods III), Yergin
└── Contrarian      → Burry, Varoufakis (what consensus is wrong about)
```

---

## Implementation Notes

When building this:
1. Thesis entries stored as YAML in `corp/data/theses/`
2. GeopoliticsExpert loads theses and cross-references against daily headlines
3. Agreement matrix computed across thinkers per category
4. Disagreement flags widen regime confidence intervals
5. Thesis confirmation/invalidation events trigger regime reassessment
6. CEO CLI can add/edit/review theses ("add thesis from Dalio on debt cycle")
7. LLM (Groq) evaluates headline-thesis alignment, not raw summarization

DO NOT try to reconcile contradictory thinkers. Disagreement is signal.

---

## Trigger for Implementation

Start building when:
- Curriculum training (3-stage, 120 gen) is complete
- Forward test infrastructure is validated
- GeopoliticsExpert has been running daily for 2+ weeks with current setup
- The thesis library format has been reviewed by the operator (CEO CLI)
