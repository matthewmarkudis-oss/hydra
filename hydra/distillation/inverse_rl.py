"""Maximum Entropy Inverse Reinforcement Learning for hedge fund objective inference.

Implements a simplified MaxEnt IRL pipeline that infers what objective function
top hedge funds optimize from their observed 13F filing behavior.  The inferred
reward weights are mapped to Hydra's RewardConfig parameters so the RL agents
can be trained to mimic the risk/return preferences of successful discretionary
managers.

Pipeline overview:
    1. Extract expert trajectories from sequential 13F filings.
    2. Compute expert feature expectations (mean feature vector).
    3. Infer reward weights via regularized least-squares feature matching.
    4. Map the learned weights to RewardConfig parameters.

Backtesting and training only.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("hydra.distillation.inverse_rl")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NUM_FEATURES = 12
_FEATURE_NAMES: list[str] = [
    "mkt_rf_exposure",       # Fama-French Mkt-RF beta
    "smb_exposure",          # Fama-French SMB beta
    "hml_exposure",          # Fama-French HML beta
    "rmw_exposure",          # Fama-French RMW beta
    "cma_exposure",          # Fama-French CMA beta
    "sector_delta_tech",     # Change in tech sector allocation %
    "sector_delta_energy",   # Change in energy sector allocation %
    "sector_delta_finance",  # Change in finance sector allocation %
    "sector_delta_healthcare",  # Change in healthcare sector allocation %
    "concentration_delta",   # HHI delta (Herfindahl-Hirschman Index)
    "turnover",              # Fraction of portfolio rebalanced
    "drawdown_proxy",        # Max quarterly loss across held positions
]

# Sector keyword mapping for rudimentary sector classification from tickers.
# In production this would use a proper GICS lookup; here we use a simplified
# heuristic that checks if a ticker string starts with common prefixes.
_SECTOR_KEYWORDS: dict[str, list[str]] = {
    "tech": [
        "AAPL", "MSFT", "GOOG", "GOOGL", "META", "AMZN", "NVDA", "AMD",
        "INTC", "CRM", "ORCL", "ADBE", "CSCO", "IBM", "TSM", "AVGO",
        "QCOM", "TXN", "NFLX", "PYPL", "SQ", "SHOP", "SNOW", "PLTR",
        "UBER", "COIN", "MARA", "NET", "DDOG", "ZS",
    ],
    "energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO",
        "OXY", "DVN", "HAL", "BKR", "FANG", "APA", "MRO",
    ],
    "finance": [
        "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW",
        "AXP", "USB", "PNC", "TFC", "COF", "BK", "STT", "FITB",
    ],
    "healthcare": [
        "UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT",
        "DHR", "BMY", "AMGN", "GILD", "ISRG", "MDT", "SYK", "CVS",
    ],
}

# RewardConfig valid ranges (drawn from schema.py Field constraints)
_REWARD_RANGES: dict[str, tuple[float, float]] = {
    "sharpe_eta": (0.001, 1.0),
    "drawdown_penalty": (0.0, 10.0),
    "transaction_penalty": (0.0, 5.0),
    "holding_penalty": (0.0, 5.0),
    "pnl_bonus_weight": (0.0, 10.0),
    "reward_scale": (1.0, 500.0),
}

# RewardConfig base (default) values matching schema.py defaults
_REWARD_DEFAULTS: dict[str, float] = {
    "sharpe_eta": 0.05,
    "drawdown_penalty": 0.5,
    "transaction_penalty": 0.1,
    "holding_penalty": 0.1,
    "pnl_bonus_weight": 1.0,
    "reward_scale": 100.0,
}

# Regularization strength for IRL
_LAMBDA_REG: float = 0.01


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _classify_sector(ticker: str) -> str | None:
    """Return sector string for *ticker* or ``None`` if unrecognised."""
    upper = ticker.upper()
    for sector, tickers in _SECTOR_KEYWORDS.items():
        if upper in tickers:
            return sector
    return None


def _portfolio_weights(holdings: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Compute portfolio weight per ticker from a filing snapshot.

    Args:
        holdings: Mapping ``{ticker: {"shares": ..., "value": ...}}``.

    Returns:
        Dict of ``{ticker: weight}`` where weights sum to 1.
    """
    total_value = sum(
        float(pos.get("value", 0)) for pos in holdings.values()
    )
    if total_value <= 0:
        return {t: 0.0 for t in holdings}
    return {t: float(pos.get("value", 0)) / total_value for t, pos in holdings.items()}


def _sector_allocations(weights: dict[str, float]) -> dict[str, float]:
    """Aggregate ticker weights into sector allocations.

    Returns:
        Dict with keys ``tech``, ``energy``, ``finance``, ``healthcare`` and
        corresponding allocation fractions.
    """
    alloc: dict[str, float] = {
        "tech": 0.0,
        "energy": 0.0,
        "finance": 0.0,
        "healthcare": 0.0,
    }
    for ticker, w in weights.items():
        sector = _classify_sector(ticker)
        if sector is not None:
            alloc[sector] += w
    return alloc


def _hhi(weights: dict[str, float]) -> float:
    """Herfindahl-Hirschman Index from a weight dictionary."""
    vals = np.array(list(weights.values()), dtype=np.float64)
    total = vals.sum()
    if total <= 0:
        return 0.0
    fracs = vals / total
    return float(np.sum(fracs ** 2))


def _turnover(weights_prev: dict[str, float], weights_next: dict[str, float]) -> float:
    """Portfolio turnover between two quarters.

    Turnover = sum(|w_{t+1} - w_t|) / 2
    """
    all_tickers = set(weights_prev) | set(weights_next)
    delta_sum = sum(
        abs(weights_next.get(t, 0.0) - weights_prev.get(t, 0.0))
        for t in all_tickers
    )
    return delta_sum / 2.0


def _approximate_fund_return(
    holdings_prev: dict[str, dict[str, Any]],
    holdings_next: dict[str, dict[str, Any]],
) -> float:
    """Approximate quarterly fund return from portfolio value changes.

    When actual fund returns are unavailable we estimate the return as
    ``(total_value_{t+1} - total_value_t) / total_value_t``.
    """
    val_prev = sum(float(p.get("value", 0)) for p in holdings_prev.values())
    val_next = sum(float(p.get("value", 0)) for p in holdings_next.values())
    if val_prev <= 0:
        return 0.0
    return (val_next - val_prev) / val_prev


def _regress_factor_exposures(
    fund_return: float,
    factor_returns: pd.DataFrame | None,
    quarter_idx: int,
) -> np.ndarray:
    """Regress a single quarterly fund return against Fama-French 5 factors.

    If *factor_returns* contains enough rows we use the quarter's factor
    returns to compute a simple projection.  Because we only have a single
    fund return observation per quarter, a full OLS is under-determined; we
    approximate the exposure vector as:

        beta_i = fund_return * factor_return_i / ||factor_returns||^2

    This gives a directional (not causal) exposure estimate which is
    sufficient for the IRL feature vector.

    Args:
        fund_return: Scalar quarterly return for the fund.
        factor_returns: DataFrame with columns for each FF5 factor.
            Expected columns (order matters): Mkt-RF, SMB, HML, RMW, CMA.
        quarter_idx: Row index into *factor_returns* for this quarter.

    Returns:
        (5,) float64 array of factor exposures.
    """
    exposures = np.zeros(5, dtype=np.float64)
    if factor_returns is None or factor_returns.empty:
        return exposures

    # Attempt to pull a row of factor returns for this quarter
    if quarter_idx < 0 or quarter_idx >= len(factor_returns):
        return exposures

    row = factor_returns.iloc[quarter_idx].values[:5].astype(np.float64)
    norm_sq = float(np.dot(row, row))
    if norm_sq < 1e-12:
        return exposures

    exposures = fund_return * row / norm_sq
    return exposures


def _max_quarterly_loss(
    holdings_prev: dict[str, dict[str, Any]],
    holdings_next: dict[str, dict[str, Any]],
) -> float:
    """Compute a drawdown proxy: max loss across held positions in the quarter.

    For each ticker held at q_t, computes per-position return using value
    changes and returns the most negative return (as a positive magnitude).
    If no position lost value, returns 0.
    """
    worst = 0.0
    for ticker, pos_prev in holdings_prev.items():
        val_prev = float(pos_prev.get("value", 0))
        if val_prev <= 0:
            continue
        pos_next = holdings_next.get(ticker, {})
        val_next = float(pos_next.get("value", 0))
        ret = (val_next - val_prev) / val_prev
        if ret < worst:
            worst = ret
    # Return magnitude (positive number representing loss depth)
    return abs(worst)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_expert_trajectories(
    filings_history: list[dict],
    factor_returns: pd.DataFrame,
) -> np.ndarray:
    """Convert quarterly 13F filing snapshots into state-action feature vectors.

    Each filing dict has the structure::

        {fund_name: {ticker: {"shares": int, "value": float}}}

    For each consecutive pair of quarters ``(q_t, q_{t+1})`` the function
    computes a 12-dimensional feature vector:

    - **Factor exposures** (dims 0-4): approximate FF5 betas from the
      quarterly fund return regressed against the factor return row.
    - **Sector allocation delta** (dims 5-8): change in tech, energy,
      finance, healthcare allocation percentages.
    - **Concentration delta** (dim 9): HHI change between quarters.
    - **Turnover** (dim 10): fraction of portfolio weight rebalanced.
    - **Drawdown proxy** (dim 11): max single-position loss in the quarter.

    Args:
        filings_history: Ordered list of quarterly 13F snapshots (oldest
            first).  Each entry is ``{fund_name: {ticker: {shares, value}}}``.
        factor_returns: DataFrame with Fama-French 5-factor quarterly
            returns.  Expected to have at least ``len(filings_history) - 1``
            rows.

    Returns:
        ``(num_transitions, 12)`` float32 array.  If insufficient data
        (fewer than 2 filings), returns a single zero-padded row and logs
        a warning.
    """
    if len(filings_history) < 2:
        logger.warning(
            "Insufficient filings history (%d snapshots); need at least 2 "
            "for transition features.  Returning zero-padded row.",
            len(filings_history),
        )
        return np.zeros((1, _NUM_FEATURES), dtype=np.float32)

    all_features: list[np.ndarray] = []

    for t in range(len(filings_history) - 1):
        filing_t = filings_history[t]
        filing_t1 = filings_history[t + 1]

        # Use the first fund present in both snapshots.  If filings
        # contain multiple funds, iterate over all of them.
        funds_t = set(filing_t.keys())
        funds_t1 = set(filing_t1.keys())
        common_funds = funds_t & funds_t1

        if not common_funds:
            logger.warning(
                "No common fund names between quarters %d and %d; "
                "padding transition with zeros.",
                t, t + 1,
            )
            all_features.append(np.zeros(_NUM_FEATURES, dtype=np.float64))
            continue

        for fund_name in sorted(common_funds):
            holdings_prev = filing_t[fund_name]
            holdings_next = filing_t1[fund_name]

            feat = np.zeros(_NUM_FEATURES, dtype=np.float64)

            # --- Factor exposures (dims 0-4) ---
            fund_ret = _approximate_fund_return(holdings_prev, holdings_next)
            feat[0:5] = _regress_factor_exposures(
                fund_ret, factor_returns, quarter_idx=t,
            )

            # --- Sector allocation delta (dims 5-8) ---
            w_prev = _portfolio_weights(holdings_prev)
            w_next = _portfolio_weights(holdings_next)

            sec_prev = _sector_allocations(w_prev)
            sec_next = _sector_allocations(w_next)

            feat[5] = sec_next["tech"] - sec_prev["tech"]
            feat[6] = sec_next["energy"] - sec_prev["energy"]
            feat[7] = sec_next["finance"] - sec_prev["finance"]
            feat[8] = sec_next["healthcare"] - sec_prev["healthcare"]

            # --- Concentration delta (dim 9) ---
            hhi_prev = _hhi(w_prev)
            hhi_next = _hhi(w_next)
            feat[9] = hhi_next - hhi_prev

            # --- Turnover (dim 10) ---
            feat[10] = _turnover(w_prev, w_next)

            # --- Drawdown proxy (dim 11) ---
            feat[11] = _max_quarterly_loss(holdings_prev, holdings_next)

            all_features.append(feat)

    if not all_features:
        logger.warning("No valid transitions extracted; returning zero-padded row.")
        return np.zeros((1, _NUM_FEATURES), dtype=np.float32)

    return np.array(all_features, dtype=np.float32)


def compute_expert_feature_expectations(trajectories: np.ndarray) -> np.ndarray:
    """Compute mean feature vector across all expert transitions.

    This is the empirical feature expectation under the expert policy,
    ``\\hat{\\mu}_E = \\frac{1}{N} \\sum_i \\phi(s_i, a_i)``.

    Args:
        trajectories: ``(N, 12)`` array of expert feature vectors.

    Returns:
        ``(12,)`` float64 array.  Returns zeros if input is empty.
    """
    if trajectories is None or trajectories.size == 0:
        logger.warning("Empty trajectories array; returning zero feature expectations.")
        return np.zeros(_NUM_FEATURES, dtype=np.float64)

    if trajectories.ndim == 1:
        return trajectories.astype(np.float64)

    return np.mean(trajectories.astype(np.float64), axis=0)


def infer_reward_weights(
    expert_features: np.ndarray,
    candidate_features: np.ndarray | None = None,
) -> np.ndarray:
    """Infer reward weights via simplified Maximum Entropy IRL.

    When *candidate_features* are provided (a matrix where each row is a
    candidate policy's feature expectation), the weights are found by
    regularised least-squares feature matching::

        w* = argmin_w ||expert_features - candidate_features @ w||^2
                      + lambda * ||w||^2
           = (Phi^T Phi + lambda I)^{-1} Phi^T r_expert

    When no candidates are available the expert feature expectation itself
    is treated as the implied reward direction and normalised to unit length.

    Args:
        expert_features: ``(12,)`` expert feature expectation vector.
        candidate_features: Optional ``(M, 12)`` matrix of candidate
            policy feature expectations.

    Returns:
        ``(12,)`` weight vector.
    """
    expert_features = np.asarray(expert_features, dtype=np.float64).ravel()
    if expert_features.shape[0] != _NUM_FEATURES:
        raise ValueError(
            f"Expected expert_features of length {_NUM_FEATURES}, "
            f"got {expert_features.shape[0]}."
        )

    if candidate_features is not None:
        Phi = np.asarray(candidate_features, dtype=np.float64)
        if Phi.ndim == 1:
            Phi = Phi.reshape(1, -1)
        if Phi.shape[1] != _NUM_FEATURES:
            raise ValueError(
                f"candidate_features must have {_NUM_FEATURES} columns, "
                f"got {Phi.shape[1]}."
            )

        # Regularised least-squares: w = (Phi^T Phi + lambda I)^{-1} Phi^T r
        PhiT = Phi.T  # (12, M)
        A = PhiT @ Phi + _LAMBDA_REG * np.eye(_NUM_FEATURES, dtype=np.float64)

        try:
            w = np.linalg.solve(A, PhiT @ expert_features)
        except np.linalg.LinAlgError:
            logger.warning(
                "Singular matrix in IRL solve; falling back to pseudo-inverse."
            )
            w = np.linalg.lstsq(A, PhiT @ expert_features, rcond=None)[0]

        return w.astype(np.float64)

    # No candidates: use expert features as implied reward direction
    norm = np.linalg.norm(expert_features)
    if norm < 1e-12:
        logger.warning(
            "Expert feature vector is near-zero; returning uniform weights."
        )
        return np.ones(_NUM_FEATURES, dtype=np.float64) / _NUM_FEATURES

    return (expert_features / norm).astype(np.float64)


def map_weights_to_reward_config(irl_weights: np.ndarray) -> dict[str, float]:
    """Map IRL weight vector to Hydra ``RewardConfig`` parameters.

    The mapping translates the 12-dimensional learned preference vector into
    the six reward parameters used by :class:`DifferentialSharpeReward`.

    Mapping logic:
        * ``w[0]`` (Mkt-RF exposure) scales ``pnl_bonus_weight``.
        * ``w[4]`` (CMA / conservative factor) inversely scales
          ``transaction_penalty`` (conservative funds tolerate higher costs).
        * ``mean(|w[5:9]|)`` (sector rotation magnitude) scales
          ``sharpe_eta`` (active rotation implies higher Sharpe focus).
        * ``w[9]`` (concentration) scales ``holding_penalty``.
        * ``w[10]`` (turnover) further modifies ``transaction_penalty``.
        * ``w[11]`` (drawdown) scales ``drawdown_penalty``.

    All values are clipped to the valid ``RewardConfig`` ranges.

    Args:
        irl_weights: ``(12,)`` weight vector from :func:`infer_reward_weights`.

    Returns:
        Dict of reward parameters ready for
        ``HydraConfig.apply_patch({"reward": result})``.
    """
    w = np.asarray(irl_weights, dtype=np.float64).ravel()
    if w.shape[0] != _NUM_FEATURES:
        raise ValueError(
            f"Expected weight vector of length {_NUM_FEATURES}, got {w.shape[0]}."
        )

    base = _REWARD_DEFAULTS.copy()

    # --- pnl_bonus_weight: influenced by Mkt-RF exposure (w[0]) ---
    pnl_bonus_weight = base["pnl_bonus_weight"] * (1.0 + 0.5 * float(w[0]))

    # --- transaction_penalty: influenced by CMA (w[4]) and turnover (w[10]) ---
    transaction_penalty = base["transaction_penalty"] * (1.0 - 0.3 * float(w[4]))
    # Turnover modifier: high turnover preference reduces transaction penalty
    # awareness (the fund trades more despite costs)
    transaction_penalty *= (1.0 - 0.2 * float(w[10]))

    # --- sharpe_eta: influenced by sector rotation activity ---
    sector_mag = float(np.mean(np.abs(w[5:9])))
    sharpe_eta = base["sharpe_eta"] * (1.0 + 2.0 * sector_mag)

    # --- holding_penalty: influenced by concentration preference (w[9]) ---
    holding_penalty = base["holding_penalty"] * (1.0 + 0.4 * float(w[9]))

    # --- drawdown_penalty: influenced by drawdown weight (w[11]) ---
    drawdown_penalty = base["drawdown_penalty"] * (1.0 + 0.6 * abs(float(w[11])))

    # --- reward_scale: keep at default ---
    reward_scale = base["reward_scale"]

    # Clip to valid ranges
    config: dict[str, float] = {
        "pnl_bonus_weight": float(np.clip(
            pnl_bonus_weight,
            _REWARD_RANGES["pnl_bonus_weight"][0],
            _REWARD_RANGES["pnl_bonus_weight"][1],
        )),
        "transaction_penalty": float(np.clip(
            transaction_penalty,
            _REWARD_RANGES["transaction_penalty"][0],
            _REWARD_RANGES["transaction_penalty"][1],
        )),
        "sharpe_eta": float(np.clip(
            sharpe_eta,
            _REWARD_RANGES["sharpe_eta"][0],
            _REWARD_RANGES["sharpe_eta"][1],
        )),
        "holding_penalty": float(np.clip(
            holding_penalty,
            _REWARD_RANGES["holding_penalty"][0],
            _REWARD_RANGES["holding_penalty"][1],
        )),
        "drawdown_penalty": float(np.clip(
            drawdown_penalty,
            _REWARD_RANGES["drawdown_penalty"][0],
            _REWARD_RANGES["drawdown_penalty"][1],
        )),
        "reward_scale": float(np.clip(
            reward_scale,
            _REWARD_RANGES["reward_scale"][0],
            _REWARD_RANGES["reward_scale"][1],
        )),
    }

    return config


def get_inference_report(
    irl_weights: np.ndarray,
    proposed_config: dict[str, float],
    num_transitions: int = 0,
) -> dict[str, Any]:
    """Generate a human-readable report of the IRL inference results.

    Args:
        irl_weights: ``(12,)`` inferred weight vector.
        proposed_config: Mapped ``RewardConfig`` parameter dict from
            :func:`map_weights_to_reward_config`.
        num_transitions: Number of expert transitions used for inference.
            Used to gauge data-sufficiency confidence.

    Returns:
        Report dict containing:
            - ``raw_weights``: the 12-dim vector as a list.
            - ``feature_names``: human-readable labels for each dimension.
            - ``proposed_config``: the mapped reward parameters.
            - ``confidence``: float in [0, 1] based on data sufficiency.
            - ``methodology``: ``"simplified_maxent_irl"``.
    """
    w = np.asarray(irl_weights, dtype=np.float64).ravel()

    # Confidence heuristic: saturates around 20 transitions.
    # 0 transitions -> 0.0, 1 -> 0.15, 4 -> 0.5, 10 -> 0.75, 20+ -> ~0.95
    if num_transitions <= 0:
        confidence = 0.0
    else:
        confidence = float(1.0 - np.exp(-0.15 * num_transitions))

    return {
        "raw_weights": w.tolist(),
        "feature_names": list(_FEATURE_NAMES),
        "proposed_config": dict(proposed_config),
        "confidence": round(confidence, 4),
        "methodology": "simplified_maxent_irl",
        "num_transitions": num_transitions,
    }


# ---------------------------------------------------------------------------
# Convenience orchestrator
# ---------------------------------------------------------------------------

class InverseRLCalibrator:
    """End-to-end inverse RL calibrator for hedge fund objective inference.

    Wraps the module-level functions into a stateful pipeline that extracts
    expert trajectories from 13F filings, infers reward weights, and produces
    a ``RewardConfig``-compatible parameter dict.

    Example::

        calibrator = InverseRLCalibrator()
        calibrator.fit(filings_history, factor_returns)
        patch = calibrator.reward_patch()  # {"reward": {...}}
        new_cfg = hydra_cfg.apply_patch(patch)

    Attributes:
        weights: Inferred 12-dim reward weight vector (available after fit).
        config: Mapped RewardConfig parameter dict (available after fit).
        report: Full inference report dict (available after fit).
    """

    def __init__(self) -> None:
        self.weights: np.ndarray | None = None
        self.config: dict[str, float] | None = None
        self.report: dict[str, Any] | None = None
        self._trajectories: np.ndarray | None = None
        self._expert_features: np.ndarray | None = None

    def fit(
        self,
        filings_history: list[dict],
        factor_returns: pd.DataFrame,
        candidate_features: np.ndarray | None = None,
    ) -> InverseRLCalibrator:
        """Run the full IRL pipeline.

        Args:
            filings_history: Ordered list of quarterly 13F filing dicts.
            factor_returns: Fama-French 5-factor quarterly returns DataFrame.
            candidate_features: Optional ``(M, 12)`` candidate policy features
                for feature-matching IRL.  If ``None``, the expert features
                are used directly as the reward direction.

        Returns:
            ``self`` for method chaining.
        """
        logger.info(
            "Starting inverse RL calibration with %d filing snapshots.",
            len(filings_history),
        )

        # Step 1: extract trajectories
        self._trajectories = extract_expert_trajectories(
            filings_history, factor_returns,
        )
        num_transitions = self._trajectories.shape[0]
        logger.info("Extracted %d expert transitions.", num_transitions)

        # Step 2: feature expectations
        self._expert_features = compute_expert_feature_expectations(
            self._trajectories,
        )

        # Step 3: infer weights
        self.weights = infer_reward_weights(
            self._expert_features,
            candidate_features=candidate_features,
        )
        logger.info("Inferred IRL weights: %s", self.weights)

        # Step 4: map to reward config
        self.config = map_weights_to_reward_config(self.weights)
        logger.info("Proposed RewardConfig: %s", self.config)

        # Step 5: generate report
        self.report = get_inference_report(
            self.weights, self.config, num_transitions=num_transitions,
        )

        return self

    def reward_patch(self) -> dict[str, dict[str, float]]:
        """Return a config patch dict suitable for ``HydraConfig.apply_patch``.

        Returns:
            ``{"reward": <mapped_config>}``

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if self.config is None:
            raise RuntimeError(
                "InverseRLCalibrator has not been fitted yet.  "
                "Call .fit() before .reward_patch()."
            )
        return {"reward": dict(self.config)}

    def generate_report(self) -> dict[str, Any]:
        """Return the inference report (convenience alias).

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if self.report is None:
            raise RuntimeError(
                "InverseRLCalibrator has not been fitted yet.  "
                "Call .fit() before .generate_report()."
            )
        return self.report

    @property
    def feature_names(self) -> list[str]:
        """Human-readable labels for the 12 feature dimensions."""
        return list(_FEATURE_NAMES)

    @property
    def num_features(self) -> int:
        """Number of features in the IRL feature space."""
        return _NUM_FEATURES
