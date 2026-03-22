"""Statistical calibration tests ported from ATHENA.

Implements:
- Probabilistic Sharpe Ratio (PSR) — Bailey & Lopez de Prado
- Deflated Sharpe Ratio (DSR) — adjusts for multiple-testing bias
- Bootstrap Confidence Intervals on Sharpe ratio
- Walk-Forward Efficiency (WFE) overfitting detector from KRONOS
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Return distribution statistics
# ---------------------------------------------------------------------------

def return_statistics(returns: np.ndarray) -> dict[str, float]:
    """Compute return distribution statistics needed by PSR/DSR.

    Args:
        returns: Array of per-step or per-trade returns.

    Returns:
        Dict with sharpe, mean, std, skewness, kurtosis, n.
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    if n < 2:
        return {"sharpe": 0.0, "mean": 0.0, "std": 0.0,
                "skewness": 0.0, "kurtosis": 3.0, "n": n}

    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    sharpe = mean / (std + 1e-12)

    # Skewness and excess kurtosis
    if std > 1e-12:
        z = (returns - mean) / std
        skew = float(np.mean(z ** 3))
        kurt = float(np.mean(z ** 4))  # raw kurtosis (normal=3)
    else:
        skew, kurt = 0.0, 3.0

    return {
        "sharpe": round(sharpe, 6),
        "mean": round(mean, 8),
        "std": round(std, 8),
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Probabilistic Sharpe Ratio (PSR)
# ---------------------------------------------------------------------------

def probabilistic_sharpe_ratio(
    sharpe_observed: float,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    sharpe_benchmark: float = 0.0,
) -> dict[str, float]:
    """Probabilistic Sharpe Ratio: P(true SR > benchmark).

    Accounts for non-normality via skewness and excess kurtosis.
    From Bailey & Lopez de Prado (2012).

    Args:
        sharpe_observed: Observed Sharpe ratio.
        n_obs: Number of return observations.
        skewness: Return skewness.
        kurtosis: Return kurtosis (normal = 3.0).
        sharpe_benchmark: Benchmark Sharpe to beat.

    Returns:
        Dict with psr, z_score, sigma_sr.
    """
    if n_obs < 5:
        return {"psr": 0.5, "z_score": 0.0, "sigma_sr": float("inf")}

    sr = sharpe_observed
    sr_star = sharpe_benchmark

    # Standard error of Sharpe ratio under non-normal returns
    sigma_sr_sq = (
        1.0
        - skewness * sr
        + ((kurtosis - 1) / 4.0) * sr ** 2
    ) / (n_obs - 1)
    sigma_sr = math.sqrt(max(sigma_sr_sq, 1e-12))

    z = (sr - sr_star) / sigma_sr

    # CDF of standard normal (no scipy dependency)
    psr = _norm_cdf(z)

    return {
        "psr": round(psr, 6),
        "z_score": round(z, 4),
        "sigma_sr": round(sigma_sr, 6),
    }


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio (DSR)
# ---------------------------------------------------------------------------

def deflated_sharpe_ratio(
    sharpe_observed: float,
    n_obs: int,
    n_trials: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    variance_of_sharpes: float = 1.0,
) -> dict[str, Any]:
    """Deflated Sharpe Ratio — adjusts for data-snooping / multiple testing.

    When selecting the best of N agents/param combos, the expected maximum
    Sharpe under pure luck grows with N. DSR adjusts for this.

    From Bailey & Lopez de Prado (2014).

    Args:
        sharpe_observed: Observed Sharpe of the selected (best) agent.
        n_obs: Number of return observations.
        n_trials: Number of agents/strategies tested (selection bias).
        skewness: Return skewness.
        kurtosis: Return kurtosis (normal = 3.0).
        variance_of_sharpes: Variance across all tried Sharpe ratios.

    Returns:
        Dict with dsr, expected_max_sharpe, is_significant.
    """
    if n_trials < 2:
        psr_result = probabilistic_sharpe_ratio(
            sharpe_observed, n_obs, skewness, kurtosis, 0.0
        )
        return {
            "dsr": psr_result["psr"],
            "expected_max_sharpe": 0.0,
            "sharpe_observed": round(sharpe_observed, 6),
            "n_trials": n_trials,
            "is_significant": psr_result["psr"] > 0.95,
        }

    # Expected maximum Sharpe under null hypothesis (no skill)
    EULER_MASCHERONI = 0.5772156649
    log_n = math.log(max(n_trials, 2))
    sqrt_2ln = math.sqrt(2 * log_n)
    correction = (math.log(math.pi) + EULER_MASCHERONI) / (2 * sqrt_2ln)
    e_max_sr = math.sqrt(variance_of_sharpes) * (sqrt_2ln - correction)

    # DSR = PSR with benchmark = expected max under null
    psr_result = probabilistic_sharpe_ratio(
        sharpe_observed, n_obs, skewness, kurtosis, e_max_sr
    )

    return {
        "dsr": psr_result["psr"],
        "expected_max_sharpe": round(e_max_sr, 6),
        "sharpe_observed": round(sharpe_observed, 6),
        "n_trials": n_trials,
        "is_significant": psr_result["psr"] > 0.95,
    }


# ---------------------------------------------------------------------------
# Bootstrap Confidence Interval
# ---------------------------------------------------------------------------

def bootstrap_sharpe_ci(
    returns: np.ndarray,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap confidence interval on Sharpe ratio.

    Args:
        returns: Array of returns.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (e.g. 0.95).
        seed: Random seed for reproducibility.

    Returns:
        Dict with sharpe_observed, ci_lower, ci_upper, p_value.
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    if n < 5:
        return {
            "sharpe_observed": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
            "ci_level": confidence, "p_value_sharpe_gt_zero": 0.5, "n_obs": n,
        }

    obs_sharpe = float(np.mean(returns) / (np.std(returns, ddof=1) + 1e-12))

    rng = np.random.RandomState(seed)
    boot_sharpes = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(returns, size=n, replace=True)
        std = np.std(sample, ddof=1)
        boot_sharpes[i] = np.mean(sample) / (std + 1e-12)

    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_sharpes, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_sharpes, 100 * (1 - alpha / 2)))
    p_value = float(np.mean(boot_sharpes <= 0))

    return {
        "sharpe_observed": round(obs_sharpe, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "ci_level": confidence,
        "p_value_sharpe_gt_zero": round(p_value, 6),
        "n_obs": n,
    }


# ---------------------------------------------------------------------------
# Walk-Forward Efficiency (WFE) — from KRONOS
# ---------------------------------------------------------------------------

def compute_wfe(
    oos_sharpe: float,
    is_sharpe: float,
    min_is_sharpe: float = 0.1,
) -> float:
    """Walk-Forward Efficiency: ratio of OOS to IS Sharpe.

    WFE > 0.40 = good generalization
    0.25 <= WFE < 0.40 = moderate overfitting
    WFE < 0.25 = severe overfitting (IS edge evaporates OOS)

    Args:
        oos_sharpe: Out-of-sample Sharpe ratio.
        is_sharpe: In-sample (training) Sharpe ratio.
        min_is_sharpe: Floor to avoid division by near-zero.

    Returns:
        WFE ratio, clamped to [0, 2].
    """
    if is_sharpe < min_is_sharpe:
        return 0.0
    wfe = oos_sharpe / is_sharpe
    return max(0.0, min(wfe, 2.0))


def diagnose_wfe(wfe: float) -> dict[str, str]:
    """Diagnose WFE and return severity + recommendation.

    Returns:
        Dict with severity, diagnosis, recommendation.
    """
    if wfe >= 0.60:
        return {
            "severity": "good",
            "diagnosis": f"Strong generalization (WFE={wfe:.2f})",
            "recommendation": "No changes needed — OOS performance retains most IS edge.",
        }
    elif wfe >= 0.40:
        return {
            "severity": "acceptable",
            "diagnosis": f"Adequate generalization (WFE={wfe:.2f})",
            "recommendation": "Acceptable. Monitor for degradation over time.",
        }
    elif wfe >= 0.25:
        return {
            "severity": "warning",
            "diagnosis": f"Moderate overfitting (WFE={wfe:.2f})",
            "recommendation": "Reduce model complexity. Equalize weights. Shorten lookback.",
        }
    else:
        return {
            "severity": "critical",
            "diagnosis": f"Severe overfitting (WFE={wfe:.2f})",
            "recommendation": "IS edge evaporates OOS. Reset parameters to defaults. "
                             "Reduce complexity aggressively. Consider simpler strategies.",
        }


# ---------------------------------------------------------------------------
# Full Calibration Report
# ---------------------------------------------------------------------------

def run_full_calibration(
    oos_returns: np.ndarray,
    n_trials: int,
    confidence: float = 0.95,
    n_bootstrap: int = 5000,
    is_sharpe: float | None = None,
) -> dict[str, Any]:
    """Run all statistical tests and return composite verdict.

    Args:
        oos_returns: Out-of-sample return array.
        n_trials: Number of agents/param combos tested (for DSR).
        confidence: Confidence level for bootstrap CI.
        n_bootstrap: Number of bootstrap resamples.
        is_sharpe: In-sample Sharpe for WFE computation (optional).

    Returns:
        Dict with all test results and overall verdict.
    """
    dist = return_statistics(oos_returns)
    boot = bootstrap_sharpe_ci(oos_returns, n_bootstrap, confidence)
    psr = probabilistic_sharpe_ratio(
        dist["sharpe"], dist["n"], dist["skewness"], dist["kurtosis"]
    )
    dsr = deflated_sharpe_ratio(
        dist["sharpe"], dist["n"], n_trials,
        dist["skewness"], dist["kurtosis"]
    )

    # WFE if IS sharpe provided
    wfe_result = None
    if is_sharpe is not None:
        wfe_val = compute_wfe(dist["sharpe"], is_sharpe)
        wfe_result = {"wfe": wfe_val, **diagnose_wfe(wfe_val)}

    # Verdict gates
    dsr_pass = dsr["dsr"] > 0.5
    ci_pass = boot["ci_lower"] > 0
    psr_pass = psr["psr"] > 0.95

    reasons = []
    if not dsr_pass:
        reasons.append(f"DSR {dsr['dsr']:.3f} <= 0.50 — edge may be data-snooping artifact")
    if not ci_pass:
        reasons.append(f"Bootstrap CI includes zero [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")
    if not psr_pass:
        reasons.append(f"PSR {psr['psr']:.3f} < 0.95 — insufficient statistical evidence")

    verdict = "PASS" if (dsr_pass and ci_pass) else "FAIL"

    return {
        "distribution": dist,
        "bootstrap": boot,
        "probabilistic_sharpe": psr,
        "deflated_sharpe": dsr,
        "wfe": wfe_result,
        "verdict": verdict,
        "reasons": reasons,
        "dsr_pass": dsr_pass,
        "ci_pass": ci_pass,
        "psr_pass": psr_pass,
    }


# ---------------------------------------------------------------------------
# Helpers (no scipy dependency)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF using the error function from math module."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
