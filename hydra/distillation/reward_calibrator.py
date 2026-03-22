"""Map academic factor loadings to HydraConfig RewardConfig parameters.

Pure Python + numpy + scipy implementation. Zero LLM cost.

Given Fama-French / Fung-Hsieh factor data (or similar), compute target
factor exposures and translate them into reward-shaping weights suitable
for ``HydraConfig.apply_patch({"reward": ...})``.

Backtesting and training only.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger("hydra.distillation.reward_calibrator")

# ---------------------------------------------------------------------------
# Factor -> reward parameter mapping specification
# ---------------------------------------------------------------------------
# Each entry: factor_name -> (reward_param, direction, sensitivity)
#   direction +1 means positive beta increases the param
#   direction -1 means negative beta increases the param
_FACTOR_MAP: dict[str, tuple[str, int, float]] = {
    "Mkt-RF": ("pnl_bonus_weight", +1, 0.5),
    "SMB": ("drawdown_penalty", -1, 0.4),
    "HML": ("holding_penalty", +1, 0.3),
    "RMW": ("sharpe_eta", +1, 0.2),
    "CMA": ("transaction_penalty", -1, 0.2),
}

# Valid ranges for reward parameters (inclusive bounds).
_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "sharpe_eta": (0.001, 0.5),
    "drawdown_penalty": (0.1, 5.0),
    "transaction_penalty": (0.005, 1.0),
    "holding_penalty": (0.0, 2.0),
    "pnl_bonus_weight": (0.0, 5.0),
    "reward_scale": (10.0, 500.0),
}

# Ordered list of optimisable parameter names (for vector-based optimisation).
_OPT_PARAMS: list[str] = [
    "sharpe_eta",
    "drawdown_penalty",
    "transaction_penalty",
    "holding_penalty",
    "pnl_bonus_weight",
    "reward_scale",
]


def _clip_param(name: str, value: float) -> float:
    """Clip *value* to the valid range for *name*."""
    lo, hi = _PARAM_BOUNDS.get(name, (-np.inf, np.inf))
    return float(np.clip(value, lo, hi))


class RewardCalibrator:
    """Translate academic factor loadings into reward configuration parameters.

    Typical workflow::

        cal = RewardCalibrator()
        loadings = cal.compute_target_profile(factor_returns, target_returns)
        proposed = cal.map_to_reward_config(loadings, current_config.reward.model_dump())
        report = cal.get_calibration_report(loadings, proposed, current_config.reward.model_dump())
        new_config = current_config.apply_patch({"reward": proposed})
    """

    # ------------------------------------------------------------------ #
    #  compute_target_profile
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_target_profile(
        factor_returns: pd.DataFrame,
        target_returns: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Compute factor loadings (betas) for a target return series.

        Parameters
        ----------
        factor_returns:
            DataFrame with columns = factor names (e.g. ``Mkt-RF``, ``SMB``, ...)
            and a DatetimeIndex (or any index aligned with *target_returns*).
        target_returns:
            Optional single-column (or Series) of target returns (e.g. HFRI
            composite). When provided an OLS regression is run::

                target = alpha + F @ beta + epsilon

            When *None*, long-term mean returns of each factor are used as
            implied weights (risk-premia approach).

        Returns
        -------
        dict[str, float]
            Mapping of factor_name -> loading (beta coefficient).
        """
        factor_returns = factor_returns.copy()

        if target_returns is not None:
            # Align indices
            if isinstance(target_returns, pd.DataFrame):
                if target_returns.shape[1] != 1:
                    raise ValueError(
                        "target_returns must be a single-column DataFrame or Series"
                    )
                target_returns = target_returns.iloc[:, 0]

            common_idx = factor_returns.index.intersection(target_returns.index)
            if len(common_idx) < 3:
                raise ValueError(
                    f"Only {len(common_idx)} overlapping observations between "
                    "factor_returns and target_returns; need at least 3."
                )

            y = target_returns.loc[common_idx].values.astype(np.float64)
            X = factor_returns.loc[common_idx].values.astype(np.float64)

            # Add intercept column
            ones = np.ones((X.shape[0], 1), dtype=np.float64)
            X_with_intercept = np.hstack([ones, X])

            # OLS via numpy lstsq
            coeffs, residuals, rank, sv = np.linalg.lstsq(
                X_with_intercept, y, rcond=None
            )

            alpha = float(coeffs[0])
            betas = coeffs[1:]
            factor_names = list(factor_returns.columns)

            loadings = {
                name: float(beta) for name, beta in zip(factor_names, betas)
            }

            logger.info(
                "OLS regression: alpha=%.6f, R^2 approx from %d obs, rank=%d",
                alpha,
                len(common_idx),
                rank,
            )
            for name, beta in loadings.items():
                logger.debug("  %s: beta=%.4f", name, beta)

            return loadings

        # No target_returns: use long-term factor risk premia (mean returns).
        means = factor_returns.mean()
        total = means.abs().sum()
        if total < 1e-12:
            logger.warning(
                "Factor means are near zero; returning uniform loadings."
            )
            n = len(factor_returns.columns)
            return {name: 1.0 / n for name in factor_returns.columns}

        loadings = {name: float(val) for name, val in means.items()}
        logger.info(
            "Implied loadings from risk premia (mean returns) over %d obs.",
            len(factor_returns),
        )
        return loadings

    # ------------------------------------------------------------------ #
    #  map_to_reward_config
    # ------------------------------------------------------------------ #
    @staticmethod
    def map_to_reward_config(
        factor_loadings: dict[str, float],
        current_config: dict[str, Any],
    ) -> dict[str, float]:
        """Map factor betas to reward parameters.

        Parameters
        ----------
        factor_loadings:
            Output of :meth:`compute_target_profile`.
        current_config:
            Current reward config as a flat dict (e.g.
            ``HydraConfig().reward.model_dump()``).

        Returns
        -------
        dict[str, float]
            New reward parameter dict, ready for
            ``HydraConfig.apply_patch({"reward": result})``.
        """
        result = dict(current_config)

        for factor_name, (param, direction, sensitivity) in _FACTOR_MAP.items():
            beta = factor_loadings.get(factor_name)
            if beta is None:
                logger.debug(
                    "Factor %r not found in loadings; skipping.", factor_name
                )
                continue

            current_value = float(result.get(param, 0.0))
            new_value = current_value * (1.0 + direction * beta * sensitivity)
            new_value = _clip_param(param, new_value)
            result[param] = new_value

            logger.debug(
                "%s: %s %.4f -> %.4f  (beta=%.4f, dir=%+d, sens=%.2f)",
                factor_name,
                param,
                current_value,
                new_value,
                beta,
                direction,
                sensitivity,
            )

        # Scale reward_scale based on implied target Sharpe magnitude.
        # Use the absolute market beta as a rough proxy for target Sharpe.
        mkt_beta = factor_loadings.get("Mkt-RF")
        if mkt_beta is not None:
            base_scale = float(result.get("reward_scale", 100.0))
            sharpe_proxy = abs(mkt_beta)
            # Scale up for higher Sharpe targets, down for lower
            adjusted_scale = base_scale * (1.0 + 0.3 * (sharpe_proxy - 1.0))
            result["reward_scale"] = _clip_param("reward_scale", adjusted_scale)

        return result

    # ------------------------------------------------------------------ #
    #  run_constrained_optimization
    # ------------------------------------------------------------------ #
    @staticmethod
    def run_constrained_optimization(
        factor_returns: pd.DataFrame,
        target_sharpe: float = 1.5,
        target_sortino: float = 2.0,
        target_max_dd: float = 0.15,
    ) -> dict[str, float]:
        """Find reward parameters that minimise divergence from a target profile.

        Uses ``scipy.optimize.minimize`` with method ``SLSQP``.

        Parameters
        ----------
        factor_returns:
            Historical factor returns DataFrame.
        target_sharpe:
            Desired annualised Sharpe ratio.
        target_sortino:
            Desired annualised Sortino ratio.
        target_max_dd:
            Target maximum drawdown (as a positive fraction, e.g. 0.15 = 15%).

        Returns
        -------
        dict[str, float]
            Calibrated reward parameter dict.
        """
        # Compute factor statistics from the data
        ann_factor = np.sqrt(252)
        factor_means = factor_returns.mean().values.astype(np.float64)
        factor_stds = factor_returns.std().values.astype(np.float64)
        factor_names = list(factor_returns.columns)

        # Downside deviations
        neg_returns = factor_returns.clip(upper=0.0)
        factor_downside_stds = neg_returns.std().values.astype(np.float64)

        # Cumulative returns for drawdown estimation
        cum_returns = (1 + factor_returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdowns = ((cum_returns - rolling_max) / rolling_max).min().values.astype(
            np.float64
        )
        factor_max_dds = np.abs(drawdowns)  # positive fractions

        # Build bounds for the optimiser (same order as _OPT_PARAMS)
        bounds = [_PARAM_BOUNDS[p] for p in _OPT_PARAMS]

        # Initial point: midpoints of bounds
        x0 = np.array([(lo + hi) / 2.0 for lo, hi in bounds], dtype=np.float64)

        # --- Objective: match implied factor profile to targets ---
        def _objective(x: np.ndarray) -> float:
            params = dict(zip(_OPT_PARAMS, x))

            # Implied Sharpe: higher sharpe_eta and pnl_bonus favour higher
            # risk-adjusted return; drawdown_penalty discourages it.
            implied_sharpe_weight = (
                params["pnl_bonus_weight"] * params["sharpe_eta"] * params["reward_scale"]
            ) / max(params["drawdown_penalty"], 1e-6)

            # Normalise to a comparable scale
            implied_sharpe = implied_sharpe_weight / 100.0

            # Implied Sortino: similar but penalise via drawdown more
            implied_sortino = implied_sharpe * (
                1.0 + params["drawdown_penalty"] / max(params["pnl_bonus_weight"], 1e-6)
            )

            # Implied max DD tolerance: higher drawdown_penalty means lower
            # tolerance.  Transaction and holding penalties also contribute.
            implied_dd = 0.3 / max(
                params["drawdown_penalty"]
                + 0.5 * params["holding_penalty"]
                + 0.3 * params["transaction_penalty"],
                1e-6,
            )

            loss_sharpe = (implied_sharpe - target_sharpe) ** 2
            loss_sortino = (implied_sortino - target_sortino) ** 2
            loss_dd = (implied_dd - target_max_dd) ** 2 * 10.0  # heavier weight

            return float(loss_sharpe + loss_sortino + loss_dd)

        result = minimize(
            _objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-10},
        )

        if not result.success:
            logger.warning(
                "Optimisation did not fully converge: %s", result.message
            )

        calibrated = {}
        for name, val in zip(_OPT_PARAMS, result.x):
            calibrated[name] = _clip_param(name, float(val))

        logger.info(
            "Constrained optimisation finished (success=%s, fun=%.6e, nit=%d).",
            result.success,
            result.fun,
            result.nit,
        )
        return calibrated

    # ------------------------------------------------------------------ #
    #  get_calibration_report
    # ------------------------------------------------------------------ #
    @staticmethod
    def get_calibration_report(
        factor_loadings: dict[str, float],
        proposed_weights: dict[str, float],
        current_weights: dict[str, float],
    ) -> dict[str, Any]:
        """Generate a human-readable calibration report.

        Parameters
        ----------
        factor_loadings:
            Factor name -> beta mapping.
        proposed_weights:
            Proposed reward parameters.
        current_weights:
            Current reward parameters.

        Returns
        -------
        dict
            Report with keys ``factor_loadings``, ``current_vs_proposed``,
            and ``rationale``.
        """
        # Build per-parameter comparison
        all_params = sorted(
            set(list(proposed_weights.keys()) + list(current_weights.keys()))
        )
        current_vs_proposed: dict[str, dict[str, Any]] = {}
        for param in all_params:
            cur = current_weights.get(param)
            prop = proposed_weights.get(param)
            if cur is None or prop is None:
                continue
            cur_f = float(cur)
            prop_f = float(prop)
            if abs(cur_f) > 1e-12:
                change_pct = (prop_f - cur_f) / cur_f * 100.0
            else:
                change_pct = 0.0 if abs(prop_f) < 1e-12 else float("inf")
            current_vs_proposed[param] = {
                "current": cur_f,
                "proposed": prop_f,
                "change_pct": round(change_pct, 2),
            }

        # Build rationale: for each parameter, explain which factor drove it
        rationale: dict[str, str] = {}
        # Invert factor map: reward_param -> list of (factor, direction, sensitivity)
        param_to_factors: dict[str, list[tuple[str, int, float]]] = {}
        for factor_name, (param, direction, sensitivity) in _FACTOR_MAP.items():
            param_to_factors.setdefault(param, []).append(
                (factor_name, direction, sensitivity)
            )

        for param in all_params:
            drivers = param_to_factors.get(param, [])
            if not drivers:
                # Check if it is reward_scale (driven by Mkt-RF magnitude)
                if param == "reward_scale" and "Mkt-RF" in factor_loadings:
                    beta = factor_loadings["Mkt-RF"]
                    rationale[param] = (
                        f"Scaled based on Mkt-RF beta magnitude "
                        f"({beta:+.4f}); higher absolute market exposure "
                        f"implies larger reward scale."
                    )
                continue

            parts = []
            for factor_name, direction, sensitivity in drivers:
                beta = factor_loadings.get(factor_name)
                if beta is None:
                    continue
                dir_label = "positive" if direction == +1 else "negative"
                effect = direction * beta * sensitivity
                if abs(effect) < 1e-6:
                    parts.append(
                        f"{factor_name} beta={beta:+.4f} (negligible effect)"
                    )
                else:
                    change_dir = "increases" if effect > 0 else "decreases"
                    parts.append(
                        f"{factor_name} beta={beta:+.4f} ({dir_label} "
                        f"mapping, sens={sensitivity}) {change_dir} {param} "
                        f"by ~{abs(effect)*100:.1f}%"
                    )
            if parts:
                rationale[param] = "; ".join(parts)

        report = {
            "factor_loadings": dict(factor_loadings),
            "current_vs_proposed": current_vs_proposed,
            "rationale": rationale,
        }

        logger.info("Calibration report generated for %d parameters.", len(current_vs_proposed))
        return report
