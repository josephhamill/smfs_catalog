# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/regression.py
#
# Regression over paired raw values, shared by variable drift and any-vs-any
# scatter views. OLS supplies the line and its analytic mean-response interval;
# Spearman supplies the default association measure. OLS intervals assume
# independent, constant-variance residuals, so acquisition-order correlation
# can make a drift interval too narrow. Qt-free and DB-free.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# One interval level across the application.
CI_PCT = 95.0

# Drift rates are reported per hour; timestamps are stored in seconds.
SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class Correlation:
    """A correlation coefficient, p-value, method, and paired sample size."""
    rho:    float
    p:      float
    n:      int
    method: str          # "pearson" | "spearman"


@dataclass(frozen=True)
class LinearFit:
    """An OLS line and the statistics needed to report and redraw its CI.

    `slope_cov` is the full covariance in (slope, intercept) order. Its
    off-diagonal term is required to reconstruct the confidence band.
    """
    n:            int
    slope:        float
    intercept:    float
    slope_se:     float
    intercept_se: float
    slope_ci:     tuple[float, float]
    slope_cov:    tuple[tuple[float, float], tuple[float, float]]
    r:            float
    r2:           float
    p_slope:      float
    resid_sd:     float      # s: residual standard deviation, in y's units
    x_mean:       float
    sxx:          float      # sum (x - x_mean)^2 — the leverage denominator
    t_crit:       float
    pct:          float

    @property
    def dof(self) -> int:
        return self.n - 2

    def predict(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return self.slope * x + self.intercept

    def band(self, x) -> tuple[np.ndarray, np.ndarray]:
        """Confidence band for the mean response, not a prediction interval."""
        x  = np.asarray(x, dtype=float)
        se = self.resid_sd * np.sqrt(1.0 / self.n + (x - self.x_mean) ** 2 / self.sxx)
        fit = self.predict(x)
        half = self.t_crit * se
        return fit - half, fit + half


def _finite_pairs(x, y) -> tuple[np.ndarray, np.ndarray]:
    """Return positions where both values are finite; never shift either axis."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("x and y must be the same length")
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def linear_fit(x, y) -> LinearFit | None:
    """OLS of y on x; None for fewer than three pairs or constant x."""
    from scipy import stats

    x, y = _finite_pairs(x, y)
    n = int(x.size)
    if n < 3:
        return None
    x_mean = float(x.mean())
    sxx    = float(np.sum((x - x_mean) ** 2))
    if not np.isfinite(sxx) or sxx <= 0.0:
        return None

    res = stats.linregress(x, y)
    slope, intercept = float(res.slope), float(res.intercept)
    dof = n - 2
    resid   = y - (slope * x + intercept)
    resid_sd = float(np.sqrt(np.sum(resid ** 2) / dof)) if dof > 0 else 0.0

    slope_se     = float(res.stderr)
    intercept_se = float(res.intercept_stderr)
    # Cov(slope, intercept) = -x_mean * Var(slope): the intercept is the line
    # extrapolated back to x = 0, so the further x sits from zero the more
    # tightly the two parameters trade off against each other.
    var_s   = slope_se ** 2
    cov_si  = -x_mean * var_s
    t_crit  = float(stats.t.ppf(0.5 + CI_PCT / 200.0, dof)) if dof > 0 else float("inf")
    half    = t_crit * slope_se

    return LinearFit(
        n            = n,
        slope        = slope,
        intercept    = intercept,
        slope_se     = slope_se,
        intercept_se = intercept_se,
        slope_ci     = (slope - half, slope + half),
        slope_cov    = ((var_s, cov_si), (cov_si, intercept_se ** 2)),
        r            = float(res.rvalue),
        r2           = float(res.rvalue) ** 2,
        p_slope      = float(res.pvalue),
        resid_sd     = resid_sd,
        x_mean       = x_mean,
        sxx          = sxx,
        t_crit       = t_crit,
        pct          = CI_PCT,
    )


def correlate(x, y, method: str = "spearman") -> Correlation | None:
    """Correlation over finite pairs.

    Spearman is the default because it measures monotonic association without
    requiring linearity and is less sensitive to individual outliers. Pearson
    remains available when linear association is specifically wanted.
    """
    from scipy import stats

    x, y = _finite_pairs(x, y)
    n = int(x.size)
    if n < 3:
        return None
    # A constant array has no correlation to report, in either method. Caught
    # here rather than left to scipy, which returns NaN with a warning — and a
    # warning printed to a terminal nobody is watching is not an answer.
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return None
    if method == "pearson":
        r, p = stats.pearsonr(x, y)
    elif method == "spearman":
        r, p = stats.spearmanr(x, y)
    else:
        raise ValueError(f"unknown correlation method: {method!r}")
    if not np.isfinite(r):
        return None
    return Correlation(rho=float(r), p=float(p), n=n, method=method)


def per_hour(fit: LinearFit) -> tuple[float, float, float]:
    """Return slope and CI in y-units/hour for second-based timestamps."""
    k = SECONDS_PER_HOUR
    lo, hi = fit.slope_ci
    return fit.slope * k, lo * k, hi * k


def manifest_fields(fit: LinearFit | None, corr: Correlation | None,
                    *, x_is_time: bool = False) -> dict:
    """Return reproducibility fields, including explicit failure state."""
    out: dict = {
        "ci_pct": CI_PCT,
        "ci_method": "ols_analytic_mean_response",
        "ci_assumptions": "independent residuals with constant variance",
        "acquisition_order_caution": (
            "for consecutive measurements, residual autocorrelation can make "
            "the OLS confidence interval a lower bound on uncertainty"
        ),
    }
    if fit is None:
        out.update(fit_ok=False,
                   fit_reason="fewer than 3 finite pairs, or x has no spread")
    else:
        out.update(
            fit_ok        = True,
            n_pairs       = fit.n,
            slope         = fit.slope,
            slope_se      = fit.slope_se,
            slope_ci_lo   = fit.slope_ci[0],
            slope_ci_hi   = fit.slope_ci[1],
            intercept     = fit.intercept,
            intercept_se  = fit.intercept_se,
            param_cov     = [list(r) for r in fit.slope_cov],
            param_cov_order = ["slope", "intercept"],
            residual_sd   = fit.resid_sd,
            r             = fit.r,
            r_squared     = fit.r2,
            p_slope       = fit.p_slope,
            dof           = fit.dof,
        )
        if x_is_time:
            s, lo, hi = per_hour(fit)
            out.update(slope_per_hour=s,
                       slope_per_hour_ci_lo=lo,
                       slope_per_hour_ci_hi=hi,
                       x_unit="s (Unix timestamp)")
    if corr is None:
        out.update(correlation_ok=False)
    else:
        out.update(correlation_ok=True,
                   correlation_method=corr.method,
                   correlation_rho=corr.rho,
                   correlation_p=corr.p,
                   correlation_n=corr.n)
    return out
