# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Guards for the shared drift and any-vs-any regression routines.

The load-bearing test here is (b), the NULL test. A drift fit that reports a
slope is trivially easy to write and useless if it reports one on data with no
drift; without a null test every other check in this file would still pass for
an estimator that always found a trend. Same reasoning as the white-noise null
in test_fit_uncertainty.py and the no-range/no-starts nulls in
test_histogram_range_and_starts.py.

Rules and directions are pinned; no cohort's measured numbers are.
"""

from __future__ import annotations

import numpy as np
import pytest

from smfs_catalog import regression as R

HOUR = 3600.0


def _series(slope_per_hour: float, n: int = 400, noise: float = 5.0,
            days: float = 5.0, seed: int = 0):
    """A value-vs-time series with a known drift rate, in the app's own units:
    x is a Unix-style timestamp in seconds, y is the variable."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, days * 24 * HOUR, n)
    y = 100.0 + (slope_per_hour / HOUR) * x + rng.normal(0.0, noise, n)
    return x, y


# ── (a) it recovers a slope that is really there ─────────────────────────────

def test_recovers_a_known_drift_rate():
    x, y = _series(0.8)
    fit = R.linear_fit(x, y)
    assert fit is not None
    slope, lo, hi = R.per_hour(fit)
    assert lo < 0.8 < hi, "the 95% interval must cover the true drift rate"
    assert slope == pytest.approx(0.8, rel=0.05)


def test_per_hour_is_exactly_the_seconds_slope_times_3600():
    """The reported rate is a unit change and nothing else — no rounding, no
    re-fit. A drift quoted per hour that isn't the fitted per-second slope
    scaled is a second number that can disagree with the first."""
    x, y = _series(0.3)
    fit = R.linear_fit(x, y)
    slope, lo, hi = R.per_hour(fit)
    assert slope == pytest.approx(fit.slope * HOUR, rel=0, abs=0)
    assert (lo, hi) == pytest.approx((fit.slope_ci[0] * HOUR,
                                      fit.slope_ci[1] * HOUR), rel=0, abs=0)


# ── (b) THE NULL TEST: no drift must read as no drift ────────────────────────

def test_null_no_drift_gives_an_interval_straddling_zero():
    """Flat data with noise must not produce a drift verdict.

    Checked over many seeds rather than one: a 95% interval is allowed to miss
    about 1 time in 20, so a single seed proves nothing either way. What is
    pinned is the coverage rate, which is the actual claim the interval makes.
    """
    misses = 0
    trials = 60
    for seed in range(trials):
        x, y = _series(0.0, seed=seed)
        fit = R.linear_fit(x, y)
        assert fit is not None
        _, lo, hi = R.per_hour(fit)
        if not (lo < 0.0 < hi):
            misses += 1
    assert misses <= trials * 0.15, (
        f"{misses}/{trials} null series reported a drift direction; a 95% "
        f"interval should miss about 1 in 20"
    )


def test_null_correlation_on_unrelated_variables_is_not_significant():
    """The same guard for the scatter's rho: two independent variables must
    not routinely come back correlated. This is what makes the fishing warning
    in the scatter tooltip a caution rather than estimator behavior."""
    rng = np.random.default_rng(7)
    hits = 0
    trials = 60
    for _ in range(trials):
        a = rng.normal(size=300)
        b = rng.normal(size=300)
        c = R.correlate(a, b)
        assert c is not None
        if c.p < 0.05:
            hits += 1
    assert hits <= trials * 0.15


# ── (c) the exported covariance must redraw the drawn band ───────────────────

def test_exported_covariance_reproduces_the_drawn_band():
    """A drawn CI band exports its limits and the full
    covariance, never just the diagonal, so someone can redraw the published
    figure from the CSV. Here that is checkable exactly — Var(fit at x) =
    x^2 Var(s) + 2x Cov(s,i) + Var(i) must equal the band we actually drew.

    The off-diagonal is not a formality: acquisition timestamps are ~1.8e9, so
    slope and intercept are almost perfectly anticorrelated and a
    diagonal-only reader would get the band wildly wrong.
    """
    x, y = _series(0.4, seed=3)
    fit = R.linear_fit(x, y)
    (var_s, cov_si), (cov_is, var_i) = fit.slope_cov
    assert cov_si == cov_is, "covariance must be symmetric"

    xs = np.linspace(x.min(), x.max(), 25)
    half_from_cov = fit.t_crit * np.sqrt(xs ** 2 * var_s + 2 * xs * cov_si + var_i)
    lo, hi = fit.band(xs)
    assert np.allclose(half_from_cov, (hi - lo) / 2.0, rtol=1e-9)

    # ... and the diagonal alone would NOT: a null test for the null test, so
    # this check cannot pass by the off-diagonal happening to be negligible.
    half_diag_only = fit.t_crit * np.sqrt(xs ** 2 * var_s + var_i)
    assert not np.allclose(half_diag_only, (hi - lo) / 2.0, rtol=0.5)


def test_stated_standard_errors_match_the_covariance_diagonal():
    x, y = _series(0.4, seed=4)
    fit = R.linear_fit(x, y)
    (var_s, _), (_, var_i) = fit.slope_cov
    assert np.sqrt(var_s) == pytest.approx(fit.slope_se)
    assert np.sqrt(var_i) == pytest.approx(fit.intercept_se)


# ── (d) it is a mean-response band, and shows it ─────────────────────────────

def test_band_is_narrowest_at_the_centre_of_the_data():
    """The bow-tie shape is what makes it a confidence band for the trend
    rather than a constant-width ribbon. If it ever comes out uniform, the
    leverage term has been dropped and the band is wrong at both ends —
    precisely where a drift claim is read."""
    x, y = _series(0.5, seed=5)
    fit = R.linear_fit(x, y)
    xs = np.linspace(x.min(), x.max(), 101)
    lo, hi = fit.band(xs)
    w = hi - lo
    assert w.argmin() == pytest.approx(50, abs=2), "narrowest at x-mean"
    assert w[0] > w[50] and w[-1] > w[50]


def test_band_narrows_as_the_square_root_of_n():
    """A property check, not a comparison against itself: 16x the curves must
    buy about 4x on the interval. Pins the estimator's behaviour without
    pinning any measured number."""
    def half_width(n):
        x, y = _series(0.5, n=n, seed=11)
        f = R.linear_fit(x, y)
        return (f.slope_ci[1] - f.slope_ci[0]) / 2.0
    ratio = half_width(200) / half_width(3200)
    assert 3.2 < ratio < 4.8, f"expected ~4x (sqrt 16), got {ratio:.2f}"


# ── (e) it declines rather than fabricating ──────────────────────────────────

@pytest.mark.parametrize("x, y", [
    ([1.0, 2.0], [3.0, 4.0]),                 # 2 points: exact fit, zero dof
    ([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]),       # no spread in x
    ([], []),
    ([np.nan, np.nan, 1.0], [1.0, 2.0, 3.0]), # only one usable pair left
])
def test_returns_none_rather_than_a_meaningless_line(x, y):
    assert R.linear_fit(x, y) is None
    assert R.correlate(x, y) is None or len(x) >= 3


def test_a_failed_fit_is_recorded_in_the_manifest_not_omitted():
    """An absent key is indistinguishable from an older export that never had
    one. A failure has to say so."""
    m = R.manifest_fields(None, None)
    assert m["fit_ok"] is False
    assert m["fit_reason"]
    assert m["correlation_ok"] is False
    assert "ci_pct" in m and "ci_method" in m


def test_manifest_carries_the_hourly_rate_only_when_x_is_time():
    x, y = _series(0.6)
    fit, corr = R.linear_fit(x, y), R.correlate(x, y)
    assert "slope_per_hour" in R.manifest_fields(fit, corr, x_is_time=True)
    assert "slope_per_hour" not in R.manifest_fields(fit, corr, x_is_time=False)
    m = R.manifest_fields(fit, corr, x_is_time=True)
    assert m["param_cov_order"] == ["slope", "intercept"]
    assert np.shape(m["param_cov"]) == (2, 2)


def test_interval_policy_is_fixed_and_exported_with_its_assumptions():
    """One published CI convention, not a hidden computation/export fork."""
    import inspect

    assert "pct" not in inspect.signature(R.linear_fit).parameters
    fit = R.linear_fit(*_series(0.2))
    m = R.manifest_fields(fit, None, x_is_time=True)
    assert fit.pct == R.CI_PCT == m["ci_pct"]
    assert "independent residuals" in m["ci_assumptions"]
    assert "lower bound" in m["acquisition_order_caution"]


def test_same_variable_scatter_does_not_manufacture_a_regression_record():
    """The UI may show X == Y, but that identity is not a fitted result."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] /
           "smfs_catalog" / "scatter_window.py").read_text(encoding="utf-8")
    assert "self._fit = None if same else _reg.linear_fit" in src
    assert "self._corr = (None if same else" in src


def test_scatter_window_does_not_offer_unreported_log_views():
    """The raw-scale analysis must not silently hide non-positive values or
    apply a logarithm to Unix timestamps through the date axis."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] /
           "smfs_catalog" / "scatter_window.py").read_text(encoding="utf-8")
    assert "log X" not in src
    assert "log Y" not in src
    assert "setLogMode" not in src
    assert '"x_log"' not in src
    assert '"y_log"' not in src


# ── (f) missing values are dropped PAIRWISE ──────────────────────────────────

def test_a_missing_value_drops_its_partner_too():
    """Dropping NaNs per-array would slide y against x and correlate the wrong
    points together — a silent, total corruption that still returns a
    confident-looking number."""
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    y = np.array([2.0, 4.0, 99.0, 8.0, 10.0])
    fit = R.linear_fit(x, y)
    assert fit.n == 4
    assert fit.slope == pytest.approx(2.0)
    assert fit.r2 == pytest.approx(1.0)

    y2 = np.array([2.0, 4.0, 6.0, np.nan, 10.0])
    assert R.linear_fit(x, y2).n == 3


def test_mismatched_lengths_raise_rather_than_truncate():
    with pytest.raises(ValueError):
        R.linear_fit([1.0, 2.0, 3.0], [1.0, 2.0])


# ── (g) the correlation defaults to rank, and says which it used ─────────────

def test_correlation_defaults_to_spearman_and_reports_n():
    x, y = _series(0.5, n=120, seed=9)
    c = R.correlate(x, y)
    assert c.method == "spearman" and c.n == 120
    assert R.correlate(x, y, method="pearson").method == "pearson"
    with pytest.raises(ValueError):
        R.correlate(x, y, method="kendall")


def test_rank_correlation_survives_an_outlier_that_moves_pearson():
    """Why spearman is the default: one wild curve must not decide whether a
    scanned pair looks related."""
    rng = np.random.default_rng(2)
    x = np.arange(200.0)
    y = x + rng.normal(0, 3, 200)
    x_out = np.append(x, 5.0)
    y_out = np.append(y, 5000.0)
    assert R.correlate(x_out, y_out, "spearman").rho > 0.9
    assert R.correlate(x_out, y_out, "pearson").rho < \
           R.correlate(x_out, y_out, "spearman").rho
