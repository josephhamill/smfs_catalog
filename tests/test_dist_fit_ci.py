# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Regression test: distribution-fit confidence intervals are bootstrapped over the
raw values, never read off curve_fit's covariance (#135).

THE DEFECT.  The fit is least squares on ~20 histogram HEIGHTS.  Nothing in that
objective records how many curves are behind each bar, so the covariance cannot
know the sample size, and a symmetric +/- 1.96 sigma interval on a bounded
parameter walks straight out of the possible range.  On a live
catalogue: a Gaussian width of 72.5 nm with a 95 % interval of [-21.4, 166.3],
and a mixing fraction -- a proportion -- reported as [-0.14, 0.86].  The worst
case was the LARGEST cohort (9,071 curves, fraction [-3.11, +4.88]): sample size
does not rescue a flat objective.

NOTE ON NUMBERS.  This file pins RULES and INVARIANTS, not any cohort's
measurements.  How wide a given interval comes out depends on the data; what
must always hold is that it cannot leave the possible range, that it is built
from refits rather than from a covariance, and that draws which fail to converge
are counted rather than dropped.

WHAT IS PINNED HERE:

 (a) THE HEADLINE, as a comparison: on data where the covariance interval puts a
     Gaussian width below zero, the bootstrap does not.  Without this the file
     would only prove the new code is self-consistent.
 (b) THE INVARIANTS, on ordinary data: no negative widths, every mixing fraction
     inside [0, 1].  These hold by construction (bounds apply to every refit,
     fractions are computed per draw) and must keep holding.
 (c) NULL TEST -- where the fit is well determined, the bootstrap agrees with
     the covariance.  A method that always disagreed would be just as wrong.
 (d) The estimator checked against a PROPERTY rather than against itself:
     interval width falls roughly as 1/sqrt(n).
 (e) LABEL SWITCHING -- every draw is ordered left-to-right before percentiles
     are taken, so "Peak 2" means the same component in every draw.
 (f) Failed draws are COUNTED, and too few converged returns None rather than
     falling back to the covariance behind the same label.
 (g) Cancellation is honoured and recorded.
 (h) The manifest names the method that actually ran -- and the covariance
     default still describes the covariance, so mean_curve_window's band (which
     genuinely uses it) is not mislabelled by this change.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import optimize

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smfs_catalog.dist_fit_core import (  # noqa: E402
    BOOTSTRAP_CI_METHOD,
    COV_CI_METHOD,
    MODELS,
    bootstrap_fit_ci,
    ci_manifest_fields,
    composite_bounds,
    composite_guess,
    make_composite,
    order_params_by_centre,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _binned(data, n_bins=20):
    counts, edges = np.histogram(data, bins=n_bins)
    bw = edges[1] - edges[0]
    centres = (edges[:-1] + edges[1:]) / 2
    density = counts / max(counts.sum() * bw, 1e-10)
    return edges, centres, density


def _fit(data, n_peaks=2, n_bins=20):
    """Fit exactly the way DistFitWindow does, and return everything after it."""
    edges, centres, density = _binned(data, n_bins)
    comps = [MODELS["Gaussian"]] * n_peaks
    fn = make_composite(comps)
    bounds = composite_bounds(comps, data)
    popt, pcov = optimize.curve_fit(
        fn, centres, density, p0=composite_guess(comps, data, centres, density),
        bounds=bounds, maxfev=20000, method="trf",
    )
    return comps, edges, bounds, order_params_by_centre(comps, popt), pcov


def _two_peaks(seed, n1=90, n2=40, mu1=100, sd1=25, mu2=150, sd2=45):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(mu1, sd1, n1), rng.normal(mu2, sd2, n2)])


SIGMA_IDX = (2, 5)      # the width parameter of each Gaussian component


# ── (a) the headline, as a comparison ─────────────────────────────────────────

def test_covariance_can_report_a_negative_width_and_the_bootstrap_cannot():
    """Both methods on one dataset: the old one leaves the possible range.

    Seeded so this is a fixed dataset, not a lucky draw. Two overlapping
    Gaussians and 130 values is an ordinary request, not a pathological one --
    which is the point.
    """
    data = _two_peaks(seed=0)
    comps, edges, bounds, popt, pcov = _fit(data)

    cov_lo = popt - 1.96 * np.sqrt(np.diag(pcov))
    assert min(cov_lo[i] for i in SIGMA_IDX) < 0, (
        "this dataset is the fixture for the defect; if the covariance no "
        "longer goes negative here, pick another one rather than deleting "
        "the check"
    )

    boot = bootstrap_fit_ci(comps, data, edges, True, popt, bounds, n_draws=200)
    assert boot is not None
    assert min(boot.lo[i] for i in SIGMA_IDX) > 0, (
        "a resample is refitted under the same bounds, so no draw can hold a "
        "negative width and no percentile of them can either"
    )


# ── (b) the invariants ────────────────────────────────────────────────────────

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_widths_stay_positive_and_fractions_stay_in_the_unit_interval(seed):
    data = _two_peaks(seed=seed)
    comps, edges, bounds, popt, _ = _fit(data)
    boot = bootstrap_fit_ci(comps, data, edges, True, popt, bounds, n_draws=150)
    assert boot is not None
    for i in SIGMA_IDX:
        assert boot.lo[i] > 0 and boot.hi[i] > boot.lo[i]
    assert np.all(boot.frac_lo >= 0.0) and np.all(boot.frac_hi <= 1.0), (
        "mixing fractions come from each draw's own amplitudes; dividing an "
        "interval by a total is what put a proportion outside [0, 1]"
    )


# ── (c) null test ─────────────────────────────────────────────────────────────

def test_agrees_with_the_covariance_where_the_fit_is_well_determined():
    """Well-separated peaks, plenty of data: the two methods should not disagree.

    This is the check that stops the bootstrap from being 'a method that always
    returns something different'. Compared on the peak CENTRES, which are what
    the fit determines best.
    """
    rng = np.random.default_rng(11)
    data = np.concatenate([rng.normal(60, 8, 3000), rng.normal(160, 12, 2000)])
    comps, edges, bounds, popt, pcov = _fit(data, n_bins=40)
    perr = np.sqrt(np.diag(pcov))
    boot = bootstrap_fit_ci(comps, data, edges, True, popt, bounds, n_draws=200)
    assert boot is not None
    for i in (1, 4):                       # the two mu parameters
        cov_w = 2 * 1.96 * perr[i]
        boot_w = boot.hi[i] - boot.lo[i]
        assert 0.2 < boot_w / cov_w < 5.0, (
            f"parameter {i}: covariance width {cov_w:.3g} vs bootstrap "
            f"{boot_w:.3g} -- these should be the same order where the "
            f"objective is well conditioned"
        )


# ── (d) checked against a property, not against itself ────────────────────────

def test_interval_width_falls_roughly_as_one_over_root_n():
    """Four times the data should roughly halve the interval.

    Checked on the peak centre of a single well-behaved Gaussian, where 1/sqrt(n)
    is the honest expectation. Generous bounds: this is asserting a trend, not a
    coefficient.
    """
    rng = np.random.default_rng(5)
    pool = rng.normal(100, 20, 8000)
    widths = []
    for n in (500, 8000):
        data = pool[:n]
        comps, edges, bounds, popt, _ = _fit(data, n_peaks=1, n_bins=30)
        boot = bootstrap_fit_ci(comps, data, edges, True, popt, bounds,
                                n_draws=150)
        assert boot is not None
        widths.append(boot.hi[1] - boot.lo[1])
    ratio = widths[0] / widths[1]
    assert 2.0 < ratio < 8.0, (
        f"16x the data narrowed the interval {ratio:.2f}x; sqrt(16) = 4 is the "
        f"expectation, so this is either not sampling variability or the "
        f"resampling is not doing anything"
    )


# ── (e) label switching ───────────────────────────────────────────────────────

def test_every_draw_is_ordered_left_to_right_before_percentiles():
    comps = [MODELS["Gaussian"]] * 2
    right_first = np.array([1.0, 200.0, 30.0, 2.0, 50.0, 10.0])
    ordered = order_params_by_centre(comps, right_first)
    assert list(ordered) == [2.0, 50.0, 10.0, 1.0, 200.0, 30.0], (
        "a draw whose components came back in the other order must be "
        "reordered, or 'Peak 2' means different things in different draws and "
        "the percentile describes neither"
    )
    # ...and an already-ordered vector is left alone.
    assert list(order_params_by_centre(comps, ordered)) == list(ordered)


def test_swapped_draws_do_not_widen_the_interval():
    """Two nearly identical peaks: without ordering, the interval spans both.

    Constructed rather than sampled, so the failure is unambiguous.
    """
    comps = [MODELS["Gaussian"]] * 2
    a = np.array([1.0, 100.0, 10.0, 1.0, 101.0, 10.0])
    b = np.array([1.0, 101.0, 10.0, 1.0, 100.0, 10.0])   # same fit, swapped
    ordered = np.array([order_params_by_centre(comps, p) for p in (a, b)])
    assert np.ptp(ordered[:, 1]) <= 1.0 and np.ptp(ordered[:, 4]) <= 1.0, (
        "after ordering, the two draws agree; before it they would look like "
        "a 1 nm spread on each of two peaks that never moved"
    )


# ── (f) failed draws are counted; too few returns None ────────────────────────

def test_failed_draws_are_counted_not_silently_dropped(monkeypatch):
    data = _two_peaks(seed=1)
    comps, edges, bounds, popt, _ = _fit(data)

    real = optimize.curve_fit
    state = {"i": 0}

    def flaky(*a, **kw):
        state["i"] += 1
        if state["i"] % 2 == 0:
            raise RuntimeError("did not converge")
        return real(*a, **kw)

    monkeypatch.setattr("smfs_catalog.dist_fit_core.optimize.curve_fit", flaky)
    boot = bootstrap_fit_ci(comps, data, edges, True, popt, bounds, n_draws=100)
    assert boot is not None
    assert boot.n_failed > 0
    assert boot.n_ok + boot.n_failed == boot.n_draws, (
        "every draw is accounted for; a reader told '100 draws' must not be "
        "looking at an interval built from an unstated number of them"
    )


def test_returns_none_rather_than_falling_back_to_the_covariance(monkeypatch):
    data = _two_peaks(seed=2)
    comps, edges, bounds, popt, _ = _fit(data)

    def always_fails(*a, **kw):
        raise RuntimeError("did not converge")

    monkeypatch.setattr("smfs_catalog.dist_fit_core.optimize.curve_fit",
                        always_fails)
    assert bootstrap_fit_ci(comps, data, edges, True, popt, bounds,
                            n_draws=50) is None, (
        "no interval is the honest answer; substituting a different method "
        "behind the same column heading is the defect this fixes"
    )


# ── (g) cancellation ──────────────────────────────────────────────────────────

def test_cancelling_is_honoured_and_recorded():
    data = _two_peaks(seed=3)
    comps, edges, bounds, popt, _ = _fit(data)

    calls = {"n": 0}

    def cancel_after_60(i, n):
        calls["n"] += 1
        return i >= 60

    boot = bootstrap_fit_ci(comps, data, edges, True, popt, bounds,
                            n_draws=400, progress=cancel_after_60)
    assert calls["n"] <= 62, "cancelling must stop the loop, not just flag it"
    assert boot is not None and boot.cancelled
    assert boot.n_ok < boot.n_draws


def test_cancelling_too_early_yields_no_interval():
    data = _two_peaks(seed=3)
    comps, edges, bounds, popt, _ = _fit(data)
    boot = bootstrap_fit_ci(comps, data, edges, True, popt, bounds,
                            n_draws=400, progress=lambda i, n: i >= 3)
    assert boot is None, (
        "three refits is not an interval; better to report none than to "
        "percentile the resampler"
    )


# ── (h) determinism and the manifest ──────────────────────────────────────────

def test_same_seed_reproduces_the_interval_exactly():
    data = _two_peaks(seed=4)
    comps, edges, bounds, popt, _ = _fit(data)
    kw = dict(n_draws=80)
    a = bootstrap_fit_ci(comps, data, edges, True, popt, bounds, **kw)
    b = bootstrap_fit_ci(comps, data, edges, True, popt, bounds, **kw)
    assert np.allclose(a.lo, b.lo) and np.allclose(a.hi, b.hi), (
        "a Monte Carlo interval that cannot be reproduced from its recorded "
        "seed is not reproducible provenance"
    )


def test_manifest_names_the_method_that_actually_ran():
    data = _two_peaks(seed=0)
    comps, edges, bounds, popt, pcov = _fit(data)
    boot = bootstrap_fit_ci(comps, data, edges, True, popt, bounds, n_draws=60)
    fields = boot.manifest_fields(pcov)

    assert fields["ci_method"] == BOOTSTRAP_CI_METHOD
    assert fields["ci_n_draws_ok"] == boot.n_ok
    assert fields["ci_n_draws_failed"] == boot.n_failed
    assert fields["ci_seed"] == boot.seed and fields["ci_pct"] == boot.pct
    assert fields["param_covariance"] is not None, (
        "the covariance still ships: it is what regenerates the interval this "
        "window reported before #135, for anyone re-checking earlier work"
    )

    # The default must still describe the covariance -- mean_curve_window's
    # band genuinely uses it, and this change must not relabel that.
    assert ci_manifest_fields(pcov, True)["ci_method"] == COV_CI_METHOD


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
