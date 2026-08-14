# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Guards for #93 — dist_fit_core's AIC/AICc/BIC must be on the SAME scale as
gmm_fit_core's, which means the same likelihood (per sample, not per bin) and
the same parameter-counting convention as sklearn's.

The load-bearing test is (b): the criteria are checked against sklearn's own
numbers on the same data, not against a second copy of our own formula. A test
that only re-derived our arithmetic would have passed happily for the whole
period the two windows disagreed.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import optimize

from smfs_catalog import dist_fit_core as D


def _bimodal(n1=1200, n2=800, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(100.0, 10.0, n1),
                           rng.normal(200.0, 15.0, n2)])


def _ls_fit(values, names, n_bins=20):
    """The app's own path: least squares on the histogram density."""
    comps = [D.MODELS[n] for n in names]
    counts, edges = np.histogram(values, bins=n_bins)
    bc = 0.5 * (edges[1:] + edges[:-1])
    bw = edges[1] - edges[0]
    dens = counts / (counts.sum() * bw)
    fn = D.make_composite(comps)
    popt, _ = optimize.curve_fit(
        fn, bc, dens, p0=D.composite_guess(comps, values, bc, dens),
        bounds=D.composite_bounds(comps, values), maxfev=40000)
    return D.fit_stats(dens, fn(bc, *popt), len(popt), values, comps, popt), comps, popt


# ── (a) n is the sample count, not the bin count ─────────────────────────────

def test_the_criteria_are_computed_over_values_not_bins():
    """The whole of #93 in one assertion. Before the fix `n` in the AIC/BIC
    formulae was the number of histogram bars."""
    values = _bimodal()
    for n_bins in (20, 60):
        gof, _, _ = _ls_fit(values, ["Gaussian", "Gaussian"], n_bins=n_bins)
        assert gof["n (for IC)"] == values.size


def test_binning_barely_moves_the_criteria():
    """A corollary worth pinning separately: tripling the bin count must not
    move a model-selection statistic much, because the statistic is about the
    values. On the old basis it moved the criteria by their own magnitude."""
    values = _bimodal()
    a, _, _ = _ls_fit(values, ["Gaussian", "Gaussian"], n_bins=20)
    b, _, _ = _ls_fit(values, ["Gaussian", "Gaussian"], n_bins=60)
    assert abs(a["AICc"] - b["AICc"]) / abs(a["AICc"]) < 0.05


# ── (b) THE CROSS-CHECK: against sklearn, not against ourselves ──────────────

def test_parameter_count_matches_sklearns_convention():
    """means + covariances + (k-1) weights. Ours carries an extra `amp` per
    component whose total is fixed by the data, so exactly one is dropped."""
    sklearn_mixture = pytest.importorskip("sklearn.mixture")
    values = _bimodal()
    for names, k in ((["Gaussian"], 1), (["Gaussian", "Gaussian"], 2)):
        gof, _, _ = _ls_fit(values, names)
        gm = sklearn_mixture.GaussianMixture(k, random_state=0).fit(
            values.reshape(-1, 1))
        assert gof["k (for IC)"] == gm._n_parameters(), (
            f"{k}-component parameter count disagrees with sklearn")


def test_a_well_specified_fit_lands_on_sklearns_own_aic():
    """Where the model FITS, the least-squares estimate is close to the
    maximum-likelihood one, so our AIC must land on sklearn's. This is the
    claim '#93 is fixed' actually makes; everything else is arithmetic."""
    sklearn_mixture = pytest.importorskip("sklearn.mixture")
    values = _bimodal()
    gof, _, _ = _ls_fit(values, ["Gaussian", "Gaussian"])
    X = values.reshape(-1, 1)
    gm = sklearn_mixture.GaussianMixture(2, random_state=0, n_init=5).fit(X)
    for ours, theirs in ((gof["AIC"], gm.aic(X)), (gof["BIC"], gm.bic(X))):
        assert abs(ours - theirs) / abs(theirs) < 0.01
    # ... and never BELOW it: sklearn's is the maximum-likelihood estimate, so
    # nothing fitted by least squares may score better. A value under it means
    # the likelihood is being computed on a different footing again.
    assert gof["AIC"] >= gm.aic(X) - 1e-6


# ── (c) it still picks the right model ───────────────────────────────────────

def test_two_components_still_beat_one_on_bimodal_data():
    values = _bimodal()
    g1, _, _ = _ls_fit(values, ["Gaussian"])
    g2, _, _ = _ls_fit(values, ["Gaussian", "Gaussian"])
    assert g2["AICc"] < g1["AICc"] and g2["BIC"] < g1["BIC"]


def test_null_one_component_wins_on_unimodal_data():
    """The null test. Without it every check above would pass for a statistic
    that always preferred more components — which is the failure mode a
    likelihood-based criterion exists to prevent."""
    rng = np.random.default_rng(4)
    values = rng.normal(150.0, 20.0, 3000)
    g1, _, _ = _ls_fit(values, ["Gaussian"])
    g2, _, _ = _ls_fit(values, ["Gaussian", "Gaussian"])
    assert g1["BIC"] < g2["BIC"], (
        "BIC must not buy a second component on genuinely unimodal data")


# ── (d) the renormalisation, and the mis-specification caveat ────────────────

def test_density_is_renormalised_because_least_squares_does_not_constrain_mass():
    """A single Gaussian LS-fitted to bimodal data takes one hump and comes
    back with total mass well under 1. The likelihood is of the shape it found,
    with the mass restored — an unnormalised curve has no likelihood at all."""
    values = _bimodal()
    _, comps, popt = _ls_fit(values, ["Gaussian"])
    assert popt[0] < 0.9, "expected the LS fit to abandon mass here"
    dens = D.composite_density(comps, popt)
    grid = np.linspace(0.0, 400.0, 40001)
    assert np.trapezoid(dens(grid), grid) == pytest.approx(1.0, abs=1e-3)


def test_an_unnormalisable_fit_reports_no_criteria_rather_than_a_number():
    values = _bimodal()
    comps = [D.MODELS["Gaussian"]]
    assert D.composite_density(comps, [0.0, 100.0, 10.0]) is None
    assert D.composite_density(comps, [np.nan, 100.0, 10.0]) is None
    gof = D.fit_stats(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 3,
                      values, comps, [0.0, 100.0, 10.0])
    assert np.isnan(gof["AICc"]) and np.isnan(gof["log-likelihood"])
    assert np.isfinite(gof["R²"]), "the binned stats are still real"


def test_values_the_model_gives_no_density_are_counted_not_dropped():
    """Dropping them would shrink n for whichever model fails worst — the
    opposite of what a model-selection statistic should do."""
    comps = [D.MODELS["Gamma"]]           # support x > 0 only
    values = np.array([-5.0, -1.0, 1.0, 2.0, 3.0, 4.0])
    gof = D.fit_stats(np.array([1.0, 1.0]), np.array([1.0, 1.0]), 3,
                      values, comps, [1.0, 2.0, 1.5])
    assert gof["n_zero_density"] == 2
    assert gof["n (for IC)"] == 6, "n must not shrink to hide the misses"
    assert np.isfinite(gof["AICc"])


# ── (e) every result says which basis produced it ────────────────────────────

def test_every_gof_is_stamped_with_its_basis():
    """A stored AICc must say which method made it: nothing in the number
    itself does, and the comparison table ranks on it. Same reasoning as
    #134's payload-version bump."""
    values = _bimodal()
    gof, _, _ = _ls_fit(values, ["Gaussian"])
    assert gof["ic_basis"] == D.IC_BASIS
    assert D.IC_BASIS != D.IC_BASIS_LEGACY


def test_the_binned_statistics_are_still_binned():
    """R2/chi-2/RSS describe the fit to the histogram — that is what least
    squares minimised — so they must still move with the bin count. Pinned so
    nobody later 'unifies' them onto the per-sample footing, where they would
    mean nothing."""
    values = _bimodal()
    a, _, _ = _ls_fit(values, ["Gaussian", "Gaussian"], n_bins=20)
    b, _, _ = _ls_fit(values, ["Gaussian", "Gaussian"], n_bins=60)
    assert a["DOF"] != b["DOF"]
    assert a["RSS"] != b["RSS"]
