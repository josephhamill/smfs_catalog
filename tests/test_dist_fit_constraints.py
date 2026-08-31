# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""User-set parameter limits on a distribution fit.

A limit is prior information the user supplied. These tests pin the three
things that makes true of the numbers downstream of it: the fit honours the
limit, the reported interval narrows because every bootstrap refit honours it
too, and a fit that never touches its limits is bit-for-bit the fit it would
have been without them.
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
    MODELS,
    at_bound_flags,
    bootstrap_fit_ci,
    composite_bounds,
    composite_guess,
    flatten_constraints,
    make_composite,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _binned(data, n_bins=20):
    counts, edges = np.histogram(data, bins=n_bins)
    bw = edges[1] - edges[0]
    centres = (edges[:-1] + edges[1:]) / 2
    density = counts / max(counts.sum() * bw, 1e-10)
    return edges, centres, density


def _fit(data, comps, constraints=None, n_bins=20):
    """Fit the way DistFitWindow does, limits included."""
    edges, centres, density = _binned(data, n_bins)
    fn = make_composite(comps)
    bounds = composite_bounds(comps, data, constraints)
    p0 = composite_guess(comps, data, centres, density)
    p0 = list(np.clip(p0, bounds[0], bounds[1]))
    popt, pcov = optimize.curve_fit(
        fn, centres, density, p0=p0, bounds=bounds, maxfev=20000, method="trf",
    )
    return edges, bounds, popt


def _skewed(seed=0, n=300):
    """Piles up at zero with a tail to the right — the shape that sends a
    Gaussian's mean below zero.

    Shape below 1 puts the mode AT the origin, so the histogram's tallest bar
    is its first. No symmetric curve centred inside that range can match it,
    and least squares slides the centre off the left end instead.
    """
    rng = np.random.default_rng(seed)
    return rng.gamma(shape=0.6, scale=20.0, size=n)


GAUSS = [MODELS["Gaussian"]]
MU = 1          # position is parameter 1 for every registered model


# ── the limit is honoured ─────────────────────────────────────────────────────

def test_a_gaussian_on_skewed_data_puts_its_mean_below_zero_unconstrained():
    """The premise. Without it the rest of this file is testing nothing."""
    _, _, popt = _fit(_skewed(), GAUSS)
    assert popt[MU] < 0.0


def test_a_lower_limit_keeps_the_mean_out_of_the_forbidden_region():
    _, bounds, popt = _fit(_skewed(), GAUSS, [[None, (0.0, None), None]])
    assert popt[MU] >= 0.0
    assert bounds[0][MU] == 0.0


def test_the_fit_lands_on_the_limit_and_says_so():
    _, bounds, popt = _fit(_skewed(), GAUSS, [[None, (0.0, None), None]])
    assert at_bound_flags(popt, bounds[0], bounds[1])[MU] == "lo"


def test_a_limit_the_fit_never_reaches_is_not_flagged():
    """Only a limit that actually stopped the fit is an imposition worth
    reporting; one the fit stayed clear of changed nothing."""
    _, bounds, popt = _fit(_skewed(), GAUSS, [[None, (-1e6, None), None]])
    assert at_bound_flags(popt, bounds[0], bounds[1])[MU] is None


# ── a limit nobody set changes nothing ────────────────────────────────────────

def test_no_constraints_reproduces_the_unconstrained_bounds_exactly():
    data = _skewed()
    assert composite_bounds(GAUSS, data) == composite_bounds(GAUSS, data, None)


def test_a_limit_the_fit_stays_clear_of_leaves_it_where_it_was():
    """A narrowing the fit never reaches must not move the answer.

    To solver tolerance, not to the bit: trf scales its trust region by the
    bound box, so any change to it perturbs the arithmetic in the last few
    digits. That is far below anything the fit reports.
    """
    data = _skewed()
    _, free_b, free = _fit(data, GAUSS)
    # Below where the fit lands, but still inside the automatic box.
    floor = (free[MU] + free_b[0][MU]) / 2
    _, _, slack = _fit(data, GAUSS, [[None, (floor, None), None]])
    assert slack[MU] > floor
    assert np.allclose(free, slack, rtol=1e-5, atol=0.0)


def test_widening_past_the_automatic_limit_can_relocate_the_fit():
    """The override replaces rather than intersects, so a user can hand the
    solver a box the automatic bounds were holding it out of — and on a
    monotone-decreasing histogram there is a degenerate solution out there: an
    enormous Gaussian centred far to the left, approximating the decay with its
    tail. The automatic limit is what normally prevents it.

    Recorded because it is the cost of letting a limit widen, and because the
    resulting parameters are visibly absurd rather than quietly wrong.
    """
    data = _skewed()
    _, _, free = _fit(data, GAUSS)
    _, _, wide = _fit(data, GAUSS, [[None, (-1e9, 1e9), None]])
    assert wide[MU] < free[MU]
    assert wide[0] > 1e12 * free[0]        # amplitude runs away with it


def test_only_the_named_parameter_moves():
    """Limiting μ must leave amp and σ on their automatic limits."""
    data = _skewed()
    auto = composite_bounds(GAUSS, data)
    lim = composite_bounds(GAUSS, data, [[None, (0.0, None), None]])
    assert lim[0][0] == auto[0][0] and lim[1][0] == auto[1][0]     # amp
    assert lim[0][2] == auto[0][2] and lim[1][2] == auto[1][2]     # sigma


# ── impossible limits are refused, not repaired ───────────────────────────────

def test_a_minimum_above_the_maximum_is_refused_by_name():
    with pytest.raises(ValueError, match="μ"):
        composite_bounds(GAUSS, _skewed(), [[None, (50.0, 10.0), None]])


def test_a_user_minimum_above_the_automatic_maximum_is_refused():
    """The override replaces rather than intersects, so it can collide with a
    limit the user never saw. It must be named, not quietly widened."""
    data = np.linspace(1.0, 10.0, 200)
    with pytest.raises(ValueError):
        composite_bounds(GAUSS, data, [[None, None, (1e9, None)]])


def test_a_limit_may_widen_past_the_automatic_one():
    data = _skewed()
    auto = composite_bounds(GAUSS, data)
    lim = composite_bounds(GAUSS, data, [[None, (None, 1e9), None]])
    assert lim[1][MU] == 1e9 > auto[1][MU]


# ── the interval is conditional on the limit ──────────────────────────────────

def test_a_binding_limit_narrows_the_reported_interval():
    """Every refit is handed the same limits, so the percentile interval cannot
    cross one. That is correct GIVEN the limit — and it is exactly why the
    window reports the interval as conditional."""
    data = _skewed()
    edges, free_b, free_p = _fit(data, GAUSS)
    _, lim_b, lim_p = _fit(data, GAUSS, [[None, (0.0, None), None]])

    free = bootstrap_fit_ci(GAUSS, data, edges, True, free_p, free_b,
                            n_draws=120, seed=3)
    lim = bootstrap_fit_ci(GAUSS, data, edges, True, lim_p, lim_b,
                           n_draws=120, seed=3)
    assert free is not None and lim is not None
    assert lim.lo[MU] >= 0.0
    assert free.lo[MU] < lim.lo[MU]
    assert (lim.hi[MU] - lim.lo[MU]) < (free.hi[MU] - free.lo[MU])


def test_no_draw_escapes_the_limit():
    data = _skewed()
    edges, bounds, popt = _fit(data, GAUSS, [[None, (0.0, None), None]])
    boot = bootstrap_fit_ci(GAUSS, data, edges, True, popt, bounds,
                            n_draws=120, seed=5)
    assert boot is not None
    assert boot.lo[MU] >= 0.0 and boot.hi[MU] >= 0.0


# ── the flat per-parameter view lines up with popt ────────────────────────────

def test_flatten_puts_each_limit_on_its_own_parameter():
    comps = [MODELS["Gaussian"], MODELS["Gamma"]]
    flat = flatten_constraints(comps, [[None, (0.0, None), None],
                                       [None, None, (1.0, 5.0)]])
    assert len(flat) == 6
    assert flat == [None, (0.0, None), None, None, None, (1.0, 5.0)]


def test_flatten_tolerates_components_nobody_constrained():
    comps = [MODELS["Gaussian"], MODELS["Weibull"]]
    assert flatten_constraints(comps, [[None, (0.0, None), None]]) == \
        [None, (0.0, None), None, None, None, None]
    assert flatten_constraints(comps, None) == [None] * 6
