# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""Regression tests for shared histogram geometry and component-start hints.

The tests cover unchanged defaults, explicit-range exclusion reporting,
single-sourced limits, movable start hints, and stable component identity
through left-to-right ordering.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smfs_catalog import histogram_binning as hb          # noqa: E402
from smfs_catalog.dist_fit_core import (                  # noqa: E402
    MODELS,
    centre_permutation,
    composite_bounds,
    composite_guess,
    sort_components_by_centre,
)

SRC = {p.name: p.read_text(encoding="utf-8")
       for p in (ROOT / "smfs_catalog").glob("*.py")}


def _hist(values, n_bins=20):
    counts, edges = np.histogram(values, bins=n_bins)
    bw = edges[1] - edges[0]
    centres = (edges[:-1] + edges[1:]) / 2
    return centres, counts / max(counts.sum() * bw, 1e-10)


@pytest.fixture(scope="module")
def values():
    rng = np.random.default_rng(0)
    return np.concatenate([rng.normal(75, 14, 480), rng.normal(140, 30, 235)])


# ── Histogram (a) null test ─────────────────────────────────────────────────────

def test_no_explicit_range_is_exactly_the_old_behaviour(values):
    """user_bins with no range must equal np.histogram(values, bins=n)."""
    for n in (5, 20, 137):
        bins = hb.user_bins(values, n)
        assert np.allclose(bins.edges, np.histogram(values, bins=n)[1])
        assert bins.n_out_of_range == 0
        assert bins.count(values).sum() == len(values), (
            "with the full range nothing may fall outside; if it does, the "
            "default answer has changed"
        )


# ── Histogram (b) narrowed range reports exclusions ─────────────────────────────

def test_a_narrowed_range_counts_what_it_excluded(values):
    lo, hi = 60.0, 120.0
    bins = hb.user_bins(values, 20, lo, hi)
    expected_below = int(np.count_nonzero(values < lo))
    expected_above = int(np.count_nonzero(values > hi))
    assert (bins.n_below, bins.n_above) == (expected_below, expected_above)
    assert bins.n_out_of_range > 0, "fixture must actually exclude something"
    assert bins.count(values).sum() == len(values) - bins.n_out_of_range, (
        "the bars hold exactly the in-range values; excluded ones are neither "
        "binned nor piled into the end bars"
    )


def test_an_inverted_range_falls_back_rather_than_inventing_edges(values):
    bins = hb.user_bins(values, 20, 200.0, 50.0)
    assert bins is not None and bins.n_bins == 20
    assert bins.range_lo < bins.range_hi


def test_the_fit_window_states_the_exclusion_on_screen_and_in_the_manifest():
    src = SRC["dist_fit_window.py"]
    assert "n_excluded_by_range" in src
    assert src.count("n_excluded_by_range") >= 2, (
        "the count belongs in the fit record AND in export_provenance -- a "
        "number that lives only on a label is one nobody reads later"
    )
    assert "_excluded_lbl" in src


def test_the_fit_uses_only_values_inside_the_selected_range():
    src = SRC["dist_fit_window.py"]
    assert "data        = self._plot.fit_data" in src
    assert "n_tot = float(self.fit_data.size)" in src
    assert "fit_stats(density, y_fit_bins, len(popt), data" in src


def test_saved_fit_preserves_non_numeric_gof_provenance_and_configuration():
    src = SRC["dist_fit_window.py"]
    assert "gof_json = json.dumps(f[\"gof\"])" in src
    assert '"normalized": self._plot.normalized' in src
    assert '"n_excluded_by_range": f["n_excluded_by_range"]' in src
    assert '"user_peak_starts": f["starts"]' in src


# ── Histogram (c) ceilings ──────────────────────────────────────────────────────

def test_the_user_bin_ceiling_is_single_sourced_and_large():
    assert hb.MAX_USER_BINS >= 20_000
    assert hb.MAX_AUTO_BINS < hb.MAX_USER_BINS, (
        "the automatic rule's taste limit and the user's ceiling are different "
        "questions and must not collapse into one number"
    )
    assert "_hb.MAX_USER_BINS" in SRC["dist_fit_window.py"], (
        "the spin box must take its ceiling from the constant, not retype one"
    )


def test_no_window_rolls_its_own_bin_count_rule():
    """The `clip(len/5, 10, 50)` fork, and anything shaped like it."""
    assert not re.search(r"min\(\s*50\s*,\s*len\(", SRC["event_summary_window.py"]), (
        "event_summary_window's ad-hoc bin rule is what made a window and its "
        "own export disagree about the bins"
    )
    for name in ("event_summary_window.py", "dist_fit_window.py"):
        assert "histogram_binning" in SRC[name], (
            f"{name} draws histograms and must take its geometry from the one "
            f"module that decides it"
        )


def test_degenerate_robust_range_reports_tails_against_actual_edges():
    values = np.r_[np.zeros(1000), 0.5]
    bins = hb.robust_bins(values)
    assert bins is not None
    assert bins.count(values).sum() == len(values)
    assert bins.n_out_of_range == 0, (
        "the padded fallback edges include 0.5, so it must not be reported as "
        "outside the displayed histogram"
    )


@pytest.mark.parametrize("outlier", [100.0, -100.0])
def test_full_range_export_keeps_rare_outlier_when_percentiles_collapse(outlier):
    values = np.r_[np.zeros(1000), outlier]
    bins = hb.full_range_bins(values)
    assert bins is not None
    assert bins.edges[0] <= values.min()
    assert bins.edges[-1] >= values.max()
    assert bins.n_out_of_range == 0
    assert bins.count(values).sum() == len(values)


# ── Starts (a) null test ────────────────────────────────────────────────────────

def test_no_starts_leaves_the_guess_exactly_as_it_was(values):
    centres, density = _hist(values)
    comps = [MODELS["Gaussian"]] * 2
    auto = composite_guess(comps, values, centres, density)
    for starts in (None, [], [None, None]):
        assert np.allclose(composite_guess(comps, values, centres, density,
                                           starts), auto), (
            "until somebody sets a value, the automatic answer must be "
            "bit-for-bit what it always was"
        )


# ── Starts (b) hint, not pin ────────────────────────────────────────────────────

def test_a_start_moves_the_starting_point_and_nothing_else(values):
    centres, density = _hist(values)
    comps = [MODELS["Gaussian"]] * 2
    before = composite_bounds(comps, values)
    guess = composite_guess(comps, values, centres, density, [None, 140.0])
    after = composite_bounds(comps, values)
    assert guess[4] == 140.0, "the component the user touched starts where they said"
    assert before == after, (
        "a starting position must not narrow a bound -- that would turn the "
        "user's prior into a fitted-looking number"
    )


def test_the_fit_walks_away_from_a_start_the_data_disagrees_with(values):
    """A start far from any real peak must not hold the component there."""
    from scipy import optimize
    from smfs_catalog.dist_fit_core import make_composite

    centres, density = _hist(values)
    comps = [MODELS["Gaussian"]] * 2
    fn = make_composite(comps)
    bounds = composite_bounds(comps, values)
    p0 = composite_guess(comps, values, centres, density, [None, 250.0])
    popt, _ = optimize.curve_fit(fn, centres, density, p0=p0, bounds=bounds,
                                 maxfev=20000, method="trf")
    assert abs(popt[4] - 250.0) > 20.0, (
        "the fit is free to move a hinted component, and on data that "
        "disagrees it must"
    )


# ── Starts (c) component identity ───────────────────────────────────────────────

def test_components_keep_their_identity_when_starts_are_out_of_order(values):
    centres, density = _hist(values)
    comps = [MODELS["Gaussian"]] * 2
    guess = composite_guess(comps, values, centres, density, [200.0, 60.0])
    assert guess[1] == 200.0 and guess[4] == 60.0, (
        "component 1 was told 200 and component 2 was told 60; re-sorting them "
        "into position order would hand each the other's start, which is "
        "exactly the mixed-component-type problem this feature exists for"
    )
    for sigma in (guess[2], guess[5]):
        assert np.isfinite(sigma) and sigma > 0, (
            "widths are derived from the spacing between components and must "
            "stay positive when that spacing runs right-to-left"
        )


def test_a_start_follows_its_component_through_the_sort():
    """centre_permutation is what keeps a start attached to its component."""
    comps = [MODELS["Gaussian"]] * 2
    popt = np.array([1.0, 200.0, 30.0, 2.0, 50.0, 10.0])   # right-hand first
    perm = centre_permutation(comps, popt)
    assert perm == [1, 0]

    starts = ["start-of-A", "start-of-B"]
    permuted = [starts[i] for i in perm]
    _, s_popt, *_ = sort_components_by_centre(
        comps, popt, np.zeros(6), np.zeros(6), np.zeros(6),
        ["c0", "c1"], ["l0", "l1"],
    )
    assert s_popt[1] == 50.0, "sort puts the left-hand component first"
    assert permuted[0] == "start-of-B", (
        "the left-hand component after sorting is the one that was second "
        "before it, so its start must travel with it"
    )


# ── Starts (d) control ──────────────────────────────────────────────────────────

def test_peak_row_start_round_trips_including_auto():
    from PyQt6.QtWidgets import QApplication
    from smfs_catalog.dist_fit_window import PeakRow

    app = QApplication.instance() or QApplication([])
    row = PeakRow(0, "Gaussian", "#2d6cdf")
    assert row.start is None, "a new component starts on 'auto', as before"
    row.set_start(140.0)
    assert row.start == pytest.approx(140.0)
    row.set_start(None)
    assert row.start is None, "there must be a way back to the automatic answer"
    del app


def test_pinning_is_not_offered_anywhere():
    """A user-supplied component start is a hint, never a pin."""
    src = SRC["dist_fit_core.py"] + SRC["dist_fit_window.py"]
    assert "setFixed" not in src.replace("setFixedWidth", "")
    assert not re.search(r"\bpin(ned)?\s*=", src), (
        "fixing a parameter and fitting the rest needs its own discussion and "
        "its own label in the output, not a quiet flag here"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
