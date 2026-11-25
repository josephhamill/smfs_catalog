# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Pass-1 step d2: the HistGrid parameter object replaces the loose
(x_bins, f_bins, x_range, f_range) arguments of the 2DH calculators, the two
cache-key builders and the grid dialogs.

Counting: argument counts below are the plan's (they leave `self` out, as
inspect.signature does for a class).  The size guard uses tests/size_limits,
which counts `self`, so `_GridDialog.__init__` (parent, grid, align_segment,
defaults) is 5 there.

HistGrid is frozen, like PR (c)'s parameter objects (CacheContext etc.), and
its ranges are tuples.

Three layers:
  * interface tests: HistGrid, the planned signatures, the deleted function;
  * byte-identity and behaviour pins: the key strings are literals recorded
    from the code before d2, and the histograms are small hand-checked arrays;
    they go through the small adapter below, the only place that spells out
    how the HistGrid is built from the loose values;
  * a size guard on the changed functions.
"""
import dataclasses
import inspect
from pathlib import Path

import numpy as np
import pytest

from smfs_catalog import base_2dh_window, event_processor as ep, physical_2dh_window
from size_limits import violations
from source_inspect import function_node

PKG = Path(ep.__file__).resolve().parent


# ── adapter: takes loose values, builds the HistGrid, calls the d2 API ───────

def _grid(spec):
    """spec is the loose (x_bins, f_bins, x_range, f_range) tuple."""
    return ep.HistGrid(*spec)


def _phys_key(f_star, spec, align_mode, align_segment):
    return ep._phys_grid_params(f_star, _grid(spec), align_mode, align_segment)


def _wlc_key(spec, align_segment):
    return ep._wlc_grid_params(_grid(spec), align_segment)


def _phys_hist(x, F, anchor, spec):
    return ep.compute_physical_histogram_at(x, F, anchor, _grid(spec))


def _wlc_hist(x, F, l_p, l_c, spec):
    return ep.compute_wlc_histogram(x, F, l_p, l_c, _grid(spec))


# ── interface: HistGrid ──────────────────────────────────────────────────────

def test_hist_grid_is_a_frozen_dataclass_with_the_planned_fields():
    assert dataclasses.is_dataclass(ep.HistGrid)
    assert [f.name for f in dataclasses.fields(ep.HistGrid)] == [
        "x_bins", "f_bins", "x_range", "f_range"]
    g = ep.HistGrid(200, 150, (-20, 80), (0, 300))
    assert (g.x_bins, g.f_bins, g.x_range, g.f_range) == (
        200, 150, (-20, 80), (0, 300))
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.x_bins = 1


def test_hist_grid_equal_grids_compare_equal():
    assert ep.HistGrid(2, 3, (0, 1), (0, 2)) == ep.HistGrid(2, 3, (0, 1), (0, 2))
    assert ep.HistGrid(2, 3, (0, 1), (0, 2)) != ep.HistGrid(2, 3, (0, 1), (0, 3))


# ── interface: planned signatures ────────────────────────────────────────────

def _names(obj):
    return list(inspect.signature(obj).parameters)


def test_compute_physical_histogram_at_takes_x_F_anchor_grid():
    assert _names(ep.compute_physical_histogram_at) == ["x", "F", "anchor", "grid"]


def test_compute_wlc_histogram_takes_x_F_lp_lc_grid():
    assert _names(ep.compute_wlc_histogram) == ["x", "F", "l_p", "l_c", "grid"]


def test_phys_grid_params_takes_fstar_grid_mode_segment():
    assert _names(ep._phys_grid_params) == [
        "F_star", "grid", "align_mode", "align_segment"]


def test_wlc_grid_params_takes_grid_segment():
    assert _names(ep._wlc_grid_params) == ["grid", "align_segment"]


def test_grid_dialog_takes_parent_grid_segment_defaults():
    assert _names(base_2dh_window._GridDialog) == [
        "parent", "grid", "align_segment", "defaults"]


def test_physical_grid_dialog_takes_six_arguments():
    # Listed under D5: parent, grid, align_segment, f_star, align_mode, defaults.
    assert _names(physical_2dh_window._PhysicalGridDialog) == [
        "parent", "grid", "align_segment", "f_star", "align_mode", "defaults"]


@pytest.mark.parametrize("fn", [
    ep.compute_physical_histogram_at, ep.compute_wlc_histogram,
    ep._phys_grid_params, ep._wlc_grid_params])
@pytest.mark.parametrize("old", ["x_bins", "f_bins", "x_range", "f_range"])
def test_loose_grid_keywords_are_gone(fn, old):
    assert old not in inspect.signature(fn).parameters


def test_unused_compute_physical_histogram_is_deleted():
    assert not hasattr(ep, "compute_physical_histogram")
    # and nothing in the package or scripts still names it
    for root in (PKG, PKG.parent / "scripts"):
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert "compute_physical_histogram(" not in text, path.name


# ── byte-identical cache keys (recorded from the pre-d2 code) ────────────────
# Ints stay ints and floats stay floats in the strings, so HistGrid must not
# coerce its fields.

PHYS_KEYS = [
    ((50.0, (200, 150, (-20, 80), (0, 300)), "rupture", "last"),
     '{"type":"phys","x_bins":200,"f_bins":150,"x_min":-20,"x_max":80,'
     '"f_min":0,"f_max":300,"F_star_pN":50.0,"align":"rupture","seg":"last","v":4}'),
    ((50.0, (200, 150, (-20, 80), (0, 300)), "fstar", "first"),
     '{"type":"phys","x_bins":200,"f_bins":150,"x_min":-20,"x_max":80,'
     '"f_min":0,"f_max":300,"F_star_pN":50.0,"align":"fstar","seg":"first","v":4}'),
    ((12.5, (64, 32, (-10.5, 40.25), (-5, 120.125)), "onset", "penult"),
     '{"type":"phys","x_bins":64,"f_bins":32,"x_min":-10.5,"x_max":40.25,'
     '"f_min":-5,"f_max":120.125,"F_star_pN":12.5,"align":"onset","seg":"penult","v":4}'),
    ((50.001, (10, 20, (0, 1), (0, 2)), "lc", "last"),
     '{"type":"phys","x_bins":10,"f_bins":20,"x_min":0,"x_max":1,'
     '"f_min":0,"f_max":2,"F_star_pN":50.001,"align":"lc","seg":"last","v":4}'),
]

WLC_KEYS = [
    (((100, 80, (-0.1, 1.2), (-5.0, 45.0)), "last"),
     '{"x_bins":100,"f_bins":80,"x_min":-0.1,"x_max":1.2,'
     '"f_min":-5.0,"f_max":45.0,"seg":"last","v":6}'),
    (((64, 32, (-0.1001, 1.25), (-5, 45.001)), "first"),
     '{"x_bins":64,"f_bins":32,"x_min":-0.1001,"x_max":1.25,'
     '"f_min":-5,"f_max":45.001,"seg":"first","v":6}'),
    (((10, 20, (0, 1), (0, 2)), "penult"),
     '{"x_bins":10,"f_bins":20,"x_min":0,"x_max":1,'
     '"f_min":0,"f_max":2,"seg":"penult","v":6}'),
]


@pytest.mark.parametrize("args, expected", PHYS_KEYS)
def test_phys_grid_params_string_is_byte_identical(args, expected):
    assert _phys_key(*args) == expected


@pytest.mark.parametrize("args, expected", WLC_KEYS)
def test_wlc_grid_params_string_is_byte_identical(args, expected):
    assert _wlc_key(*args) == expected


def test_phys_worked_example_hist_grid_call_gives_the_old_call_string():
    """The plan's worked example.  The old keyword call
    _phys_grid_params(50.0, x_bins=200, f_bins=150, x_range=(-20, 80),
    f_range=(0, 300)) returned this exact string (recorded before d2)."""
    new = ep._phys_grid_params(
        50.0, ep.HistGrid(200, 150, (-20, 80), (0, 300)),
        ep.PHYS_ALIGN_DEFAULT, ep.ALIGN_SEG_DEFAULT)
    assert new == PHYS_KEYS[0][1]


# ── behaviour pins for the histograms (same arrays as before d2) ─────────────

def test_physical_histogram_pin_counts_and_dtype():
    H = _phys_hist(np.array([0.0, 1.0, 2.0, 3.0]), np.array([1.0, 1.0, 2.0, 2.0]),
                   1.0, (2, 2, (-1.0, 3.0), (0.0, 4.0)))
    np.testing.assert_array_equal(H, [[2, 0], [0, 2]])
    assert H.dtype == np.uint32


def test_physical_histogram_pin_none_anchor_drops_the_curve():
    assert _phys_hist(np.array([1.0, 2.0]), np.array([1.0, 2.0]),
                      None, (2, 2, (0.0, 2.0), (0.0, 2.0))) is None


def test_physical_histogram_pin_zero_lp_anchor_path_is_none():
    # b7: an l_p == 0 fit gives no F* anchor; that None drops the curve here.
    from smfs_catalog.models import ChainFit
    x = np.linspace(1.0, 40.0, 30)
    F = np.linspace(1.0, 30.0, 30)
    anchor = ep.PHYS_ALIGN_ANCHORS["fstar"](x, F, ChainFit("wlc", (0.0, 50.0)), 5.0)
    assert _phys_hist(x, F, anchor, (4, 4, (-50.0, 50.0), (0.0, 40.0))) is None


def test_wlc_histogram_pin_counts_and_dtype():
    from smfs_catalog.models import _k_B, _TEMPERATURE
    kT = _k_B * _TEMPERATURE
    H = _wlc_hist(np.array([5.0, 1.0]), np.array([kT / 2.0, kT / 2.0]),
                  2.0, 10.0, (2, 2, (0.0, 1.0), (0.0, 2.0)))
    np.testing.assert_array_equal(H, [[0, 1], [0, 1]])
    assert H.dtype == np.uint32


# ── size guard (CLAUDE.md limits) on what d2 changes ─────────────────────────

@pytest.mark.parametrize("name", [
    "compute_physical_histogram_at", "compute_wlc_histogram",
    "_phys_grid_params", "_wlc_grid_params"])
def test_changed_calculator_is_within_the_size_limits(name):
    assert violations(name, function_node(ep.__file__, name)) == []


def test_grid_dialog_init_is_within_the_size_limits():
    # Today 63 lines: d2 must bring it to 50 (split out helpers).
    assert violations("_GridDialog.__init__",
                      function_node(base_2dh_window.__file__, "__init__",
                                    "_GridDialog")) == []


def test_physical_grid_dialog_init_is_within_the_other_limits():
    # Excluded: the argument limit (6 by the plan's count, D5).  Lines,
    # decision points and nesting are still checked.
    found = violations("_PhysicalGridDialog.__init__",
                       function_node(physical_2dh_window.__file__, "__init__",
                                     "_PhysicalGridDialog"))
    assert [v for v in found if "arguments" not in v] == []


# ── the window builds its HistGrid from its own attributes ───────────────────

def test_window_hist_grid_is_built_from_the_window_attributes_in_order():
    from types import SimpleNamespace
    stand_in = SimpleNamespace(
        _x_bins=7, _f_bins=9, _x_min=-1.5, _x_max=2.5, _f_min=-3.0, _f_max=8.0)
    grid = base_2dh_window._TwoDHWindowBase._hist_grid.fget(stand_in)
    assert grid == ep.HistGrid(7, 9, (-1.5, 2.5), (-3.0, 8.0))
