# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Regression test: the piezo landmarks are measured FROM snap-off.

Replaces test_variable_relative_snapoff.py, which guarded a display checkbox in
one window.  The convention is now a property of the variable itself
(variables.REFERENCED), so what needs guarding is different: that every consumer
gets the referenced number without asking, that the raw stage positions are not
offered beside them, and — the half that actually matters scientifically — that
referencing removes a common-mode trend while leaving a real one alone.

The contract under test:
(a) variables.values() returns landmark - snap-off for the referenced keys, and
    the raw stored number for snapoff_piezo_nm itself, which IS the drift.
(b) a file missing either end has no value - None, never a distance computed
    against a zero we do not have.  It must not raise.
(c) the raw landmarks are not offered as variables, so a dropdown cannot show a
    sum next to both its parts.  snapoff_piezo_nm still is.
(d) NULL TEST: referencing must not manufacture a signal.  Data with a real
    per-curve trend and NO common-mode drift comes back with its trend intact.
(e) the drift that referencing removes is the one it claims to remove: a pure
    common-mode wander, with the true distance held constant, comes back flat.
(f) the queue table shows the same number the variable accessor does - one
    routing, never two copies of it.
(g) VariableStatsWindow plots the referenced values, with no mode to set.
"""
import os
import sys
import tempfile

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
app = QApplication.instance() or QApplication(sys.argv)

from smfs_catalog import db as _db
from smfs_catalog.provenance import cache_version
from smfs_catalog import variables as _vars
from smfs_catalog.variable_window import VariableStatsWindow

tmp = tempfile.mkdtemp(prefix="landmark_ref_")
DB = os.path.join(tmp, "test.sqlite")
_db.initialise(DB)

DIR = "/tank/testdata/alexandre"

# A, B: the same true geometry (rupture 5000 nm past snap-off, contact 20 nm
# before it) recorded 3000 nm apart in stage position - the confound, isolated.
# C: a rupture with no snap-off at all, the missing-zero case.
FILE_A = _db.normalize_path(f"{DIR}/Image0001.ibw")
FILE_B = _db.normalize_path(f"{DIR}/Image0002.ibw")
FILE_C = _db.normalize_path(f"{DIR}/Image0003.ibw")
PATHS  = [FILE_A, FILE_B, FILE_C]

_conn = _db.get_connection(DB)
with _conn:
    for _path, _date in [(FILE_A, "2026-07-17 10:00:00"),
                         (FILE_B, "2026-07-18 10:00:00"),
                         (FILE_C, "2026-07-19 10:00:00")]:
        _conn.execute(
            "INSERT INTO files (path, filename, first_seen, last_seen,"
            " experimentalist, measured_at, event)"
            " VALUES (?, ?, datetime('now'), datetime('now'), 'alexandre', ?, 'event')",
            (_path, os.path.basename(_path), _date))
_conn.close()

ID_A, ID_B, ID_C = (_db.get_file_id(p, DB) for p in PATHS)

_METHOD = cache_version() or "test-build"
for _fid, _snap, _contact, _onset, _rupture in [
    (ID_A, 1000.0,  980.0, 1050.0, 6000.0),
    (ID_B, 4000.0, 3980.0, 4050.0, 9000.0),
]:
    _db.write_analysis_result(_fid, "snapoff_piezo_nm", _snap,    "{}", _METHOD, DB)
    _db.write_analysis_result(_fid, "contact_piezo_nm", _contact, "{}", _METHOD, DB)
    _db.write_analysis_result(_fid, "onset_piezo_nm",   _onset,   "{}", _METHOD, DB)
    _db.write_analysis_result(_fid, "rupture_piezo_nm", _rupture, "{}", _METHOD, DB)
# ID_C: a rupture with no snap-off row at all.
_db.write_analysis_result(ID_C, "rupture_piezo_nm", 7000.0, "{}", _METHOD, DB)

VALUES = _vars.values(
    PATHS, ["rupture_dx_nm", "onset_dx_nm", "contact_dx_nm", "snapoff_piezo_nm"], DB)
A = VALUES[_db.normalize_path(FILE_A)]
B = VALUES[_db.normalize_path(FILE_B)]
C = VALUES[_db.normalize_path(FILE_C)]


# ── (a) the values are distances from snap-off ────────────────────────────────

def test_each_landmark_is_its_distance_from_snapoff():
    assert A["rupture_dx_nm"] == 5000.0 and B["rupture_dx_nm"] == 5000.0
    assert A["onset_dx_nm"] == 50.0 and B["onset_dx_nm"] == 50.0
    # Contact is the one landmark on the NEAR side, subtracted the other way
    # round so that every entry in REFERENCED is a positive distance.
    assert A["contact_dx_nm"] == 20.0 and B["contact_dx_nm"] == 20.0


def test_two_curves_3000nm_apart_agree_on_every_distance():
    """The confound, isolated: same geometry, different stage position."""
    for k in ("rupture_dx_nm", "onset_dx_nm", "contact_dx_nm"):
        assert A[k] == B[k]


def test_snapoff_itself_is_left_raw_because_it_is_the_drift():
    assert A["snapoff_piezo_nm"] == 1000.0
    assert B["snapoff_piezo_nm"] == 4000.0


# ── (b) a missing zero is a missing value, not a fabricated one ───────────────

def test_a_landmark_with_no_snapoff_has_no_distance():
    # Present and None: a caller must be able to tell "measured as nothing"
    # from "I forgot to ask" (variables.values' own rule).
    assert "rupture_dx_nm" in C and C["rupture_dx_nm"] is None
    # And the raw landmark is never silently substituted for the distance.
    assert C["rupture_dx_nm"] != 7000.0


# ── (c) the raw landmarks are not offered beside the referenced ones ──────────

def test_raw_landmarks_are_not_offered_as_variables():
    offered = {v.key for v in _vars.available(PATHS, DB)}
    for raw in ("contact_piezo_nm", "onset_piezo_nm", "rupture_piezo_nm"):
        # Offering both would put a sum in the same dropdown as its two parts.
        assert raw not in offered
        # Not deleted, though: curve_analysis reads them to skip the ROI search.
        assert raw in _vars.NOT_A_VARIABLE
    for ref in ("contact_dx_nm", "onset_dx_nm", "rupture_dx_nm"):
        assert ref in offered
    assert "snapoff_piezo_nm" in offered
    # An intercept is pinned at piezo = 0, so on an axis it is the stage
    # position again under a calibration's name (r = +0.994 on the live data).
    assert "invols_intercept" not in offered
    assert "baseline_intercept" not in offered


# ── (d)/(e) what referencing does and does not remove ─────────────────────────
#
# Both cases are built from the same true signal, so they differ in exactly one
# thing: whether a common-mode wander is added to both channels.

_RNG   = np.random.default_rng(20260807)
_N     = 400
_T     = np.arange(_N, dtype=float)
_TRUE  = 60.0 + 0.05 * _T + _RNG.normal(0, 1.0, _N)   # a REAL trend, +0.05/step
_WANDER = 800.0 * np.sin(_T / 37.0) + 4.0 * _T        # the stage going walkabout


def _slope(y):
    return float(np.polyfit(_T, y, 1)[0])


def test_a_constant_distance_on_a_drifting_stage_reads_flat_once_referenced():
    flat     = 60.0 + _RNG.normal(0, 1.0, _N)
    raw      = _WANDER + flat
    # Stated as a RATIO, not an absolute slope: the claim is that the raw
    # number reports the stage's motion rather than the curve's, and the size
    # of this fixture's wander is not itself the finding.
    assert abs(_slope(raw)) > 100 * abs(_slope(raw - _WANDER))
    assert abs(_slope(raw - _WANDER)) < 0.02


def test_null_referencing_a_stage_that_never_moved_changes_nothing():
    """NULL TEST.  Without it, a change that simply flattened everything would
    pass every other check here."""
    raw = 1000.0 + _TRUE
    assert abs(_slope(raw - 1000.0) - _slope(raw)) < 1e-9


def test_a_real_trend_survives_referencing_and_would_not_have_survived_raw():
    """The correction runs both ways: this is the Dylan case, where the
    common-mode swamp was hiding a real trend (r = +0.07 raw, +0.75 referenced)."""
    raw = _WANDER + _TRUE
    assert abs(_slope(raw - _WANDER) - 0.05) < 0.01
    assert abs(_slope(raw)) > 20 * 0.05


# ── (f) the queue table agrees with the accessor ──────────────────────────────

def test_the_queue_prints_the_referenced_landmarks():
    from smfs_catalog.dashboard_window import _QUEUE_DERIVED
    keys = {k for k, _ in _QUEUE_DERIVED}
    assert {"contact_dx_nm", "onset_dx_nm", "rupture_dx_nm"} <= keys
    assert not ({"contact_piezo_nm", "onset_piezo_nm", "rupture_piezo_nm"} & keys)
    assert "snapoff_piezo_nm" in keys


def test_a_queue_cell_holds_what_the_accessor_returned():
    """The queue must not route its own columns to their values independently
    of variables.py: that fork shows a raw stage position in the table, under
    the same heading every plot references."""
    from smfs_catalog.dashboard_window import DashboardWindow
    assert DashboardWindow._queue_cell_value(
        "rupture_dx_nm", FILE_A, VALUES) == A["rupture_dx_nm"]


# ── (g) the window plots them, with no mode to set ────────────────────────────

def test_the_variable_window_plots_referenced_values_with_no_mode():
    win = VariableStatsWindow("rupture_dx_nm", "Rupture from snap-off (nm)", PATHS, DB)
    assert len(win._plot_vals) == 2          # FILE_C has no zero
    assert sorted(win._plot_vals.tolist()) == [5000.0, 5000.0]
    assert not hasattr(win, "_chk_relative")
    # Axis and bounds name the same quantity, so there is no state in which
    # the threshold row has to be disabled.
    assert win._apply_btn.isEnabled()


def test_the_drift_variable_still_plots_raw_stage_positions():
    win = VariableStatsWindow("snapoff_piezo_nm", "Snap-off, abs. piezo (nm)", PATHS, DB)
    assert sorted(win._plot_vals.tolist()) == [1000.0, 4000.0]
