# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Shared builders for the invOLS ensemble tests: temporary catalogs holding files
with instrument values, per-curve slopes and measured times.

Rows are written with plain SQL, not through the helpers under test, so a test
of those helpers does not stand on them.  Every function takes the catalog path
explicitly; nothing here ever touches the app's own catalog.

make_force_curve builds a synthetic ForceCurve: a noiseless linear approach
contact of a known slope and a retract with one rupture.
"""
import json
import os
import sqlite3

import numpy as np

from smfs_catalog import db

CODE = "C"
SLOPE_PARAMS = '{"invols_offset_pts": 0, "invols_window_pts": 100}'


def make_catalog(tmp, files):
    """Create a catalog under `tmp` holding one row per entry of `files`.

    Each entry is a dict; every key is optional:
      instrument   files.inv_ols_nm_v (nm/V), default None
      slope        a stored `invols_slope` row (nm/nm), default no row
      measured_at  files.measured_at text, default None
      event_params the params_json of a stored `event` row, default no row
      queued       whether the file is in the analysis queue, default True
      curve_type   files.curve_type text (e.g. "continuous_stretch"), default None
    Returns (db_path, file_ids), the ids in the order of `files`.
    """
    db_path = os.path.join(str(tmp), "synth_catalog.db")
    db.initialise(db_path)
    ids = []
    con = sqlite3.connect(db_path)
    try:
        with con:
            for number, spec in enumerate(files, start=1):
                ids.append(_add_file(con, number, spec))
    finally:
        con.close()
    return db_path, ids


def path_of(number):
    """The catalog path of the `number`-th file (1-based) of make_catalog."""
    return db.normalize_path(f"/synth/Image{number:04d}.ibw")


def _add_file(con, number, spec):
    cur = con.execute(
        "INSERT INTO files (path, filename, first_seen, last_seen, "
        "inv_ols_nm_v, measured_at) VALUES (?, ?, 'x', 'x', ?, ?)",
        (path_of(number), f"Image{number:04d}.ibw",
         spec.get("instrument"), spec.get("measured_at")))
    file_id = cur.lastrowid
    if spec.get("curve_type") is not None:
        con.execute("UPDATE files SET curve_type = ? WHERE id = ?",
                    (spec["curve_type"], file_id))
    if spec.get("slope") is not None:
        _add_result(con, file_id, "invols_slope", spec["slope"], SLOPE_PARAMS)
    if spec.get("event_params") is not None:
        _add_result(con, file_id, "event", 1.0, spec["event_params"])
    if spec.get("queued", True):
        con.execute("INSERT INTO analysis_queue (file_id, enqueued_at) "
                    "VALUES (?, 'x')", (file_id,))
    return file_id


def _add_result(con, file_id, analysis_type, value, params_json):
    con.execute(
        "INSERT INTO analysis_results (file_id, analysis_type, value, "
        "params_json, code_version, computed_at) VALUES (?, ?, ?, ?, ?, 'x')",
        (file_id, analysis_type, value, params_json, CODE))


def add_group(db_path, file_ids, transform_nm_v, average="arithmetic"):
    """Record an applied group: its row, its members, and one assignment per
    member (a file already assigned moves to this group, as a later apply
    does).  Returns the group id."""
    con = sqlite3.connect(db_path)
    try:
        with con:
            group_id = con.execute(
                "INSERT INTO invols_groups (created_at, method, n, average, "
                "transform_nm_v, fit_settings_json) VALUES ('x', 'all', ?, ?, "
                "?, '{}')", (len(file_ids), average, transform_nm_v)).lastrowid
            for file_id in file_ids:
                con.execute("INSERT INTO invols_group_members (group_id, "
                            "file_id) VALUES (?, ?)", (group_id, file_id))
                con.execute("INSERT OR REPLACE INTO invols_assignments "
                            "(file_id, group_id) VALUES (?, ?)",
                            (file_id, group_id))
    finally:
        con.close()
    return group_id


def current_fit_key(db_path):
    """The invOLS fit key for the fit settings now in force in `db_path`: the
    text a stored slope row carries when its fit was made under them."""
    from smfs_catalog import curve_analysis
    params = curve_analysis.pipeline_params_from(db.load_analysis_params(db_path))
    return params.params_invols


def stamp_fit_key(db_path, file_ids):
    """Rewrite the stored slope rows of `file_ids` under the fit key in force
    now, as if the queue had just re-run those files."""
    key = current_fit_key(db_path)
    con = sqlite3.connect(db_path)
    try:
        with con:
            for file_id in file_ids:
                con.execute("UPDATE analysis_results SET params_json = ? "
                            "WHERE file_id = ? AND analysis_type = "
                            "'invols_slope'", (key, file_id))
    finally:
        con.close()


def add_result(db_path, file_id, analysis_type, value, params_json="{}"):
    """Write one analysis_results row with plain SQL."""
    con = sqlite3.connect(db_path)
    try:
        with con:
            _add_result(con, file_id, analysis_type, value, params_json)
    finally:
        con.close()


def add_other_rows(db_path, file_id):
    """Give `file_id` an event_map row and an event_histograms row (plain SQL),
    the stored results an applied ensemble must leave alone."""
    con = sqlite3.connect(db_path)
    try:
        with con:
            con.execute("INSERT INTO event_map (file_id, params_json, "
                        "code_version, payload_json, computed_at) "
                        "VALUES (?, 'P', 'C', '[1, 2]', 'x')", (file_id,))
            con.execute("INSERT INTO event_histograms (file_id, histogram, "
                        "x_bins, f_bins, params_json, computed_at) "
                        "VALUES (?, ?, 4, 4, 'P', 'x')",
                        (file_id, b"\x01\x02\x03"))
    finally:
        con.close()


def put_result(db_path, file_id, analysis_type, value, params_json="{}"):
    """Write one analysis_results row with plain SQL, replacing any row the file
    already has of that type (the table keeps one per file and type)."""
    con = sqlite3.connect(db_path)
    try:
        with con:
            con.execute("DELETE FROM analysis_results WHERE file_id = ? AND "
                        "analysis_type = ?", (file_id, analysis_type))
            _add_result(con, file_id, analysis_type, value, params_json)
    finally:
        con.close()


def unassign(db_path, file_ids):
    """Take `file_ids` back to their own fit: delete their assignments."""
    con = sqlite3.connect(db_path)
    try:
        with con:
            for file_id in file_ids:
                con.execute("DELETE FROM invols_assignments WHERE file_id = ?",
                            (file_id,))
    finally:
        con.close()


def set_instrument(db_path, file_id, value):
    """Overwrite files.inv_ols_nm_v of one file (None, 0, a negative number...)."""
    con = sqlite3.connect(db_path)
    try:
        with con:
            con.execute("UPDATE files SET inv_ols_nm_v = ? WHERE id = ?",
                        (value, file_id))
    finally:
        con.close()


def event_map_doc(n_ruptures):
    """The stored text of an event_map document: one ROI with `n_ruptures`
    ruptures and as many segments."""
    from smfs_catalog import roi_events
    ruptures = [roi_events.Rupture(idx=10 + i, piezo_nm=5.0 + i, d1_height=1.0,
                                   force_pN=50.0, extension_nm=20.0 + i)
                for i in range(n_ruptures)]
    segments = [roi_events.Segment(left_idx=i, right_idx=10 + i,
                                   left_piezo_nm=float(i),
                                   right_piezo_nm=5.0 + i)
                for i in range(n_ruptures)]
    roi = roi_events.ROI(onset_idx=0, return_idx=50, onset_piezo_nm=0.0,
                         return_piezo_nm=9.0, ruptures=ruptures,
                         segments=segments)
    events = roi_events.CurveEvents(rois=[roi], detector="synthetic")
    return json.dumps(roi_events.events_to_payload(events))


def put_event_map(db_path, file_id, params_json, payload_json):
    """Make `payload_json` the file's one event_map row, under `params_json`."""
    con = sqlite3.connect(db_path)
    try:
        with con:
            con.execute("DELETE FROM event_map WHERE file_id = ?", (file_id,))
            con.execute("INSERT INTO event_map (file_id, params_json, "
                        "code_version, payload_json, computed_at) "
                        "VALUES (?, ?, ?, ?, 'x')",
                        (file_id, params_json, CODE, payload_json))
    finally:
        con.close()


def fetch(db_path, sql, params=()):
    """Every row `sql` returns, as plain tuples, read on a connection of its own."""
    con = sqlite3.connect(db_path)
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()


# ── a synthetic force curve ──────────────────────────────────────────────────

_POINTS = 3000          # per approach and per retract; 0.2 nm of piezo each
_TRAVEL_NM = 600.0
_CONTACT_NM = 200.0     # piezo position of the contact point
_PULL_START_NM = 320.0  # the tether pull begins here (past the post-snap-off mask)
_PULL_END_NM = 420.0    # ... and ruptures here (before the baseline region)
_PULL_DEPTH_NM = 80.0   # deflection at the rupture, towards the surface
_SPRING_CONSTANT = 10.0  # pN/nm


def make_force_curve(slope=-0.95, instrument=199.14, deep_slope=None,
                     deep_pts=200):
    """A ForceCurve with a known approach contact slope and one rupture.

    Approach: piezo falls from 600 nm to 0; free (noisy, seeded) down to the
    contact point at 200 nm, then a noiseless line of slope `slope` (nm/nm).
    With `deep_slope`, the last `deep_pts` approach samples follow that slope
    instead, continuous at the join, so a fit window's value depends on where
    it sits.  Retract: the contact line back out to the contact point, flat
    noisy baseline, one tether pull from 320 nm that ruptures at 420 nm.
    The instrument value travels on the curve as `inv_ols_nm_v`.
    """
    rng = np.random.default_rng(7)
    piezo_appr = np.linspace(_TRAVEL_NM, 0.0, _POINTS)
    piezo_retr = np.linspace(0.0, _TRAVEL_NM, _POINTS)
    defl_appr = np.where(piezo_appr < _CONTACT_NM,
                         slope * (piezo_appr - _CONTACT_NM), 0.0)
    if deep_slope is not None:
        join = piezo_appr[_POINTS - deep_pts]
        deep = slope * (join - _CONTACT_NM) + deep_slope * (piezo_appr - join)
        defl_appr = np.where(piezo_appr < join, deep, defl_appr)
    defl_retr = np.where(piezo_retr < _CONTACT_NM,
                         slope * (piezo_retr - _CONTACT_NM), 0.0)
    pull = (piezo_retr >= _PULL_START_NM) & (piezo_retr < _PULL_END_NM)
    ramp = (piezo_retr - _PULL_START_NM) / (_PULL_END_NM - _PULL_START_NM)
    defl_retr = np.where(pull, -_PULL_DEPTH_NM * ramp, defl_retr)
    defl_appr = defl_appr + np.where(piezo_appr >= _CONTACT_NM,
                                     rng.normal(0.0, 0.4, _POINTS), 0.0)
    defl_retr = defl_retr + np.where(piezo_retr >= _CONTACT_NM,
                                     rng.normal(0.0, 0.4, _POINTS), 0.0)
    from smfs_catalog.curve_loader import ForceCurve
    return ForceCurve(
        path="synthetic.ibw", piezo_appr=piezo_appr, defl_appr=defl_appr,
        piezo_retr=piezo_retr, defl_retr=defl_retr,
        spring_constant=_SPRING_CONSTANT, sample_rate_hz=20000.0,
        inv_ols_nm_v=instrument)
