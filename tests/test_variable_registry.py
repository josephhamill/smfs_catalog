# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Guards for smfs_catalog.variables — the one accessor answering "what per-file
variables exist, and give me some of them for this cohort" (#143).

The load-bearing test is (e): criteria_gate._values must still return exactly
what its own hand-written routing returned, because that routing decides which
curves are hits. Everything else here would pass for a rewrite that quietly
changed the gate.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from smfs_catalog import db as _db
from smfs_catalog import variables as _vars
from smfs_catalog import criteria_gate as _gate

DB = os.path.join(tempfile.mkdtemp(), "vars.db")
_db.initialise(DB)

FULL = _db.normalize_path("/tank/testdata/vars/full.ibw")      # every source populated
SPARSE = _db.normalize_path("/tank/testdata/vars/sparse.ibw")  # analysis result only
BARE = _db.normalize_path("/tank/testdata/vars/bare.ibw")      # nothing but the row
ALL    = [FULL, SPARSE, BARE]

_conn = _db.get_connection(DB)
with _conn:
    for path, k, at, date in (
        (FULL,   42.5, "2026-07-01 09:30:00", "2026-07-01"),
        (SPARSE, None, None,                  "2026-07-02"),
        (BARE,   None, None,                  None),
    ):
        _conn.execute(
            "INSERT INTO files (path, filename, first_seen,"
            " last_seen, event, experimentalist, spring_constant_pn_nm,"
            " measured_at, measured_date)"
            " VALUES (?,?,datetime('now'),datetime('now'),'event','sam',?,?,?)",
            (path, os.path.basename(path), k, at, date))
    for path, val in ((FULL, 1.25), (SPARSE, 2.5)):
        fid = _conn.execute("SELECT id FROM files WHERE path=?", (path,)).fetchone()[0]
        _conn.execute(
            "INSERT INTO analysis_results (file_id, analysis_type, value,"
            " params_json, code_version, computed_at)"
            " VALUES (?,'baseline_rms',?,'{}','test',datetime('now'))", (fid, val))
    # A real upgraded catalog can still contain these retired result rows.
    # Their presence, rather than their absence from a synthetic DB, is what
    # the exclusion contract needs to survive.
    fid = _conn.execute("SELECT id FROM files WHERE path=?", (FULL,)).fetchone()[0]
    for key in ("wlc_l_c_nm", "wlc_l_p_nm", "wlc_l_c_err", "wlc_l_p_err",
                "rupture_force_pn", "baseline_r2", "invols_r2"):
        _conn.execute(
            "INSERT INTO analysis_results (file_id, analysis_type, value,"
            " params_json, code_version, computed_at)"
            " VALUES (?,?,1.0,'{}','legacy',datetime('now'))", (fid, key))
_conn.close()

# Criteria use the active queue cohort's owner. Establish the cohort this file
# claims to test instead of relying on an implicit per-file profile.
_db.enqueue_files([_db.get_file_id(path, DB) for path in ALL], DB)


# ── (a) routing ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("key, source", [
    ("seg_l_c_nm",            _vars.SOURCE_SEGMENT),
    ("seg_edge_pinned",       _vars.SOURCE_SEGMENT),
    ("baseline_rms",          _vars.SOURCE_ANALYSIS),
    ("snapoff_piezo_nm",      _vars.SOURCE_ANALYSIS),
    ("spring_constant_pn_nm", _vars.SOURCE_FILE),
    (_vars.TIME_KEY,          _vars.SOURCE_FILE),
])
def test_every_key_routes_to_the_right_store(key, source):
    """seg_* CANNOT be served by the analysis_results query — it follows the
    live segment selection and any manual override — so a mis-route reads as
    'this curve has no value', silently, for the most interesting half of the
    catalog."""
    assert _vars.source_of(key) == source


def test_all_three_sources_are_offered():
    got = {v.source for v in _vars.available(ALL, DB)}
    assert got == {_vars.SOURCE_SEGMENT, _vars.SOURCE_ANALYSIS, _vars.SOURCE_FILE}


def test_nothing_in_not_a_variable_is_ever_offered():
    """The dead, the plumbing and the misleading, in one place.

    Written as a loop over the real set rather than as a handful of named
    keys, because the first attempt at this exclusion was a hand-written copy
    that invented two key names not in the database (`event_verdict`,
    `is_event`) and missed all fifteen real ones — including `event`, which
    has 132,453 rows. A test naming keys by hand could have made the same
    mistake and still passed.
    """
    keys = {v.key for v in _vars.available(ALL, DB)}
    assert not (keys & _vars.NOT_A_VARIABLE)


def test_the_dashboard_and_the_registry_share_one_exclusion_object():
    """Not "have equal contents" — the SAME object. Two lists that merely
    agree today are the fork this module exists to close;
    identity is the only version of this that cannot drift apart."""
    pytest.importorskip("PyQt6.QtWidgets")
    from smfs_catalog import dashboard_window as _dash
    assert _dash._QUEUE_HIDE is _vars.NOT_A_VARIABLE


def test_the_dead_fitter_a_columns_can_never_be_plotted_beside_the_live_ones():
    """wlc_l_c_nm is Fitter A's contour length: fit to the RAW retract
    (sawtooth noise), one event per curve, on a cache key of frozen literals
    that never invalidated (#80). It was deleted and its rows are
    frozen at that date. seg_l_c_nm is the SAME physical quantity measured
    correctly. Offering both puts a corpse on an axis beside the living thing
    under near-identical names, which is worse than offering neither."""
    keys = {v.key for v in _vars.available(ALL, DB)}
    for dead in ("wlc_l_c_nm", "wlc_l_p_nm", "wlc_l_c_err", "wlc_l_p_err",
                 "rupture_force_pn"):
        assert dead not in keys
    assert "seg_l_c_nm" in keys and "seg_l_p_nm" in keys and "seg_force_pN" in keys


def test_units_come_from_quantities_never_declared_here():
    from smfs_catalog import quantities as _q
    for v in _vars.available(ALL, DB):
        assert v.unit == _q.unit_of(v.key)


def test_every_offered_variable_is_an_explicit_quantity():
    """GENERIC keeps an unknown value renderable, but an offered variable is
    part of the public UI and must not silently acquire an unknown/blank unit."""
    from smfs_catalog import quantities as _q
    missing = [v.key for v in _vars.available(ALL, DB)
               if v.key not in _q.QUANTITIES]
    assert not missing


# ── (b) missing means present-and-None, never absent ─────────────────────────

def test_a_missing_value_is_none_not_a_missing_key():
    """A caller must be able to tell 'measured as nothing' from 'I forgot to
    ask'. An absent key silently reads as the first."""
    got = _vars.values(ALL, ["baseline_rms", "spring_constant_pn_nm"], DB)
    for path in ALL:
        rp = _db.normalize_path(path)
        assert set(got[rp]) == {"baseline_rms", "spring_constant_pn_nm"}
    assert got[_db.normalize_path(FULL)]["baseline_rms"] == 1.25
    assert got[_db.normalize_path(BARE)]["baseline_rms"] is None
    assert got[_db.normalize_path(SPARSE)]["spring_constant_pn_nm"] is None


def test_columns_are_aligned_with_the_paths_and_nan_where_missing():
    """Aligned by construction is the point: masking per-array rather than
    per-pair slides one axis against the other and confidently correlates the
    wrong curves together."""
    order, cols = _vars.columns(ALL, ["baseline_rms", "spring_constant_pn_nm"], DB)
    assert order == [_db.normalize_path(p) for p in ALL]
    assert cols["baseline_rms"][0] == 1.25
    assert np.isnan(cols["baseline_rms"][2])
    assert cols["spring_constant_pn_nm"][0] == 42.5
    assert np.isnan(cols["spring_constant_pn_nm"][1])
    assert all(a.shape == (len(ALL),) for a in cols.values())


# ── (c) acquisition time ─────────────────────────────────────────────────────

def test_time_prefers_the_full_timestamp_and_falls_back_to_the_date():
    """measured_at is second-resolution; measured_date is the day-only
    fallback for files not yet re-scanned with time capture. Preferring the
    wrong one silently quantises every drift fit to 24-hour steps."""
    got = _vars.values(ALL, [_vars.TIME_KEY], DB)
    full   = got[_db.normalize_path(FULL)][_vars.TIME_KEY]
    sparse = got[_db.normalize_path(SPARSE)][_vars.TIME_KEY]
    bare   = got[_db.normalize_path(BARE)][_vars.TIME_KEY]
    import datetime as _dt
    assert full == _dt.datetime(2026, 7, 1, 9, 30, 0).timestamp()
    assert sparse == _dt.datetime(2026, 7, 2).timestamp()
    assert np.isnan(bare), "no date at all must be NaN, never epoch zero"


def test_time_is_an_ordinary_variable_so_drift_is_a_special_case():
    """#13 is #143 with x = time. If time ever stops being offered, the drift
    fit becomes a second implementation of the same arithmetic."""
    keys = {v.key for v in _vars.available(ALL, DB)}
    assert _vars.TIME_KEY in keys
    assert next(v for v in _vars.available(ALL, DB) if v.key == _vars.TIME_KEY).is_time


# ── (d) get_file_columns refuses what it does not know ───────────────────────

def test_unknown_file_columns_raise_rather_than_read_as_missing():
    """These names reach an f-string in the SELECT. Beyond injection, a
    silently dropped column reads downstream as 'this file has no value',
    which is a quieter and worse kind of wrong."""
    with pytest.raises(ValueError):
        _db.get_file_columns([FULL], ["spring_constant_pn_nm", "nope"], DB)
    with pytest.raises(ValueError):
        _db.get_file_columns([FULL], ["1); DROP TABLE files;--"], DB)


def test_get_file_columns_is_bulk_and_skips_unknown_paths():
    got = _db.get_file_columns(ALL + ["/tank/nope.ibw"],
                               ["spring_constant_pn_nm"], DB)
    assert set(got) == {_db.normalize_path(p) for p in ALL}
    assert _db.get_file_columns([], ["spring_constant_pn_nm"], DB) == {}
    assert _db.get_file_columns(ALL, [], DB) == {}


# ── (e) THE ONE THAT MATTERS: the gate did not change ────────────────────────

def test_the_gate_reads_exactly_what_its_own_routing_used_to_read():
    """criteria_gate._values now delegates to variables.values. It decides
    which curves are hits, so 'equivalent' has to be checked, not assumed.

    This reproduces the pre-delegation implementation inline and compares.
    """
    from smfs_catalog.roi_pipeline import (
        SEG_SUMMARY_KEYS, SEG_SUMMARY_FIELD, read_segment_select,
        segment_summary_bulk,
    )

    def _old_values(event_paths, checked, db_path):
        plain = [k for k in checked if k not in SEG_SUMMARY_KEYS]
        segk  = [k for k in checked if k in SEG_SUMMARY_KEYS]
        out = {_db.normalize_path(p): {} for p in event_paths}
        if plain:
            derived = _db.get_derived_results_bulk_latest(event_paths, plain, db_path)
            for rp in out:
                d = derived.get(rp, {})
                for k in plain:
                    e = d.get(k)
                    out[rp][k] = e[0] if e is not None else None
        if segk:
            seg = segment_summary_bulk(event_paths, read_segment_select(db_path), db_path)
            for rp in out:
                d = seg.get(rp, {})
                for k in segk:
                    out[rp][k] = d.get(SEG_SUMMARY_FIELD[k])
        return out

    checked = ["baseline_rms", "seg_l_c_nm", "seg_force_pN", "seg_n_segments"]
    assert _gate._values(ALL, checked, DB) == _old_values(ALL, checked, DB)


def test_the_gate_still_produces_the_same_hit_split():
    """The end-to-end version of the above — the answer the app actually
    acts on."""
    _gate.set_criterion("baseline_rms", True, "sam", DB)
    _db.set_threshold("baseline_rms", 1.0, 2.0, "Baseline RMS", "sam", DB)
    hits, non_hits = _gate.evaluate(ALL, DB)
    assert _db.normalize_path(FULL) in hits           # 1.25, in bounds
    assert _db.normalize_path(SPARSE) in non_hits     # 2.5, out of bounds
    assert _db.normalize_path(BARE) in non_hits       # missing -> non-hit
