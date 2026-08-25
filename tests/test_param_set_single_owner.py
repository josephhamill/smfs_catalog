# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
ONE parameter set, fetched in ONE read, from the ONE place that decides whose.

The failure this pins down with the app open:

  * The analysis queue held Dylan's files; an ROI window was open on one of
    Anastasiia's curves.
  * display_roi took its nine on-screen knobs from the curve's owner and its
    other seven parameters (baseline anchor, cutoff, turnaround trim, var
    window, snap-off threshold, invOLS window) through db.get_param, which
    resolves via the queue.
  * The resulting parameter set was half of each — a combination neither
    person had chosen — and it decided where the search band went.  The ROI
    window said "no ROI found" while the raw-curve window, running entirely
    on the queue's set, drew two ROIs on the same curve at the same moment.

Cause: 8473c92 removed the settings-table mirror (correctly) and repointed
every read at the queue, without noticing that a window whose knobs were
loaded from a different source now disagreed with it.

The rule, as the user states it: **nobody owns the queue; the file
at position one of it decides which parameter set is used** — for the batch
worker and for every window alike.  Clear the queue, add, delete, restart:
recheck.  Otherwise one answer.  There is no per-curve resolution.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smfs_catalog import db as _db
from smfs_catalog.curve_analysis import pipeline_params_from
from smfs_catalog.roi_pipeline import event_params_from


# Two deliberately disjoint sets, so any mixing is visible in the values.
ANA = {
    "baseline_anchor_nm": 40.0, "spectral_cutoff_hz": 2000.0,
    "turnaround_trim_pts": 5000.0, "var_window_ms": 5.0,
    "detection_threshold_appr": 0.0182, "detection_threshold_retr": 0.0181,
    "invols_offset_pts": 400.0, "invols_window_pts": 400.0,
    "roi_window_pts": 31.0, "roi_threshold_nm_per_nm": 4.1728,
    "roi_inner_threshold_nm_per_nm": 2.2916,
    "roi_post_snapoff_mask_nm": 20.0, "roi_onset_threshold_nm": -0.2,
    "roi_detector_mode_idx": 1.0, "roi_prominence": 1.0,
    "roi_min_distance_pts": 6.0,
}
DYL = {
    "baseline_anchor_nm": 20.0, "spectral_cutoff_hz": 1500.0,
    "turnaround_trim_pts": 1596.0, "var_window_ms": 3.0,
    "detection_threshold_appr": 0.0079, "detection_threshold_retr": 0.0075,
    "invols_offset_pts": 180.0, "invols_window_pts": 180.0,
    "roi_window_pts": 61.0, "roi_threshold_nm_per_nm": 0.6907,
    "roi_inner_threshold_nm_per_nm": 0.4907,
    "roi_post_snapoff_mask_nm": 15.0, "roi_onset_threshold_nm": 0.0,
    "roi_detector_mode_idx": 2.0, "roi_prominence": 0.5,
    "roi_min_distance_pts": 12.0,
}
ANA_PATH = _db.normalize_path("/x/ana.ibw")
DYL_PATH = _db.normalize_path("/x/dyl.ibw")


@pytest.fixture()
def dbp(tmp_path):
    """Two experimentalists, one file each, ANASTASIIA at queue position one."""
    p = str(tmp_path / "t.db")
    _db.initialise(p)
    _db.merge_experimentalist_profile("Anastasiia", ANA, p)
    _db.merge_experimentalist_profile("Dylan", DYL, p)
    conn = _db.get_connection(p)
    with conn:
        for path, who in ((ANA_PATH, "Anastasiia"), (DYL_PATH, "Dylan")):
            conn.execute(
                "INSERT INTO files (path, filename, experimentalist, curve_type,"
                " first_seen, last_seen) VALUES (?,?,?,'force_extension','t','t')",
                (path, os.path.basename(path), who))
    conn.close()
    _db.enqueue_files([_db.get_file_id(ANA_PATH, p)], p)
    _db.enqueue_files([_db.get_file_id(DYL_PATH, p)], p)   # bottom of queue
    assert _db.active_param_owner(p) == "Anastasiia"
    return p


def _numbers(obj) -> set:
    d = obj if isinstance(obj, dict) else obj.__dict__
    return {float(v) for v in d.values()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}


def test_row_one_decides_not_the_last_added(dbp):
    """Adding to the bottom must not move the answer."""
    assert _db.active_param_owner(dbp) == "Anastasiia"
    _db.clear_analysis_queue(dbp)
    _db.enqueue_files([_db.get_file_id(DYL_PATH, dbp)], dbp)
    assert _db.active_param_owner(dbp) == "Dylan"


def test_analysis_snapshot_is_complete_and_one_owner(dbp):
    got = _db.load_analysis_params(dbp)
    assert set(got) == set(_db.PARAM_DEFAULTS), "must return EVERY parameter"
    for key, want in ANA.items():
        assert got[key] == want, f"{key} did not come from the row-one owner"
    only_dylans = set(DYL.values()) - set(ANA.values())
    assert not (only_dylans & set(got.values())), "another profile leaked in"


def test_event_projection_is_one_owner(dbp):
    ep = event_params_from(_db.load_analysis_params(dbp))
    assert ep.anchor_nm == 40.0 and ep.trim_pts == 5000
    assert ep.d1_threshold == ANA["roi_threshold_nm_per_nm"]
    assert ep.invols_offset_pts == 400
    leaked = (set(DYL.values()) - set(ANA.values())) & _numbers(ep)
    assert not leaked, f"values from a second profile leaked in: {leaked}"


def test_pipeline_projection_is_one_owner(dbp):
    p = pipeline_params_from(_db.load_analysis_params(dbp))
    assert p.anchor_nm == 40.0 and p.trim_pts == 5000 and p.roi_win_pts == 31
    nums = {
        float(v) for v in json.loads(p.all_params).values()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    leaked = (set(DYL.values()) - set(ANA.values())) & nums
    assert not leaked, f"a second profile leaked into the cache key: {leaked}"


def test_nothing_asks_a_different_question():
    """There must be exactly one way to decide whose set is in force.

    No module may resolve parameters from the curve on screen — that was the
    rejected second rule.  db.active_param_owner is the only answer, reached
    through db.load_analysis_params.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent / "smfs_catalog"
    for f in sorted(root.glob("*.py")):
        src = f.read_text()
        assert "get_param_for(" not in src, (
            f"{f.name} resolves parameters for a named person — second rule")
        assert "owner_of_file(" not in src, (
            f"{f.name} resolves parameters from the curve — second rule")


def test_signature_changes_when_a_parameter_changes(dbp):
    """A stored result is reused only on a signature match, so every parameter
    the finder reads must move the signature."""
    from dataclasses import replace
    from smfs_catalog.roi_pipeline import event_map_params_json

    base = event_params_from(_db.load_analysis_params(dbp))
    base_sig = event_map_params_json(base)
    for field, bump in (("anchor_nm", 99.0), ("d1_threshold", 9.9),
                        ("trim_pts", 7777), ("invols_window_pts", 999)):
        assert event_map_params_json(replace(base, **{field: bump})) != base_sig, (
            f"changing {field} did not change the stored signature")
