# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
The invOLS store: the trace table, applying an ensemble value, reverting to the
own fit, the average option, the source labels and the curve label.

Fixture (the plan's): four files, instrument 199.14 nm/V each, slopes -0.95,
-1.00, -1.05 and none for file 4, acquired an hour apart.  Expected values are
hand calculations (199.14 / 0.95 = 209.62105..., 199.14 / 1.05 = 189.65714...,
the three-value mean 199.47273 and so on), never numbers read back from the
code.  Every call passes an explicit temporary catalog.
"""
import datetime
import json
import os
import re
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import invols_synth as synth                                 # noqa: E402
from smfs_catalog import db, variables                       # noqa: E402
from smfs_catalog import invols_store as store               # noqa: E402

INSTRUMENT = 199.14
OWN = {1: 209.62105, 2: 199.14, 3: 189.65714}      # 199.14 / |slope|
MEAN_3 = 199.47273
TRANSFORM = "transform_invols_nm_v"
UNUSABLE = [None, 0.0, -5.0, float("inf")]


@pytest.fixture(autouse=True)
def _pinned_code_version(monkeypatch):
    """The code identity is pinned, so the file does not depend on a git
    checkout (a built or exported copy has none)."""
    from smfs_catalog import provenance
    monkeypatch.setattr(provenance, "cache_version", lambda: "test-build")
    monkeypatch.setattr(store, "cache_version", lambda: "test-build",
                        raising=False)


def _no_code_identity(monkeypatch):
    from smfs_catalog import provenance
    monkeypatch.setattr(provenance, "cache_version", lambda: None)
    monkeypatch.setattr(store, "cache_version", lambda: None, raising=False)


def _hour(n):
    """Unix seconds of 2026-03-18 `n`:00 local time (the fixture's times)."""
    return datetime.datetime(2026, 3, 18, n).timestamp()


def _stamp(n):
    return f"2026-03-18 {n:02d}:00:00"


def _standard(tmp_path, slopes=(-0.95, -1.0, -1.05, None), **extra):
    """The four-file fixture; slope rows are stamped as made under the fit
    settings in force now."""
    files = [{"instrument": INSTRUMENT, "slope": s,
              "measured_at": _stamp(10 + i), **extra}
             for i, s in enumerate(slopes)]
    db_path, ids = synth.make_catalog(tmp_path, files)
    synth.stamp_fit_key(db_path, ids)
    return db_path, ids


def _request(db_path, members=(1, 2, 3), targets=(1, 2, 3, 4), **opts):
    """An EnsembleRequest; `opts` may give method, box and average."""
    o = {"method": "all", "box": None, "average": "arithmetic", **opts}
    assert set(o) == {"method", "box", "average"}, f"unknown options {opts}"
    return store.EnsembleRequest(tuple(members), tuple(targets), o["method"],
                                 o["box"], store.fit_settings_now(db_path),
                                 o["average"])


def _rows(db_path, table, order="1"):
    return synth.fetch(db_path, f"SELECT * FROM {table} ORDER BY {order}")


def _transform_rows(db_path):
    return synth.fetch(db_path, "SELECT file_id, value, params_json "
                       "FROM analysis_results WHERE analysis_type = ? "
                       "ORDER BY file_id", (TRANSFORM,))


def _state(db_path):
    """Everything an apply may write, as plain rows."""
    return {t: _rows(db_path, t) for t in
            ("invols_groups", "invols_group_members", "invols_assignments")} | {
        TRANSFORM: _transform_rows(db_path)}


def _paths(n=4):
    return [synth.path_of(i) for i in range(1, n + 1)]


# ── The trace table ────────────────────────────────────────────────────────────

def test_trace_rows_are_in_time_order_with_own_values(tmp_path):
    db_path, ids = _standard(tmp_path)
    synth.add_result(db_path, ids[0], "invols_rms", 0.1948)
    rows = store.trace_rows(list(reversed(_paths())), db_path)
    assert [r.file_id for r in rows] == ids
    assert [r.path for r in rows] == _paths()
    assert [r.time_s for r in rows] == [_hour(10), _hour(11), _hour(12), _hour(13)]
    assert all(r.instrument_nm_v == INSTRUMENT for r in rows)
    for row, expected in zip(rows[:3], OWN.values()):
        assert row.own_nm_v == pytest.approx(expected, rel=1e-6)
    assert rows[3].own_nm_v is None
    assert [r.source for r in rows] == ["own"] * 4
    assert [r.group_id for r in rows] == [None] * 4
    assert [r.fit_current for r in rows[:3]] == [True] * 3
    assert rows[3].fit_current is False
    assert rows[0].rms_nm == pytest.approx(0.1948)
    assert rows[1].rms_nm is None


def test_trace_rows_order_by_time_then_path(tmp_path):
    files = [{"instrument": INSTRUMENT, "measured_at": _stamp(h)}
             for h in (12, 10, 11, 11)]
    db_path, ids = synth.make_catalog(tmp_path, files)
    rows = store.trace_rows(list(reversed(_paths())), db_path)
    assert [r.file_id for r in rows] == [ids[1], ids[2], ids[3], ids[0]]


def test_applied_ensemble_shows_in_the_trace_rows(tmp_path):
    db_path, ids = _standard(tmp_path)
    result = store.apply_ensemble(_request(db_path), db_path)
    rows = store.trace_rows(_paths(), db_path)
    assert [r.source for r in rows] == ["ensemble"] * 4
    assert [r.group_id for r in rows] == [result.group_id] * 4
    for r in rows:
        assert r.in_use_nm_v == pytest.approx(MEAN_3, rel=1e-6)
    for row, expected in zip(rows[:3], OWN.values()):      # the fit is untouched
        assert row.own_nm_v == pytest.approx(expected, rel=1e-6)


# ── Old fit settings (research NM-16) ──────────────────────────────────────────

def test_changed_fit_settings_make_the_stored_fits_not_current(tmp_path):
    db_path, _ = _standard(tmp_path)
    db.update_analysis_param("invols_window_pts", 150, db_path)
    rows = store.trace_rows(_paths(), db_path)
    assert [r.fit_current for r in rows] == [False] * 4
    for row, expected in zip(rows[:3], OWN.values()):
        assert row.own_nm_v == pytest.approx(expected, rel=1e-6)


def test_apply_with_no_current_member_raises_and_writes_nothing(tmp_path):
    db_path, _ = _standard(tmp_path)
    db.update_analysis_param("invols_window_pts", 150, db_path)
    before = _state(db_path)
    with pytest.raises(store.InvolsApplyError):
        store.apply_ensemble(_request(db_path), db_path)
    assert _state(db_path) == before
    assert before["invols_groups"] == []


def test_one_rerun_member_applies_alone_and_the_others_are_reported(tmp_path):
    db_path, ids = _standard(tmp_path)
    db.update_analysis_param("invols_window_pts", 150, db_path)
    synth.stamp_fit_key(db_path, [ids[0]])              # only file 1 re-run
    result = store.apply_ensemble(_request(db_path), db_path)
    assert result.summary.n == 1
    assert result.left_out_old_settings == (2, 3)
    (n, mean), = synth.fetch(db_path, "SELECT n, transform_nm_v FROM invols_groups")
    assert n == 1
    assert mean == pytest.approx(OWN[1], rel=1e-6)
    (settings_json,), = synth.fetch(db_path, "SELECT fit_settings_json FROM invols_groups")
    assert json.loads(settings_json)["invols_window_pts"] == 150
    assert store.stale_setting_groups(_paths(), db_path) == []


# ── Applying ───────────────────────────────────────────────────────────────────

def test_apply_arithmetic_writes_group_members_assignments_and_values(tmp_path):
    db_path, ids = _standard(tmp_path)
    result = store.apply_ensemble(_request(db_path), db_path)
    (gid, n, average, value, sd, se, method, box), = synth.fetch(
        db_path, "SELECT id, n, average, transform_nm_v, sd_nm_v, se_nm_v, "
        "method, box_json FROM invols_groups")
    assert (n, average, method, box) == (3, "arithmetic", "all", None)
    assert value == pytest.approx(MEAN_3, rel=1e-6)
    assert sd == pytest.approx(9.98611, rel=1e-5)
    assert se == pytest.approx(5.7655, rel=1e-4)
    assert synth.fetch(db_path, "SELECT file_id FROM invols_group_members "
                       "WHERE group_id = ? ORDER BY file_id", (gid,)) == [(1,), (2,), (3,)]
    assert synth.fetch(db_path, "SELECT file_id, group_id FROM "
                       "invols_assignments ORDER BY file_id") == [(i, gid) for i in ids]
    rows = _transform_rows(db_path)
    assert [r[0] for r in rows] == ids
    for _, v, params in rows:
        assert v == value                       # the one stored group value
        assert json.loads(params) == {"invols_group": gid}
    assert result.group_id == gid
    assert result.n_written == 4
    assert result.summary.n == 3
    assert result.skipped_no_instrument == ()
    assert result.left_out_old_settings == ()


def test_apply_harmonic_stores_the_harmonic_mean(tmp_path):
    db_path, _ = _standard(tmp_path)
    store.apply_ensemble(_request(db_path, average="harmonic"), db_path)
    (average, value), = synth.fetch(
        db_path, "SELECT average, transform_nm_v FROM invols_groups")
    assert average == "harmonic"
    assert value == pytest.approx(199.14, rel=1e-12)
    assert [r[1] for r in _transform_rows(db_path)] == [value] * 4


def test_unknown_average_raises_and_writes_nothing(tmp_path):
    db_path, _ = _standard(tmp_path)
    before = _state(db_path)
    with pytest.raises(ValueError):
        store.apply_ensemble(_request(db_path, average="median"), db_path)
    assert _state(db_path) == before


def test_applying_leaves_results_and_histograms_untouched(tmp_path):
    db_path, ids = _standard(tmp_path, event_params="P")
    for file_id in ids[:3]:
        synth.add_other_rows(db_path, file_id)
    tables = ("event_map", "event_histograms", "files")
    kept = "SELECT * FROM analysis_results WHERE analysis_type != ? ORDER BY 1, 2"
    before = ({t: _rows(db_path, t) for t in tables},
              synth.fetch(db_path, kept, (TRANSFORM,)))
    assert len([r for r in before[1] if r[1] == "event"]) == 4
    assert len([r for r in before[1] if r[1] == "invols_slope"]) == 3
    assert len(before[0]["event_map"]) == 3 and len(before[0]["event_histograms"]) == 3

    store.apply_ensemble(_request(db_path), db_path)

    after = ({t: _rows(db_path, t) for t in tables},
             synth.fetch(db_path, kept, (TRANSFORM,)))
    assert after == before


def test_target_without_instrument_is_skipped_and_listed(tmp_path):
    files = [{"instrument": INSTRUMENT, "slope": s, "measured_at": _stamp(10 + i)}
             for i, s in enumerate((-0.95, -1.0, -1.05))]
    files.append({"instrument": None, "measured_at": _stamp(13)})
    db_path, ids = synth.make_catalog(tmp_path, files)
    synth.stamp_fit_key(db_path, ids)
    result = store.apply_ensemble(_request(db_path), db_path)
    assert result.skipped_no_instrument == (4,)
    assert result.n_written == 3
    assert synth.fetch(db_path, "SELECT file_id FROM invols_assignments "
                       "ORDER BY file_id") == [(1,), (2,), (3,)]


@pytest.mark.parametrize("bad", UNUSABLE)
def test_target_with_unusable_instrument_is_skipped(tmp_path, bad):
    db_path, ids = _standard(tmp_path)
    synth_sql = "UPDATE files SET inv_ols_nm_v = ? WHERE id = ?"
    _execute(db_path, synth_sql, (bad, ids[3]))
    result = store.apply_ensemble(_request(db_path), db_path)
    assert result.skipped_no_instrument == (4,)
    assert result.n_written == 3
    assert synth.fetch(db_path, "SELECT file_id FROM invols_assignments "
                       "WHERE file_id = 4") == []
    assert [r[0] for r in _transform_rows(db_path)] == [1, 2, 3]


@pytest.mark.parametrize("bad", UNUSABLE)
def test_member_with_unusable_instrument_is_dropped_quietly(tmp_path, bad):
    db_path, ids = _standard(tmp_path)
    _execute(db_path, "UPDATE files SET inv_ols_nm_v = ? WHERE id = ?", (bad, ids[2]))
    result = store.apply_ensemble(
        _request(db_path, members=(1, 2, 3), targets=(1, 2)), db_path)
    assert result.summary.n == 2
    assert result.left_out_old_settings == ()
    (n, value), = synth.fetch(db_path, "SELECT n, transform_nm_v FROM invols_groups")
    assert n == 2
    assert value == pytest.approx((OWN[1] + OWN[2]) / 2, rel=1e-6)
    assert synth.fetch(db_path, "SELECT file_id FROM invols_group_members "
                       "ORDER BY file_id") == [(1,), (2,)]


def test_member_with_a_positive_slope_is_dropped(tmp_path):
    db_path, _ = _standard(tmp_path, slopes=(-0.95, -1.0, 0.9, None))
    result = store.apply_ensemble(_request(db_path), db_path)
    assert result.summary.n == 2
    assert result.left_out_old_settings == ()
    (n, value), = synth.fetch(db_path, "SELECT n, transform_nm_v FROM invols_groups")
    assert n == 2
    assert value == pytest.approx((OWN[1] + OWN[2]) / 2, rel=1e-6)
    assert synth.fetch(db_path, "SELECT file_id FROM invols_group_members "
                       "ORDER BY file_id") == [(1,), (2,)]


def test_no_eligible_member_raises_and_writes_nothing(tmp_path):
    db_path, _ = _standard(tmp_path)
    before = _state(db_path)
    with pytest.raises(store.InvolsApplyError):
        store.apply_ensemble(_request(db_path, members=(4,)), db_path)
    assert _state(db_path) == before


def test_failure_inside_the_transaction_leaves_nothing(tmp_path, monkeypatch):
    db_path, ids = _standard(tmp_path)
    for file_id in ids:                                  # an earlier value each
        synth.add_result(db_path, file_id, TRANSFORM, 100.0 + file_id, "old")
    before = _state(db_path)
    assert len(before[TRANSFORM]) == 4
    real = db.write_analysis_results_multi
    calls = []

    def fails_on_the_second_write(*args, **kwargs):
        if calls:
            raise RuntimeError("disk full")
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(db, "write_analysis_results_multi", fails_on_the_second_write)
    monkeypatch.setattr(store, "write_analysis_results_multi",
                        fails_on_the_second_write, raising=False)
    with pytest.raises(Exception):
        store.apply_ensemble(_request(db_path), db_path)
    assert calls == [1]                                  # it did get that far
    assert _state(db_path) == before
    assert before["invols_groups"] == [] and before["invols_assignments"] == []


def test_box_method_stores_the_box_as_four_floats(tmp_path):
    db_path, _ = _standard(tmp_path)
    box = (_hour(10), _hour(12), 150, 250)
    store.apply_ensemble(_request(db_path, method="box", box=box), db_path)
    (method, box_json), = synth.fetch(
        db_path, "SELECT method, box_json FROM invols_groups")
    assert method == "box"
    stored = json.loads(box_json)
    assert stored == [_hour(10), _hour(12), 150.0, 250.0]
    assert all(isinstance(x, float) for x in stored)


@pytest.mark.parametrize("method, box", [("box", None), ("median", None),
                                         ("median", (0.0, 1.0, 0.0, 1.0))])
def test_bad_method_raises_and_writes_nothing(tmp_path, method, box):
    db_path, _ = _standard(tmp_path)
    before = _state(db_path)
    with pytest.raises(ValueError):
        store.apply_ensemble(_request(db_path, method=method, box=box), db_path)
    assert _state(db_path) == before


def test_one_member_stores_no_sd_or_se(tmp_path):
    db_path, _ = _standard(tmp_path)
    result = store.apply_ensemble(_request(db_path, members=(1,)), db_path)
    assert result.summary.n == 1
    (n, value, sd, se), = synth.fetch(
        db_path, "SELECT n, transform_nm_v, sd_nm_v, se_nm_v FROM invols_groups")
    assert (n, sd, se) == (1, None, None)
    assert value == pytest.approx(OWN[1], rel=1e-6)


def test_applying_twice_keeps_both_groups_and_points_at_the_second(tmp_path):
    db_path, ids = _standard(tmp_path)
    first = store.apply_ensemble(_request(db_path), db_path)
    second = store.apply_ensemble(_request(db_path, members=(1, 2)), db_path)
    assert first.group_id != second.group_id
    groups = synth.fetch(db_path, "SELECT id, n FROM invols_groups ORDER BY id")
    assert groups == [(first.group_id, 3), (second.group_id, 2)]
    assert synth.fetch(db_path, "SELECT file_id, group_id FROM invols_assignments "
                       "ORDER BY file_id") == [(i, second.group_id) for i in ids]
    assert synth.fetch(db_path, "SELECT COUNT(*) FROM invols_group_members "
                       "WHERE group_id = ?", (first.group_id,)) == [(3,)]
    expected = (OWN[1] + OWN[2]) / 2
    for _, v, _p in _transform_rows(db_path):
        assert v == pytest.approx(expected, rel=1e-6)


# ── Back to the own fit ────────────────────────────────────────────────────────

def test_apply_own_removes_assignments_and_restores_own_values(tmp_path):
    db_path, ids = _standard(tmp_path)
    store.apply_ensemble(_request(db_path), db_path)
    slope_keys = dict(synth.fetch(db_path, "SELECT file_id, params_json FROM "
                                  "analysis_results WHERE analysis_type = 'invols_slope'"))
    assert store.apply_own((1, 2, 3, 4), db_path) == 4
    assert _rows(db_path, "invols_assignments") == []
    assert synth.fetch(db_path, "SELECT COUNT(*) FROM invols_groups") == [(1,)]
    rows = _transform_rows(db_path)
    assert [r[0] for r in rows] == ids
    for (file_id, value, params), expected in zip(rows[:3], OWN.values()):
        assert value == pytest.approx(expected, rel=1e-6)
        assert params == slope_keys[file_id]
    assert rows[3][1] is None
    assert all(r.source == "own" for r in store.trace_rows(_paths(), db_path))


# ── The averaging option ───────────────────────────────────────────────────────

def test_average_defaults_to_arithmetic(tmp_path):
    db_path, _ = synth.make_catalog(tmp_path, [])
    assert store.get_average(db_path) == "arithmetic"


def test_average_choice_is_remembered(tmp_path):
    db_path, _ = synth.make_catalog(tmp_path, [])
    store.set_average("harmonic", db_path)
    assert store.get_average(db_path) == "harmonic"
    assert db.get_app_setting("invols_average", "", db_path) == "harmonic"


def test_unknown_average_is_refused_and_the_choice_kept(tmp_path):
    db_path, _ = synth.make_catalog(tmp_path, [])
    store.set_average("harmonic", db_path)
    with pytest.raises(ValueError):
        store.set_average("median", db_path)
    assert store.get_average(db_path) == "harmonic"


# ── Labels and warnings ────────────────────────────────────────────────────────

def test_source_labels_own_then_ensemble(tmp_path):
    db_path, ids = _standard(tmp_path)
    assert store.source_labels(ids, db_path) == {i: "own fit" for i in ids}
    store.apply_ensemble(_request(db_path), db_path)
    labels = store.source_labels(ids, db_path)
    assert set(labels) == set(ids)
    for text in labels.values():
        assert re.match(r"^ensemble of 3 traces, \d{4}-\d{2}-\d{2}$", text)


def test_stale_setting_groups_empty_then_names_the_group(tmp_path):
    db_path, _ = _standard(tmp_path)
    result = store.apply_ensemble(_request(db_path), db_path)
    assert store.stale_setting_groups(_paths(), db_path) == []
    db.update_analysis_param("invols_window_pts", 150, db_path)
    assert store.stale_setting_groups(_paths(), db_path) == [result.group_id]


def test_fit_settings_now_reports_the_settings_in_force(tmp_path):
    db_path, _ = synth.make_catalog(tmp_path, [])
    db.update_analysis_param("invols_window_pts", 150, db_path)
    now = store.fit_settings_now(db_path)
    assert set(now) == {"invols_offset_pts", "invols_window_pts",
                        "spectral_cutoff_hz"}
    assert now["invols_window_pts"] == 150
    assert now["invols_offset_pts"] == db.load_analysis_params(db_path).invols_offset_pts


# curve_label.  Hand calculation: 199.14 / 0.9852 = 202.1315 -> "202.13".
OWN_LABEL = "InvOLS in use 202.13 nm/V (own fit) · slope=-0.9852 · RMS=0.1948 nm"
ENSEMBLE_LABEL = ("InvOLS in use 210.00 nm/V (ensemble of 3 traces) · "
                  "own fit 202.13 nm/V · slope=-0.9852 · RMS=0.1948 nm")


def _label_catalog(tmp_path):
    db_path, ids = _standard(tmp_path)
    synth.add_result(db_path, ids[0], "invols_rms", 0.1948)
    # the stored fit is the one the label's slope comes from
    synth.put_result(db_path, ids[0], "invols_slope", -0.9852,
                     synth.current_fit_key(db_path))
    return db_path, ids


def test_curve_label_for_an_own_fit_trace(tmp_path):
    db_path, _ = _label_catalog(tmp_path)
    label = store.curve_label(synth.path_of(1), -0.9852, INSTRUMENT, db_path)
    assert OWN_LABEL in label


def test_curve_label_for_an_ensemble_trace(tmp_path):
    db_path, ids = _label_catalog(tmp_path)
    synth.add_group(db_path, ids[:3], 210.0)
    label = store.curve_label(synth.path_of(1), -0.9852, INSTRUMENT, db_path)
    assert ENSEMBLE_LABEL in label


def test_curve_label_for_an_ensemble_trace_with_no_own_fit(tmp_path):
    db_path, ids = _label_catalog(tmp_path)
    synth.add_group(db_path, ids[:3], 210.0)
    _execute(db_path, "INSERT INTO invols_assignments (file_id, group_id) "
             "VALUES (4, 1)")
    label = store.curve_label(synth.path_of(4), None, INSTRUMENT, db_path)
    assert "InvOLS in use 210.00 nm/V (ensemble of 3 traces) · own fit —" in label


def test_curve_label_takes_the_in_use_name_from_variables(tmp_path, monkeypatch):
    monkeypatch.setattr(variables, "IN_USE_INVOLS_LABEL", "In-use SENTINEL",
                        raising=False)
    db_path, ids = _label_catalog(tmp_path)
    own = store.curve_label(synth.path_of(1), -0.9852, INSTRUMENT, db_path)
    synth.add_group(db_path, ids[:3], 210.0)
    ensemble = store.curve_label(synth.path_of(1), -0.9852, INSTRUMENT, db_path)
    assert own.startswith("In-use SENTINEL 202.13 nm/V (own fit)")
    assert ensemble.startswith("In-use SENTINEL 210.00 nm/V (ensemble of 3 traces)")
    assert "InvOLS in use" not in own + ensemble


# ── An assigned file whose instrument value has become unusable ────────────────

@pytest.fixture(params=UNUSABLE, ids=["null", "zero", "negative", "inf"])
def broken_after_apply(request, tmp_path):
    """Apply an ensemble to all four files, then make file 1's instrument value
    unusable, as a re-read of the file can."""
    db_path, ids = _standard(tmp_path)
    store.apply_ensemble(_request(db_path), db_path)
    _execute(db_path, "UPDATE files SET inv_ols_nm_v = ? WHERE id = 1",
             (request.param,))
    # A stored Transform InvOLS row exists for file 1, and must still not show.
    assert [r for r in _transform_rows(db_path) if r[0] == 1][0][1] is not None
    return db_path, ids


def test_unusable_instrument_means_no_ensemble_in_trace_rows(broken_after_apply):
    db_path, _ = broken_after_apply
    rows = {r.file_id: r for r in store.trace_rows(_paths(), db_path)}
    assert rows[1].source == "own"
    assert rows[1].group_id is None
    assert rows[1].in_use_nm_v is None
    assert rows[2].source == "ensemble"                 # the others are unaffected


def test_unusable_instrument_is_labelled_own_fit(broken_after_apply):
    db_path, _ = broken_after_apply
    assert store.source_labels([1, 2], db_path)[1] == "own fit"
    label = store.curve_label(synth.path_of(1), -0.95, 199.14, db_path)
    assert "(own fit)" in label
    assert "ensemble" not in label


def test_reading_does_not_delete_the_assignment(broken_after_apply):
    db_path, _ = broken_after_apply
    store.trace_rows(_paths(), db_path)
    store.source_labels([1, 2, 3, 4], db_path)
    store.curve_label(synth.path_of(1), -0.95, 199.14, db_path)
    assert synth.fetch(db_path, "SELECT group_id FROM invols_assignments "
                       "WHERE file_id = 1") == [(1,)]


# ── One settings snapshot, taken inside the transaction ───────────────────────

def _window_of(params_json):
    return json.loads(params_json)["window_pts"]


def test_settings_changing_during_apply_cannot_mislabel_the_group(
        tmp_path, monkeypatch):
    """Fits are stored under window 150 and the setting is now 100.  A second
    connection sets it back to 150 right after apply's first read of the
    settings.  Either apply refuses, or the group names the window the members
    were actually fitted under."""
    db_path, ids = _standard(tmp_path)
    db.update_analysis_param("invols_window_pts", 150, db_path)
    synth.stamp_fit_key(db_path, ids)                    # fitted under 150
    db.update_analysis_param("invols_window_pts", 100, db_path)
    request = _request(db_path)
    real, seen = db.load_analysis_params, []

    def flips_after_the_first_read(*args, **kwargs):
        result = real(*args, **kwargs)
        has_conn = kwargs.get("conn") is not None or len(args) > 1
        if not has_conn and not seen:
            seen.append(1)
            db.update_analysis_param("invols_window_pts", 150, db_path)
        return result

    monkeypatch.setattr(db, "load_analysis_params", flips_after_the_first_read)
    try:
        store.apply_ensemble(request, db_path)
    except store.InvolsApplyError:
        assert _rows(db_path, "invols_groups") == []
        return
    (settings_json,), = synth.fetch(db_path, "SELECT fit_settings_json FROM invols_groups")
    member_windows = {_window_of(p) for (p,) in synth.fetch(
        db_path, "SELECT params_json FROM analysis_results WHERE "
        "analysis_type = 'invols_slope' AND file_id IN (1, 2, 3)")}
    assert member_windows == {150}
    assert json.loads(settings_json)["invols_window_pts"] == 150


def test_setting_changed_just_before_the_transaction_opens_refuses(
        tmp_path, monkeypatch):
    db_path, _ = _standard(tmp_path)                     # fitted under the default
    request = _request(db_path)
    before = _state(db_path)
    real = store._write_transaction

    def changes_the_setting_first(path):
        db.update_analysis_param("invols_window_pts", 150, path)
        return real(path)

    monkeypatch.setattr(store, "_write_transaction", changes_the_setting_first)
    with pytest.raises(store.InvolsApplyError):
        store.apply_ensemble(request, db_path)
    assert _state(db_path) == before


# ── in_use_nm_v is the stored Transform InvOLS ────────────────────────────────

def test_in_use_is_the_stored_row_for_an_own_fit_file(tmp_path):
    db_path, ids = _standard(tmp_path)
    synth.add_result(db_path, ids[0], TRANSFORM, 123.45, "{}")   # not 209.62
    rows = {r.file_id: r for r in store.trace_rows(_paths(), db_path)}
    assert rows[1].source == "own"
    assert rows[1].in_use_nm_v == 123.45


def test_in_use_is_the_stored_row_for_an_ensemble_file(tmp_path):
    db_path, ids = _standard(tmp_path)
    synth.add_group(db_path, ids[:3], 210.0)
    synth.add_result(db_path, ids[0], TRANSFORM, 111.11, "{}")   # not 210
    rows = {r.file_id: r for r in store.trace_rows(_paths(), db_path)}
    assert rows[1].source == "ensemble"
    assert rows[1].in_use_nm_v == 111.11


def test_in_use_is_none_for_an_own_fit_file_with_no_stored_row(tmp_path):
    db_path, _ = _standard(tmp_path)
    assert _transform_rows(db_path) == []
    rows = store.trace_rows(_paths(), db_path)
    assert [r.in_use_nm_v for r in rows] == [None] * 4


# ── Smaller items ─────────────────────────────────────────────────────────────

def test_every_target_skipped_writes_nothing(tmp_path):
    db_path, ids = _standard(tmp_path)
    _execute(db_path, "UPDATE files SET inv_ols_nm_v = NULL WHERE id IN (3, 4)")
    before = _state(db_path)
    result = store.apply_ensemble(
        _request(db_path, members=(1, 2), targets=(3, 4)), db_path)
    assert result.group_id is None
    assert result.n_written == 0
    assert result.skipped_no_instrument == (3, 4)
    assert _state(db_path) == before
    assert before["invols_groups"] == [] and before["invols_group_members"] == []


def test_the_same_path_twice_gives_one_row(tmp_path):
    db_path, _ = _standard(tmp_path)
    rows = store.trace_rows([synth.path_of(1), synth.path_of(1)], db_path)
    assert [r.file_id for r in rows] == [1]


def test_no_code_identity_refuses_apply_ensemble(tmp_path, monkeypatch):
    db_path, _ = _standard(tmp_path)
    request = _request(db_path)
    before = _state(db_path)
    _no_code_identity(monkeypatch)
    with pytest.raises(store.InvolsApplyError):
        store.apply_ensemble(request, db_path)
    assert _state(db_path) == before


def test_no_code_identity_refuses_apply_own(tmp_path, monkeypatch):
    db_path, _ = _standard(tmp_path)
    store.apply_ensemble(_request(db_path), db_path)
    before = _state(db_path)
    _no_code_identity(monkeypatch)
    with pytest.raises(store.InvolsApplyError):
        store.apply_own((1, 2, 3, 4), db_path)
    assert _state(db_path) == before
    assert len(before["invols_assignments"]) == 4


def test_request_settings_that_differ_from_those_in_force_are_ignored(tmp_path):
    db_path, _ = _standard(tmp_path)
    wrong = {"invols_offset_pts": 1.0, "invols_window_pts": 9.0,
             "spectral_cutoff_hz": 5.0}
    request = store.EnsembleRequest((1, 2, 3), (1, 2, 3, 4), "all", None, wrong)
    store.apply_ensemble(request, db_path)
    (settings_json,), = synth.fetch(db_path, "SELECT fit_settings_json FROM invols_groups")
    assert json.loads(settings_json) == store.fit_settings_now(db_path)
    assert json.loads(settings_json) != wrong


def _execute(db_path, sql, params=()):
    import sqlite3
    con = sqlite3.connect(db_path)
    try:
        with con:
            con.execute(sql, params)
    finally:
        con.close()


# The RMS belongs to the stored fit: shown only for the slope that fit made.

def test_curve_label_shows_no_rms_for_a_slope_that_is_not_the_stored_fit(
        tmp_path):
    db_path, _ = _label_catalog(tmp_path)        # stored slope -0.9852
    label = store.curve_label(synth.path_of(1), -0.9000, INSTRUMENT, db_path)
    assert "slope=-0.9000" in label
    assert "RMS=" not in label


def test_curve_label_shows_the_rms_for_the_stored_slope(tmp_path):
    db_path, _ = _label_catalog(tmp_path)
    label = store.curve_label(synth.path_of(1), -0.9852, INSTRUMENT, db_path)
    assert "RMS=0.1948 nm" in label


def test_curve_label_shows_no_rms_when_there_is_no_own_slope(tmp_path):
    db_path, _ = _label_catalog(tmp_path)
    label = store.curve_label(synth.path_of(1), None, INSTRUMENT, db_path)
    assert "RMS=" not in label


# curve_label(..., own_rms): the RMS of the fit on screen, when the caller has it.

def test_curve_label_shows_a_given_own_rms(tmp_path):
    db_path, _ = _label_catalog(tmp_path)        # stored RMS 0.1948
    label = store.curve_label(synth.path_of(1), -0.9852, INSTRUMENT, db_path,
                              own_rms=0.3)
    assert "RMS=0.3" in label
    assert "0.1948" not in label


def test_curve_label_shows_a_given_own_rms_whatever_is_stored(tmp_path):
    db_path, _ = _label_catalog(tmp_path)
    label = store.curve_label(synth.path_of(1), -0.9, INSTRUMENT, db_path,
                              own_rms=0.3)
    assert "RMS=0.3" in label


def test_curve_label_shows_a_given_own_rms_for_an_unknown_file(tmp_path):
    db_path, _ = _label_catalog(tmp_path)
    label = store.curve_label(synth.path_of(99), -0.9852, INSTRUMENT, db_path,
                              own_rms=0.3)
    assert "RMS=0.3" in label


def test_curve_label_without_own_rms_falls_back_to_the_stored_comparison(
        tmp_path):
    db_path, _ = _label_catalog(tmp_path)
    match = store.curve_label(synth.path_of(1), -0.9852, INSTRUMENT, db_path,
                              own_rms=None)
    other = store.curve_label(synth.path_of(1), -0.9, INSTRUMENT, db_path,
                              own_rms=None)
    assert "RMS=0.1948 nm" in match
    assert "RMS=" not in other


def test_curve_label_shows_no_rms_without_a_slope_even_if_one_is_given(tmp_path):
    db_path, _ = _label_catalog(tmp_path)
    label = store.curve_label(synth.path_of(1), None, INSTRUMENT, db_path,
                              own_rms=0.3)
    assert "RMS=" not in label


def test_curve_label_shows_no_rms_for_a_nan_slope_even_with_an_own_rms(tmp_path):
    db_path, _ = _label_catalog(tmp_path)
    label = store.curve_label(synth.path_of(1), float("nan"), INSTRUMENT,
                              db_path, own_rms=0.3)
    assert "RMS=" not in label


# ── A group of one trace reads in the singular ─────────────────────────────────

def test_source_label_of_a_group_of_one_trace_is_singular(tmp_path):
    db_path, ids = _standard(tmp_path)
    synth.add_group(db_path, ids[:1], 210.0)
    text = store.source_labels(ids[:1], db_path)[ids[0]]
    assert text.startswith("ensemble of 1 trace,")


def test_source_label_of_a_group_of_two_traces_stays_plural(tmp_path):
    db_path, ids = _standard(tmp_path)
    synth.add_group(db_path, ids[:2], 210.0)
    text = store.source_labels(ids[:1], db_path)[ids[0]]
    assert text.startswith("ensemble of 2 traces,")


def test_curve_label_of_a_group_of_one_trace_is_singular(tmp_path):
    db_path, ids = _label_catalog(tmp_path)
    synth.add_group(db_path, ids[:1], 210.0)
    label = store.curve_label(synth.path_of(1), -0.9852, INSTRUMENT, db_path)
    assert "InvOLS in use 210.00 nm/V (ensemble of 1 trace) · own fit" in label
