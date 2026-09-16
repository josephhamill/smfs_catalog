"""Explore Events explains each missing curve, for whatever axes are plotted."""

import json
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from smfs_catalog import db as _db
from smfs_catalog import roi_pipeline as _rp
from smfs_catalog import variables as _vars
from smfs_catalog.event_summary_window import (
    EventSummaryWindow, drop_breakdown_lines, fit_outcome_text,
)
from smfs_catalog.ledger import Ledger
from smfs_catalog.provenance import cache_version
from smfs_catalog.roi_events import (
    ROI, CurveEvents, Rupture, Segment, events_to_payload,
)

X, Y = "seg_x_rupture_nm", "seg_force_pN"


class _Window(EventSummaryWindow):
    _x_key = X
    _y_key = Y


def _outcome(status, detail=None, *, n_segments=1):
    return {"n_segments": n_segments, "fit_status": status, "fit_detail": detail}


def _window(outcomes):
    win = _Window.__new__(_Window)
    n = len(outcomes)
    win._results = [{"path": f"/data/c{i}.ibw"} for i in range(n)]
    for name in ("_force_arr", "_length_arr", "_x_arr", "_y_arr"):
        setattr(win, name, np.full(n, np.nan))
    win._seg_outcome = list(outcomes)
    win._segment_select = "ultimate"
    win._var_by_key = {}
    return win


def _drops(led):
    return {d.path: (d.reason, d.detail) for d in led.drops()}


def test_each_drop_names_its_stored_reason():
    win = _window([
        _outcome("fit_available"),                                   # plotted
        None,                                                        # never loaded
        _outcome(None, n_segments=None),                             # no analysis
        _outcome(None, n_segments=1),                                # no such segment
        _outcome("no_fit", "no force peak"),
        _outcome("not_attempted"),
    ])
    win._x_arr[0], win._y_arr[0] = 12.0, 50.0

    drops = _drops(win._plottability_ledger())

    assert "/data/c0.ibw" not in drops
    assert drops["/data/c1.ibw"] == (
        "not_finite", f"no {_vars.label(X)}, no {_vars.label(Y)}; segment: ultimate")
    assert drops["/data/c2.ibw"][0] == "no_stored_segments"
    assert drops["/data/c3.ibw"] == ("no_segment_chosen", "segment: ultimate")
    assert drops["/data/c4.ibw"] == ("fit_not_attempted", "no force peak; segment: ultimate")
    assert drops["/data/c5.ibw"] == ("fit_not_attempted", "fitter did not run; segment: ultimate")


def test_a_failed_fit_is_plotted_on_fit_free_axes_but_not_downstream(monkeypatch):
    win = _window([_outcome("no_fit", "optimizer failed")])
    win._x_arr[0], win._y_arr[0] = 12.0, 50.0        # force peak was found
    monkeypatch.setattr(win, "_live_hit_mask", lambda: np.array([True]), raising=False)

    assert win._plottability_ledger().n_dropped == 0
    assert _drops(win.population_ledger("hit")) == {
        "/data/c0.ibw": ("fit_failed", "optimizer failed; segment: ultimate")}


def test_a_value_the_fit_outcome_does_not_explain_is_named():
    win = _window([_outcome("fit_available"), _outcome("no_fit", "no force peak")])
    win._y_arr[:] = 50.0
    axes = [("seg_dF_pN", win._x_arr), ("contact_dx_nm", win._y_arr * np.nan)]

    led = Ledger("t", ["/data/c0.ibw", "/data/c1.ibw"])
    win._record_missing(led, 0, "/data/c0.ibw", axes)
    win._record_missing(led, 1, "/data/c1.ibw", axes)

    drops = _drops(led)
    assert drops["/data/c0.ibw"] == (
        "not_finite",
        f"no {_vars.label('seg_dF_pN')}, no {_vars.label('contact_dx_nm')}; segment: ultimate")
    assert drops["/data/c1.ibw"][0] == "not_finite"


def test_the_dialog_line_leads_with_the_stored_reason():
    win = _window([_outcome("no_fit", "insufficient loading ramp")])

    assert drop_breakdown_lines(win._plottability_ledger()) == [
        "1 × fit not attempted (insufficient loading ramp; segment: ultimate)"]


def test_breakdown_splits_a_reason_by_stored_outcome():
    led = Ledger("t", ["a", "b", "c"])
    led.drop("a", "no_fit", "no force peak; segment: ultimate")
    led.drop("b", "no_fit", "no force peak; segment: ultimate")
    led.drop("c", "no_fit", "insufficient loading ramp; segment: ultimate")

    lines = drop_breakdown_lines(led)

    assert len(lines) == 2
    assert lines[0].startswith("2 × ") and "no force peak" in lines[0]
    assert lines[1].startswith("1 × ") and "insufficient loading ramp" in lines[1]


@pytest.mark.parametrize("status, detail, text", [
    ("fit_available", None, "fit"),
    ("review", "peak at segment boundary", "review: peak at segment boundary"),
    ("no_fit", "optimizer failed", "optimizer failed"),
    ("not_attempted", None, "not attempted"),
])
def test_outcome_text(status, detail, text):
    assert fit_outcome_text({"fit_status": status, "fit_detail": detail}) == text


def _catalog_with_one_curve(tmp_path):
    """One stored curve: penultimate segment fitted (review), ultimate segment's
    force peak found but its WLC optimizer failed."""
    db = str(tmp_path / "t.sqlite")
    _db.initialise(db)
    path = _db.normalize_path("/data/curve.ibw")
    conn = _db.get_connection(db)
    with conn:
        conn.execute(
            "INSERT INTO files (path, filename, first_seen, last_seen, event)"
            " VALUES (?, 'curve.ibw', datetime('now'), datetime('now'), 'event')",
            (path,))
    fid = conn.execute("SELECT id FROM files WHERE path=?", (path,)).fetchone()[0]
    conn.close()

    def seg(left, right, **kw):
        return Segment(left_idx=left, right_idx=right,
                       left_piezo_nm=float(left), right_piezo_nm=float(right), **kw)

    events = CurveEvents(detector="test", rois=[ROI(
        onset_idx=0, return_idx=300, onset_piezo_nm=0.0, return_piezo_nm=300.0,
        ruptures=[Rupture(idx=i, piezo_nm=float(i), d1_height=1.0, prominence=1.0,
                          force_pN=f, extension_nm=x)
                  for i, f, x in ((100, 80.0, 40.0), (200, 60.0, 90.0))],
        segments=[seg(0, 100, l_p_nm=0.4, l_c_nm=50.0, left_extension_nm=0.0,
                      fit_status="review", fit_detail="peak at segment boundary"),
                  seg(100, 200, fit_status="no_fit", fit_detail="optimizer failed")],
    )])
    _db.write_event_map(fid, json.dumps(events_to_payload(events)),
                        json.dumps({"tag": "t"}), cache_version() or "test", db)
    return db, path


def test_axes_choose_what_is_plotted_but_not_the_downstream_cohort(tmp_path):
    from PyQt6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication([])
    db, path = _catalog_with_one_curve(tmp_path)

    win = EventSummaryWindow([{"path": path}], db)

    assert (win._x_key, win._y_key) == ("seg_x_rupture_nm", "seg_force_pN")
    assert (win._x_arr[0], win._y_arr[0]) == (90.0, 60.0)
    assert win._event_list.count() == 1
    assert win.population_paths("hit") == []

    win._x_combo.setCurrentIndex(win._x_combo.findData("seg_l_c_nm"))

    assert win._event_list.count() == 0
    (drop,) = win._plottability_ledger().drops()
    assert (drop.reason, drop.detail) == ("fit_failed", "optimizer failed; segment: ultimate")

    win._on_swap_axes()
    assert (win._x_key, win._y_key) == ("seg_force_pN", "seg_l_c_nm")
    win.close()


def _catalog_with_curves(tmp_path, n):
    """`n` fully fitted curves, each with a different force and extension."""
    db = str(tmp_path / "many.sqlite")
    _db.initialise(db)
    paths = []
    for i in range(n):
        path = _db.normalize_path(f"/data/curve{i}.ibw")
        conn = _db.get_connection(db)
        with conn:
            conn.execute(
                "INSERT INTO files (path, filename, first_seen, last_seen, event)"
                " VALUES (?, ?, datetime('now'), datetime('now'), 'event')",
                (path, f"curve{i}.ibw"))
        fid = conn.execute("SELECT id FROM files WHERE path=?", (path,)).fetchone()[0]
        conn.close()
        events = CurveEvents(detector="test", rois=[ROI(
            onset_idx=0, return_idx=300, onset_piezo_nm=0.0, return_piezo_nm=300.0,
            ruptures=[Rupture(idx=100, piezo_nm=100.0, d1_height=1.0, prominence=1.0,
                              force_pN=50.0 + 10.0 * i, extension_nm=20.0 + 5.0 * i)],
            segments=[Segment(left_idx=0, right_idx=100, left_piezo_nm=0.0,
                              right_piezo_nm=100.0, l_p_nm=0.4,
                              l_c_nm=60.0 + 4.0 * i, left_extension_nm=0.0,
                              fit_status="fit_available")],
        )])
        _db.write_event_map(fid, json.dumps(events_to_payload(events)),
                            json.dumps({"tag": "t"}), cache_version() or "test", db)
        paths.append(path)
    return db, paths


def test_cluster_colours_are_placed_on_the_chosen_axes(tmp_path):
    from PyQt6.QtWidgets import QApplication
    from smfs_catalog import clustering as _cl
    _app = QApplication.instance() or QApplication([])
    db, paths = _catalog_with_curves(tmp_path, 4)
    win = EventSummaryWindow([{"path": p} for p in paths], db)
    try:
        _cl.set_current(_cl.Clustering(
            labels={p: i % 2 for i, p in enumerate(paths)},
            k=2, seed=1, n_pcs=3, sklearn_version="test"))
        win._cluster_bar._chk.setChecked(True)          # triggers a rebuild

        assert win._cluster_bar.is_active()
        drawn = {(round(s.pos().x(), 6), round(s.pos().y(), 6))
                 for s in win._scatter_pass.points()}
        assert drawn == {(round(float(x), 6), round(float(y), 6))
                         for x, y in zip(win._x_arr, win._y_arr)}

        win._y_combo.setCurrentIndex(win._y_combo.findData("seg_l_c_nm"))
        moved = {(round(s.pos().x(), 6), round(s.pos().y(), 6))
                 for s in win._scatter_pass.points()}
        assert moved == {(round(float(x), 6), round(float(y), 6))
                         for x, y in zip(win._x_arr, win._y_arr)}
        assert moved != drawn
    finally:
        _cl.clear()
        win.close()


def test_the_scatter_reports_and_exports_its_correlation(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from smfs_catalog import export_utils
    _app = QApplication.instance() or QApplication([])
    db, paths = _catalog_with_curves(tmp_path, 4)
    out = tmp_path / "exports"
    out.mkdir()
    export_utils.set_export_dir_override(str(out), db)
    monkeypatch.setattr("smfs_catalog.event_summary_window.QMessageBox.information",
                        lambda *a, **k: None)

    win = EventSummaryWindow([{"path": p} for p in paths], db)

    assert win._corr is not None and win._fit is not None
    assert "Spearman" in win._stats_label.text() and "R²" in win._stats_label.text()

    win._on_export_scatter()
    (manifest_path,) = out.glob("scatter_*_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["slope"] == pytest.approx(win._fit.slope)
    assert manifest["ci_pct"] and "fit_ci_lo" in manifest["columns"]

    # X against itself is an identity, so it must not be reported as a result.
    win._y_combo.setCurrentIndex(win._y_combo.findData(win._x_key))
    assert win._fit is None and win._corr is None
    assert "Spearman" not in win._stats_label.text()
    win.close()


def test_exports_follow_the_axes(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from smfs_catalog import export_utils
    _app = QApplication.instance() or QApplication([])
    db, path = _catalog_with_one_curve(tmp_path)
    conn = _db.get_connection(db)
    with conn:
        conn.execute("UPDATE files SET measured_at='2026-03-18 10:00:00' WHERE path=?", (path,))
    conn.close()
    out = tmp_path / "exports"
    out.mkdir()
    export_utils.set_export_dir_override(str(out), db)
    monkeypatch.setattr("smfs_catalog.event_summary_window.QMessageBox.information",
                        lambda *a, **k: None)

    win = EventSummaryWindow([{"path": path}], db)
    win._y_combo.setCurrentIndex(win._y_combo.findData(_vars.TIME_KEY))
    win._on_export_scatter()
    win._on_export_x_hist()

    (scatter,) = out.glob("scatter_*_manifest.json")
    manifest = json.loads(scatter.read_text())
    assert (manifest["x_variable"], manifest["y_variable"]) == ("seg_x_rupture_nm", _vars.TIME_KEY)
    assert list(out.glob("hist_seg_x_rupture_nm_*_manifest.json"))
    win.close()


def test_segment_summary_reports_the_selected_segments_outcome(tmp_path):
    db, path = _catalog_with_one_curve(tmp_path)

    ult = _rp.segment_summary_bulk([path], "ultimate", db)[path]
    pen = _rp.segment_summary_bulk([path], "penultimate", db)[path]

    assert (ult["fit_status"], ult["fit_detail"]) == ("no_fit", "optimizer failed")
    assert (pen["fit_status"], pen["fit_detail"]) == ("review", "peak at segment boundary")
    assert "fit_status" not in _rp.SEG_SUMMARY_FIELD.values()
