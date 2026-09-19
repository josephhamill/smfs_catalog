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
    EventSummaryWindow, drop_breakdown_lines, fit_outcome_rows, fit_outcome_text,
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
    for name in ("_x_arr", "_y_arr"):
        setattr(win, name, np.full(n, np.nan))
    win._seg_outcome = list(outcomes)
    win._segment_select = "ultimate"
    win._var_by_key = {}
    return win


def _plotted_ledger(win):
    return win._plotted_ledger(np.ones(len(win._results), dtype=bool))


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

    drops = _drops(_plotted_ledger(win))

    assert "/data/c0.ibw" not in drops
    assert drops["/data/c1.ibw"] == (
        "not_finite", f"no {_vars.label(X)}, no {_vars.label(Y)}; segment: ultimate")
    assert drops["/data/c2.ibw"][0] == "no_stored_segments"
    assert drops["/data/c3.ibw"] == ("no_segment_chosen", "segment: ultimate")
    assert drops["/data/c4.ibw"] == ("fit_not_attempted", "no force peak; segment: ultimate")
    assert drops["/data/c5.ibw"] == ("fit_not_attempted", "fitter did not run; segment: ultimate")


def test_a_failed_fit_is_plotted_and_handed_on(monkeypatch):
    win = _window([_outcome("no_fit", "optimizer failed")])
    win._x_arr[0], win._y_arr[0] = 12.0, 50.0        # force peak was found
    monkeypatch.setattr(win, "_live_hit_mask", lambda: np.array([True]), raising=False)

    assert _plotted_ledger(win).n_dropped == 0
    # The population is membership only; a consumer that needs the fit says so.
    assert win.population_ledger("hit").n_dropped == 0
    lc = np.array([np.nan])
    assert _drops(win.population_ledger("hit", [("seg_l_c_nm", lc)])) == {
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

    assert drop_breakdown_lines(_plotted_ledger(win)) == [
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


def test_outcome_counts_add_up_to_plotted_and_not_plotted(monkeypatch):
    win = _window([
        _outcome("fit_available"), _outcome("fit_available"),
        _outcome("review", "peak at segment boundary"),
        _outcome("no_fit", "optimizer failed"),               # force kept, plotted
        _outcome("no_fit", "no force peak"),
        _outcome(None, n_segments=None),
        _outcome(None, n_segments=1),
    ])
    win._x_arr[:4], win._y_arr[:4] = 12.0, 50.0
    win._active_population = "all"
    monkeypatch.setattr(win, "_live_hit_mask", lambda: np.ones(7, dtype=bool),
                        raising=False)
    in_pop = win._population_mask()
    shown = win._plotted_mask(in_pop)
    led = win._plotted_ledger(in_pop)

    rows = fit_outcome_rows(win._seg_outcome, shown)

    assert rows[0] == ("fit", 2, 2)
    assert {t: (n, p) for t, n, p in rows[1:]} == {
        "review: peak at segment boundary": (1, 1),
        "optimizer failed": (1, 1),
        "no force peak": (1, 0),
        "no stored analysis": (1, 0),
        "no such segment": (1, 0),
    }
    assert sum(n for _, n, _ in rows) == led.n_asked == 7
    assert sum(p for _, _, p in rows) == led.n_kept == int(shown.sum()) == 4
    assert led.n_kept + led.n_dropped == led.n_asked


def test_the_tally_counts_only_the_population_on_screen(monkeypatch):
    win = _window([_outcome("fit_available")] * 2 + [_outcome("no_fit", "no force peak")] * 2)
    win._x_arr[[0, 2]], win._y_arr[[0, 2]] = 12.0, 50.0
    win._active_population = "hit"
    monkeypatch.setattr(win, "_live_hit_mask",
                        lambda: np.array([True, True, False, False]), raising=False)
    in_pop = win._population_mask()

    led = win._plotted_ledger(in_pop)

    assert (led.n_asked, led.n_kept, led.n_dropped) == (2, 1, 1)
    assert led.n_kept == int(win._plotted_mask(in_pop).sum())
    assert [d.path for d in led.drops()] == ["/data/c1.ibw"]


def _catalog_with_one_curve(tmp_path):
    """One stored curve: penultimate segment fitted (review); ultimate segment's
    force peak found and its ramp regains the first rupture's force (a
    measured reload distance of 30 nm), but its WLC optimizer failed."""
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
                  for i, f, x in ((100, 60.0, 40.0), (200, 80.0, 90.0))],
        segments=[seg(0, 100, l_p_nm=0.4, l_c_nm=50.0, left_extension_nm=0.0,
                      fit_status="review", fit_detail="peak at segment boundary"),
                  seg(100, 200, isoforce_x_nm=70.0,
                      fit_status="no_fit", fit_detail="optimizer failed")],
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
    assert (win._x_arr[0], win._y_arr[0]) == (90.0, 80.0)
    assert win._event_list.count() == 1
    assert win.population_paths("hit") == [path]

    win._x_combo.setCurrentIndex(win._x_combo.findData("seg_l_c_nm"))
    assert win.population_paths("hit") == [path]      # the axes never filter it

    assert win._event_list.count() == 0
    (drop,) = _plotted_ledger(win).drops()
    assert (drop.reason, drop.detail) == ("fit_failed", "optimizer failed; segment: ultimate")

    win._on_swap_axes()
    assert (win._x_key, win._y_key) == ("seg_force_pN", "seg_l_c_nm")
    win.close()


def test_the_outcomes_dialog_states_scope_and_totals(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QApplication, QMessageBox
    _app = QApplication.instance() or QApplication([])
    db, path = _catalog_with_one_curve(tmp_path)
    shown = {}
    monkeypatch.setattr(QMessageBox, "exec", lambda box: shown.update(
        text=box.text(), info=box.informativeText(), detail=box.detailedText()))

    win = EventSummaryWindow([{"path": path}], db)
    win._x_combo.setCurrentIndex(win._x_combo.findData("seg_l_c_nm"))
    win._on_show_outcomes()

    assert "Hits · segment: Ultimate · 1 curves" in shown["text"]
    assert "optimizer failed" in shown["info"]
    assert "      1        0            1  total" in shown["info"]
    assert f"{path}\toptimizer failed\tfit failed" in shown["detail"]
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
    assert "Spearman" in win._fit_label.text() and "R²" in win._fit_label.text()
    assert win._fit_label.text().startswith("fit — Hits:")

    win._on_export_scatter()
    (manifest_path,) = out.glob("scatter_*_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["slope"] == pytest.approx(win._fit.slope)
    assert manifest["ci_pct"] and "fit_ci_lo" in manifest["columns"]

    # X against itself is an identity, so it must not be reported as a result.
    win._y_combo.setCurrentIndex(win._y_combo.findData(win._x_key))
    assert win._fit is None and win._corr is None
    assert win._fit_label.text() == ""
    win.close()


def test_one_control_draws_and_analyses_the_same_curves(tmp_path, monkeypatch):
    """What is plotted is what is analysed: the scatter, the event list, the
    count and the fit all describe the population control's population."""
    from PyQt6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication([])
    db, paths = _catalog_with_curves(tmp_path, 6)

    win = EventSummaryWindow([{"path": p} for p in paths], db)
    # The gate passes everything with no bounds set, so the split is
    # stated here instead: the first three curves are hits.
    monkeypatch.setattr(win, "_live_hit_mask",
                        lambda: np.array([True] * 3 + [False] * 3), raising=False)
    win._rebuild()

    for pop, n_drawn in (("hit", 3), ("non_hit", 3), ("all", 6)):
        win._pop_btns[pop].setChecked(True)
        drawn = len(win._scatter_pass.points()) + len(win._scatter_fail.points())
        assert drawn == n_drawn
        assert win._event_list.count() == n_drawn
        assert f"{n_drawn} plotted" in win._stats_label.text()
        assert win._corr.n == n_drawn
        assert win._fit_label.text().startswith(
            f"fit — {'Hits' if pop == 'hit' else 'Non-Hits' if pop == 'non_hit' else 'All events'}:")
        # The WLC navigator takes the same population.
        assert len(win._current_event_paths()) == n_drawn
    win.close()


def test_all_events_exports_one_file_naming_each_row(tmp_path, monkeypatch):
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
    win._pop_btns["all"].setChecked(True)

    assert win._active_population == "all"
    assert win._fit_chk.text() == "Linear fit (All events)"
    # A mixed ensemble is a legitimate one: these build over every event.
    assert win._isoforce_btn.isEnabled()
    assert win._norm_2dh_btn.isEnabled()
    assert win._phys_2dh_btn.isEnabled()
    assert sorted(win.population_paths("all")) == sorted(paths)

    win._on_export_scatter()
    (csv_path,) = out.glob("scatter_*_all_*.csv")
    header, *rows = csv_path.read_text().strip().splitlines()
    assert header.split(",")[1] == "hit"
    assert len(rows) == len(paths)
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


def test_downstream_windows_take_the_population_and_apply_their_own_needs(
        tmp_path, monkeypatch):
    """A curve whose fit failed is a member: the fit-free 2DH keeps it, the
    normalized 2DH drops it as its own no_fit, Isoforce keeps it for its
    measured reload distance, and View individual events lists it."""
    from PyQt6.QtWidgets import QApplication
    from smfs_catalog.normalized_2dh_window import Normalized2DHWindow
    from smfs_catalog.physical_2dh_window import Physical2DHWindow
    _app = QApplication.instance() or QApplication([])
    db, path = _catalog_with_one_curve(tmp_path)
    win = EventSummaryWindow([{"path": path}], db)
    # No curve file exists here, so a histogram is supplied rather than read.
    for cls in (Normalized2DHWindow, Physical2DHWindow):
        monkeypatch.setattr(cls, "_compute_from_curve",
                            lambda self, *a, **k: np.ones((2, 2), dtype=np.uint32))

    physical = Physical2DHWindow([{"path": path}], db, population="hit")
    assert physical._align_mode not in physical._FIT_DEPENDENT_ALIGN_MODES
    physical.sync_from_event_summary(win)
    assert list(physical._event_histograms) == [path]

    normalized = Normalized2DHWindow([{"path": path}], db, population="hit")
    normalized.sync_from_event_summary(win)
    assert normalized._event_histograms == {}
    assert [d.reason for d in normalized._ledger.drops()] == ["no_fit"]

    assert win._isoforce_paths("hit") == [path]
    assert win._current_event_paths() == [path]
    for w in (physical, normalized, win):
        w.close()


def test_fit_x_uses_exactly_the_plotted_curves(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication([])
    db, paths = _catalog_with_curves(tmp_path, 6)
    win = EventSummaryWindow([{"path": p} for p in paths], db)
    win._y_arr[0] = np.nan          # has an X value but is not plotted
    win._rebuild()
    opened = []
    monkeypatch.setattr(win, "_open_fit_window",
                        lambda label, units, values, paths=None, axes=None:
                        opened.append((values, paths)))

    win._on_fit_x()

    (values, fitted_paths), = opened
    assert len(values) == win._event_list.count() == 5
    assert paths[0] not in fitted_paths
    win.close()


def test_segment_summary_reports_the_selected_segments_outcome(tmp_path):
    db, path = _catalog_with_one_curve(tmp_path)

    ult = _rp.segment_summary_bulk([path], "ultimate", db)[path]
    pen = _rp.segment_summary_bulk([path], "penultimate", db)[path]

    assert (ult["fit_status"], ult["fit_detail"]) == ("no_fit", "optimizer failed")
    assert (pen["fit_status"], pen["fit_detail"]) == ("review", "peak at segment boundary")
    assert "fit_status" not in _rp.SEG_SUMMARY_FIELD.values()
