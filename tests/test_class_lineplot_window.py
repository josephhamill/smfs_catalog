"""Focused contracts for the non-event audit browser."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

import tmpdirs

from smfs_catalog import class_lineplot_window as _window
from smfs_catalog import db as _real_db
from smfs_catalog.curve_loader import LoadError


_app = QApplication.instance() or QApplication([])

# A real catalog, because the window reads the deflection-histogram store even
# when its queue listing is stubbed. Empty, so both populations sum to nothing.
_DB = os.path.join(tmpdirs.mkdtemp(prefix="smfs_nonevents_"), "catalog.db")
_real_db.initialise(_DB)


def _row(path: str, event: str) -> dict:
    return {
        "path": path,
        "filename": path.rsplit("/", 1)[-1],
        "event": event,
        "status": "done",
    }


def test_browser_filters_to_non_events_and_loads_only_the_selected_curve(monkeypatch):
    rows = [
        _row("/data/negative.ibw", "non_event"),
        _row("/data/event.ibw", "event"),
    ]
    loaded = []
    monkeypatch.setattr(_window._db, "list_queue", lambda _db: rows)

    def load(path):
        loaded.append(path)
        return SimpleNamespace(
            piezo_retr=np.array([1.0, 2.0]),
            defl_retr=np.array([3.0, 4.0]),
        )

    monkeypatch.setattr(_window, "load_force_curve", load)
    win = _window.ClassLinePlotWindow("non_event", _DB)
    try:
        assert win.windowTitle() == "SMFS — Non-events"
        assert [row["path"] for row in win._rows] == ["/data/negative.ibw"]
        assert loaded == ["/data/negative.ibw"]
        assert win._counter.text() == "1 / 1"
    finally:
        win.close()


def test_empty_state_disables_actions_and_refresh_reports_load_failure(monkeypatch):
    rows = []
    monkeypatch.setattr(_window._db, "list_queue", lambda _db: rows)
    win = _window.ClassLinePlotWindow("non_event", _DB)
    try:
        assert not win._btn_auto_fwd.isEnabled()
        assert not win._export_btn.isEnabled()
        assert "No analysed non-events" in win._status_lbl.text()

        rows.append(_row("/data/missing.ibw", "non_event"))

        def fail(_path):
            raise LoadError("drive unavailable")

        monkeypatch.setattr(_window, "load_force_curve", fail)
        win.refresh()

        assert win._export_btn.isEnabled()
        assert "drive unavailable" in win._status_lbl.text()
        assert win._curve.xData is None or win._curve.xData.size == 0
    finally:
        win.close()


def test_browser_rejects_verdicts_it_does_not_mean_to_display():
    try:
        _window.ClassLinePlotWindow("event", _DB)
    except ValueError as exc:
        assert "non_event" in str(exc)
    else:
        raise AssertionError("unsupported verdict was accepted")


def _fresh_catalog(paths):
    """A real catalog holding `paths` as files, with no histograms stored."""
    db_path = os.path.join(tmpdirs.mkdtemp(prefix="smfs_nonevents_"), "c.db")
    _real_db.initialise(db_path)
    now = "2026-01-01 00:00:00"
    for p in paths:
        _real_db.upsert_file(
            {"path": p, "filename": p.rsplit("/", 1)[-1], "parse_ok": 1,
             "first_seen": now, "last_seen": now}, db_path=db_path)
    return db_path


def test_a_shown_curve_with_no_stored_histogram_is_stored_and_joins_the_total(
        monkeypatch):
    from smfs_catalog import event_processor as ep

    paths = ["/data/a.ibw", "/data/b.ibw"]
    db_path = _fresh_catalog(paths)
    rows = [_row(p, "non_event") for p in paths]
    monkeypatch.setattr(_window._db, "list_queue", lambda _db: rows)
    monkeypatch.setattr(_window, "load_force_curve", lambda _p: SimpleNamespace(
        piezo_retr=np.array([1.0, 2.0, 3.0]),
        defl_retr=np.array([0.0, 1.0, 2.0])))

    win = _window.ClassLinePlotWindow("non_event", db_path)
    try:
        key = ep.defl_grid_params()
        fid_a = _real_db.get_file_id("/data/a.ibw", db_path)
        assert _real_db.get_deflection_histogram(fid_a, key, db_path) is not None
        assert win._n_binned == 1
        assert int(win._cohort_counts.sum()) == 3

        win._go_next()
        assert win._n_binned == 2
        assert int(win._cohort_counts.sum()) == 6
        assert "2 of 2 binned" in win._hist_lbl.text()

        # Paging back to a curve that is now stored adds nothing twice.
        win._go_prev()
        assert win._n_binned == 2
        assert int(win._cohort_counts.sum()) == 6
    finally:
        win.close()


def test_a_shown_curve_with_a_stored_histogram_uses_it_and_does_not_rebin(
        monkeypatch):
    from smfs_catalog import event_processor as ep

    path = "/data/stored.ibw"
    db_path = _fresh_catalog([path])
    fid = _real_db.get_file_id(path, db_path)
    stored = np.zeros(ep.DEFL_HIST_BINS, dtype=np.uint32)
    stored[500] = 42
    _real_db.write_deflection_histogram(
        fid, stored, 0, 0, ep.defl_grid_params(), db_path)

    monkeypatch.setattr(_window._db, "list_queue",
                        lambda _db: [_row(path, "non_event")])
    monkeypatch.setattr(_window, "load_force_curve", lambda _p: SimpleNamespace(
        piezo_retr=np.array([1.0, 2.0]), defl_retr=np.array([3.0, 4.0])))

    def must_not_bin(*_a, **_k):
        raise AssertionError("a stored histogram was binned again")

    monkeypatch.setattr(_window._ep, "compute_deflection_histogram",
                        must_not_bin)

    win = _window.ClassLinePlotWindow("non_event", db_path)
    try:
        assert win._n_binned == 1
        assert int(win._cohort_counts.sum()) == 42
    finally:
        win.close()
