from __future__ import annotations

import numpy as np
from types import SimpleNamespace
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from smfs_catalog.curve_loader import ForceCurve
from smfs_catalog import display_roi
from smfs_catalog import curve_analysis
from smfs_catalog import dashboard_window
from smfs_catalog import rawcurve_window as raw


class _Worker(QObject):
    playhead_changed = pyqtSignal(int)
    queue_empty = pyqtSignal()
    file_done = pyqtSignal(int, str, bool)
    file_error = pyqtSignal(int, str)
    data_unavailable = pyqtSignal(int, str, str)
    paused_changed = pyqtSignal(bool)
    direction_changed = pyqtSignal(int)
    throttle_changed = pyqtSignal(int)
    queue_changed = pyqtSignal()

    def queue_ids(self):
        return []

    def playhead(self):
        return None

    def throttle_ms(self):
        return 0

    def is_paused(self):
        return True

    def direction(self):
        return 1

    def set_paused(self, _paused):
        pass

    def set_direction(self, _direction):
        pass

    def set_throttle_ms(self, _ms):
        pass

    def notify_work_available(self):
        pass


def _curve(*, xpos=0.0, ypos=0.0):
    return ForceCurve(
        path="curve.ibw",
        piezo_appr=np.array([0.0, 1.0]),
        defl_appr=np.array([0.0, 0.1]),
        piezo_retr=np.array([1.0, 0.0]),
        defl_retr=np.array([0.1, 0.0]),
        spring_constant=10.0,
        xpos=xpos,
        ypos=ypos,
    )


def _window(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(
        display_roi,
        "ROIWindow",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("disabled")),
    )
    win = raw.RawCurveWindow([], worker=_Worker())
    win._test_app = app
    return win


def test_worker_mode_draws_raw_curve_without_running_analysis(monkeypatch):
    win = _window(monkeypatch)
    monkeypatch.setattr(raw, "load_force_curve", lambda _path: _curve())
    monkeypatch.setattr(win, "_draw_persisted_overlays", lambda _fid: False)
    monkeypatch.setattr(
        win,
        "_draw_derived",
        lambda *_args: (_ for _ in ()).throw(AssertionError("analysis ran in GUI")),
    )
    win._paths = ["curve.ibw"]
    win._current_file_id = 7

    try:
        win._do_draw(0)
        assert win._curve_appr.xData.tolist() == [0.0, 1.0]
        assert win._status_label.text() == "analysis in progress…"
    finally:
        win.close()


def test_late_worker_completion_cannot_redraw_an_old_curve(monkeypatch):
    win = _window(monkeypatch)
    calls = []
    monkeypatch.setattr(
        win,
        "_draw_persisted_overlays",
        lambda file_id, **_kwargs: calls.append(file_id) or True,
    )
    win._current_file_id = 8

    try:
        win._on_worker_file_done(7, "event", False)
        win._on_worker_file_done(8, "non_event", False)
        assert calls == [8]
    finally:
        win.close()


def test_zero_stage_origin_is_displayed_but_missing_coordinates_are_not(monkeypatch):
    win = _window(monkeypatch)
    try:
        win._draw(_curve(xpos=0.0, ypos=0.0))
        assert win._meta_vals["xy"].text() == "(0.00, 0.00) µm"

        win._draw(_curve(xpos=None, ypos=None))
        assert win._meta_vals["xy"].text() == "—"
    finally:
        win.close()


def test_current_worker_failure_is_visible(monkeypatch):
    win = _window(monkeypatch)
    win._current_file_id = 5
    try:
        win._on_worker_file_error(4, "old failure")
        assert "old failure" not in win._status_label.text()
        win._on_worker_file_error(5, "contact fit failed")
        assert "contact fit failed" in win._status_label.text()
    finally:
        win.close()


def test_overlay_read_failure_is_visible(monkeypatch):
    win = _window(monkeypatch)
    win._current_file_id = 5
    monkeypatch.setattr(
        win,
        "_draw_persisted_overlays",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad map")),
    )
    try:
        win._on_worker_file_done(5, "event", False)
        assert "ValueError" in win._status_label.text()
        assert "bad map" in win._status_label.text()
    finally:
        win.close()


def test_persisted_non_event_overlays_are_read_without_curve_analysis(monkeypatch):
    win = _window(monkeypatch)
    params = object()
    projected = SimpleNamespace(all_params="all", params_cd="contact")
    monkeypatch.setattr(raw, "cache_version", lambda: "clean-build")
    monkeypatch.setattr(raw._db, "load_analysis_params", lambda _path: params)
    monkeypatch.setattr(curve_analysis, "pipeline_params_from", lambda _p: projected)
    monkeypatch.setattr(raw._db, "get_analysis_result", lambda *_args: 0.0)
    monkeypatch.setattr(
        raw._db,
        "get_analysis_results_multi",
        lambda *_args: {
            "contact_piezo_nm": 12.5,
            "snapoff_piezo_nm": 8.0,
        },
    )
    monkeypatch.setattr(
        win,
        "_draw_derived",
        lambda *_args: (_ for _ in ()).throw(AssertionError("analysis ran in GUI")),
    )

    try:
        assert win._draw_persisted_overlays(7, event="non_event")
        assert win._contact_appr_line.value() == 12.5
        assert win._contact_retr_line.value() == 8.0
    finally:
        win.close()


def test_inspection_navigation_pauses_before_stepping(monkeypatch):
    calls = []
    worker = SimpleNamespace(
        set_paused=lambda paused: calls.append(("paused", paused)),
        step_to=lambda file_id: calls.append(("step", file_id)),
    )
    host = SimpleNamespace(
        _db_path="catalog.db",
        _worker=worker,
        _queue_id_to_row={7: 0},
        _on_open_viewer=lambda: calls.append(("open", None)),
    )
    monkeypatch.setattr(dashboard_window._db, "get_file_id", lambda *_args: 7)

    dashboard_window.DashboardWindow._open_raw_viewer(host, "curve.ibw")

    assert calls == [("open", None), ("paused", True), ("step", 7)]
