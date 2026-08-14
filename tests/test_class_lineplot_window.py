"""Focused contracts for the non-event audit browser."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from smfs_catalog import class_lineplot_window as _window
from smfs_catalog.curve_loader import LoadError


_app = QApplication.instance() or QApplication([])


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
    win = _window.ClassLinePlotWindow("non_event", "unused.sqlite")
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
    win = _window.ClassLinePlotWindow("non_event", "unused.sqlite")
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
        _window.ClassLinePlotWindow("event", "unused.sqlite")
    except ValueError as exc:
        assert "non_event" in str(exc)
    else:
        raise AssertionError("unsupported verdict was accepted")
