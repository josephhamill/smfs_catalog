"""The Go to: row sends the curve on screen to the right window."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

import tmpdirs

from smfs_catalog import class_lineplot_window as _window
from smfs_catalog import db as _real_db
from smfs_catalog import navigation


_app = QApplication.instance() or QApplication([])

_DB = os.path.join(tmpdirs.mkdtemp(prefix="smfs_navlinks_"), "catalog.db")
_real_db.initialise(_DB)


class _FakeDashboard:
    """Stands in for the singleton dashboard.

    Handed to the jump helpers by patching find_dashboard rather than by being
    a real window: discovery scans every top-level widget for one named
    DashboardWindow, so a dashboard another test left alive would answer first.
    """

    def __init__(self) -> None:
        self.revealed: list[tuple[str, str]] = []

    def reveal_raw_at(self, path):    self.revealed.append(("raw", path))
    def reveal_roi_at(self, path):    self.revealed.append(("roi", path))
    def reveal_decomp_at(self, path): self.revealed.append(("decomp", path))


def _row(path: str) -> dict:
    return {"path": path, "filename": path.rsplit("/", 1)[-1],
            "event": "non_event", "status": "done"}


def _curve(_path):
    return SimpleNamespace(piezo_retr=np.array([1.0, 2.0]),
                           defl_retr=np.array([3.0, 4.0]))


def test_each_button_reveals_the_current_curve_in_its_own_window(monkeypatch):
    rows = [_row("/data/first.ibw"), _row("/data/second.ibw")]
    monkeypatch.setattr(_window._db, "list_queue", lambda _db: rows)
    monkeypatch.setattr(_window, "load_force_curve", _curve)

    dash = _FakeDashboard()
    monkeypatch.setattr(navigation, "find_dashboard", lambda: dash)

    win = _window.ClassLinePlotWindow("non_event", _DB)
    try:
        raw_btn, roi_btn, decomp_btn = win._go_btns
        raw_btn.click()
        roi_btn.click()
        decomp_btn.click()
        assert dash.revealed == [
            ("raw",    "/data/first.ibw"),
            ("roi",    "/data/first.ibw"),
            ("decomp", "/data/first.ibw"),
        ]

        # The row acts on whatever curve is showing NOW, not the one that was
        # showing when it was built.
        dash.revealed.clear()
        win._go_next()
        roi_btn.click()
        assert dash.revealed == [("roi", "/data/second.ibw")]
    finally:
        win.close()


def test_an_empty_cohort_cannot_send_a_curve_anywhere(monkeypatch):
    monkeypatch.setattr(_window._db, "list_queue", lambda _db: [])
    win = _window.ClassLinePlotWindow("non_event", _DB)
    try:
        assert win._current_path() is None
        assert not any(btn.isEnabled() for btn in win._go_btns)
    finally:
        win.close()


def test_a_jump_with_no_dashboard_and_no_raw_window_warns_instead_of_raising(
        monkeypatch):
    """No target is a message, not a traceback."""
    warned = []
    monkeypatch.setattr(navigation, "find_dashboard", lambda: None)
    monkeypatch.setattr(navigation.QMessageBox, "information",
                        lambda *a, **k: warned.append(a[-1]))
    navigation.go_to_decomp(None, "/data/first.ibw")
    assert warned and "decomposition" in warned[0]
