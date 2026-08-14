"""Behavioral guards for shared Qt helpers and their raw-view integration.

The geometry cases exercise the helper directly, including work areas smaller
than the normal minimum. The raw-view case protects the metadata panel's Qt
ownership: labels in a parentless form layout reserve blank space but never
paint, which is visually indistinguishable from an intentionally empty panel.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QRect
from PyQt6.QtWidgets import QApplication

from smfs_catalog.qt_utils import _make_session_header, fit_on_screen
from smfs_catalog.rawcurve_window import RawCurveWindow


class _Screen:
    def __init__(self, available: QRect) -> None:
        self._available = available

    def availableGeometry(self) -> QRect:
        return QRect(self._available)


class _Window:
    def __init__(self, available: QRect, frame: QRect) -> None:
        self._screen = _Screen(available)
        self._frame = QRect(frame)

    def screen(self):
        return self._screen

    def resize(self, width: int, height: int) -> None:
        self._frame.setSize(QRect(0, 0, width, height).size())

    def frameGeometry(self) -> QRect:
        return QRect(self._frame)

    def move(self, x: int, y: int) -> None:
        self._frame.moveTopLeft(QRect(x, y, 0, 0).topLeft())


def test_fit_on_screen_bounds_size_and_position() -> None:
    available = QRect(100, 50, 800, 600)
    win = _Window(available, QRect(850, 620, 100, 100))

    fit_on_screen(win, 1400, 900)

    frame = win.frameGeometry()
    assert frame.size().width() == 704
    assert frame.size().height() == 504
    assert frame.left() >= available.left()
    assert frame.top() >= available.top()
    assert frame.x() + frame.width() <= available.x() + available.width()
    assert frame.y() + frame.height() <= available.y() + available.height()


def test_fit_on_screen_does_not_force_minimum_past_small_work_area() -> None:
    win = _Window(QRect(0, 0, 400, 300), QRect(0, 0, 100, 100))

    fit_on_screen(win, 1400, 900)

    assert win.frameGeometry().size().width() == 304
    assert win.frameGeometry().size().height() == 204


def test_session_header_handles_nullable_metadata() -> None:
    app = QApplication.instance() or QApplication([])
    header = _make_session_header({
        "experimentalist": None,
        "directory": "",
        "analyte": "DNA",
        "technique": None,
        "n_curves": 12,
    })

    assert header is not None
    assert header.text() == "—  ·  —  ·  DNA  ·  —  ·  12 curves"
    assert app is not None


def test_raw_curve_metadata_labels_belong_to_visible_panel() -> None:
    app = QApplication.instance() or QApplication([])
    win = RawCurveWindow([], worker=None)
    win.show()
    app.processEvents()

    try:
        assert win._meta_vals
        assert all(label.parentWidget() is not None
                   for label in win._meta_vals.values())
        assert all(label.isVisible() for label in win._meta_vals.values())
    finally:
        win.close()
