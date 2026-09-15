# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/navigation.py
#
# The "Go to:" row — one curve, seen in another window.
#
# Every inspection window shows a cohort and a current curve; the question
# "what does THIS curve look like in the raw viewer / the ROI search / the
# decomposition view" is the same question in all of them, so it is answered
# once here rather than per window.
#
# Routing goes through the dashboard, not through window-to-window references.
# The dashboard owns the singleton viewer and the worker playhead, and its
# reveal_*_at methods move that playhead; the ROI and decomposition windows
# follow the playhead through their WorkerNavBar and sync on show. So a jump is
# "move the playhead, then reveal the window", and no inspection window needs a
# handle on any other. A directly-linked raw window, where a caller has one, is
# a fallback for the case where no dashboard is alive — a bare window in a test
# or a minimal environment.
#
# The dashboard is located by class name among the top-level widgets rather than
# by import, because importing it here would be circular: the dashboard builds
# the windows that call this.
#
# Tooltips live here for the same reason the handlers do. When they were copied
# per window they drifted, and one copy set the same tooltip twice with the
# longer text overwritten by the shorter.

from __future__ import annotations

from typing import Callable

from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
)

TIP_RAW = "Pause analysis and open this curve in the scan (raw) window"
TIP_ROI = (
    "Open the event search — the detection signals and thresholds that decide "
    "where this curve's ROIs are, and therefore which ruptures and segments it "
    "has."
)
TIP_DECOMP = (
    "Open the decomposition view on this curve — the baseline, contact and "
    "retract segments the landmark detection works from."
)
TIP_DASHBOARD = "Bring the dashboard window to the front"


def find_dashboard():
    """The singleton dashboard among the app's top-level windows, or None."""
    for w in QApplication.topLevelWidgets():
        if type(w).__name__ == "DashboardWindow":
            return w
    return None


def _warn_no_target(parent, what: str) -> None:
    QMessageBox.information(
        parent, f"Go to {what}", f"No {what} window is available.")


def go_to_raw(parent, path: str, raw_win=None) -> None:
    """Reveal `path` in the raw viewer."""
    if not path:
        return
    dash = find_dashboard()
    if dash is not None and hasattr(dash, "reveal_raw_at"):
        dash.reveal_raw_at(path)
        return
    if raw_win is not None and raw_win.go_to_path(path):
        return
    _warn_no_target(parent, "scan (raw)")


def go_to_roi(parent, path: str, raw_win=None) -> None:
    """Reveal `path` in the ROI detection window."""
    if not path:
        return
    dash = find_dashboard()
    if dash is not None and hasattr(dash, "reveal_roi_at"):
        dash.reveal_roi_at(path)
        return
    if raw_win is not None:
        opener = getattr(raw_win, "open_roi_window", None)
        if callable(opener):
            opener()
        raw_win.go_to_path(path)
        return
    _warn_no_target(parent, "ROI")


def go_to_decomp(parent, path: str, raw_win=None) -> None:
    """Reveal `path` in the decomposition window."""
    if not path:
        return
    dash = find_dashboard()
    if dash is not None and hasattr(dash, "reveal_decomp_at"):
        dash.reveal_decomp_at(path)
        return
    if raw_win is not None:
        opener = getattr(raw_win, "open_decomp_window", None)
        if callable(opener):
            opener()
        raw_win.go_to_path(path)
        return
    _warn_no_target(parent, "decomposition")


def go_to_dashboard(parent) -> None:
    """Raise the dashboard."""
    dash = find_dashboard()
    if dash is not None:
        dash.show()
        dash.raise_()
        dash.activateWindow()


def build_go_to_row(
    parent,
    path_fn:  Callable[[], str | None],
    *,
    raw_win_fn: Callable[[], object] | None = None,
    decomp:   bool = False,
) -> tuple[QHBoxLayout, list[QPushButton]]:
    """The "Go to:" label and its buttons, as a layout plus the buttons.

    `path_fn` is called at click time, not now, so the row always acts on
    whatever curve the window is currently showing. `raw_win_fn` supplies the
    fallback raw window for callers that own one.

    The returned buttons are the curve-dependent ones, for the caller to enable
    and disable with the rest of its navigation. Dashboard is not among them: it
    needs no curve, so an empty cohort is no reason to close that door.
    """
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(QLabel("Go to:"))

    def _raw_win():
        return raw_win_fn() if raw_win_fn is not None else None

    specs = [("Raw", TIP_RAW, go_to_raw), ("ROI", TIP_ROI, go_to_roi)]
    if decomp:
        specs.append(("Decomp", TIP_DECOMP, go_to_decomp))

    buttons: list[QPushButton] = []
    for label, tip, fn in specs:
        btn = QPushButton(label)
        btn.setToolTip(tip)
        btn.clicked.connect(
            lambda _checked=False, f=fn: f(parent, path_fn(), _raw_win()))
        row.addWidget(btn)
        buttons.append(btn)

    dash_btn = QPushButton("Dashboard")
    dash_btn.setToolTip(TIP_DASHBOARD)
    dash_btn.clicked.connect(lambda _checked=False: go_to_dashboard(parent))
    row.addWidget(dash_btn)

    return row, buttons
