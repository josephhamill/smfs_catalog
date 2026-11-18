# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""Focused guards for decomposition-window interactions and control ranges."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smfs_catalog import decomposition_window as dw  # noqa: E402


class _Line:
    def __init__(self, value=0.0):
        self._value = value
        self.visible = False

    def value(self):
        return self._value

    def setValue(self, value):
        self._value = value

    def setVisible(self, visible):
        self.visible = visible


class _SpinBox:
    def __init__(self, maximum=9999):
        self._maximum = maximum
        self.value = None

    def maximum(self):
        return self._maximum

    def blockSignals(self, _blocked):
        pass

    def setValue(self, value):
        self.value = value


class _Signal:
    def __init__(self):
        self.count = 0

    def emit(self):
        self.count += 1


def test_trim_line_drag_recomputes_landmarks(monkeypatch):
    """Moving only the guide must not leave old contact markers on screen."""
    writes = []
    monkeypatch.setattr(
        dw._db, "update_analysis_param",
        lambda key, value, path: writes.append((key, value, path)),
    )

    probe = type("Probe", (), {})()
    probe._n_appr = 100
    probe._trim_pts = 5
    probe._trim_spinbox = _SpinBox(maximum=100)
    probe._trim_left = _Line()
    probe._trim_right = _Line()
    probe._current_curve = object()
    probe._db_path = "catalog.db"
    probe._save_user_profile = lambda: None
    probe.analysis_params_changed = _Signal()
    redrawn = []
    probe.update_curve = redrawn.append

    dw.DecompositionWindow._on_trim_line_moved(probe, _Line(88), left=True)

    assert probe._trim_pts == 12
    assert writes == [("turnaround_trim_pts", 12.0, "catalog.db")]
    assert redrawn == [probe._current_curve]
    assert probe.analysis_params_changed.count == 1


def test_threshold_guides_remain_editable_without_landmarks():
    """Threshold inputs are visible independently of detection success."""
    probe = type("Probe", (), {})()
    probe._thresh_appr_val = 0.25
    probe._thresh_retr_val = 0.75
    probe._thresh_appr = _Line()
    probe._thresh_retr = _Line()

    dw.DecompositionWindow._refresh_threshold_guides(probe)

    assert probe._thresh_appr.value() == dw._to_shown(0.25)
    assert probe._thresh_retr.value() == dw._to_shown(0.75)
    assert probe._thresh_appr.visible
    assert probe._thresh_retr.visible


def test_fit_and_baseline_regions_have_no_fixed_cap(tmp_path):
    """The invOLS region and baseline width take any size their methods can use.

    Each floor is the method's own: an offset of 0, a two-point line, a width
    above 0 nm.  The top is the largest value the box can hold, so a long dwell
    at full sample rate fits.
    """
    from PyQt6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication([])  # named, so it outlives the window
    db_path = str(tmp_path / "catalog.sqlite")
    dw._db.initialise(db_path)
    win = dw.DecompositionWindow(db_path)
    try:
        cases = [
            (win._invols_offset_spinbox, "invols_offset_pts", 0, 60_000),
            (win._invols_window_spinbox, "invols_window_pts", 2, 60_000),
            (win._anchor_spinbox, "baseline_anchor_nm", 1, 5_000),
        ]
        for box, key, floor, above_old_cap in cases:
            assert box.minimum() == floor, key
            assert box.maximum() == dw._quant.SPIN_INT_MAX, key
            box.setValue(above_old_cap)
            assert box.value() == above_old_cap, key
            assert dw._db.load_analysis_params(db_path)[key] == above_old_cap, key
    finally:
        win.close()
