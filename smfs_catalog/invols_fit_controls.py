# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
The one pair of invOLS fit spin boxes (offset and window), shared by every
window that shows them.

A user edit writes the catalog and the active owner's profile and emits
`changed` once.  set_values and refresh_from_db only move what is shown; they
emit nothing and write nothing.
"""
from __future__ import annotations

from PyQt6.QtCore import QSignalBlocker, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QSpinBox, QWidget

from . import db as _db
from . import quantities as _quant
from .widgets import LabeledControl

_OFFSET_KEY = "invols_offset_pts"
_WINDOW_KEY = "invols_window_pts"

_OFFSET_TIP = (
    "Samples skipped back from the turnaround before the invOLS fit "
    "window begins. The deepest-contact samples carry piezo ringing and "
    "any sample deformation at peak load, both of which bend the line "
    "that converts deflection into force.\n\n"
    "If the offset and window together reach past the start of the "
    "approach, no slope can be fitted and the curve is not analysed: "
    "its readouts show \"—\".")
_WINDOW_TIP = (
    "Number of samples in the invOLS straight-line fit, counted back "
    "from the offset above toward shallower contact.\n\n"
    "This slope is what turns raw deflection into force, so it scales "
    "every force in the app. The fitted slope and RMS residual are shown "
    "beside it. If the window reaches past the start of the approach, "
    "no slope can be fitted and the curve is not analysed.")


class InvolsFitControls(QWidget):
    changed = pyqtSignal()

    def __init__(self, db_path: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db_path = db_path
        params = _db.load_analysis_params(db_path)

        self.offset_spinbox = self._make_box(
            0, _OFFSET_KEY, params.invols_offset_pts, _OFFSET_TIP)
        self.window_spinbox = self._make_box(
            2, _WINDOW_KEY, params.invols_window_pts, _WINDOW_TIP)
        self.offset_spinbox.valueChanged.connect(
            lambda v: self._on_edit(_OFFSET_KEY, v))
        self.window_spinbox.valueChanged.connect(
            lambda v: self._on_edit(_WINDOW_KEY, v))

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        lay.addWidget(LabeledControl("invOLS off:", self.offset_spinbox))
        lay.addWidget(LabeledControl("invOLS win:", self.window_spinbox))

    @staticmethod
    def _make_box(minimum: int, key: str, value: float, tip: str) -> QSpinBox:
        box = QSpinBox()
        box.setRange(minimum, _quant.SPIN_INT_MAX)
        _quant.configure_spinbox(box, key)
        box.setValue(int(value))
        box.setToolTip(tip)
        return box

    def _on_edit(self, key: str, value: int) -> None:
        _db.update_analysis_param(key, float(value), self._db_path)
        self.changed.emit()

    def offset_pts(self) -> int:
        return self.offset_spinbox.value()

    def window_pts(self) -> int:
        return self.window_spinbox.value()

    def set_values(self, offset_pts: int, window_pts: int) -> None:
        for box, value in ((self.offset_spinbox, offset_pts),
                           (self.window_spinbox, window_pts)):
            if box.value() == int(value):
                continue
            box.blockSignals(True)
            box.setValue(int(value))
            box.blockSignals(False)

    def load_values(self, offset_pts: int, window_pts: int) -> None:
        for box, value in ((self.offset_spinbox, offset_pts),
                           (self.window_spinbox, window_pts)):
            with QSignalBlocker(box):
                box.setValue(int(value))

    def refresh_from_db(self) -> None:
        params = _db.load_analysis_params(self._db_path)
        self.set_values(params.invols_offset_pts, params.invols_window_pts)
