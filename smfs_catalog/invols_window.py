# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""The invOLS window: one point per trace, the mean of all or of the boxed
traces, and Apply.  It reads and writes the catalog only through
``invols_store`` and does nothing until the user presses Apply."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup, QComboBox, QGroupBox, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QRadioButton, QVBoxLayout, QWidget,
)

from . import db as _db
from . import invols_store as store
from . import quantities, style, variables
from .invols_fit_controls import InvolsFitControls
from .invols_plot import InvolsPlot
from .invols_stats import (
    AVERAGES, TRANSFORM_KEY, Summary, applied_value, box_cut_count,
    in_time_span, is_drawn, points_in_box, summarise, usable_instrument,
)
from .qt_utils import fit_on_screen
from .scope import scope_to_query

_NO_VALUE = "—"
_CHOICES = (
    ("own", "Each trace's own fit",
     "Use when the contact is deep and clean on every trace, or the value "
     "may change between traces."),
    ("all", "Mean of all traces shown",
     "Use when the scope has been curated to one cantilever and one laser "
     "alignment, and the points agree."),
    ("box", "Mean of the boxed traces",
     "Use when some traces have a poor or missing contact region, or the "
     "setup changed partway: click and drag a box on the plot."),
)
_AVERAGE_NAMES = {"arithmetic": "Arithmetic mean", "harmonic": "Harmonic mean"}
_AVERAGE_TIP = (
    "Which average is applied to every trace in the window. The arithmetic "
    "mean and the harmonic mean differ by about the square of the relative "
    "scatter. The harmonic mean is the less biased when the scatter comes "
    "from the per-curve slope fits.")
_APPLY_NOTE = ("Forces and extensions update the next time the queue visits "
               "each trace. Own-fit values are kept.")
_OPEN_TIP = ("Opens this trace in the curve view. The trace is added to the "
             "analysis queue and the worker is paused so the curve stays on "
             "screen.")
_NO_POINT = "No trace selected. Click a point on the plot to see its details."
_DRAW_BOX = ("Draw a box on the plot: click and drag over the traces to "
             "include.")
_OWN_HINT = ("Each trace keeps its own fit. Choose a mean above to see its "
             "statistics.")


# ── the readout: pure text ───────────────────────────────────────────────────

@dataclass(frozen=True)
class ReadoutInput:
    summary: Summary | None
    instrument_nm_v: tuple[float, ...]
    method: str
    average: str
    n_cut: int
    n_old_settings: int


def _pick(n: int, one: str, many: str) -> str:
    return one if n == 1 else many


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {_pick(n, one, many)}"


def _nm_v(value: float | None) -> str:
    if value is None:
        return _NO_VALUE
    return quantities.format_value(TRANSFORM_KEY, value, with_unit=True) or _NO_VALUE


def _instrument_text(values: Sequence[float | None]) -> str:
    shown = [v for v in values if v is not None]
    if not shown:
        return _NO_VALUE
    low, high = (quantities.format_value(TRANSFORM_KEY, f(shown)) for f in (min, max))
    unit = quantities.get(TRANSFORM_KEY).shown_unit
    return f"{low} {unit}" if low == high else f"{low} – {high} {unit}"


def _average_lines(summary: Summary, average: str) -> list[str]:
    rows = (("arithmetic", "arithmetic mean", summary.mean),
            ("harmonic", "harmonic mean", summary.harmonic_mean))
    return [f"{label}: {_nm_v(value)}" + (" (applied)" if key == average else "")
            for key, label, value in rows]


def _spread_lines(summary: Summary) -> list[str]:
    return [
        f"standard deviation of the per-trace values: {_nm_v(summary.sd)}",
        "standard error of the mean of the per-trace values (scatter only; "
        f"shared calibration errors are not included): {_nm_v(summary.se)}",
    ]


def _cut_line(n_cut: int) -> str:
    unit = quantities.get(TRANSFORM_KEY).shown_unit
    return (f"{_plural(n_cut, 'trace', 'traces')} inside the box's time span "
            f"{_pick(n_cut, 'lies', 'lie')} outside its {unit} limits; the "
            "standard deviation and standard error describe only the traces "
            "inside the limits")


def _old_settings_line(n_old: int) -> str:
    return (f"{_plural(n_old, 'trace', 'traces')} "
            f"{_pick(n_old, 'has', 'have')} an own fit made under earlier fit "
            f"settings (drawn hollow); {_pick(n_old, 'it is', 'they are')} "
            f"left out until the queue visits {_pick(n_old, 'it', 'them')}")


def readout_text(info: ReadoutInput) -> str:
    summary = info.summary if info.summary is not None else summarise([])
    lines = [f"n = {summary.n}",
             f"instrument value {_instrument_text(info.instrument_nm_v)}"]
    if summary.n:
        lines += _average_lines(summary, info.average)
        lines += _spread_lines(summary)
    if info.method == "box" and info.n_cut > 0:
        lines.append(_cut_line(info.n_cut))
    if info.n_old_settings > 0:
        lines.append(_old_settings_line(info.n_old_settings))
    return "\n".join(lines)


# ── which traces a method uses ───────────────────────────────────────────────

@dataclass(frozen=True)
class _Selection:
    member_ids: tuple[int, ...]
    summary: Summary
    n_cut: int
    n_old: int


def _in_box(rows: Sequence[store.TraceRow],
            box: tuple[float, float, float, float]) -> tuple[list, int]:
    times = [r.time_s for r in rows]
    values = [r.own_nm_v for r in rows]
    inside = points_in_box(times, values, box)
    return ([r for r, keep in zip(rows, inside) if keep],
            box_cut_count(times, values, box))


def _in_span(rows: Sequence[store.TraceRow],
             box: tuple[float, float, float, float]) -> list:
    keep = in_time_span([r.time_s for r in rows], [r.own_nm_v for r in rows], box)
    return [r for r, inside in zip(rows, keep) if inside]


def _select(rows: Sequence[store.TraceRow], method: str,
            box: tuple[float, float, float, float] | None) -> _Selection:
    """Members and counts for 'all' or 'box'.  Membership is decided on the
    real own-fit values, never on where a pinned marker is drawn."""
    drawn = [r for r in rows if is_drawn(r.own_nm_v, r.time_s)]
    if method == "box":
        drawn = _in_span(drawn, box)
    current = [r for r in drawn if r.fit_current]
    n_old, n_cut = len(drawn) - len(current), 0
    if method == "box":
        current, n_cut = _in_box(current, box)
    return _Selection(tuple(r.file_id for r in current),
                      summarise(r.own_nm_v for r in current), n_cut, n_old)


def _no_instrument_text(n: int) -> str:
    if n == 0:
        return ""
    return (f"{_plural(n, 'trace', 'traces')} {_pick(n, 'has', 'have')} no "
            f"instrument InvOLS in {_pick(n, 'its', 'their')} file: "
            f"{_pick(n, 'it keeps its', 'they keep their')} own fit and "
            "cannot take an ensemble value.")


def _no_time_text(n: int) -> str:
    if n == 0:
        return ""
    return (f"{_plural(n, 'trace', 'traces')} {_pick(n, 'has', 'have')} no "
            f"acquisition time: {_pick(n, 'it is', 'they are')} not drawn and "
            f"not part of the mean, but {_pick(n, 'it receives', 'they receive')} "
            "the applied value.")


def _selection_text(row: store.TraceRow | None, source: str) -> str:
    if row is None:
        return _NO_POINT
    rms = quantities.format_value("invols_rms", row.rms_nm, with_unit=True)
    lines = [Path(row.path).name,
             f"{variables.OWN_INVOLS_LABEL}: {_nm_v(row.own_nm_v)}",
             f"{variables.label('invols_rms')}: {rms or _NO_VALUE}",
             f"Source: {source}"]
    if row.source == "ensemble":
        lines.append(f"{variables.IN_USE_INVOLS_LABEL}: {_nm_v(row.in_use_nm_v)}")
    return "\n".join(lines)


def _stale_group_text(group_ids: Sequence[int]) -> str:
    if not group_ids:
        return ""
    n = len(group_ids)
    named = ", ".join(f"#{i}" for i in group_ids)
    return (f"Warning: {_pick(n, 'group', 'groups')} {named} "
            f"{_pick(n, 'was', 'were')} applied under different fit settings "
            "than those in force now.")


# ── the window ───────────────────────────────────────────────────────────────

class InvolsWindow(QMainWindow):
    applied = pyqtSignal(int)

    def __init__(self, paths: Sequence[str], db_path: str, *,
                 open_trace: Callable[[str], None] | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._paths = list(paths)
        self._db = db_path
        self._open_trace = open_trace
        self._rows: list[store.TraceRow] = []
        self._method = "own"
        self._average = store.get_average(db_path)
        self._summary: Summary | None = None
        self._applied = False
        self._selected_id: int | None = None
        self._build()
        self._reload()
        fit_on_screen(self, 980, 820)

    # -- building --------------------------------------------------------

    def _build(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addLayout(self._build_header())
        self._stale_label = self._note_label(emphasis=True)
        layout.addWidget(self._stale_label)
        self._plot = InvolsPlot()
        self._plot.box_drawn.connect(lambda *_: self._refresh_view())
        self._plot.point_selected.connect(self._on_point_selected)
        self._plot.point_opened.connect(self._on_point_opened)
        layout.addWidget(self._plot, 1)
        layout.addLayout(self._build_middle())
        layout.addLayout(self._build_selection_row())
        self._no_instrument_label = self._note_label()
        self._no_time_label = self._note_label()
        layout.addWidget(self._no_instrument_label)
        layout.addWidget(self._no_time_label)
        self._build_apply(layout)
        self.setCentralWidget(root)

    @staticmethod
    def _note_label(text: str = "", emphasis: bool = False) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet(style.qss_emphasis() if emphasis else style.qss_text())
        return label

    def _build_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._owner_label = QLabel()
        row.addWidget(self._owner_label)
        self.fit_controls = InvolsFitControls(self._db)
        self.fit_controls.changed.connect(self._reload)
        row.addWidget(self.fit_controls)
        row.addStretch(1)
        return row

    def _build_middle(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(self._build_choices(), 1)
        self._readout_label = QLabel()
        self._readout_label.setWordWrap(True)
        self._readout_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(self._readout_label, 1)
        return row

    def _build_choices(self) -> QGroupBox:
        box = QGroupBox("How to set the InvOLS")
        layout = QVBoxLayout(box)
        self._radios: dict[str, QRadioButton] = {}
        group = QButtonGroup(box)
        for key, title, when in _CHOICES:
            radio = QRadioButton(title)
            radio.setChecked(key == self._method)
            radio.toggled.connect(lambda on, k=key: self._on_choice(k, on))
            group.addButton(radio)
            self._radios[key] = radio
            layout.addWidget(radio)
            layout.addWidget(self._note_label(when))
        layout.addLayout(self._build_average_row())
        return box

    def _build_average_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        label = QLabel("Average:")
        self._average_box = QComboBox()
        for key in AVERAGES:
            self._average_box.addItem(_AVERAGE_NAMES[key], key)
        self._average_box.setCurrentIndex(AVERAGES.index(self._average))
        for widget in (label, self._average_box):
            widget.setToolTip(_AVERAGE_TIP)
        self._average_box.currentIndexChanged.connect(
            lambda i: self.set_average(self._average_box.itemData(i)))
        row.addWidget(label)
        row.addWidget(self._average_box)
        row.addStretch(1)
        return row

    def _build_selection_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._selection_label = QLabel()
        self._selection_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self._open_button = QPushButton("Open this trace")
        self._open_button.clicked.connect(self._on_open_clicked)
        row.addWidget(self._selection_label, 1)
        row.addWidget(self._open_button)
        return row

    def _build_apply(self, layout: QVBoxLayout) -> None:
        self._apply_button = QPushButton()
        self._apply_button.setStyleSheet(style.QSS_PRIMARY_ACTION)
        self._apply_button.clicked.connect(self._on_apply)
        layout.addWidget(self._apply_button)
        layout.addWidget(self._note_label(_APPLY_NOTE))
        self._status_label = self._note_label()
        layout.addWidget(self._status_label)

    # -- the public contract ---------------------------------------------

    def apply_scope(self, scope: dict) -> None:
        rows = _db.list_files(db_path=self._db, **scope_to_query(scope))
        self._paths = [row["path"] for row in rows]
        self._plot.clear_box()
        self._reload()

    def refresh_params(self) -> None:
        self.fit_controls.refresh_from_db()
        self._reload()

    def method(self) -> str:
        return self._method

    def set_method(self, method: str) -> None:
        if method not in self._radios:
            raise ValueError(f"unknown method: {method!r}")
        self._radios[method].setChecked(True)

    def average(self) -> str:
        return self._average

    def set_average(self, average: str) -> None:
        store.set_average(average, self._db)
        self._average = average
        self._average_box.blockSignals(True)
        self._average_box.setCurrentIndex(AVERAGES.index(average))
        self._average_box.blockSignals(False)
        self._refresh_view()

    def readout(self) -> Summary | None:
        return self._summary

    # -- refreshing ------------------------------------------------------

    def _on_choice(self, method: str, checked: bool) -> None:
        if checked:
            self._method = method
            self._refresh_view()

    def _reload(self, summary: Summary | None = None) -> None:
        self._rows = store.trace_rows(self._paths, self._db)
        self._plot.set_traces(self._rows)
        self.setWindowTitle(
            f"InvOLS — {_plural(len(self._rows), 'trace', 'traces')}")
        self._owner_label.setText(
            f"Parameters: {_db.active_param_owner(self._db)}'s set")
        self._stale_label.setText(_stale_group_text(
            store.stale_setting_groups(self._paths, self._db)))
        self._refresh_view(summary)

    def _targets(self) -> list[store.TraceRow]:
        return [r for r in self._rows if usable_instrument(r.instrument_nm_v)]

    def _current_selection(self) -> _Selection | None:
        if self._method == "own":
            return None
        box = self._plot.box()
        if self._method == "box" and box is None:
            return None
        return _select(self._rows, self._method, box)

    def _refresh_view(self, summary: Summary | None = None) -> None:
        selection = self._current_selection()
        self._summary = None if selection is None else (summary or selection.summary)
        self._applied = summary is not None
        self._readout_label.setText(self._readout_message(selection))
        shown = self._summary
        self._plot.set_mean_line(
            applied_value(shown, self._average) if shown and shown.n else None)
        self._update_notes()
        self._update_apply(selection)
        self._update_selection()
        self._show_status("")

    def _show_status(self, text: str, emphasis: bool = False) -> None:
        self._status_label.setStyleSheet(
            style.qss_emphasis() if emphasis else style.qss_text())
        self._status_label.setText(text)

    def _readout_message(self, selection: _Selection | None) -> str:
        if self._method == "own":
            return _OWN_HINT
        if selection is None:
            return _DRAW_BOX
        return readout_text(ReadoutInput(
            self._summary, tuple(r.instrument_nm_v for r in self._targets()),
            self._method, self._average, selection.n_cut, selection.n_old))

    def _update_notes(self) -> None:
        no_instrument = [r for r in self._rows
                         if not usable_instrument(r.instrument_nm_v)]
        no_time = [r for r in self._targets() if r.time_s is None]
        self._no_instrument_label.setText(_no_instrument_text(len(no_instrument)))
        self._no_time_label.setText(_no_time_text(len(no_time)))

    # -- Apply -----------------------------------------------------------

    def _block_reason(self, selection: _Selection | None) -> str | None:
        if not self._targets():
            return ("No trace in the window has an instrument InvOLS, so no "
                    "value can be applied.")
        if self._method == "own":
            if any(r.source == "ensemble" for r in self._rows):
                return None
            return ("Nothing to revert: every trace is already on its own fit. "
                    "Choose a mean to apply an ensemble value.")
        if selection is None:
            return "Draw a box on the plot first: click and drag over the traces to include."
        if self._applied:
            return "Already applied. Change the method, average, box or scope to apply again."
        if selection.summary.n:
            return None
        if selection.n_old:
            return ("No trace has an own fit made under the fit settings in "
                    "force now; they are left out until the queue visits them.")
        return "No trace has an own fit to take a mean of."

    def _update_apply(self, selection: _Selection | None) -> None:
        reason = self._block_reason(selection)
        count = _plural(len(self._targets()), "trace", "traces")
        self._apply_button.setText(f"Apply to all {count} in the window")
        self._apply_button.setEnabled(reason is None)
        self._apply_button.setToolTip(
            reason or f"Stores the value for all {count} in the window.")

    def _on_apply(self) -> None:
        try:
            summary, message, written = self._write()
        except store.InvolsApplyError as exc:
            self._reload()
            self._show_status(str(exc), emphasis=True)
            return
        self._reload(summary)
        self._show_status(message)
        self.applied.emit(written)

    def _write(self) -> tuple[Summary | None, str, int]:
        target_ids = tuple(r.file_id for r in self._targets())
        if self._method == "own":
            written = store.apply_own(target_ids, self._db)
            return None, f"{_plural(written, 'trace', 'traces')} returned to their own fits.", written
        selection = self._current_selection()
        box = self._plot.box() if self._method == "box" else None
        result = store.apply_ensemble(store.EnsembleRequest(
            selection.member_ids, target_ids, self._method, box,
            store.fit_settings_now(self._db), self._average), self._db)
        value = _nm_v(applied_value(result.summary, self._average))
        message = f"Applied {value} to {_plural(result.n_written, 'trace', 'traces')}."
        return result.summary, message, result.n_written

    # -- selecting and opening a trace -----------------------------------

    def _on_point_selected(self, index: int) -> None:
        if 0 <= index < len(self._rows):
            self._selected_id = self._rows[index].file_id
            self._update_selection()

    def _on_point_opened(self, index: int) -> None:
        self._on_point_selected(index)
        self._on_open_clicked()

    def _selected_row(self) -> store.TraceRow | None:
        return next((r for r in self._rows if r.file_id == self._selected_id), None)

    def _update_selection(self) -> None:
        row = self._selected_row()
        source = "" if row is None else store.source_labels(
            [row.file_id], self._db)[row.file_id]
        self._selection_label.setText(_selection_text(row, source))
        if row is None:
            reason = "Select a trace first: click a point on the plot."
        elif self._open_trace is None:
            reason = "No curve view is available to open this trace in."
        else:
            reason = None
        self._open_button.setEnabled(reason is None)
        self._open_button.setToolTip(reason or _OPEN_TIP)

    def _on_open_clicked(self) -> None:
        row = self._selected_row()
        if row is not None and self._open_trace is not None:
            self._open_trace(row.path)
