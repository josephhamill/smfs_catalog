# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/class_lineplot_window.py
#
# ClassLinePlotWindow — inspection window for the stage-1 non-event cohort.
# Non-events have no well-defined rupture-force × contour-length summary.
# This window makes the negative cohort inspectable without inventing an
# aggregate quantity: one retract deflection-vs-piezo trace is shown at a time.
#
# Scoped to queue ∩ non-event, pre-filled from the DB.  Curves are
# loaded lazily on selection — one at a time, never the whole cohort.
#
# Single-click a row -> plot inline.  Double-click -> open in the full raw
# curve viewer via view_file_requested.

from __future__ import annotations

from pathlib import Path

import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from . import sample_marks
from . import style
from .widgets import SampleMarksToggle
from . import db as _db
from . import export_utils as _export
from .export_utils import slug as _slug
from .curve_loader import LoadError, load_force_curve
from .qt_utils import _make_session_header, set_si_label, fit_on_screen
from . import quantities as _quant
# This window drives its OWN QTimer over its own curve list — it is not the
# analysis worker's playhead, so it shares navigator_bar's slider maths but
# deliberately not its NavigatorBar.
from .navigator_bar import (
    slider_to_interval_ms, rate_to_slider, rate_label,
    SLIDER_MIN, SLIDER_MAX, DEFAULT_RATE_HZ,
)

_LIST_QSS = style.LIST_QSS


class ClassLinePlotWindow(QMainWindow):
    """
    Per-curve inspection window for the stage-1 negative cohort.

    event : 'non_event' (the stage-1 negative cohort)
    """

    # Emitted on row double-click; the dashboard connects this to its singleton
    # worker viewer.
    view_file_requested = pyqtSignal(str)

    def __init__(
        self,
        event: str,
        db_path:        str | None  = None,
        session_info:   dict | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        if event != "non_event":
            raise ValueError(
                "ClassLinePlotWindow only supports the 'non_event' verdict"
            )
        self._classification = event
        self._db_path        = db_path or _db.DEFAULT_DB_PATH
        self._session_info   = session_info
        self._rows: list[dict] = []
        self._index          = -1
        self._auto_dir       = 1

        self._nav_timer = QTimer(self)
        self._nav_timer.timeout.connect(self._auto_step)

        self.setWindowFlag(Qt.WindowType.Window)
        self.setWindowTitle("SMFS — Non-events")
        fit_on_screen(self, 1100, 700)
        style.apply_plot_defaults()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(4)
        root.setContentsMargins(8, 6, 8, 6)

        hdr = _make_session_header(session_info)
        if hdr is not None:
            root.addWidget(hdr)

        self._count_lbl = QLabel("")
        font = style.font(self._count_lbl.font(), size_pt=style.FONT_SMALL_PT)
        self._count_lbl.setFont(font)
        root.addWidget(self._count_lbl)

        purpose = QLabel(
            "Classifier-negative audit: inspect every curve that was analysed "
            "but had no validated rupture event. No aggregate distribution is "
            "shown because no scientifically justified non-event summary has "
            "been defined."
        )
        purpose.setWordWrap(True)
        purpose.setStyleSheet(style.qss_inset())
        root.addWidget(purpose)

        root.addLayout(self._build_nav_row(font))

        # ── Outer split: current retract trace | file list ─────────────────────────
        outer = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(outer, stretch=1)

        plot_panel = QWidget()
        plot_root = QVBoxLayout(plot_panel)
        plot_root.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(plot_panel)

        # This is the complete scientific surface. Aggregate panels belong
        # here only after their interpretation has been established.
        self._plot = pg.PlotWidget()
        set_si_label(self._plot, "bottom", "Piezo",      _quant.NM)
        set_si_label(self._plot, "left",   "Deflection", _quant.NM)
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._curve = sample_marks.trace(self._plot, color=style.SIG_RETRACT)
        plot_root.addWidget(self._plot, stretch=1)
        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setFont(font)
        plot_root.addWidget(self._status_lbl)

        self._list = QListWidget()
        self._list.setMinimumWidth(140)
        self._list.setMaximumWidth(240)
        self._list.setStyleSheet(_LIST_QSS)
        self._list.currentRowChanged.connect(self._on_row_changed)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        outer.addWidget(self._list)
        outer.setStretchFactor(0, 1)
        outer.setStretchFactor(1, 0)
        outer.setSizes([860, 240])

        self._populate()

    # ── Navigation row (copied from WlcViewWindow) ────────────────────────────

    def _build_nav_row(self, font) -> QHBoxLayout:
        nav = QHBoxLayout()

        self._btn_auto_rev = QPushButton("◀◀")
        self._btn_auto_rev.setCheckable(True)
        self._btn_auto_rev.setFixedWidth(44)
        self._btn_auto_rev.clicked.connect(lambda c: self._toggle_auto(-1, c))

        self._prev_btn = QPushButton("◀ Prev")
        self._next_btn = QPushButton("Next ▶")
        self._prev_btn.clicked.connect(self._go_prev)
        self._next_btn.clicked.connect(self._go_next)

        self._btn_auto_fwd = QPushButton("▶▶")
        self._btn_auto_fwd.setCheckable(True)
        self._btn_auto_fwd.setFixedWidth(44)
        self._btn_auto_fwd.clicked.connect(lambda c: self._toggle_auto(1, c))

        lbl_slow = QLabel("Slow"); lbl_slow.setFont(font)
        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(SLIDER_MIN, SLIDER_MAX)
        self._speed_slider.setValue(rate_to_slider(DEFAULT_RATE_HZ))
        self._speed_slider.setFixedWidth(100)
        self._speed_slider.valueChanged.connect(self._on_speed_change)
        lbl_fast = QLabel("Fast"); lbl_fast.setFont(font)
        self._speed_label = QLabel(rate_label(self._speed_slider.value()))
        self._speed_label.setFixedWidth(64); self._speed_label.setFont(font)

        self._counter = QLabel(); self._counter.setFont(font)
        self._fname_label = QLabel(); self._fname_label.setFont(font)

        nav.addWidget(self._btn_auto_rev)
        nav.addWidget(self._prev_btn)
        nav.addWidget(self._next_btn)
        nav.addWidget(self._btn_auto_fwd)
        nav.addSpacing(12)
        nav.addWidget(lbl_slow)
        nav.addWidget(self._speed_slider)
        nav.addWidget(lbl_fast)
        nav.addWidget(self._speed_label)
        nav.addSpacing(16)
        nav.addWidget(self._counter)
        nav.addSpacing(16)
        nav.addWidget(self._fname_label)
        nav.addStretch()
        nav.addWidget(SampleMarksToggle())
        self._export_btn = QPushButton("Export list…")
        self._export_btn.setToolTip(
            "Write this cohort's file list to the export folder, with a "
            "manifest. Reloadable with the dashboard's Load Queue."
        )
        self._export_btn.clicked.connect(self._on_export)
        nav.addWidget(self._export_btn)
        return nav

    # ── Export ───────────────────────────────────────────────────────────────

    def export_provenance(self) -> dict:
        """This window's settings, for an export manifest — same protocol
        method as the other exporting windows."""
        return {
            "window":         "class_lineplot",
            "classification": self._classification,
            "cohort":         "queue ∩ classification",
        }

    def _on_export(self) -> None:
        """Export this cohort as a file list.

        This window plots one curve at a time from a class cohort; the thing
        worth taking out of it is WHICH curves are in that cohort. The `path`
        column matches the rest of the app's exports, so the dashboard's Load
        Queue reads it straight back."""
        if not self._rows:
            QMessageBox.information(self, "Export", "No curves in this cohort.")
            return
        paths = [r["path"] for r in self._rows]
        with _export.export_group(
            self._db_path, f"cohort_{_slug(self._classification)}", [".csv"],
            kind="class_cohort",
        ) as g:
            g.contributing_files(paths)
            g.note_dict(self.export_provenance())
            g.note(n_curves=len(paths))
            g.table(".csv", ["path", "filename", "status"],
                    [(r["path"], r["filename"] or "", r["status"] or "")
                     for r in self._rows])
        QMessageBox.information(self, "Export", g.message())

    # ── Placeholder panel ─────────────────────────────────────────────────────

    # ── Population ─────────────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Called by the dashboard whenever the analysis queue changes."""
        self._populate()

    def _populate(self) -> None:
        # Preserve the current curve across refreshes (e.g. a refresh triggered
        # by enqueuing a file from a double-click) so the user stays where they
        # left off instead of snapping back to the top.
        prev_path = (self._rows[self._index]["path"]
                     if 0 <= self._index < len(self._rows) else None)

        self._rows = [r for r in _db.list_queue(self._db_path)
                      if r["event"] == self._classification]
        self._list.blockSignals(True)
        self._list.clear()
        for row in self._rows:
            item = QListWidgetItem(row["filename"] or row["path"])
            item.setData(Qt.ItemDataRole.UserRole, row["path"])
            self._list.addItem(item)
        self._list.blockSignals(False)
        n_rows = len(self._rows)
        self._count_lbl.setText(
            f"{n_rows} non-event curve{'s' if n_rows != 1 else ''} in queue"
        )

        if not self._rows:
            self._stop_auto()
            self._index = -1
            self._clear_plot()
            self._counter.setText("")
            self._fname_label.setText("")
            self._status_lbl.setText(
                "No analysed non-events are currently in the queue."
            )
            self._set_navigation_enabled(False)
            return

        self._set_navigation_enabled(True)

        new_index = next(
            (k for k, r in enumerate(self._rows) if r["path"] == prev_path), -1
        )
        if new_index >= 0:
            # Same curve still present — keep position, don't reload the plot.
            self._index = new_index
            self._show_current(reload=False)
        else:
            self._index = 0
            self._show_current()

    # ── Navigation ─────────────────────────────────────────────────────────────

    def _go_prev(self) -> None:
        self._stop_auto()
        if self._index > 0:
            self._index -= 1
            self._show_current()

    def _go_next(self) -> None:
        self._stop_auto()
        if self._index < len(self._rows) - 1:
            self._index += 1
            self._show_current()

    def _on_row_changed(self, row: int) -> None:
        # Fires only for genuine user clicks — programmatic moves block signals.
        if row < 0 or row >= len(self._rows):
            return
        self._stop_auto()
        self._index = row
        self._show_current()

    def _toggle_auto(self, direction: int, checked: bool) -> None:
        if not self._rows:
            self._stop_auto()
            return
        if checked:
            self._auto_dir = direction
            (self._btn_auto_rev if direction > 0 else self._btn_auto_fwd).setChecked(False)
            self._nav_timer.start(slider_to_interval_ms(self._speed_slider.value()))
        else:
            self._nav_timer.stop()

    def _stop_auto(self) -> None:
        self._nav_timer.stop()
        self._btn_auto_fwd.setChecked(False)
        self._btn_auto_rev.setChecked(False)

    def _auto_step(self) -> None:
        if self._auto_dir > 0:
            if self._index < len(self._rows) - 1:
                self._index += 1
                self._show_current()
            else:
                self._nav_timer.stop(); self._btn_auto_fwd.setChecked(False)
        else:
            if self._index > 0:
                self._index -= 1
                self._show_current()
            else:
                self._nav_timer.stop(); self._btn_auto_rev.setChecked(False)

    def _on_speed_change(self, value: int) -> None:
        self._speed_label.setText(rate_label(value))
        if self._nav_timer.isActive():
            self._nav_timer.setInterval(slider_to_interval_ms(value))

    def _set_navigation_enabled(self, enabled: bool) -> None:
        """Keep an empty cohort from presenting controls that cannot act."""
        self._btn_auto_rev.setEnabled(enabled)
        self._btn_auto_fwd.setEnabled(enabled)
        self._prev_btn.setEnabled(enabled and self._index > 0)
        self._next_btn.setEnabled(enabled and self._index < len(self._rows) - 1)
        self._speed_slider.setEnabled(enabled)
        self._export_btn.setEnabled(enabled)

    # ── Display ────────────────────────────────────────────────────────────────

    def _show_current(self, reload: bool = True) -> None:
        n = len(self._rows)
        self._counter.setText(f"{self._index + 1} / {n}")
        self._prev_btn.setEnabled(self._index > 0)
        self._next_btn.setEnabled(self._index < n - 1)

        self._list.blockSignals(True)
        self._list.setCurrentRow(self._index)
        self._list.blockSignals(False)
        self._list.scrollTo(self._list.currentIndex())

        row  = self._rows[self._index]
        path = row["path"]
        name = row["filename"] or Path(path).name
        self._fname_label.setText(name)
        if not reload:
            return
        self._status_lbl.setText("")
        try:
            curve = load_force_curve(path)
        except LoadError as exc:
            self._clear_plot()
            self._plot.setTitle(name)
            self._status_lbl.setText(f"Could not load this curve: {exc}")
            return
        self._curve.setData(curve.piezo_retr, curve.defl_retr)
        self._plot.setTitle(name)

    def _clear_plot(self) -> None:
        self._curve.setData([], [])
        self._plot.setTitle("")

    def _on_double_click(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.view_file_requested.emit(path)
