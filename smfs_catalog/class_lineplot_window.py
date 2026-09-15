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
# Non-events have no well-defined rupture-force × contour-length summary, so
# one retract deflection-vs-piezo trace is shown at a time, beside the one
# aggregate a non-event does support: the distribution of its deflection.
#
# Scoped to queue ∩ non-event, pre-filled from the DB.  Curves are
# loaded lazily on selection — one at a time, never the whole cohort.  The
# population histograms cost no curve reads at all: they are summed from the
# stored per-file rows, which import writes and which this window also writes
# for any curve it shows that does not have one yet.
#
# Single-click a row -> plot inline.  Double-click -> open in the full raw
# curve viewer via view_file_requested.

from __future__ import annotations

from pathlib import Path

import numpy as np
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
from . import event_processor as _ep
# This window drives its OWN QTimer over its own curve list — it is not the
# analysis worker's playhead, so it shares navigator_bar's slider maths but
# deliberately not its NavigatorBar.
from .navigator_bar import (
    slider_to_interval_ms, rate_to_slider, rate_label,
    SLIDER_MIN, SLIDER_MAX, DEFAULT_RATE_HZ,
)
from .navigation import build_go_to_row

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
        self._cohort_paths: list[str] = []
        self._event_paths:  list[str] = []
        self._cohort_sig = None
        # The non-event total as last summed, kept so a row this window stores
        # can be added to it without re-summing the cohort.
        self._cohort_counts = np.zeros(_ep.DEFL_HIST_BINS, dtype=np.uint64)
        self._cohort_below  = 0
        self._cohort_above  = 0
        self._n_binned      = 0
        self._n_events      = 0

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
            "but had no validated rupture event. Beside each trace is the "
            "distribution of its retract deflection, against the same "
            "distribution for the whole negative cohort and for the events — "
            "a negative population shaped like the positive one is evidence "
            "the criteria are not separating on deflection. No rupture-force "
            "or contour-length summary is shown: a non-event has neither."
        )
        purpose.setWordWrap(True)
        purpose.setStyleSheet(style.qss_inset())
        root.addWidget(purpose)

        root.addLayout(self._build_nav_row(font))

        # ── Outer split: trace + deflection histogram | file list ─────────────────
        outer = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(outer, stretch=1)

        # One GraphicsLayoutWidget holding both plots, the way linked plots are
        # built elsewhere: plots in one layout share its row geometry, so a
        # deflection sits at the same height in the trace and the histogram.
        # The filename is a label spanning both columns rather than the trace's
        # title, which would shorten the trace's view alone.
        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground(style.SURFACE)
        self._title_lbl = self._glw.addLabel(" ", row=0, col=0, colspan=2)

        self._plot = self._glw.addPlot(row=1, col=0)
        set_si_label(self._plot, "bottom", "Piezo",      _quant.NM)
        set_si_label(self._plot, "left",   "Deflection", _quant.NM)
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._curve = sample_marks.trace(self._plot, color=style.SIG_RETRACT)

        self._build_hist_plot()
        self._glw.ci.layout.setColumnStretchFactor(0, 5)
        self._glw.ci.layout.setColumnStretchFactor(1, 2)
        outer.addWidget(self._glw)

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

        # Under both plots, full width: under either column alone a label would
        # shorten that plot and break the alignment the shared layout gives.
        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setFont(font)
        root.addWidget(self._status_lbl)
        self._hist_lbl = QLabel("")
        self._hist_lbl.setWordWrap(True)
        self._hist_lbl.setFont(font)
        root.addWidget(self._hist_lbl)

        self._populate()

    # ── Deflection histogram panel ────────────────────────────────────────────

    def _build_hist_plot(self) -> None:
        """Deflection distribution beside the trace, on the trace's own Y axis.

        Answers the question the single trace cannot: is this curve flat, and
        is the negative population as a whole flat? The event cohort is drawn
        alongside because that comparison is the actual diagnostic — two
        populations with the same deflection distribution mean the criteria
        separating them are not separating on deflection at all.

        Everything is plotted as a FRACTION of each series' own samples. One
        curve against thousands cannot share a count axis, and the fraction is
        the comparable quantity anyway.

        The fraction axis is LOGARITHMIC. A non-event is mostly flat baseline,
        so on a linear axis one bin holds nearly every sample and the whole
        distribution renders as a single spike — the tails, which are the only
        place structure could show, are drawn one pixel wide. The question this
        panel answers lives four or five decades below the peak.

        Steps rather than filled bars: three overlapping series are unreadable
        filled, and at 0.25 nm bins a step outline IS the histogram.

        The Y axis is linked to the trace, so a bin lines up with the deflection
        it describes and both panels zoom together.
        """
        self._hist_plot = self._glw.addPlot(row=1, col=1)
        self._hist_plot.showGrid(x=True, y=True, alpha=0.2)
        self._hist_plot.setLabel("bottom", "Fraction of samples")
        self._hist_plot.hideAxis("left")
        self._hist_plot.setLogMode(x=True, y=False)
        # A log axis carries its own decades; an SI prefix on top of them would
        # relabel 10^-3 as "1" and hide the factor in the axis title.
        self._hist_plot.getAxis("bottom").enableAutoSIPrefix(False)
        self._hist_plot.setYLink(self._plot)

        # Staircase corners: every bin edge twice, so a count spans its bin
        # rather than being drawn at a point in the middle of it.
        edges = _ep.defl_bin_edges()
        self._hist_steps = np.repeat(edges, 2)[1:-1]

        # Added back to front: the populations are context, the current curve is
        # the subject and must stay legible on top of them.
        self._hist_events = self._hist_plot.plot(
            [], [], pen=pg.mkPen(style.SIG_APPROACH, width=1))
        self._hist_cohort = self._hist_plot.plot(
            [], [], pen=pg.mkPen(150, 150, 150, width=4))
        self._hist_curve = self._hist_plot.plot(
            [], [], pen=pg.mkPen(style.SIG_RETRACT, width=1))

    def _draw_hist(self, item, counts) -> None:
        """Draw one series as a staircase of per-bin fractions.

        Empty bins become NaN rather than zero: on a log axis zero has no
        position, and a gap is the honest picture of a bin nothing landed in.
        """
        total = float(np.asarray(counts).sum())
        if total <= 0:
            item.setData([], [])
            return
        frac = np.asarray(counts, dtype=float) / total
        frac[frac <= 0] = np.nan
        item.setData(np.repeat(frac, 2), self._hist_steps)

    def _load_or_compute_histogram(
        self, path: str, defl
    ) -> tuple["np.ndarray", int, int, bool]:
        """(counts, n_below, n_above, stored_now) for one curve, using the
        deflection already loaded for the trace. A file missing from the
        catalog is binned for display and not stored."""
        file_id = _db.get_file_id(path, self._db_path)
        if file_id is None:
            return (*_ep.compute_deflection_histogram(defl), False)
        return _db.get_or_store_deflection_histogram(
            file_id, defl, self._db_path)

    def _set_curve_histogram(self, path: str, defl) -> None:
        counts, below, above, stored_now = self._load_or_compute_histogram(
            path, defl)
        self._draw_hist(self._hist_curve, counts)
        if stored_now:
            # Every curve this window shows is a cohort member, and a row stored
            # just now was not there when the cohort was summed, so it adds
            # without double-counting.
            self._cohort_counts += counts
            self._cohort_below  += below
            self._cohort_above  += above
            self._n_binned      += 1
            self._draw_hist(self._hist_cohort, self._cohort_counts)
            self._update_hist_caption()

    def _clear_curve_histogram(self) -> None:
        self._hist_curve.setData([], [])

    def _refresh_population_histograms(self) -> None:
        """Sum the stored per-curve rows for both cohorts.

        Skipped when neither cohort has changed since the last sum: refresh()
        fires on every batch the worker classifies, and summing a full queue
        measures ~200 ms on the GUI thread — the one cost this panel could
        plausibly impose.

        Cohort membership is not the only thing that can change the answer,
        though: a backfill adds rows for curves already in the cohort. Showing
        the window clears the cache for exactly that reason.
        """
        sig = (tuple(self._cohort_paths), tuple(self._event_paths))
        if sig == self._cohort_sig:
            return
        self._cohort_sig = sig

        key   = _ep.defl_grid_params()
        bins  = _ep.DEFL_HIST_BINS
        conn  = _db.get_connection(self._db_path)
        try:
            cohort, below, above, n_binned = _db.sum_deflection_histograms(
                self._cohort_paths, key, bins, self._db_path, conn=conn)
            events, _eb, _ea, n_events = _db.sum_deflection_histograms(
                self._event_paths, key, bins, self._db_path, conn=conn)
        finally:
            conn.close()

        self._cohort_counts = cohort
        self._cohort_below  = below
        self._cohort_above  = above
        self._n_binned      = n_binned
        self._n_events      = n_events
        self._draw_hist(self._hist_cohort, cohort)
        self._draw_hist(self._hist_events, events)
        self._update_hist_caption()

    def _update_hist_caption(self) -> None:
        self._hist_lbl.setText(self._hist_caption(
            self._n_binned, self._cohort_below, self._cohort_above,
            self._n_events))

    def _hist_caption(
        self, n_binned: int, below: int, above: int, n_events: int
    ) -> str:
        """What the bars are, and what is missing from them.

        A population drawn from only part of its cohort is not the population,
        so the shortfall is stated rather than left to look complete.
        """
        n_cohort = len(self._cohort_paths)
        if n_cohort == 0:
            return ""
        parts = [f"Non-events: {n_binned:,} of {n_cohort:,} binned"]
        if n_binned < n_cohort:
            parts.append(
                f"{n_cohort - n_binned:,} not yet binned — a curve is binned "
                f"whenever it is read: on import, analysis, re-check, or "
                f"viewing here")
        if n_events:
            parts.append(f"events: {n_events:,}")
        if below or above:
            lo, hi = _ep.DEFL_HIST_RANGE
            parts.append(
                f"{below + above:,} samples outside {lo:g}–{hi:g} nm")
        return " · ".join(parts)

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
        nav.addSpacing(16)
        go_row, self._go_btns = build_go_to_row(
            self, self._current_path, decomp=True)
        nav.addLayout(go_row)
        nav.addSpacing(16)
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
        lo, hi = _ep.DEFL_HIST_RANGE
        return {
            "window":            "class_lineplot",
            "classification":    self._classification,
            "cohort":            "queue ∩ classification",
            "defl_hist_bins":    _ep.DEFL_HIST_BINS,
            "defl_hist_min_nm":  lo,
            "defl_hist_max_nm":  hi,
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

    # ── Population ─────────────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Called by the dashboard whenever the analysis queue changes."""
        self._populate()

    def _populate(self) -> None:
        # Preserve the current curve across refreshes (e.g. a refresh triggered
        # by enqueuing a file from a double-click) so the user stays where they
        # left off instead of snapping back to the top.
        prev_path = self._current_path()

        # One queue read serves both cohorts: this window lists the negatives,
        # and the histogram panel draws the positives beside them for contrast.
        queue = _db.list_queue(self._db_path)
        self._rows = [r for r in queue if r["event"] == self._classification]
        self._cohort_paths = [r["path"] for r in self._rows]
        self._event_paths  = [r["path"] for r in queue if r["event"] == "event"]
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

        self._refresh_population_histograms()

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

    def _current_path(self) -> str | None:
        """The path of the curve on screen, or None when the cohort is empty."""
        if 0 <= self._index < len(self._rows):
            return self._rows[self._index]["path"]
        return None

    def _set_navigation_enabled(self, enabled: bool) -> None:
        """Keep an empty cohort from presenting controls that cannot act."""
        self._btn_auto_rev.setEnabled(enabled)
        self._btn_auto_fwd.setEnabled(enabled)
        for btn in self._go_btns:
            btn.setEnabled(enabled)
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
            self._title_lbl.setText(name)
            self._status_lbl.setText(f"Could not load this curve: {exc}")
            return
        self._curve.setData(curve.piezo_retr, curve.defl_retr)
        self._set_curve_histogram(path, curve.defl_retr)
        self._title_lbl.setText(name)

    def _clear_plot(self) -> None:
        self._curve.setData([], [])
        self._title_lbl.setText(" ")
        self._clear_curve_histogram()

    def showEvent(self, event) -> None:
        """Re-sum the populations on every show.

        Curves already in the cohort can acquire histograms after the fact,
        when a re-check measures files imported before the store existed. That
        does not change cohort membership, so nothing else would notice.
        """
        super().showEvent(event)
        self._cohort_sig = None
        self._refresh_population_histograms()

    def _on_double_click(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.view_file_requested.emit(path)
