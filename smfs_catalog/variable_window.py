# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/variable_window.py
#
# VariableStatsWindow — lightweight, DB-backed single-variable viewer.
#
# Opened from a queue-table column header. Shows one
# analysis variable across the current queue scope two ways:
#   left  : drift-vs-time scatter  (value vs measured_date)
#   right : 1-D histogram          (value on Y, linked to the scatter's Y)
#
# Values use the shared variables router (analysis results, live segment
# summaries, or file metadata). Times prefer files.measured_at and fall back to
# measured_date for older scans. No curve files are loaded.
#
# Thresholds are editable in-place: each bound has an enable
# checkbox + numeric spinbox + a draggable line shown on both plots.  An
# unchecked bound is *absent* (None in the DB), not infinite — so a variable we
# don't gate simply has no constraint.  "Apply thresholds" persists the bounds
# via db.set_threshold and recomputes the pass/fail scatter + split histogram.
# Pass/fail itself is never stored — it is always derived from value vs bounds.

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QSplitter,
    QVBoxLayout, QWidget,
)

from . import db as _db
from . import export_utils as _export
from . import histogram_binning as _hb
from . import quantities as _quant
from . import regression as _reg
from .export_utils import slug as _slug
from . import style
from . import clustering as _clustering
from .widgets import ClusterColourBar
from .qt_utils import _DateAxis, _make_session_header, set_si_label, fit_on_screen
from . import variables as _vars
from .roi_pipeline import SEG_SUMMARY_KEYS, read_segment_select

# Referencing is a property of the variable registry, so the plotted axis and
# persisted thresholds always name the same quantity.

_PEN_NONE     = pg.mkPen(None)
# Pass/fail by TONE, matching Explore Events — same concept, so the same
# encoding, rather than blue/red here and something else there.
_BRUSH_PASS   = pg.mkBrush(*style.rgba(style.INK_STRONG, 255))
_BRUSH_FAIL   = pg.mkBrush(*style.rgba(style.INK_FAINT, 255))
_BRUSH_PASS_H = pg.mkBrush(*style.rgba(style.INK_STRONG, 190))   # histogram pass fill
_BRUSH_FAIL_H = pg.mkBrush(*style.rgba(style.INK_FAINT, 190))    # histogram fail fill
_PEN_THRESH   = style.guide_pen(style.LM_THRESHOLD)
_PEN_SEL      = pg.mkPen(style.INK, width=2)    # selection ring on the timeseries
_LIST_QSS   = style.LIST_QSS
# The drift fit is a model, so it uses model weight over neutral data
# (style rule 2).  The band is the same hue, translucent — a fit and its own
# uncertainty are one object and must not read as two unrelated series.
_DRIFT_HUE    = style.series_line(0)
_PEN_DRIFT    = style.model_pen(_DRIFT_HUE)
_BRUSH_DRIFT  = style.band_brush(_DRIFT_HUE)


def _ts_to_date(ts: float) -> str:
    """Unix timestamp → 'YYYY-MM-DD HH:MM:SS' for export. The inverse of
    _date_to_ts below; the plotted arrays hold timestamps, but a date string
    is what belongs in a CSV someone else will read."""
    try:
        return datetime.datetime.fromtimestamp(float(ts)).isoformat(sep=" ",
                                                                    timespec="seconds")
    except (ValueError, OSError, OverflowError, TypeError):
        return ""


def _date_to_ts(d: str | None) -> float:
    """
    Acquisition datetime string → Unix timestamp; NaN if absent/unparseable.

    Accepts the full 'YYYY-MM-DD HH:MM:SS' (preferred, second resolution) and
    the day-only 'YYYY-MM-DD' fallback for files not yet re-scanned with time
    capture.
    """
    if not d:
        return float("nan")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(d.strip(), fmt).timestamp()
        except (ValueError, OSError, OverflowError):
            continue
    return float("nan")


class VariableStatsWindow(QMainWindow):
    """One analysis variable, DB-backed, over an arbitrary scope of paths."""

    # Emitted on double-click of a file (list or scatter point); the dashboard
    # connects this to its singleton worker viewer, like the class windows.
    view_file_requested = pyqtSignal(str)
    # Emitted after Apply persists new bounds, so the criteria gate can
    # recompute its hit/non-hit split live without polling the DB.
    thresholds_changed = pyqtSignal()

    def __init__(
        self,
        variable_key: str,
        label:        str,
        paths:        list[str],
        db_path:      str,
        session_info: dict | None = None,
        experimentalist: str | None = None,
    ) -> None:
        super().__init__()
        self.setWindowFlag(Qt.WindowType.Window)
        self._variable_key = variable_key
        self._label        = label
        self._db_path      = db_path
        # Thresholds are per-experimentalist. Because `paths` can span owners,
        # resolve whose bounds Apply will write and state that owner in the UI.
        self._experimentalist = (
            experimentalist
            if experimentalist is not None
            else _db.resolve_common_experimentalist(paths, db_path)
        )
        self._owner_label = (self._experimentalist
                             or "shared default — mixed/unknown owners")
        self.setWindowTitle(f"SMFS — variable — {label}")
        fit_on_screen(self, 1100, 600)
        # Per-plotted-point state — one entry per scatter point (≡ one file with
        # both a value and a timestamp).  Index space is shared by the timeseries
        # scatter, the side file-list, and the selection ring.
        self._plot_paths: list[str] = []
        self._plot_ts:    np.ndarray = np.empty(0)
        self._plot_vals:  np.ndarray = np.empty(0)
        self._plot_params: list[str] = []   # params_json per plotted point (rug colour)
        self._selected:   int | None = None

        # Param-set → tab10 colour, assigned on first sight — drives the rug strip
        # under the timeseries so a day's evolving parameter sets are legible at a
        # glance (a value analysed under different params gets a different colour).
        self._param_color_map: dict[str, tuple[int, int, int]] = {}
        self._param_color_idx = 0

        # Histogram source (all finite values) + threshold state.  _lo/_hi are
        # the *applied* bounds (None = inactive); the spinboxes hold the live,
        # not-yet-applied edit.  _updating guards the line ↔ spinbox sync loop.
        self._finite_v: np.ndarray = np.empty(0)
        self._finite_paths: list[str] = []
        self._lo: float | None = None
        self._hi: float | None = None
        self._seed_lo = 0.0
        self._seed_hi = 0.0
        self._updating = False
        self._fit_win = None   # DistFitWindow, opened on demand (single instance)

        # Drift fit over the plotted (value, acquisition-time) pairs.
        # Recomputed in _render from whatever is currently plotted, never
        # cached across a mode change: it describes the points on screen, and a
        # fit that outlived the data it was fitted to would be the same defect
        # class as a stale gate verdict.
        self._drift_fit:  _reg.LinearFit  | None = None
        self._drift_corr: _reg.Correlation | None = None

        self._raw_vals:   np.ndarray = np.empty(0)
        self._raw_ts:     np.ndarray = np.empty(0)
        self._raw_paths:  list[str] = []
        self._raw_params: list[str] = []
        self._n_missing_value = 0

        style.apply_plot_defaults()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(4)
        root.setContentsMargins(8, 6, 8, 6)

        hdr = _make_session_header(session_info)
        if hdr is not None:
            root.addWidget(hdr)

        # What this variable actually is, ON SCREEN rather than on hover.
        # This whole window is about one variable, so the description is the
        # subject rather than an aside — and a reader who opened it because
        # they did not recognise a column heading should not have to know to
        # hover something to find out.  Same register (variables.DESCRIPTIONS)
        # the queue header, the scatter axes and the criteria boxes read.
        _desc = _vars.describe(variable_key)
        if _desc:
            self._desc_label = QLabel(_desc)
            self._desc_label.setWordWrap(True)
            self._desc_label.setStyleSheet(style.qss_text(style.UI_MUTED))
            root.addWidget(self._desc_label)
        self._info = QLabel("")
        font = style.font(self._info.font(), size_pt=style.FONT_SMALL_PT)
        self._info.setFont(font)
        root.addWidget(self._info)

        # Readout for the user's current selection (file + value + date).
        self._sel_label = QLabel("")
        self._sel_label.setFont(font)
        root.addWidget(self._sel_label)

        # Options row — the drift-fit toggle's home.
        rel_row = QHBoxLayout()

        # Drift fit. On by default: the slope is the answer to "is this
        # variable moving over the session", and a diagnostic nobody switches on
        # is a diagnostic nobody reads — same call as wlc_view_window's
        # per-segment CI envelope.  A display toggle that informs, never gates.
        self._chk_drift = QCheckBox("Drift fit")
        self._chk_drift.setFont(font)
        self._chk_drift.setChecked(True)
        self._chk_drift.setToolTip(
            "OLS trend through the timeseries; its slope is the drift rate per "
            "hour. The 95% band describes the mean trend, not individual curves. "
            "It assumes independent residuals with constant variance; consecutive "
            "measurements can make the interval too narrow."
        )
        self._chk_drift.toggled.connect(self._on_drift_toggled)
        rel_row.addWidget(self._chk_drift)
        self._drift_lbl = QLabel("")
        self._drift_lbl.setFont(font)
        self._drift_lbl.setStyleSheet(style.qss_text())
        rel_row.addWidget(self._drift_lbl)
        rel_row.addStretch()
        root.addLayout(rel_row)

        # Cluster colouring projects the 2DH clustering onto this
        # variable's timeseries, so drift can be read per cluster.
        self._cluster_bar = ClusterColourBar()
        self._cluster_bar.changed.connect(self._render)
        root.addWidget(self._cluster_bar)

        # Outer split: the two plots (drift | histogram) on the left, the side
        # file-list on the right (matching EventSummaryWindow / the class windows).
        outer = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(outer, stretch=1)

        # Left column: the drift|hist plots, with a thin param-set rug beneath,
        # X-linked to the drift so it pans/zooms with the timeseries.
        left_col = QWidget()
        left_v   = QVBoxLayout(left_col)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.setSpacing(2)

        hsplit = QSplitter(Qt.Orientation.Horizontal)
        left_v.addWidget(hsplit, stretch=1)

        # ── Left: drift-vs-time scatter ───────────────────────────────────────
        self._drift = pg.PlotWidget(axisItems={"bottom": _DateAxis()})
        # The unit comes from the variable's own registration rather than
        # being baked into `label` by the caller.  si=False: the two bound
        # lines on this axis are dragged and typed into the bounds spin
        # boxes, which show plain units and cannot carry an SI prefix.
        set_si_label(self._drift, "left", style.mathify(label),
                     key=variable_key, si=False)
        self._drift.setLabel("bottom", "Acquisition time")   # no math to typeset
        self._drift.showGrid(x=True, y=True, alpha=0.2)
        self._sc_pass = pg.ScatterPlotItem(size=style.DOT_SIZE, pen=_PEN_NONE, brush=_BRUSH_PASS)
        self._sc_fail = pg.ScatterPlotItem(size=style.DOT_SIZE, pen=_PEN_NONE, brush=_BRUSH_FAIL)
        self._drift.addItem(self._sc_fail, ignoreBounds=True)
        self._drift.addItem(self._sc_pass)
        self._sc_pass.sigClicked.connect(self._on_scatter_clicked)
        self._sc_fail.sigClicked.connect(self._on_scatter_clicked)
        # Selection ring — a hollow circle around the user-selected point.
        self._sel_marker = pg.ScatterPlotItem(
            size=16, symbol="o", pen=_PEN_SEL, brush=pg.mkBrush(None),
        )
        self._sel_marker.hide()
        self._drift.addItem(self._sel_marker, ignoreBounds=True)
        # Drift fit and band ignore bounds: a fit is drawn
        # over the data and must never be what sets the view range — an
        # extrapolated band edge would silently rescale the axis away from the
        # points it describes.
        # Seeded with a degenerate pair, not left empty: FillBetweenItem builds
        # its path in __init__ from whatever the two curves hold at that moment.
        self._drift_band_lo = pg.PlotCurveItem(x=[0.0, 1.0], y=[0.0, 0.0],
                                               pen=pg.mkPen(None))
        self._drift_band_hi = pg.PlotCurveItem(x=[0.0, 1.0], y=[0.0, 0.0],
                                               pen=pg.mkPen(None))
        self._drift_band    = pg.FillBetweenItem(
            self._drift_band_lo, self._drift_band_hi, brush=_BRUSH_DRIFT)
        self._drift_line    = pg.PlotCurveItem(pen=_PEN_DRIFT)
        for it in (self._drift_band_lo, self._drift_band_hi,
                   self._drift_band, self._drift_line):
            self._drift.addItem(it, ignoreBounds=True)
        self._th_lo_d = pg.InfiniteLine(angle=0, movable=True, pen=_PEN_THRESH); self._th_lo_d.hide()
        self._th_hi_d = pg.InfiniteLine(angle=0, movable=True, pen=_PEN_THRESH); self._th_hi_d.hide()
        self._drift.addItem(self._th_lo_d); self._drift.addItem(self._th_hi_d)
        hsplit.addWidget(self._drift)

        # ── Right: transposed histogram (value on Y, linked to scatter) ───────
        self._hist = pg.PlotWidget()
        self._hist.setLabel("bottom", "Count")
        self._hist.getAxis("left").setStyle(showValues=False)
        self._hist.showGrid(x=True, y=True, alpha=0.2)
        self._hist.getViewBox().setYLink(self._drift.getViewBox())
        self._bars_fail = pg.BarGraphItem(x0=[], x1=[], y0=[], y1=[], pen=_PEN_NONE, brush=_BRUSH_FAIL_H)
        self._bars_pass = pg.BarGraphItem(x0=[], x1=[], y0=[], y1=[], pen=_PEN_NONE, brush=_BRUSH_PASS_H)
        self._hist.addItem(self._bars_fail)
        self._hist.addItem(self._bars_pass)
        self._th_lo_h = pg.InfiniteLine(angle=0, movable=True, pen=_PEN_THRESH); self._th_lo_h.hide()
        self._th_hi_h = pg.InfiniteLine(angle=0, movable=True, pen=_PEN_THRESH); self._th_hi_h.hide()
        self._hist.addItem(self._th_lo_h); self._hist.addItem(self._th_hi_h)
        hsplit.addWidget(self._hist)
        hsplit.setSizes([800, 300])

        # ── Param-set rug — one '+' per plotted point at its acquisition time ──
        self._rug = pg.PlotWidget()
        self._rug.setFixedHeight(18)
        self._rug.hideAxis("left")
        self._rug.hideAxis("bottom")
        self._rug.setMouseEnabled(x=False, y=False)
        self._rug.getViewBox().setXLink(self._drift.getViewBox())
        self._rug_scatter = pg.ScatterPlotItem(symbol="+", size=8, pen=_PEN_NONE)
        self._rug.addItem(self._rug_scatter)
        left_v.addWidget(self._rug)
        outer.addWidget(left_col)

        # ── Right: side file-list (one row per plotted point) ─────────────────
        self._list = QListWidget()
        self._list.setStyleSheet(_LIST_QSS)
        self._list.currentRowChanged.connect(self._on_list_row_changed)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        outer.addWidget(self._list)
        outer.setSizes([900, 200])

        # ── Threshold controls ───────────────────────────────────────────
        root.addLayout(self._build_threshold_row(font))

        self._load(paths)

    # ── Threshold controls ──────────────────────────────────────────────────────

    def _build_threshold_row(self, font) -> QHBoxLayout:
        """Per-bound enable checkbox + numeric entry + Apply, wired to the lines."""
        row = QHBoxLayout()
        row.addWidget(QLabel("Thresholds:"))

        self._chk_lo  = QCheckBox("Lower")
        self._spin_lo = QDoubleSpinBox()
        self._chk_hi  = QCheckBox("Upper")
        self._spin_hi = QDoubleSpinBox()
        # One widget serves every gate variable, so precision and step must
        # follow the registered quantity rather than a global constant.
        #
        # Seeded per-instance because a bound already in the thresholds table
        # may carry more digits than the quantity warrants. decimals_for widens to fit
        # rather than rounding: a box that displayed a rounded version of a
        # stored bound and wrote it back on the next touch would change the
        # threshold. audit_stored_precision reports these so
        # they can be tidied deliberately — rounding
        # `seg_n_segments <= 3.704446` to `<= 4` changes which curves are hits.
        # Declared precision here; _init_threshold_controls widens it once the
        # stored bounds are known.  No unit suffix — the variable's own label
        # above already carries it, and these boxes are narrow.
        for sp in (self._spin_lo, self._spin_hi):
            sp.setRange(-1e12, 1e12)
            _quant.configure_spinbox(sp, self._variable_key, suffix=False)
            # Minimum, not fixed: 120px could not fit a piezo bound at six
            # decimals ("-1234.567890"), so the box clipped its own value.
            sp.setMinimumWidth(120)
            sp.setEnabled(False)
        self._chk_lo.setFont(font)
        self._chk_hi.setFont(font)

        row.addWidget(self._chk_lo); row.addWidget(self._spin_lo)
        row.addSpacing(16)
        row.addWidget(self._chk_hi); row.addWidget(self._spin_hi)
        # Show the applied bounds beside the editing controls, using the same
        # quantity formatting so any discrepancy is visible in place.
        self._applied_lbl = QLabel("")
        self._applied_lbl.setStyleSheet(style.qss_text())
        row.addWidget(self._applied_lbl)

        row.addStretch()
        self._apply_btn = QPushButton("Apply thresholds")
        self._apply_btn.clicked.connect(self._apply_thresholds)
        row.addWidget(self._apply_btn)
        self._fit_btn = QPushButton("Fit histogram…")
        self._fit_btn.clicked.connect(self._on_fit_histogram)
        row.addWidget(self._fit_btn)
        self._export_btn = QPushButton("Export…")
        self._export_btn.setToolTip(
            "Write this variable's timeseries and histogram to the export "
            "folder, with a manifest recording the bounds, mode and file list."
        )
        self._export_btn.clicked.connect(self._on_export)
        row.addWidget(self._export_btn)

        # Two-way sync: a move on any line, or a spinbox edit, propagates to the
        # other two via _sync_lo/_sync_hi (the _updating guard breaks the loop).
        self._spin_lo.valueChanged.connect(self._sync_lo)
        self._spin_hi.valueChanged.connect(self._sync_hi)
        self._th_lo_d.sigPositionChanged.connect(lambda ln: self._sync_lo(ln.value()))
        self._th_lo_h.sigPositionChanged.connect(lambda ln: self._sync_lo(ln.value()))
        self._th_hi_d.sigPositionChanged.connect(lambda ln: self._sync_hi(ln.value()))
        self._th_hi_h.sigPositionChanged.connect(lambda ln: self._sync_hi(ln.value()))
        self._chk_lo.toggled.connect(self._on_chk_lo)
        self._chk_hi.toggled.connect(self._on_chk_hi)
        return row

    def _sync_lo(self, val: float) -> None:
        if self._updating:
            return
        self._updating = True
        self._spin_lo.setValue(val)
        self._th_lo_d.setValue(val)
        self._th_lo_h.setValue(val)
        self._updating = False

    def _sync_hi(self, val: float) -> None:
        if self._updating:
            return
        self._updating = True
        self._spin_hi.setValue(val)
        self._th_hi_d.setValue(val)
        self._th_hi_h.setValue(val)
        self._updating = False

    def _on_chk_lo(self, checked: bool) -> None:
        self._spin_lo.setEnabled(checked)
        self._th_lo_d.setVisible(checked)
        self._th_lo_h.setVisible(checked)
        if checked and not self._updating:
            self._sync_lo(self._spin_lo.value())

    def _on_chk_hi(self, checked: bool) -> None:
        self._spin_hi.setEnabled(checked)
        self._th_hi_d.setVisible(checked)
        self._th_hi_h.setVisible(checked)
        if checked and not self._updating:
            self._sync_hi(self._spin_hi.value())

    def _init_threshold_controls(self, lo: float | None, hi: float | None) -> None:
        """Seed the controls from the DB bounds (None = inactive bound)."""
        # Widen the boxes' precision, if needed, BEFORE any setValue: this is
        # the first moment the real stored bounds exist (they are None when the
        # row is built).  QDoubleSpinBox.value() returns its value rounded to
        # its decimals, so seeding a coarser box with a finer stored bound
        # would display one number while the database held another, and write
        # the displayed one back on the next touch. Seed values are also
        # passed: they are p5/p95 of the live data and become the bound if the
        # user ticks the box without typing.
        dec = _quant.decimals_for(self._variable_key, lo, hi,
                                  self._seed_lo, self._seed_hi)
        for sp in (self._spin_lo, self._spin_hi):
            _quant.configure_spinbox(sp, self._variable_key, decimals=dec,
                                     suffix=False)

        self._updating = True
        self._chk_lo.setChecked(lo is not None)
        self._spin_lo.setEnabled(lo is not None)
        self._spin_lo.setValue(lo if lo is not None else self._seed_lo)
        self._th_lo_d.setValue(self._spin_lo.value()); self._th_lo_h.setValue(self._spin_lo.value())
        self._th_lo_d.setVisible(lo is not None); self._th_lo_h.setVisible(lo is not None)

        self._chk_hi.setChecked(hi is not None)
        self._spin_hi.setEnabled(hi is not None)
        self._spin_hi.setValue(hi if hi is not None else self._seed_hi)
        self._th_hi_d.setValue(self._spin_hi.value()); self._th_hi_h.setValue(self._spin_hi.value())
        self._th_hi_d.setVisible(hi is not None); self._th_hi_h.setVisible(hi is not None)
        self._updating = False

        self._lo, self._hi = lo, hi
        self._refresh_applied_label()

    def _apply_thresholds(self) -> None:
        """Persist the current bounds and recompute the pass/fail split."""
        lo = self._spin_lo.value() if self._chk_lo.isChecked() else None
        hi = self._spin_hi.value() if self._chk_hi.isChecked() else None
        if lo is not None and hi is not None and lo > hi:
            QMessageBox.warning(
                self, "Invalid thresholds",
                "The lower threshold must not be greater than the upper threshold.",
            )
            return
        self._lo, self._hi = lo, hi
        _db.set_threshold(self._variable_key, self._lo, self._hi, self._label,
                          self._experimentalist, self._db_path)
        self._refresh_applied_label()
        self._render()
        self.thresholds_changed.emit()

    def _refresh_applied_label(self) -> None:
        """Echo the bounds now in force, in the quantity's own digits."""
        f = lambda v: _quant.format_value(self._variable_key, v, with_unit=True)
        if self._lo is None and self._hi is None:
            self._applied_lbl.setText("applied: none — passes all")
        elif self._lo is not None and self._hi is not None:
            self._applied_lbl.setText(f"applied: {f(self._lo)} ≤ x ≤ {f(self._hi)}")
        elif self._lo is not None:
            self._applied_lbl.setText(f"applied: x ≥ {f(self._lo)}")
        else:
            self._applied_lbl.setText(f"applied: x ≤ {f(self._hi)}")

    def _on_fit_histogram(self) -> None:
        """Open (or raise) a DistFitWindow on the currently-passing values."""
        fv = self._finite_v
        keep = self._pass_mask()
        pass_values = fv[keep]
        if pass_values.size < 5:
            QMessageBox.information(
                self, "Fit histogram",
                "At least five finite values within the applied thresholds are required.",
            )
            return
        if self._fit_win is not None and self._fit_win.isVisible():
            self._fit_win.raise_()
            self._fit_win.activateWindow()
            return
        from .dist_fit_window import DistFitWindow
        self._fit_win = DistFitWindow(
            self._display_label(), _quant.get(self._variable_key).shown_unit,
            pass_values, self._db_path,
            caption=self._provenance_caption(int(pass_values.size)),
            paths=[p for p, k in zip(self._finite_paths, keep) if k],
        )
        self._fit_win.show()

    def _pass_mask(self) -> np.ndarray:
        """Which of _finite_v currently pass the applied bounds."""
        fv = self._finite_v
        lo, hi = self._lo, self._hi
        keep = np.ones(fv.size, dtype=bool)
        if lo is not None: keep &= fv >= lo
        if hi is not None: keep &= fv <= hi
        return keep

    # ── Export ────────────────────────────────────────────────────────────────

    def _provenance_caption(self, n: int) -> str:
        bits = [f"{n} values", f"variable: {self._display_label()}"]
        if self._lo is not None or self._hi is not None:
            bits.append(f"bounds: {self._lo if self._lo is not None else '−∞'}"
                        f" … {self._hi if self._hi is not None else '+∞'}")
        return "  |  ".join(bits)

    def export_provenance(self) -> dict:
        """This window's settings, for an export manifest — same protocol
        method as the other exporting windows."""
        return {
            "window":        "variable_stats",
            "variable":      self._variable_key,
            "label":         self._display_label(),
            "bound_lo":      self._lo,
            "bound_hi":      self._hi,
            # Whether the band was on SCREEN, separately from whether it was
            # fitted — a figure and its CSV must agree about what was drawn,
            # the same reason mean_curve_window records which of +-1sigma/+-SE
            # was displayed.
            "drift_fit_shown": bool(self._chk_drift.isChecked()),
            **_clustering.provenance(self._plot_paths,
                                     self._cluster_bar.is_active()),
        }

    def _on_export(self) -> None:
        """Both of this window's figures, as data.

        _timeseries.csv is one row per file in scope, including files with a
        missing/non-finite value or date, so the complete bound outcome and
        drift plot can both be reconstructed. _histogram.csv is the binned
        distribution beside it. The
        pass/fail column is derived from the current bounds, never stored."""
        if self._raw_vals.size == 0:
            QMessageBox.information(self, "Export", "No values to export.")
            return
        keep = self._pass_mask()
        with _export.export_group(
            self._db_path, f"variable_{_slug(self._variable_key)}",
            ["_timeseries.csv", "_histogram.csv"], kind="variable_stats",
        ) as g:
            g.contributing_files(self._raw_paths)
            g.note_dict(self.export_provenance())
            g.note(n_values=int(self._finite_v.size),
                   n_pass=int(keep.sum()),
                   n_missing_value=self._n_missing_value,
                   n_plotted_with_date=len(self._plot_paths),
                   n_missing_date=self._n_no_date)

            # The drift fit and its band travel with the points. The band
            # is drawn, so its lo/hi ship as columns and its full 2x2 parameter
            # covariance ships in the manifest — the diagonal alone cannot
            # redraw it, exactly as for the mixture fits' band.
            g.note_dict(_reg.manifest_fields(self._drift_fit, self._drift_corr,
                                             x_is_time=True))
            fit = self._drift_fit
            if fit is not None:
                f_lo, f_hi = fit.band(self._plot_ts)
                f_val      = fit.predict(self._plot_ts)
            else:
                nan = np.full(self._plot_ts.shape, np.nan)
                f_lo = f_hi = f_val = nan

            # Written whether or not the colouring is switched on: a display
            # toggle must not decide what a data file contains, and the
            # clustering does not survive the session.
            _cl = _clustering.current()
            pass_by_path = {p: bool(k) for p, k in zip(self._finite_paths, keep)}
            missing_passes = self._lo is None and self._hi is None
            fit_by_path = {
                p: (float(fv), float(flo), float(fhi))
                for p, fv, flo, fhi in zip(self._plot_paths, f_val, f_lo, f_hi)
            }
            g.table(
                "_timeseries.csv",
                ["path", "measured_date", "value", "passes_bounds", "params",
                 "drift_fit", "drift_ci_lo", "drift_ci_hi", "cluster"],
                [(p, _ts_to_date(t), "" if not np.isfinite(v) else float(v),
                  pass_by_path.get(p, missing_passes), par,
                  *fit_by_path.get(p, ("", "", "")),
                  "" if _cl is None or _cl.label_for(p) is None
                  else _cl.label_for(p))
                 for p, t, v, par in zip(
                     self._raw_paths, self._raw_ts, self._raw_vals,
                     self._raw_params)],
            )

            bins = _hb.full_range_bins(self._finite_v)
            if bins is not None:
                counts = bins.count(self._finite_v)
                g.histogram("_histogram.csv", bins.edges, counts)
                g.note(n_bins=int(bins.n_bins),
                       bin_range=[float(bins.edges[0]), float(bins.edges[-1])])
            else:
                g.table(
                    "_histogram.csv", ["bin_left", "bin_right", "count"], []
                )
                g.note(n_bins=0)

        QMessageBox.information(self, "Export", g.message())

    # ── Data ──────────────────────────────────────────────────────────────────

    def _load(self, paths: list[str]) -> None:
        key = self._variable_key
        resolved = [_db.normalize_path(p) for p in paths]
        if key in SEG_SUMMARY_KEYS:
            # seg_* keys are read from each curve's event_map rather than
            # analysis_results. The rug identifies the producing event-map
            # parameters and the reporting policy used to project its scalar.
            provenance = _db.get_event_map_provenance_bulk(resolved, self._db_path)
            selection = read_segment_select(self._db_path)
            params_by_path = {
                rp: json.dumps({
                    "event_params": provenance.get(rp, {}).get("params_json"),
                    "code_version": provenance.get(rp, {}).get("code_version"),
                    "reported_segment": selection,
                }, sort_keys=True, separators=(",", ":"))
                for rp in resolved
            }
        else:
            # params_json for the rug only — the VALUE comes from variables.py
            # below, whichever of the three stores it lives in and whether or
            # not it is referenced to snap-off (variables.REFERENCED).
            prov = _db.get_derived_results_bulk_latest(
                resolved, [_vars.provenance_key(key)], self._db_path)
            params_by_path = {
                rp: (d.get(_vars.provenance_key(key)) or (None, ""))[1]
                for rp, d in prov.items()
            }
        values = _vars.values(resolved, [key], self._db_path)
        dates  = _db.get_measured_datetimes(resolved, self._db_path)

        ts_list, val_list, path_list, params_list = [], [], [], []
        for rp in resolved:
            v = values.get(rp, {}).get(key)
            val_list.append(float("nan") if v is None else float(v))
            ts_list.append(_date_to_ts(dates.get(rp)))
            path_list.append(rp)
            params_list.append(params_by_path.get(rp, ""))

        self._raw_vals   = np.asarray(val_list, dtype=float)
        self._raw_ts     = np.asarray(ts_list,  dtype=float)
        self._raw_paths  = path_list
        self._raw_params = params_list

        self._recompute_display(seed_thresholds=True)

    def _display_label(self) -> str:
        return self._label

    def _recompute_display(self, seed_thresholds: bool) -> None:
        """
        Rebuild every plotted/finite array from the cached raw arrays.
        """
        vals = self._raw_vals
        ts   = self._raw_ts

        fin = np.isfinite(vals)
        self._n_missing_value = int((~fin).sum())
        self._finite_v = vals[fin]
        # Same index space as _finite_v — the curves behind the histogram, so
        # an export (and the fit window opened from here) can name its data.
        self._finite_paths = [self._raw_paths[i] for i in np.where(fin)[0]]

        # Plotted points: those with BOTH a value and a date.  They define the
        # shared index space for the scatter, the side list, and the ring.
        plot_m = fin & np.isfinite(ts)
        pm_idx = np.where(plot_m)[0]
        self._plot_paths  = [self._raw_paths[i] for i in pm_idx]
        self._plot_ts     = ts[plot_m]
        self._plot_vals   = vals[plot_m]
        self._plot_params = [self._raw_params[i] for i in pm_idx]
        self._n_no_date   = int((fin & ~np.isfinite(ts)).sum())

        set_si_label(self._drift, "left", style.mathify(self._display_label()),
                     key=self._variable_key, si=False)
        self._populate_list()
        self._update_rug()

        # Seed default bound positions (5th / 95th percentile) so a freshly
        # enabled bound lands somewhere grabbable — not persisted until Apply.
        if seed_thresholds and self._finite_v.size:
            self._seed_lo = float(np.percentile(self._finite_v, 5))
            self._seed_hi = float(np.percentile(self._finite_v, 95))
            # Arrow-key steps come from the quantity registry; dragging and
            # typing remain available for larger changes.

        if seed_thresholds:
            row = _db.get_threshold(self._variable_key, self._experimentalist, self._db_path)
            lo = row["lower_bound"] if row is not None else None
            hi = row["upper_bound"] if row is not None else None
            self._init_threshold_controls(lo, hi)

        self._render()

    # ── Pass/fail rendering (re-run on Apply) ───────────────────────────────────

    def _render(self) -> None:
        """Recompute the pass/fail split from the applied bounds and redraw."""
        lo, hi = self._lo, self._hi

        # Scatter — split the plotted points.  Per-point `data` carries the local
        # index back into self._plot_* so a scatter click maps to the list row.
        pv = self._plot_vals
        keep = np.ones(pv.size, dtype=bool)
        if lo is not None: keep &= pv >= lo
        if hi is not None: keep &= pv <= hi
        pass_local = np.where(keep)[0]
        fail_local = np.where(~keep)[0]
        self._cluster_bar.refresh(self._plot_paths)
        if self._cluster_bar.is_active():
            # Cluster hue replaces the pass/fail TONE. The pass/fail split
            # itself is untouched — a failing curve keeps failing, it just
            # shows which cluster it is in. Unlabelled curves keep the faint
            # tone rather than borrowing a cluster's hue.
            cl = _clustering.current()

            def _spots(local):
                out = []
                for i in local:
                    lbl = cl.label_for(self._plot_paths[i]) if cl else None
                    brush = (style.scatter_brush(style.series_labeled(lbl))
                             if lbl is not None else _BRUSH_FAIL)
                    out.append({"pos": (float(self._plot_ts[i]), float(pv[i])),
                                "data": int(i), "brush": brush,
                                "pen": _PEN_NONE})
                return out
            self._sc_pass.setData(_spots(pass_local))
            self._sc_fail.setData(_spots(fail_local))
        else:
            self._sc_pass.setData(self._plot_ts[pass_local], pv[pass_local], data=pass_local.tolist())
            self._sc_fail.setData(self._plot_ts[fail_local], pv[fail_local], data=fail_local.tolist())

        self._render_drift_fit()

        # Histogram: pass/fail bars over robust shared bin edges. The
        # geometry comes from histogram_binning: edges span the 1st–99th pct
        # (so a narrow bulk resolves); outliers remain in the scatter but are
        # omitted from the bars and reported in the summary below.
        # Pass/fail split shares the one geometry; the same call reproduces the
        # axes at report time from the cached values.
        fv = self._finite_v
        bins = _hb.robust_bins(fv) if fv.size else None
        if bins is not None:
            fv_keep = np.ones(fv.size, dtype=bool)
            if lo is not None: fv_keep &= fv >= lo
            if hi is not None: fv_keep &= fv <= hi
            cp = bins.count(fv[fv_keep])
            cf = bins.count(fv[~fv_keep])
            y0, y1 = bins.edges[:-1], bins.edges[1:]
            # Stacked end-to-end (pass 0→cp, fail cp→cp+cf) so both colors stay
            # visible and the combined bar still reads as the bin's full count.
            self._bars_pass.setOpts(x0=np.zeros(len(cp)), x1=cp, y0=y0, y1=y1)
            self._bars_fail.setOpts(x0=cp, x1=cp + cf, y0=y0, y1=y1)
        else:
            self._bars_pass.setOpts(x0=[], x1=[], y0=[], y1=[])
            self._bars_fail.setOpts(x0=[], x1=[], y0=[], y1=[])

        parts = [f"Owner: {self._owner_label}", self._display_label(),
                 f"{int(fv.size)} values"]
        if lo is not None or hi is not None:
            within = int(np.count_nonzero(
                (fv >= lo if lo is not None else True) &
                (fv <= hi if hi is not None else True)
            )) if fv.size else 0
            parts.append(f"{within} within thresholds")
        if fv.size:
            parts.append(
                f"median {_quant.format_value(self._variable_key, np.median(fv), with_unit=True)}"
            )
        if bins is not None and bins.n_out_of_range:
            parts.append(
                f"{bins.n_out_of_range} beyond 1–99 pct range "
                f"(in scatter, not binned)"
            )
        if self._n_no_date:
            parts.append(f"{self._n_no_date} without a date (histogram only)")
        if self._n_missing_value:
            parts.append(
                f"{self._n_missing_value} missing/non-finite "
                f"(fails when used as a bounded criterion)"
            )
        self._info.setText("   —   ".join(parts))

    # ── Drift fit ────────────────────────────────────────────────────────────────────

    def _on_drift_toggled(self, _checked: bool) -> None:
        """Show/hide only — the fit itself is recomputed either way, so the
        export and the readout stay available without the band on screen."""
        self._render_drift_fit()

    def _render_drift_fit(self) -> None:
        """Refit the drift line over the plotted points and redraw it.

        Fitted over EVERY plotted point, pass and fail alike. The bounds are a
        criteria question — which curves count as hits — and drift is a
        question about the instrument over the session; letting a threshold
        decide which points the trend line sees would make the slope a function
        of a gate that has nothing to do with it, and would flatten real drift
        by construction (the bounds cut exactly the excursions drift produces).
        """
        ts, vals = self._plot_ts, self._plot_vals
        self._drift_fit  = _reg.linear_fit(ts, vals)
        self._drift_corr = _reg.correlate(ts, vals, method="spearman")

        fit  = self._drift_fit
        show = self._chk_drift.isChecked() and fit is not None
        for it in (self._drift_line, self._drift_band):
            it.setVisible(show)
        if not show:
            self._drift_lbl.setText(
                "" if fit is not None else "drift fit: too few dated points"
            )
            return

        # Drawn across the observed time span only — a band is a claim about
        # where the trend is, and extending it past the last curve would be
        # extrapolation the data does not support.
        xs = np.linspace(float(np.min(ts)), float(np.max(ts)), 200)
        lo, hi = fit.band(xs)
        self._drift_line.setData(xs, fit.predict(xs))
        self._drift_band_lo.setData(xs, lo)
        self._drift_band_hi.setData(xs, hi)
        self._drift_lbl.setText(self._drift_summary())

    def _drift_summary(self) -> str:
        """'slope ± CI per hour', plus rho — the readout and the caption both."""
        fit = self._drift_fit
        if fit is None:
            return "drift fit: too few dated points"
        slope, lo, hi = _reg.per_hour(fit)
        key = self._variable_key
        unit = _quant.get(key).shown_unit
        per  = f" {unit}/h" if unit else " /h"
        f = lambda v: _quant.format_value(key, v)
        # An interval containing zero means the data does not establish a
        # direction.  Say that in words rather than leaving a reader to notice
        # the sign change across the bracket themselves.
        verdict = "" if (lo > 0 or hi < 0) else "  (consistent with no drift)"
        bits = [f"drift {f(slope)}{per}  [{f(lo)}, {f(hi)}] {fit.pct:.0f}% CI{verdict}"]
        if self._drift_corr is not None:
            c = self._drift_corr
            bits.append(f"Spearman ρ {c.rho:+.2f} (n={c.n})")
        return "   —   ".join(bits)

    # ── Param-set rug ────────────────────────────────────────────────────────────

    def _color_for_params(self, params_json: str) -> tuple[int, int, int]:
        """tab10 RGB assigned to this params_json, allocated on first sight."""
        if params_json not in self._param_color_map:
            self._param_color_map[params_json] = style.rgba(
                style.series_labeled(self._param_color_idx))[:3]
            self._param_color_idx += 1
        return self._param_color_map[params_json]

    def _update_rug(self) -> None:
        """
        Rebuild the rug: one '+' per plotted point at its acquisition time,
        coloured by the param set its value was computed under.  Distinct colours
        make an evolving-params history immediately visible.
        """
        spots = []
        for ts, params in zip(self._plot_ts, self._plot_params):
            if not np.isfinite(ts):
                continue
            rgb = self._color_for_params(params if params is not None else "")
            spots.append({
                "pos":   (float(ts), 0.0),
                "brush": pg.mkBrush(*rgb, 255),
                "pen":   _PEN_NONE,
            })
        self._rug_scatter.setData(spots)

    # ── Selection / inspection linking ──────────────────────────────────────────

    def _populate_list(self) -> None:
        """Repopulate the side list — one row per plotted point, in scatter order."""
        self._list.blockSignals(True)
        self._list.clear()
        for path in self._plot_paths:
            item = QListWidgetItem(Path(path).name)
            self._list.addItem(item)
        self._list.blockSignals(False)
        self._selected = None
        self._sel_marker.hide()
        self._sel_label.setText("")

    def _on_list_row_changed(self, row: int) -> None:
        if 0 <= row < len(self._plot_paths):
            self._select(row, from_list=True)

    def _on_scatter_clicked(self, _scatter, points) -> None:
        # `points` is an array of every spot under the cursor — when zoomed out
        # a single click can land on several.  Just take the first.
        if len(points) == 0:
            return
        i = points[0].data()
        if i is not None:
            self._select(int(i), from_list=False)

    def _select(self, i: int, from_list: bool) -> None:
        if not (0 <= i < len(self._plot_paths)):
            return
        self._selected = i
        self._sel_marker.setData([self._plot_ts[i]], [self._plot_vals[i]])
        self._sel_marker.show()
        name = Path(self._plot_paths[i]).name
        self._sel_label.setText(
            f"Selected: {name}   —   {self._display_label()} "
            f"{_quant.format_value(self._variable_key, self._plot_vals[i], with_unit=True)}"
        )
        if not from_list:
            self._list.blockSignals(True)
            self._list.setCurrentRow(i)
            self._list.blockSignals(False)
            self._list.scrollTo(self._list.currentIndex())

    def _on_double_click(self, _item: QListWidgetItem) -> None:
        if self._selected is not None and 0 <= self._selected < len(self._plot_paths):
            self.view_file_requested.emit(self._plot_paths[self._selected])


    def closeEvent(self, event):
        self._cluster_bar.detach()
        super().closeEvent(event)
