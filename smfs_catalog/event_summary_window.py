# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/event_summary_window.py
#
# EventSummaryWindow — shows all detected molecular events together, split
# live into Hits / Non-hits by the criteria gate. Events are not a third
# population: criteria_gate.evaluate() with nothing checked already returns
# everything as a hit, so turning off every filter IS the events view.
#
# 2×2 grid layout:
#   Upper-left  : scatter — any variable on X and Y (variables.available),
#                 opening on extension at rupture vs rupture force
#   Upper-right : Y histogram (transposed: X = count; Y linked to scatter)
#   Lower-left  : X histogram (Y = count; X linked to scatter)
#   Lower-right : empty (reserved)
#
# Hits are drawn red, non-hits gray — the REAL criteria_gate.evaluate()
# split, not a proxy.
#
# ONE population control (Hits / Non-Hits / Both): what is drawn IS what is
# analysed. The scatter, both histograms, the event list, the plotted count,
# the fit on the scatter, Fit X/Y/2D, Isoforce, the 2DH builds and every
# export all take that one population, so no number on screen can describe
# curves that are not on screen. Under Both the 2DH builds and Isoforce take
# every event: an ensemble is a histogram over the curves it is handed, and
# cluster structure that crosses the criteria only appears in a mixed one.
#
# A crosshair cursor in the scatter + matching lines in both histograms
# marks the current curve when it is a confirmed event.
#
# The plotted X/Y come from variables.columns, so seg_* values follow each
# curve's CURRENTLY SELECTED segment (Ultimate/Penultimate and manual
# overrides) exactly as the queue table does. A curve contributes a point only
# when both values exist — never a fabricated value — and the Why dialog says
# why each missing curve is missing.
#
# The cohort handed to the 2DH windows, Isoforce and the WLC navigator is NOT
# the plotted set: it is still the curves with a selected-segment rupture force
# AND contour length (_force_arr/_length_arr, population_ledger), whatever the
# axes show, so changing an axis never changes a downstream cohort.
#
# Pre-populated from the DB at open time (no curve loading required).

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from . import db as _db
from . import criteria_gate as _gate
from . import export_utils as _export
from . import ledger as _ledger
from . import histogram_binning as _hb
from . import style
from . import clustering as _clustering
from . import regression as _reg
from . import variables as _vars
from .widgets import ClusterColourBar, FlowLayout, LabeledControl, VariableCombo
from .qt_utils import _DateAxis, _make_session_header, set_si_label, fit_on_screen
from .quantities import format_value as _q   # ONE formatter: unit and
# meaningful digits come from quantities.py, so the same measurement
# cannot print as 166, 166.2 and 166.20 in three different windows.
from . import quantities as _quant

# Hits and non-hits are separated by tone, not hue. Red is reserved for status.
# With
# both populations neutral, EVERY palette hue stays available for whatever is
# drawn on top, and density still reads correctly because tone is what a dense
# scatter encodes anyway.  The two are always named in the legend/toggle
# row as well, so the distinction is never carried by the marks alone.
_EVENT_RGBA = style.HIT_RGBA
_HIST_HIT_RGBA     = style.rgba(style.INK_STRONG, 190)   # bars: opaque enough to read
_HIST_NON_HIT_RGBA = style.rgba(style.INK_FAINT, 190)
_CURS_PEN   = pg.mkPen(style.INK, width=1.5, style=Qt.PenStyle.DashLine)

# Model-free by default: both are read off the force peak before any WLC fit,
# so a curve whose fit failed is still plotted.
_DEFAULT_X = "seg_x_rupture_nm"
_DEFAULT_Y = "seg_force_pN"

# seg_* variables that exist only once the selected segment got far enough
# through roi_events.fit_segments, so a missing value is explained by that
# segment's stored fit outcome. The rest (ROI-level deltas, segment count) can
# be missing for reasons the fit outcome does not describe.
_SEGMENT_FIT_KEYS = frozenset({
    "seg_l_p_nm", "seg_l_c_nm", "seg_l_p_err", "seg_l_c_err",
    "seg_force_pN", "seg_x_rupture_nm", "seg_x_junction_nm",
    "seg_tau", "seg_z_max", "seg_x_max_nm", "seg_edge_pinned",
})


def _vsep() -> QFrame:
    """Thin vertical rule — groups the action row's button clusters without
    a QLabel caption above each one (kept to a single row)."""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


def _let_text_wrap(label: QLabel) -> None:
    """Let a long readout reflow instead of widening the window.

    A QLabel reports its whole single-line text as its width hint, and Qt will
    not shrink a window below that, so a readout that grows a few words wide
    can put the window's right-hand edge off the screen.
    """
    label.setWordWrap(True)
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)


def fit_outcome_text(outcome: dict) -> str:
    """A stored segment fit outcome as a short phrase: "fit",
    "review: peak at segment boundary", "optimizer failed", "not attempted"."""
    status, detail = outcome.get("fit_status"), outcome.get("fit_detail")
    if status == "fit_available":
        return "fit"
    if status == "review":
        return f"review: {detail}" if detail else "review"
    if status == "not_attempted":
        return "not attempted"
    return detail or "no fit"


def drop_breakdown_lines(led) -> list[str]:
    """One `38 × <reason> (<detail>)` line per distinct drop label, largest
    first — finer than Ledger.breakdown_lines, so drops sharing a reason are
    split by the stored fit outcome in their detail."""
    counts = Counter(d.label for d in led.drops())
    return [f"{n:,} × {label}" for label, n in counts.most_common()]


class EventSummaryWindow(QMainWindow):
    """
    Displays all confirmed rupture events together as a scatter of any two
    variables, with linked marginal histograms — see module docstring.

    Pre-populates from the DB on construction; live updates go through
    reload_paths(), which re-reads every value from the DB — never recomputed
    locally in this window.

    Navigation in RawCurveWindow moves a crosshair cursor to the current
    curve's position in the scatter (or hides it for non-events).
    """

    # Emitted when a WLC fit is completed for a event: (index, l_p_nm, l_c_nm, params_json)
    wlc_result_ready = pyqtSignal(int, float, float, str)

    def __init__(
        self,
        prepass_results: list[dict],
        db_path:         str | None  = None,
        session_info:    dict | None = None,
    ) -> None:
        super().__init__()

        self._results      = prepass_results
        self._db_path      = db_path or _db.DEFAULT_DB_PATH
        self._session_info = session_info
        self._criteria_opener = None   # set via set_criteria_opener()
        self._experimentalist_resolved = False
        self._experimentalist: str | None = None
        self._raw_win      = None   # set via set_raw_window()
        self._wlc_view      = None   # keep reference to prevent GC
        self._isoforce_win  = None   # keep reference to prevent GC
        # Population-keyed ("hit"/"non_hit") so fitting/2DH-building the
        # scope selector's populations never collide or silently reuse
        # each other's window — see module docstring.
        self._gmm_wins:      dict   = {}   # "population:x_key:y_key" → GmmFitWindow
        self._norm_2dh_wins: dict   = {}   # population → Normalized2DHWindow
        self._phys_2dh_wins: dict   = {}   # population → Physical2DHWindow
        self._2dh_wins:     list   = []   # registered via set_2dh_window()
        self._fit_wins:      dict   = {}   # "label (population)" → DistFitWindow
        n                  = len(prepass_results)

        # Per-curve event arrays — NaN = no value for the selected segment.
        # Force/length define the downstream cohort (module docstring); X/Y are
        # what is plotted.
        self._force_arr  = np.full(n, np.nan)   # selected segment's rupture force (pN)
        self._length_arr = np.full(n, np.nan)   # selected segment's WLC contour length (nm)
        self._x_arr      = np.full(n, np.nan)
        self._y_arr      = np.full(n, np.nan)
        # The drawn points' correlation and OLS line, recomputed every
        # _rebuild over what is on screen; None when there is nothing honest
        # to report (see _render_fit).
        self._fit:  _reg.LinearFit   | None = None
        self._corr: _reg.Correlation | None = None
        self._fit_paths: list[str] = []
        self._fishing_note = ""
        self._vars = _vars.available(
            [r.get("path") for r in prepass_results if r.get("path")], self._db_path)
        self._var_by_key = {v.key: v for v in self._vars}
        # Selected segment's stored fit outcome, None until loaded — see
        # _prepopulate and _record_missing.
        self._seg_outcome: list[dict | None] = [None] * n
        # Real criteria_gate.evaluate() split, recomputed every _rebuild() —
        # True = hit. Kept over the FULL self._results (not just plotted
        # points): a curve can have a real hit/non-hit verdict without a
        # usable segment fit, so it's still counted even when it can't be
        # drawn.
        self._hit_mask = np.zeros(n, dtype=bool)
        # "hit" | "non_hit" | "both" — the population every result here is
        # computed over (see the scope selector below).
        self._active_population = "hit"
        self._current_index    = 0
        self._selected_index: int | None = None   # user selection (≠ playhead cursor)
        # Which segment (Ultimate/Penultimate, the dashboard's global Segment
        # combo) produced self._force_arr/_length_arr — set in _prepopulate(),
        # re-read every reload_paths() so it can never silently go stale
        # relative to what's actually plotted (see the module docstring).
        self._segment_select: str | None = None
        self._data_revision = 0
        self._data_signature = None
        self._load_error: str | None = None

        self._update_title()
        self.setWindowFlag(Qt.WindowType.Window)
        fit_on_screen(self, 1100, 700)
        style.apply_plot_defaults()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(4)
        root.setContentsMargins(8, 6, 8, 6)

        _hdr = _make_session_header(session_info)
        if _hdr is not None:
            root.addWidget(_hdr)

        # The explanation button sits beside the count it explains,
        # not in a menu, because the moment someone needs it is the moment
        # they read a number that doesn't match the one above.
        stats_row = QHBoxLayout()
        stats_row.setContentsMargins(0, 0, 0, 0)
        self._stats_label = QLabel("0 events")
        font = self._stats_label.font()
        font.setPointSize(style.FONT_SMALL_PT)
        self._stats_label.setFont(font)
        # A one-line label sets the window's minimum width from its own text,
        # which is how this window's minimum reached 2056 px on a 1920 screen.
        # Wrapping plus an ignored width hint lets the text reflow instead.
        _let_text_wrap(self._stats_label)
        stats_row.addWidget(self._stats_label, 1)
        self._why_btn = QPushButton("Why?")
        self._why_btn.setFont(font)
        self._why_btn.setToolTip(
            "Which curves this window was given but could not plot, and why.")
        self._why_btn.clicked.connect(self._on_show_drops)
        self._why_btn.setEnabled(False)
        stats_row.addWidget(self._why_btn)
        stats_row.addStretch()
        root.addLayout(stats_row)

        # Readout for the user's current selection (independent of the playhead
        # crosshair, which tracks the curve being analysed/navigated).
        self._sel_label = QLabel("")
        self._sel_label.setFont(font)
        _let_text_wrap(self._sel_label)
        root.addWidget(self._sel_label)

        # Cluster colouring is placed here rather than in the action row:
        # it changes how everything above is DRAWN, and the action row is
        # about what the buttons below it compute over.
        self._cluster_bar = ClusterColourBar()
        self._cluster_bar.changed.connect(self._rebuild)
        root.addWidget(self._cluster_bar)

        # Axis choice — what is plotted; the downstream cohort handed to the
        # 2DH windows and the WLC navigator does not follow it (module docstring).
        axis_row = QHBoxLayout()
        axis_row.setContentsMargins(0, 0, 0, 0)
        self._x_combo = VariableCombo(self._vars, _DEFAULT_X)
        self._y_combo = VariableCombo(self._vars, _DEFAULT_Y)
        self._swap_btn = QPushButton("Swap axes")
        self._swap_btn.clicked.connect(self._on_swap_axes)
        self._fit_chk = QCheckBox("Linear fit")
        self._fit_chk.setToolTip(
            "OLS line with a 95% confidence band for the mean trend, not for "
            "individual curves. The interval assumes independent residuals "
            "with constant variance; consecutive measurements can make it "
            "too narrow."
        )
        self._fit_chk.toggled.connect(self._rebuild)
        axis_row.addWidget(LabeledControl("X:", self._x_combo))
        axis_row.addWidget(LabeledControl("Y:", self._y_combo))
        axis_row.addWidget(self._swap_btn)
        axis_row.addWidget(self._fit_chk)
        axis_row.addStretch()
        root.addLayout(axis_row)

        # The fit's own readout, carrying the population it was computed over:
        # the fit is the one number here a reader will compare against an
        # exported file, so it says which curves it describes.
        self._fit_label = QLabel("")
        self._fit_label.setFont(font)
        _let_text_wrap(self._fit_label)
        root.addWidget(self._fit_label)

        # The fishing caution, stated where the number is read: with this many
        # variables on offer, a p-value found by scanning pairs is not the
        # p-value it looks like.
        self._warn_label = QLabel("")
        self._warn_label.setStyleSheet(style.qss_text(style.TEXT_WARNING))
        self._warn_label.setFont(style.font(font, size_pt=style.FONT_SMALL_PT))
        _let_text_wrap(self._warn_label)
        root.addWidget(self._warn_label)
        self._x_combo.currentIndexChanged.connect(self._on_axes_changed)
        self._y_combo.currentIndexChanged.connect(self._on_axes_changed)

        # ── Action row — Filtering, the population control, fit
        # buttons, 2DH buttons, exports, View individual events.
        #
        # A FlowLayout, because as one QHBoxLayout these fourteen buttons and
        # their separators set a minimum width of 1822 px and the window could
        # not be made narrower than that on any screen.  It wraps onto as many
        # rows as the current width needs.
        action_row = FlowLayout(margin=0, h_spacing=6, v_spacing=4)

        self._criteria_btn = QPushButton("Filtering…")
        self._criteria_btn.setToolTip(
            "Open the criteria dialog — tune which variables gate hit/non-hit "
            "and watch this window's split update live."
        )
        self._criteria_btn.clicked.connect(self._on_open_criteria)
        action_row.addWidget(self._criteria_btn)

        # The window's one population control: what is drawn IS what is
        # analysed (module docstring).
        self._pop_btns = {}
        self._pop_group = QButtonGroup(self)
        self._pop_group.setExclusive(True)
        for pop in ("hit", "non_hit", "both"):
            btn = QPushButton("Both" if pop == "both"
                              else _ledger.population_label(pop))
            btn.setCheckable(True)
            btn.setChecked(pop == "hit")
            btn.setToolTip(
                "The population this window draws AND computes over: the "
                "scatter, both histograms, the event list, the fit, "
                "Fit X/Y/2D, Isoforce, the 2DH builds and every export."
            )
            btn.toggled.connect(
                lambda checked, p=pop: self._on_population_toggled(checked, p))
            self._pop_group.addButton(btn)
            self._pop_btns[pop] = btn
        # The caption travels with its buttons so a wrap cannot strand it.
        action_row.addWidget(LabeledControl(
            "Population (drawn and analysed):", *self._pop_btns.values()))

        action_row.addWidget(_vsep())

        self._fit_x_btn = QPushButton("Fit X…")
        self._fit_x_btn.clicked.connect(self._on_fit_x)
        action_row.addWidget(self._fit_x_btn)
        self._fit_y_btn = QPushButton("Fit Y…")
        self._fit_y_btn.clicked.connect(self._on_fit_y)
        action_row.addWidget(self._fit_y_btn)
        self._fit_2d_btn = QPushButton("Fit 2D…")
        self._fit_2d_btn.clicked.connect(self._on_fit_2d)
        action_row.addWidget(self._fit_2d_btn)
        self._isoforce_btn = QPushButton("Isoforce…")
        self._isoforce_btn.setToolTip(
            "Curves with an adjacent isoforce pair on the last ROI: the current "
            "manual Primary/Secondary pair when complete, otherwise the last two "
            "ruptures, marked on the curve."
        )
        self._isoforce_btn.clicked.connect(self._on_view_isoforce)
        action_row.addWidget(self._isoforce_btn)

        action_row.addWidget(_vsep())

        # 2D-histogram windows — each cascades into PCA → K-means → clustering.
        # Two registered representations of the same hit: unitless and physical.
        self._norm_2dh_btn = QPushButton("2DH (normalized)")
        self._norm_2dh_btn.setToolTip("Open the total unitless 2D histogram for these traces.")
        self._norm_2dh_btn.clicked.connect(self._on_open_normalized_2dh)
        action_row.addWidget(self._norm_2dh_btn)
        self._phys_2dh_btn = QPushButton("2DH (physical)")
        self._phys_2dh_btn.setToolTip("Open the total physical-units 2D histogram for these traces.")
        self._phys_2dh_btn.clicked.connect(self._on_open_physical_2dh)
        action_row.addWidget(self._phys_2dh_btn)

        action_row.addWidget(_vsep())

        # Export writes to the configured database export directory.
        # override folder set from the dashboard's "Export folder…" button.
        # Reads self._population_mask()/self._active_population, same as the
        # Fit buttons above and the same curves the scatter draws.
        self._export_scatter_btn = QPushButton("Export scatter…")
        self._export_scatter_btn.clicked.connect(self._on_export_scatter)
        action_row.addWidget(self._export_scatter_btn)
        self._export_x_hist_btn = QPushButton("Export X hist…")
        self._export_x_hist_btn.clicked.connect(self._on_export_x_hist)
        action_row.addWidget(self._export_x_hist_btn)
        self._export_y_hist_btn = QPushButton("Export Y hist…")
        self._export_y_hist_btn.clicked.connect(self._on_export_y_hist)
        action_row.addWidget(self._export_y_hist_btn)
        self._export_rois_btn = QPushButton("Export ROI/segment rows…")
        self._export_rois_btn.setToolTip(
            "One row per ROI segment, not per curve: every rupture in every ROI "
            "of every curve in this population, with its WLC fit, its force and "
            "the step to the rupture before it."
        )
        self._export_rois_btn.clicked.connect(self._on_export_roi_segments)
        action_row.addWidget(self._export_rois_btn)

        action_row.addWidget(_vsep())

        # Moved up from the side panel (was "View fit") — same button/slot,
        # new name and position.
        self._view_fit_btn = QPushButton("View individual events")
        self._view_fit_btn.setEnabled(False)
        self._view_fit_btn.clicked.connect(self._on_view_fit)
        action_row.addWidget(self._view_fit_btn)

        # No addStretch(): FlowLayout packs from the left and QLayout has none.
        _action_bar = QWidget()
        _action_bar.setLayout(action_row)
        _action_bar.setSizePolicy(QSizePolicy.Policy.Preferred,
                                  QSizePolicy.Policy.Minimum)
        root.addWidget(_action_bar)

        # ── 2×2 grid via nested splitters ─────────────────────────────────────
        # Outer horizontal splitter: plot grid | right-hand side panel.
        # Plot grid left column  : scatter (top) + X histogram (bottom)
        # Plot grid right column : Y histogram (top)  + empty widget (bottom)
        # Side panel (on the RIGHT, matching WlcViewWindow's track list): fit
        # status over the event file-list (click a event to inspect it,
        # double-click to open the WLC fit view) — "View individual events"
        # itself lives in the action row above now.
        outer = QSplitter(Qt.Orientation.Horizontal)

        hsplit      = QSplitter(Qt.Orientation.Horizontal)
        vsplit_left = QSplitter(Qt.Orientation.Vertical)
        vsplit_right = QSplitter(Qt.Orientation.Vertical)
        hsplit.addWidget(vsplit_left)
        hsplit.addWidget(vsplit_right)
        outer.addWidget(hsplit)

        side = QWidget()
        side_lay = QVBoxLayout(side)
        side_lay.setContentsMargins(0, 0, 0, 0)
        self._fit_status = QLabel("")
        self._fit_status.setFont(font)
        self._event_list = QListWidget()
        self._event_list.setStyleSheet(style.LIST_QSS)
        self._event_list.currentRowChanged.connect(self._on_list_row_changed)
        # Double-click a event to jump straight into the WLC fit navigator (the
        # "View individual events" button in the action row above still works
        # too). The double-click also fires currentRowChanged first, so the
        # selection is already set.
        self._event_list.itemDoubleClicked.connect(lambda _item: self._on_view_fit())
        side_lay.addWidget(self._fit_status)
        side_lay.addWidget(self._event_list, stretch=1)
        outer.addWidget(side)

        outer.setSizes([900, 200])
        root.addWidget(outer, stretch=1)

        # ── Upper-left: scatter ────────────────────────────────────────────────
        self._scatter_plot = pg.PlotWidget()
        self._scatter_plot.showGrid(x=True, y=True, alpha=0.2)

        # Small translucent dots make density readable as tone.
        self._scatter_fail = pg.ScatterPlotItem(size=style.DOT_SIZE, pen=pg.mkPen(None))
        self._scatter_pass = pg.ScatterPlotItem(size=style.DOT_SIZE, pen=pg.mkPen(None))
        self._scatter_plot.addItem(self._scatter_fail, ignoreBounds=True)
        self._scatter_plot.addItem(self._scatter_pass)
        self._scatter_pass.sigClicked.connect(self._on_scatter_clicked)
        self._scatter_fail.sigClicked.connect(self._on_scatter_clicked)

        # Selection marker — a ring around the user-selected point (distinct
        # from the dashed playhead crosshair).
        self._sel_marker = pg.ScatterPlotItem(
            size=16, symbol="o",
            pen=pg.mkPen(style.INK, width=2), brush=pg.mkBrush(None),
        )
        self._sel_marker.hide()
        self._scatter_plot.addItem(self._sel_marker, ignoreBounds=True)

        # OLS line and its band, both ignoreBounds: a model drawn over the data
        # must never be what sets the view range.
        self._band_lo = pg.PlotCurveItem(x=[0.0, 1.0], y=[0.0, 0.0], pen=pg.mkPen(None))
        self._band_hi = pg.PlotCurveItem(x=[0.0, 1.0], y=[0.0, 0.0], pen=pg.mkPen(None))
        self._band    = pg.FillBetweenItem(self._band_lo, self._band_hi,
                                           brush=style.band_brush(style.series_line(0)))
        self._fit_line = pg.PlotCurveItem(pen=style.model_pen(style.series_line(0)))
        for it in (self._band_lo, self._band_hi, self._band, self._fit_line):
            self._scatter_plot.addItem(it, ignoreBounds=True)

        # Crosshair cursor
        self._cursor_v = pg.InfiniteLine(angle=90, movable=False, pen=_CURS_PEN)
        self._cursor_h = pg.InfiniteLine(angle=0,  movable=False, pen=_CURS_PEN)
        self._cursor_v.hide()
        self._cursor_h.hide()
        self._scatter_plot.addItem(self._cursor_v)
        self._scatter_plot.addItem(self._cursor_h)

        vsplit_left.addWidget(self._scatter_plot)

        # ── Lower-left: X histogram (X linked to scatter X) ────────────────────
        self._x_hist_plot = pg.PlotWidget()
        self._x_hist_plot.setLabel("left", "Count")
        self._x_hist_plot.showGrid(x=True, y=True, alpha=0.2)
        self._x_hist_plot.getAxis("bottom").setStyle(showValues=False)
        self._x_hist_plot.getViewBox().setXLink(self._scatter_plot.getViewBox())

        self._x_bar_pass = pg.BarGraphItem(
            x0=[], x1=[], y0=[], y1=[],
            pen=pg.mkPen(None), brush=pg.mkBrush(*_HIST_HIT_RGBA),
        )
        self._x_bar_fail = pg.BarGraphItem(
            x0=[], x1=[], y0=[], y1=[],
            pen=pg.mkPen(None), brush=pg.mkBrush(*_HIST_NON_HIT_RGBA),
        )
        self._x_hist_plot.addItem(self._x_bar_fail)
        self._x_hist_plot.addItem(self._x_bar_pass)

        # Cursor line in X histogram (vertical — tracks X)
        self._x_hist_cursor = pg.InfiniteLine(angle=90, movable=False, pen=_CURS_PEN)
        self._x_hist_cursor.hide()
        self._x_hist_plot.addItem(self._x_hist_cursor)

        vsplit_left.addWidget(self._x_hist_plot)
        vsplit_left.setSizes([480, 180])

        # ── Upper-right: Y histogram (Y linked to scatter Y) ──────────────────
        self._y_hist_plot = pg.PlotWidget()
        self._y_hist_plot.setLabel("bottom", "Count")
        self._y_hist_plot.showGrid(x=True, y=True, alpha=0.2)
        self._y_hist_plot.getAxis("left").setStyle(showValues=False)
        self._y_hist_plot.getViewBox().setYLink(self._scatter_plot.getViewBox())

        self._y_bar_pass = pg.BarGraphItem(
            x0=[], x1=[], y0=[], y1=[],
            pen=pg.mkPen(None), brush=pg.mkBrush(*_HIST_HIT_RGBA),
        )
        self._y_bar_fail = pg.BarGraphItem(
            x0=[], x1=[], y0=[], y1=[],
            pen=pg.mkPen(None), brush=pg.mkBrush(*_HIST_NON_HIT_RGBA),
        )
        # Per-cluster 1DH outlines, created empty and populated by
        # _draw_cluster_curves.  Held as lists so a change of k adds or removes
        # curves without either panel needing to know k in advance.
        self._y_hist_curves: list = []
        self._x_hist_curves:  list = []
        self._y_hist_plot.addItem(self._y_bar_fail)
        self._y_hist_plot.addItem(self._y_bar_pass)

        # Cursor line in Y histogram (horizontal — tracks Y)
        self._y_hist_cursor = pg.InfiniteLine(angle=0, movable=False, pen=_CURS_PEN)
        self._y_hist_cursor.hide()
        self._y_hist_plot.addItem(self._y_hist_cursor)

        vsplit_right.addWidget(self._y_hist_plot)

        # ── Lower-right: empty (reserved) ─────────────────────────────────────
        vsplit_right.addWidget(QWidget())
        vsplit_right.setSizes([480, 180])

        hsplit.setSizes([780, 280])

        self._apply_axis_labels()
        self._prepopulate()

    # ── Pre-population ────────────────────────────────────────────────────────

    def _prepopulate(self) -> None:
        """Load the downstream force/length, each curve's stored fit outcome,
        and the plotted X/Y, all for the currently selected segment. Missing
        values stay NaN (e.g. Penultimate on a one-segment curve), never a
        fabricated value."""
        try:
            from .roi_pipeline import read_segment_select, segment_summary_bulk

            paths  = [r["path"] for r in self._results]
            select = read_segment_select(self._db_path)
            self._segment_select = select
            seg    = segment_summary_bulk(paths, select, self._db_path)
            for i, r in enumerate(self._results):
                sd     = seg.get(_db.normalize_path(r["path"]), {})
                force  = sd.get("force_pN")
                length = sd.get("l_c_nm")
                if force is not None and length is not None:
                    self._force_arr[i]  = force
                    self._length_arr[i] = length
                self._seg_outcome[i] = {
                    "n_segments": sd.get("n_segments"),
                    "fit_status": sd.get("fit_status"),
                    "fit_detail": sd.get("fit_detail"),
                }
            self._load_axes()
            self._load_error = None
        except Exception as exc:
            # A summary window is an inspector, so a DB/read failure must not
            # take down the dashboard.  It must not masquerade as a genuine
            # empty cohort either: _update_stats exposes this state visibly.
            self._load_error = f"{type(exc).__name__}: {exc}"

        self._rebuild()
        signature = self._current_data_signature()
        if signature != self._data_signature:
            self._data_signature = signature
            self._data_revision += 1
            self._mark_fit_windows_stale()
        if len(self._results):
            self._update_cursor(self._current_index)
        else:
            self._hide_cursor()

        # Closed Qt windows are normally only hidden.  They are not useful as
        # live 2DH subscribers: reopening creates and rebuilds a fresh window.
        self._2dh_wins = [win for win in self._2dh_wins
                          if self._window_is_visible(win)]
        if self._load_error is None:
            for win in self._2dh_wins:
                win.sync_from_event_summary(self)

    # ── Axes ──────────────────────────────────────────────────────────────────

    @property
    def _x_key(self) -> str:
        return self._x_combo.currentData()

    @property
    def _y_key(self) -> str:
        return self._y_combo.currentData()

    def _axis_label(self, key: str) -> str:
        return self._var_by_key[key].label if key in self._var_by_key else _vars.label(key)

    def _load_axes(self) -> None:
        """Read the plotted X/Y for every loaded curve (NaN where missing)."""
        n = len(self._results)
        self._x_arr = np.full(n, np.nan)
        self._y_arr = np.full(n, np.nan)
        idx = [i for i, r in enumerate(self._results) if r.get("path")]
        if not idx:
            return
        xk, yk = self._x_key, self._y_key
        _order, cols = _vars.columns([self._results[i]["path"] for i in idx],
                                     list(dict.fromkeys((xk, yk))), self._db_path)
        self._x_arr[idx] = cols[xk]
        self._y_arr[idx] = cols[yk]

    def _apply_axis_labels(self) -> None:
        xk, yk = self._x_key, self._y_key
        for side, key in (("bottom", xk), ("left", yk)):
            if key == _vars.TIME_KEY:
                self._scatter_plot.setAxisItems({side: _DateAxis(orientation=side)})
                self._scatter_plot.setLabel(side, "Acquisition time")
            else:
                self._scatter_plot.setAxisItems({side: pg.AxisItem(side)})
                set_si_label(self._scatter_plot, side,
                             style.mathify(self._axis_label(key)), key=key, si=False)
        for btn, what, key in ((self._fit_x_btn, "Fit a distribution to", xk),
                               (self._fit_y_btn, "Fit a distribution to", yk),
                               (self._export_x_hist_btn, "Histogram of", xk),
                               (self._export_y_hist_btn, "Histogram of", yk)):
            btn.setToolTip(f"{what} {self._axis_label(key)}, over the selected population.")
        self._fit_2d_btn.setToolTip(
            f"Fit a 2D Gaussian Mixture Model to {self._axis_label(yk)} × "
            f"{self._axis_label(xk)}.")
        self._export_scatter_btn.setToolTip(
            f"path, {self._axis_label(xk)}, {self._axis_label(yk)} — one row per point.")
        self._fit_chk.setText(f"Linear fit ({self._population_label()})")

    def _on_axes_changed(self) -> None:
        try:
            self._load_axes()
            self._load_error = None
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
        self._apply_axis_labels()
        # A new axis is a different view, not new data: open fit snapshots are
        # keyed by their variable and stay valid, so no revision bump here.
        self._data_signature = self._current_data_signature()
        self._rebuild()
        self._update_cursor(self._current_index)

    def _on_swap_axes(self) -> None:
        VariableCombo.swap(self._x_combo, self._y_combo)
        self._on_axes_changed()

    @staticmethod
    def _window_is_visible(win) -> bool:
        try:
            return win is not None and win.isVisible()
        except RuntimeError:
            return False

    def _current_data_signature(self) -> tuple:
        """Identity of the values/populations used by statistical children.

        Dashboard flushes also occur for non-events.  Comparing content keeps
        those no-op refreshes from falsely labelling a fit as outdated.
        """
        def _value(v: float):
            return None if np.isnan(v) else float(v)

        return (
            self._segment_select,
            tuple(r.get("path") for r in self._results),
            tuple(_value(v) for v in self._force_arr),
            tuple(_value(v) for v in self._length_arr),
            (self._x_key, self._y_key),
            tuple(_value(v) for v in self._x_arr),
            tuple(_value(v) for v in self._y_arr),
            tuple(bool(v) for v in self._hit_mask),
            self._load_error,
        )

    def _mark_fit_windows_stale(self) -> None:
        """Label open statistical results as snapshots; never recompute them.

        Live analysis can add a curve on every worker flush.  Automatically
        refitting here would make an open GMM/distribution window launch a new
        calculation for every trace. Instead the snapshot remains available
        for inspection and the next explicit Fit action replaces it.
        """
        for win in [*self._fit_wins.values(), *self._gmm_wins.values()]:
            if not self._window_is_visible(win):
                continue
            if getattr(win, "_event_summary_revision", None) == self._data_revision:
                continue
            title = win.windowTitle()
            if not title.endswith(" — outdated snapshot"):
                win.setWindowTitle(f"{title} — outdated snapshot")

    # ── Scatter + histogram rebuild ───────────────────────────────────────────

    def _rebuild(self) -> None:
        """Recompute the real hit/non-hit split (criteria_gate.evaluate(),
        not a proxy) and redraw all three panels."""
        paths = [r.get("path") for r in self._results]
        hits, _non_hits = _gate.evaluate([p for p in paths if p], self._db_path)
        hit_set = set(hits)
        self._hit_mask = np.array([bool(p) and p in hit_set for p in paths], dtype=bool)

        valid = ~np.isnan(self._y_arr) & ~np.isnan(self._x_arr)
        idx_v = np.where(valid)[0]
        y_v   = self._y_arr[valid]
        x_v   = self._x_arr[valid]
        hit_v = self._hit_mask[valid]

        # One population control (module docstring): these two are the drawn
        # tones AND the analysed set, so nothing on screen can describe curves
        # that are not on it. The hit/non-hit split itself is untouched.
        analysed = self._population_mask()[valid]
        pas  = hit_v  & analysed
        fail = ~hit_v & analysed

        # Scatter marks are translucent (thousands of overlapping points —
        # density should read as tone); histogram bars are the same two tones
        # opaque, since a bar is one solid area and gains nothing from alpha.
        event_brush = pg.mkBrush(*_EVENT_RGBA)
        fail_brush  = pg.mkBrush(*style.NON_HIT_RGBA)
        bar_hit_brush = pg.mkBrush(*_HIST_HIT_RGBA)
        bar_non_brush = pg.mkBrush(*_HIST_NON_HIT_RGBA)

        # Scatter — per-point `data` carries the index back into self._results
        # so scatter clicks can be mapped to a file.
        # Cluster colouring projects the 2DH clustering back onto these
        # two scalars.  It replaces the hit/non-hit TONE, not the hit/non-hit
        # split: the population control still decides what is drawn, so a
        # non-hit stays off the plot under Hits.  An unlabelled curve keeps the
        # neutral tone rather than borrowing a cluster's hue.
        cluster_on = self._cluster_bar.is_active()
        cl = _clustering.current() if cluster_on else None
        if cl is not None:
            def _spots(sel, fallback):
                out = []
                for j in np.where(sel)[0]:
                    i = int(idx_v[j])
                    lbl = cl.label_for(paths[i]) if paths[i] else None
                    brush = (style.scatter_brush(style.series_labeled(lbl))
                             if lbl is not None else fallback)
                    out.append({"pos": (float(x_v[j]), float(y_v[j])),
                                "data": i, "brush": brush,
                                "pen": pg.mkPen(None)})
                return out
            self._scatter_pass.setData(_spots(pas, event_brush))
            self._scatter_fail.setData(_spots(fail, fail_brush))
        else:
            self._scatter_pass.setData(x=x_v[pas].tolist(),  y=y_v[pas].tolist(),
                                       data=idx_v[pas].tolist(),  brush=event_brush)
            self._scatter_fail.setData(x=x_v[fail].tolist(), y=y_v[fail].tolist(),
                                       data=idx_v[fail].tolist(), brush=fail_brush)

        self._fit_paths = [paths[int(i)] for i in idx_v[analysed]]
        self._render_fit(x_v[analysed], y_v[analysed])

        # Geometry from histogram_binning, the same module the EXPORT of these
        # very histograms already used (_hb.full_range_bins below) and the same
        # convention variable_window draws on screen. Robust range plus
        # Freedman-Diaconis width; what falls outside is counted and reported,
        # never silently dropped.
        y_bins = _hb.robust_bins(y_v) if len(y_v) else None
        x_bins = _hb.robust_bins(x_v) if len(x_v) else None

        if y_bins is not None and x_bins is not None:
            self._hist_n_out = (y_bins.n_out_of_range, x_bins.n_out_of_range)

            # Y histogram (transposed: bars along Y axis) — hit/non-hit
            # stacked end-to-end (hit 0→cp, non-hit cp→cp+cf) so both remain
            # visible. Bin edges span ALL valid points regardless of the Show
            # checkboxes, so toggling visibility never moves the axis.
            y_edges = y_bins.edges
            cp_y = y_bins.count(y_v[pas])
            cf_y = y_bins.count(y_v[fail])
            y0, y1  = y_edges[:-1], y_edges[1:]
            self._y_bar_pass.setOpts(x0=np.zeros(len(cp_y)), x1=cp_y, y0=y0, y1=y1,
                                        brush=bar_hit_brush)
            self._y_bar_fail.setOpts(x0=cp_y, x1=cp_y + cf_y, y0=y0, y1=y1,
                                        brush=bar_non_brush)
            # Draw one curve per cluster for both axes.
            # Overlaid step outlines rather than stacked bars — stacking hides
            # the very shapes being compared, which is the whole point of
            # drawing them per cluster.  The hit/non-hit bars stay underneath
            # as the substrate; the cluster curves are the reading.
            shown = pas | fail
            self._draw_cluster_curves(cl, paths, idx_v, shown,
                                      y_v, y_bins, self._y_hist_curves, transposed=True)
            self._draw_cluster_curves(cl, paths, idx_v, shown,
                                      x_v, x_bins, self._x_hist_curves, transposed=False)

            # X histogram (standard: bars along X axis) — same stacking.
            x_edges = x_bins.edges
            cp_x = x_bins.count(x_v[pas])
            cf_x = x_bins.count(x_v[fail])
            x0, x1  = x_edges[:-1], x_edges[1:]
            self._x_bar_pass.setOpts(x0=x0, x1=x1, y0=np.zeros(len(cp_x)), y1=cp_x,
                                       brush=bar_hit_brush)
            self._x_bar_fail.setOpts(x0=x0, x1=x1, y0=cp_x, y1=cp_x + cf_x,
                                       brush=bar_non_brush)
        else:
            self._hist_n_out = (0, 0)
            for bar in (self._y_bar_pass, self._y_bar_fail,
                        self._x_bar_pass,  self._x_bar_fail):
                bar.setOpts(x0=[], x1=[], y0=[], y1=[])
            self._draw_cluster_curves(None, paths, idx_v, None, None, None,
                                      self._y_hist_curves, transposed=True)
            self._draw_cluster_curves(None, paths, idx_v, None, None, None,
                                      self._x_hist_curves, transposed=False)

        self._cluster_bar.refresh([p for p in paths if p])
        self._update_title()   # before _update_stats, which shows its summary
        self._update_stats()
        self._rebuild_list()
        self._update_sel_marker()

    def _render_fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Correlation and OLS line for the analysed population's points.

        Recomputed even when the line is hidden, so the readout and the export
        stay available without it. A variable against itself is a legitimate
        temporary axis choice, but its perfect line and rho are identities
        rather than analyses and must not reach either.
        """
        same = self._x_key == self._y_key
        self._fit = None if same or len(x) < 3 else _reg.linear_fit(x, y)
        self._corr = (None if same or len(x) < 3 else
                      _reg.correlate(x, y, method="spearman"))
        show = self._fit_chk.isChecked() and self._fit is not None
        for it in (self._fit_line, self._band):
            it.setVisible(show)
        if show:
            xs = np.linspace(float(x.min()), float(x.max()), 200)
            lo, hi = self._fit.band(xs)
            self._fit_line.setData(xs, self._fit.predict(xs))
            self._band_lo.setData(xs, lo)
            self._band_hi.setData(xs, hi)
        n_pairs = len(self._vars) * (len(self._vars) - 1) // 2
        self._fishing_note = (
            f"⚠ {len(self._vars)} variables here make {n_pairs} possible "
            f"pairs, so ~{max(1, round(n_pairs * 0.05))} would clear p < 0.05 "
            f"by chance alone. A correlation found by scanning pairs needs "
            f"confirming on an independent cohort before it means anything."
            if self._corr is not None and self._corr.p < 0.05 else ""
        )

    def _fit_text(self) -> str:
        """Spearman rho, slope and R², named for the population they describe.
        Empty without a fit."""
        bits = []
        if self._corr is not None:
            c = self._corr
            bits.append(f"Spearman ρ {c.rho:+.3f} (n={c.n}, p={c.p:.3g})")
        if self._fit is not None:
            f = self._fit
            if self._x_key == _vars.TIME_KEY:
                s, lo, hi = _reg.per_hour(f)
                unit = _quant.unit_of(self._y_key)
                bits.append(f"slope {s:.4g} [{lo:.4g}, {hi:.4g}] "
                            f"{unit + '/h' if unit else '/h'}")
            else:
                lo, hi = f.slope_ci
                bits.append(f"slope {f.slope:.4g} [{lo:.4g}, {hi:.4g}]")
            bits.append(f"R² {f.r2:.3f}")
        if not bits:
            return ""
        return f"fit — {self._population_label()}:   " + "   |   ".join(bits)

    def _draw_cluster_curves(self, cl, paths, idx_v, shown, values, bins,
                             store: list, *, transposed: bool) -> None:
        """One overlaid step outline per cluster, or clear them all.

        `cl` None (colouring off, or no clustering) clears and returns — the
        curves are removed from the plot rather than left with empty data, so
        a stale legend entry cannot outlive the thing it described.

        `transposed` because the force panel draws its bars along Y and the
        length panel along X; the counts are identical, only the axes swap.
        """
        plot = self._y_hist_plot if transposed else self._x_hist_plot
        for item in store:
            plot.removeItem(item)
        store.clear()
        if cl is None or bins is None or values is None or shown is None:
            return

        labels = np.array(
            [(cl.label_for(paths[int(i)]) if paths[int(i)] else None)
             for i in idx_v])
        edges = bins.edges
        # Step outline: repeat each edge so the curve traces the bin tops,
        # the same shape a bar chart's silhouette has.
        for c in sorted({int(v) for v in labels if v is not None}):
            sel = shown & np.array([v == c for v in labels])
            if not sel.any():
                continue
            counts = bins.count(values[sel])
            step_pos = np.repeat(edges, 2)[1:-1]
            step_cnt = np.repeat(counts, 2)
            pen = style.model_pen(style.series_labeled(c))
            item = (pg.PlotCurveItem(x=step_cnt, y=step_pos, pen=pen)
                    if transposed else
                    pg.PlotCurveItem(x=step_pos, y=step_cnt, pen=pen))
            plot.addItem(item)
            store.append(item)

    def _update_title(self) -> None:
        """Refresh the stable title and visible population/segment summary."""
        n_hit = int(self._hit_mask.sum())
        n_non = len(self._results) - n_hit
        seg = {"ultimate": "Ultimate", "penultimate": "Penultimate"}.get(
            self._segment_select, self._segment_select or "?"
        )
        self.setWindowTitle("SMFS — event summary")
        self._population_summary = (
            f"{n_hit} hits, {n_non} non-hits   |   segment: {seg}")

    def _plottability_ledger(self) -> _ledger.Ledger:
        """Every loaded curve, and why it cannot be drawn when it cannot.

        Population-blind on purpose: this explains the gap between the count
        in the population summary (the whole events population) and the count in the stats
        line.
        """
        paths = [r.get("path") or "" for r in self._results]
        led = _ledger.Ledger("Explore Events plottability", paths)
        axes = [(self._x_key, self._x_arr), (self._y_key, self._y_arr)]
        for i, p in enumerate(paths):
            if p:
                self._record_missing(led, i, p, axes)
        return led

    def _record_missing(self, led: _ledger.Ledger, i: int, p: str, axes) -> None:
        """Drop curve i from `led` if any (key, values) in `axes` lacks a value
        for it, with the reason.

        For a missing seg_* value the reason comes from the selected segment's
        stored outcome when it has been loaded: no stored analysis, no such
        segment, or the fit outcome the pipeline recorded. Otherwise the drop
        names the variables with no value.
        """
        missing = [k for k, arr in axes if np.isnan(arr[i])]
        if not missing:
            return
        seg_txt = f"segment: {self._segment_select or '?'}"
        seg_missing = [k for k in missing
                       if _vars.source_of(k) == _vars.SOURCE_SEGMENT]
        o = self._seg_outcome[i] if i < len(self._seg_outcome) else None
        if o is not None and seg_missing:
            if o["n_segments"] is None:
                led.drop(p, "no_stored_segments")
                return
            if o["fit_status"] is None:
                led.drop(p, "no_segment_chosen", seg_txt)
                return
            if (o["fit_status"] in ("no_fit", "not_attempted")
                    and any(k in _SEGMENT_FIT_KEYS for k in seg_missing)):
                reason = ("fit_failed" if o["fit_detail"] == "optimizer failed"
                          else "fit_not_attempted")
                led.drop(p, reason,
                         f"{o['fit_detail'] or 'fitter did not run'}; {seg_txt}")
                return
        detail = "no " + ", no ".join(self._axis_label(k) for k in dict.fromkeys(missing))
        led.drop(p, "not_finite", f"{detail}; {seg_txt}" if seg_missing else detail)

    def _on_show_drops(self) -> None:
        """The tally and the journey, for the curves this window couldn't plot."""
        led = self._plottability_ledger()
        if led.n_dropped == 0:
            QMessageBox.information(
                self, "Dropped curves",
                f"Nothing dropped — all {led.n_asked:,} curves have "
                f"{self._axis_label(self._x_key)} and "
                f"{self._axis_label(self._y_key)}.")
            return
        lines = [led.summary("plotted"), ""]
        lines += drop_breakdown_lines(led)
        lines.append("")
        lines.append("Curves (first 40):")
        for d in led.drops()[:40]:
            lines.append(f"  {Path(d.path).name} — {d.label}")
        if led.n_dropped > 40:
            lines.append(f"  … and {led.n_dropped - 40:,} more "
                         f"(export the scatter for the full list)")
        box = QMessageBox(self)
        box.setWindowTitle("Dropped curves")
        box.setText(f"<b>{led.summary('plotted')}</b>")
        box.setInformativeText("\n".join(lines[2:]))
        box.setDetailedText("\n".join(f"{d.path}\t{d.reason}\t{d.detail}"
                                      for d in led.drops()))
        box.exec()

    def _update_stats(self) -> None:
        """Stats describe the population control's population, which is also
        what is drawn, so the numbers on screen always match the plot
        underneath them — and say what was dropped to get there, so the gap
        between this number and the population summary is accounted for rather
        than left to be noticed."""
        if self._load_error is not None:
            self._stats_label.setText(
                f"Could not load event summary values — {self._load_error}")
            self._stats_label.setToolTip(self._load_error)
        else:
            self._stats_label.setToolTip("")

        # Written before the early returns below: a stale fit line under a
        # "0 events shown" stats line would describe curves that are gone.
        self._fit_label.setText(self._fit_text())
        self._warn_label.setText(self._fishing_note)

        led = self._plottability_ledger()
        self._why_btn.setEnabled(led.n_dropped > 0)
        self._why_btn.setToolTip(led.report() if led.n_dropped else
                                 "Nothing was dropped — every curve is plotted.")
        drop_txt = (f"   |   asked {led.n_asked:,}, "
                    f"{led.n_dropped:,} not plottable" if led.n_dropped else "")
        # The histograms use the robust range, so their tails sit outside the
        # bars while remaining in the scatter and in every number here.  Say so
        # rather than leave the shorter bar count unexplained.
        n_out_y, n_out_x = getattr(self, "_hist_n_out", (0, 0))
        bin_txt = (f"   |   histogram range excludes "
                   f"{n_out_x} X / {n_out_y} Y outliers"
                   if (n_out_x or n_out_y) else "")

        shown = self._plotted_mask()
        n = int(shown.sum())
        if self._load_error is not None:
            return
        if n == 0:
            self._stats_label.setText(
                f"{self._population_summary}   |   0 events plotted{drop_txt}")
            return
        parts = []
        for key, arr in ((self._x_key, self._x_arr), (self._y_key, self._y_arr)):
            v = arr[shown]
            mean, med = (_q(key, s, with_unit=True) for s in (np.mean(v), np.median(v)))
            parts.append(f"{self._axis_label(key)}: mean {mean}  median {med}")
        self._stats_label.setText(
            f"{self._population_summary}   |   {n} plotted{drop_txt}{bin_txt}   |   "
            + "   |   ".join(parts)
        )

    # ── Selection / inspection linking ────────────────────────────────────────

    def _plotted_mask(self) -> np.ndarray:
        """The curves actually on the scatter: the selected population's, with
        both plotted values. The side list, the count in the stats line and
        the scatter itself must not be able to disagree about this."""
        return self._population_mask() & ~np.isnan(self._x_arr) & ~np.isnan(self._y_arr)

    def _rebuild_list(self) -> None:
        """Repopulate the side event-list from what is plotted, preserving the
        selection by path."""
        idx_v = np.where(self._plotted_mask())[0]
        prev_path = None
        if self._selected_index is not None and 0 <= self._selected_index < len(self._results):
            prev_path = self._results[self._selected_index].get("path")

        self._event_list.blockSignals(True)
        self._event_list.clear()
        restore_row = -1
        for row, i in enumerate(idx_v):
            path = self._results[int(i)].get("path") or ""
            tag  = "hit" if self._hit_mask[int(i)] else "non-hit"
            item = QListWidgetItem(f"{Path(path).name}  [{tag}]")
            item.setData(Qt.ItemDataRole.UserRole, int(i))
            self._event_list.addItem(item)
            if path and path == prev_path:
                restore_row = row
        self._event_list.blockSignals(False)

        if restore_row >= 0:
            self._event_list.setCurrentRow(restore_row)
        else:
            # Previously-selected event no longer present (e.g. reclassified).
            self._selected_index = None
            self._sel_marker.hide()
            self._sel_label.setText("")
            self._fit_status.setText("")

    def _on_list_row_changed(self, row: int) -> None:
        item = self._event_list.item(row) if row >= 0 else None
        if item is None:
            return
        i = item.data(Qt.ItemDataRole.UserRole)
        if i is not None:
            self._select_index(int(i), from_list=True)

    def _on_scatter_clicked(self, _scatter, points) -> None:
        # `points` is an array of every spot under the cursor — when zoomed out
        # a single click can land on several.  Just take the first.
        if len(points) == 0:
            return
        i = points[0].data()
        if i is not None:
            self._select_index(int(i), from_list=False)

    def _select_index(self, i: int, from_list: bool) -> None:
        self._selected_index = i
        self._update_sel_marker()
        self._update_sel_readout()
        if not from_list:
            self._event_list.blockSignals(True)
            for row in range(self._event_list.count()):
                if self._event_list.item(row).data(Qt.ItemDataRole.UserRole) == i:
                    self._event_list.setCurrentRow(row)
                    break
            self._event_list.blockSignals(False)

    def _update_sel_marker(self) -> None:
        i = self._selected_index
        if i is None or not (0 <= i < len(self._x_arr)):
            self._sel_marker.hide()
            return
        x = self._x_arr[i]; y = self._y_arr[i]
        if np.isnan(x) or np.isnan(y):
            self._sel_marker.hide()
            return
        self._sel_marker.setData(x=[x], y=[y])
        self._sel_marker.show()

    def _update_sel_readout(self) -> None:
        i = self._selected_index
        o = self._seg_outcome[i] if i is not None and 0 <= i < len(self._seg_outcome) else None
        self._fit_status.setText(
            f"Fit: {fit_outcome_text(o)}" if o and o["fit_status"] else "")
        if i is None or not (0 <= i < len(self._results)):
            self._sel_label.setText("")
            return
        name = Path(self._results[i].get("path") or "").name
        x = self._x_arr[i]; y = self._y_arr[i]
        if np.isnan(x) or np.isnan(y):
            self._sel_label.setText(f"Selected: {name}")
        else:
            self._sel_label.setText(
                f"Selected: {name}   —   "
                f"{self._axis_label(self._x_key)} {_q(self._x_key, x, with_unit=True)}, "
                f"{self._axis_label(self._y_key)} {_q(self._y_key, y, with_unit=True)}"
            )

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _update_cursor(self, index: int) -> None:
        if not (0 <= index < len(self._x_arr)):
            self._hide_cursor()
            self._view_fit_btn.setEnabled(False)
            return
        x = self._x_arr[index]
        y = self._y_arr[index]
        if not np.isnan(x) and not np.isnan(y):
            self._cursor_v.setValue(x); self._cursor_v.show()
            self._cursor_h.setValue(y); self._cursor_h.show()
            self._y_hist_cursor.setValue(y); self._y_hist_cursor.show()
            self._x_hist_cursor.setValue(x);  self._x_hist_cursor.show()
        else:
            self._hide_cursor()
        self._view_fit_btn.setEnabled(
            bool(np.any(~np.isnan(self._force_arr) & ~np.isnan(self._length_arr))))

    def _hide_cursor(self) -> None:
        self._cursor_v.hide()
        self._cursor_h.hide()
        self._y_hist_cursor.hide()
        self._x_hist_cursor.hide()

    def set_criteria_opener(self, cb) -> None:
        """Register the dashboard's Criteria-dialog opener so the in-window
        'Filtering…' button raises the same singleton dialog instead of the
        window owning its own copy."""
        self._criteria_opener = cb

    def _on_open_criteria(self) -> None:
        if self._criteria_opener is not None:
            self._criteria_opener()

    def _on_population_toggled(self, checked: bool, pop: str) -> None:
        """A population button was selected — the others clear via _pop_group.

        Governs what is drawn and every result computed from it (see module
        docstring). Under "both" the 2DH builds and Isoforce take every event:
        an ensemble is a histogram over the curves it is given, and cluster
        structure that crosses the criteria only shows up in a mixed one.
        """
        if not checked:
            return
        self._active_population = pop
        self._fit_chk.setText(f"Linear fit ({self._population_label()})")
        self._rebuild()

    def set_raw_window(self, win) -> None:
        """Register the RawCurveWindow so WlcViewWindow can navigate back to a event."""
        self._raw_win = win

    def set_2dh_window(self, win) -> None:
        """Register a 2DH window.  Triggers an immediate full sync."""
        self._2dh_wins = [old for old in self._2dh_wins
                          if self._window_is_visible(old) and old is not win]
        self._2dh_wins.append(win)
        win.sync_from_event_summary(self)

    def set_results(self, prepass_results: list[dict]) -> None:
        """Called after construction when prepass_results become available."""
        self.reload_paths([r.get("path") for r in prepass_results if r.get("path")])

    def reload_paths(self, paths: list[str]) -> None:
        """
        Re-scope to `paths` and re-read event state from the DB.  Used by the
        dashboard for LIVE updates: as the worker classifies curves, the set of
        queue∩events grows, so the scatter must pick up new points.

        Reallocates the per-curve arrays (the event set can grow past the size set
        at construction) and preserves the user's selection by path.  Does NOT
        re-trigger WLC auto-fitting — the scatter needs only force/length, and
        per-flush curve loads would be far too heavy.
        """
        prev_path = None
        if self._selected_index is not None and 0 <= self._selected_index < len(self._results):
            prev_path = self._results[self._selected_index].get("path")

        self._results    = [{"path": p} for p in paths]
        n                = len(self._results)
        self._force_arr  = np.full(n, np.nan)
        self._length_arr = np.full(n, np.nan)
        self._x_arr      = np.full(n, np.nan)
        self._y_arr      = np.full(n, np.nan)
        self._seg_outcome = [None] * n
        if self._current_index >= n:
            self._current_index = 0

        self._selected_index = None
        if prev_path is not None:
            for i, r in enumerate(self._results):
                if r["path"] == prev_path:
                    self._selected_index = i
                    break

        self._prepopulate()

    # ── Distribution fit pop-outs ─────────────────────────────────────────────

    def _population_label(self) -> str:
        return _ledger.population_label(self._active_population)

    def _live_hit_mask(self) -> np.ndarray:
        """Boolean mask over self._results, asked of the gate RIGHT NOW.

        self._hit_mask is a render cache rebuilt in _rebuild() and used to
        colour the scatter. Anything that produces a RESULT (an export, a
        fit, a 2DH build) must not read that cache, because a missed refresh
        would silently put the wrong curves in a file on disk. Those callers
        come here instead and get the gate's current answer."""
        paths = [r.get("path") for r in self._results]
        live = [p for p in paths if p]
        if not live:
            return np.zeros(len(paths), dtype=bool)
        hits, _non_hits = _gate.evaluate(live, self._db_path)
        hit_set = set(hits)
        return np.array([bool(p) and p in hit_set for p in paths], dtype=bool)

    def _population_mask(self) -> np.ndarray:
        """Boolean mask over self._results for whichever population the scope
        selector currently points to — what every result here is computed
        over. Sourced from the gate at call time, not from the render cache."""
        live = self._live_hit_mask()
        if self._active_population == "both":
            return np.array([bool(r.get("path")) for r in self._results], dtype=bool)
        return live if self._active_population == "hit" else ~live

    def _cluster_caption(self) -> str:
        """The clustering, for the on-canvas caption. Empty when not shown."""
        return self._cluster_bar.legend_text(
            [r.get("path") for r in self._results if r.get("path")])

    def _provenance_caption(self, n: int | None = None, axes=None) -> str:
        """What produced the values a fit window is about to run on —
        population + the Ultimate/Penultimate segment (see _update_title's
        note). Threaded into DistFitWindow/GmmFitWindow as an on-canvas
        caption so it isn't lost the moment a fit pops out into its own
        window. `axes` are the (key, values) fitted; see population_ledger."""
        seg = {"ultimate": "Ultimate", "penultimate": "Penultimate"}.get(
            self._segment_select, self._segment_select or "?"
        )
        parts = [self._population_label(), f"segment: {seg}"]
        if n is not None:
            # Both fit windows put the cohort caption on-canvas and into their
            # export manifest, so saying it here reaches both.
            led = self.population_ledger(self._active_population, axes)
            parts.append(f"{n} points of {led.n_asked} events"
                         if led.n_asked != n else f"{n} points")
            if led.n_dropped:
                parts.append(f"{led.n_dropped} dropped")
        cl = self._cluster_caption()
        if cl:
            parts.append(cl)
        return "   ·   ".join(parts)

    def _paths_for_mask(self, sel: np.ndarray) -> list[str]:
        """The curve paths behind a selection mask, in the same order as the
        values it selects — so a fit window can name the data it fitted."""
        return [self._results[int(i)].get("path", "") for i in np.where(sel)[0]]

    def _on_fit_x(self) -> None:
        self._fit_axis(self._x_key, self._x_arr)

    def _on_fit_y(self) -> None:
        self._fit_axis(self._y_key, self._y_arr)

    def _fit_axis(self, key: str, arr: np.ndarray) -> None:
        sel = self._population_mask() & ~np.isnan(arr)
        valid = arr[sel]
        if len(valid) < 5:
            return
        self._open_fit_window(self._axis_label(key), _quant.unit_of(key), valid,
                              self._paths_for_mask(sel), axes=[(key, arr)])

    def _on_fit_2d(self) -> None:
        xk, yk = self._x_key, self._y_key
        sel = self._population_mask() & ~np.isnan(self._x_arr) & ~np.isnan(self._y_arr)
        pop_xy = np.column_stack([self._x_arr[sel], self._y_arr[sel]])
        if len(pop_xy) < 5:
            return

        pop = self._active_population
        gmm_key = f"{pop}:{xk}:{yk}"
        existing = self._gmm_wins.get(gmm_key)
        if existing is not None and existing.isVisible():
            if getattr(existing, "_event_summary_revision", None) == self._data_revision:
                existing.raise_()
                existing.activateWindow()
                return
            existing.close()

        from .gmm_fit_window import GmmFitWindow
        axes = [(xk, self._x_arr), (yk, self._y_arr)]
        win = GmmFitWindow(pop_xy, self._db_path,
                           caption=self._provenance_caption(len(pop_xy), axes),
                           paths=self._paths_for_mask(sel),
                           x_key=xk, y_key=yk)
        self._gmm_wins[gmm_key] = win
        win._event_summary_revision = self._data_revision
        win.show()

    def _open_fit_window(self, label: str, units: str, pass_values: np.ndarray,
                         paths: list[str] | None = None, axes=None) -> None:
        if len(pass_values) < 5:
            return
        from .dist_fit_window import DistFitWindow
        key = f"{label} ({self._population_label()})"
        existing = self._fit_wins.get(key)
        if existing is not None and existing.isVisible():
            if getattr(existing, "_event_summary_revision", None) == self._data_revision:
                existing.raise_()
                existing.activateWindow()
                return
            existing.close()
        win = DistFitWindow(
            key, units, pass_values, self._db_path,
            caption=self._provenance_caption(len(pass_values), axes),
            paths=paths,
        )
        self._fit_wins[key] = win
        win._event_summary_revision = self._data_revision
        win.show()

    # ── Export ────────────────────────────────────────────────────────────────

    def export_provenance(self, axes=None) -> dict:
        """This window's settings, for an export manifest — the same protocol
        method the 2DH windows implement (base_2dh_window.export_provenance).
        Segment selection is here because it silently decides what every
        seg_* number in this window MEANS; an export that didn't record it
        would be ambiguous the moment the toggle moved. `axes` are the
        (key, values) the export draws on; see population_ledger."""
        # The drop tally travels with every export from this window.
        # A manifest is read months later by someone with no access to the
        # window that produced it, so a row count with nothing saying what it
        # was drawn FROM is exactly the unverifiable claim this issue is
        # about — the same reason the file list is already in here.
        led = self.population_ledger(self._active_population, axes)
        return {
            "window":         "explore_events",
            "population":     self._active_population,
            "segment_select": self._segment_select,
            "x_variable":     self._x_key,
            "y_variable":     self._y_key,
            "fit_shown":      bool(self._fit_chk.isChecked()),
            # How many pairs were on offer, so a reader can judge a p-value
            # that was found by scanning rather than predicted in advance.
            "n_variables_offered": len(self._vars),
            "n_pairs_possible":    len(self._vars) * (len(self._vars) - 1) // 2,
            "n_events_loaded": len(self._results),
            "population_drops": led.manifest(),
            **_clustering.provenance(
                [r.get('path') for r in self._results if r.get('path')],
                self._cluster_bar.is_active()),
        }

    def _on_export_scatter(self) -> None:
        xk, yk = self._x_key, self._y_key
        sel = self._population_mask() & ~np.isnan(self._x_arr) & ~np.isnan(self._y_arr)
        idx = np.where(sel)[0]
        if len(idx) == 0:
            QMessageBox.information(self, "Export scatter",
                                     "No points in the selected population.")
            return
        # Full path, not Path(...).name — a basename can't identify a curve
        # in a catalog where the same filename recurs across directories.
        sel_paths = [self._results[int(i)]["path"] for i in idx]
        # A WLC parameter on an axis travels with its fit uncertainty, read
        # from the same register as the plotted values: without it the export
        # gives a point with no error bar while the number sits in the DB.
        errs = {k: k[:-len("_nm")] + "_err" for k in dict.fromkeys((xk, yk))
                if k.endswith("_nm") and k[:-len("_nm")] + "_err" in _SEGMENT_FIT_KEYS}
        err_cols = (_vars.columns(sel_paths, list(errs.values()), self._db_path)[1]
                    if errs else {})

        # Which population each row belongs to, always — a Both export would
        # otherwise be two cohorts with nothing telling them apart, and a
        # single-population file still gains by saying so in the data.
        live_hit = self._live_hit_mask()
        cols, series = ["path", "hit"], [sel_paths, [int(live_hit[i]) for i in idx]]
        for key, arr in dict(((xk, self._x_arr), (yk, self._y_arr))).items():
            cols.append(key)
            series.append([float(v) for v in arr[idx]])
            if key in errs:
                cols.append(errs[key])
                series.append(["" if np.isnan(v) else float(v)
                               for v in err_cols[errs[key]]])

        # Refitted over THESE rows. They are the analysed population, which is
        # what the screen fit uses too, so the two agree by construction — but
        # an exported fit column must be computed from the file it ships in.
        # Same rule as the screen fit: an axis against itself is an identity,
        # not a result.
        x, y = self._x_arr[idx], self._y_arr[idx]
        fit = None if xk == yk or len(x) < 3 else _reg.linear_fit(x, y)
        corr = (None if xk == yk or len(x) < 3 else
                _reg.correlate(x, y, method="spearman"))
        if fit is not None:
            f_lo, f_hi = fit.band(x)
            f_val = fit.predict(x)
        else:
            f_lo = f_hi = f_val = np.full(x.shape, np.nan)
        cols += ["fit", "fit_ci_lo", "fit_ci_hi", "cluster"]
        # The cluster travels with the row whether or not the colouring is
        # switched on — a display toggle must not decide what a data file
        # contains, and the clustering itself does not survive the session.
        cl = _clustering.current()
        series += [
            [float(v) for v in f_val], [float(v) for v in f_lo],
            [float(v) for v in f_hi],
            ["" if cl is None or cl.label_for(p) is None else cl.label_for(p)
             for p in sel_paths],
        ]
        rows = list(zip(*series))
        axes = [(xk, self._x_arr), (yk, self._y_arr)]
        with _export.export_group(
            self._db_path,
            f"scatter_{_export.slug(yk)}_vs_{_export.slug(xk)}_{self._active_population}",
            [".csv"], kind="scatter_xy",
        ) as g:
            g.contributing_files(sel_paths)
            g.note_dict(self.export_provenance(axes))
            g.note_dict(_reg.manifest_fields(fit, corr, x_is_time=xk == _vars.TIME_KEY))
            g.note(columns=cols, units=_quant.units_for(cols[1:]), n_points=len(rows))
            g.table(".csv", cols, rows)
        QMessageBox.information(
            self, "Export scatter", f"{len(rows)} points.\n\n{g.message()}")

    # One row per (curve, ROI, segment).  The header names are the file's
    # column names, so they carry their own units and stay plain text — a
    # manifest and CSV are read by scripts rather than rendered by pyqtgraph.
    _ROI_SEGMENT_COLUMNS = [
        ("path",              "path"),
        ("roi_index",         "roi_index"),
        ("n_ruptures",        "n_ruptures"),
        ("ordering",          "ordering"),
        ("seg_index",         "seg_index"),
        ("position",          "position"),
        ("l_p_nm",            "l_p_nm"),
        ("l_p_err_nm",        "l_p_err"),
        ("l_c_nm",            "l_c_nm"),
        ("l_c_err_nm",        "l_c_err"),
        ("n_fit_pts",         "n_fit_pts"),
        ("fit_status",        "fit_status"),
        ("fit_detail",        "fit_detail"),
        ("rupture_force_pN",  "rupture_force_pN"),
        ("dX_from_prev_nm",   "dX_from_prev_nm"),
        ("dF_from_prev_pN",   "dF_from_prev_pN"),
    ]

    def _on_export_roi_segments(self) -> None:
        """Export every ROI segment of every curve in the selected population.

        This is the finer grain of the same cohort every other export here
        uses: the queue table and the classification report are one row per
        CURVE, and a curve's inner ruptures — the sub-events the multi-event
        fitter already found and stored.

        Reads only what `event_map` already holds: `assemble_rows` projects
        stored documents, it never runs the detector or the fitter. So this
        button cannot start a batch job, which is precisely what the window
        it replaces did.

        Curves with no stored document under the CURRENT parameter set are
        skipped by `assemble_rows` — they are counted here and reported both
        on screen and in the manifest, rather than silently shrinking the
        cohort.
        """
        title = "Export ROI/segment rows"
        paths = self._paths_for_mask(self._population_mask())
        paths = [p for p in paths if p]
        if not paths:
            QMessageBox.information(self, title,
                                     "No curves in the selected population.")
            return

        from .roi_pipeline import assemble_rows
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            rows = assemble_rows(paths, self._db_path, mode="all")
        finally:
            QApplication.restoreOverrideCursor()

        if not rows:
            QMessageBox.information(
                self, title,
                f"None of the {len(paths)} curves in this population has "
                "stored ROI segments under the current parameter set.\n\n"
                "Run the analysis over them first."
            )
            return

        # Same reason as the scatter export's: under Both these rows are two
        # cohorts, and nothing else in the file says which is which.
        live_hit = self._live_hit_mask()
        hit_by_path = {r.get("path"): int(live_hit[i])
                       for i, r in enumerate(self._results) if r.get("path")}
        for row in rows:
            row["hit"] = hit_by_path.get(row.get("path"), "")

        covered = {r.get("path") for r in rows if r.get("path")}
        headers = [h for h, _k in self._ROI_SEGMENT_COLUMNS] + ["hit"]
        keys    = [k for _h, k in self._ROI_SEGMENT_COLUMNS] + ["hit"]
        with _export.export_group(
            self._db_path,
            f"roi_segments_{self._active_population}",
            [".csv"], kind="roi_segment_rows",
        ) as g:
            g.contributing_files(covered)
            g.note_dict(self.export_provenance())
            g.note(
                grain="one row per (curve, ROI, segment)",
                columns=headers,
                column_keys=keys,
                n_rows=len(rows),
                n_curves_in_population=len(paths),
                n_curves_with_stored_segments=len(covered),
                n_curves_without_stored_segments=len(paths) - len(covered),
            )
            g.dict_table(".csv", headers, keys, rows)
        QMessageBox.information(
            self, title,
            f"{len(rows)} rows from {len(covered)} of {len(paths)} curves "
            f"({len(paths) - len(covered)} had no stored ROI segments under "
            f"the current parameter set).\n\n{g.message()}"
        )

    def _on_export_x_hist(self) -> None:
        self._export_histogram(self._x_key, self._x_arr)

    def _on_export_y_hist(self) -> None:
        self._export_histogram(self._y_key, self._y_arr)

    def _export_histogram(self, key: str, arr: np.ndarray) -> None:
        stem_suffix, label = _export.slug(key), self._axis_label(key)
        sel = self._population_mask() & ~np.isnan(arr)
        values = arr[sel]
        title = f"Export {label} histogram"
        if len(values) == 0:
            QMessageBox.information(self, title, "No values in the selected population.")
            return
        # Record the curves behind the bins so the exported cohort is recoverable.
        contributing = [
            self._results[int(i)]["path"]
            for i in np.where(sel)[0]
            if self._results[int(i)].get("path")
        ]
        bins = _hb.full_range_bins(values)
        if bins is None:
            QMessageBox.information(self, title, "Nothing to bin.")
            return
        counts = bins.count(values)
        with _export.export_group(
            self._db_path,
            f"hist_{stem_suffix}_{self._active_population}",
            [".csv"], kind=f"histogram_{stem_suffix}",
        ) as g:
            g.contributing_files(contributing)
            g.note_dict(self.export_provenance([(key, arr)]))
            # A histogram has no rows to carry a hit column, so the split goes
            # in the manifest: under Both these bars hold both populations.
            binned_hit = self._live_hit_mask()[sel]
            g.note(
                variable=key,
                unit=_quant.unit_of(key),
                n_values=int(len(values)),
                n_hit=int(np.count_nonzero(binned_hit)),
                n_non_hit=int(np.count_nonzero(~binned_hit)),
                n_binned=int(counts.sum()),
                n_bins=int(bins.n_bins),
                bin_range=[float(bins.edges[0]), float(bins.edges[-1])],
            )
            g.histogram(".csv", bins.edges, counts)
        QMessageBox.information(
            self, title,
            f"{bins.n_bins} bins; {int(counts.sum())} of {len(values)} values "
            f"(all included, none dropped) from {len(contributing)} files.\n\n"
            f"{g.message()}"
        )

    # ── 2D-histogram windows ──────────────────────────────────────────────────

    def _experimentalist_id(self) -> str | None:
        """
        The experimentalist whose data is in view.
        The 2DH windows persist their grid settings under this identity, matching
        the analysis_runner path so settings carry across both launch routes.
        Derived once from the cohort's first file; None if it can't be resolved.
        Cohorts spanning >1 experimentalist are rare and simply key off the first.
        """
        if not self._experimentalist_resolved:
            self._experimentalist_resolved = True
            for r in self._results:
                path = r.get("path")
                if not path:
                    continue
                try:
                    who = _db.get_experimentalist_for_file(path, self._db_path)
                except Exception:
                    who = None
                if who:
                    self._experimentalist = who
                break
        return self._experimentalist

    def population_ledger(self, which: str, axes=None) -> _ledger.Ledger:
        """Who is in `which` population, and why anyone else is not.

        Membership in a population ("is it a hit") and completeness ("does it
        have the values this consumer needs") are different questions with
        different remedies — retune the criteria vs. re-analyse the curve —
        and answering both with one list meant a 2DH received an
        already-filtered cohort and could not tell that it had been filtered.

        `axes` is the (key, values) list a curve must have. The default is the
        selected segment's rupture force and contour length: the downstream
        cohort, which does not follow the plotted axes (module docstring).

        `asked` is the whole loaded events population, so the ledger reports
        against the number in the visible population summary rather than an
        already-narrowed set. Membership comes from the gate at call time,
        not from this window's render cache; a 2DH built from a
        stale mask would be a wrong figure, not just a wrong-looking screen.
        """
        paths = [r.get("path") or "" for r in self._results]
        led = _ledger.Ledger("Explore Events population", paths)
        if axes is None:
            axes = [("seg_force_pN", self._force_arr), ("seg_l_c_nm", self._length_arr)]

        live = self._live_hit_mask()
        # "both" asks of every loaded event, so nothing is dropped for
        # membership and the ledger reports only missing values.
        mask = (np.ones(len(paths), dtype=bool) if which == "both"
                else live if which == "hit" else ~live)
        other = "non-hit" if which == "hit" else "hit"
        for i, p in enumerate(paths):
            if not p:
                continue
            if not mask[i]:
                led.drop(p, "not_in_population", other)
                continue
            self._record_missing(led, i, p, axes)
        return led

    def population_paths(self, which: str) -> list[str]:
        """Paths with a usable segment fit belonging to `which` population
        ("hit"/"non_hit"/"both"), independent of the population control's
        CURRENT setting. A 2DH window remembers which population it was opened
        for and asks for exactly that one on every refresh, so it stays
        correctly scoped even after the control moves underneath it.

        The survivors of population_ledger() — callers wanting to report what
        they were given, not only what they got, should ask for the ledger
        instead. Kept as the convenience form because most callers genuinely
        only need the list."""
        return self.population_ledger(which).kept()

    def _isoforce_paths(self, which: str) -> list[str]:
        """Subset of population_paths(which) with a usable adjacent isoforce
        pair — the current manual pair when complete, otherwise the last two
        ruptures. In concrete terms, roi_pipeline.segment_summary_bulk's
        dX_iso_nm is non-None (the same rule IsoforceWindow uses to draw).
        Sorted by measured date, like
        _current_event_paths."""
        paths = self.population_paths(which)
        if not paths:
            return []
        from .roi_pipeline import segment_summary_bulk, read_segment_select
        seg = segment_summary_bulk(paths, read_segment_select(self._db_path), self._db_path)
        qualifying = [
            p for p in paths
            if seg.get(_db.normalize_path(p), {}).get("dX_iso_nm") is not None
        ]
        dates = _db.get_measured_dates(qualifying, self._db_path)
        return sorted(qualifying, key=lambda p: (dates.get(p) or "", p))

    def _on_open_normalized_2dh(self) -> None:
        pop = self._active_population
        win = self._norm_2dh_wins.get(pop)
        if win is not None and win.isVisible():
            win.raise_(); win.activateWindow()
            return
        self._drop_2dh_window(win)   # discard a stale, closed one
        from .normalized_2dh_window import Normalized2DHWindow
        win = Normalized2DHWindow(
            self._results, self._db_path, self._session_info,
            experimentalist=self._experimentalist_id(),
            population=pop,
        )
        # Show the (empty) window BEFORE syncing: the build can take many seconds
        # when per-curve histograms must be recomputed, and its "Building… i/n"
        # progress is only useful if the window is already on screen.  Building
        # invisibly looks like a frozen app (and gets the user restarting).
        win.show()
        QApplication.processEvents()
        self._norm_2dh_wins[pop] = win
        # Registers + syncs this population, and keeps it fed on later changes.
        self.set_2dh_window(win)

    def _on_open_physical_2dh(self) -> None:
        pop = self._active_population
        win = self._phys_2dh_wins.get(pop)
        if win is not None and win.isVisible():
            win.raise_(); win.activateWindow()
            return
        self._drop_2dh_window(win)   # discard a stale, closed one
        from .physical_2dh_window import Physical2DHWindow
        win = Physical2DHWindow(
            self._results, self._db_path, self._session_info,
            experimentalist=self._experimentalist_id(),
            population=pop,
        )
        # Show before syncing so the "Building… i/n" progress is visible during
        # the (possibly long) histogram build — see _on_open_normalized_2dh.
        win.show()
        QApplication.processEvents()
        self._phys_2dh_wins[pop] = win
        self.set_2dh_window(win)

    def _drop_2dh_window(self, win) -> None:
        """Unregister a (typically closed) 2DH window so it stops receiving syncs."""
        if win is not None and win in self._2dh_wins:
            self._2dh_wins.remove(win)

    # ── WLC view ──────────────────────────────────────────────────────────────

    def _current_event_paths(self) -> list[str]:
        """The selected population's curves with a selected-segment rupture
        force and contour length — the WLC navigator's cohort, which keeps
        that requirement whatever the plotted axes are (module docstring)."""
        sel = (self._population_mask()
               & ~np.isnan(self._force_arr) & ~np.isnan(self._length_arr))
        paths = [p for i in np.where(sel)[0]
                 if (p := self._results[int(i)].get("path"))]
        dates = _db.get_measured_dates(paths, self._db_path)
        return sorted(paths, key=lambda p: (dates.get(p) or "", p))

    def _on_view_fit(self) -> None:
        from .wlc_view_window import WlcViewWindow
        event_paths = self._current_event_paths()
        if not event_paths:
            return
        # Open at the user's selection if there is one, else the playhead curve.
        target = self._selected_index if self._selected_index is not None else self._current_index
        current_path = self._results[target].get("path")
        try:
            nav_index = event_paths.index(current_path)
        except ValueError:
            nav_index = 0
        self._wlc_view = WlcViewWindow(
            event_paths, nav_index, self._db_path, self._session_info,
            two_dh_win=self._2dh_wins[0] if self._2dh_wins else None,
            raw_window=self._raw_win,
        )
        self._wlc_view.show()

    def _on_view_isoforce(self) -> None:
        from .isoforce_window import IsoforceWindow
        pop = self._active_population
        paths = self._isoforce_paths(pop)
        if not paths:
            other = "non_hit" if pop == "hit" else "hit"
            other_name = "Non-Hits" if pop == "hit" else "Hits"
            msg = (f"No curves in the current {self._population_label()} "
                   "population have a usable adjacent isoforce pair.")
            n_other = len(self._isoforce_paths(other))
            if n_other:
                msg += (f"\n\n{n_other} curve(s) in {other_name} do. "
                        f"Switch to {other_name} to view them.")
            QMessageBox.information(self, "Isoforce", msg)
            return
        target = self._selected_index if self._selected_index is not None else self._current_index
        current_path = self._results[target].get("path") if 0 <= target < len(self._results) else None
        try:
            nav_index = paths.index(current_path)
        except ValueError:
            nav_index = 0
        if self._isoforce_win is not None and self._isoforce_win.isVisible():
            self._isoforce_win.close()
        self._isoforce_win = IsoforceWindow(
            paths, nav_index, self._db_path, self._session_info,
            raw_window=self._raw_win, population=pop,
        )
        self._isoforce_win.show()

    # WLC fitting is an analysis step persisted in event_map by
    # roi_events.fit_segments
    # during the worker's own pass; this window has no business re-running it,
    # on a raw-signal fitter, as a side effect of being viewed. `_fit_status`
    # shows the selected segment's stored outcome, read in _prepopulate.


    def closeEvent(self, event):
        self._cluster_bar.detach()
        super().closeEvent(event)
