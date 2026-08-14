# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/dist_fit_window.py
#
# DistFitWindow — pop-out histogram fitting window for a single Stats variable.
#
# Opened from AnalysisWindow ("Fit…" button on the Raw or Derived tab).
# Receives the pass-only values directly; no file loading, no column selector.
# One window per variable; re-opening the same variable raises the existing window.
#
# Sandbox: add/remove peak components, re-fit freely.
# Model comparison table is quantitative (AICc, BIC) — not visual.
# Prior saved fits for this variable are loaded from DB at open time so the
# comparison table is pre-populated with historical context.
# "Save fit" commits the current result to the distribution_fits DB table.

from __future__ import annotations

import json

import numpy as np
from scipy import optimize

import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from . import style
from . import db as _db
from . import histogram_binning as _hb
from . import quantities as _quant
from . import export_utils as _export
from .export_utils import slug as _slug
from .dist_fit_core import (
    MODELS, MODEL_NAMES, ci_manifest_fields, bootstrap_fit_ci, CI_N_DRAWS,
    centre_permutation, composite_bounds, composite_guess, fit_stats,
    make_composite, sort_components_by_centre,
)
from . import dist_fit_core as _dfc
from .qt_utils import CancelableProgress, set_plot_title, set_si_label, fit_on_screen


PEAK_COLORS = list(style.SERIES_LABELED)
HIST_BRUSH = style.rgba(style.DATA, 150)
TOTAL_PEN = pg.mkPen(style.INK, width=style.W_MODEL)


class PeakRow(QFrame):
    """A model component and its optional starting-position hint.

    A typed or picked position replaces only the automatic initial guess; it
    does not constrain the fitted parameter. The guess calculation itself
    remains in :mod:`dist_fit_core`.
    """

    remove_clicked = pyqtSignal(object)
    pick_requested = pyqtSignal(object)
    start_changed = pyqtSignal(object)
    AUTO = -1e12

    def __init__(self, index: int, model_name: str, color: str, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.color = color
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        dot = QLabel("●")
        dot.setStyleSheet(style.qss_text(color, size_px=style.DOT_LABEL_SIZE_PX))
        lay.addWidget(dot)
        self._lbl = QLabel()
        lay.addWidget(self._lbl, 1)
        lay.addWidget(QLabel("start:"))
        self._start = QDoubleSpinBox()
        self._start.setRange(self.AUTO, 1e12)
        self._start.setValue(self.AUTO)
        self._start.setSpecialValueText("auto")
        _quant.configure_spinbox(self._start, decimals=4, suffix=False)
        self._start.setToolTip(
            "Where this component starts the fit — a hint, not a pin.\n"
            "The fit remains free to move it. 'auto' uses peak detection."
        )
        self._start.valueChanged.connect(lambda _v: self.start_changed.emit(self))
        lay.addWidget(self._start)
        self._pick = QPushButton("⌖")
        self._pick.setFixedWidth(28)
        self._pick.setCheckable(True)
        self._pick.setToolTip(
            "Click a position on the histogram to set the starting point."
        )
        self._pick.clicked.connect(lambda: self.pick_requested.emit(self))
        lay.addWidget(self._pick)
        remove_btn = QPushButton("✕")
        remove_btn.setFixedWidth(28)
        remove_btn.clicked.connect(lambda: self.remove_clicked.emit(self))
        lay.addWidget(remove_btn)
        self.set_index(index)

    def set_index(self, index: int) -> None:
        self.index = index
        self._lbl.setText(f"Peak {index + 1}:  {self.model_name}")

    @property
    def start(self) -> float | None:
        value = float(self._start.value())
        return None if value == self.AUTO else value

    def set_start(self, value: float | None) -> None:
        self._start.blockSignals(True)
        self._start.setValue(self.AUTO if value is None else float(value))
        self._start.blockSignals(False)

    @property
    def picking(self) -> bool:
        return self._pick.isChecked()

    def set_picking(self, on: bool) -> None:
        self._pick.setChecked(bool(on))


# ── Plot pane ─────────────────────────────────────────────────────────────────

class _PlotPane(QWidget):
    """
    Histogram + residuals subplot for a fixed dataset.
    Data is passed in at construction; no file loading.
    """

    # x of a click made while the pane is armed for picking. Emitted
    # once and then disarmed: picking is a deliberate act, not a mode the user
    # can forget they are in and then move a component by clicking to pan.
    position_picked = pyqtSignal(float)

    def __init__(self, data: np.ndarray, label: str, units: str, caption: str = "", parent=None):
        super().__init__(parent)

        self._caption:      str              = caption
        self._data:        np.ndarray       = data
        self._label:       str              = label
        self._units:       str              = units
        self._bin_centers: np.ndarray | None = None
        self._density:     np.ndarray | None = None
        self._edges:       np.ndarray | None = None
        self._bins:        _hb.HistogramBins | None = None
        self._bin_width:   float             = 1.0

        self._hist_item:       pg.BarGraphItem | None = None
        self._main_fit_items:  list = []
        self._res_items:       list = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        # Controls
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Bins:"))
        self._bins_spin = QSpinBox()
        # The ceiling is deliberately huge. Whenever the range is
        # automatic, bin count is the ONLY resolution knob, so a low cap caps
        # the only control there is.
        self._bins_spin.setRange(1, _hb.MAX_USER_BINS)
        _quant.configure_spinbox(self._bins_spin)
        self._bins_spin.setValue(20)
        self._bins_spin.setToolTip(
            "Number of histogram bins.\n"
            "More bins = finer detail but noisier bars.\n"
            "For ~100 measurements, 15-25 bins is usually good.\n\n"
            "The fit is computed on these bins, so a very fine binning makes the "
            "fit itself noisier. The confidence interval widens honestly when it "
            "does."
        )
        self._bins_spin.valueChanged.connect(self._replot)
        ctrl.addWidget(self._bins_spin)
        ctrl.addSpacing(12)

        # Explicit range. Unchecked, the edges span the data's own
        # min/max — exactly what np.histogram(values, bins=n) did before — so
        # the default answer is unchanged and only a deliberate narrowing
        # changes anything.
        self._auto_range_chk = QCheckBox("Auto range")
        self._auto_range_chk.setChecked(True)
        self._auto_range_chk.setToolTip(
            "Span the full range of the values.\n"
            "Uncheck to set the range by hand — but note that values outside\n"
            "it leave the FIT as well as the picture, because the fit sees\n"
            "the bins and nothing else. The count is reported beside this."
        )
        self._auto_range_chk.toggled.connect(self._on_auto_range_toggled)
        ctrl.addWidget(self._auto_range_chk)

        finite = self._data[np.isfinite(self._data)]
        lo_seed = float(finite.min()) if finite.size else 0.0
        hi_seed = float(finite.max()) if finite.size else 1.0
        self._range_lo_spin = QDoubleSpinBox()
        self._range_hi_spin = QDoubleSpinBox()
        for spin, seed in ((self._range_lo_spin, lo_seed),
                           (self._range_hi_spin, hi_seed)):
            spin.setRange(-1e12, 1e12)
            # decimals_for widens the box to hold the seeded value exactly
            # rather than rounding it — the "widen, never round" rule.
            _quant.configure_spinbox(
                spin, decimals=_quant.decimals_for(self._label, seed), suffix=False)
            spin.setValue(seed)
            spin.setEnabled(False)
            spin.valueChanged.connect(self._replot)
        ctrl.addWidget(QLabel("min:"))
        ctrl.addWidget(self._range_lo_spin)
        ctrl.addWidget(QLabel("max:"))
        ctrl.addWidget(self._range_hi_spin)
        ctrl.addSpacing(12)
        self._norm_chk = QCheckBox("Normalize to density")
        self._norm_chk.setChecked(True)
        self._norm_chk.setToolTip(
            "Must be checked for the fitted curves to have the correct scale.\n"
            "Uncheck only to see raw counts."
        )
        self._norm_chk.toggled.connect(self._replot)
        ctrl.addWidget(self._norm_chk)
        self._last_fit = None
        self._ci_chk = QCheckBox("Show 95% CI")
        self._ci_chk.setChecked(True)
        self._ci_chk.setToolTip(
            "Shade the 95% confidence band of the total fitted curve.\n"
            "Bootstrapped from the measured values — the same resamples the\n"
            "results table reports as '95% CI' and the export writes out."
        )
        self._ci_chk.toggled.connect(self._redraw_fit)
        ctrl.addWidget(self._ci_chk)
        ctrl.addStretch()
        # Says what the range is costing, right where the range is set.  Blank
        # when nothing is excluded, so it reads as a warning rather than as
        # furniture.
        self._excluded_lbl = QLabel("")
        self._excluded_lbl.setStyleSheet(style.qss_text(style.TEXT_WARNING))
        ctrl.addWidget(self._excluded_lbl)
        lay.addLayout(ctrl)

        # Graphics: main plot (top) + residuals (bottom)
        self._gw = pg.GraphicsLayoutWidget()
        lay.addWidget(self._gw, 1)

        self._plot = self._gw.addPlot(row=0, col=0)
        # si=False: the results table beside this plot prints every fitted
        # peak position in the plain unit, so the axis must not float to a
        # different prefix and show the same fit at two scales.
        set_si_label(self._plot, "bottom", style.mathify(label), units, si=False)
        self._plot.setLabel("left", "Density")
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._legend = self._plot.addLegend(offset=(10, 10))
        # On-canvas (survives pyqtgraph's Export…, unlike the header QLabel
        # below) — what population/segment this fit's pass_values came from.
        set_plot_title(self._plot, caption=self._caption)

        self._res = self._gw.addPlot(row=1, col=0)
        self._res.setLabel("left", "Residual")
        self._res.setMaximumHeight(110)
        self._res.setXLink(self._plot)
        self._res.showGrid(x=True, y=True, alpha=0.25)
        self._res.addItem(
            pg.InfiniteLine(
                pos=0, angle=0, pen=style.hair_pen()
            )
        )

        self._gw.ci.layout.setRowStretchFactor(0, 4)
        self._gw.ci.layout.setRowStretchFactor(1, 1)

        self._picking = False
        self._plot.scene().sigMouseClicked.connect(self._on_scene_click)

        self._replot()

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def fit_data(self) -> np.ndarray:
        """Finite values inside the histogram range, i.e. the fitted sample."""
        return self._data[self.fit_mask]

    @property
    def fit_mask(self) -> np.ndarray:
        if self._edges is None:
            return np.zeros(self._data.shape, dtype=bool)
        return (
            np.isfinite(self._data)
            & (self._data >= self._edges[0])
            & (self._data <= self._edges[-1])
        )

    @property
    def bin_centers(self) -> np.ndarray | None:
        return self._bin_centers

    @property
    def density(self) -> np.ndarray | None:
        return self._density

    @property
    def bin_width(self) -> float:
        return self._bin_width

    @property
    def edges(self) -> np.ndarray | None:
        """The bin EDGES as plotted.  The bootstrap re-bins every resample on
        these same edges — see bootstrap_fit_ci for why they must not move."""
        return self._edges

    @property
    def normalized(self) -> bool:
        return self._norm_chk.isChecked()

    @property
    def bins(self) -> "_hb.HistogramBins | None":
        """The geometry as drawn, including how many values it excluded."""
        return self._bins

    @property
    def range_is_auto(self) -> bool:
        return self._auto_range_chk.isChecked()

    @property
    def n_bins(self) -> int:
        return self._bins_spin.value()

    # ── Fit overlay ───────────────────────────────────────────────────────────

    def show_fit(self, x_fine, y_total, y_components, colors, labels,
                 y_fit_at_bins, ci_band=None):
        # Remembered so toggling the CI checkbox can redraw without
        # re-running the fit (the fit is unchanged; only its rendering is).
        self._last_fit = (x_fine, y_total, y_components, colors, labels,
                          y_fit_at_bins, ci_band)
        self.clear_fit()
        self._legend.clear()

        # 95% band on the total fit, drawn UNDER the curves so it reads as
        # uncertainty around them rather than as another series.
        if ci_band is not None and self._ci_chk.isChecked():
            lo_b, hi_b = ci_band
            up = pg.PlotDataItem(x_fine, hi_b)
            lo = pg.PlotDataItem(x_fine, lo_b)
            band = pg.FillBetweenItem(up, lo,
                                      brush=style.band_brush(style.INK, alpha=45))
            self._plot.addItem(band)
            self._main_fit_items.append(band)

        for y_comp, color, label in zip(y_components, colors, labels):
            item = self._plot.plot(
                x_fine, y_comp,
                pen=style.guide_pen(color, width=style.W_MODEL, alpha=style.A_MODEL),
                name=label,
            )
            self._main_fit_items.append(item)

        item = self._plot.plot(x_fine, y_total, pen=TOTAL_PEN, name="Total fit")
        self._main_fit_items.append(item)

        residuals = self._density - y_fit_at_bins
        pos       = residuals >= 0
        bw        = self._bin_width * 0.7

        # ±1σ counting-noise reference on the residual panel.  Without it the
        # residual bars have no scale: a reader cannot tell a bar that is
        # ordinary Poisson scatter from one that is real, systematic misfit.
        # σ is the counting error of each bin, carried into whatever units the
        # residual is currently in (density or raw counts).
        n_tot = float(self.fit_data.size)
        if n_tot > 0 and self._bin_width > 0:
            counts = (self._density * n_tot * self._bin_width
                      if self._norm_chk.isChecked() else self._density)
            sigma = np.sqrt(np.clip(counts, 0.0, None))
            if self._norm_chk.isChecked():
                sigma = sigma / (n_tot * self._bin_width)
            up_s = pg.PlotDataItem(self._bin_centers, sigma)
            lo_s = pg.PlotDataItem(self._bin_centers, -sigma)
            sband = pg.FillBetweenItem(
                up_s, lo_s, brush=style.band_brush(style.INK_MUTED, alpha=45))
            self._res.addItem(sband)
            self._res_items.append(sband)

        if pos.any():
            r = pg.BarGraphItem(
                x=self._bin_centers[pos], height=residuals[pos],
                width=bw, brush=pg.mkBrush(*style.rgba(style.STATUS_GOOD, 180)),
                pen=pg.mkPen(None),
            )
            self._res.addItem(r)
            self._res_items.append(r)

        neg = ~pos
        if neg.any():
            r = pg.BarGraphItem(
                x=self._bin_centers[neg], height=residuals[neg],
                width=bw, brush=pg.mkBrush(*style.rgba(style.STATUS_CRITICAL, 180)),
                pen=pg.mkPen(None),
            )
            self._res.addItem(r)
            self._res_items.append(r)

    def arm_pick(self, on: bool) -> None:
        """Next click on the histogram reports its x, instead of panning."""
        self._picking = bool(on)
        self._gw.setCursor(Qt.CursorShape.CrossCursor if on
                           else Qt.CursorShape.ArrowCursor)

    def _on_scene_click(self, ev) -> None:
        if not self._picking:
            return
        vb = self._plot.getViewBox()
        if vb is None or not self._plot.sceneBoundingRect().contains(ev.scenePos()):
            return
        x = float(vb.mapSceneToView(ev.scenePos()).x())
        ev.accept()
        self.arm_pick(False)
        self.position_picked.emit(x)

    def _redraw_fit(self, _checked=False):
        if getattr(self, "_last_fit", None) is not None:
            self.show_fit(*self._last_fit)

    def clear_fit(self):
        self._last_fit = None
        for item in self._main_fit_items:
            self._plot.removeItem(item)
        self._main_fit_items.clear()
        for item in self._res_items:
            self._res.removeItem(item)
        self._res_items.clear()
        self._legend.clear()

    # ── Private ───────────────────────────────────────────────────────────────

    def _on_auto_range_toggled(self, auto: bool) -> None:
        """Hand control of the range over, seeded with what is on screen.

        Seeding from the current edges rather than from a default means
        unchecking the box never moves the histogram — it only makes the range
        editable, so the first thing the user sees is the picture they already
        had.
        """
        for spin in (self._range_lo_spin, self._range_hi_spin):
            spin.setEnabled(not auto)
        if not auto and self._edges is not None:
            for spin, v in ((self._range_lo_spin, float(self._edges[0])),
                            (self._range_hi_spin, float(self._edges[-1]))):
                spin.blockSignals(True)
                spin.setValue(v)
                spin.blockSignals(False)
        self._replot()

    def _replot(self):
        # user_bins with no range is exactly np.histogram(values, bins=n); the
        # difference is that it also reports what a NARROWED range excluded.
        auto = self._auto_range_chk.isChecked()
        bins = _hb.user_bins(
            self._data, self._bins_spin.value(),
            None if auto else float(self._range_lo_spin.value()),
            None if auto else float(self._range_hi_spin.value()),
        )
        if bins is None:
            return
        edges = bins.edges
        counts = bins.count(self._data)
        self._bins = bins
        self._bin_width  = edges[1] - edges[0]
        self._edges      = edges
        self._bin_centers = (edges[:-1] + edges[1:]) / 2

        # The fit is computed on these bins, so an excluded value is excluded
        # from the FIT, not just from the drawing.  Never let that be silent.
        n_out = bins.n_out_of_range
        self._excluded_lbl.setText(
            "" if n_out == 0 else
            f"⚠ {n_out} of {int(np.isfinite(self._data).sum())} values are "
            f"outside this range — excluded from the fit"
        )

        if self._norm_chk.isChecked():
            total         = counts.sum() * self._bin_width
            self._density = counts / max(total, 1e-10)
            self._plot.setLabel("left", "Density")
        else:
            self._density = counts.astype(float)
            self._plot.setLabel("left", "Counts")

        if self._hist_item is not None:
            self._plot.removeItem(self._hist_item)

        self._hist_item = pg.BarGraphItem(
            x=self._bin_centers,
            height=self._density,
            width=self._bin_width * 0.92,
            brush=pg.mkBrush(*HIST_BRUSH),
            pen=pg.mkPen(style.INK_MUTED, width=0.8),
        )
        self._plot.addItem(self._hist_item)
        self.clear_fit()


# ── Model pane ────────────────────────────────────────────────────────────────

class _ModelPane(QWidget):
    """
    Model builder, fit results, model comparison, and DB save.
    Mirrors the standalone ModelPane but saves to the catalog DB.
    """

    def __init__(
        self,
        plot_pane:     _PlotPane,
        variable:      str,
        units:         str,
        db_path:       str,
        parent=None,
        paths:         list[str] | None = None,
        caption:       str = "",
    ):
        super().__init__(parent)
        self._plot     = plot_pane
        self._variable = variable
        self._units    = units
        self._db_path  = db_path
        self._paths    = list(paths) if paths else []
        self._caption  = caption
        self._rows:     list[PeakRow] = []
        self._last_fit: dict          = {}
        self._pick_row: PeakRow | None = None
        plot_pane.position_picked.connect(self._on_position_picked)
        self._fit_history: list[dict] = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        # ── Model Builder ─────────────────────────────────────────────────────
        bg = QGroupBox("Model Builder")
        bl = QVBoxLayout(bg)

        add_row = QHBoxLayout()
        add_row.addWidget(QLabel("Add peak:"))
        self._model_combo = QComboBox()
        self._model_combo.addItems(MODEL_NAMES)
        add_row.addWidget(self._model_combo, 1)
        add_btn = QPushButton("+ Add")
        add_btn.clicked.connect(self._add_peak)
        add_row.addWidget(add_btn)
        bl.addLayout(add_row)

        self._peaks_container = QWidget()
        self._peaks_lay = QVBoxLayout(self._peaks_container)
        self._peaks_lay.setContentsMargins(0, 0, 0, 0)
        self._peaks_lay.setSpacing(3)
        self._peaks_lay.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(self._peaks_container)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(80)
        scroll.setMaximumHeight(160)
        bl.addWidget(scroll)

        fit_btn = QPushButton("▶  Fit!")
        fit_btn.setStyleSheet(style.QSS_PRIMARY_ACTION)
        fit_btn.clicked.connect(self._fit)
        bl.addWidget(fit_btn)
        lay.addWidget(bg)

        # ── Fit Results ───────────────────────────────────────────────────────
        rg = QGroupBox("Fit Results")
        rl = QVBoxLayout(rg)

        self._params_tbl = QTableWidget(0, 4)
        self._params_tbl.setHorizontalHeaderLabels(
            ["Parameter", "Value", "± Std Err", "95% CI"]
        )
        self._params_tbl.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._params_tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._params_tbl.setAlternatingRowColors(True)
        self._params_tbl.setMinimumHeight(110)
        rl.addWidget(self._params_tbl)

        self._stats_box = QTextEdit()
        self._stats_box.setReadOnly(True)
        self._stats_box.setMaximumHeight(130)
        self._stats_box.setFont(style.font(
            self._stats_box.font(), size_pt=style.FONT_SMALL_PT, mono=True))
        rl.addWidget(self._stats_box)

        btn_row = QHBoxLayout()
        self._save_btn = QPushButton("Save fit to DB")
        self._save_btn.setEnabled(False)
        self._save_btn.setToolTip("Save current fit result to the catalog database.")
        self._save_btn.clicked.connect(self._save_fit)
        btn_row.addWidget(self._save_btn)
        # "Save fit to DB" keeps the result INSIDE the app (distribution_fits,
        # read back by this window's comparison table). "Export…" gets it OUT,
        # into the report/publication the fit was run for. Two different
        # destinations; both are needed.
        self._export_btn = QPushButton("Export…")
        self._export_btn.setEnabled(False)
        self._export_btn.setToolTip(
            "Write this fit to the export folder: fitted parameters with "
            "uncertainties, the binned data, the fitted curve, and a manifest "
            "(model, goodness-of-fit, file list)."
        )
        self._export_btn.clicked.connect(self._export_fit)
        btn_row.addWidget(self._export_btn)
        self._save_status = QLabel("")
        self._save_status.setStyleSheet(style.qss_text(style.TEXT_GOOD, size_px=10))
        btn_row.addWidget(self._save_status)
        btn_row.addStretch()
        rl.addLayout(btn_row)
        lay.addWidget(rg)

        # ── Model Comparison ──────────────────────────────────────────────────
        cg = QGroupBox("Model Comparison")
        cl = QVBoxLayout(cg)

        guide = QLabel(
            "Lower AICc / BIC = better model.  "
            "ΔAIC > 2 meaningful, > 6 strong, > 10 decisive.  "
            "★ = current session best.  ✔ = previously saved."
        )
        guide.setWordWrap(True)
        guide.setStyleSheet(style.qss_text(size_px=10))
        cl.addWidget(guide)

        self._cmp_tbl = QTableWidget(0, 7)
        self._cmp_tbl.setHorizontalHeaderLabels(
            ["Model", "k", "n", "R²", "AICc", "ΔAICc", "ΔBIC"]
        )
        self._cmp_tbl.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self._cmp_tbl.horizontalHeader().setStretchLastSection(True)
        self._cmp_tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._cmp_tbl.setAlternatingRowColors(True)
        self._cmp_tbl.setMinimumHeight(100)
        cl.addWidget(self._cmp_tbl)

        clr_btn = QPushButton("Clear session history")
        clr_btn.setToolTip(
            "Remove unsaved session fits from the comparison table.\n"
            "Previously saved fits are reloaded from the DB."
        )
        clr_btn.clicked.connect(self._clear_session_history)
        cl.addWidget(clr_btn)
        lay.addWidget(cg, 1)

        # Pre-populate comparison table from DB
        self._load_saved_fits()

    # ── Model builder ─────────────────────────────────────────────────────────

    def _add_peak(self):
        name  = self._model_combo.currentText()
        color = PEAK_COLORS[len(self._rows) % len(PEAK_COLORS)]
        row   = PeakRow(len(self._rows), name, color)
        row.remove_clicked.connect(self._remove_peak)
        row.pick_requested.connect(self._arm_pick)
        self._peaks_lay.insertWidget(self._peaks_lay.count() - 1, row)
        self._rows.append(row)

    def _arm_pick(self, row: PeakRow):
        """Point the next histogram click at THIS component's start.

        Only one component can be armed at a time — otherwise a click would
        have to guess which of two waiting rows meant it.
        """
        for other in self._rows:
            if other is not row:
                other.set_picking(False)
        self._pick_row = row if row.picking else None
        self._plot.arm_pick(row.picking)

    def _on_position_picked(self, x: float):
        row, self._pick_row = self._pick_row, None
        if row is None or row not in self._rows:
            return
        row.set_start(x)
        row.set_picking(False)

    def _remove_peak(self, row: PeakRow):
        if getattr(self, "_pick_row", None) is row:
            self._pick_row = None
            self._plot.arm_pick(False)
        self._rows.remove(row)
        self._peaks_lay.removeWidget(row)
        row.deleteLater()
        for i, r in enumerate(self._rows):
            r.set_index(i)
        self._plot.clear_fit()
        self._stats_box.clear()
        self._params_tbl.setRowCount(0)
        self._last_fit = {}
        self._save_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self._save_status.setText("")

    # ── Fitting ───────────────────────────────────────────────────────────────

    def _run_bootstrap(self, components, data, popt, bounds, x_fine):
        """Bootstrap the reported interval, with a cancellable progress dialog.

        Measured 4–5 s for 400 resamples on real cohorts of 350–9,000 curves,
        so this runs on the GUI thread behind a progress dialog rather than a
        worker: a thread would have to marshal the result back for a wait the
        user can already watch and abort.  Cancelling yields no interval (see
        bootstrap_fit_ci) — never a quiet fall back to the covariance.
        """
        prog = CancelableProgress(
            self, f"Bootstrapping the confidence interval — {CI_N_DRAWS} refits…",
            CI_N_DRAWS,
        )
        try:
            boot = bootstrap_fit_ci(
                components, data, self._plot.edges, self._plot.normalized,
                popt, bounds, x_fine,
                progress=lambda i, n: prog.tick(i, n),
            )
        finally:
            prog.close()

        if boot is None:
            self._save_status.setText(
                "No confidence interval — too few resamples converged."
                if not prog.cancelled else "Confidence interval cancelled.")
        elif boot.n_failed:
            self._save_status.setText(
                f"{boot.n_failed} of {boot.n_draws} resamples did not converge; "
                f"interval from the {boot.n_ok} that did.")
        return boot

    def _fit(self):
        if not self._rows:
            QMessageBox.warning(self, "No model", "Add at least one peak component.")
            return
        if self._plot.bin_centers is None:
            return

        data        = self._plot.fit_data
        if data.size < 5:
            QMessageBox.warning(
                self, "Too few values",
                "At least five finite values inside the histogram range are required.",
            )
            return
        bin_centers = self._plot.bin_centers
        density     = self._plot.density
        bw          = self._plot.bin_width

        bins_geom  = self._plot.bins
        components = [MODELS[r.model_name] for r in self._rows]
        colors     = [r.color             for r in self._rows]
        labels     = [f"Peak {i + 1} ({r.model_name})"
                      for i, r in enumerate(self._rows)]

        from collections import Counter
        counts      = Counter(r.model_name for r in self._rows)
        model_label = " + ".join(
            f"{v}× {k}" if v > 1 else k for k, v in sorted(counts.items())
        )

        fn     = make_composite(components)
        bounds = composite_bounds(components, data)
        # Whatever the user typed or picked replaces the finder's
        # position for that component, and nothing else about the fit changes.
        starts = [r.start for r in self._rows]
        p0     = composite_guess(components, data, bin_centers, density, starts)

        try:
            popt, pcov = optimize.curve_fit(
                fn, bin_centers, density,
                p0=p0, bounds=bounds,
                maxfev=20000, method="trf",
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Fit failed",
                f"{e}\n\nTry adjusting the number of bins or a different model."
            )
            return

        # Placeholders only: sort_components_by_centre needs four arrays of the
        # right shape to permute alongside popt.  The reported interval comes
        # from the bootstrap below. The covariance remains a property of the
        # fit for the manifest, not the reported confidence interval.
        perr  = np.sqrt(np.diag(pcov))
        ci_lo = popt - 1.96 * perr
        ci_hi = popt + 1.96 * perr

        # Same permutation the sort applies, so a start stays attached to the
        # component it was set for rather than to a slot number.
        starts = [starts[i] for i in centre_permutation(components, popt)]
        components, popt, perr, ci_lo, ci_hi, colors, labels = \
            sort_components_by_centre(components, popt, perr, ci_lo, ci_hi, colors, labels)

        # Rebuild BOTH from the sorted component list.  `fn` and `bounds` above
        # were built before the sort. With mixed component types, their
        # parameter blocks no longer align with sorted popt unless rebuilt.
        fn     = make_composite(components)
        bounds = composite_bounds(components, data)

        x0     = bin_centers[0]  - bw
        x1     = bin_centers[-1] + bw
        x_fine = np.linspace(x0, x1, 500)

        boot = self._run_bootstrap(components, data, popt, bounds, x_fine)
        if boot is not None:
            perr, ci_lo, ci_hi = boot.sd, boot.lo, boot.hi
        else:
            # No interval rather than the wrong one.  Falling back to the
            # covariance here would put a different method behind the same
            # column heading.
            nan = np.full(len(popt), np.nan)
            perr, ci_lo, ci_hi = nan, nan.copy(), nan.copy()

        y_components_fine = []
        y_components_bins = []
        idx = 0
        for comp in components:
            n = comp.n_params
            p = popt[idx:idx + n]
            y_components_fine.append(comp.pdf_fn(x_fine, *p))
            y_components_bins.append(comp.pdf_fn(bin_centers, *p))
            idx += n

        y_total_fine = np.sum(y_components_fine, axis=0)
        y_fit_bins   = np.sum(y_components_bins, axis=0)

        # `data` (the raw values), not just the bin heights: the information
        # criteria are per-sample, so they can be compared with the GMM
        # window's. `density`/`y_fit_bins` still carry the binned R2.
        gof = fit_stats(density, y_fit_bins, len(popt), data, components, popt)

        # The band comes from the SAME draws as the parameter intervals.  A
        # covariance band drawn beside bootstrap intervals would put two
        # different uncertainty claims in one figure, and the picture is what
        # gets published.
        ci_band = boot.band if boot is not None else None
        self._plot.show_fit(
            x_fine, y_total_fine, y_components_fine,
            colors, labels, y_fit_bins, ci_band,
        )
        self._update_table(components, popt, perr, ci_lo, ci_hi, colors, boot,
                           starts)
        self._update_stats(gof)
        self._record_fit(model_label, len(data), gof, saved=False)
        self._save_status.setText("")

        param_labels = [
            f"{labels[i]}/{comp.param_names[j]}"
            for i, comp in enumerate(components)
            for j in range(comp.n_params)
        ]

        self._last_fit = {
            "model_label":        model_label,
            "n_peaks":            len(components),
            "n_values":           len(data),
            "n_bins":             self._plot.n_bins,
            # What the histogram range excluded.  These values are gone from
            # the fit, not just from the picture, so the number travels
            # with the result into the manifest rather than living only on a
            # label somebody may not have looked at.
            "range_lo":           float(self._plot.edges[0]),
            "range_hi":           float(self._plot.edges[-1]),
            "range_is_auto":      bool(self._plot.range_is_auto),
            # Where each component was told to start, in the same order as
            # popt. A user-set start is therefore recorded explicitly.
            "starts":             list(starts),
            "n_excluded_by_range": int(bins_geom.n_out_of_range) if bins_geom else 0,
            "param_labels":       param_labels,
            "popt":               popt,
            "perr":               perr,
            "ci_lo":              ci_lo,
            "ci_hi":              ci_hi,
            "gof":                gof,
            "bin_centers":        bin_centers,
            "density":            density,
            "y_fit_bins":         y_fit_bins,
            "x_fine":             x_fine,
            "y_total_fine":       y_total_fine,
            "y_components_bins":  y_components_bins,
            "labels":             labels,
            # Kept so the export can carry the band that was DRAWN, and the
            # covariance it came from. `perr`/`ci_lo`/`ci_hi` above are the
            # covariance's DIAGONAL and cannot regenerate this band —
            # total_fit_ci samples the full matrix precisely because mixture
            # components are correlated and the diagonal overstates the
            # spread. Exporting only the diagonal would let someone redraw
            # this figure with a visibly wrong (too wide) band.
            # The covariance is still recorded — it is a real property of the
            # fit. It is not the reported confidence interval; the manifest's
            # ci_method identifies the interval stored in the CI columns.
            "pcov":               pcov,
            "ci_band":            ci_band,
            "boot":               boot,
        }
        self._save_btn.setEnabled(True)
        self._export_btn.setEnabled(True)

    # ── Results display ───────────────────────────────────────────────────────

    # A start that moves by less than this fraction of its own confidence
    # interval has not really been tested by the data: the objective is flat
    # enough here that the fit had no reason to go anywhere. It is reported
    # but never used to gate or reject the fit.
    _FLAT_MOVE_FRACTION = 0.01

    def _start_note(self, started, converged, lo, hi) -> str:
        """'started 140 → 186.8' plus, where it applies, that it barely moved."""
        if started is None or not np.isfinite(converged):
            return ""
        moved = abs(converged - float(started))
        note = f"   (started {float(started):.4g})"
        width = (hi - lo) if np.isfinite(lo) and np.isfinite(hi) else np.nan
        if np.isfinite(width) and width > 0 and moved < self._FLAT_MOVE_FRACTION * width:
            note += " — barely moved; the fit had little to say here"
        return note

    def _update_table(self, components, popt, perr, ci_lo, ci_hi, colors,
                      boot=None, starts=None):
        # Locate each component's amplitude (always param index 0) and sum them
        # so we can express each as a fraction of the total.
        amp_idx = []
        idx = 0
        for comp in components:
            amp_idx.append(idx)
            idx += comp.n_params
        total_amp = max(sum(popt[i] for i in amp_idx), 1e-12)

        def _ci_text(lo, hi):
            return "—" if not (np.isfinite(lo) and np.isfinite(hi)) \
                   else f"[{lo:.4g},  {hi:.4g}]"

        # Build rows as (name, val_str, err_str, ci_str, color)
        rows = []
        idx  = 0
        for i, (comp, color) in enumerate(zip(components, colors)):
            for j, pname in enumerate(comp.param_names):
                val = popt[idx + j]
                err = perr[idx + j]
                lo  = ci_lo[idx + j]
                hi  = ci_hi[idx + j]
                label = f"Peak {i+1} ({comp.name})  "

                if j == 0:   # amplitude → fraction
                    frac = val / total_amp
                    # The fraction's interval comes from the draws themselves —
                    # each resample's own fraction — not from dividing this
                    # parameter's interval by the total.  That division is how a
                    # proportion came to be reported as [−0.40, +1.22]: scaling
                    # an interval that starts below zero leaves it below zero.
                    if boot is not None:
                        f_lo, f_hi = boot.frac_lo[i], boot.frac_hi[i]
                        # width/3.92 is the 95 % interval read back as a 1σ
                        # spread, so the stderr column stays comparable with
                        # the other rows'.
                        f_err = ((f_hi - f_lo) / 3.92
                                 if np.isfinite(f_hi - f_lo) else float("nan"))
                    else:
                        f_lo = f_hi = f_err = float("nan")
                    rows.append((
                        label + "fraction",
                        f"{frac:.3f}  ({frac * 100:.1f} %)",
                        "—" if not np.isfinite(f_err) else f"{f_err:.3g}",
                        _ci_text(f_lo, f_hi),
                        color,
                    ))

                elif comp.name == "LogNormal" and pname == "μ_log":
                    started = (starts[i] if starts and j == 1 else None)
                    # exp(μ_log) = median of the distribution in linear units.
                    # SE by delta method: SE(exp(x)) = exp(x) · SE(x)
                    med     = np.exp(val)
                    med_err = med * err
                    rows.append((
                        label + "μ_log  (ln-space)",
                        f"{val:.4g}  →  median = {med:.4g}"
                        + self._start_note(started, np.exp(val), np.exp(lo)
                                           if np.isfinite(lo) else np.nan,
                                           np.exp(hi) if np.isfinite(hi) else np.nan),
                        "—" if not np.isfinite(err) else f"{err:.3g}  (±{med_err:.3g})",
                        _ci_text(lo, hi) if not np.isfinite(lo + hi) else
                        f"[{lo:.4g}, {hi:.4g}]  →  [{np.exp(lo):.4g}, {np.exp(hi):.4g}]",
                        color,
                    ))

                elif comp.name == "LogNormal" and pname == "σ_log":
                    # exp(σ_log) = geometric SD factor (GSD):
                    # ~68 % of data lies within [median/GSD, median·GSD].
                    gsd = np.exp(val)
                    rows.append((
                        label + "σ_log  (ln-space)",
                        f"{val:.4g}  →  GSD = {gsd:.3g}×",
                        "—" if not np.isfinite(err) else f"{err:.3g}",
                        _ci_text(lo, hi),
                        color,
                    ))

                else:
                    # A component's POSITION is parameter 1 for every model in
                    # the registry, which is the one the user set a start for.
                    started = (starts[i] if starts and j == 1 else None)
                    rows.append((
                        label + pname,
                        f"{val:.5g}" + self._start_note(started, val, lo, hi),
                        "—" if not np.isfinite(err) else f"{err:.3g}",
                        _ci_text(lo, hi),
                        color,
                    ))
            idx += comp.n_params

        self._params_tbl.setRowCount(len(rows))
        for r, (name, val_str, err_str, ci_str, color) in enumerate(rows):
            for c, text in enumerate([name, val_str, err_str, ci_str]):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if c == 0:
                    item.setForeground(QColor(color))
                self._params_tbl.setItem(r, c, item)

    def _update_stats(self, gof: dict):
        # Two blocks because these describe different things: R2/chi-2/RSS are
        # about the fit to the HISTOGRAM (what least squares minimised), the
        # information criteria are about the fit to the VALUES.
        integers = {"DOF", "n (for IC)", "k (for IC)", "n_zero_density"}
        binned   = ("R²", "Reduced χ²", "RSS", "DOF")
        per_samp = ("log-likelihood", "AIC", "AICc", "BIC", "n (for IC)", "k (for IC)")

        def fmt(key):
            val = gof.get(key)
            if val is None:
                return None
            return (f"  {key:<16} {int(val)}" if key in integers
                    else f"  {key:<16} {val:.5g}")

        lines = ["Goodness-of-fit — to the histogram", "─" * 40]
        lines += [t for t in (fmt(k) for k in binned) if t]
        lines += ["", "Information criteria — over the values", "─" * 40]
        lines += [t for t in (fmt(k) for k in per_samp) if t]

        if gof.get("n_zero_density"):
            lines += ["",
                      f"  ⚠ {int(gof['n_zero_density'])} value(s) fall where this "
                      f"model has no density.",
                      "    The likelihood is floored there, so the criteria "
                      "below understate how",
                      "    badly the model misses them."]
        lines += [
            "",
            "Guide: lower AICc / BIC = better model",
            "  |ΔAIC| > 2 meaningful · > 6 strong · > 10 decisive",
            "  Use both AICc and BIC — agreement = robust conclusion",
            "",
            "Computed from the per-sample log-likelihood, so these are on the",
            "same scale as the 2-D GMM window's. Legacy fits used a",
            "bin-count basis and do not compare — the",
            "comparison table marks them.",
            "Note the PARAMETERS are still estimated by least squares on the",
            "bins; only the reported statistic is per-sample.",
        ]
        self._stats_box.setPlainText("\n".join(lines))

    # ── Comparison table ──────────────────────────────────────────────────────

    def _record_fit(self, model_label: str, n_values: int,
                    gof: dict, saved: bool = False):
        self._fit_history.append({
            "model":   model_label,
            "k":       gof["k (for IC)"],
            "n":       n_values,
            "R²":      gof["R²"],
            "AICc":    gof["AICc"],
            "BIC":     gof["BIC"],
            # Which method made this AICc — see _update_comparison_table.
            "ic_basis": gof.get("ic_basis", _dfc.IC_BASIS_LEGACY),
            "saved":   saved,
        })
        self._update_comparison_table()

    def _update_comparison_table(self):
        h = self._fit_history
        if not h:
            self._cmp_tbl.setRowCount(0)
            return
        # Only rows computed on the current information-criterion basis may be
        # ranked against each other. Legacy fits use a Gaussian-SSE surrogate
        # over bin heights; the current basis uses per-sample likelihood, so
        # the two live on entirely different scales.  Ranking them together
        # would put the green "best" tint on whichever scale happens to run
        # smaller — a wrong conclusion, drawn silently, in the one table whose
        # whole job is choosing a model.  Old rows are still SHOWN: they are a
        # record of a fit that really happened, and deleting them would be
        # discarding the user's own work.  They just do not compete, and say so.
        current = [x for x in h if x.get("ic_basis") == _dfc.IC_BASIS]
        best_aicc = min((x["AICc"] for x in current), default=float("nan"))
        best_bic  = min((x["BIC"]  for x in current), default=float("nan"))
        self._cmp_tbl.setRowCount(len(h))
        for r, entry in enumerate(h):
            comparable = entry.get("ic_basis") == _dfc.IC_BASIS
            d_aicc  = entry["AICc"] - best_aicc if comparable else float("nan")
            d_bic   = entry["BIC"]  - best_bic  if comparable else float("nan")
            is_best = comparable and d_aicc < 1e-9
            is_last = r == len(h) - 1 and not entry["saved"]
            saved   = entry["saved"]
            label   = ("✔ " if saved else "") + entry["model"]
            if comparable:
                aicc_txt = f"{entry['AICc']:.1f}"
                d_aicc_txt = "★ best" if is_best      else f"+{d_aicc:.1f}"
                d_bic_txt  = "★ best" if d_bic < 1e-9 else f"+{d_bic:.1f}"
            else:
                aicc_txt = f"{entry['AICc']:.1f} (old basis)"
                d_aicc_txt = d_bic_txt = "not comparable"
            vals = [
                label,
                str(entry["k"]),
                str(entry["n"]),
                f"{entry['R²']:.3f}",
                aicc_txt,
                d_aicc_txt,
                d_bic_txt,
            ]
            for c, text in enumerate(vals):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if saved:
                    item.setBackground(QColor(style.TABLE_TINT_SAVED))   # pale green — saved
                elif is_best:
                    item.setBackground(QColor(style.TABLE_TINT_BEST))   # green — best AICc
                elif is_last:
                    item.setBackground(QColor(style.TABLE_TINT_RECENT))   # yellow — most recent
                self._cmp_tbl.setItem(r, c, item)

    def _clear_session_history(self):
        """Keep only previously-saved fits; remove unsaved session entries."""
        self._fit_history = [e for e in self._fit_history if e["saved"]]
        self._update_comparison_table()

    def _load_saved_fits(self):
        """Pre-populate comparison table with fits saved to DB for this variable."""
        try:
            rows = _db.get_distribution_fits(self._variable, self._db_path)
            for row in rows:
                gof = json.loads(row["gof_json"])
                self._fit_history.append({
                    "model": row["model_label"],
                    "k":     gof.get("k (for IC)", row["n_peaks"]),
                    "n":     row["n_values"],
                    "R²":    gof.get("R²", float("nan")),
                    "AICc":  gof.get("AICc", float("nan")),
                    "BIC":   gof.get("BIC",  float("nan")),
                    # An unstamped row cannot acquire a basis retroactively;
                    # absence identifies the legacy basis.
                    "ic_basis": gof.get("ic_basis", _dfc.IC_BASIS_LEGACY),
                    "saved": True,
                })
            if self._fit_history:
                self._update_comparison_table()
        except Exception:
            pass

    # ── DB save ───────────────────────────────────────────────────────────────

    def export_provenance(self) -> dict:
        """This fit's settings, for an export manifest — same protocol method
        as the other exporting windows."""
        f = self._last_fit
        return {
            "window":      "dist_fit",
            "variable":    self._variable,
            "units":       self._units,
            "model_label": f.get("model_label"),
            "n_peaks":     f.get("n_peaks"),
            "n_values":    f.get("n_values"),
            "n_bins":      f.get("n_bins"),
            "range_lo":    f.get("range_lo"),
            "range_hi":    f.get("range_hi"),
            "range_is_auto":       f.get("range_is_auto"),
            "n_excluded_by_range": f.get("n_excluded_by_range"),
            "user_peak_starts":    [None if v is None else float(v)
                                    for v in (f.get("starts") or [])],
            "caption":     self._caption,
        }

    def _export_fit(self):
        """Write the current fit out as data files.

        Deliberately NOT built on a shared "fit export" helper with
        gmm_fit_window: the two windows' fits are different objects (1-D peak
        components with per-parameter CIs vs. a 2-D mixture with means,
        covariances and weights), and forcing one schema over both would
        produce a file that describes neither well. What IS shared — the
        export folder, the naming, the CSV writer, the manifest — is shared,
        via export_utils.ExportGroup. The plotting/model UI remains specific
        to each fit type."""
        if not self._last_fit:
            return
        f = self._last_fit
        parts = ["_params.csv", "_histogram.csv", "_curve.csv"]
        stem  = f"fit_{_slug(self._variable)}"
        with _export.export_group(
            self._db_path, stem, parts, kind="distribution_fit",
        ) as g:
            g.contributing_files([
                path for path, keep in zip(self._paths, self._plot.fit_mask) if keep
            ])
            g.note_dict(self.export_provenance())
            g.note(goodness_of_fit=f["gof"],
                   components=list(f["labels"]))
            # The interval's own provenance names the method that produced
            # it and carries the covariance needed to reconstruct the fit.
            boot = f.get("boot")
            g.note_dict(boot.manifest_fields(f.get("pcov")) if boot is not None
                        else ci_manifest_fields(f.get("pcov"), False,
                                                method="none — no interval computed"))

            g.table(
                "_params.csv",
                ["parameter", "value", "stderr", "ci_lo", "ci_hi"],
                [(name, float(v), float(e), float(lo), float(hi))
                 for name, v, e, lo, hi in zip(
                     f["param_labels"], f["popt"], f["perr"],
                     f["ci_lo"], f["ci_hi"])],
            )

            # The binned data as plotted, the total fit at those bins, and
            # each component — everything needed to redraw the figure.
            comp_labels = list(f["labels"])
            header = (["bin_center", "density", "fit_total"]
                      + [f"fit_{_slug(l)}" for l in comp_labels])
            rows = []
            for i in range(len(f["bin_centers"])):
                row = [float(f["bin_centers"][i]), float(f["density"][i]),
                       float(f["y_fit_bins"][i])]
                row += [float(c[i]) for c in f["y_components_bins"]]
                rows.append(row)
            g.table("_histogram.csv", header, rows)

            # The smooth fitted curve, sampled finely — a fit line drawn from
            # the binned values alone would be visibly coarser than the one
            # on screen — plus the confidence band drawn around it, so the
            # exported figure can show the same uncertainty the screen did.
            band = f.get("ci_band")
            header = ["x", "y_total"] + (["ci_lo", "ci_hi"] if band else [])
            rows = []
            for i in range(len(f["x_fine"])):
                row = [float(f["x_fine"][i]), float(f["y_total_fine"][i])]
                if band:
                    row += [float(band[0][i]), float(band[1][i])]
                rows.append(row)
            g.table("_curve.csv", header, rows)

        QMessageBox.information(self, "Export fit", g.message())

    def _save_fit(self):
        if not self._last_fit:
            return
        f = self._last_fit
        params_json = json.dumps([
            {"name": name, "value": float(v),
             "stderr": float(e), "ci_lo": float(lo), "ci_hi": float(hi)}
            for name, v, e, lo, hi in zip(
                f["param_labels"], f["popt"], f["perr"], f["ci_lo"], f["ci_hi"]
            )
        ])
        gof_json = json.dumps(f["gof"])
        fit_config_json = json.dumps({
            "n_bins": f["n_bins"],
            "normalized": self._plot.normalized,
            "range_lo": f["range_lo"],
            "range_hi": f["range_hi"],
            "range_is_auto": f["range_is_auto"],
            "n_excluded_by_range": f["n_excluded_by_range"],
            "user_peak_starts": f["starts"],
        })

        try:
            _db.save_distribution_fit(
                variable        = self._variable,
                units           = self._units,
                n_values        = f["n_values"],
                n_peaks         = f["n_peaks"],
                model_label     = f["model_label"],
                params_json     = params_json,
                gof_json        = gof_json,
                fit_config_json = fit_config_json,
                db_path         = self._db_path,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return

        # Mark the most recently added unsaved entry as saved
        for entry in reversed(self._fit_history):
            if not entry["saved"]:
                entry["saved"] = True
                break
        self._update_comparison_table()
        self._save_btn.setEnabled(False)
        self._save_status.setText("Saved ✔")


# ── Main window ───────────────────────────────────────────────────────────────

class DistFitWindow(QMainWindow):
    """
    Pop-out histogram fitting sandbox for one Stats variable.

    Locked to the pass values supplied at construction.
    Double-clicking "Fit…" again for the same variable raises this window.
    """

    def __init__(
        self,
        variable_name: str,
        units:         str,
        pass_values:   np.ndarray,
        db_path:       str,
        caption:       str = "",
        paths:         list[str] | None = None,
    ) -> None:
        super().__init__()
        self._variable = variable_name
        # The curves these values came from, positionally aligned with
        # pass_values. Carried purely so an export can say WHICH curves it
        # fitted — without it a fit result is a set of numbers nobody can
        # trace back to data. Callers that genuinely have no file list (rare)
        # may omit it; the manifest then records an empty list rather than
        # silently implying the export covers everything.
        self._paths = list(paths) if paths else []

        n     = len(pass_values)
        title = f"SMFS — distribution fit — {variable_name}"
        if units:
            title += f"  [{units}]"
        self.setWindowTitle(title)
        self.setWindowFlag(Qt.WindowType.Window)
        fit_on_screen(self, 1200, 720)
        style.apply_plot_defaults()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(4)
        root.setContentsMargins(8, 6, 8, 6)

        # Header
        hdr = QLabel(
            f"{variable_name}   |   {n} pass values   |   "
            f"min {pass_values.min():.4g}   "
            f"mean {pass_values.mean():.4g}   "
            f"median {float(np.median(pass_values)):.4g}   "
            f"max {pass_values.max():.4g}"
            + (f"   [{units}]" if units else "")
        )
        hdr.setFont(style.font(hdr.font(), size_pt=style.FONT_SMALL_PT))
        root.addWidget(hdr)

        # Splitter: plot left, model right
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._plot_pane  = _PlotPane(pass_values, variable_name, units, caption=caption)
        self._model_pane = _ModelPane(
            self._plot_pane, variable_name, units, db_path,
            paths=self._paths, caption=caption,
        )

        splitter.addWidget(self._plot_pane)
        splitter.addWidget(self._model_pane)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([600, 600])
        root.addWidget(splitter, stretch=1)
