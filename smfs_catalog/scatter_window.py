# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/scatter_window.py
#
# ScatterWindow — any per-file variable against any other.
#
# Two dropdowns, a scatter of every curve with BOTH values, Spearman rho with
# n on the canvas, and an ordinary-least-squares line with its 95 % confidence
# band.  Everything exports, with both axis choices in the manifest.
#
# ACQUISITION TIME IS ONE OF THE VARIABLES, deliberately.  Put it on X and the
# fitted slope is the drift rate. Both this window and the variable drift view
# use regression.py.
#
# THE FISHING PROBLEM IS REAL AND IS STATED ON SCREEN.  With ~30 variables
# there are ~400 pairs, so several will clear p < 0.05 by chance alone.  The
# window says so next to the number, and the manifest records how many
# variables were on offer, which is the figure a reader needs to judge a
# p-value found by scanning. Nothing is gated in this exploratory view.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from . import db as _db
from . import export_utils as _export
from . import quantities as _quant
from . import regression as _reg
from . import style
from . import variables as _vars
from . import clustering as _clustering
from .widgets import ClusterColourBar
from .export_utils import slug as _slug
from .qt_utils import _DateAxis, _make_session_header, set_plot_title, set_si_label, fit_on_screen

_PEN_NONE = pg.mkPen(None)
_BRUSH_PT = pg.mkBrush(*style.rgba(style.INK_STRONG, style.DOT_ALPHA))
_PEN_SEL  = pg.mkPen(style.INK, width=2)
_FIT_HUE  = style.series_line(0)
_PEN_FIT  = style.model_pen(_FIT_HUE)
_BRUSH_FIT = style.band_brush(_FIT_HUE)

# Start with a commonly available force/length pair rather than a blank plot.
_DEFAULT_X = "seg_l_c_nm"
_DEFAULT_Y = "seg_force_pN"


class ScatterWindow(QMainWindow):
    """Any per-file variable against any other, over a fixed cohort."""

    view_file_requested = pyqtSignal(str)

    def __init__(
        self,
        paths:        list[str],
        db_path:      str,
        caption:      str = "",
        session_info: dict | None = None,
    ) -> None:
        super().__init__()
        self.setWindowFlag(Qt.WindowType.Window)
        self._db_path = db_path
        self._paths   = [_db.normalize_path(p) for p in paths]
        self._caption = caption

        self._owner = (_db.resolve_common_experimentalist(self._paths, db_path)
                       or "mixed/unknown owners")
        self.setWindowTitle("SMFS — variable comparison")
        fit_on_screen(self, 1150, 700)
        self._vars   = _vars.available(self._paths, db_path)
        self._by_key = {v.key: v for v in self._vars}

        # Plotted state — one entry per point, i.e. per curve having BOTH
        # values.  Shared index space for the scatter, the side list and the
        # selection ring, exactly as variable_window does it.
        self._plot_paths: list[str] = []
        self._x = np.empty(0)
        self._y = np.empty(0)
        self._selected: int | None = None
        self._fit:  _reg.LinearFit  | None = None
        self._corr: _reg.Correlation | None = None
        self._n_missing_x = self._n_missing_y = self._n_missing_both = 0

        style.apply_plot_defaults()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(4)
        root.setContentsMargins(8, 6, 8, 6)

        hdr = _make_session_header(session_info)
        if hdr is not None:
            root.addWidget(hdr)

        root.addLayout(self._build_axis_row())

        self._info = QLabel("")
        info_font = style.font(
            self._info.font(), size_pt=style.FONT_SMALL_PT)
        self._info.setFont(info_font)
        self._info.setStyleSheet(style.qss_text())
        root.addWidget(self._info)

        self._sel_label = QLabel("")
        self._sel_label.setFont(info_font)
        root.addWidget(self._sel_label)

        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split, stretch=1)

        self._plot = pg.PlotWidget()
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._scatter = pg.ScatterPlotItem(size=style.DOT_SIZE, pen=_PEN_NONE,
                                           brush=_BRUSH_PT)
        self._scatter.sigClicked.connect(self._on_scatter_clicked)
        self._plot.addItem(self._scatter)
        self._sel_marker = pg.ScatterPlotItem(size=16, symbol="o", pen=_PEN_SEL,
                                              brush=pg.mkBrush(None))
        self._sel_marker.hide()
        self._plot.addItem(self._sel_marker, ignoreBounds=True)
        # Fit + band, both ignoreBounds: a model drawn over the data must never
        # be what sets the view range.
        self._band_lo = pg.PlotCurveItem(x=[0.0, 1.0], y=[0.0, 0.0], pen=_PEN_NONE)
        self._band_hi = pg.PlotCurveItem(x=[0.0, 1.0], y=[0.0, 0.0], pen=_PEN_NONE)
        self._band    = pg.FillBetweenItem(self._band_lo, self._band_hi,
                                           brush=_BRUSH_FIT)
        self._line    = pg.PlotCurveItem(pen=_PEN_FIT)
        for it in (self._band_lo, self._band_hi, self._band, self._line):
            self._plot.addItem(it, ignoreBounds=True)
        split.addWidget(self._plot)

        self._list = QListWidget()
        self._list.setStyleSheet(style.LIST_QSS)
        self._list.currentRowChanged.connect(self._on_list_row_changed)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        split.addWidget(self._list)
        split.setSizes([900, 220])

        # The clustering ran on the 2DH ensemble and is projected back onto
        # these two scalars. It is session-only, and the bar says when absent.
        self._cluster_bar = ClusterColourBar()
        self._cluster_bar.changed.connect(self._render)
        root.addWidget(self._cluster_bar)

        root.addLayout(self._build_action_row())
        self._reload()

    # ── Controls ─────────────────────────────────────────────────────────────

    def _fill_combo(self, combo: QComboBox, default_key: str) -> None:
        for i, v in enumerate(self._vars):
            combo.addItem(v.label, v.key)
            # Per-ITEM hover, so the description is readable while choosing
            # rather than only after committing to an axis.  Same register the
            # queue header reads (variables.DESCRIPTIONS).
            if v.description:
                combo.setItemData(i, v.description, Qt.ItemDataRole.ToolTipRole)
        i = combo.findData(default_key)
        combo.setCurrentIndex(i if i >= 0 else 0)
        self._sync_combo_tooltip(combo)
        combo.currentIndexChanged.connect(
            lambda _i, c=combo: self._sync_combo_tooltip(c))

    @staticmethod
    def _sync_combo_tooltip(combo: QComboBox) -> None:
        """
        Qt does not carry an item's tooltip onto the closed combo box, so
        without this the description is reachable only while the popup is
        open — i.e. never, once an axis has been chosen.
        """
        combo.setToolTip(
            combo.itemData(combo.currentIndex(), Qt.ItemDataRole.ToolTipRole) or "")

    def _build_axis_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel("X:"))
        self._x_combo = QComboBox()
        self._fill_combo(self._x_combo, _DEFAULT_X)
        self._x_combo.currentIndexChanged.connect(self._reload)
        row.addWidget(self._x_combo)

        row.addSpacing(12)
        row.addWidget(QLabel("Y:"))
        self._y_combo = QComboBox()
        self._fill_combo(self._y_combo, _DEFAULT_Y)
        self._y_combo.currentIndexChanged.connect(self._reload)
        row.addWidget(self._y_combo)

        row.addSpacing(16)
        self._chk_fit = QCheckBox("Linear fit")
        self._chk_fit.setChecked(True)
        self._chk_fit.setToolTip(
            "OLS line with a 95% confidence band for the mean trend, not for "
            "individual curves. The interval assumes independent residuals "
            "with constant variance; consecutive measurements can make it "
            "too narrow."
        )
        self._chk_fit.toggled.connect(self._render)
        row.addWidget(self._chk_fit)

        row.addStretch()
        return row

    def _build_action_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._warn = QLabel("")
        self._warn.setStyleSheet(style.qss_text(style.TEXT_WARNING))
        self._warn.setFont(style.font(
            self._warn.font(), size_pt=style.FONT_SMALL_PT))
        self._warn.setWordWrap(True)
        row.addWidget(self._warn, 1)
        self._swap_btn = QPushButton("Swap axes")
        self._swap_btn.clicked.connect(self._on_swap)
        row.addWidget(self._swap_btn)
        self._export_btn = QPushButton("Export…")
        self._export_btn.setToolTip(
            "Write the plotted pairs, the fitted line and its band to the "
            "export folder, with a manifest recording both axes, the "
            "correlation and the file list."
        )
        self._export_btn.clicked.connect(self._on_export)
        row.addWidget(self._export_btn)
        return row

    def _on_swap(self) -> None:
        xi, yi = self._x_combo.currentIndex(), self._y_combo.currentIndex()
        for combo, i in ((self._x_combo, yi), (self._y_combo, xi)):
            combo.blockSignals(True)
            combo.setCurrentIndex(i)
            combo.blockSignals(False)
        self._reload()

    # ── Data ─────────────────────────────────────────────────────────────────

    @property
    def _x_key(self) -> str:
        return self._x_combo.currentData()

    @property
    def _y_key(self) -> str:
        return self._y_combo.currentData()

    def _reload(self) -> None:
        """Refetch both axes for the cohort and redraw."""
        xk, yk = self._x_key, self._y_key
        keys = [xk] if xk == yk else [xk, yk]
        order, cols = _vars.columns(self._paths, keys, self._db_path)
        xs, ys = cols[xk], cols[yk]

        # PAIRWISE completeness, and the three shortfalls counted separately:
        # "no x", "no y" and "neither" have different remedies (retune the
        # detector, re-analyse, or this curve simply predates the variable),
        # and one merged number hides which one you are looking at — the
        # distinction between gate participation and plot completeness.
        fin_x, fin_y = np.isfinite(xs), np.isfinite(ys)
        keep = fin_x & fin_y
        self._n_missing_x    = int(np.count_nonzero(~fin_x & fin_y))
        self._n_missing_y    = int(np.count_nonzero(fin_x & ~fin_y))
        self._n_missing_both = int(np.count_nonzero(~fin_x & ~fin_y))

        idx = np.where(keep)[0]
        self._plot_paths = [order[i] for i in idx]
        self._x, self._y = xs[keep], ys[keep]
        self._selected = None

        self._apply_axis_labels()
        self._populate_list()
        self._render()

    def _apply_axis_labels(self) -> None:
        xk, yk = self._x_key, self._y_key
        xv, yv = self._by_key[xk], self._by_key[yk]
        if xv.is_time:
            self._plot.setAxisItems({"bottom": _DateAxis()})
            self._plot.setLabel("bottom", "Acquisition time")
        else:
            self._plot.setAxisItems({"bottom": pg.AxisItem("bottom")})
            # si=False on both: the readouts beside the plot print these
            # numbers in plain units, and an axis free to relabel itself in
            # micrometres beside a label reading nanometres shows one value at
            # two scales.
            set_si_label(self._plot, "bottom", style.mathify(xv.label),
                         key=xk, si=False)
        set_si_label(self._plot, "left", style.mathify(yv.label), key=yk, si=False)

    # ── Render ───────────────────────────────────────────────────────────────

    def _render(self) -> None:
        xk, yk = self._x_key, self._y_key
        same = xk == yk

        self._cluster_bar.refresh(self._plot_paths)
        if self._cluster_bar.is_active():
            # Per-point brushes: SERIES_LABELED, because a cluster colour
            # always ships a legend.  An unlabelled curve keeps the neutral
            # tone rather than borrowing a cluster's hue — it is not a cluster
            # of its own, and the coverage line says how many there are.
            cl = _clustering.current()
            spots = []
            for i, path in enumerate(self._plot_paths):
                lbl = cl.label_for(path) if cl else None
                brush = (style.scatter_brush(style.series_labeled(lbl))
                         if lbl is not None else _BRUSH_PT)
                spots.append({"pos": (float(self._x[i]), float(self._y[i])),
                              "data": i, "brush": brush, "pen": _PEN_NONE})
            self._scatter.setData(spots)
        else:
            self._scatter.setData(self._x, self._y,
                                  data=list(range(len(self._plot_paths))))
        self._update_sel_marker()

        # Recomputed even when the fit is hidden, so the readout and the export
        # stay available without the band on screen. A variable against itself
        # is still a useful temporary axis choice, but its perfect line/rho are
        # identities, not analyses, and must not leak into the export.
        self._fit = None if same else _reg.linear_fit(self._x, self._y)
        self._corr = (None if same else
                      _reg.correlate(self._x, self._y, method="spearman"))

        show = self._chk_fit.isChecked() and self._fit is not None and not same
        for it in (self._line, self._band):
            it.setVisible(show)
        if show:
            xs = np.linspace(float(self._x.min()), float(self._x.max()), 200)
            lo, hi = self._fit.band(xs)
            self._line.setData(xs, self._fit.predict(xs))
            self._band_lo.setData(xs, lo)
            self._band_hi.setData(xs, hi)

        self._update_info(same)
        set_plot_title(self._plot,
                       f"{self._by_key[yk].label} vs {self._by_key[xk].label}",
                       self._provenance_caption())

    def _update_info(self, same: bool) -> None:
        n = len(self._plot_paths)
        bits = [f"Owner: {self._owner}",
                f"{n} of {len(self._paths)} curves have both values"]
        for count, what in ((self._n_missing_x, "no X"),
                            (self._n_missing_y, "no Y"),
                            (self._n_missing_both, "neither")):
            if count:
                bits.append(f"{count} {what}")
        if same:
            bits.append("same variable on both axes")
        elif self._corr is not None:
            c = self._corr
            bits.append(f"Spearman ρ {c.rho:+.3f}  (n={c.n}, p={c.p:.3g})")
        if self._fit is not None and not same:
            f = self._fit
            if self._by_key[self._x_key].is_time:
                s, lo, hi = _reg.per_hour(f)
                unit = self._by_key[self._y_key].unit
                bits.append(f"slope {s:.4g} [{lo:.4g}, {hi:.4g}] "
                            f"{unit + '/h' if unit else '/h'}")
            else:
                lo, hi = f.slope_ci
                bits.append(f"slope {f.slope:.4g} [{lo:.4g}, {hi:.4g}]")
            bits.append(f"R² {f.r2:.3f}")
        self._info.setText("   —   ".join(bits))
        self._update_warning(same)

    def _update_warning(self, same: bool) -> None:
        """The fishing caution, sized to what is actually on offer.

        Stated where the number is read, not only in the manual: a p-value
        found by scanning pairs is not the p-value it looks like, and the
        moment somebody needs to know that is the moment they see a small
        one.
        """
        if same or self._corr is None:
            self._warn.setText("")
            return
        n_pairs = len(self._vars) * (len(self._vars) - 1) // 2
        if self._corr.p < 0.05:
            self._warn.setText(
                f"⚠ {len(self._vars)} variables here make {n_pairs} possible "
                f"pairs, so ~{max(1, round(n_pairs * 0.05))} would clear "
                f"p < 0.05 by chance alone. A correlation found by scanning "
                f"pairs needs confirming on an independent cohort before it "
                f"means anything."
            )
        else:
            self._warn.setText("")

    # ── Selection ────────────────────────────────────────────────────────────

    def _populate_list(self) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        for p in self._plot_paths:
            self._list.addItem(QListWidgetItem(Path(p).name))
        self._list.blockSignals(False)
        self._sel_marker.hide()
        self._sel_label.setText("")

    def _on_list_row_changed(self, row: int) -> None:
        if 0 <= row < len(self._plot_paths):
            self._select(row, from_list=True)

    def _on_scatter_clicked(self, _sc, points) -> None:
        if len(points) == 0:
            return
        i = points[0].data()
        if i is not None:
            self._select(int(i), from_list=False)

    def _select(self, i: int, from_list: bool) -> None:
        if not (0 <= i < len(self._plot_paths)):
            return
        self._selected = i
        self._update_sel_marker()
        xq = _quant.format_value(self._x_key, self._x[i], with_unit=True)
        yq = _quant.format_value(self._y_key, self._y[i], with_unit=True)
        self._sel_label.setText(
            f"Selected: {Path(self._plot_paths[i]).name}   —   "
            f"{self._by_key[self._x_key].label} {xq}   —   "
            f"{self._by_key[self._y_key].label} {yq}"
        )
        if not from_list:
            self._list.blockSignals(True)
            self._list.setCurrentRow(i)
            self._list.blockSignals(False)
            self._list.scrollTo(self._list.currentIndex())

    def _update_sel_marker(self) -> None:
        i = self._selected
        if i is None or not (0 <= i < len(self._plot_paths)):
            self._sel_marker.hide()
            return
        self._sel_marker.setData([self._x[i]], [self._y[i]])
        self._sel_marker.show()

    def _on_double_click(self, _item) -> None:
        if self._selected is not None and 0 <= self._selected < len(self._plot_paths):
            self.view_file_requested.emit(self._plot_paths[self._selected])

    # ── Export ───────────────────────────────────────────────────────────────

    def _provenance_caption(self) -> str:
        bits = [f"{len(self._plot_paths)} curves"]
        if self._corr is not None and self._x_key != self._y_key:
            bits.append(f"Spearman ρ {self._corr.rho:+.3f} (n={self._corr.n})")
        if self._caption:
            bits.append(self._caption)
        legend = self._cluster_bar.legend_text(self._plot_paths)
        if legend:
            bits.append(legend)
        return "  |  ".join(bits)

    def export_provenance(self) -> dict:
        """This window's settings, for an export manifest.

        Both axes, always: a scatter of two user-chosen variables is
        meaningless without them, and which pair was chosen is exactly the
        setting that silently decides what the picture means.
        """
        xv, yv = self._by_key[self._x_key], self._by_key[self._y_key]
        return {
            "window":            "scatter",
            "x_variable":        xv.key,
            "x_label":           xv.label,
            "x_unit":            xv.unit,
            "x_source":          xv.source,
            "y_variable":        yv.key,
            "y_label":           yv.label,
            "y_unit":            yv.unit,
            "y_source":          yv.source,
            "fit_shown":         bool(self._chk_fit.isChecked()),
            "n_curves_in_cohort":      len(self._paths),
            "n_curves_plotted":        len(self._plot_paths),
            "n_missing_x":             self._n_missing_x,
            "n_missing_y":             self._n_missing_y,
            "n_missing_both":          self._n_missing_both,
            # How many pairs were on offer, so a reader can judge a p-value
            # that was found by scanning rather than predicted in advance.
            "n_variables_offered":     len(self._vars),
            "n_pairs_possible":        len(self._vars) * (len(self._vars) - 1) // 2,
            **_clustering.provenance(self._plot_paths,
                                     self._cluster_bar.is_active()),
        }

    def _on_export(self) -> None:
        if not self._plot_paths:
            QMessageBox.information(self, "Export", "No pairs to export.")
            return
        xk, yk = self._x_key, self._y_key
        with _export.export_group(
            self._db_path,
            f"scatter_{_slug(xk)}_vs_{_slug(yk)}", ["_pairs.csv"],
            kind="scatter",
        ) as g:
            g.contributing_files(self._plot_paths)
            g.note_dict(self.export_provenance())
            g.note_dict(_reg.manifest_fields(
                self._fit, self._corr,
                x_is_time=self._by_key[xk].is_time))

            # The band is DRAWN, so its lo/hi ship as columns and its full
            # parameter covariance ships in the manifest (manifest_fields) —
            # the diagonal alone cannot redraw it.
            if self._fit is not None:
                f_lo, f_hi = self._fit.band(self._x)
                f_val      = self._fit.predict(self._x)
            else:
                nan = np.full(self._x.shape, np.nan)
                f_lo = f_hi = f_val = nan
            # The cluster travels with the row, always — it is the join key
            # back to pca_window's _scores.csv, and the clustering itself does
            # not survive the session, so an export is the only place it can
            # be kept.  Written whether or not the colouring is switched ON:
            # a display toggle must not decide what a data file contains.
            cl = _clustering.current()
            g.table(
                "_pairs.csv",
                ["path", xk, yk, "fit", "fit_ci_lo", "fit_ci_hi", "cluster"],
                [(p, float(a), float(b), float(v), float(lo), float(hi),
                  "" if cl is None or cl.label_for(p) is None else cl.label_for(p))
                 for p, a, b, v, lo, hi in zip(
                     self._plot_paths, self._x, self._y, f_val, f_lo, f_hi)],
            )
        QMessageBox.information(self, "Export", g.message())


    def closeEvent(self, event):
        self._cluster_bar.detach()
        super().closeEvent(event)
