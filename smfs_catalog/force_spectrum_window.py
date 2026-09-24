# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/force_spectrum_window.py
#
# Rupture force against ln(loading rate) for one population, with Bell-Evans
# and both DHS shapes fitted and compared (force_spectrum_core). The fit uses
# exactly the curves it is handed; the population is chosen upstream.

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView, QGroupBox, QHeaderView, QLabel, QMainWindow, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from . import db as _db
from . import force_spectrum_core as _fs
from . import quantities as _quant
from . import style
from . import variables as _vars
from .qt_utils import fit_on_screen, set_plot_title, set_si_label

FORCE_KEY = "seg_force_pN"
RATE_KEY  = "seg_loading_rate_pN_s"


class ForceSpectrumWindow(QMainWindow):
    def __init__(self, paths: list[str], db_path: str, caption: str = "",
                 parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("SMFS — force spectrum")
        self.setWindowFlag(Qt.WindowType.Window)
        fit_on_screen(self, 1200, 720)
        style.apply_plot_defaults()

        _order, cols = _vars.columns(paths, [FORCE_KEY, RATE_KEY], db_path)
        meta = _db.get_file_metadata_bulk(paths, list(_fs.TEMPERATURE_KEYS), db_path)
        temp_K, n_temp = _fs.temperature_K(list(meta.values()))
        kT = _fs.kT_of(temp_K)
        rate, force = cols[RATE_KEY], cols[FORCE_KEY]
        self._fits, n_dropped = _fs.fit_all(rate, force, kT)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(4)
        root.setContentsMargins(8, 6, 8, 6)

        n_used = len(paths) - n_dropped
        temp_txt = (f"T = {temp_K:.1f} K (median of {n_temp} curves)"
                    if n_temp else "no stored temperature — cannot fit")
        hdr = QLabel(f"{n_used} ruptures fitted   |   {n_dropped} dropped "
                     f"(no rate, or rate ≤ 0)   |   {temp_txt}")
        hdr.setFont(style.font(hdr.font(), size_pt=style.FONT_SMALL_PT))
        root.addWidget(hdr)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_plot(rate, force, caption))
        splitter.addWidget(self._build_tables())
        splitter.setSizes([700, 500])
        root.addWidget(splitter, stretch=1)

    def _build_plot(self, rate, force, caption: str) -> QWidget:
        plot = pg.PlotWidget()
        rate_unit = _quant.unit_of(RATE_KEY)
        plot.setLabel("bottom", f"ln({_vars.label(RATE_KEY)} / {rate_unit})")
        set_si_label(plot, "left", _vars.label(FORCE_KEY), _quant.unit_of(FORCE_KEY), si=False)
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.addLegend(offset=(10, 10))
        set_plot_title(plot, caption=caption)

        ok = np.isfinite(rate) & np.isfinite(force) & (rate > 0.0)
        ln_r = np.log(rate[ok])
        dot = QColor(style.INK_MUTED)
        dot.setAlpha(style.DOT_ALPHA)
        plot.addItem(pg.ScatterPlotItem(ln_r, force[ok], size=style.DOT_SIZE,
                                        pen=None, brush=dot))
        if ln_r.size:
            grid = np.linspace(ln_r.min(), ln_r.max(), 300)
            for fit, color in zip(self._fits, style.SERIES_LABELED):
                # Only where the model gives a force of its own; the clamped
                # stretches are not drawn.
                y = np.where(fit.in_domain(grid), fit.predict(grid), np.nan)
                plot.plot(grid, y, pen=pg.mkPen(color, width=2),
                          name=fit.label, connect="finite")
        return plot

    def _build_tables(self) -> QWidget:
        pane = QWidget()
        lay = QVBoxLayout(pane)
        lay.setContentsMargins(0, 0, 0, 0)

        pg_box = QGroupBox("Parameters (±1σ)")
        pl = QVBoxLayout(pg_box)
        headers = ["Model"] + [f"{n} ({u})" for n, u in zip(_fs.PARAM_NAMES, _fs.PARAM_UNITS)]
        ptbl = _table(headers)
        ptbl.setRowCount(len(self._fits))
        for r, fit in enumerate(self._fits):
            cells = [fit.label]
            for name in _fs.PARAM_NAMES:
                if name not in fit.params:
                    cells.append("—")
                    continue
                v, e = fit.params[name]
                txt = f"{v:.3g} ± {e:.2g}"
                if name in fit.at_bound:
                    txt += f" (at {fit.at_bound[name]} bound)"
                cells.append(txt)
            _fill_row(ptbl, r, cells)
        pl.addWidget(ptbl)
        lay.addWidget(pg_box)

        cg = QGroupBox("Model Comparison")
        cl = QVBoxLayout(cg)
        guide = QLabel("Lower AICc / BIC = better model.  "
                       "ΔAIC > 2 meaningful, > 6 strong, > 10 decisive.")
        guide.setWordWrap(True)
        guide.setStyleSheet(style.qss_text(size_px=10))
        cl.addWidget(guide)
        ctbl = _table(["Model", "k", "n", "R²", "AICc", "ΔAICc", "ΔBIC"])
        ctbl.setRowCount(len(self._fits))
        best_aicc = min((f.stats["AICc"] for f in self._fits), default=float("nan"))
        best_bic  = min((f.stats["BIC"]  for f in self._fits), default=float("nan"))
        for r, fit in enumerate(self._fits):
            s = fit.stats
            d_aicc, d_bic = s["AICc"] - best_aicc, s["BIC"] - best_bic
            _fill_row(ctbl, r, [
                fit.label, str(s["k (for IC)"]), str(s["n (for IC)"]),
                f"{s['R²']:.3f}", f"{s['AICc']:.1f}",
                "★ best" if d_aicc < 1e-9 else f"+{d_aicc:.1f}",
                "★ best" if d_bic < 1e-9 else f"+{d_bic:.1f}",
            ], tint=style.TABLE_TINT_BEST if d_aicc < 1e-9 else None)
        cl.addWidget(ctbl)
        lay.addWidget(cg)
        lay.addStretch()
        return pane


def _table(headers: list[str]) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    t.horizontalHeader().setStretchLastSection(True)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setAlternatingRowColors(True)
    t.setMinimumHeight(100)
    return t


def _fill_row(table: QTableWidget, row: int, cells: list[str], tint: str | None = None) -> None:
    for c, text in enumerate(cells):
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if tint:
            item.setBackground(QColor(tint))
        table.setItem(row, c, item)
