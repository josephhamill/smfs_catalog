# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/categorical_window.py
#
# CategoricalStatsWindow — the non-numeric sibling of VariableStatsWindow
# A numeric queue column opens the drift-vs-time scatter and robust
# histogram; a CATEGORICAL column (Class, Status) has no continuous axis, so it
# gets the categorical analogue:
#
#   left  : time-series scatter, one horizontal band per category, points
#           colour-coded by category (so drift between classes stays visible).
#   right : bar chart of per-category counts, Y-linked to the bands so each
#           bar sits beside its band (mirrors the transposed histogram).
#   side  : one row per file, double-click routes to the worker viewer.
#
# Counts use every categorised file in scope; the scatter shows only those that
# also carry an acquisition date (same value-and-date rule as the numeric
# window).  No thresholds — categories are not gated here.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPushButton, QSplitter, QVBoxLayout, QWidget,
)
from PyQt6.QtWidgets import QLabel

from . import db as _db
from . import export_utils as _export
from .export_utils import slug as _slug
from . import style
from .qt_utils import _DateAxis, _make_session_header, fit_on_screen
from .variable_window import (
    _date_to_ts, _ts_to_date, _LIST_QSS, _PEN_NONE, _PEN_SEL,
)

# Category colours.  Known Class/Status values match the queue-table tints (but
# saturated for points); anything else cycles the shared TAB10 palette so new
# categories still get a distinct, stable-per-session colour.
_KNOWN: dict[str, tuple[int, int, int]] = {
    "event":        (46, 160, 67),
    "hit":          (46, 160, 67),
    "non_hit":      (200, 60, 60),
    "—":            (190, 190, 190),
    "non_event":    (140, 140, 140),
    "unavailable":  (230, 120, 60),
    "unusable":     (120, 145, 175),
    "unclassified": (190, 190, 190),
    "running":      (230, 180, 0),
    "done":         (46, 160, 67),
    "pending":      (120, 120, 200),
    "error":        (200, 60, 60),
}


def _jitter(n: int, rng: np.random.Generator) -> np.ndarray:
    """Vertical spread within a band so stacked same-category points show."""
    return rng.uniform(-0.32, 0.32, size=n)


class CategoricalStatsWindow(QMainWindow):
    """Counts and drift over time for one categorical queue column."""

    # Double-click of a file routes to the dashboard's singleton worker viewer,
    # matching VariableStatsWindow / the class windows.
    view_file_requested = pyqtSignal(str)

    def __init__(
        self,
        field_key:    str,
        label:        str,
        pairs:        list[tuple[str, str]],   # (path, category) for the scope
        db_path:      str,
        session_info: dict | None = None,
    ) -> None:
        super().__init__()
        self.setWindowFlag(Qt.WindowType.Window)
        self._field_key = field_key
        self._label     = label
        self._db_path   = db_path
        self.setWindowTitle(f"SMFS — category — {label}")
        fit_on_screen(self, 1100, 600)
        # Shared index space for the scatter, side list, and selection ring.
        self._plot_paths: list[str] = []
        self._selected:   int | None = None
        self._resolved:   list = []
        self._counts_by:  dict = {}
        self._cat_order:  list = []
        self._point_ts:   list = []
        self._n_no_date:  int  = 0

        style.apply_plot_defaults()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(4)
        root.setContentsMargins(8, 6, 8, 6)

        hdr = _make_session_header(session_info)
        if hdr is not None:
            root.addWidget(hdr)

        self._info = QLabel("")
        font = style.font(self._info.font(), size_pt=style.FONT_SMALL_PT)
        self._info.setFont(font)
        self._info.setFont(font)
        root.addWidget(self._info)

        sel_row = QHBoxLayout()
        self._sel_label = QLabel("")
        self._sel_label.setFont(font)
        sel_row.addWidget(self._sel_label)
        sel_row.addStretch()
        self._export_btn = QPushButton("Export…")
        self._export_btn.setToolTip(
            "Write the per-category counts and the per-file category/date "
            "table to the export folder, with a manifest."
        )
        self._export_btn.clicked.connect(self._on_export)
        sel_row.addWidget(self._export_btn)
        root.addLayout(sel_row)

        outer = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(outer, stretch=1)
        hsplit = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(hsplit)

        # ── Left: time-series scatter, one band per category ──────────────────
        self._drift = pg.PlotWidget(axisItems={"bottom": _DateAxis()})
        self._drift.setLabel("left", label)
        self._drift.setLabel("bottom", "Acquisition time")
        self._drift.showGrid(x=True, y=True, alpha=0.2)
        self._sc = pg.ScatterPlotItem(size=style.DOT_SIZE + 2, pen=_PEN_NONE)
        self._drift.addItem(self._sc)
        self._sc.sigClicked.connect(self._on_scatter_clicked)
        self._sel_marker = pg.ScatterPlotItem(
            size=16, symbol="o", pen=_PEN_SEL, brush=pg.mkBrush(None),
        )
        self._sel_marker.hide()
        self._drift.addItem(self._sel_marker, ignoreBounds=True)
        hsplit.addWidget(self._drift)

        # ── Right: per-category count bars, Y-linked to the bands ─────────────
        self._counts = pg.PlotWidget()
        self._counts.setLabel("bottom", "Count")
        self._counts.getAxis("left").setStyle(showValues=False)
        self._counts.showGrid(x=True, y=True, alpha=0.2)
        self._counts.getViewBox().setYLink(self._drift.getViewBox())
        self._bars = pg.BarGraphItem(x0=[], x1=[], y0=[], y1=[], pen=_PEN_NONE)
        self._counts.addItem(self._bars)
        hsplit.addWidget(self._counts)
        hsplit.setSizes([800, 300])

        # ── Right: side file-list ─────────────────────────────────────────────
        self._list = QListWidget()
        self._list.setStyleSheet(_LIST_QSS)
        self._list.currentRowChanged.connect(self._on_list_row_changed)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        outer.addWidget(self._list)
        outer.setSizes([900, 200])

        self._load(pairs)

    # ── Colour mapping ──────────────────────────────────────────────────────────

    def _colour(self, cat: str) -> tuple[int, int, int]:
        if cat in self._cat_colour:
            return self._cat_colour[cat]
        if cat in _KNOWN:
            rgb = _KNOWN[cat]
        else:
            rgb = style.rgba(style.series_labeled(len(self._cat_colour)))[:3]
        self._cat_colour[cat] = rgb
        return rgb

    # ── Data ──────────────────────────────────────────────────────────────────

    def _load(self, pairs: list[tuple[str, str]]) -> None:
        resolved = [(_db.normalize_path(p), (c or "unclassified")) for p, c in pairs]
        if not resolved:
            self._info.setText(f"{self._label}: nothing in the current scope")
            return

        # Stable category order: by descending count, so the biggest band sits
        # at the top and the legend reads naturally.
        cats = sorted({c for _, c in resolved})
        counts_by = {c: sum(1 for _, cc in resolved if cc == c) for c in cats}
        order = sorted(cats, key=lambda c: (-counts_by[c], c))
        self._cat_index = {c: i for i, c in enumerate(order)}
        self._cat_colour: dict[str, tuple[int, int, int]] = {}

        # Count bars — every categorised file, one bar per category.
        brushes = [pg.mkBrush(*self._colour(c)) for c in order]
        y_centres = np.arange(len(order), dtype=float)
        widths = np.array([counts_by[c] for c in order], dtype=float)
        self._bars.setOpts(
            x0=np.zeros(len(order)), x1=widths,
            y0=y_centres - 0.4, y1=y_centres + 0.4, brushes=brushes,
        )
        # Band labels on the scatter's Y axis.
        ticks = [(i, c) for c, i in self._cat_index.items()]
        self._drift.getAxis("left").setTicks([ticks])
        self._drift.setYRange(-0.6, len(order) - 0.4)

        # Scatter — files that also carry an acquisition date.
        dates = _db.get_measured_datetimes([rp for rp, _ in resolved], self._db_path)
        rng = np.random.default_rng(0)
        xs, ys, pt_brushes, paths = [], [], [], []
        n_no_date = 0
        for rp, c in resolved:
            ts = _date_to_ts(dates.get(rp))
            if not np.isfinite(ts):
                n_no_date += 1
                continue
            xs.append(ts)
            ys.append(self._cat_index[c])
            pt_brushes.append(pg.mkBrush(*self._colour(c)))
            paths.append(rp)
        ys = np.asarray(ys, dtype=float)
        if ys.size:
            ys = ys + _jitter(ys.size, rng)
        self._sc.setData(np.asarray(xs, dtype=float), ys,
                         brush=pt_brushes, data=list(range(len(paths))))
        self._plot_paths = paths
        # Kept for the export: the resolved (path, category) pairs, the
        # per-category counts, and each plotted point's date. The counts bar
        # chart and the drift scatter are two views of these.
        self._resolved   = list(resolved)
        self._counts_by  = dict(counts_by)
        self._cat_order  = list(order)
        self._point_ts   = list(xs)
        self._n_no_date  = n_no_date

        self._populate_list()

        parts = [self._label,
                 f"{len(resolved):,} files",
                 f"{len(order)} categories",
                 "  ·  ".join(f"{c}: {counts_by[c]:,}" for c in order)]
        if n_no_date:
            parts.append(f"{n_no_date:,} without a date (counts only)")
        self._info.setText("   —   ".join(parts))

    # ── Export ────────────────────────────────────────────────────────────────

    def export_provenance(self) -> dict:
        """This window's settings, for an export manifest — same protocol
        method as the other exporting windows."""
        return {
            "window":     "categorical_stats",
            "field":      self._field_key,
            "label":      self._label,
            "categories": list(self._cat_order),
        }

    def _on_export(self) -> None:
        """Both views as data: the per-category counts (the bar chart) and one
        row per file with its category and date (the drift scatter)."""
        if not self._resolved:
            QMessageBox.information(self, "Export", "Nothing to export.")
            return
        ts_by_path = dict(zip(self._plot_paths, self._point_ts))
        with _export.export_group(
            self._db_path, f"category_{_slug(self._field_key)}",
            ["_counts.csv", "_files.csv"], kind="categorical_stats",
        ) as g:
            g.contributing_files(rp for rp, _c in self._resolved)
            g.note_dict(self.export_provenance())
            g.note(n_files=len(self._resolved),
                   n_categories=len(self._cat_order),
                   n_missing_date=self._n_no_date)
            g.table("_counts.csv", ["category", "count"],
                    [(c, int(self._counts_by.get(c, 0))) for c in self._cat_order])
            g.table(
                "_files.csv", ["path", "category", "measured_date"],
                [(rp, c, _ts_to_date(ts_by_path[rp]) if rp in ts_by_path else "")
                 for rp, c in self._resolved],
            )
        QMessageBox.information(self, "Export", g.message())

    # ── Side list + selection ─────────────────────────────────────────────────

    def _populate_list(self) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        for path in self._plot_paths:
            self._list.addItem(QListWidgetItem(Path(path).name))
        self._list.blockSignals(False)
        self._selected = None
        self._sel_marker.hide()
        self._sel_label.setText("")

    def _select(self, idx: int | None) -> None:
        self._selected = idx
        if idx is None or not (0 <= idx < len(self._plot_paths)):
            self._sel_marker.hide()
            self._sel_label.setText("")
            return
        pts = self._sc.getData()
        self._sel_marker.setData([pts[0][idx]], [pts[1][idx]])
        self._sel_marker.show()
        self._sel_label.setText(Path(self._plot_paths[idx]).name)

    def _on_scatter_clicked(self, _plot, points) -> None:
        if not len(points):
            return
        idx = points[0].data()
        if idx is None:
            return
        self._list.blockSignals(True)
        self._list.setCurrentRow(int(idx))
        self._list.blockSignals(False)
        self._select(int(idx))

    def _on_list_row_changed(self, row: int) -> None:
        self._select(row if row >= 0 else None)

    def _on_double_click(self, _item) -> None:
        if self._selected is not None and 0 <= self._selected < len(self._plot_paths):
            self.view_file_requested.emit(self._plot_paths[self._selected])
