# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/trace_overlay_panel.py
#
# TraceOverlayPanel — a checkbox list of individual curves that plots each
# one's own chronological (x, F) trace over a 2D histogram when ticked.
# Traces are transformed lazily by the host window and remain display-only;
# this panel owns checkbox and plot-item state, not analysis or persistence.
# It is shared by the live 2DH windows and MeanCurveWindow popouts.

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QCheckBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from PyQt6.QtCore import Qt

from . import style


class TraceOverlayPanel(QWidget):
    """Fixed-width side panel: "Clear All Curves" + a scrollable checkbox
    list, one row per path currently available. Ticking a box calls
    `fetch_fn(path)` for that curve's (x, F) trace, already transformed into
    the host plot's coordinates and original sample order, and adds it to
    `plot_widget`; unticking removes it. `fetch_fn` returning None (e.g. no
    stored fit for that curve) unchecks the box instead of raising.
    """

    def __init__(
        self,
        plot_widget: pg.PlotWidget,
        fetch_fn: "Callable[[str], tuple | None]",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._plot     = plot_widget
        self._fetch_fn = fetch_fn
        self._checks: "dict[str, QCheckBox]" = {}
        self._series_index: "dict[str, int]" = {}
        # Each overlay is TWO items: a white casing and the coloured line
        # on top of it (style.py § H) — removed together.
        self._items:  "dict[str, tuple[pg.PlotDataItem, ...]]" = {}

        self.setFixedWidth(190)
        panel_l = QVBoxLayout(self)
        panel_l.setContentsMargins(0, 0, 0, 0)
        panel_l.setSpacing(2)

        clear_btn = QPushButton("Clear All Curves")
        clear_btn.setToolTip("Uncheck every overlaid trace.")
        clear_btn.clicked.connect(self.clear_all)
        panel_l.addWidget(clear_btn)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        self._list_l = QVBoxLayout(inner)
        self._list_l.setContentsMargins(4, 4, 4, 4)
        self._list_l.setSpacing(1)
        self._list_l.addStretch(1)
        scroll.setWidget(inner)
        panel_l.addWidget(scroll, stretch=1)

    def set_paths(self, paths) -> None:
        """Rebuild the checkbox list to match `paths`, preserving checked
        state (and the plotted line) for any path that's still present.
        Cheap — just widgets, no recompute."""
        current = set(paths)
        # Assign once for this panel's lifetime. A refresh can add/remove paths
        # while checked overlays remain visible; retaining the assignment keeps
        # their identity stable and prevents index shifts when an earlier
        # filename enters or leaves the sorted checkbox list.
        next_index = max(self._series_index.values(), default=-1) + 1
        for path in sorted(current - set(self._series_index)):
            self._series_index[path] = next_index
            next_index += 1
        stale = set(self._checks) - current
        for path in stale:
            chk = self._checks.pop(path)
            self._list_l.removeWidget(chk)
            chk.deleteLater()
            for item in self._items.pop(path, ()):
                self._plot.removeItem(item)

        new_paths = sorted(current - set(self._checks))
        for path in new_paths:
            chk = QCheckBox(Path(path).name)
            chk.setToolTip(path)
            chk.toggled.connect(lambda on, p=path: self._on_toggled(p, on))
            self._list_l.insertWidget(self._list_l.count() - 1, chk)
            self._checks[path] = chk

    def _on_toggled(self, path: str, on: bool) -> None:
        if not on:
            for item in self._items.pop(path, ()):
                self._plot.removeItem(item)
            return
        aligned = self._fetch_fn(path)
        if aligned is None:
            self._checks[path].setChecked(False)
            return
        x_aligned, F = aligned
        # SERIES_LABELED, not a cycled tab10: this panel IS the legend (one
        # checkbox per filename), which is what buys the extra slots.  The old
        # rota shared yellow/orange/brown/red with the 2DH's own YlOrBr ramp
        # underneath, so a trace could land on a background of its own colour.
        # The ramp is monochrome now (style.py § J) AND each trace is drawn with
        # a white casing under it — the ramp alone isn't enough, since every hue
        # falls under 3:1 somewhere in the middle of any monochrome ramp.
        # Identity is assigned once, not derived from the number currently
        # shown. Toggling or refreshing another trace must not recolour this one
        # or give two visible traces the same colour/dash identity.
        n = self._series_index[path]
        color = style.series_labeled(n)
        item = pg.PlotDataItem(
            x_aligned, F,
            pen=pg.mkPen(style.CASING_COLOR, width=style.W_CASING),
        )
        top = pg.PlotDataItem(
            x_aligned, F,
            pen=pg.mkPen(
                style.rgba(color, 255), width=style.W_OVERLAY,
                style=(Qt.PenStyle.DashLine
                       if style.series_dashed(n, style.SERIES_LABELED)
                       else Qt.PenStyle.SolidLine),
            ),
            name=Path(path).name,
        )
        self._plot.addItem(item)
        self._plot.addItem(top)
        self._items[path] = (item, top)

    def clear_all(self) -> None:
        """Uncheck every overlay checkbox — each toggle removes its own plot
        item via _on_toggled, so this needs no separate cleanup."""
        for chk in self._checks.values():
            chk.setChecked(False)
