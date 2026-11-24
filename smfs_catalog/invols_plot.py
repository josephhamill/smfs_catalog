# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""The invOLS plot: one point per trace, time on x and the own-fit nm/V on y,
with a box drawn by a left drag.  It takes ``TraceRow``s and touches no
database."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import pyqtgraph as pg
from PyQt6.QtCore import QPointF, Qt, pyqtSignal

from . import quantities, style, variables
from .invols_stats import TRANSFORM_KEY, display_range, is_drawn
from .qt_utils import _DateAxis, set_si_label

if TYPE_CHECKING:
    from .invols_store import TraceRow

_DEFAULT_Y = (0.0, 1.0)
_DEFAULT_X_HALF_SPAN_S = 3600.0
_SINGLE_TIME_HALF_SPAN_S = 60.0
_PIN_INSET = 0.02
_POINT_SIZE = 8
_MARKER_SIZE = 11
_CLICK_PX = 4.0
_SOURCE_TEXT = {"own": "own fit", "ensemble": "ensemble value"}


class _OutlierScatter(pg.ScatterPlotItem):
    """Counts for the x auto-range (an outlier's time stays in view) but not the
    y.  Its own bounding rectangle, which hit-testing uses, keeps both."""

    _full_bounds = False

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        if ax == 1 and not self._full_bounds:
            return (None, None)
        return super().dataBounds(ax, frac, orthoRange)

    def boundingRect(self):
        self._full_bounds = True
        try:
            return super().boundingRect()
        finally:
            self._full_bounds = False


class _BoxViewBox(pg.ViewBox):
    box_dragged = pyqtSignal(float, float, float, float, bool)
    box_cancelled = pyqtSignal()
    click_at = pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._moved_far = False

    def _clamped(self, pos):
        r = self.rect()
        return QPointF(min(max(pos.x(), r.left()), r.right()),
                       min(max(pos.y(), r.top()), r.bottom()))

    def mouseDragEvent(self, ev, axis=None):
        if axis is not None or ev.button() != Qt.MouseButton.LeftButton:
            super().mouseDragEvent(ev, axis)
            return
        ev.accept()
        start, now = ev.buttonDownPos(), self._clamped(ev.pos())
        far = pg.Point(now - start).length() >= _CLICK_PX
        if ev.isStart():
            self._moved_far = False
        self._moved_far = self._moved_far or far
        if not self._moved_far:
            if ev.isFinish():
                self.click_at.emit(ev.buttonDownScenePos())
        elif ev.isFinish() and not far:
            self.box_cancelled.emit()
        else:
            self._emit_box(start, now, ev.isFinish())

    def _emit_box(self, start, now, finished: bool) -> None:
        a, b = self.mapToView(start), self.mapToView(now)
        self.box_dragged.emit(a.x(), b.x(), a.y(), b.y(), finished)


def _drawn(rows: Sequence[TraceRow]) -> list[tuple[int, TraceRow]]:
    return [(i, r) for i, r in enumerate(rows)
            if is_drawn(r.own_nm_v, r.time_s)]


def _y_view(values: list[float]) -> tuple[float, float, tuple[int, ...]]:
    if not values:
        return (*_DEFAULT_Y, ())
    shown = display_range(values)
    if not (math.isfinite(shown.lo) and math.isfinite(shown.hi) and shown.hi > shown.lo):
        return (*_DEFAULT_Y, (0,) * len(values))
    return shown.lo, shown.hi, shown.side


def _x_view(times: list[float]) -> tuple[float, float]:
    if not times:
        now = time.time()
        return now - _DEFAULT_X_HALF_SPAN_S, now + _DEFAULT_X_HALF_SPAN_S
    lo, hi = min(times), max(times)
    pad = 0.05 * (hi - lo) or _SINGLE_TIME_HALF_SPAN_S
    return lo - pad, hi + pad


def _tip(x, y, data) -> str:
    own = quantities.format_value(TRANSFORM_KEY, data["value"], with_unit=True)
    lines = [data["name"], f"{variables.OWN_INVOLS_LABEL}: {own}",
             f"Source: {_SOURCE_TEXT.get(data['source'], data['source'])}"]
    in_use = quantities.format_value(TRANSFORM_KEY, data["in_use"], with_unit=True)
    if data["source"] == "ensemble" and in_use:
        lines.append(f"{variables.IN_USE_INVOLS_LABEL}: {in_use}")
    if data["side"]:
        lines.append("Outside the plotted range, shown at its edge")
    if not data["current"]:
        lines.append("Fit made under earlier fit settings")
    return "\n".join(lines)


def _spot(index: int, row: TraceRow, side: int, lo: float, hi: float) -> dict:
    colour = style.INVOLS_ENSEMBLE if row.source == "ensemble" else style.INVOLS_OWN
    inset = _PIN_INSET * (hi - lo)
    y = hi - inset if side > 0 else lo + inset if side < 0 else row.own_nm_v
    return {
        "pos": (row.time_s, y),
        "symbol": "t1" if side > 0 else "t" if side < 0 else "o",
        "size": _MARKER_SIZE if side else _POINT_SIZE,
        "pen": pg.mkPen(colour, width=1.5),
        "brush": pg.mkBrush(colour) if row.fit_current else pg.mkBrush(None),
        "data": {"index": index, "value": row.own_nm_v, "side": side,
                 "current": row.fit_current, "source": row.source,
                 "in_use": row.in_use_nm_v, "name": Path(row.path).name},
    }


class InvolsPlot(pg.PlotWidget):
    box_drawn = pyqtSignal(float, float, float, float)
    point_selected = pyqtSignal(int)
    point_opened = pyqtSignal(int)

    def __init__(self, parent=None):
        self._view_box = _BoxViewBox()
        super().__init__(parent, viewBox=self._view_box,
                         axisItems={"bottom": _DateAxis()})
        set_si_label(self, "left", quantities.with_unit(
            variables.OWN_INVOLS_LABEL, quantities.unit_of(TRANSFORM_KEY)),
            si=False)
        self._view_box.disableAutoRange()
        self._scatters = []
        self._outlier_scatter = None
        self._outliers = []
        self._mean_line = None
        self._box = None
        self._box_item = pg.QtWidgets.QGraphicsRectItem()
        self._box_item.setPen(pg.mkPen(style.INK_STRONG, width=1))
        self._box_item.setVisible(False)
        self._view_box.addItem(self._box_item, ignoreBounds=True)
        self._view_box.box_dragged.connect(self._on_box_dragged)
        self._view_box.click_at.connect(self._on_slow_click)
        self._view_box.box_cancelled.connect(self._restore_box)
        self._view_box.sigYRangeChanged.connect(self._place_outliers)
        self._apply_view([], [])

    def set_traces(self, rows: Sequence[TraceRow]) -> None:
        for scatter in self._scatters:
            self._view_box.removeItem(scatter)
        self._scatters, self._outlier_scatter, self._outliers = [], None, []
        self._view_box.setToolTip("")
        drawn = _drawn(rows)
        lo, hi, side = _y_view([r.own_nm_v for _, r in drawn])
        self._apply_view([r.time_s for _, r in drawn], [lo, hi])
        self._draw(drawn, side, lo, hi)

    def _draw(self, drawn, side, lo: float, hi: float) -> None:
        ordinary = [_spot(i, r, 0, lo, hi) for (i, r), s in zip(drawn, side) if not s]
        self._outliers = [(i, r) for (i, r), s in zip(drawn, side) if s]
        if ordinary:
            self._add_scatter(ordinary, False)
        if self._outliers:
            self._outlier_scatter = self._add_scatter(self._outlier_spots(), True)

    def _add_scatter(self, spots: list[dict], outlier: bool):
        cls = _OutlierScatter if outlier else pg.ScatterPlotItem
        scatter = cls(spots=spots, tip=_tip, hoverable=True)
        scatter.sigClicked.connect(self._on_clicked)
        self._view_box.addItem(scatter)
        self._scatters.append(scatter)
        return scatter

    def _outlier_spots(self) -> list[dict]:
        lo, hi = self._view_box.viewRange()[1]
        spots = []
        for index, row in self._outliers:
            side = 1 if row.own_nm_v > hi else -1 if row.own_nm_v < lo else 0
            spots.append(_spot(index, row, side, lo, hi))
        return spots

    def _place_outliers(self, *_) -> None:
        if self._outlier_scatter is not None:
            self._outlier_scatter.setData(spots=self._outlier_spots())

    def _apply_view(self, times: list[float], y_limits: list[float]) -> None:
        x0, x1 = _x_view(times)
        y0, y1 = (y_limits[0], y_limits[1]) if y_limits else _DEFAULT_Y
        self._view_box.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)

    def _point_at(self, scene_pos) -> int | None:
        if not self._view_box.sceneBoundingRect().contains(scene_pos):
            return None
        for item in self.scene().items(scene_pos):
            if item not in self._scatters:
                continue
            points = item.pointsAt(item.mapFromScene(scene_pos))
            if len(points):
                return int(points[0].data()["index"])
        return None

    def _on_clicked(self, _scatter, points, ev) -> None:
        if not ev.double():
            self.point_selected.emit(int(points[0].data()["index"]))

    def _on_slow_click(self, scene_pos) -> None:
        index = self._point_at(scene_pos)
        if index is not None:
            self.point_selected.emit(index)

    def mouseDoubleClickEvent(self, ev) -> None:
        super().mouseDoubleClickEvent(ev)
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        index = self._point_at(self.mapToScene(ev.position().toPoint()))
        if index is not None:
            self.point_opened.emit(index)

    def _on_box_dragged(self, t0, t1, y0, y1, finished) -> None:
        if finished:
            self.commit_box(t0, t1, y0, y1)
        else:
            self._show_box(t0, t1, y0, y1)

    def _restore_box(self) -> None:
        if self._box is None:
            self._box_item.setVisible(False)
        else:
            self._show_box(*self._box)

    def _show_box(self, t0, t1, y0, y1) -> None:
        self._box_item.setRect(min(t0, t1), min(y0, y1), abs(t1 - t0), abs(y1 - y0))
        self._box_item.setVisible(True)

    def commit_box(self, t0: float, t1: float, y0: float, y1: float) -> None:
        self._box = (float(min(t0, t1)), float(max(t0, t1)),
                     float(min(y0, y1)), float(max(y0, y1)))
        self._show_box(*self._box)
        self.box_drawn.emit(*self._box)

    def box(self) -> tuple[float, float, float, float] | None:
        return self._box

    def clear_box(self) -> None:
        self._box = None
        self._box_item.setVisible(False)

    def set_mean_line(self, value: float | None) -> None:
        if self._mean_line is not None:
            self._view_box.removeItem(self._mean_line)
            self._mean_line = None
        if value is None:
            return
        pen = pg.mkPen(style.INK_STRONG, width=1, style=Qt.PenStyle.DashLine)
        self._mean_line = pg.InfiniteLine(pos=value, angle=0, movable=False, pen=pen)
        self._view_box.addItem(self._mean_line, ignoreBounds=True)
