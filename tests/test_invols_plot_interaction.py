# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.


"""
The invOLS plot, interaction: the box drawn by a left drag, clicks and double
clicks reporting the row index, axis drags, zoom and pan with pinned markers,
and overlapping marks.  Builders and readers are in invols_plot_synth.py;
drawing is in test_invols_plot.py.
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pg = pytest.importorskip("pyqtgraph")
pytest.importorskip("PyQt6")

import numpy as np                                                  # noqa: E402
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt                # noqa: E402
from PyQt6.QtGui import QMouseEvent                                 # noqa: E402
from PyQt6.QtTest import QTest                                      # noqa: E402
from PyQt6.QtWidgets import QApplication                            # noqa: E402

from smfs_catalog import quantities, style                          # noqa: E402
from smfs_catalog.invols_store import TraceRow                      # noqa: E402
from smfs_catalog.qt_utils import _DateAxis                         # noqa: E402
from invols_plot_synth import (  # noqa: E402
    HOUR,
    LEFT,
    MIDDLE,
    RIGHT,
    T0,
    TRIANGLES,
    _Spy,
    _at,
    _bulk_with,
    _drag,
    _markers,
    _points,
    _row,
    _rows,
    _scatters,
    _send,
    _settle,
    _spots,
    _tip,
    _view_y,
    pg,
)
from invols_plot_synth import plot, qapp  # noqa: E402,F401  (fixtures)


# ── the box ──────────────────────────────────────────────────────────────────

def test_commit_box_normalises_stores_and_emits_once(plot):
    spy = _Spy(plot.box_drawn)
    plot.commit_box(5, 1, 300, 100)
    assert spy.calls == [(1.0, 5.0, 100.0, 300.0)]
    assert plot.box() == (1.0, 5.0, 100.0, 300.0)


def test_clear_box_gives_none(plot):
    plot.commit_box(5, 1, 300, 100)
    plot.clear_box()
    assert plot.box() is None


def test_no_box_before_one_is_drawn(plot):
    assert plot.box() is None


def _prepare_for_drag(plot):
    plot.set_traces(_rows([200.0, 201.0, 199.0, 200.5]))
    plot.setXRange(T0, T0 + 4 * HOUR, padding=0)
    plot.setYRange(0, 40, padding=0)
    _settle(plot)


def test_left_drag_draws_a_box_in_data_coordinates(plot):
    _prepare_for_drag(plot)
    spy = _Spy(plot.box_drawn)
    # from the upper right to the lower left, so normalising is exercised too
    _drag(plot, _at(plot, T0 + 3.5 * HOUR, 35), _at(plot, T0 + 0.5 * HOUR, 5))
    assert len(spy.calls) == 1
    t0, t1, y0, y1 = spy.calls[0]
    span_t, span_y = 4 * HOUR, 40
    assert t0 == pytest.approx(T0 + 0.5 * HOUR, abs=0.02 * span_t)
    assert t1 == pytest.approx(T0 + 3.5 * HOUR, abs=0.02 * span_t)
    assert y0 == pytest.approx(5, abs=0.02 * span_y)
    assert y1 == pytest.approx(35, abs=0.02 * span_y)
    assert plot.box() == (t0, t1, y0, y1)


@pytest.mark.parametrize("button", [RIGHT, MIDDLE], ids=["right", "middle"])
def test_right_and_middle_drag_do_not_draw_a_box(plot, button):
    _prepare_for_drag(plot)
    spy = _Spy(plot.box_drawn)
    _drag(plot, _at(plot, T0 + 3.5 * HOUR, 35), _at(plot, T0 + 0.5 * HOUR, 5),
          button=button)
    assert spy.calls == []
    assert plot.box() is None


# ── clicks report the index into the rows last given ─────────────────────────

def _click_at_spot(plot, spot, dy=0, double=False):
    p = _at(plot, spot.pos().x(), spot.pos().y())
    p = QPoint(p.x(), p.y() + dy)
    click = QTest.mouseDClick if double else QTest.mouseClick
    click(plot.viewport(), LEFT, pos=p)
    QApplication.processEvents()


def _spot_at_time(plot, time_s):
    for _, pt in _spots(plot):
        if math.isclose(pt.pos().x(), time_s, abs_tol=1.0):
            return pt
    raise AssertionError(f"no spot at time {time_s}")


def test_click_reports_the_row_index_even_when_an_earlier_row_is_undrawn(plot):
    rows = _rows([200.0, None, 201.0, 199.0])
    plot.set_traces(rows)
    _settle(plot)
    spy = _Spy(plot.point_selected)
    _click_at_spot(plot, _spot_at_time(plot, rows[2].time_s))
    assert spy.calls == [(2,)]


def test_double_click_opens_the_row_index_even_when_an_earlier_row_is_undrawn(plot):
    rows = _rows([200.0, None, 201.0, 199.0])
    plot.set_traces(rows)
    _settle(plot)
    opened = _Spy(plot.point_opened)
    _click_at_spot(plot, _spot_at_time(plot, rows[2].time_s), double=True)
    assert opened.calls == [(2,)]


def test_index_follows_the_rows_last_given(plot):
    plot.set_traces(_rows([200.0, 201.0, 199.0]))
    rows = [_row(5, 205.0), _row(6, None), _row(7, 206.0)]
    plot.set_traces(rows)
    _settle(plot)
    spy = _Spy(plot.point_selected)
    _click_at_spot(plot, _spot_at_time(plot, rows[2].time_s))
    assert spy.calls == [(2,)]


def test_click_on_a_pinned_marker_reports_its_row_index(plot):
    values = [None] + _bulk_with(9957.0)
    rows = _rows(values)
    plot.set_traces(rows)
    _settle(plot)
    (_, marker), = _markers(plot)
    spy = _Spy(plot.point_selected)
    _click_at_spot(plot, marker, dy=3)        # a little inside the edge
    assert spy.calls == [(len(rows) - 1,)]


def test_double_click_on_a_pinned_marker_opens_its_row_index(plot):
    rows = _rows([None] + _bulk_with(9957.0))
    plot.set_traces(rows)
    _settle(plot)
    (_, marker), = _markers(plot)
    opened = _Spy(plot.point_opened)
    _click_at_spot(plot, marker, dy=3, double=True)
    assert opened.calls == [(len(rows) - 1,)]


# ── I1: a left drag on an axis is not a box ──────────────────────────────────

def _axis_centre(plot, name) -> QPoint:
    scene = plot.getAxis(name).sceneBoundingRect().center()
    return plot.mapFromScene(scene)


class _DragStub:
    """A drag event as pyqtgraph's AxisItem hands it to the view box
    (``ViewBox.mouseDragEvent(ev, axis=...)``).  Real mouse events do reach an
    axis item offscreen when they are paced about 25 ms apart (see the real-event
    test below); this stub drives the same path without waiting, and was checked
    to pan a plain ``pg.ViewBox`` along the axis it is given."""

    def __init__(self, last, pos, down, *, start=False, finish=False):
        self._last, self._pos, self._down = last, pos, down
        self._start, self._finish = start, finish
        self.accepted = False

    def button(self): return LEFT
    def buttons(self): return LEFT
    def pos(self): return pg.Point(self._pos)
    def lastPos(self): return pg.Point(self._last)
    def buttonDownPos(self, *a): return pg.Point(self._down)
    def scenePos(self): return pg.Point(self._pos)
    def lastScenePos(self): return pg.Point(self._last)
    def buttonDownScenePos(self, *a): return pg.Point(self._down)
    def isStart(self): return self._start
    def isFinish(self): return self._finish
    def accept(self): self.accepted = True
    def ignore(self): pass
    def modifiers(self): return Qt.KeyboardModifier.NoModifier


def _axis_drag(vb, axis, step):
    """Drag by `step` pixels along `axis` (0 = x, 1 = y) in three events."""
    a, b = (100, 100), (100 + step[0], 100 + step[1])
    vb.mouseDragEvent(_DragStub(a, a, a, start=True), axis=axis)
    vb.mouseDragEvent(_DragStub(a, b, a), axis=axis)
    vb.mouseDragEvent(_DragStub(b, b, a, finish=True), axis=axis)


@pytest.mark.parametrize("axis, step", [(1, (0, 40)), (0, (40, 0))],
                         ids=["left-axis", "bottom-axis"])
@pytest.mark.parametrize("has_box", [False, True], ids=["no-box", "box-kept"])
def test_left_drag_on_an_axis_pans_and_draws_no_box(plot, axis, step, has_box):
    _prepare_for_drag(plot)
    if has_box:
        plot.commit_box(T0, T0 + HOUR, 10, 20)
    before_box = plot.box()
    before_view = plot.getViewBox().viewRange()
    spy = _Spy(plot.box_drawn)
    _axis_drag(plot.getViewBox(), axis, step)
    assert plot.box() == before_box
    assert spy.calls == []
    after = plot.getViewBox().viewRange()
    tol = (1.0, 1e-6)                    # seconds on x, nm/V on y
    assert after[axis] != pytest.approx(before_view[axis], abs=tol[axis])  # panned
    assert after[1 - axis] == pytest.approx(before_view[1 - axis],
                                            abs=tol[1 - axis])


# ── I2: a slow click with a tiny movement keeps the box ──────────────────────

def test_slow_click_with_one_pixel_movement_selects_and_keeps_the_box(plot):
    rows = _rows([200.0, 201.0, 199.0])
    plot.set_traces(rows)
    _settle(plot)
    plot.commit_box(T0 - 10, T0 + 10, 100, 300)
    box = plot.box()
    boxes = _Spy(plot.box_drawn)
    picked = _Spy(plot.point_selected)
    p = _at(plot, rows[1].time_s, 201.0)
    _send(plot, QEvent.Type.MouseButtonPress, p, LEFT, LEFT)
    QTest.qWait(650)                         # past pyqtgraph's 0.5 s drag time
    _send(plot, QEvent.Type.MouseMove, QPoint(p.x() + 1, p.y()),
          Qt.MouseButton.NoButton, LEFT)
    _send(plot, QEvent.Type.MouseButtonRelease, QPoint(p.x() + 1, p.y()), LEFT,
          Qt.MouseButton.NoButton)
    assert plot.box() == box
    assert boxes.calls == []
    assert picked.calls == [(1,)]


# ── I3: a double click outside the plot area opens nothing ───────────────────

def test_double_click_on_the_axis_beneath_an_offscreen_point_opens_nothing(plot):
    rows = _rows([200.0, 201.0, 199.0])
    plot.set_traces(rows)
    vb = plot.getViewBox()
    per_px = (4 * HOUR) / vb.width()
    plot.setXRange(T0 + 20 * per_px, T0 + 20 * per_px + 4 * HOUR, padding=0)
    _settle(plot)
    p = _at(plot, rows[0].time_s, 200.0)
    scene = plot.mapToScene(p)
    assert plot.getAxis("left").sceneBoundingRect().contains(scene), \
        "test set-up: the point must lie beneath the left axis"
    opened = _Spy(plot.point_opened)
    QTest.mouseDClick(plot.viewport(), LEFT, pos=p)
    QApplication.processEvents()
    assert opened.calls == []


# ── I4: pinned markers stay honest after zoom or pan ─────────────────────────

OUTLIER_AT = 29


def _outlier_spot(plot):
    for sc, pt in _spots(plot):
        if math.isclose(pt.pos().x(), T0 + OUTLIER_AT * HOUR, abs_tol=1.0):
            return sc, pt
    raise AssertionError("the outlier has no spot")


def _view(plot, lo, hi):
    plot.setYRange(lo, hi, padding=0)
    _settle(plot)


def test_outlier_inside_a_zoomed_out_view_is_at_its_real_position(plot):
    plot.set_traces(_rows(_bulk_with(9957.0)))
    _settle(plot)
    _view(plot, 0, 11000)
    sc, pt = _outlier_spot(plot)
    assert pt.pos().y() == pytest.approx(9957.0)
    assert pt.symbol() not in TRIANGLES
    assert not _markers(plot)


def test_outlier_goes_back_to_the_top_edge_when_the_view_returns_to_the_bulk(plot):
    plot.set_traces(_rows(_bulk_with(9957.0)))
    _view(plot, 0, 11000)
    _view(plot, 190, 210)
    sc, pt = _outlier_spot(plot)
    assert pt.symbol() in TRIANGLES
    assert 190 <= pt.pos().y() <= 210
    assert pt.pos().y() >= 210 - 0.1 * 20
    assert "9957" in _tip(sc, pt)


def test_zooming_into_the_bulk_pins_only_the_outlier(plot):
    values = _bulk_with(9957.0)
    plot.set_traces(_rows(values))
    _view(plot, 199.5, 200.5)
    markers = _markers(plot)
    assert len(markers) == 1
    assert math.isclose(markers[0][1].pos().x(), T0 + OUTLIER_AT * HOUR, abs_tol=1.0)
    real = sorted(values[:OUTLIER_AT])
    drawn = sorted(pt.pos().y() for _, pt in _points(plot))
    assert drawn == pytest.approx(real)           # off-screen, not pinned


def test_box_around_a_zoomed_out_outlier_contains_it(plot):
    from smfs_catalog.invols_stats import points_in_box
    values = _bulk_with(9957.0)
    times = [T0 + i * HOUR for i in range(len(values))]
    plot.set_traces(_rows(values))
    _view(plot, 0, 11000)
    _, pt = _outlier_spot(plot)
    box = (T0 - HOUR, T0 + 31 * HOUR, 9000.0, 10500.0)
    assert box[2] <= pt.pos().y() <= box[3]
    mask = points_in_box(times, values, box)
    assert mask[OUTLIER_AT] and mask.sum() == 1
    bulk_box = (T0 - HOUR, T0 + 31 * HOUR, 190.0, 210.0)
    mask = points_in_box(times, values, bulk_box)
    assert not mask[OUTLIER_AT] and mask.sum() == OUTLIER_AT


def test_low_outlier_behaves_the_same_at_the_bottom_edge(plot):
    values = _bulk_with(12.5)
    plot.set_traces(_rows(values))
    _view(plot, 0, 300)
    sc, pt = _outlier_spot(plot)
    assert pt.pos().y() == pytest.approx(12.5)
    assert pt.symbol() not in TRIANGLES
    _view(plot, 190, 210)
    sc, pt = _outlier_spot(plot)
    assert pt.symbol() in TRIANGLES
    assert 190 <= pt.pos().y() <= 190 + 0.1 * 20
    assert "12.5" in _tip(sc, pt)
    _view(plot, 199.5, 200.5)
    assert len(_markers(plot)) == 1


# ── n: press, release, double-click, release on a point ──────────────────────

def test_click_then_double_click_sequence_selects_once_and_opens_once(plot):
    rows = _rows([200.0, 201.0, 199.0])
    plot.set_traces(rows)
    _settle(plot)
    picked, opened = _Spy(plot.point_selected), _Spy(plot.point_opened)
    p = _at(plot, rows[1].time_s, 201.0)
    _send(plot, QEvent.Type.MouseButtonPress, p, LEFT, LEFT)
    _send(plot, QEvent.Type.MouseButtonRelease, p, LEFT, Qt.MouseButton.NoButton)
    _send(plot, QEvent.Type.MouseButtonDblClick, p, LEFT, LEFT)
    _send(plot, QEvent.Type.MouseButtonRelease, p, LEFT, Qt.MouseButton.NoButton)
    assert picked.calls == [(1,)]
    assert opened.calls == [(1,)]


# ── F2: a box dragged past the view edge stops at the edge ───────────────────

N_MANY = 401


def _many_rows():
    values = [float(v) for v in np.linspace(190, 210, N_MANY)]
    rows = [_row(i, v, time_s=T0 + i * HOUR) for i, v in enumerate(values)]
    return rows, values


def _selected(plot, rows, values):
    from smfs_catalog.invols_stats import points_in_box
    times = [r.time_s for r in rows]
    mask = points_in_box(times, values, plot.box())
    return np.asarray(times)[mask], np.asarray(values)[mask]


@pytest.mark.parametrize("overshoot", [15, -15], ids=["below", "above"])
def test_box_dragged_past_the_y_edge_is_limited_to_the_view(plot, overshoot):
    rows, values = _many_rows()
    plot.set_traces(rows)
    plot.setXRange(T0, T0 + (N_MANY - 1) * HOUR, padding=0)
    plot.setYRange(198, 200, padding=0)
    _settle(plot)
    ylo, yhi = _view_y(plot)
    start = _at(plot, T0 + 100 * HOUR, 199)
    edge = _at(plot, T0 + 200 * HOUR, ylo if overshoot > 0 else yhi)
    _drag(plot, start, QPoint(edge.x(), edge.y() + overshoot))
    assert plot.box() is not None
    assert plot.box()[2] >= ylo - 1e-6 and plot.box()[3] <= yhi + 1e-6
    _, picked = _selected(plot, rows, values)
    assert len(picked) > 0
    assert picked.min() >= ylo - 1e-6 and picked.max() <= yhi + 1e-6


@pytest.mark.parametrize("side", ["left", "right"])
def test_box_dragged_past_the_time_edge_is_limited_to_the_view(plot, side):
    rows, values = _many_rows()
    plot.set_traces(rows)
    plot.setXRange(T0 + 100 * HOUR, T0 + 300 * HOUR, padding=0)
    plot.setYRange(189, 211, padding=0)
    _settle(plot)
    (xlo, xhi), _ = plot.getViewBox().viewRange()
    start = _at(plot, T0 + 200 * HOUR, 200)
    edge = _at(plot, xlo if side == "left" else xhi, 195)
    _drag(plot, start, QPoint(edge.x() + (-15 if side == "left" else 15), edge.y()))
    box = plot.box()
    assert box is not None
    assert box[0] >= xlo - 1.0 and box[1] <= xhi + 1.0
    picked, _ = _selected(plot, rows, values)
    assert len(picked) > 0
    assert picked.min() >= xlo - 1.0 and picked.max() <= xhi + 1.0


def test_commit_box_from_code_keeps_limits_beyond_the_view(plot):
    plot.set_traces(_rows([199.0, 200.0, 201.0]))
    plot.setYRange(198, 200, padding=0)
    _settle(plot)
    plot.commit_box(T0 - 5 * HOUR, T0 + 50 * HOUR, 100, 300)
    assert plot.box() == (T0 - 5 * HOUR, T0 + 50 * HOUR, 100.0, 300.0)


def test_drag_that_returns_to_its_start_keeps_the_earlier_box(plot):
    _prepare_for_drag(plot)
    plot.commit_box(T0, T0 + HOUR, 10, 20)
    box = plot.box()
    spy = _Spy(plot.box_drawn)
    p = _at(plot, T0 + 2 * HOUR, 20)
    _send(plot, QEvent.Type.MouseButtonPress, p, LEFT, LEFT)
    out = QPoint(p.x() + 10, p.y() + 10)
    _send(plot, QEvent.Type.MouseMove, out, Qt.MouseButton.NoButton, LEFT)
    _send(plot, QEvent.Type.MouseMove, p, Qt.MouseButton.NoButton, LEFT)
    _send(plot, QEvent.Type.MouseButtonRelease, p, LEFT, Qt.MouseButton.NoButton)
    assert plot.box() == box
    assert spy.calls == []


# ── F3: one click and the double click agree on the topmost mark ─────────────

def _overlapping_marks(plot):
    """An ordinary point and an outlier marker at the same time and the same
    pixel.  Returns (pixel, index of the topmost mark)."""
    values = [float(v) for v in np.linspace(199, 201, 29)]
    rows = [_row(i, v) for i, v in enumerate(values)]
    rows.append(_row(29, 9957.0, time_s=rows[28].time_s))
    plot.set_traces(rows)
    marker_y = []
    for hi in (201.5, 203.0):
        plot.setYRange(199.0, hi, padding=0)
        _settle(plot)
        marker_y.append(_markers(plot)[0][1].pos().y())
    slope = (marker_y[1] - marker_y[0]) / (203.0 - 201.5)
    plot.setYRange(199.0, 201.5 + (values[28] - marker_y[0]) / slope, padding=0)
    _settle(plot)
    marker = _markers(plot)[0][1]
    assert marker.pos().y() == pytest.approx(values[28], abs=1e-6)
    pixel = _at(plot, rows[28].time_s, values[28])
    scene = plot.mapToScene(pixel)
    found = {}
    for sc in _scatters(plot):
        pts = sc.pointsAt(sc.mapFromScene(scene))
        if len(pts):
            found[sc] = int(pts[0].data()["index"])
    assert sorted(found.values()) == [28, 29], "test set-up: marks must overlap"
    top = next(it for it in plot.scene().items(scene)
               if isinstance(it, pg.ScatterPlotItem))
    return pixel, found[top]


def test_fast_click_on_overlapping_marks_selects_the_topmost(plot):
    pixel, top = _overlapping_marks(plot)
    spy = _Spy(plot.point_selected)
    QTest.mouseClick(plot.viewport(), LEFT, pos=pixel)
    QApplication.processEvents()
    assert spy.calls == [(top,)]


def test_slow_click_on_overlapping_marks_selects_the_topmost(plot):
    pixel, top = _overlapping_marks(plot)
    spy = _Spy(plot.point_selected)
    _send(plot, QEvent.Type.MouseButtonPress, pixel, LEFT, LEFT)
    QTest.qWait(650)
    moved = QPoint(pixel.x() + 1, pixel.y())
    _send(plot, QEvent.Type.MouseMove, moved, Qt.MouseButton.NoButton, LEFT)
    _send(plot, QEvent.Type.MouseButtonRelease, moved, LEFT, Qt.MouseButton.NoButton)
    assert spy.calls == [(top,)]


def test_double_click_sequence_on_overlapping_marks_agrees_on_the_topmost(plot):
    pixel, top = _overlapping_marks(plot)
    picked, opened = _Spy(plot.point_selected), _Spy(plot.point_opened)
    _send(plot, QEvent.Type.MouseButtonPress, pixel, LEFT, LEFT)
    _send(plot, QEvent.Type.MouseButtonRelease, pixel, LEFT, Qt.MouseButton.NoButton)
    _send(plot, QEvent.Type.MouseButtonDblClick, pixel, LEFT, LEFT)
    _send(plot, QEvent.Type.MouseButtonRelease, pixel, LEFT, Qt.MouseButton.NoButton)
    assert picked.calls == [(top,)]
    assert opened.calls == [(top,)]


# ── a real, paced left drag on an axis pans and draws no box ─────────────────

def _paced_drag(plot, start, end, pace_ms=25):
    _send(plot, QEvent.Type.MouseButtonPress, start, LEFT, LEFT)
    QTest.qWait(pace_ms)
    dx, dy = end.x() - start.x(), end.y() - start.y()
    for frac in (0.1, 0.5, 1.0):
        pos = QPoint(start.x() + int(dx * frac), start.y() + int(dy * frac))
        _send(plot, QEvent.Type.MouseMove, pos, Qt.MouseButton.NoButton, LEFT)
        QTest.qWait(pace_ms)
    _send(plot, QEvent.Type.MouseButtonRelease, end, LEFT, Qt.MouseButton.NoButton)


@pytest.mark.parametrize("axis, name, step",
                         [(1, "left", (0, 40)), (0, "bottom", (40, 0))])
def test_real_paced_left_drag_on_an_axis_pans_and_draws_no_box(plot, axis, name, step):
    _prepare_for_drag(plot)
    plot.commit_box(T0, T0 + HOUR, 10, 20)
    box, before = plot.box(), plot.getViewBox().viewRange()
    spy = _Spy(plot.box_drawn)
    start = _axis_centre(plot, name)
    _paced_drag(plot, start, QPoint(start.x() + step[0], start.y() + step[1]))
    tol = (1.0, 1e-6)
    after = plot.getViewBox().viewRange()
    assert plot.box() == box and spy.calls == []
    assert after[axis] != pytest.approx(before[axis], abs=tol[axis])
    assert after[1 - axis] == pytest.approx(before[1 - axis], abs=tol[1 - axis])
