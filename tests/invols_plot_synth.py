# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.


"""
Shared builders, readers, mouse helpers and fixtures for the invOLS plot tests
(drawing in test_invols_plot.py, interaction in test_invols_plot_interaction.py).

The widget needs no catalog, so the rows are built by hand.  Point state is
read the way pyqtgraph itself exposes it, so the tests do not depend on how the
widget arranges its items:

* every ``ScatterPlotItem`` added to the plot's view box is searched; each of
  its spots gives ``pos()``, ``pen()``, ``brush()``, ``symbol()`` and ``data()``;
* a pinned-edge marker is a spot whose symbol is a triangle;
* a tooltip is what the scatter's ``opts['tip']`` callable returns for the spot
  (the same call pyqtgraph makes when the pointer hovers it);
* clicks, double clicks and drags are real mouse events sent to the viewport.
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

T0 = 1_700_000_000.0          # an arbitrary Unix time; rows are an hour apart
HOUR = 3600.0
TRIANGLES = {"t", "t1", "t2", "t3"}
LEFT, RIGHT, MIDDLE = (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton,
                       Qt.MouseButton.MiddleButton)


# ── builders and readers ─────────────────────────────────────────────────────

def _row(i, own, **opts):
    """A TraceRow; `opts` may give source, current, time_s and in_use."""
    o = {"source": "own", "current": True, "time_s": "auto", "in_use": "own",
         **opts}
    assert set(o) == {"source", "current", "time_s", "in_use"}, f"unknown {opts}"
    t = T0 + i * HOUR if o["time_s"] == "auto" else o["time_s"]
    return TraceRow(file_id=i + 1, path=f"/data/f{i + 1}.ibw", time_s=t,
                    instrument_nm_v=199.14, own_nm_v=own,
                    fit_current=o["current"], rms_nm=0.2, source=o["source"],
                    group_id=None if o["source"] == "own" else 1,
                    in_use_nm_v=own if o["in_use"] == "own" else o["in_use"])


def _rows(values, **kw):
    return [_row(i, v, **kw) for i, v in enumerate(values)]


def _items(plot):
    """Every item in the view's child group, including those added with
    ignoreBounds=True (which ``ViewBox.addedItems`` leaves out)."""
    return list(plot.getViewBox().childGroup.childItems())


def _scatters(plot):
    return [it for it in _items(plot) if isinstance(it, pg.ScatterPlotItem)]


def _spots(plot):
    return [(sc, pt) for sc in _scatters(plot) for pt in sc.points()]


def _points(plot):
    """Drawn spots that are not pinned-edge markers."""
    return [(sc, pt) for sc, pt in _spots(plot) if pt.symbol() not in TRIANGLES]


def _markers(plot):
    return [(sc, pt) for sc, pt in _spots(plot) if pt.symbol() in TRIANGLES]


def _tip(scatter, spot) -> str:
    tip = scatter.opts["tip"]
    assert callable(tip), "the scatter has no tooltip callable (opts['tip'])"
    return str(tip(x=spot.pos().x(), y=spot.pos().y(), data=spot.data()))


def _tips(plot):
    return [_tip(sc, pt) for sc, pt in _spots(plot)]


def _is_filled(spot) -> bool:
    brush = spot.brush()
    return (brush.style() != Qt.BrushStyle.NoBrush
            and brush.color().alpha() > 0)


def _pen_hex(spot) -> str:
    return spot.pen().color().name().lower()


def _at(plot, t, y) -> QPoint:
    scene = plot.getViewBox().mapViewToScene(QPointF(t, y))
    return plot.mapFromScene(scene)


def _view_y(plot):
    return plot.getViewBox().viewRange()[1]


def _horizontal_lines(plot, value):
    """Visible horizontal lines at ``value`` (an InfiniteLine or a flat curve)."""
    found = []
    for it in _items(plot):
        if not it.isVisible():
            continue
        if isinstance(it, pg.InfiniteLine) and it.angle == 0:
            if math.isclose(float(it.value()), value, rel_tol=1e-9):
                found.append(it)
        elif isinstance(it, (pg.PlotDataItem, pg.PlotCurveItem)):
            y = it.getData()[1]
            if y is not None and len(y) and np.allclose(y, value, rtol=1e-9):
                found.append(it)
    return found


def _send(plot, kind, pos, button, buttons):
    vp = plot.viewport()
    ev = QMouseEvent(kind, QPointF(pos), QPointF(vp.mapToGlobal(pos)), button,
                     buttons, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(vp, ev)
    QApplication.processEvents()


def _drag(plot, start: QPoint, end: QPoint, button=LEFT):
    """A press, a few moves and a release.  The first move is a few pixels
    from the press, as a real hand's is."""
    _send(plot, QEvent.Type.MouseButtonPress, start, button, button)
    dx, dy = end.x() - start.x(), end.y() - start.y()
    for frac in (0.03, 0.5, 1.0):
        pos = QPoint(start.x() + int(dx * frac), start.y() + int(dy * frac))
        _send(plot, QEvent.Type.MouseMove, pos, Qt.MouseButton.NoButton, button)
    _send(plot, QEvent.Type.MouseButtonRelease, end, button,
          Qt.MouseButton.NoButton)


class _Spy:
    def __init__(self, signal):
        self.calls = []
        signal.connect(lambda *a: self.calls.append(tuple(a)))


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def plot(qapp):
    from smfs_catalog.invols_plot import InvolsPlot
    w = InvolsPlot()
    w.resize(640, 420)
    w.show()
    QApplication.processEvents()
    yield w
    w.close()
    w.deleteLater()


def _settle(plot):
    QApplication.processEvents()


def _bulk_with(outlier):
    return [float(v) for v in np.linspace(199, 201, 29)] + [outlier]
