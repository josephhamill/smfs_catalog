# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/sample_marks.py — line, or the samples themselves.
#
# A line segment asserts the tip moved between two samples.  Through a held
# segment it did not: the piezo sits at one position for thousands of samples,
# so the line draws motion that never happened.  Dots say only where the
# samples are.
#
# Dots cost more to draw — one repaint of a ~100k-sample curve measures 13 ms
# as a line against 48 ms as dots, and ~40 MB more peak RSS.  Stepping curve to
# curve is one repaint and does not notice; dragging a pan across a full curve
# is roughly 20 fps and does.  Hence a toggle rather than a fixed choice, and
# hence the toggle is offered in every window that draws samples rather than
# once on the dashboard: it is wanted where the heavy curve is.
#
# Switching on visible sample count automatically was rejected — the plot then
# changes appearance on its own and the rule has to be explained.
#
# ── Why this is not in style.py ───────────────────────────────────────────────
#
# style.py answers "what does a data trace look like?" and owns both answers
# (style.data_marks).  This module answers "which one is in force right now?",
# which is application state with a user attached to it, and applies it to the
# live traces.  Same boundary that keeps QtWidgets out of style.py.
#
# ── Why traces are made here rather than by their windows ─────────────────────
#
# A trace built by hand is a trace the toggle cannot reach: it would keep its
# pen and quietly stay a line while everything else switched.  So this module
# builds them, and tests/test_sample_marks.py asserts no window calls
# style.data_pen directly.

from __future__ import annotations

import weakref

import pyqtgraph as pg
from PyQt6.QtCore import QObject, pyqtSignal

from . import db as _db
from . import style

# The stored value is a word, not a bool, because app_settings holds text and a
# figure setting is read by a human looking at the row.
APP_SETTING = _db.APP_SETTING_SAMPLE_MARKS
_DOTS  = "dots"
_LINES = "lines"

# Dots by default.  The held segments are the reason the app exists; the
# default should not draw motion through them.
_DEFAULT_DOTS = True

_dots: bool = _DEFAULT_DOTS
_db_path: str | None = None

# Every live data trace, against the appearance it was asked for.  Weak, so a
# closed window's plots are collected rather than kept alive by this registry —
# on this machine that matters more than the redraw cost.
_tracked: "weakref.WeakKeyDictionary[pg.PlotDataItem, tuple]" = (
    weakref.WeakKeyDictionary()
)


class _Broadcast(QObject):
    changed = pyqtSignal(bool)


_broadcast = _Broadcast()

# Emitted with the new mode whenever it changes, so that every open window's
# toggle agrees without any window knowing about any other.
changed = _broadcast.changed


def dots() -> bool:
    """True if data traces are currently drawn as their samples."""
    return _dots


def load(db_path: str) -> bool:
    """Adopt the mode stored for this catalog and remember where to save it.

    Called once at startup, before any window exists.  Until it is called the
    mode is the default and nothing is persisted, which is what a test or a
    headless import wants.
    """
    global _dots, _db_path
    _db_path = db_path
    stored = _db.get_app_setting(APP_SETTING, "", db_path)
    if stored in (_DOTS, _LINES):
        _dots = stored == _DOTS
    _apply()
    return _dots


def set_dots(on: bool) -> None:
    """Switch every live data trace, and remember the choice."""
    global _dots
    on = bool(on)
    if on == _dots:
        return
    _dots = on
    _apply()
    if _db_path is not None:
        _db.set_app_setting(APP_SETTING, _DOTS if on else _LINES, _db_path)
    changed.emit(on)


def trace(plot, *, color=style.DATA, width: float = style.W_DATA,
          alpha: int = 255, name: str | None = None, x=None, y=None):
    """Add one rule-1 data trace to `plot`, dressed in the current mode.

    Every trace of a curve's samples is made here.  Pass the SERIES_LINE hue
    when several signals share a panel, and `alpha` when they are stacked under
    markers that have to read through them.
    """
    item = plot.plot([] if x is None else x, [] if y is None else y, name=name)
    _tracked[item] = (color, width, alpha)
    _dress(item, color, width, alpha)
    return item


def _dress(item, color, width, alpha) -> None:
    marks = style.data_marks(color, width, alpha, dots=_dots)
    item.setPen(marks["pen"])
    item.setSymbol(marks["symbol"])
    if marks["symbol"] is not None:
        item.setSymbolSize(marks["symbolSize"])
        item.setSymbolPen(marks["symbolPen"])
        item.setSymbolBrush(marks["symbolBrush"])


def _apply() -> None:
    for item, spec in list(_tracked.items()):
        try:
            _dress(item, *spec)
        except RuntimeError:
            # The C++ side went away with its window while the wrapper lived on.
            _tracked.pop(item, None)
