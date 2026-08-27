# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Guard + behaviour: a data trace is drawn as the samples it is made of, or as a
line, and every one of them can be switched.

WHY THE GUARD EXISTS.  A trace built with a hand-rolled pen is a trace the
toggle cannot reach: it keeps its pen and quietly stays a line while every
other trace switches, and nothing fails — it just looks wrong on one panel.

So: style.data_pen belongs to sample_marks.trace(), and this asserts it.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from smfs_catalog import sample_marks, style

_app = QApplication.instance() or QApplication([])

PKG = Path(__file__).resolve().parents[1] / "smfs_catalog"

# style.py defines it; sample_marks.py is the only caller, because it is what
# registers the trace for the toggle to find later.
EXEMPT = {"style.py", "sample_marks.py"}

DATA_PEN = re.compile(r"\bdata_pen\s*\(")


@pytest.fixture
def lines_then_restore():
    """Run with a known mode, and leave the process as it was found."""
    was = sample_marks.dots()
    yield
    sample_marks.set_dots(was)


@pytest.mark.parametrize(
    "path", sorted(p for p in PKG.glob("*.py") if p.name not in EXEMPT),
    ids=lambda p: p.name,
)
def test_no_window_builds_its_own_data_pen(path: Path):
    """Data traces are made by sample_marks.trace(), not pen by pen."""
    hits = [
        f"{path.name}:{i}  {line.strip()[:70]!r}"
        for i, line in enumerate(path.read_text().splitlines(), 1)
        if DATA_PEN.search(line)
    ]
    assert not hits, (
        "a data trace built here cannot be switched to dots — use "
        "sample_marks.trace(plot, color=…, width=…, alpha=…):\n  "
        + "\n  ".join(hits)
    )


def test_dots_is_the_default():
    """The held segments are the reason the app exists, and a line drawn
    through one asserts motion that never happened."""
    assert sample_marks._DEFAULT_DOTS is True


def test_data_marks_covers_both_forms():
    line = style.data_marks(dots=False)
    dots = style.data_marks(dots=True)

    assert line["pen"] is not None and line["symbol"] is None
    assert dots["pen"] is None and dots["symbol"] == "o"
    # No ring: 100k outlined dots read as a black band, and the outline is the
    # expensive half of drawing them.
    assert dots["symbolPen"] is None


def test_data_pen_carries_alpha():
    """The parameter display_roi.py needed and did not have."""
    opaque = style.data_pen(style.DATA).color().alpha()
    faded = style.data_pen(style.DATA, alpha=220).color().alpha()
    assert opaque == 255, "rule 1's default is still opaque"
    assert faded == 220


def _draws_a_line(item) -> bool:
    """pg.mkPen(None) is a QPen with NoPen, not None — so ask the pen, not the
    slot, whether anything joins the samples up."""
    return item.opts["pen"].style() != Qt.PenStyle.NoPen


def test_toggling_switches_a_live_trace(lines_then_restore):
    """The toggle reaches a trace that already exists — the whole mechanism."""
    plot = pg.PlotWidget()
    item = sample_marks.trace(plot, color=style.SIG_RETRACT, name="retract")

    sample_marks.set_dots(True)
    assert not _draws_a_line(item)
    assert item.opts["symbol"] == "o"

    sample_marks.set_dots(False)
    assert _draws_a_line(item)
    assert item.opts["symbol"] is None

    # …and back, because a toggle is not a one-way door.
    sample_marks.set_dots(True)
    assert not _draws_a_line(item)
    assert item.opts["symbol"] == "o"


def test_the_registry_does_not_hold_a_closed_window_open():
    """Weak, so stepping through curves cannot accumulate plots.  On this
    machine that matters more than the redraw cost."""
    import gc

    gc.collect()
    baseline = len(sample_marks._tracked)

    plot = pg.PlotWidget()
    sample_marks.trace(plot)
    assert len(sample_marks._tracked) == baseline + 1

    del plot
    gc.collect()
    assert len(sample_marks._tracked) == baseline


def test_a_closed_window_does_not_break_the_next_toggle(lines_then_restore):
    """Every toggle listens to one module-level signal for as long as it
    lives.  If the connection outlived the widget, the first switch after
    closing a window would reach a deleted C++ object and take the app down."""
    import gc

    from PyQt6.QtWidgets import QWidget

    from smfs_catalog.widgets import SampleMarksToggle

    host = QWidget()
    SampleMarksToggle(host)
    host.deleteLater()
    del host
    _app.processEvents()
    gc.collect()

    sample_marks.set_dots(False)     # would raise RuntimeError if it leaked
    sample_marks.set_dots(True)


def test_the_setting_survives_a_restart(tmp_path, lines_then_restore):
    """A catalog reopens in the mode it was left in."""
    from smfs_catalog import db as _db

    db_path = str(tmp_path / "catalog.db")
    _db.initialise(db_path)

    sample_marks.load(db_path)
    assert sample_marks.dots() is True, "a fresh catalog starts on the default"

    sample_marks.set_dots(False)
    assert _db.get_app_setting(_db.APP_SETTING_SAMPLE_MARKS, "", db_path) == "lines"

    # A new process would start on the default and adopt the stored value.
    sample_marks._dots = True
    assert sample_marks.load(db_path) is False
