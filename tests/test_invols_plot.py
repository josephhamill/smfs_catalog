# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.


"""
The invOLS plot, drawing: one point per trace, time on x and nm/V on y; the
colours, hollow points, pinned outlier markers, the display range, the mean
line and the tooltips.  Builders and readers are in invols_plot_synth.py;
clicks, drags and the box are in test_invols_plot_interaction.py.
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

from smfs_catalog import quantities, style, variables               # noqa: E402
from smfs_catalog.invols_store import TraceRow                      # noqa: E402
from smfs_catalog.qt_utils import _DateAxis                         # noqa: E402
from invols_plot_synth import (  # noqa: E402
    HOUR,
    T0,
    _bulk_with,
    _horizontal_lines,
    _is_filled,
    _markers,
    _pen_hex,
    _points,
    _row,
    _rows,
    _settle,
    _spots,
    _tip,
    _tips,
    _view_y,
)
from invols_plot_synth import plot, qapp  # noqa: E402,F401  (fixtures)


# ── points, colours, hollow ──────────────────────────────────────────────────

def test_three_rows_with_own_values_give_three_points(plot):
    plot.set_traces(_rows([209.62105, 199.14, 189.65714]))
    _settle(plot)
    assert len(_points(plot)) == 3
    assert not _markers(plot)
    xs = sorted(pt.pos().x() for _, pt in _points(plot))
    assert xs == pytest.approx([T0, T0 + HOUR, T0 + 2 * HOUR], abs=1.0)
    ys = sorted(pt.pos().y() for _, pt in _points(plot))
    assert ys == pytest.approx([189.65714, 199.14, 209.62105])


def test_rows_with_no_own_value_add_no_point(plot):
    plot.set_traces(_rows([209.0, None, 199.0, None]))
    _settle(plot)
    assert len(_spots(plot)) == 2


def test_axes_are_time_and_nm_per_v(plot):
    assert isinstance(plot.getAxis("bottom"), _DateAxis)
    left = plot.getAxis("left")
    label = f"{left.labelText} {left.labelUnits}"
    assert quantities.unit_of("transform_invols_nm_v") in label


def test_own_and_ensemble_use_two_different_named_style_colours(plot):
    plot.set_traces([_row(0, 200.0, source="own"),
                     _row(1, 201.0, source="ensemble"),
                     _row(2, 199.0, source="own")])
    _settle(plot)
    by_x = {round(pt.pos().x()): _pen_hex(pt) for _, pt in _points(plot)}
    own, ens = by_x[round(T0)], by_x[round(T0 + HOUR)]
    assert by_x[round(T0 + 2 * HOUR)] == own
    assert own != ens
    palette = {c.lower() for c in (*style.SERIES_LINE, *style.SERIES_LABELED)}
    palette |= {c.lower() for c in (style.INK, style.INK_STRONG, style.INK_MUTED,
                                    style.INK_FAINT, style.DATA, style.GRID)}
    for colour in (own, ens):
        named = [n for n, v in vars(style).items()
                 if isinstance(v, str) and v.lower() == colour]
        assert named, f"{colour} is not a named style.py token"
        assert colour in palette, f"{colour} is not from the existing palette"


def test_plot_module_names_no_colour_literal():
    import smfs_catalog.invols_plot as mod
    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert not re.search(r"#[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?\b", src)
    assert not re.search(r"QColor\(\s*\d", src)


def test_fit_current_false_is_hollow_and_true_is_filled(plot):
    plot.set_traces([_row(0, 200.0, current=True),
                     _row(1, 201.0, current=False)])
    _settle(plot)
    spots = {round(pt.pos().x()): (sc, pt) for sc, pt in _points(plot)}
    filled = spots[round(T0)][1]
    hollow = spots[round(T0 + HOUR)][1]
    assert _is_filled(filled)
    assert not _is_filled(hollow)
    assert hollow.pen().color().alpha() > 0      # an outline is still drawn


def test_hollow_tooltip_says_earlier_fit_settings_and_filled_does_not(plot):
    plot.set_traces([_row(0, 200.0, current=True),
                     _row(1, 201.0, current=False)])
    _settle(plot)
    tips = {round(pt.pos().x()): _tip(sc, pt) for sc, pt in _points(plot)}
    assert "earlier fit settings" in tips[round(T0 + HOUR)]
    assert "earlier fit settings" not in tips[round(T0)]


# ── outliers pinned to the edge ──────────────────────────────────────────────


def test_high_outlier_is_one_pinned_marker_with_its_real_number(plot):
    plot.set_traces(_rows(_bulk_with(9957.0)))
    _settle(plot)
    lo, hi = _view_y(plot)
    assert hi < 300
    assert lo <= 199.0 and hi >= 201.0
    markers = _markers(plot)
    assert len(markers) == 1
    sc, marker = markers[0]
    assert lo <= marker.pos().y() <= hi
    assert marker.pos().y() >= hi - 0.1 * (hi - lo)       # at the top edge
    assert "9957" in _tip(sc, marker)
    assert len(_spots(plot)) == 30                         # one spot per trace


def test_low_outlier_is_pinned_to_the_bottom_edge(plot):
    plot.set_traces(_rows(_bulk_with(12.5)))
    _settle(plot)
    lo, hi = _view_y(plot)
    markers = _markers(plot)
    assert len(markers) == 1
    sc, marker = markers[0]
    assert lo <= marker.pos().y() <= lo + 0.1 * (hi - lo)
    assert "12.5" in _tip(sc, marker)


def test_no_marker_when_nothing_is_an_outlier(plot):
    plot.set_traces(_rows([float(v) for v in np.linspace(199, 201, 30)]))
    _settle(plot)
    assert not _markers(plot)


# ── guards on the display range ──────────────────────────────────────────────

def _assert_finite_view(plot):
    xr, yr = plot.getViewBox().viewRange()
    for lim in (*xr, *yr):
        assert math.isfinite(lim)
    assert xr[0] < xr[1] and yr[0] < yr[1]


def test_no_rows_gives_a_default_view_without_error(plot):
    plot.set_traces([])
    _settle(plot)
    assert _spots(plot) == []
    _assert_finite_view(plot)


def test_rows_all_without_own_value_give_a_finite_view(plot):
    plot.set_traces(_rows([None, None, None]))
    _settle(plot)
    assert _spots(plot) == []
    _assert_finite_view(plot)


def test_a_single_row_gives_a_finite_view_containing_it(plot):
    plot.set_traces(_rows([199.14]))
    _settle(plot)
    _assert_finite_view(plot)
    lo, hi = _view_y(plot)
    assert lo < 199.14 < hi
    assert len(_points(plot)) == 1


def test_non_finite_own_values_are_not_drawn_and_do_not_break_the_view(plot):
    plot.set_traces(_rows([200.0, float("nan"), 201.0, float("inf"), 199.0]))
    _settle(plot)
    _assert_finite_view(plot)
    for _, pt in _spots(plot):
        assert math.isfinite(pt.pos().x()) and math.isfinite(pt.pos().y())
    assert len(_points(plot)) == 3


def test_row_with_no_time_raises_nothing_and_adds_no_non_finite_point(plot):
    # The plan does not say whether a trace without an acquisition time is
    # drawn.  Only the safe part is asserted: no error, no point at a
    # non-finite position, the view stays finite, the timed rows are drawn.
    rows = [_row(0, 200.0), _row(1, 201.0, time_s=None), _row(2, 199.0)]
    plot.set_traces(rows)
    _settle(plot)
    for _, pt in _spots(plot):
        assert math.isfinite(pt.pos().x()) and math.isfinite(pt.pos().y())
    xs = {round(pt.pos().x()) for _, pt in _points(plot)}
    assert {round(T0), round(T0 + 2 * HOUR)} <= xs
    _assert_finite_view(plot)


# ── a second set_traces leaves nothing behind ────────────────────────────────

def test_second_set_traces_leaves_no_leftovers(plot):
    first = _rows(_bulk_with(9957.0))
    first[3] = _row(3, first[3].own_nm_v, current=False)
    plot.set_traces(first)
    _settle(plot)
    assert _markers(plot) and len(_spots(plot)) == 30

    plot.set_traces(_rows([150.0, 151.0]))
    _settle(plot)
    assert len(_spots(plot)) == 2
    assert not _markers(plot)
    tips = " ".join(_tips(plot))
    assert "9957" not in tips
    assert "earlier fit settings" not in tips
    lo, hi = _view_y(plot)
    assert lo < 150.0 and hi > 151.0 and hi < 300


def test_set_traces_with_no_rows_after_rows_clears_everything(plot):
    plot.set_traces(_rows(_bulk_with(9957.0)))
    plot.set_traces([])
    _settle(plot)
    assert _spots(plot) == []


# ── the mean line ────────────────────────────────────────────────────────────

def test_mean_line_shows_for_a_value_and_goes_for_none(plot):
    plot.set_traces(_rows([199.0, 200.0, 201.0]))
    assert _horizontal_lines(plot, 200.0) == []
    plot.set_mean_line(200.0)
    assert len(_horizontal_lines(plot, 200.0)) == 1
    plot.set_mean_line(None)
    assert _horizontal_lines(plot, 200.0) == []


def test_mean_line_moves_to_a_new_value_and_never_doubles(plot):
    plot.set_traces(_rows([199.0, 200.0, 201.0]))
    plot.set_mean_line(200.0)
    plot.set_mean_line(200.5)
    assert _horizontal_lines(plot, 200.0) == []
    assert len(_horizontal_lines(plot, 200.5)) == 1


def test_mean_line_survives_a_new_set_traces_without_duplicating(plot):
    plot.set_mean_line(200.0)
    plot.set_traces(_rows([199.0, 200.0, 201.0]))
    assert len(_horizontal_lines(plot, 200.0)) == 1


# ── C1: the points are the own-fit value, and are named so ───────────────────

def _left_label(plot) -> str:
    left = plot.getAxis("left")
    return f"{left.labelText} {left.labelUnits}"


def test_own_invols_label_is_a_named_constant():
    from smfs_catalog import variables
    assert variables.OWN_INVOLS_LABEL == "Own-fit InvOLS"


def test_left_axis_is_named_own_fit_invols_with_its_unit(plot):
    label = _left_label(plot)
    assert "Own-fit InvOLS" in label
    assert "nm/V" in label
    assert "Transform InvOLS" not in label


def test_own_row_tooltip_gives_the_own_fit_value(plot):
    plot.set_traces([_row(0, 209.62105)])
    _settle(plot)
    (sc, pt), = _spots(plot)
    assert "Own-fit InvOLS: 209.62 nm/V" in _tip(sc, pt)


def test_ensemble_row_tooltip_gives_both_values(plot):
    plot.set_traces([_row(0, 209.62105, source="ensemble", in_use=199.47273)])
    _settle(plot)
    (sc, pt), = _spots(plot)
    tip = _tip(sc, pt)
    assert "Own-fit InvOLS: 209.62 nm/V" in tip
    assert "InvOLS in use: 199.47 nm/V" in tip


def test_in_use_label_is_a_named_constant():
    assert variables.IN_USE_INVOLS_LABEL == "InvOLS in use"


def test_ensemble_tooltip_takes_the_in_use_name_from_variables(plot, monkeypatch):
    monkeypatch.setattr(variables, "IN_USE_INVOLS_LABEL", "In-use SENTINEL",
                        raising=False)
    plot.set_traces([_row(0, 209.62105, source="ensemble", in_use=199.47273)])
    _settle(plot)
    (sc, pt), = _spots(plot)
    tip = _tip(sc, pt)
    assert "In-use SENTINEL: 199.47 nm/V" in tip
    assert "InvOLS in use" not in tip


def test_ensemble_row_without_in_use_value_invents_no_number(plot):
    plot.set_traces([_row(0, 209.62105, source="ensemble", in_use=None)])
    _settle(plot)
    (sc, pt), = _spots(plot)
    tip = _tip(sc, pt)
    assert "Own-fit InvOLS: 209.62 nm/V" in tip
    assert not re.search(r"in use:?\s*[-\d]", tip, re.I)
    assert "nan" not in tip.lower() and "none" not in tip.lower()


# ── I5: auto-range follows the bulk ──────────────────────────────────────────

def test_auto_range_stays_near_the_bulk_with_a_mean_line_and_an_outlier(plot):
    plot.set_traces(_rows(_bulk_with(9957.0)))
    plot.set_mean_line(525.23)
    plot.autoRange()
    _settle(plot)
    lo, hi = _view_y(plot)
    assert hi < 300
    assert lo <= 199.0 and hi >= 201.0


# ── F1: auto-range follows the bulk on y but keeps the outlier's time in x ───

MINUTE = 60.0
TWO_DAYS = 2 * 86400.0


def _session_with_outlier(after: bool):
    """29 traces a minute apart and a 9957 outlier two days after (or before)."""
    values = [float(v) for v in np.linspace(199, 201, 29)]
    rows = [_row(i, v, time_s=T0 + i * MINUTE) for i, v in enumerate(values)]
    t_out = T0 + 28 * MINUTE + TWO_DAYS if after else T0 - TWO_DAYS
    rows.append(_row(29, 9957.0, time_s=t_out))
    return rows, t_out


def _assert_outlier_visible_and_bulk_framed(plot, t_out):
    (xlo, xhi), (ylo, yhi) = plot.getViewBox().viewRange()
    assert xlo <= t_out <= xhi
    assert yhi < 300 and ylo <= 199.0 and yhi >= 201.0
    markers = _markers(plot)
    assert len(markers) == 1
    _, marker = markers[0]
    assert math.isclose(marker.pos().x(), t_out, abs_tol=1.0)
    assert xlo <= marker.pos().x() <= xhi
    assert ylo <= marker.pos().y() <= yhi


@pytest.mark.parametrize("after", [True, False], ids=["later", "earlier"])
def test_auto_range_keeps_a_distant_in_time_outlier_in_view(plot, after):
    rows, t_out = _session_with_outlier(after)
    plot.set_traces(rows)
    plot.autoRange()
    _settle(plot)
    _assert_outlier_visible_and_bulk_framed(plot, t_out)


@pytest.mark.parametrize("after", [True, False], ids=["later", "earlier"])
def test_enable_auto_range_keeps_a_distant_in_time_outlier_in_view(plot, after):
    rows, t_out = _session_with_outlier(after)
    plot.set_traces(rows)
    plot.enableAutoRange()
    QTest.qWait(100)
    _assert_outlier_visible_and_bulk_framed(plot, t_out)


def test_mean_line_does_not_pull_the_auto_range(plot):
    rows, t_out = _session_with_outlier(True)
    plot.set_traces(rows)
    plot.set_mean_line(525.0)
    plot.autoRange()
    _settle(plot)
    _assert_outlier_visible_and_bulk_framed(plot, t_out)
