# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
The invOLS window after the feature review: one test per finding.

  1. a unit on screen comes from `quantities`;
  2. names on screen come from `variables`;
  3. one "drawn" rule: the plot and the window agree on which rows count;
  4. the box readout counts only stale traces inside the box's time span;
  5. a new scope clears a drawn box;
  6. one double click writes one group; Apply waits for a change;
  7. an Apply error is emphasised and the window reloads;
  8. the stale-settings warning's grammar;
  9. the instrument value comes from the traces that take the value.

Same fixture as test_invols_window.py: four files, instrument 199.14 nm/V,
slopes -0.95, -1.00, -1.05 and none, an hour apart (10:00 to 13:00).  Expected
numbers are hand calculations: 199.14 / 0.95 = 209.62105, 199.14 / 1.00 =
199.14, 199.14 / 1.05 = 189.65714, mean of the three 199.47273.  Every catalog
is a temporary file passed explicitly.
"""
from __future__ import annotations

import dataclasses
import importlib
import math
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import invols_gui_synth as g                                  # noqa: E402
import invols_synth as synth                                  # noqa: E402
from PyQt6.QtWidgets import QLabel                            # noqa: E402
from smfs_catalog import db, invols_stats, quantities, style, variables  # noqa: E402
from smfs_catalog import invols_store as store                # noqa: E402
from smfs_catalog.scope import new_scope                      # noqa: E402

TRANSFORM_KEY = "transform_invols_nm_v"
SENTINEL = "SENTINEL"
BOX_FILES_1_2 = (g.hour(10) - 60, g.hour(11) + 60, 150.0, 250.0)
BOX_FILES_1_TO_3 = (g.hour(10) - 60, g.hour(12) + 60, 150.0, 250.0)
STALE_TAIL = " applied under different fit settings than those in force now."


def _iw():
    return importlib.import_module("smfs_catalog.invols_window")


@pytest.fixture(autouse=True)
def _pinned(monkeypatch):
    g.pin_code_version(monkeypatch)


@pytest.fixture(scope="module")
def qapp():
    return g.qapp()


@pytest.fixture
def make(qapp):
    """Factory: an InvolsWindow on a catalog, shown, closed at teardown."""
    made = []

    def _make(cat, **kwargs):
        win = _iw().InvolsWindow(cat.paths, cat.db, **kwargs)
        win.applied_counts = []
        win.applied.connect(lambda n: win.applied_counts.append(n))
        win.show()
        g.pump()
        made.append(win)
        return win

    yield _make
    for win in made:
        win.close()
        win.deleteLater()
    g.pump()


@pytest.fixture
def cat(tmp_path):
    return g.build(tmp_path, rms=[0.1948, 0.2113, 0.25, None])


def _groups(cat):
    return synth.fetch(cat.db, "SELECT id FROM invols_groups ORDER BY id")


def _stale(cat, index):
    """File `index`'s stored fit, re-stamped as made under other fit settings."""
    synth.put_result(cat.db, cat.ids[index], "invols_slope", g.SLOPES[index],
                     "old-key")


def _box_window(make, cat, box):
    win = make(cat)
    win.set_method("box")
    g.plot_of(win).commit_box(*box)
    g.pump()
    return win


def _sentinel_unit(monkeypatch):
    """quantities answers with a sentinel unit for the Transform InvOLS."""
    real = quantities.get

    def get(key):
        quantity = real(key)
        if key == TRANSFORM_KEY:
            return dataclasses.replace(quantity, display_unit=SENTINEL)
        return quantity

    monkeypatch.setattr(quantities, "get", get)


def _readout_input(**opts):
    o = {"method": "box", "average": "arithmetic", "n_cut": 2, "n_old": 0, **opts}
    summary = invols_stats.summarise([209.62105, 199.14, 189.65714])
    return _iw().ReadoutInput(summary=summary,
                              instrument_nm_v=(199.14, 199.14, 199.14),
                              method=o["method"], average=o["average"],
                              n_cut=o["n_cut"], n_old_settings=o["n_old"])


# ═════════════════════════════════════════════════════════════════════════════
# 1. A unit on screen comes from quantities
# ═════════════════════════════════════════════════════════════════════════════

def test_the_cut_line_takes_its_unit_from_quantities(monkeypatch):
    _sentinel_unit(monkeypatch)
    text = _iw().readout_text(_readout_input())
    (line,) = g.lines_with(text, "limits")
    assert f"outside its {SENTINEL} limits" in line
    assert "nm/V" not in line


def test_the_cut_line_still_reads_nm_per_volt_by_default():
    text = _iw().readout_text(_readout_input())
    (line,) = g.lines_with(text, "limits")
    assert "2 traces inside the box's time span lie outside its nm/V limits" in line


# ═════════════════════════════════════════════════════════════════════════════
# 2. Names on screen come from variables
# ═════════════════════════════════════════════════════════════════════════════

def _select_an_ensemble_trace(make, cat, index=0):
    win = make(cat)
    win.set_method("all")
    g.apply_button(win).click()
    g.pump()
    g.plot_of(win).point_selected.emit(index)
    g.pump()
    return win


def test_the_selection_panel_takes_the_in_use_name_from_variables(
        make, cat, monkeypatch):
    monkeypatch.setattr(variables, "IN_USE_INVOLS_LABEL", "In-use SENTINEL",
                        raising=False)
    win = _select_an_ensemble_trace(make, cat)
    shown = g.screen(win)
    assert "In-use SENTINEL: 199.47 nm/V" in shown
    assert "InvOLS in use" not in shown


def test_the_selection_panel_shows_the_default_in_use_wording(make, cat):
    shown = g.screen(_select_an_ensemble_trace(make, cat))
    assert "InvOLS in use: 199.47 nm/V" in shown


def test_the_selection_panel_takes_the_rms_name_from_the_registry(
        make, cat, monkeypatch):
    real = variables.label
    monkeypatch.setattr(
        variables, "label",
        lambda key: "RMS SENTINEL" if key == "invols_rms" else real(key))
    win = make(cat)
    g.plot_of(win).point_selected.emit(1)
    g.pump()
    shown = g.screen(win)
    assert "RMS SENTINEL: 0.2113 nm" in shown
    assert "Fit RMS" not in shown


def test_the_selection_panel_names_the_rms_as_the_registry_does(make, cat):
    win = make(cat)
    g.plot_of(win).point_selected.emit(1)
    g.pump()
    shown = g.screen(win)
    assert "InvOLS RMS: 0.2113 nm" in shown          # variables.py: "InvOLS RMS"
    assert "Fit RMS" not in shown


# ═════════════════════════════════════════════════════════════════════════════
# 3. One "drawn" rule: the plot and the window agree
# ═════════════════════════════════════════════════════════════════════════════

# (id, own value, acquisition time in seconds or None).  Only "normal" is drawn.
_T0 = 1_700_000_000.0
ROW_CASES = [
    ("normal", 200.0, _T0),
    ("no-value", None, _T0),
    ("inf-value", math.inf, _T0),
    ("nan-value", math.nan, _T0),
    ("no-time", 200.0, None),
    ("inf-time", 200.0, math.inf),
]


def _trace(i, own, time_s):
    return store.TraceRow(
        file_id=i + 1, path=f"/data/f{i + 1}.ibw", time_s=time_s,
        instrument_nm_v=199.14, own_nm_v=own, fit_current=True, rms_nm=0.2,
        source="own", group_id=None, in_use_nm_v=own)


def _window_on(rows, make, cat, monkeypatch):
    """A window whose catalog read returns exactly `rows`."""
    monkeypatch.setattr(store, "trace_rows", lambda paths, db_path: list(rows))
    win = make(cat)
    win.set_method("all")
    g.pump()
    return win


@pytest.mark.parametrize("case", ROW_CASES, ids=[c[0] for c in ROW_CASES])
def test_the_window_counts_a_row_exactly_when_the_plot_draws_it(
        make, cat, monkeypatch, case):
    name, own, time_s = case
    win = _window_on([_trace(0, own, time_s)], make, cat, monkeypatch)
    drawn = len(g.spots(g.plot_of(win)))
    assert drawn == (1 if name == "normal" else 0)
    assert win.readout().n == drawn


def test_the_window_and_the_plot_agree_on_a_mixed_list(make, cat, monkeypatch):
    rows = [_trace(0, 200.0, _T0)]
    rows += [_trace(i + 1, own, t) for i, (_n, own, t) in enumerate(ROW_CASES[1:])]
    rows.append(_trace(len(rows), 201.0, _T0 + 60.0))
    win = _window_on(rows, make, cat, monkeypatch)
    assert len(g.spots(g.plot_of(win))) == 2
    assert win.readout().n == 2
    assert win.readout().mean == pytest.approx(200.5, rel=1e-9)
    assert "n = 2" in g.screen(win)


@pytest.mark.parametrize("own", [math.inf, math.nan], ids=["inf", "nan"])
def test_a_non_finite_stale_row_is_not_counted_as_an_earlier_fit(
        make, cat, monkeypatch, own):
    stale = dataclasses.replace(_trace(1, own, _T0 + 60.0), fit_current=False)
    win = _window_on([_trace(0, 200.0, _T0), stale], make, cat, monkeypatch)
    assert win.readout().n == 1
    assert "earlier fit settings" not in g.screen(win)


def test_a_non_finite_row_is_still_a_target(make, cat, monkeypatch):
    rows = [_trace(0, 200.0, _T0), _trace(1, math.inf, _T0 + 60.0)]
    win = _window_on(rows, make, cat, monkeypatch)
    assert g.apply_button(win).text() == "Apply to all 2 traces in the window"


# ═════════════════════════════════════════════════════════════════════════════
# 4. The box readout counts only stale traces inside the box's time span
# ═════════════════════════════════════════════════════════════════════════════

def test_a_stale_trace_outside_the_box_is_not_in_the_readout(make, cat):
    _stale(cat, 2)                    # file 3, 12:00: outside a 10:00-11:00 box
    win = _box_window(make, cat, BOX_FILES_1_2)
    assert win.readout().n == 2
    assert "earlier fit settings" not in g.screen(win)


def test_only_the_stale_traces_inside_the_box_span_are_counted(make, cat):
    _stale(cat, 1)                    # file 2, 11:00: inside
    _stale(cat, 2)                    # file 3, 12:00: outside
    win = _box_window(make, cat, BOX_FILES_1_2)
    assert win.readout().n == 1
    assert ("1 trace has an own fit made under earlier fit settings "
            "(drawn hollow); it is left out until the queue visits it"
            ) in g.screen(win)


def test_under_all_every_stale_trace_is_counted(make, cat):
    _stale(cat, 1)
    _stale(cat, 2)
    win = make(cat)
    win.set_method("all")
    g.pump()
    assert "2 traces have an own fit made under earlier fit settings" in g.screen(win)


def test_an_empty_box_does_not_blame_stale_traces_outside_it(make, cat):
    _stale(cat, 2)                    # outside the box below
    win = _box_window(make, cat, (g.hour(10) - 60, g.hour(10) + 60, 300.0, 400.0))
    assert win.readout().n == 0
    tip = g.apply_button(win).toolTip()
    assert not g.apply_button(win).isEnabled()
    assert "fit settings" not in tip
    assert "No trace has an own fit to take a mean of." in tip


def test_an_empty_box_over_stale_traces_blames_the_fit_settings(make, cat):
    _stale(cat, 1)                    # file 2, 11:00: the only trace in the box
    win = _box_window(make, cat, (g.hour(11) - 60, g.hour(11) + 60, 150.0, 250.0))
    assert win.readout().n == 0
    assert not g.apply_button(win).isEnabled()
    assert ("No trace has an own fit made under the fit settings in force now"
            in g.apply_button(win).toolTip())


# ═════════════════════════════════════════════════════════════════════════════
# 5. apply_scope clears a drawn box
# ═════════════════════════════════════════════════════════════════════════════

def test_apply_scope_clears_a_drawn_box(make, tmp_path):
    cat = g.build(tmp_path, users=["A", "A", "B", "B"])
    win = _box_window(make, cat, BOX_FILES_1_2)
    plot = g.plot_of(win)
    assert plot.box() is not None and win.readout().n == 2
    win.apply_scope({**new_scope(), "users": ["A"]})
    g.pump()
    assert plot.box() is None
    assert win.readout() is None
    assert "draw a box" in g.screen(win).lower()
    assert not g.apply_button(win).isEnabled()


# ═════════════════════════════════════════════════════════════════════════════
# 6. One double click writes one group; Apply waits for a change
# ═════════════════════════════════════════════════════════════════════════════

def _applied_under_all(make, cat):
    win = make(cat)
    win.set_method("all")
    g.pump()
    return win


def test_two_apply_clicks_in_a_row_write_one_group(make, cat):
    win = _applied_under_all(make, cat)
    button = g.apply_button(win)
    button.click()
    button.click()
    g.pump()
    assert len(_groups(cat)) == 1
    assert win.applied_counts == [4]


def test_apply_is_disabled_after_a_successful_ensemble_apply(make, cat):
    win = _applied_under_all(make, cat)
    assert g.apply_button(win).isEnabled()
    g.apply_button(win).click()
    g.pump()
    assert not g.apply_button(win).isEnabled()
    assert g.apply_button(win).toolTip()          # and it says something


def _change_method(win, cat):
    win.set_method("own")


def _change_average(win, cat):
    win.set_average("harmonic")


def _change_scope(win, cat):
    win.apply_scope(new_scope())


def _change_settings(win, cat):
    db.update_analysis_param("invols_window_pts", 150, cat.db)
    synth.stamp_fit_key(cat.db, cat.ids)          # the queue re-ran every trace
    win.refresh_params()


def _reload(win, cat):
    win.refresh_params()


CHANGES = [_change_method, _change_average, _change_scope, _change_settings,
           _reload]


@pytest.mark.parametrize("change", CHANGES, ids=[c.__name__[1:] for c in CHANGES])
def test_apply_is_enabled_again_after_something_changes(make, cat, change):
    win = _applied_under_all(make, cat)
    g.apply_button(win).click()
    g.pump()
    assert not g.apply_button(win).isEnabled()
    change(win, cat)
    g.pump()
    assert g.apply_button(win).isEnabled()


def test_apply_is_enabled_again_after_a_new_box(make, cat):
    win = _box_window(make, cat, BOX_FILES_1_2)
    g.apply_button(win).click()
    g.pump()
    assert not g.apply_button(win).isEnabled()
    g.plot_of(win).commit_box(*BOX_FILES_1_TO_3)
    g.pump()
    assert g.apply_button(win).isEnabled()


def test_a_second_apply_after_a_change_writes_a_second_group(make, cat):
    win = _applied_under_all(make, cat)
    g.apply_button(win).click()
    g.pump()
    win.set_average("harmonic")
    g.apply_button(win).click()
    g.pump()
    assert len(_groups(cat)) == 2


# ═════════════════════════════════════════════════════════════════════════════
# 7. An Apply error is emphasised and the window reloads
# ═════════════════════════════════════════════════════════════════════════════

ERROR_TEXT = "no trace with a current own fit can give a value"


def _label_showing(win, needle):
    found = [w for w in win.findChildren(QLabel) if needle in w.text()]
    assert len(found) == 1, f"expected one label showing {needle!r}, got {len(found)}"
    return found[0]


def _apply_that_fails(make, cat):
    """Another window changes the fit window and the queue has not re-run the
    traces; this window is told nothing, so Apply finds no current fit."""
    win = _applied_under_all(make, cat)
    db.update_analysis_param("invols_window_pts", 150, cat.db)
    g.apply_button(win).click()
    g.pump()
    return win


def test_an_apply_error_is_shown_with_emphasis(make, cat):
    win = _apply_that_fails(make, cat)
    label = _label_showing(win, ERROR_TEXT)
    assert style.qss_emphasis() in label.styleSheet()
    assert _groups(cat) == [] and win.applied_counts == []


def test_an_apply_error_reloads_the_window_from_the_catalog(make, cat):
    win = _apply_that_fails(make, cat)
    assert win.readout().n == 0                    # every fit is now an earlier one
    assert not any(g.is_filled(p) for p in g.points(g.plot_of(win)))
    assert "earlier fit settings" in g.screen(win)
    assert not g.apply_button(win).isEnabled()
    assert ERROR_TEXT in g.screen(win)             # the reload keeps the message


# ═════════════════════════════════════════════════════════════════════════════
# 8. The stale-settings warning's grammar
# ═════════════════════════════════════════════════════════════════════════════

def test_one_stale_group_reads_in_the_singular(make, cat):
    synth.add_group(cat.db, cat.ids[:2], 200.0)
    shown = g.screen(make(cat))
    assert "Warning: group #1 was" + STALE_TAIL in shown


def test_two_stale_groups_read_in_the_plural(make, cat):
    synth.add_group(cat.db, cat.ids[:2], 200.0)
    synth.add_group(cat.db, cat.ids[2:3], 200.0)
    shown = g.screen(make(cat))
    assert "Warning: groups #1, #2 were" + STALE_TAIL in shown
    assert "group #1, group #2" not in shown


# ═════════════════════════════════════════════════════════════════════════════
# 9. The instrument value comes from the traces that take the value
# ═════════════════════════════════════════════════════════════════════════════

def test_with_no_member_the_readout_still_gives_the_instrument_value(make, cat):
    db.update_analysis_param("invols_window_pts", 150, cat.db)   # every fit stale
    win = make(cat)
    win.set_method("all")
    g.pump()
    shown = g.screen(win)
    assert "n = 0" in shown
    assert "instrument value 199.14 nm/V" in shown


def test_the_instrument_range_covers_the_targets_not_the_members(make, tmp_path):
    # File 4 has no slope, so it is no member, but it takes the value.
    cat = g.build(tmp_path, instruments=[199.14, 199.14, 199.14, 205.5])
    win = make(cat)
    win.set_method("all")
    g.pump()
    assert win.readout().n == 3
    assert "instrument value 199.14 – 205.50 nm/V" in g.screen(win)


def test_the_instrument_range_under_box_covers_every_target(make, tmp_path):
    cat = g.build(tmp_path, instruments=[199.14, 199.14, 205.5, 199.14])
    win = _box_window(make, cat, BOX_FILES_1_2)
    assert win.readout().n == 2
    assert "instrument value 199.14 – 205.50 nm/V" in g.screen(win)


def test_a_trace_without_an_instrument_value_is_not_in_the_range(make, tmp_path):
    cat = g.build(tmp_path, instruments=[199.14, 199.14, 199.14, None])
    win = make(cat)
    win.set_method("all")
    g.pump()
    assert "instrument value 199.14 nm/V" in g.screen(win)
