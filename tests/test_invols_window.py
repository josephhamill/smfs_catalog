# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
The invOLS window (Task 9) and its pure readout text.

Fixture (the plan's): four files, instrument 199.14 nm/V each, slopes -0.95,
-1.00, -1.05 and none for file 4, acquired an hour apart.  Expected numbers are
hand calculations (199.14 / 0.95 = 209.62105..., the three-value mean
199.47273, sd 9.98611, se 5.7655), never numbers read back from the code.

The window is read the way a user reads it: button texts, label texts, the
plot's drawn spots.  The tests name only the plan's public interface
(InvolsWindow, applied, apply_scope, refresh_params, method, set_method,
average, set_average, readout, ReadoutInput, readout_text) plus the widgets'
types (InvolsPlot, InvolsFitControls, QPushButton), found by type or by their
button text.  Every catalog is a temporary file passed explicitly.
"""
from __future__ import annotations

import importlib
import json
import os
import re
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import invols_gui_synth as g                                  # noqa: E402
import invols_synth as synth                                  # noqa: E402
from smfs_catalog import db, invols_stats, style, variables  # noqa: E402
from smfs_catalog import invols_store as store                # noqa: E402
from smfs_catalog.scope import new_scope                      # noqa: E402

TRANSFORM = "transform_invols_nm_v"


def _iw():
    """The module under test, imported when a test runs so that each test
    fails on its own while the module does not exist."""
    return importlib.import_module("smfs_catalog.invols_window")


@pytest.fixture(autouse=True)
def _pinned(monkeypatch):
    g.pin_code_version(monkeypatch)


@pytest.fixture(scope="module")
def qapp():
    return g.qapp()


class _Calls:
    def __init__(self):
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)


@pytest.fixture
def make(qapp):
    """Factory: an InvolsWindow on a catalog, shown, closed at teardown."""
    made = []

    def _make(cat, **kwargs):
        spy = _Calls()
        kwargs.setdefault("open_trace", spy)
        win = _iw().InvolsWindow(cat.paths, cat.db, **kwargs)
        win.opened = spy
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


def _group_rows(cat):
    return synth.fetch(cat.db, "SELECT method, n, average, transform_nm_v, "
                       "box_json, fit_settings_json FROM invols_groups "
                       "ORDER BY id")


def _info(summary=None, instrument=(199.14, 199.14, 199.14), **opts):
    """A ReadoutInput; `opts` may give method, average, n_cut and n_old."""
    o = {"method": "all", "average": "arithmetic", "n_cut": 0, "n_old": 0,
         **opts}
    assert set(o) == {"method", "average", "n_cut", "n_old"}, f"unknown {opts}"
    summary = (invols_stats.summarise([209.62105, 199.14, 189.65714])
               if summary is None else summary)
    return _iw().ReadoutInput(summary=summary, instrument_nm_v=tuple(instrument),
                              method=o["method"], average=o["average"],
                              n_cut=o["n_cut"], n_old_settings=o["n_old"])


def _unitless_decimals(text: str) -> list[str]:
    """Decimal numbers in `text` not followed by a unit (a range such as
    '199.14 – 205.50 nm/V' gives its unit to both ends)."""
    bad = []
    for m in re.finditer(r"\d+\.\d+", text):
        tail = text[m.end():m.end() + 16]
        if re.match(r"\s*(nm/V|nm|V\b)", tail):
            continue
        if re.match(r"\s*[–-]\s*\d+\.\d+\s*nm/V", tail):
            continue
        bad.append(text[max(0, m.start() - 20):m.end() + 12])
    return bad


# ═════════════════════════════════════════════════════════════════════════════
# readout_text: a pure function
# ═════════════════════════════════════════════════════════════════════════════

def test_readout_text_all_shows_the_plans_numbers_and_labels():
    text = _iw().readout_text(_info())
    for needle in ("instrument value 199.14 nm/V", "199.47", "199.14", "9.99",
                   "5.77",
                   "standard deviation of the per-trace values",
                   "standard error of the mean of the per-trace values "
                   "(scatter only"):
        assert needle in text
    assert re.search(r"\bn\s*=\s*3\b", text)
    assert "median" not in text.lower()


def test_readout_text_lines_carry_their_units():
    text = _iw().readout_text(_info())
    assert _unitless_decimals(text) == []
    for label in ("arithmetic mean", "harmonic mean",
                  "standard deviation of the per-trace values",
                  "standard error of the mean"):
        lines = g.lines_with(text, label)
        assert lines, label
        assert all("nm/V" in ln for ln in lines), lines


@pytest.mark.parametrize("average, applied, other", [
    ("arithmetic", "arithmetic mean", "harmonic mean"),
    ("harmonic", "harmonic mean", "arithmetic mean"),
])
def test_readout_text_marks_only_the_average_that_will_be_applied(
        average, applied, other):
    text = _iw().readout_text(_info(average=average))
    applied_lines = g.lines_with(text, applied)
    other_lines = g.lines_with(text, other)
    assert len(applied_lines) == 1 and "applied" in applied_lines[0]
    assert len(other_lines) == 1 and "applied" not in other_lines[0]


def test_readout_text_harmonic_line_holds_the_harmonic_value():
    text = _iw().readout_text(_info(average="harmonic"))
    line = g.lines_with(text, "harmonic mean")[0]
    assert "199.14 nm/V" in line


def test_readout_text_shows_a_range_when_the_files_differ():
    text = _iw().readout_text(_info(instrument=(199.14, 205.5, 199.14)))
    assert "instrument value" in text
    assert "199.14" in text and "205.50" in text


def test_readout_text_for_an_empty_result_formats_no_missing_number():
    empty = invols_stats.summarise([])
    text = _iw().readout_text(_info(summary=empty, instrument=(), n_old=3))
    assert isinstance(text, str) and text.strip()
    assert "None" not in text and "nan" not in text.lower()
    assert re.search(r"\bn\s*=\s*0\b", text)


def test_readout_text_for_an_empty_result_still_gives_the_old_settings_line():
    empty = invols_stats.summarise([])
    text = _iw().readout_text(_info(summary=empty, n_old=3))
    assert ("3 traces have an own fit made under earlier fit settings "
            "(drawn hollow); they are left out until the queue visits them"
            ) in text


def test_readout_text_for_one_trace_shows_sd_and_se_as_no_value():
    one = invols_stats.summarise([199.14])
    text = _iw().readout_text(_info(summary=one, instrument=(199.14,)))
    assert "None" not in text and "nan" not in text.lower()
    assert re.search(r"\bn\s*=\s*1\b", text)
    for label in ("standard deviation", "standard error"):
        lines = g.lines_with(text, label)
        assert len(lines) == 1, label
        assert "—" in lines[0]
        assert not re.search(r"\d", lines[0]), lines[0]
    assert "199.14 nm/V" in g.lines_with(text, "arithmetic mean")[0]


@pytest.mark.parametrize("n_cut, lead", [(1, "1 trace"), (3, "3 traces")])
def test_readout_text_box_names_the_cut_traces(n_cut, lead):
    text = _iw().readout_text(_info(method="box", n_cut=n_cut))
    assert lead in text
    for needle in ("inside the box's time span", "outside its nm/V limits",
                   "the standard deviation and standard error describe only "
                   "the traces inside the limits"):
        assert needle in text


def test_readout_text_has_no_cut_line_without_cut_traces_or_for_all():
    box = _iw().readout_text(_info(method="box", n_cut=0))
    every = _iw().readout_text(_info(method="all", n_cut=2))
    for text in (box, every):
        assert "outside its nm/V limits" not in text


def test_readout_text_old_settings_line_for_several_traces_is_verbatim():
    text = _iw().readout_text(_info(n_old=3))
    assert ("3 traces have an own fit made under earlier fit settings "
            "(drawn hollow); they are left out until the queue visits them"
            ) in text


def test_readout_text_old_settings_line_for_one_trace_keeps_its_meaning():
    text = _iw().readout_text(_info(n_old=1))
    for needle in ("1 trace", "earlier fit settings", "drawn hollow",
                   "left out until the queue visits"):
        assert needle in text


def test_readout_text_has_no_old_settings_line_when_none_are_old():
    assert "earlier fit settings" not in _iw().readout_text(_info(n_old=0))


def test_readout_text_uses_no_banned_word_and_no_median():
    for kwargs in ({}, {"method": "box", "n_cut": 2}, {"n_old": 2},
                   {"average": "harmonic"}):
        text = _iw().readout_text(_info(**kwargs)).lower()
        assert not [w for w in g.BANNED if w in text]
        assert "median" not in text


# ═════════════════════════════════════════════════════════════════════════════
# Opening, the three methods, the readout
# ═════════════════════════════════════════════════════════════════════════════

def test_window_opens_with_three_points_and_names_the_four_traces(make, cat):
    win = make(cat)
    assert len(g.points(g.plot_of(win))) == 3
    assert re.search(r"\b4\b", win.windowTitle())
    assert g.apply_button(win).text() == "Apply to all 4 traces in the window"


def test_the_apply_note_is_shown_verbatim(make, cat):
    win = make(cat)
    assert ("Forces and extensions update the next time the queue visits each "
            "trace. Own-fit values are kept.") in g.screen(win)


def test_each_choice_says_when_to_use_it(make, cat):
    win = make(cat)
    shown = g.screen(win)
    for needle in ("Each trace's own fit", "Mean of all traces shown",
                   "Mean of the boxed traces",
                   "deep and clean on every trace",
                   "one cantilever and one laser alignment",
                   "poor or missing contact region"):
        assert needle in shown


def test_the_default_method_is_own_with_apply_disabled_and_a_reason(make, cat):
    win = make(cat)
    assert win.method() == "own"
    assert win.readout() is None
    apply_btn = g.apply_button(win)
    assert not apply_btn.isEnabled()
    assert "ensemble" in apply_btn.toolTip().lower()


def test_method_all_gives_the_three_trace_readout(make, cat):
    win = make(cat)
    win.set_method("all")
    g.pump()
    assert win.method() == "all"
    summary = win.readout()
    assert summary.n == 3
    assert summary.mean == pytest.approx(g.MEAN_3, rel=1e-6)
    assert summary.sd == pytest.approx(9.98611, rel=1e-5)
    assert summary.se == pytest.approx(5.7655, rel=1e-4)
    shown = g.screen(win)
    for needle in ("instrument value 199.14 nm/V",
                   "standard error of the mean of the per-trace values "
                   "(scatter only", "199.47", "199.14", "9.99", "5.77"):
        assert needle in shown
    assert "median" not in g.everything(win).lower()
    assert g.apply_button(win).isEnabled()


def test_default_average_is_arithmetic_and_the_text_marks_it_applied(make, cat):
    win = make(cat)
    win.set_method("all")
    g.pump()
    assert win.average() == "arithmetic"
    shown = g.screen(win)
    arithmetic = g.lines_with(shown, "arithmetic mean")
    harmonic = g.lines_with(shown, "harmonic mean")
    assert any("applied" in ln for ln in arithmetic)
    assert not any("applied" in ln for ln in harmonic)


def test_the_average_selector_explains_itself(make, cat):
    win = make(cat)
    assert "Average" in g.screen(win)
    tips = g.everything(win)
    assert "square of the relative scatter" in tips
    assert "harmonic mean" in tips and "less biased" in tips


def test_harmonic_average_is_marked_and_applied(make, cat):
    win = make(cat)
    win.set_method("all")
    win.set_average("harmonic")
    g.pump()
    assert win.average() == "harmonic"
    shown = g.screen(win)
    assert any("applied" in ln for ln in g.lines_with(shown, "harmonic mean"))
    assert not any("applied" in ln
                   for ln in g.lines_with(shown, "arithmetic mean"))
    assert "199.14" in g.lines_with(shown, "harmonic mean")[0]
    g.apply_button(win).click()
    g.pump()
    (method, n, average, value, _box, _fit), = _group_rows(cat)
    assert (method, n, average) == ("all", 3, "harmonic")
    assert value == pytest.approx(199.14, rel=1e-12)


def test_the_chosen_average_is_remembered_by_the_next_window(make, cat):
    first = make(cat)
    first.set_average("harmonic")
    assert store.get_average(cat.db) == "harmonic"
    assert make(cat).average() == "harmonic"


def test_an_unknown_average_is_refused_and_changes_nothing(make, cat):
    win = make(cat)
    with pytest.raises(ValueError):
        win.set_average("median")
    assert win.average() == "arithmetic"
    assert store.get_average(cat.db) == "arithmetic"


def test_box_method_with_no_box_disables_apply_and_asks_for_a_box(make, cat):
    win = make(cat)
    win.set_method("box")
    g.pump()
    assert win.method() == "box"
    assert not g.apply_button(win).isEnabled()
    assert "draw a box" in g.screen(win).lower()
    assert "box" in g.apply_button(win).toolTip().lower()


def test_box_around_two_traces_reads_only_those_two(make, cat):
    win = make(cat)
    win.set_method("box")
    g.plot_of(win).commit_box(g.hour(10) - 60, g.hour(11) + 60, 150.0, 250.0)
    g.pump()
    summary = win.readout()
    assert summary.n == 2
    assert summary.mean == pytest.approx(204.380525, rel=1e-6)
    assert g.apply_button(win).isEnabled()
    assert "outside its nm/V limits" not in g.screen(win)


def test_box_whose_limits_cut_a_trace_in_its_span_says_so(make, cat):
    win = make(cat)
    win.set_method("box")
    # Trace 3 (189.66 nm/V) is inside the time span but below 195 nm/V.
    g.plot_of(win).commit_box(g.hour(10) - 60, g.hour(12) + 60, 195.0, 215.0)
    g.pump()
    assert win.readout().n == 2
    shown = g.screen(win)
    assert "1 trace" in shown
    assert "outside its nm/V limits" in shown
    assert "describe only the traces inside the limits" in shown


def test_apply_under_box_stores_the_box_and_the_members_it_holds(make, cat):
    win = make(cat)
    win.set_method("box")
    g.plot_of(win).commit_box(g.hour(10) - 60, g.hour(11) + 60, 150.0, 250.0)
    g.apply_button(win).click()
    g.pump()
    (method, n, average, value, box_json, _fit), = _group_rows(cat)
    assert (method, n, average) == ("box", 2, "arithmetic")
    assert value == pytest.approx(204.380525, rel=1e-6)
    assert len(json.loads(box_json)) == 4
    members = synth.fetch(cat.db, "SELECT file_id FROM invols_group_members "
                          "ORDER BY file_id")
    assert members == [(cat.ids[0],), (cat.ids[1],)]
    # Every trace in the window still receives the value, boxed or not.
    assert len(synth.fetch(cat.db, "SELECT * FROM invols_assignments")) == 4


def test_box_membership_uses_real_values_not_a_pinned_outlier_marker(
        make, tmp_path):
    slopes = (-0.98, -0.99, -1.0, -1.0, -1.01, -1.02, -1.0, -0.02)   # last: 9957 nm/V
    cat = g.build(tmp_path, slopes=slopes)
    win = make(cat)
    plot = g.plot_of(win)
    assert [p for p in g.spots(plot) if p.symbol() in g.TRIANGLES], \
        "the 9957 nm/V trace should be a pinned marker"
    win.set_method("box")
    # The pinned marker sits near the top of the view (about 205 nm/V), inside
    # this box; its real value, 9957 nm/V, is outside the limits.
    plot.commit_box(g.hour(9), g.hour(20), 150.0, 250.0)
    g.pump()
    assert win.readout().n == 7
    shown = g.screen(win)
    assert "1 trace" in shown and "outside its nm/V limits" in shown


def test_apply_under_all_writes_one_group_and_every_trace_is_an_ensemble_trace(
        make, cat):
    win = make(cat)
    win.set_method("all")
    before = {r.file_id: r.source for r in store.trace_rows(cat.paths, cat.db)}
    assert set(before.values()) == {"own"}
    g.apply_button(win).click()
    g.pump()
    (method, n, average, value, box_json, _fit), = _group_rows(cat)
    assert (method, n, average, box_json) == ("all", 3, "arithmetic", None)
    assert value == pytest.approx(g.MEAN_3, rel=1e-6)
    assert len(synth.fetch(cat.db, "SELECT * FROM invols_assignments")) == 4
    assert win.applied_counts == [4]
    rows = store.trace_rows(cat.paths, cat.db)
    assert [r.source for r in rows] == ["ensemble"] * 4


def test_points_change_colour_when_the_traces_move_to_the_ensemble(make, cat):
    win = make(cat)
    win.set_method("all")
    plot = g.plot_of(win)
    own_colours = {g.pen_hex(p) for p in g.points(plot)}
    g.apply_button(win).click()
    g.pump()
    ensemble_colours = {g.pen_hex(p) for p in g.points(plot)}
    assert len(own_colours) == 1 and len(ensemble_colours) == 1
    assert own_colours != ensemble_colours
    assert own_colours == {style.INVOLS_OWN.lower()}
    assert ensemble_colours == {style.INVOLS_ENSEMBLE.lower()}


def test_after_apply_the_window_reports_the_apply_summary_not_its_readout(
        make, cat):
    win = make(cat)
    win.set_method("all")
    assert win.readout().mean == pytest.approx(g.MEAN_3, rel=1e-6)
    # The queue re-runs file 3 between the readout and Apply: slope -1.5.
    synth.put_result(cat.db, cat.ids[2], "invols_slope", -1.5,
                     synth.current_fit_key(cat.db))
    g.apply_button(win).click()
    g.pump()
    (_m, _n, _a, value, _b, _f), = _group_rows(cat)
    expected = (199.14 / 0.95 + 199.14 + 199.14 / 1.5) / 3
    assert value == pytest.approx(expected, rel=1e-9)
    assert win.readout().n == 3
    assert win.readout().mean == pytest.approx(value, rel=1e-12)


def test_stored_fit_settings_are_the_ones_in_force_not_the_windows_memory(
        make, cat):
    win = make(cat)
    win.set_method("all")
    # Another window changes the window length and the queue re-runs all three
    # traces under it; this window is told nothing.
    db.update_analysis_param("invols_window_pts", 150, cat.db)
    synth.stamp_fit_key(cat.db, cat.ids)
    g.apply_button(win).click()
    g.pump()
    (_m, n, _a, _v, _b, fit_json), = _group_rows(cat)
    assert n == 3
    assert json.loads(fit_json)["invols_window_pts"] == 150


def test_opening_and_closing_without_applying_leaves_the_catalog_unchanged(
        make, cat):
    before = g.dump(cat.db)
    win = make(cat)
    win.set_method("all")
    g.plot_of(win).point_selected.emit(1)
    g.pump()
    win.close()
    g.pump()
    assert g.dump(cat.db) == before


def test_own_method_reverts_the_windows_traces_to_their_own_fits(make, cat):
    win = make(cat)
    win.set_method("all")
    g.apply_button(win).click()
    g.pump()
    group_rows = _group_rows(cat)
    win.set_method("own")
    g.pump()
    assert g.apply_button(win).isEnabled()
    g.apply_button(win).click()
    g.pump()
    assert synth.fetch(cat.db, "SELECT * FROM invols_assignments") == []
    assert _group_rows(cat) == group_rows
    assert win.applied_counts == [4, 4]
    assert [r.source for r in store.trace_rows(cat.paths, cat.db)] == ["own"] * 4
    values = dict(synth.fetch(cat.db, "SELECT file_id, value FROM "
                              "analysis_results WHERE analysis_type = ?",
                              (TRANSFORM,)))
    assert values[cat.ids[0]] == pytest.approx(209.62105, rel=1e-6)


# ═════════════════════════════════════════════════════════════════════════════
# Files without an instrument value, files without an acquisition time
# ═════════════════════════════════════════════════════════════════════════════

def test_a_file_without_an_instrument_value_is_named_and_not_counted(
        make, tmp_path):
    cat = g.build(tmp_path, slopes=(-0.95, -1.0, -1.05, -1.0),
                  instruments=[199.14, 199.14, 199.14, None])
    win = make(cat)
    shown = g.screen(win)
    assert "1 trace" in shown
    assert "no instrument InvOLS in" in shown
    assert "cannot take an ensemble value" in shown
    assert g.apply_button(win).text() == "Apply to all 3 traces in the window"


def test_no_instrument_line_is_absent_when_every_file_has_one(make, cat):
    assert "no instrument InvOLS" not in g.screen(make(cat))


def _no_time_catalog(tmp_path, count):
    slopes = (-0.95, -1.0, -1.05, None, -1.2, -0.9)[:4 + count]
    hours = [10, 11, 12, 13] + [None] * count
    return g.build(tmp_path, slopes=slopes, hours=hours)


def test_one_trace_without_a_time_is_not_drawn_but_is_a_target(make, tmp_path):
    cat = _no_time_catalog(tmp_path, 1)
    win = make(cat)
    assert len(g.points(g.plot_of(win))) == 3
    assert g.apply_button(win).text() == "Apply to all 5 traces in the window"
    win.set_method("all")
    g.pump()
    assert win.readout().n == 3
    assert win.readout().mean == pytest.approx(g.MEAN_3, rel=1e-6)
    assert ("1 trace has no acquisition time: it is not drawn and not part of "
            "the mean, but it receives the applied value.") in g.screen(win)
    g.apply_button(win).click()
    g.pump()
    members = [r[0] for r in synth.fetch(
        cat.db, "SELECT file_id FROM invols_group_members ORDER BY file_id")]
    assert members == cat.ids[:3]
    assigned = {r[0] for r in synth.fetch(
        cat.db, "SELECT file_id FROM invols_assignments")}
    assert assigned == set(cat.ids)
    assert win.applied_counts == [5]
    values = dict(synth.fetch(cat.db, "SELECT file_id, value FROM "
                              "analysis_results WHERE analysis_type = ?",
                              (TRANSFORM,)))
    assert values[cat.ids[4]] == pytest.approx(g.MEAN_3, rel=1e-6)


def test_several_traces_without_a_time_are_counted_in_one_exact_line(
        make, tmp_path):
    cat = _no_time_catalog(tmp_path, 2)
    win = make(cat)
    win.set_method("all")
    g.pump()
    assert win.readout().n == 3
    assert ("2 traces have no acquisition time: they are not drawn and not "
            "part of the mean, but they receive the applied value."
            ) in g.screen(win)


def test_a_trace_without_a_time_is_never_a_box_member(make, tmp_path):
    cat = _no_time_catalog(tmp_path, 1)       # its own value is 165.95 nm/V
    win = make(cat)
    win.set_method("box")
    g.plot_of(win).commit_box(g.hour(9), g.hour(14), 100.0, 250.0)
    g.pump()
    assert win.readout().n == 3
    assert "no acquisition time" in g.screen(win)


def test_no_time_line_is_absent_when_every_trace_has_a_time(make, cat):
    win = make(cat)
    win.set_method("all")
    g.pump()
    assert "acquisition time" not in g.screen(win)


# ═════════════════════════════════════════════════════════════════════════════
# Fit settings: hollow points, the readout drops them, the warning
# ═════════════════════════════════════════════════════════════════════════════

def test_changed_fit_settings_turn_points_hollow_and_empty_the_readout(
        make, cat):
    win = make(cat)
    win.set_method("all")
    plot = g.plot_of(win)
    assert all(g.is_filled(p) for p in g.points(plot))
    db.update_analysis_param("invols_window_pts", 150, cat.db)
    win.refresh_params()
    g.pump()
    assert len(g.points(plot)) == 3
    assert not any(g.is_filled(p) for p in g.points(plot))
    assert win.readout().n == 0
    shown = g.screen(win)
    assert "3 traces have an own fit made under earlier fit settings" in shown
    apply_btn = g.apply_button(win)
    assert not apply_btn.isEnabled()
    assert "fit settings" in apply_btn.toolTip().lower()
    win.set_method("box")
    plot.commit_box(g.hour(9), g.hour(14), 100.0, 300.0)
    g.pump()
    assert win.readout().n == 0
    assert not apply_btn.isEnabled()


def test_after_one_trace_is_rerun_the_readout_has_it_alone(make, cat):
    win = make(cat)
    win.set_method("all")
    db.update_analysis_param("invols_window_pts", 150, cat.db)
    win.refresh_params()
    synth.stamp_fit_key(cat.db, [cat.ids[0]])
    win.refresh_params()
    g.pump()
    summary = win.readout()
    assert summary.n == 1
    assert summary.mean == pytest.approx(209.62105, rel=1e-6)
    shown = g.screen(win)
    assert "2 traces have an own fit made under earlier fit settings" in shown
    standard_deviation = g.lines_with(shown, "standard deviation")[0]
    assert "—" in standard_deviation and not re.search(r"\d", standard_deviation)
    assert sum(1 for p in g.points(g.plot_of(win)) if g.is_filled(p)) == 1


def test_editing_the_fit_controls_in_the_window_turns_stale_traces_hollow(
        make, cat):
    win = make(cat)
    win.set_method("all")
    controls = g.controls_of(win)
    controls.window_spinbox.setValue(150)
    g.pump()
    assert not any(g.is_filled(p) for p in g.points(g.plot_of(win)))
    assert win.readout().n == 0
    assert "earlier fit settings" in g.screen(win)


def test_refresh_params_pulls_the_fit_controls_from_the_catalog_silently(
        make, cat):
    win = make(cat)
    controls = g.controls_of(win)
    seen = _Calls()
    controls.changed.connect(lambda: seen())
    db.update_analysis_param("invols_window_pts", 120, cat.db)
    win.refresh_params()
    g.pump()
    assert controls.window_pts() == 120
    assert seen.calls == []


def test_a_group_applied_under_other_fit_settings_raises_a_warning(make, cat):
    win = make(cat)
    win.set_method("all")
    g.apply_button(win).click()
    g.pump()
    assert "different fit settings" not in g.screen(win).lower()
    db.update_analysis_param("invols_window_pts", 150, cat.db)
    win.refresh_params()
    g.pump()
    shown = g.screen(win).lower()
    assert "different fit settings" in shown
    assert re.search(r"group\s*#?\s*1\b", shown)


def test_no_warning_before_any_group_exists(make, cat):
    win = make(cat)
    db.update_analysis_param("invols_window_pts", 150, cat.db)
    win.refresh_params()
    g.pump()
    assert "different fit settings" not in g.screen(win).lower()


# ═════════════════════════════════════════════════════════════════════════════
# Selecting a point, Open this trace
# ═════════════════════════════════════════════════════════════════════════════

def test_with_nothing_selected_open_is_disabled_and_says_why(make, cat):
    btn = g.open_button(make(cat))
    assert not btn.isEnabled()
    assert "select" in btn.toolTip().lower()


def test_selecting_a_point_shows_its_facts_and_opens_the_trace(make, cat):
    win = make(cat)
    g.plot_of(win).point_selected.emit(1)
    g.pump()
    shown = g.screen(win)
    for needle in ("Image0002.ibw", "199.14 nm/V", "0.2113 nm", "own fit"):
        assert needle in shown
    btn = g.open_button(win)
    assert btn.isEnabled()
    btn.click()
    assert win.opened.calls == [(cat.paths[1],)]


def test_a_selected_ensemble_trace_shows_its_source(make, cat):
    win = make(cat)
    win.set_method("all")
    g.apply_button(win).click()
    g.pump()
    g.plot_of(win).point_selected.emit(0)
    g.pump()
    assert "ensemble" in g.screen(win).lower()
    assert "0.1948 nm" in g.screen(win)


def test_double_click_on_a_point_opens_the_trace(make, cat):
    win = make(cat)
    g.plot_of(win).point_opened.emit(2)
    assert win.opened.calls == [(cat.paths[2],)]


def test_a_real_click_selects_the_right_trace_when_an_earlier_one_is_undrawn(
        make, tmp_path):
    # File 1 has no fit and is not drawn; the drawn spots are files 2, 3, 4.
    cat = g.build(tmp_path, slopes=(None, -0.95, -1.0, -1.05))
    win = make(cat)
    plot = g.plot_of(win)
    g.click_point(plot, g.spot_for_path(plot, cat.paths[2]))
    g.pump()
    btn = g.open_button(win)
    assert btn.isEnabled()
    btn.click()
    assert win.opened.calls == [(cat.paths[2],)]


def test_open_trace_is_optional_and_the_button_does_nothing_harmful(
        qapp, cat):
    win = _iw().InvolsWindow(cat.paths, cat.db)
    try:
        g.plot_of(win).point_selected.emit(0)
        g.pump()
        g.open_button(win).click()          # must not raise
    finally:
        win.close()


# ═════════════════════════════════════════════════════════════════════════════
# apply_scope, the parameter owner, wording
# ═════════════════════════════════════════════════════════════════════════════

def test_apply_scope_replaces_the_traces(make, tmp_path):
    cat = g.build(tmp_path, users=["A", "A", "B", "B"])
    win = make(cat)
    assert g.apply_button(win).text() == "Apply to all 4 traces in the window"
    win.apply_scope({**new_scope(), "users": ["A"]})
    g.pump()
    assert g.apply_button(win).text() == "Apply to all 2 traces in the window"
    assert re.search(r"\b2\b", win.windowTitle())
    assert len(g.points(g.plot_of(win))) == 2


def test_apply_scope_keeps_the_readout_in_step(make, tmp_path):
    cat = g.build(tmp_path, users=["A", "A", "B", "B"])
    win = make(cat)
    win.set_method("all")
    win.apply_scope({**new_scope(), "users": ["A"]})
    g.pump()
    assert win.readout().n == 2
    assert win.readout().mean == pytest.approx(204.380525, rel=1e-6)


def test_the_window_names_whose_parameter_set_is_in_force(make, tmp_path):
    cat = g.build(tmp_path, users=["Alpha", "Beta", "Beta", "Beta"])
    db.enqueue_files([cat.ids[1]], cat.db)
    owner = db.active_param_owner(cat.db)
    assert owner == "Beta"
    assert f"Parameters: {owner}'s set" in g.screen(make(cat))


def test_opening_the_window_does_not_change_the_parameter_owner(
        make, tmp_path):
    cat = g.build(tmp_path, users=["Alpha", "Beta", "Beta", "Beta"])
    db.enqueue_files([cat.ids[1]], cat.db)
    before = db.active_param_owner(cat.db)
    win = make(cat)
    win.set_method("all")
    g.plot_of(win).point_selected.emit(0)
    g.pump()
    assert db.active_param_owner(cat.db) == before == "Beta"


def test_no_label_uses_a_banned_word_in_any_state(make, cat):
    win = make(cat)
    states = []
    states.append(g.everything(win))
    win.set_method("all")
    states.append(g.everything(win))
    win.set_average("harmonic")
    states.append(g.everything(win))
    win.set_method("box")
    g.plot_of(win).commit_box(g.hour(10) - 60, g.hour(12) + 60, 195.0, 215.0)
    g.plot_of(win).point_selected.emit(1)
    states.append(g.everything(win))
    win.set_method("all")
    g.apply_button(win).click()
    db.update_analysis_param("invols_window_pts", 150, cat.db)
    win.refresh_params()
    states.append(g.everything(win))
    for text in states:
        low = text.lower()
        assert not [w for w in g.BANNED if w in low]
        assert "median" not in low


def test_every_decimal_on_screen_carries_a_unit(make, cat):
    win = make(cat)
    win.set_method("all")
    g.plot_of(win).point_selected.emit(0)
    g.pump()
    labels = []
    from PyQt6.QtWidgets import QLabel
    labels += [lb.text() for lb in win.findChildren(QLabel)]
    for text in (g.screen(win), *labels):
        assert _unitless_decimals(g._plain(text)) == []


def test_the_window_calls_the_plotted_values_own_fit_invols(make, cat):
    win = make(cat)
    win.set_method("all")
    g.apply_button(win).click()
    g.pump()
    g.plot_of(win).point_selected.emit(1)
    g.pump()
    assert "Transform InvOLS" not in g.everything(win)
    axis_text = g.plot_of(win).getAxis("left").labelText
    assert variables.OWN_INVOLS_LABEL == "Own-fit InvOLS"
    assert variables.OWN_INVOLS_LABEL in axis_text
    assert "Transform InvOLS" not in axis_text
    assert "nm/V" in axis_text
