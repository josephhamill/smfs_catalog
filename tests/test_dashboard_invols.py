# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
The dashboard side of the invOLS window (Task 9): the button, the single
reused window, the "InvOLS source" column, the refresh after an apply, the
queue-table export, and the two-way link with the curve view's fit controls.

Fixture: four files (instrument 199.14 nm/V, slopes -0.95, -1.00, -1.05, -1.00),
the first three belonging to experimentalist Alpha and the fourth to Beta, all
four queued and "up to date" (a stored event row under today's parameters and
the pinned code identity).  The scope is Alpha, so the invOLS window holds
files 1-3; file 4 is a queued bystander whose status must not change.

Every catalog is a temporary file passed explicitly.  Message boxes are
recorded, never shown; the dashboard's close prompt is answered "yes".
"""
from __future__ import annotations

import csv
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
from PyQt6.QtWidgets import QApplication, QMessageBox, QPushButton  # noqa: E402
from smfs_catalog import curve_analysis, db                   # noqa: E402
from smfs_catalog import dashboard_window as dash_mod         # noqa: E402
from smfs_catalog.scope import new_scope                      # noqa: E402

BUTTON_TEXT = "InvOLS for these files…"
SOURCE_RE = re.compile(r"^ensemble of 3 traces, \d{4}-\d{2}-\d{2}$")
USERS = ["Alpha", "Alpha", "Alpha", "Beta"]


@pytest.fixture(autouse=True)
def _pinned(monkeypatch):
    g.pin_code_version(monkeypatch)


@pytest.fixture(scope="module")
def qapp():
    return g.qapp()


class Env:
    """A running dashboard on the fixture catalog, plus what it was asked."""


def invols_windows():
    mod = sys.modules.get("smfs_catalog.invols_window")
    if mod is None:                 # nothing can have opened one yet
        return []
    return [w for w in QApplication.topLevelWidgets()
            if isinstance(w, mod.InvolsWindow) and w.isVisible()]


@pytest.fixture
def env(qapp, tmp_path, monkeypatch):
    e = Env()
    e.cat = g.build(tmp_path, slopes=(-0.95, -1.0, -1.05, -1.0), users=USERS,
                    rms=[0.1948, 0.2113, 0.25, 0.2])
    key = curve_analysis.pipeline_params_from(
        db.load_analysis_params(e.cat.db)).all_params
    for file_id in e.cat.ids:
        synth.put_result(e.cat.db, file_id, "event", 1.0, key)
    e.tmp = tmp_path
    e.info, e.warnings, e.errors = [], [], []
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: e.info.append(a)))
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: e.warnings.append(a)))
    monkeypatch.setattr(dash_mod.DashboardWindow, "_confirm_queue_saved",
                        lambda self: True)
    # An exception inside a Qt slot would otherwise abort the process.
    previous_hook = sys.excepthook
    sys.excepthook = lambda *exc: e.errors.append(exc)

    e.dash = dash_mod.DashboardWindow(e.cat.db)
    e.dash.show()
    db.enqueue_files(e.cat.ids, e.cat.db)
    e.dash._worker.invalidate_queue_cache()
    e.dash._refresh_queue_table()
    e.dash._scope = {**new_scope(), "users": ["Alpha"]}
    g.pump()
    yield e
    sys.excepthook = previous_hook
    for win in invols_windows():
        win.close()
    e.dash.close()
    g.pump()
    assert e.errors == [], e.errors


def click_invols_button(e):
    buttons = [b for b in e.dash.findChildren(QPushButton)
               if b.text() == BUTTON_TEXT]
    assert len(buttons) == 1, "the dashboard has no single 'InvOLS for these files…' button"
    buttons[0].click()
    g.pump()


def open_window(e):
    click_invols_button(e)
    wins = invols_windows()
    assert len(wins) == 1
    return wins[0]


def cell(e, file_id, col):
    row = e.dash._queue_id_to_row[file_id]
    return e.dash._queue_table.item(row, col).text()


def headers(e):
    t = e.dash._queue_table
    return [t.horizontalHeaderItem(i).text() for i in range(t.columnCount())]


def apply_all(e, win):
    win.set_method("all")
    g.apply_button(win).click()
    g.pump(5)


# ── the button and the window ────────────────────────────────────────────────

def test_the_button_exists_with_the_exact_text(env):
    assert BUTTON_TEXT in [b.text() for b in env.dash.findChildren(QPushButton)]


def test_an_empty_scope_gives_an_information_box_and_no_window(env):
    env.dash._scope = {**new_scope(), "search": "no-such-file-anywhere"}
    click_invols_button(env)
    assert len(env.info) == 1
    assert invols_windows() == []


def test_with_files_in_scope_one_window_opens_on_the_scopes_paths(env):
    win = open_window(env)
    assert env.info == []
    assert g.apply_button(win).text() == "Apply to all 3 traces in the window"
    assert len(g.points(g.plot_of(win))) == 3


def test_a_second_click_reuses_the_open_window(env):
    first = open_window(env)
    click_invols_button(env)
    second = invols_windows()
    assert second == [first]


def test_after_closing_the_window_a_click_opens_one_again(env):
    open_window(env).close()
    g.pump()
    assert invols_windows() == []
    click_invols_button(env)
    assert len(invols_windows()) == 1


def test_a_scope_edit_reaches_the_open_window(env):
    win = open_window(env)
    env.dash._scope = {**new_scope(), "users": ["Alpha"]}
    env.dash._search.setText("Image0001")
    env.dash._on_scope_edit()
    g.pump()
    assert re.fullmatch(r"Apply to all 1 traces? in the window",
                        g.apply_button(win).text())


def test_opening_the_window_changes_nothing_in_the_catalog(env):
    before = g.dump(env.cat.db)
    win = open_window(env)
    win.close()
    g.pump()
    assert g.dump(env.cat.db) == before


# ── the InvOLS source column ─────────────────────────────────────────────────

def test_the_fixed_columns_keep_their_places_and_the_new_one_is_fifth(env):
    assert headers(env)[:5] == ["Filename", "Status", "Event", "Hit",
                                "InvOLS source"]
    header_item = env.dash._queue_table.horizontalHeaderItem(4)
    assert header_item.toolTip().strip()


def test_before_any_apply_every_source_cell_reads_own_fit_and_rows_are_fresh(env):
    for file_id in env.cat.ids:
        assert cell(env, file_id, 4) == "own fit"
        assert cell(env, file_id, 1).startswith("up to date")


def test_after_an_apply_the_cells_read_the_ensemble_and_only_those_rows_are_stale(env):
    win = open_window(env)
    apply_all(env, win)
    for file_id in env.cat.ids[:3]:
        assert SOURCE_RE.match(cell(env, file_id, 4)), cell(env, file_id, 4)
        assert cell(env, file_id, 1).startswith("stale params")
    bystander = env.cat.ids[3]
    assert cell(env, bystander, 4) == "own fit"
    assert cell(env, bystander, 1).startswith("up to date")


def test_the_apply_adds_the_transform_invols_column(env):
    win = open_window(env)
    apply_all(env, win)
    assert "Transform InvOLS (nm/V)" in headers(env)


def test_reverting_to_own_fits_returns_the_cells_to_own_fit(env):
    win = open_window(env)
    apply_all(env, win)
    win.set_method("own")
    g.apply_button(win).click()
    g.pump(5)
    for file_id in env.cat.ids:
        assert cell(env, file_id, 4) == "own fit"


# ── the export ───────────────────────────────────────────────────────────────

def _find_key(node, key):
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            found = _find_key(value, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_key(value, key)
            if found is not None:
                return found
    return None


def test_the_queue_table_export_has_the_source_column_right_after_hit(env):
    win = open_window(env)
    apply_all(env, win)
    before = set(os.listdir(env.tmp))
    env.dash._on_export_queue_table()
    g.pump()
    new = sorted(set(os.listdir(env.tmp)) - before)
    csv_files = [n for n in new if n.endswith(".csv")]
    json_files = [n for n in new if n.endswith(".json")]
    assert len(csv_files) == 1 and json_files
    with open(env.tmp / csv_files[0], newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    assert header[:7] == ["path", "filename", "status", "freshness", "event",
                          "hit", "invols_source"]
    by_path = {r[0]: r[6] for r in rows[1:]}
    assert SOURCE_RE.match(by_path[env.cat.paths[0]])
    assert by_path[env.cat.paths[3]] == "own fit"
    keys = None
    for name in json_files:
        keys = _find_key(json.loads((env.tmp / name).read_text("utf-8")),
                         "column_keys")
        if keys:
            break
    assert keys[:7] == ["path", "filename", "status", "freshness", "event",
                        "hit", "invols_source"]


def _export_rows(e):
    """Run the queue-table export; return (header, {path: row dict}, keys)."""
    before = set(os.listdir(e.tmp))
    e.dash._on_export_queue_table()
    g.pump()
    new = sorted(set(os.listdir(e.tmp)) - before)
    csv_name = [n for n in new if n.endswith(".csv")][0]
    with open(e.tmp / csv_name, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    keys = None
    for name in [n for n in new if n.endswith(".json")]:
        keys = _find_key(json.loads((e.tmp / name).read_text("utf-8")),
                         "column_keys")
        if keys:
            break
    return header, {r[0]: dict(zip(header, r)) for r in rows[1:]}, keys


WORDS = {"fresh": "up to date", "stale": "stale params", "new": "not analysed"}


def _live_freshness(e):
    from smfs_catalog.curve_analysis import current_signature
    params_json, code_ver = current_signature(e.cat.db)
    return db.queue_freshness(params_json, code_ver, e.cat.db)


def test_the_export_freshness_equals_the_db_answer_and_the_status_cell(env):
    win = open_window(env)
    apply_all(env, win)
    header, by_path, keys = _export_rows(env)
    assert keys[3] == "freshness"
    live = _live_freshness(env)
    assert set(live.values()) == {"fresh", "stale"}      # the check can tell
    for file_id, path in zip(env.cat.ids, env.cat.paths):
        word = by_path[path]["freshness"]
        assert word == WORDS[live[file_id]]
        assert cell(env, file_id, 1).startswith(word)


def test_after_an_apply_assigned_files_export_stale_and_the_bystander_up_to_date(env):
    win = open_window(env)
    apply_all(env, win)
    _header, by_path, _keys = _export_rows(env)
    for path in env.cat.paths[:3]:
        assert by_path[path]["freshness"] == "stale params"
        assert SOURCE_RE.match(by_path[path]["invols_source"])
    assert by_path[env.cat.paths[3]]["freshness"] == "up to date"
    assert by_path[env.cat.paths[3]]["invols_source"] == "own fit"


def test_before_any_apply_every_exported_row_is_up_to_date(env):
    _header, by_path, _keys = _export_rows(env)
    assert [by_path[p]["freshness"] for p in env.cat.paths] == ["up to date"] * 4


def test_a_file_never_analysed_exports_not_analysed(env):
    conn = db.get_connection(env.cat.db)
    try:
        conn.execute("DELETE FROM analysis_results WHERE file_id = ? "
                     "AND analysis_type = 'event'", (env.cat.ids[3],))
        conn.commit()
    finally:
        conn.close()
    _header, by_path, _keys = _export_rows(env)
    assert by_path[env.cat.paths[3]]["freshness"] == "not analysed"
    assert by_path[env.cat.paths[0]]["freshness"] == "up to date"


def test_the_queue_state_column_is_unchanged_beside_the_freshness(env):
    _header, by_path, _keys = _export_rows(env)
    queued = {r["path"]: r["status"] or "" for r in db.list_queue(env.cat.db)}
    for path in env.cat.paths:
        assert by_path[path]["status"] == queued[path]


# ── Open this trace ──────────────────────────────────────────────────────────

def test_open_this_trace_calls_the_dashboards_reveal_with_the_path(
        env, monkeypatch):
    opened = []
    monkeypatch.setattr(dash_mod.DashboardWindow, "reveal_decomp_at",
                        lambda self, path: opened.append(path))
    win = open_window(env)
    g.plot_of(win).point_selected.emit(1)
    g.pump()
    btn = g.open_button(win)
    tip = btn.toolTip().lower()
    assert "queue" in tip and "paus" in tip
    btn.click()
    assert opened == [env.cat.paths[1]]


def test_open_this_trace_does_not_change_the_parameter_owner_of_a_busy_queue(
        env):
    # The queue already holds Beta's file 4 first; the opened trace is Alpha's.
    db.dequeue_files(env.cat.ids[:3], env.cat.db)
    env.dash._worker.invalidate_queue_cache()
    env.dash._refresh_queue_table()
    assert db.active_param_owner(env.cat.db) == "Beta"
    win = open_window(env)
    assert db.active_param_owner(env.cat.db) == "Beta"
    g.plot_of(win).point_selected.emit(1)
    g.pump()
    g.open_button(win).click()
    g.pump(5)
    assert db.active_param_owner(env.cat.db) == "Beta"
    assert env.cat.paths[1] in db.queue_paths(env.cat.db)
    assert env.dash._worker.is_paused()


# ── the link with the curve view's fit controls ─────────────────────────────

def _open_curve_view(e, with_decomp=True):
    e.dash._on_open_viewer()
    viewer = e.dash._viewer
    if with_decomp:
        viewer._toggle_decomp(True)
        assert viewer._decomp_win is not None
    g.pump()
    return viewer


class _Count:
    def __init__(self, monkeypatch, cls, name):
        self.n = 0
        original = getattr(cls, name)

        def wrapper(obj, *args, **kwargs):
            self.n += 1
            return original(obj, *args, **kwargs)
        monkeypatch.setattr(cls, name, wrapper)


def _classes():
    from smfs_catalog.decomposition_window import DecompositionWindow
    return DecompositionWindow, importlib.import_module(
        "smfs_catalog.invols_window").InvolsWindow


def test_an_edit_in_the_window_refreshes_the_curve_view_once(env, monkeypatch):
    decomp_cls, _ = _classes()
    seen = _Count(monkeypatch, decomp_cls, "refresh_invols_fit")
    viewer = _open_curve_view(env)
    win = open_window(env)
    g.controls_of(win).window_spinbox.setValue(130)
    g.pump()
    assert seen.n == 1
    assert viewer._decomp_win._invols_window_spinbox.value() == 130


def test_an_edit_in_the_curve_view_refreshes_the_window_once_without_an_echo(
        env, monkeypatch):
    decomp_cls, invols_cls = _classes()
    to_window = _Count(monkeypatch, invols_cls, "refresh_params")
    to_view = _Count(monkeypatch, decomp_cls, "refresh_invols_fit")
    viewer = _open_curve_view(env)
    win = open_window(env)
    viewer._decomp_win._invols_window_spinbox.setValue(140)
    g.pump()
    assert to_window.n == 1
    assert to_view.n == 0
    assert g.controls_of(win).window_pts() == 140


def test_an_edit_in_the_window_does_not_loop_between_the_two_sides(
        env, monkeypatch):
    decomp_cls, invols_cls = _classes()
    to_window = _Count(monkeypatch, invols_cls, "refresh_params")
    to_view = _Count(monkeypatch, decomp_cls, "refresh_invols_fit")
    _open_curve_view(env)
    win = open_window(env)
    g.controls_of(win).offset_spinbox.setValue(60)
    g.pump(5)
    assert to_view.n == 1
    assert to_window.n <= 1


def test_the_window_works_when_there_is_no_curve_view_at_all(env):
    win = open_window(env)
    g.controls_of(win).window_spinbox.setValue(130)
    g.pump()
    assert db.load_analysis_params(env.cat.db).invols_window_pts == 130


def test_the_window_works_when_the_curve_view_has_no_fit_window_yet(env):
    viewer = _open_curve_view(env, with_decomp=False)
    assert viewer._decomp_win is None
    win = open_window(env)
    g.controls_of(win).window_spinbox.setValue(130)
    g.pump()
    assert db.load_analysis_params(env.cat.db).invols_window_pts == 130
    assert viewer._decomp_win is None


def test_a_fit_window_built_after_the_invols_window_is_linked_too(
        env, monkeypatch):
    decomp_cls, invols_cls = _classes()
    to_window = _Count(monkeypatch, invols_cls, "refresh_params")
    to_view = _Count(monkeypatch, decomp_cls, "refresh_invols_fit")
    viewer = _open_curve_view(env, with_decomp=False)
    win = open_window(env)
    viewer._toggle_decomp(True)
    g.pump()
    to_window.n = 0
    to_view.n = 0
    g.controls_of(win).window_spinbox.setValue(150)
    g.pump()
    assert to_view.n == 1
    viewer._decomp_win._invols_window_spinbox.setValue(160)
    g.pump()
    assert to_window.n == 1
    assert g.controls_of(win).window_pts() == 160
