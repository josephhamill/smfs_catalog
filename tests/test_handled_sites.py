# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Guard test: the window handlers that used to swallow errors now show them.

WHY THIS EXISTS.  PR (b) / D1: a broad `except Exception: return` in a window
made a failure look like an empty state ("No curve", "nothing stale", "no
existing values").  Each site now binds the exception and calls
`report_handled(<the window's own setter>, <context>, exc)`, which logs the
traceback and shows "<context> failed: <Name>: <msg>", then carries on as
before.  Windows are not constructed in most tests (see
test_names_are_imported.py), so the first half of this file reads the source:
for each (module, function, which try) it finds the handler and checks it.
The second half drives a few of the handlers on a stand-in `self` (no
QApplication) by forcing the failing dependency to raise.

Context wording is NOT asserted (the plan does not fix it), except that the
dashboard's two child-window handlers name the child's title.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from smfs_catalog import crashlog
from smfs_catalog import dashboard_window as dash
from smfs_catalog import decomposition_window as dw
from smfs_catalog import display_roi as roi
from smfs_catalog import event_summary_window as esw

PKG = Path(__file__).resolve().parents[1] / "smfs_catalog"


# ── AST helpers ──────────────────────────────────────────────────────────────

def _function(module: str, name: str) -> ast.FunctionDef:
    tree = ast.parse((PKG / f"{module}.py").read_text(encoding="utf-8"))
    found = [n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(found) == 1, f"{module}.{name}: expected one definition, got {len(found)}"
    return found[0]


def _handler(module: str, func: str, marker: str) -> ast.ExceptHandler:
    """The broad handler of the `try` in `func` whose body mentions `marker`."""
    fn = _function(module, func)
    hits = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Try):
            continue
        body_src = "\n".join(ast.unparse(s) for s in node.body)
        if marker in body_src:
            hits += node.handlers
    assert len(hits) == 1, f"{module}.{func}: {len(hits)} handlers for try containing {marker!r}"
    return hits[0]


def _dotted(node) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _call_name(call: ast.Call) -> str:
    return _dotted(call.func).split(".")[-1]


def _calls(handler: ast.ExceptHandler, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(handler)
            if isinstance(n, ast.Call) and _call_name(n) == name]


def _local_callable(handler, node):
    """The nested `def` (by name) or Lambda that a report_handled `show` arg is."""
    if isinstance(node, ast.Lambda):
        return node
    if isinstance(node, ast.Name):
        for n in ast.walk(handler):
            if isinstance(n, ast.FunctionDef) and n.name == node.id:
                return n
    return None


def _calls_setter(tree, setter: str) -> bool:
    return any(isinstance(n, ast.Call) and _dotted(n.func) == setter
               for n in ast.walk(tree))


def _assert_reports_to(handler, setter: str, function=None) -> None:
    """Handler binds the exception and gives report_handled the bound exception;
    the `show` argument is the literal `setter`, or a local callable that calls
    it.  When the local callable only records the text (the dashboard's child
    errors, shown after the loop) pass `function`: it must call `setter`."""
    assert handler.name, "handler must bind the exception (`as exc`)"
    calls = _calls(handler, "report_handled")
    assert calls, "handler must call report_handled"
    call = calls[0]
    assert len(call.args) >= 3, "report_handled(show, context, exc)"
    last = call.args[2]
    assert isinstance(last, ast.Name) and last.id == handler.name, \
        "report_handled must be given the bound exception"
    show = call.args[0]
    if _dotted(show) == setter:
        return
    local = _local_callable(handler, show)
    assert local is not None, \
        f"show must be {setter} or a local callable, got {ast.unparse(show)}"
    target = function if function is not None else local
    assert _calls_setter(target, setter), f"{setter} is never called by the show path"


# (module, function, marker in the try body, setter text, what "carry on" looks like)
SURFACE_SITES = [
    ("decomposition_window", "update_curve", "decompose_curve",
     "self._context_label.setText"),
    ("decomposition_window", "update_curve", "analyse_curve",
     "self._context_label.setText"),
    ("decomposition_window", "_on_nav_curve_selected", "load_force_curve",
     "self._context_label.setText"),
    ("decomposition_window", "_sync_profile_owner", "active_param_owner",
     "self._context_label.setText"),
    ("display_roi", "_sync_profile_owner", "active_param_owner",
     "self._params_label.setText"),
    ("display_roi", "update_curve", "load_force_curve",
     "self._status_label.setText"),
    ("display_roi", "_recompute_and_draw", "compute_curve_events_coords",
     "self._status_label.setText"),
    ("dist_fit_window", "_load_saved_fits", "get_distribution_fits",
     "self._save_status.setText"),
    ("dashboard_window", "_compute_freshness", "queue_freshness",
     "self._worker_status_lbl.setText"),
    ("dashboard_window", "_refresh_facets", "facets",
     "self._scope_count_lbl.setText"),
    ("dashboard_window", "_on_scope_edit", "apply_scope",
     "self._scope_count_lbl.setText"),
    ("dashboard_window", "_refresh_population_children", "reload_paths",
     "self._scope_count_lbl.setText"),
    ("wlc_view_window", "_load_and_fit", "compute_curve_events_coords",
     "self._top.setTitle"),
    ("isoforce_window", "_load_and_mark", "compute_curve_events_coords",
     "self._plot.setTitle"),
    ("event_summary_window", "_experimentalist_id", "get_experimentalist_for_file",
     "self._warn_label.setText"),
]


@pytest.mark.parametrize(
    "module,func,marker,setter", SURFACE_SITES,
    ids=[f"{m}.{f}[{k}]" for m, f, k, _ in SURFACE_SITES])
def test_handler_reports_to_its_widget(module, func, marker, setter):
    # The child-scope handler only records the text; the function writes it after the loop.
    whole = (_function(module, func)
             if (module, func) == ("dashboard_window", "_on_scope_edit") else None)
    _assert_reports_to(_handler(module, func, marker), setter, whole)


@pytest.mark.parametrize("func,marker", [
    ("_on_scope_edit", "apply_scope"),
    ("_refresh_population_children", "reload_paths"),
])
def test_dashboard_child_handlers_name_the_child_window(func, marker):
    # b7: the title is read into a local before the try (a destroyed child's
    # windowTitle() would raise inside the handler); the context uses that local.
    fn = _function("dashboard_window", func)
    h = _handler("dashboard_window", func, marker)
    assert not _calls(h, "windowTitle"), "windowTitle() must not be called in the handler"
    titles = {t.id for n in ast.walk(fn) if isinstance(n, ast.Assign)
              and any(_call_name(c) == "windowTitle"
                      for c in ast.walk(n.value) if isinstance(c, ast.Call))
              for t in n.targets if isinstance(t, ast.Name)}
    assert titles, "windowTitle() must be read into a local before the try"
    context = _calls(h, "report_handled")[0].args[1]
    used = {n.id for n in ast.walk(context) if isinstance(n, ast.Name)}
    assert used & titles, "context must include the child's title"


# Fallbacks that must survive ("otherwise carry on as today").
def _returns(h): return any(isinstance(n, ast.Return) for n in ast.walk(h))
def _clears(h): return bool(_calls(h, "_clear"))
def _assigns(name):
    def check(h):
        return any(isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name
            for tgt in n.targets for t in ast.walk(tgt))
            for n in ast.walk(h))
    return check


FALLBACKS = [
    ("decomposition_window", "update_curve", "decompose_curve", _clears),
    ("decomposition_window", "update_curve", "analyse_curve", _assigns("stage1")),
    ("decomposition_window", "_on_nav_curve_selected", "load_force_curve", _clears),
    ("decomposition_window", "_sync_profile_owner", "active_param_owner", _returns),
    ("display_roi", "_sync_profile_owner", "active_param_owner", _returns),
    ("display_roi", "update_curve", "load_force_curve", _clears),
    ("dashboard_window", "_compute_freshness", "queue_freshness", _returns),
    ("dashboard_window", "_refresh_facets", "facets", _returns),
    ("wlc_view_window", "_load_and_fit", "compute_curve_events_coords", _returns),
    ("isoforce_window", "_load_and_mark", "compute_curve_events_coords", _returns),
    ("event_summary_window", "_experimentalist_id", "get_experimentalist_for_file",
     _assigns("who")),
]


@pytest.mark.parametrize(
    "module,func,marker,check", FALLBACKS,
    ids=[f"{m}.{f}[{k}]" for m, f, k, _ in FALLBACKS])
def test_handler_keeps_todays_fallback(module, func, marker, check):
    h = _handler(module, func, marker)
    assert check(h), "the fallback behaviour was dropped"
    assert _calls(h, "report_handled"), "and it must also report"


# ── display_roi specifics ────────────────────────────────────────────────────

def test_display_roi_update_curve_no_longer_names_load_error():
    h = _handler("display_roi", "update_curve", "load_force_curve")
    names = {n.id for n in ast.walk(h.type) if isinstance(n, ast.Name)}
    assert "LoadError" not in names
    assert "Exception" in names


def test_display_roi_recompute_does_not_print_to_stderr():
    h = _handler("display_roi", "_recompute_and_draw", "compute_curve_events_coords")
    assert not _calls(h, "print")
    assert "stderr" not in ast.unparse(h)
    assert _calls(h, "report_handled")


# ── bulk_metadata_dialog: log + placeholder ──────────────────────────────────

def test_bulk_metadata_lookup_failure_is_logged_and_marked_in_the_box():
    h = _handler("bulk_metadata_dialog", "_make_value_box", "get_distinct_values")
    assert h.name, "handler must bind the exception"
    log = _calls(h, "log_handled")
    assert log, "handler must call crashlog.log_handled"
    assert isinstance(log[-1].args[-1], ast.Name) and log[-1].args[-1].id == h.name
    assert _placeholder_set_on_failure(), "box placeholder must read the failure text"
    assert _assigns("existing")(h), "falls back to an empty list as before"


def _placeholder_set_on_failure() -> bool:
    """The failure text reaches setPlaceholderText (in the handler, or in the
    function guarded by a flag the handler sets)."""
    src = ast.unparse(_function("bulk_metadata_dialog", "_make_value_box"))
    return "setPlaceholderText" in src and "couldn't load existing values" in src


# ── behavioural: handlers driven on a stand-in `self` ────────────────────────

class _Label:
    def __init__(self):
        self.texts = []

    def setText(self, text):
        self.texts.append(text)

    def setTitle(self, text):
        self.texts.append(text)


@pytest.fixture(autouse=True)
def _no_real_crashlog(monkeypatch):
    monkeypatch.setattr(crashlog, "log_handled", lambda ctx, exc: None)


def _boom(*_a, **_k):
    raise OSError("boom")


def _probe(**attrs):
    p = type("Probe", (), {})()
    for k, v in attrs.items():
        setattr(p, k, v)
    return p


def test_compute_freshness_failure_shown_and_returns_empty(monkeypatch):
    monkeypatch.setattr(dash._db, "queue_freshness", _boom)
    import smfs_catalog.curve_analysis as ca
    monkeypatch.setattr(ca, "current_signature", lambda _db: ("{}", "v"))
    lbl = _Label()
    p = _probe(_db_path="x.db", _worker_status_lbl=lbl)
    assert dash.DashboardWindow._compute_freshness(p) == {}
    assert len(lbl.texts) == 1
    assert lbl.texts[0].endswith(" failed: OSError: boom")


def test_nav_load_failure_shown_clears_and_does_not_update(monkeypatch):
    monkeypatch.setattr(dw, "load_force_curve", _boom)
    lbl, cleared, updated = _Label(), [], []
    p = _probe(_context_label=lbl, _sync_profile_owner=lambda path: None,
               _clear=lambda: cleared.append(1), update_curve=updated.append)
    dw.DecompositionWindow._on_nav_curve_selected(p, "a.ibw", 1)
    assert cleared == [1] and updated == []
    assert len(lbl.texts) == 1
    assert lbl.texts[0].endswith(" failed: OSError: boom")


def test_sync_profile_owner_failure_shown_in_each_window(monkeypatch):
    for mod, cls, attr in ((dw, dw.DecompositionWindow, "_context_label"),
                           (roi, roi.DisplayROI if hasattr(roi, "DisplayROI") else None,
                            "_params_label")):
        if cls is None:
            cls = next(v for v in vars(roi).values()
                       if isinstance(v, type) and "_sync_profile_owner" in vars(v))
        monkeypatch.setattr(mod._db, "active_param_owner", _boom)
        lbl = _Label()
        p = _probe(_db_path="x.db", _owner_synced=False, **{attr: lbl})
        assert cls._sync_profile_owner(p, "a.ibw") is None
        assert len(lbl.texts) == 1, mod.__name__
        assert lbl.texts[0].endswith(" failed: OSError: boom"), mod.__name__
        assert p._owner_synced is False, "previous owner kept, as today"


def test_experimentalist_lookup_failure_shown_and_returns_none(monkeypatch):
    monkeypatch.setattr(esw._db, "get_experimentalist_for_file", _boom)
    lbl = _Label()
    p = _probe(_experimentalist_resolved=False, _results=[{"path": "a.ibw"}],
               _db_path="x.db", _warn_label=lbl, _experimentalist=None)
    assert esw.EventSummaryWindow._experimentalist_id(p) is None
    assert len(lbl.texts) == 1
    assert lbl.texts[0].endswith(" failed: OSError: boom")


def test_child_apply_scope_failure_names_the_child_and_siblings_still_run():
    class Bad:
        def isVisible(self): return True
        def windowTitle(self): return "Bad Child Window"
        def apply_scope(self, _s): raise OSError("boom")

    class Good(Bad):
        got = None
        def windowTitle(self): return "Good"
        def apply_scope(self, s): Good.got = s

    class Sig:
        def emit(self, *_): pass

    class Timer:
        def start(self): pass

    lbl = _Label()
    kids = [Bad(), Good()]
    p = _probe(_current_scope=lambda: {"k": 1}, _count_timer=Timer(),
               scope_changed=Sig(), _prune_children=lambda: None,
               _children=kids, _scope_count_lbl=lbl)
    dash.DashboardWindow._on_scope_edit(p)
    assert Good.got == {"k": 1}, "the next child still gets the scope"
    assert len(lbl.texts) == 1
    assert "Bad Child Window" in lbl.texts[0]
    assert lbl.texts[0].endswith(" failed: OSError: boom")
