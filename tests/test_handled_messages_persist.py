# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
PR (b) / b6: a D1 failure message must stay visible until the same action
next succeeds.

WHY THIS EXISTS.  `report_handled` writes "<context> failed: <Name>: <msg>" to
a label.  At four sites the label is rewritten by ordinary code a moment later
(facet refresh -> the count line; a child's apply_scope -> the 250 ms count
timer; sync_profile_owner -> the per-curve redraw; queue freshness -> the
worker status line), so the message was barely seen.  Each test forces the
failure, runs the code that used to overwrite the label, and asserts the
failure text is STILL showing on some label; then it lets the action succeed
and asserts the failure text is gone.

The tests do not say WHICH label shows the message (the plan allows "write
after the overwrite" or "another existing label"), only that one of the
window's real QLabel attributes does.  The stand-in `self` hands out a fake
label for any attribute the source creates with `= QLabel(`; any other
unknown attribute reads as None (as an attribute initialised to None in
__init__ would).  Only the wording after " failed: " is asserted.

Two of the stand-ins (decomposition, display_roi) replace the heavy redraw
with a stub that does just the overwrite the real redraw does
(decomposition_window.update_curve writes `_context_label` unconditionally;
display_roi._recompute_and_draw writes `_status_label`); the "premise" tests
check the source still has that overwrite.

_save_status: the failure text must not use TEXT_GOOD; a later success
("Saved") must use TEXT_GOOD again.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from smfs_catalog import crashlog, style
from smfs_catalog import dashboard_window as dash
from smfs_catalog import decomposition_window as dw
from smfs_catalog import dist_fit_window as dfw
from smfs_catalog import display_roi as roi
from smfs_catalog.scope import new_scope

PKG = Path(__file__).resolve().parents[1] / "smfs_catalog"
FAILED = " failed: OSError: boom"


# -- stand-in `self` ----------------------------------------------------------

class _Label:
    def __init__(self):
        self.text = ""

    def setText(self, text):
        self.text = text

    def setStyleSheet(self, _qss): pass
    def setToolTip(self, _tip): pass
    def setVisible(self, _v): pass


class _Probe:
    """Unbound-method `self`: fake labels for real QLabel attributes, None else."""

    def __init__(self, module: str, **attrs):
        src = (PKG / f"{module}.py").read_text(encoding="utf-8")
        object.__setattr__(self, "_label_names",
                           set(re.findall(r"self\.(\w+) = QLabel\(", src)))
        object.__setattr__(self, "_made", {})
        for k, v in attrs.items():
            setattr(self, k, v)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        if name in self._label_names:
            return self._made.setdefault(name, _Label())
        return None

    def shown(self) -> list[str]:
        return [lbl.text for lbl in self._made.values() if lbl.text]


def _failure_visible(p) -> bool:
    return any(FAILED in t for t in p.shown())


def _boom(*_a, **_k):
    raise OSError("boom")


@pytest.fixture(autouse=True)
def _no_real_crashlog(monkeypatch):
    monkeypatch.setattr(crashlog, "log_handled", lambda ctx, exc: None)


# -- dashboard: facets failure vs the count refresh ---------------------------

class _Model:
    def set_rows(self, _rows): pass


def _dash_probe(**extra):
    p = _Probe("dashboard_window", _db_path="x.db", _scope=new_scope(),
               _lists={}, _db_model=_Model(),
               _refresh_queue_table=lambda: None, **extra)
    p._refresh_facets = lambda: dash.DashboardWindow._refresh_facets(p)
    return p


def _facets(monkeypatch, fn):
    monkeypatch.setattr(dash._db, "get_facet_options", fn)
    monkeypatch.setattr(dash._db, "list_files", lambda *_a, **_k: [1, 2, 3])


def test_facets_failure_survives_the_count_refresh(monkeypatch):
    _facets(monkeypatch, _boom)
    p = _dash_probe()
    dash.DashboardWindow._refresh_db_and_counts(p)
    assert _failure_visible(p), p.shown()


def test_facets_failure_clears_when_facets_next_succeed(monkeypatch):
    _facets(monkeypatch, _boom)
    p = _dash_probe()
    dash.DashboardWindow._refresh_db_and_counts(p)
    _facets(monkeypatch, lambda *_a, **_k: {})
    dash.DashboardWindow._refresh_db_and_counts(p)
    assert not _failure_visible(p), p.shown()
    assert any("3 files" in t for t in p.shown()), "normal count text still shown"


# -- dashboard: child apply_scope failure vs the count timer ------------------

class _Sig:
    def emit(self, *_): pass


class _Timer:
    def start(self): pass


class _Child:
    def __init__(self, fail):
        self.fail = fail

    def isVisible(self): return True
    def windowTitle(self): return "Some Child"

    def apply_scope(self, _scope):
        if self.fail:
            raise OSError("boom")


def _scope_probe(monkeypatch, children):
    _facets(monkeypatch, lambda *_a, **_k: {})
    return _dash_probe(_current_scope=lambda: new_scope(), _count_timer=_Timer(),
                       scope_changed=_Sig(), _prune_children=lambda: None,
                       _children=children)


def test_child_failure_survives_the_count_timer(monkeypatch):
    p = _scope_probe(monkeypatch, [_Child(fail=True)])
    dash.DashboardWindow._on_scope_edit(p)
    dash.DashboardWindow._refresh_db_and_counts(p)   # what the 250 ms timer runs
    assert _failure_visible(p), p.shown()


def test_child_failure_clears_when_apply_scope_next_succeeds(monkeypatch):
    kid = _Child(fail=True)
    p = _scope_probe(monkeypatch, [kid])
    dash.DashboardWindow._on_scope_edit(p)
    dash.DashboardWindow._refresh_db_and_counts(p)
    kid.fail = False
    dash.DashboardWindow._on_scope_edit(p)
    dash.DashboardWindow._refresh_db_and_counts(p)
    assert not _failure_visible(p), p.shown()


def test_child_failure_not_cleared_by_a_healthy_sibling(monkeypatch):
    p = _scope_probe(monkeypatch, [_Child(fail=True), _Child(fail=False)])
    dash.DashboardWindow._on_scope_edit(p)
    dash.DashboardWindow._refresh_db_and_counts(p)
    assert _failure_visible(p), p.shown()


# -- dashboard: freshness failure vs the worker status line -------------------

def _freshness_probe():
    return _Probe("dashboard_window", _db_path="x.db", _worker=None,
                  _current_direction=1, _current_playhead_id=None,
                  _loaded_count=0, _cached_count=0, _queue_id_to_row={},
                  _queue_total=0, _current_rate=lambda: 0.0,
                  _eta_text=lambda: None)


def _freshness(monkeypatch, fn):
    import smfs_catalog.curve_analysis as ca
    monkeypatch.setattr(ca, "current_signature", lambda _db: ("{}", "v"))
    monkeypatch.setattr(dash._db, "queue_freshness", fn)


def test_freshness_failure_survives_the_worker_status_update(monkeypatch):
    _freshness(monkeypatch, _boom)
    p = _freshness_probe()
    assert dash.DashboardWindow._compute_freshness(p) == {}
    dash.DashboardWindow._update_location_label(p)
    assert _failure_visible(p), p.shown()


def test_freshness_failure_clears_when_freshness_next_succeeds(monkeypatch):
    _freshness(monkeypatch, _boom)
    p = _freshness_probe()
    dash.DashboardWindow._compute_freshness(p)
    dash.DashboardWindow._update_location_label(p)
    _freshness(monkeypatch, lambda *_a, **_k: {1: "fresh"})
    assert dash.DashboardWindow._compute_freshness(p) == {1: "fresh"}
    dash.DashboardWindow._update_location_label(p)
    assert not _failure_visible(p), p.shown()
    assert any(t.startswith("Worker:") for t in p.shown()), "status line still shown"


# -- _sync_profile_owner in the two windows -----------------------------------

def _function(module: str, cls: str, name: str) -> ast.FunctionDef:
    tree = ast.parse((PKG / f"{module}.py").read_text(encoding="utf-8"))
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in klass.body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def _writes_label(fn: ast.FunctionDef, label: str) -> bool:
    return any(isinstance(n, ast.Call) and ast.unparse(n.func) == f"self.{label}.setText"
               for n in ast.walk(fn))


def test_premise_decomposition_update_curve_rewrites_context_label():
    assert _writes_label(_function("decomposition_window", "DecompositionWindow",
                                   "update_curve"), "_context_label")


def test_premise_display_roi_redraw_rewrites_status_label():
    assert _writes_label(_function("display_roi", "ROIWindow",
                                   "_recompute_and_draw"), "_status_label")


def _decomp_probe(monkeypatch):
    monkeypatch.setattr(dw, "load_force_curve", lambda path: SimpleNamespace(path=path))
    p = _Probe("decomposition_window", _db_path="x.db",
               _owner_synced=True, _experimentalist="alice")
    p._sync_profile_owner = lambda path: dw.DecompositionWindow._sync_profile_owner(p, path)
    p._clear = lambda: None
    # What the real update_curve does to the label on every normal curve.
    p.update_curve = lambda curve: p._context_label.setText(
        f"{curve.path}   |   parameters: alice")
    return p


def _roi_probe(monkeypatch):
    monkeypatch.setattr(roi, "load_force_curve", lambda path: SimpleNamespace(path=path))
    p = _Probe("display_roi", _db_path="x.db", _results=[{"path": "a.ibw"}],
               _owner_synced=True, _experimentalist="alice")
    p._set_file_label = lambda path: None
    p._sync_profile_owner = lambda path: roi.ROIWindow._sync_profile_owner(p, path)
    p._clear = lambda: None
    # What the real _recompute_and_draw does to the label on every normal curve.
    p._recompute_and_draw = lambda curve: p._status_label.setText(
        "detector: x   |   no ROI found")
    return p


def test_decomposition_owner_failure_survives_the_curve_redraw(monkeypatch):
    p = _decomp_probe(monkeypatch)
    monkeypatch.setattr(dw._db, "active_param_owner", _boom)
    dw.DecompositionWindow._on_nav_curve_selected(p, "a.ibw", 1)
    assert _failure_visible(p), p.shown()


def test_decomposition_owner_failure_clears_when_lookup_next_succeeds(monkeypatch):
    p = _decomp_probe(monkeypatch)
    monkeypatch.setattr(dw._db, "active_param_owner", _boom)
    dw.DecompositionWindow._on_nav_curve_selected(p, "a.ibw", 1)
    monkeypatch.setattr(dw._db, "active_param_owner", lambda _p: "alice")
    dw.DecompositionWindow._on_nav_curve_selected(p, "a.ibw", 1)
    assert not _failure_visible(p), p.shown()


def test_display_roi_owner_failure_survives_the_curve_redraw(monkeypatch):
    p = _roi_probe(monkeypatch)
    monkeypatch.setattr(roi._db, "active_param_owner", _boom)
    roi.ROIWindow.update_curve(p, 0)
    assert _failure_visible(p), p.shown()


def test_display_roi_owner_failure_clears_when_lookup_next_succeeds(monkeypatch):
    p = _roi_probe(monkeypatch)
    monkeypatch.setattr(roi._db, "active_param_owner", _boom)
    roi.ROIWindow.update_curve(p, 0)
    monkeypatch.setattr(roi._db, "active_param_owner", lambda _p: "alice")
    roi.ROIWindow.update_curve(p, 0)   # same owner as before: early-return path
    assert not _failure_visible(p), p.shown()


# -- dist_fit_window._save_status colour --------------------------------------

class _StyledLabel(_Label):
    def __init__(self):
        super().__init__()
        self.qss = style.qss_text(style.TEXT_GOOD, size_px=10)   # as built

    def setStyleSheet(self, qss):
        self.qss = qss


class _Btn:
    def setEnabled(self, _v): pass


def _colour_in(qss: str, colour: str) -> bool:
    return colour.lower() in qss.lower()


def _fit_probe():
    lbl = _StyledLabel()
    f = {"param_labels": ["mu"], "popt": [1.0], "perr": [0.1], "ci_lo": [0.8],
         "ci_hi": [1.2], "gof": {"R2": 0.9}, "n_bins": 10, "range_lo": 0.0,
         "range_hi": 1.0, "range_is_auto": True, "n_excluded_by_range": 0,
         "starts": [], "n_values": 50, "n_peaks": 1, "model_label": "G1"}
    p = SimpleNamespace(
        _save_status=lbl, _variable="v", _units="u",
        _context=SimpleNamespace(db_path="x.db"),
        _fit_history=[{"saved": False}], _last_fit=f,
        _plot=SimpleNamespace(normalized=False), _save_btn=_Btn(),
        _update_comparison_table=lambda: None)
    return p, lbl


def test_load_failure_text_is_not_in_the_success_colour(monkeypatch):
    monkeypatch.setattr(dfw._db, "get_distribution_fits", _boom)
    p, lbl = _fit_probe()
    dfw._ModelPane._load_saved_fits(p)
    assert lbl.text.endswith(FAILED), lbl.text
    assert not _colour_in(lbl.qss, style.TEXT_GOOD), lbl.qss


def test_load_failure_text_uses_the_error_or_warning_colour(monkeypatch):
    monkeypatch.setattr(dfw._db, "get_distribution_fits", _boom)
    p, lbl = _fit_probe()
    dfw._ModelPane._load_saved_fits(p)
    assert (_colour_in(lbl.qss, style.TEXT_BAD)
            or _colour_in(lbl.qss, style.TEXT_WARNING)), lbl.qss


def test_saved_message_is_in_the_success_colour_after_a_failure(monkeypatch):
    monkeypatch.setattr(dfw._db, "get_distribution_fits", _boom)
    monkeypatch.setattr(dfw._db, "save_distribution_fit", lambda *_a, **_k: None)
    p, lbl = _fit_probe()
    dfw._ModelPane._load_saved_fits(p)
    dfw._ModelPane._save_fit(p)
    assert lbl.text.startswith("Saved"), lbl.text
    assert _colour_in(lbl.qss, style.TEXT_GOOD), lbl.qss
    assert not _colour_in(lbl.qss, style.TEXT_BAD), lbl.qss
    assert not _colour_in(lbl.qss, style.TEXT_WARNING), lbl.qss


class _Prog:
    cancelled = False

    def __init__(self, *_a, **_k): pass
    def tick(self, *_a): pass
    def close(self): pass


def _bootstrap_probe(monkeypatch, boot):
    monkeypatch.setattr(dfw, "CancelableProgress", _Prog)
    monkeypatch.setattr(dfw, "bootstrap_fit_ci", lambda *_a, **_k: boot)
    monkeypatch.setattr(dfw._db, "get_distribution_fits", _boom)
    p, lbl = _fit_probe()
    p._plot = SimpleNamespace(edges=None, normalized=False)
    dfw._ModelPane._load_saved_fits(p)
    assert _colour_in(lbl.qss, style.TEXT_BAD), "precondition: failure is red"
    return p, lbl


@pytest.mark.parametrize("boot,expected", [
    (None, "No confidence interval"),
    (SimpleNamespace(n_failed=3, n_draws=10, n_ok=7), "did not converge"),
], ids=["no-interval", "some-did-not-converge"])
def test_non_failure_message_after_a_load_failure_is_in_the_success_colour(
        monkeypatch, boot, expected):
    p, lbl = _bootstrap_probe(monkeypatch, boot)
    dfw._ModelPane._run_bootstrap(p, [], [], [], [], [])
    assert expected in lbl.text, lbl.text
    assert _colour_in(lbl.qss, style.TEXT_GOOD), lbl.qss
    assert not _colour_in(lbl.qss, style.TEXT_BAD), lbl.qss
