"""b7 hardening.

(i) A stored wlc/bwlc segment with l_p == 0.0 makes `kT / l_p` raise
ZeroDivisionError inside ChainFit.x_at_force. Before PR (b) that gave "anchor
undefined" (the curve is dropped); it must still. `_fit_drop_reason` is not
involved: it still accepts such a fit.

(ii) The dashboard's two child-window handlers must not call windowTitle()
inside the except block: a deleted child's windowTitle() raises RuntimeError,
which would escape the handler. The title is read before the try.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from smfs_catalog import crashlog
from smfs_catalog import dashboard_window as dash
from smfs_catalog import event_processor as ep
from smfs_catalog.base_2dh_window import _TwoDHWindowBase
from smfs_catalog.models import ChainFit

PKG = Path(__file__).resolve().parents[1] / "smfs_catalog"


# -- (i) l_p == 0 -------------------------------------------------------------

@pytest.mark.parametrize("model", ["wlc", "bwlc"])
def test_x_at_force_zero_persistence_length_is_none(model):
    assert ChainFit(model, (0.0, 50.0)).x_at_force(5.0) is None


@pytest.mark.parametrize("model", ["wlc", "bwlc"])
def test_zero_lp_segment_gives_undefined_fstar_anchor(model):
    x = np.linspace(1.0, 40.0, 30)
    F = np.linspace(1.0, 30.0, 30)
    anchor = ep.PHYS_ALIGN_ANCHORS["fstar"](x, F, ChainFit(model, (0.0, 50.0)), 5.0)
    assert anchor is None
    # ... and a None anchor drops the curve (no histogram).
    grid = ep.HistGrid(ep.PHYS_X_BINS, ep.PHYS_F_BINS,
                       ep.PHYS_X_RANGE, ep.PHYS_F_RANGE)
    assert ep.compute_physical_histogram_at(x, F, anchor, grid) is None


def test_zero_lp_fit_is_not_given_a_new_drop_reason():
    # The drop reason recorded is unchanged: l_p <= 0 is not rejected up front.
    fit = ChainFit("wlc", (0.0, 50.0))
    assert _TwoDHWindowBase._fit_drop_reason(None, fit) is None


# -- (ii) windowTitle read before the try ------------------------------------

class _Label:
    def __init__(self):
        self.texts = []

    def setText(self, text):
        self.texts.append(text)


@pytest.fixture(autouse=True)
def _no_real_crashlog(monkeypatch):
    monkeypatch.setattr(crashlog, "log_handled", lambda ctx, exc: None)


def _probe(**attrs):
    p = type("Probe", (), {})()
    for k, v in attrs.items():
        setattr(p, k, v)
    return p


class _Doomed:
    """Its failing method also destroys it: windowTitle() raises afterwards."""
    dead = False

    def isVisible(self): return True

    def windowTitle(self):
        if self.dead:
            raise RuntimeError("wrapped C/C++ object has been deleted")
        return "Doomed Child"

    def _fail(self, *_a):
        self.dead = True
        raise OSError("boom")

    apply_scope = reload_paths = set_event_paths = refresh = _fail


def test_scope_edit_survives_deleted_child_and_names_its_title():
    class Good:
        got = None
        def isVisible(self): return True
        def windowTitle(self): return "Good"
        def apply_scope(self, s): Good.got = s

    class Sig:
        def emit(self, *_): pass

    class Timer:
        def start(self): pass

    lbl = _Label()
    p = _probe(_current_scope=lambda: {"k": 1}, _count_timer=Timer(),
               scope_changed=Sig(), _prune_children=lambda: None,
               _children=[_Doomed(), Good()], _scope_count_lbl=lbl)
    dash.DashboardWindow._on_scope_edit(p)
    assert Good.got == {"k": 1}, "the next child still gets the scope"
    assert len(lbl.texts) == 1
    assert "Doomed Child" in lbl.texts[0]
    assert lbl.texts[0].endswith(" failed: OSError: boom")


@pytest.mark.parametrize("method", ["reload_paths", "set_event_paths", "refresh"])
def test_refresh_children_survives_deleted_child_and_names_its_title(method):
    class Doomed(_Doomed):
        pass
    for other in ("reload_paths", "set_event_paths", "refresh"):
        if other != method:
            setattr(Doomed, other, property(lambda self: (_ for _ in ()).throw(
                AttributeError(other))))   # hasattr() is False for these

    class Good:
        called = False
        def isVisible(self): return True
        def windowTitle(self): return "Good"
        def refresh(self): Good.called = True

    lbl = _Label()
    p = _probe(_prune_children=lambda: None, _queue_event_paths=lambda: ["a"],
               _children=[Doomed(), Good()], _scope_count_lbl=lbl)
    dash.DashboardWindow._refresh_population_children(p)
    assert Good.called, "the next child is still refreshed"
    assert len(lbl.texts) == 1
    assert "Doomed Child" in lbl.texts[0]
    assert lbl.texts[0].endswith(" failed: OSError: boom")


def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse((PKG / "dashboard_window.py").read_text(encoding="utf-8"))
    found = [n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(found) == 1
    return found[0]


def _title_calls(node):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == "windowTitle"]


@pytest.mark.parametrize("func", ["_on_scope_edit", "_refresh_population_children"])
def test_window_title_not_called_inside_except_handler(func):
    fn = _function(func)
    handlers = [h for t in ast.walk(fn) if isinstance(t, ast.Try) for h in t.handlers]
    assert handlers
    for h in handlers:
        assert not _title_calls(h), "windowTitle() must be read before the try"
    assert _title_calls(fn), "the title is still read (before the try)"
