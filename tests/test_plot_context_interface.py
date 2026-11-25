# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Pass-1 step d3: the plot-window context dataclasses in
`smfs_catalog/plot_context.py` and the five constructors that take them.

Argument counts are the plan's (they leave `self` out, as inspect.signature
does for a class).  The four dataclasses are frozen, like PR (c)'s parameter
objects and d2's HistGrid: no window mutates a context (each copies the lists
and dicts it keeps, and the popouts build a new ExportContext with
dataclasses.replace).

Open points settled here, for the coder:
  * PCAWindow's `grid` is d2's `event_processor.HistGrid`;
  * MeanCurveWindow no longer takes `display`: every caller passed
    sqrt(counts), so the window derives it;
  * PCAWindow's `display` argument (the full-grid histograms behind a
    selection window) has no planned dataclass; its type is left to the coder
    and no test constructs it.
"""
import dataclasses
import importlib
import inspect
from pathlib import Path

import pytest

from smfs_catalog import event_processor
from size_limits import arg_count, violations
from source_inspect import function_node, imported_names

PKG = Path(event_processor.__file__).resolve().parent
_PLOT_CONTEXT = PKG / "plot_context.py"


def _ctx():
    assert _PLOT_CONTEXT.exists(), "smfs_catalog/plot_context.py does not exist"
    return importlib.import_module("smfs_catalog.plot_context")


# -- the four dataclasses -----------------------------------------------------

_FIELDS = {
    "AxisLabels":    ["x_label", "y_label", "x_unit", "y_unit"],
    "HistView":      ["counts", "auto_max", "z_pct", "lut", "x_range", "f_range"],
    "ExportContext": ["caption", "provenance", "paths", "db_path", "physical",
                      "overlay_fn"],
    "FitContext":    ["db_path", "paths", "caption"],
}


@pytest.mark.parametrize("name", _FIELDS)
def test_dataclass_has_exactly_the_planned_fields_in_order(name):
    cls = getattr(_ctx(), name)
    assert dataclasses.is_dataclass(cls)
    assert [f.name for f in dataclasses.fields(cls)] == _FIELDS[name]


@pytest.mark.parametrize("name", _FIELDS)
def test_dataclass_is_frozen(name):
    cls = getattr(_ctx(), name)
    assert cls.__dataclass_params__.frozen


def test_axis_labels_unit_fields_default_to_empty():
    axes = _ctx().AxisLabels("Extension", "Force")
    assert (axes.x_label, axes.y_label, axes.x_unit, axes.y_unit) == (
        "Extension", "Force", "", "")
    with pytest.raises(dataclasses.FrozenInstanceError):
        axes.x_label = "other"


def test_hist_view_stores_what_it_is_given():
    import numpy as np
    counts, lut = np.ones((2, 3)), np.zeros((256, 4), dtype=np.uint8)
    view = _ctx().HistView(counts, 2.5, 80, lut, (0.0, 1.0), (-1.0, 1.0))
    assert view.counts is counts and view.lut is lut
    assert (view.auto_max, view.z_pct) == (2.5, 80)
    assert (view.x_range, view.f_range) == ((0.0, 1.0), (-1.0, 1.0))


def test_export_context_is_all_optional_and_empty_by_default():
    ctx = _ctx().ExportContext()
    assert ctx.caption == ""
    assert not ctx.provenance and not ctx.paths
    assert ctx.db_path is None and ctx.physical is False
    assert ctx.overlay_fn is None


def test_fit_context_needs_only_the_database():
    ctx = _ctx().FitContext("catalog.db")
    assert ctx.db_path == "catalog.db"
    assert not ctx.paths and ctx.caption == ""


def test_plot_context_imports_no_qt_and_no_database():
    assert _PLOT_CONTEXT.exists(), "smfs_catalog/plot_context.py does not exist"
    banned = {"pyqt6", "pyqtgraph", "sqlite3", "db", "qt_utils"}
    bad = [n for n in imported_names(_PLOT_CONTEXT)
           if banned & {part.lower() for part in n.split(".")}]
    assert not bad, f"plot_context.py imports {bad}"


# -- the five constructors, by inspection -------------------------------------

def _params(module, cls):
    mod = importlib.import_module(f"smfs_catalog.{module}")
    return inspect.signature(getattr(mod, cls)).parameters


_SIGNATURES = [
    ("mean_curve_window", "MeanCurveWindow", ["title", "view", "axes", "context"]),
    ("pca_window", "PCAWindow", ["histograms", "grid", "axes", "context", "display"]),
    ("gmm_fit_window", "_ScatterPane", ["xy", "axes", "caption", "parent"]),
    ("gmm_fit_window", "_ModelPane",
     ["scatter_pane", "xy", "axes", "context", "parent"]),
    ("dist_fit_window", "_ModelPane",
     ["plot_pane", "variable", "units", "context", "parent"]),
]
_IDS = [f"{m}.{c}" for m, c, _ in _SIGNATURES]


@pytest.mark.parametrize("module,cls,names", _SIGNATURES, ids=_IDS)
def test_constructor_takes_the_planned_arguments(module, cls, names):
    assert list(_params(module, cls)) == names


def test_constructor_defaults_are_the_planned_ones():
    assert _params("pca_window", "PCAWindow")["display"].default is None
    assert _params("gmm_fit_window", "_ScatterPane")["caption"].default == ""
    for module, cls in [("gmm_fit_window", "_ScatterPane"),
                        ("gmm_fit_window", "_ModelPane"),
                        ("dist_fit_window", "_ModelPane")]:
        assert _params(module, cls)["parent"].default is None


def test_pca_window_grid_is_the_d2_hist_grid():
    """The decision recorded in the plan: PCAWindow's grid argument is
    event_processor.HistGrid (x_bins, f_bins, x_range, f_range)."""
    ann = _params("pca_window", "PCAWindow")["grid"].annotation
    assert "HistGrid" in (ann if isinstance(ann, str) else ann.__name__)


# -- size guard ---------------------------------------------------------------
# Argument count only (self left out, as in the plan).  Every constructor but
# gmm's _ScatterPane is already over 50 lines today (PCAWindow also over 10
# decision points), and d3 targets arguments, so only _ScatterPane, which meets
# the other limits now, is held to all of them.

def _constructor(module, cls):
    return function_node(PKG / f"{module}.py", "__init__", cls)


@pytest.mark.parametrize("module,cls,names", _SIGNATURES, ids=_IDS)
def test_constructor_has_at_most_five_arguments(module, cls, names):
    assert arg_count(_constructor(module, cls)) - 1 <= 5


def test_scatter_pane_constructor_meets_every_limit():
    fn = _constructor("gmm_fit_window", "_ScatterPane")
    assert violations("_ScatterPane.__init__", fn) == []
