# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Pass-1 step d3 behaviour pins: what the five windows do with the labels, units,
caption, provenance, paths and database they are given.

Written through the small adapters below, which were switched from the old
keyword API to the AxisLabels / HistView / ExportContext / FitContext / HistGrid
form when d3 landed.  The tests themselves did not change.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication

from smfs_catalog.event_processor import HistGrid
from smfs_catalog.plot_context import (
    AxisLabels, ExportContext, FitContext, HistView,
)
from smfs_catalog.dist_fit_window import _ModelPane as _DistModelPane, _PlotPane
from smfs_catalog.gmm_fit_window import _ModelPane as _GmmModelPane, _ScatterPane
from smfs_catalog.mean_curve_window import MeanCurveWindow
from smfs_catalog.pca_window import PCAWindow

_APP = QApplication.instance() or QApplication([])

_LUT = np.zeros((256, 4), dtype=np.uint8)
_PROV = {"population": "pass", "segment": "extension"}


def _overlay(path):
    return None


# -- adapters: the only place that knows today's keyword API -------------------

def _mean_curve(counts, **kw):
    """kw: title, x_range, f_range, x_label, f_label, x_unit, f_unit, caption,
    physical, paths, provenance, db_path; optional overlay_fn, z_pct, auto_max."""
    return MeanCurveWindow(
        kw["title"],
        HistView(counts, kw.get("auto_max", 1.0), kw.get("z_pct", 100), _LUT,
                 kw["x_range"], kw["f_range"]),
        AxisLabels(kw["x_label"], kw["f_label"], kw["x_unit"], kw["f_unit"]),
        ExportContext(kw["caption"], kw["provenance"] or {}, kw["paths"] or [],
                      kw["db_path"], kw["physical"], kw.get("overlay_fn")))


def _pca(histograms, **kw):
    """kw: x_bins, f_bins, x_range, f_range, x_label, f_label, x_unit, f_unit,
    caption, provenance, db_path, physical, overlay_fn."""
    return PCAWindow(
        histograms,
        HistGrid(kw["x_bins"], kw["f_bins"], kw["x_range"], kw["f_range"]),
        AxisLabels(kw["x_label"], kw["f_label"], kw["x_unit"], kw["f_unit"]),
        ExportContext(kw["caption"], kw["provenance"], [], kw["db_path"],
                      kw["physical"], kw["overlay_fn"]))


def _scatter(xy, labels, caption):
    """labels is (x_label, y_label, x_unit, y_unit)."""
    return _ScatterPane(xy, AxisLabels(*labels), caption)


def _gmm_model(scatter, xy, labels, context):
    """labels as for _scatter; context is (db_path, paths, caption)."""
    return _GmmModelPane(scatter, xy, AxisLabels(*labels), FitContext(*context))


def _dist_model(plot, variable, units, context):
    """context is (db_path, paths, caption)."""
    return _DistModelPane(plot, variable, units, FitContext(*context))


def _paths_of(window):
    return window._context.paths


def _db_of(pane):
    return pane._context.db_path


# -- MeanCurveWindow -----------------------------------------------------------

def _mc(**over):
    kw = dict(title="Cluster 3  (n=7)", x_range=(0.0, 10.0), f_range=(-3.0, 3.0),
              x_label="Extension", f_label="Force", x_unit="nm", f_unit="pN",
              caption="pass | extension | 200x150", physical=True,
              paths=["a.ibw", "b.ibw"], provenance=dict(_PROV),
              db_path="catalog.db", overlay_fn=_overlay)
    kw.update(over)
    return _mean_curve(np.ones((5, 6)), **kw)


def test_mean_curve_provenance_records_title_labels_units_caption_ranges():
    window = _mc(z_pct=80)
    prov = window.export_provenance()
    window.close()
    assert prov["window"] == "mean_curve"
    assert prov["view_title"] == "Cluster 3  (n=7)"
    assert prov["caption"] == "pass | extension | 200x150"
    assert prov["physical"] is True
    assert (prov["x_label"], prov["f_label"]) == ("Extension", "Force")
    assert (prov["x_unit"], prov["f_unit"]) == ("nm", "pN")
    assert prov["x_range"] == [0.0, 10.0] and prov["f_range"] == [-3.0, 3.0]
    assert prov["z_clip_pct"] == 80


def test_mean_curve_image_is_the_sqrt_of_counts_clipped_at_auto_max():
    """The display array is sqrt(counts): sqrt = 0,1,2 / 3,4,5, clip level 4,
    so 254*[0,1,2,3,4]/4 truncated = 0,63,127,190,254 and the 5 (> clip) is 255."""
    counts = np.array([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]])
    kw = dict(title="t", x_range=(0.0, 1.0), f_range=(0.0, 1.0), x_label="x",
              f_label="f", x_unit="", f_unit="", caption="", physical=False,
              paths=None, provenance=None, db_path=None)
    window = _mean_curve(counts, auto_max=4.0, z_pct=100, **kw)
    image = np.asarray(window._img.image)
    window.close()
    np.testing.assert_array_equal(image, [[0, 63, 127], [190, 254, 255]])


def test_mean_curve_provenance_keeps_the_parents_entries():
    window = _mc()
    prov = window.export_provenance()
    window.close()
    assert prov["population"] == "pass" and prov["segment"] == "extension"


def test_mean_curve_physical_flag_controls_the_wlc_button():
    on, off = _mc(physical=True), _mc(physical=False)
    has_on, has_off = hasattr(on, "_wlc_btn"), hasattr(off, "_wlc_btn")
    on.close()
    off.close()
    assert has_on and not has_off


def test_mean_curve_overlay_panel_needs_paths_and_an_overlay_function():
    both = _mc()
    no_paths = _mc(paths=None)
    no_fn = _mc(overlay_fn=None)
    got = [w._overlay_panel is not None for w in (both, no_paths, no_fn)]
    for w in (both, no_paths, no_fn):
        w.close()
    assert got == [True, False, False]


def test_mean_curve_axes_carry_the_labels_and_units():
    window = _mc()
    bottom = window._pw.getAxis("bottom")
    left = window._pw.getAxis("left")
    got = (bottom.labelText, bottom.labelUnits, left.labelText, left.labelUnits)
    window.close()
    assert got == ("Extension", "nm", "Force", "pN")


# -- PCAWindow: the popouts it builds ------------------------------------------

def _pca_window(**over):
    rng = np.random.default_rng(7)
    histograms = {f"trace-{i}": rng.poisson(0.4, size=(5, 4)).astype(np.float32)
                  for i in range(6)}
    kw = dict(x_bins=5, f_bins=4, x_range=(1.25, 3.75), f_range=(-2.0, 6.0),
              x_label="Ext", f_label="Frc", x_unit="nm", f_unit="pN",
              caption="pass | 5x4", provenance=dict(_PROV), db_path="catalog.db",
              physical=True, overlay_fn=_overlay)
    kw.update(over)
    return _pca(histograms, **kw)


def test_pca_provenance_records_grid_physical_caption_and_parent_entries():
    window = _pca_window()
    prov = window.export_provenance()
    window.close()
    assert prov["window"] == "pca" and prov["physical"] is True
    assert (prov["x_bins"], prov["f_bins"]) == (5, 4)
    assert prov["x_range"] == [1.25, 3.75] and prov["f_range"] == [-2.0, 6.0]
    assert prov["caption"] == "pass | 5x4"
    assert prov["population"] == "pass" and prov["n_events"] == 6


def test_pca_total_popout_inherits_labels_caption_physical_and_provenance():
    window = _pca_window()
    window._open_total_popout()
    popout = window._popout_wins[-1]
    prov = popout.export_provenance()
    has_overlay = popout._overlay_panel is not None
    popout.close()
    window.close()
    assert prov["view_title"] == "Total  (n=6)"
    assert prov["source"] == "pca_total" and prov["population"] == "pass"
    assert (prov["x_label"], prov["f_label"]) == ("Ext", "Frc")
    assert (prov["x_unit"], prov["f_unit"]) == ("nm", "pN")
    assert prov["caption"] == "pass | 5x4" and prov["physical"] is True
    assert prov["x_range"] == [1.25, 3.75] and prov["f_range"] == [-2.0, 6.0]
    assert has_overlay


def test_pca_cluster_popout_records_the_cluster_and_its_paths():
    window = _pca_window()
    window._run_kmeans()
    window._open_cluster_popout(0)
    popout = window._popout_wins[-1]
    prov = popout.export_provenance()
    n_paths = len(_paths_of(popout))
    popout.close()
    window.close()
    assert prov["source"] == "pca_cluster" and prov["cluster"] == 0
    assert prov["view_title"].startswith("Cluster 0  (n=")
    assert prov["view_title"] == f"Cluster 0  (n={n_paths})"
    assert 0 < n_paths < 6
    assert prov["population"] == "pass" and prov["physical"] is True


# -- gmm panes -----------------------------------------------------------------

def _gmm_panes(paths=("a.ibw", "b.ibw", "c.ibw")):
    rng = np.random.default_rng(1)
    xy = rng.normal([80.0, 60.0], [3.0, 4.0], size=(len(paths), 2))
    labels = ("Length", "Force", "nm", "pN")
    scatter = _scatter(xy, labels, "pass | gmm")
    pane = _gmm_model(scatter, xy, labels,
                      ("catalog.db", list(paths), "pass | gmm"))
    return scatter, pane


def test_gmm_scatter_axes_carry_the_labels_and_units():
    scatter, _ = _gmm_panes()
    bottom, left = scatter._plot.getAxis("bottom"), scatter._plot.getAxis("left")
    assert (bottom.labelText, bottom.labelUnits) == ("Length", "nm")
    assert (left.labelText, left.labelUnits) == ("Force", "pN")


def test_gmm_model_pane_provenance_names_the_variables_and_caption():
    _, pane = _gmm_panes()
    prov = pane.export_provenance()
    assert prov["window"] == "gmm_fit"
    assert (prov["x_variable"], prov["y_variable"]) == ("Length", "Force")
    assert prov["caption"] == "pass | gmm"


def test_gmm_model_pane_keeps_database_and_paths():
    _, pane = _gmm_panes()
    assert _db_of(pane) == "catalog.db"
    assert list(_paths_of(pane)) == ["a.ibw", "b.ibw", "c.ibw"]


# -- dist fit model pane -------------------------------------------------------

def _dist_panes():
    data = np.array([1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 2.2, 2.8])
    plot = _PlotPane(data, "Rupture force", "pN", caption="pass | dist")
    pane = _dist_model(plot, "Rupture force", "pN",
                       ("catalog.db", [f"{i}.ibw" for i in range(len(data))],
                        "pass | dist"))
    return plot, pane


def test_dist_model_pane_provenance_names_the_variable_units_and_caption():
    _, pane = _dist_panes()
    prov = pane.export_provenance()
    assert prov["window"] == "dist_fit"
    assert (prov["variable"], prov["units"]) == ("Rupture force", "pN")
    assert prov["caption"] == "pass | dist"


def test_dist_model_pane_keeps_database_and_paths():
    _, pane = _dist_panes()
    assert _db_of(pane) == "catalog.db"
    assert len(_paths_of(pane)) == 8
