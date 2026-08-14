"""Focused contracts for mean-curve coordinates and trace overlays."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication

from smfs_catalog.mean_curve_window import MeanCurveWindow, _bin_centres
from smfs_catalog.trace_overlay_panel import TraceOverlayPanel


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_bin_coordinates_are_histogram_centres():
    assert _bin_centres((-2.0, 6.0), 4) == pytest.approx([-1.0, 1.0, 3.0, 5.0])
    assert _bin_centres((10.0, 14.0), 1) == pytest.approx([12.0])


def test_trace_overlay_preserves_acquisition_order_and_stable_identity(qapp):
    import pyqtgraph as pg

    plot = pg.PlotWidget()
    traces = {
        "b.ibw": (np.array([2.0, 0.0, 1.0]), np.array([20.0, 0.0, 10.0])),
        "a.ibw": (np.array([3.0, 1.0]), np.array([30.0, 10.0])),
    }
    panel = TraceOverlayPanel(plot, traces.get)
    panel.set_paths(traces)

    panel._checks["b.ibw"].setChecked(True)
    _, top = panel._items["b.ibw"]
    assert top.xData == pytest.approx(traces["b.ibw"][0])
    assert top.yData == pytest.approx(traces["b.ibw"][1])
    original_pen = top.opts["pen"]

    panel._checks["a.ibw"].setChecked(True)
    panel._checks["b.ibw"].setChecked(False)
    panel._checks["b.ibw"].setChecked(True)
    reticked_pen = panel._items["b.ibw"][1].opts["pen"]
    assert reticked_pen.color() == original_pen.color()
    assert reticked_pen.style() == original_pen.style()

    panel.close()
    plot.close()


def test_input_changes_invalidate_wlc_and_provenance_records_controls(qapp):
    counts = np.ones((5, 6), dtype=float)
    window = MeanCurveWindow(
        title="test",
        display=np.sqrt(counts),
        counts=counts,
        auto_max=1.0,
        z_pct=100,
        lut=np.zeros((256, 4), dtype=np.uint8),
        x_range=(0.0, 10.0),
        f_range=(-3.0, 3.0),
        x_label="Extension",
        f_label="Force",
        physical=True,
    )
    window._wlc_fit = {"sentinel": True}
    window._wlc_curve = (np.array([]), np.array([]), None)
    window._region.setRegion((3.0, 8.0))
    assert window._wlc_fit is None
    assert window._wlc_curve is None

    provenance = window.export_provenance()
    assert provenance["wlc_fit_region"] == pytest.approx([3.0, 8.0])
    assert provenance["corner_mask_position"] == pytest.approx(
        [window._corner.pos().x(), window._corner.pos().y()]
    )
    assert provenance["ridge_estimator"] == "per-x-bin Gaussian centre"

    window.close()
