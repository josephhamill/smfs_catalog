"""Focused contracts for PCA calculation and loading-map coordinates."""

from __future__ import annotations

import numpy as np
import pytest


pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QApplication

from smfs_catalog.pca_window import (
    PCAWindow,
    _PCA_SVD_SOLVER,
    _relative_frequency_rows,
)
from smfs_catalog.qt_utils import FixedDomainPlot


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _window() -> PCAWindow:
    rng = np.random.default_rng(7)
    histograms = {
        f"trace-{i}": rng.poisson(0.2, size=(5, 4)).astype(np.float32)
        for i in range(4)
    }
    return PCAWindow(
        histograms=histograms,
        x_bins=5,
        f_bins=4,
        x_range=(1.25, 3.75),
        f_range=(-2.0, 6.0),
    )


def test_pca_uses_documented_randomized_solver(qapp):
    window = _window()
    assert window.export_provenance()["pca_svd_solver"] == _PCA_SVD_SOLVER
    window.close()


def test_loading_plot_covers_exact_feature_space(qapp):
    window = _window()
    plot = window._make_loading_plot(0, embedded=False)

    assert plot.viewRange()[0] == pytest.approx([1.25, 3.75])
    assert plot.viewRange()[1] == pytest.approx([-2.0, 6.0])

    image = next(item for item in plot.items() if hasattr(item, "image"))
    mapped = image.mapRectToParent(image.boundingRect())
    assert mapped == QRectF(1.25, -2.0, 2.5, 8.0)

    # A deliberate zoom remains possible, but any later layout resize must
    # restore the complete loading matrix rather than crop it by widget size.
    plot.show()
    qapp.processEvents()
    plot.getViewBox().scaleBy((2.0, 2.0))
    assert plot.viewRange()[0] != pytest.approx([1.25, 3.75])
    plot.resize(plot.width() + 137, plot.height() + 91)
    qapp.processEvents()
    assert plot.viewRange()[0] == pytest.approx([1.25, 3.75])
    assert plot.viewRange()[1] == pytest.approx([-2.0, 6.0])

    # Plot titles/axes can resize the inner ViewBox without another outer
    # PlotWidget resize. This is the event ordering used by the Windows UI.
    plot.getViewBox().scaleBy((2.0, 2.0))
    inner = plot.getViewBox()
    inner.resize(inner.size().width() - 17, inner.size().height() - 11)
    qapp.processEvents()
    assert plot.viewRange()[0] == pytest.approx([1.25, 3.75])
    assert plot.viewRange()[1] == pytest.approx([-2.0, 6.0])

    plot.close()
    window.close()


def test_embedded_loading_ranges_survive_parent_window_resize(qapp):
    window = _window()
    window.show()
    qapp.processEvents()

    def assert_full_ranges():
        plots = window.findChildren(FixedDomainPlot)
        assert len(plots) == 3
        for plot in plots:
            assert plot.viewRange()[0] == pytest.approx([1.25, 3.75])
            assert plot.viewRange()[1] == pytest.approx([-2.0, 6.0])

    assert_full_ranges()
    window.resize(1900, 1000)
    qapp.processEvents()
    assert_full_ranges()
    window.resize(1100, 760)
    qapp.processEvents()
    assert_full_ranges()
    window.close()


def test_pca_layout_shrinks_without_clipping_plots(qapp):
    window = _window()
    window._caption = (
        "Hits · 29 events · seg: Last · grid: 128×128 · z-clip: 100% "
        "· align: rupture · PCA feature space: 67×105 bins "
        "· Δx: −239.0625–75 nm · F: −75–581.25 pN"
    )
    for width, height in ((1100, 760), (600, 500)):
        window.resize(width, height)
        window.show()
        qapp.processEvents()
        assert window.width() == width
        for plot in window.findChildren(FixedDomainPlot):
            assert plot.parentWidget().rect().contains(plot.geometry())
            assert plot.centralWidget.geometry().width() <= plot.viewport().width()
            assert plot.viewRange()[0] == pytest.approx([1.25, 3.75])
            assert plot.viewRange()[1] == pytest.approx([-2.0, 6.0])
    window.close()


def test_pca_component_count_is_bounded_by_live_features(qapp):
    histograms = {
        f"trace-{i}": np.array([[i + 1, 4 - i], [0, 0]], dtype=np.uint32)
        for i in range(4)
    }
    window = PCAWindow(histograms, 2, 2, (0.0, 2.0), (0.0, 2.0))
    assert window._scores.shape == (4, 2)
    window.close()


def test_pca_rejects_an_empty_feature_space(qapp):
    histograms = {
        f"trace-{i}": np.zeros((2, 2), dtype=np.float32)
        for i in range(2)
    }
    with pytest.raises(ValueError, match="no non-zero bins"):
        PCAWindow(histograms, 2, 2, (0.0, 2.0), (0.0, 2.0))


def test_pca_profiles_give_each_trace_equal_total_weight():
    counts = np.array([[1, 3], [10, 30], [0, 0]], dtype=np.uint32)
    profiles = _relative_frequency_rows(counts)
    np.testing.assert_allclose(profiles[:2], [[0.25, 0.75], [0.25, 0.75]])
    np.testing.assert_array_equal(profiles[2], [0.0, 0.0])


def test_duplicating_a_cohort_does_not_change_pca_profiles():
    counts = np.array([[1, 3], [3, 1]], dtype=np.uint32)
    original = _relative_frequency_rows(counts)
    duplicated = _relative_frequency_rows(np.vstack([counts, counts]))
    np.testing.assert_allclose(duplicated, np.vstack([original, original]))
