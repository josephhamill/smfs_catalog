import numpy as np

from smfs_catalog.event_processor import (
    compute_physical_histogram_at,
    compute_wlc_histogram,
    phys_anchor_lc,
    phys_anchor_onset,
    phys_anchor_rupture,
    phys_anchor_snapoff,
)
from smfs_catalog.base_2dh_window import _TwoDHWindowBase, _counts_per_trace
from smfs_catalog.ledger import Ledger


def test_normalized_histogram_uses_wlc_coordinates():
    from smfs_catalog.models import _k_B, _TEMPERATURE

    kT = _k_B * _TEMPERATURE
    H = compute_wlc_histogram(
        np.array([5.0]), np.array([kT / 2.0]), l_p=2.0, l_c=10.0,
        x_bins=2, f_bins=2, x_range=(0.0, 1.0), f_range=(0.0, 2.0),
    )

    assert H.dtype == np.uint32
    np.testing.assert_array_equal(H, [[0.0, 0.0], [0.0, 1.0]])


def test_physical_histogram_subtracts_the_supplied_anchor():
    H = compute_physical_histogram_at(
        np.array([10.0, 11.0]), np.array([2.0, 3.0]), anchor=10.0,
        x_bins=2, f_bins=2, x_range=(0.0, 2.0), f_range=(2.0, 4.0),
    )

    np.testing.assert_array_equal(H, [[1.0, 0.0], [0.0, 1.0]])
    assert H.dtype == np.uint32


def test_observed_physical_anchors_do_not_require_a_wlc_fit():
    x = np.array([4.0, 5.0])
    F = np.array([1.0, 2.0])

    assert phys_anchor_onset(x, F, None, None, 50.0) == 4.0
    assert phys_anchor_snapoff(x, F, None, None, 50.0) == 0.0
    assert phys_anchor_rupture(x, F, None, None, 50.0, 5.0) == 5.0
    assert phys_anchor_lc(x, F, None, 12.0, 50.0) == 12.0


def test_incremental_2dh_uses_the_same_fit_policy_as_full_rebuild():
    class IncrementalWindow:
        add_event = _TwoDHWindowBase.add_event

        def __init__(self, needs_fit):
            self._needs_fit = needs_fit
            self._event_histograms = {}
            self.computed = self.refreshed = False

        def _stored_segment_fit(self, path):
            return None, None, 17

        def _requires_wlc_fit(self):
            return self._needs_fit

        def _load_or_compute(self, path):
            self.computed = True
            return np.ones((2, 2), dtype=np.uint32)

        def _refresh(self):
            self.refreshed = True

    observed_anchor = IncrementalWindow(needs_fit=False)
    observed_anchor.add_event("fit-free.ibw")
    assert observed_anchor.computed
    assert observed_anchor.refreshed
    assert "fit-free.ibw" in observed_anchor._event_histograms

    fit_dependent = IncrementalWindow(needs_fit=True)
    fit_dependent.add_event("needs-fit.ibw")
    assert not fit_dependent.computed
    assert not fit_dependent.refreshed
    assert "needs-fit.ibw" not in fit_dependent._event_histograms


def test_out_of_window_trace_remains_a_zero_content_histogram():
    H = compute_physical_histogram_at(
        np.array([100.0]), np.array([100.0]), anchor=0.0,
        x_bins=2, f_bins=2, x_range=(0.0, 2.0), f_range=(0.0, 2.0),
    )

    assert H is not None
    assert H.sum() == 0.0


def test_counts_per_trace_is_unchanged_when_the_cohort_is_duplicated():
    histograms = [
        np.array([[1, 2], [0, 1]], dtype=np.uint32),
        np.array([[3, 0], [2, 1]], dtype=np.uint32),
    ]
    expected = _counts_per_trace(histograms)
    np.testing.assert_array_equal(
        _counts_per_trace(histograms + histograms), expected)


def test_each_2dh_export_provenance_preserves_its_own_cohort_tally():
    def provenance(physical, kept, dropped):
        win = _TwoDHWindowBase.__new__(_TwoDHWindowBase)
        win._physical = physical
        win._population = "non_hit"
        win._align_segment = "last"
        win._event_histograms = {path: np.ones((2, 2)) for path in kept}
        win._x_bins = win._f_bins = 2
        win._x_min = win._f_min = 0.0
        win._x_max = win._f_max = 2.0
        win._z_pct = 100
        win._ledger = Ledger("2DH build", [*kept, *dropped])
        for path in dropped:
            win._ledger.drop(path, "no_fit")
        return win.export_provenance()

    normalized = provenance(False, ["a"], ["b"])
    physical = provenance(True, ["a", "b"], [])

    assert normalized["window"] == "normalized"
    assert normalized["n_events"] == 1
    assert normalized["drops"]["n_dropped"] == 1
    assert physical["window"] == "physical"
    assert physical["n_events"] == 2
    assert physical["drops"]["n_dropped"] == 0
