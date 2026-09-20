"""What each plot lets set its scale.

Autorange fits every item in a view, so whichever item has the widest extent
decides what the others look like.  In three plots that item is not the one
being read: the CI envelope dwarfs the WLC fit inside it, the contact ramp
dwarfs the retract, and one large rupture flattens every smaller d1 peak.

These are behavioural: they call the real drawing code and read the resulting
view range, so they fail if an item is re-admitted to the autorange, whatever
the spelling.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pg = pytest.importorskip("pyqtgraph")
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication            # noqa: E402

from smfs_catalog import display_roi as droi       # noqa: E402
from smfs_catalog import rawcurve_window as raw    # noqa: E402
from smfs_catalog import wlc_view_window as wvw     # noqa: E402
from smfs_catalog.curve_loader import ForceCurve    # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# ── WLC fit: the envelope is drawn, not measured against ─────────────────────

class _CiProbe:
    """Only the state _draw_fit_ci reads.  A real WlcViewWindow needs a
    database, a catalogued curve and a fit; a guard that expensive is a guard
    that gets skipped."""

    _draw_fit_ci = wvw.WlcViewWindow._draw_fit_ci

    def __init__(self):
        self._top = pg.PlotWidget()
        self._ci_chk = SimpleNamespace(isChecked=lambda: True)


def _y_range(plot) -> tuple[float, float]:
    vb = plot.getPlotItem().getViewBox()
    vb.updateAutoRange()
    return tuple(vb.viewRange()[1])


def _fitted_segment():
    """A fit whose l_c - sigma corner puts the WLC pole just past the data, so
    the envelope diverges inside the fitted range.  This is the ordinary case,
    not a contrived one: l_c_err is a fraction of a nm on a real fit."""
    return SimpleNamespace(l_p_nm=0.4, l_p_err=0.05, l_c_nm=105.0, l_c_err=4.0)


def test_the_ci_envelope_does_not_set_the_wlc_plot_scale(qapp):
    xm  = np.linspace(10.0, 100.0, 200)
    seg = _fitted_segment()
    fit = np.asarray(wvw.wlc(xm, seg.l_p_nm, seg.l_c_nm), dtype=float)

    p = _CiProbe()
    p._top.plot(xm.tolist(), fit.tolist())
    band = p._draw_fit_ci(xm, seg, (0, 0, 255))
    assert band, "no envelope drawn — the guard would pass vacuously"

    lo, hi = _y_range(p._top)
    assert hi < 3.0 * float(fit.max()), (
        f"the envelope set the scale: view reaches {hi:,.0f} pN for a fit "
        f"peaking at {fit.max():,.0f} pN")


def test_the_envelope_is_still_drawn(qapp):
    """Excluded from the autorange is not the same as absent: the band is the
    reason the plot is being looked at with the checkbox on."""
    xm  = np.linspace(10.0, 100.0, 200)
    p = _CiProbe()
    band = p._draw_fit_ci(xm, _fitted_segment(), (0, 0, 255))
    assert band
    assert band[0] in p._top.getPlotItem().items


# ── Raw curve: the contact ramp is not what the markers were drawn for ───────

def _event_curve() -> ForceCurve:
    """A ramp whose contact region reaches 25 nm and whose events sit within
    2 nm of the baseline — the ordinary proportions of a real curve (measured
    on catalog curve 134114: 34.95 nm of view for 9.81 nm of content)."""
    # Contact at low piezo, free baseline at high piezo, events in between —
    # the arrangement curve 134114 has (contact near 2250 nm, ROIs at 2340 and
    # 2549, baseline beyond 2600).
    piezo_a = np.linspace(0.0, 100.0, 400)
    defl_a  = np.where(piezo_a < 30.0, (30.0 - piezo_a) * 0.85, 0.0)
    piezo_r = np.linspace(0.0, 100.0, 400)
    defl_r  = np.where(piezo_r < 30.0, (30.0 - piezo_r) * 0.85, 0.0)
    # Two ruptures out where the tether is stretched, small against the ramp.
    defl_r[(piezo_r > 60.0) & (piezo_r < 62.0)] = -1.8
    defl_r[(piezo_r > 75.0) & (piezo_r < 77.0)] = -1.2
    return ForceCurve(
        path="curve.ibw",
        piezo_appr=piezo_a, defl_appr=defl_a,
        piezo_retr=piezo_r, defl_retr=defl_r,
        spring_constant=10.0,
    )


class _FrameProbe:
    """Only the state _frame_events and _draw_event_marker_coords read."""

    _frame_events            = raw.RawCurveWindow._frame_events
    _draw_event_marker_coords = raw.RawCurveWindow._draw_event_marker_coords
    _ramp_series             = staticmethod(raw.RawCurveWindow._ramp_series)

    def __init__(self, axes=raw._AXES[0][1]):
        self._plot = pg.PlotWidget()
        self._axes = axes
        self._drawn = _event_curve()
        self._rupture_lines: list = []
        self._onset_lines: list = []
        self._plot.plot(self._drawn.piezo_retr.tolist(),
                        self._drawn.defl_retr.tolist())
        self._plot.plot(self._drawn.piezo_appr.tolist(),
                        self._drawn.defl_appr.tolist())


_COORDS = [(61.0, 60.5), (76.0, 75.5)]


def test_the_contact_ramp_does_not_set_the_raw_curve_scale(qapp):
    p = _FrameProbe()
    loose = _y_range(p._plot)
    p._draw_event_marker_coords(_COORDS)
    framed = p._plot.getPlotItem().getViewBox().viewRange()[1]

    assert np.ptp(framed) < np.ptp(loose) / 2.0, (
        f"the ramp still sets the scale: {np.ptp(framed):.2f} nm framed "
        f"against {np.ptp(loose):.2f} nm loose")
    assert framed[1] < 5.0, "the contact ramp is still inside the view"


def test_a_curve_with_no_events_is_left_alone(qapp):
    """Most of the catalog has no ROI. Nothing is known about where to look on
    those curves, so nothing is claimed."""
    p = _FrameProbe()
    before = _y_range(p._plot)
    p._draw_event_marker_coords([])
    assert _y_range(p._plot) == before


def test_framing_is_piezo_only(qapp):
    """A mark is a piezo position. On a time axis it names a moment that was
    never measured, which is the guard _draw_persisted_overlays already applies
    to the marker lines themselves."""
    p = _FrameProbe(axes=raw._AXES[1][1])
    before = _y_range(p._plot)
    p._frame_events([61.0, 76.0])
    assert _y_range(p._plot) == before


# ── d1: the thresholds are the scale, not the tallest peak ───────────────────

class _D1Probe:
    """Only the state _frame_d1 reads."""

    _frame_d1 = droi.ROIWindow._frame_d1

    def __init__(self, outer=1.4, inner=0.946, d1=None):
        self._d1_plot = pg.PlotWidget()
        self._threshold_nm_per_nm = outer
        self._inner_threshold_nm_per_nm = inner
        if d1 is not None:
            self._d1_plot.plot(np.arange(len(d1)).tolist(), d1.tolist())


def _d1_with_one_giant_peak() -> np.ndarray:
    """Catalog curve 134115's proportions: a 14.0 nm/nm rupture over a trace
    that otherwise lives between -1.1 and about 2, against a 1.4 threshold."""
    d1 = np.random.default_rng(0).normal(0.0, 0.35, 4000)
    d1[1000:1010] = 14.0
    d1[2000:2010] = 1.9
    d1[3000:3010] = 1.6
    return np.clip(d1, -1.09, None)


def test_one_giant_peak_does_not_flatten_the_smaller_ones(qapp):
    d1 = _d1_with_one_giant_peak()
    p = _D1Probe()
    p._frame_d1(d1)
    lo, hi = p._d1_plot.getPlotItem().getViewBox().viewRange()[1]

    assert hi < float(d1.max()) / 2.0, (
        f"the tallest peak still sets the scale: view reaches {hi:.2f} for a "
        f"peak of {d1.max():.2f}")
    assert hi > p._threshold_nm_per_nm, (
        "the threshold the peaks are judged against must stay in view")


def test_framing_never_widens_the_d1_view(qapp):
    """d1 does not always hold a peak that dwarfs the rest.  Measured across
    five catalog curves, three never reach the cap, and opening the view out to
    it would spread a small signal thinner than autorange already does."""
    modest = np.clip(np.random.default_rng(1).normal(0.0, 0.4, 2000), -1.5, 3.22)
    p = _D1Probe(d1=modest)
    loose = _y_range(p._d1_plot)          # what autorange alone would show
    p._frame_d1(modest)
    framed = p._d1_plot.getPlotItem().getViewBox().viewRange()[1]

    # Spans, so pyqtgraph's own padding is on both sides of the comparison.
    assert np.ptp(framed) <= np.ptp(loose) + 1e-9, (
        f"widened the view: {np.ptp(framed):.2f} framed against "
        f"{np.ptp(loose):.2f} loose")


def test_an_empty_d1_leaves_the_panel_autoranging(qapp):
    p = _D1Probe()
    p._frame_d1(np.array([]))
    assert p._d1_plot.getPlotItem().getViewBox().autoRangeEnabled()[1]
