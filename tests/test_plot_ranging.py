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

from smfs_catalog import wlc_view_window as wvw     # noqa: E402


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
