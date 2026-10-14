"""The drawdown walk-back algorithm, and fit_segments running it.

A clean ramp with a fall of several samples has one right answer: the last
sample still on the ramp.  The status must say when an answer rests on
something other than a sample found at ramp height below a real top.
"""

from types import SimpleNamespace

import numpy as np

from smfs_catalog.drawdown_walk_back import (
    AT_TOP, DID_NOT_MOVE, FIT_CUT_SHORT, FOUND, NO_NOISE, TOP_ON_EDGE,
    drawdown_walk_back, largest_fall,
)
from smfs_catalog.roi_events import CurveEvents, ROI, Rupture, Segment, fit_segments
from smfs_catalog.rupture_algorithms import (
    RUPTURE_ALGORITHM_BY_IDX, RUPTURE_ALGORITHMS, fit_deflection,
)

LAST_ON_RAMP = 300          # the last sample before the fall
LEFT, RIGHT = 290, 319      # the d1 crossings bracketing it (width 30)


def _ramp(noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """A ramp that levels off with a small ripple, a 4-sample fall after
    LAST_ON_RAMP, then a flat level.  The ripple puts the highest sample
    before the last one, as noise does on a real ramp."""
    i = np.arange(400, dtype=float)
    f = np.minimum(0.5 * i, 140.0) + 0.5 * np.sin(i)
    top = f[LAST_ON_RAMP]
    for k, frac in enumerate((0.75, 0.5, 0.25, 0.0), start=1):
        f[LAST_ON_RAMP + k] = top * frac
    f[LAST_ON_RAMP + 5:] = 0.0
    return f + np.random.default_rng(seed).normal(0.0, noise, f.size)


def test_largest_fall_is_the_highest_then_the_lowest_after_it() -> None:
    f = np.array([10., 12., 11., 13., 5., 2., 4.])
    assert largest_fall(f, 0, 6) == (3, 5)


def test_it_stops_on_the_last_sample_of_the_ramp() -> None:
    got = drawdown_walk_back(_ramp(), 0, LEFT, RIGHT, 3, 5.0)
    assert got.idx == LAST_ON_RAMP
    assert got.status == FOUND
    assert got.top_idx < LAST_ON_RAMP
    assert got.fit_samples == 90
    assert got.bottom_idx > LAST_ON_RAMP


def test_the_search_ignores_a_larger_fall_outside_its_region() -> None:
    f = _ramp()
    f[50:60] += 500.0       # a spike and fall far to the left of the region
    assert drawdown_walk_back(f, 0, LEFT, RIGHT, 3, 5.0).idx == LAST_ON_RAMP


def test_a_short_segment_runs_on_what_it_has_and_says_so() -> None:
    got = drawdown_walk_back(_ramp(), LEFT - 20, LEFT, RIGHT, 3, 5.0)
    assert got.idx == LAST_ON_RAMP
    assert got.fit_samples == 20
    assert FIT_CUT_SHORT in got.status


def test_no_noise_means_no_answer() -> None:
    got = drawdown_walk_back(_ramp(), LEFT - 2, LEFT, RIGHT, 3, 5.0)
    assert got.idx is None and got.status == NO_NOISE


def test_the_status_names_what_the_answer_rests_on() -> None:
    # A ramp rising right up to the fall: the answer is the top itself.
    i = np.arange(400, dtype=float)
    rising = np.where(i <= LAST_ON_RAMP, 0.5 * i, 0.0)
    got = drawdown_walk_back(rising, 0, LEFT, RIGHT, 3, 5.0)
    assert got.idx == LAST_ON_RAMP and got.status == AT_TOP
    # A falling stretch: the highest sample is the first one in the region.
    falling = np.linspace(100.0, 0.0, 400)
    got = drawdown_walk_back(falling, 200, 290, 319, 3, 5.0)
    assert TOP_ON_EDGE in got.status and AT_TOP in got.status
    # A fall smaller than the margin: the bottom itself passes the test.
    f = _ramp(noise=5.0)
    f[LAST_ON_RAMP + 1:] = f[LAST_ON_RAMP] - 2.0
    assert DID_NOT_MOVE in drawdown_walk_back(f, 0, LEFT, RIGHT, 3, 50.0).status


def test_the_registry_keeps_the_low_pass_maximum_first() -> None:
    assert RUPTURE_ALGORITHM_BY_IDX[0] == "low_pass_max"
    assert RUPTURE_ALGORITHMS["drawdown_walk_back"].settings == (
        "walk_back_fit_widths", "walk_back_noise_multiple")


def _events() -> CurveEvents:
    rup = Rupture(idx=303, piezo_nm=303.0, d1_height=1.0,
                  rise_idx=LEFT, fall_idx=RIGHT)
    seg = Segment(left_idx=0, right_idx=LEFT, left_piezo_nm=0.0,
                  right_piezo_nm=float(LEFT))
    return CurveEvents(detector="test", rois=[ROI(
        onset_idx=0, return_idx=RIGHT, onset_piezo_nm=0.0,
        return_piezo_nm=float(RIGHT), ruptures=[rup], segments=[seg])])


def test_fit_segments_uses_the_algorithm_in_force() -> None:
    raw = _ramp()
    curve = SimpleNamespace(defl_retr=raw, piezo_retr=np.arange(400, dtype=float),
                            spring_constant=1.0, sample_rate_hz=1000.0)
    smooth = np.convolve(raw, np.ones(31) / 31, mode="same")

    walk = _events()
    fit_segments(curve, walk, 0.0, 1.0, 0.0, low_retr=smooth,
                 rupture_algorithm="drawdown_walk_back")
    rup, seg = walk.rois[0].ruptures[0], walk.rois[0].segments[0]
    assert rup.force_idx == LAST_ON_RAMP and seg.right_idx == LAST_ON_RAMP
    assert rup.force_pN == raw[LAST_ON_RAMP], "the saved signal's value"
    assert rup.point_status == FOUND

    old = _events()
    fit_segments(curve, old, 0.0, 1.0, 0.0, low_retr=smooth)
    rup, seg = old.rois[0].ruptures[0], old.rois[0].segments[0]
    assert rup.force_idx <= LEFT and seg.right_idx == LEFT
    assert rup.force_pN == smooth[rup.force_idx]
    assert rup.point_status is None


def test_under_the_walk_back_the_fit_and_the_slopes_ignore_the_low_pass_signal() -> None:
    """The chain fit and the loading slopes run on the raw signal, so what the
    low-pass signal holds cannot change them."""
    raw = _ramp()
    curve = SimpleNamespace(defl_retr=raw, piezo_retr=np.arange(400, dtype=float),
                            spring_constant=1.0, sample_rate_hz=1000.0)
    got = []
    for smooth in (np.convolve(raw, np.ones(31) / 31, mode="same"), raw * 0.5):
        events = _events()
        fit_segments(curve, events, 0.0, 1.0, 0.0, low_retr=smooth,
                     rupture_algorithm="drawdown_walk_back")
        seg, rup = events.rois[0].segments[0], events.rois[0].ruptures[0]
        got.append((rup.force_pN, rup.extension_nm, seg.loading_rate_pN_s,
                    seg.loading_stiffness_pN_nm, seg.rate_lo_idx, seg.l_p_nm,
                    seg.l_c_nm, seg.fit_lo_idx, seg.fit_hi_idx))
    assert got[0][2] is not None, "no loading rate was measured"
    assert got[0] == got[1]


def test_plots_draw_the_signal_each_algorithm_fits_on() -> None:
    raw, low = np.zeros(3), np.ones(3)
    assert fit_deflection("drawdown_walk_back", raw, low) is raw
    assert fit_deflection("low_pass_max", raw, low) is low
