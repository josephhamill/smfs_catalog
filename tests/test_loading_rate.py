# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Guard: the loading ramp has ONE slope, reported in the two presentations the
field uses, and they agree.

A rupture force is not a property of a bond — the same bond breaks at higher
force when pulled faster — so the force is only comparable between curves that
were loaded at comparable rates.  That makes the rate a number the rest of the
analysis leans on, and these checks pin the three ways it can quietly go wrong:

  * the two presentations drifting apart, so rate != stiffness * velocity and
    a cohort's Bell-Evans axis disagrees with its own stiffness column;
  * the window creeping down the ramp into the WLC's curvature, which biases
    the slope low without failing anything;
  * a missing sample clock being papered over with a guess instead of leaving
    the rate absent.

Run standalone:

    python tests/test_loading_rate.py
"""

from __future__ import annotations

import numpy as np

from smfs_catalog.roi_events import (
    RAMP_SLOPE_FORCE_FRAC,
    RAMP_SLOPE_MIN_PTS,
    ramp_loading_slopes,
    ramp_slope_window,
)


def _slopes(force, piezo, lo, peak_idx, rate_hz, frac=RAMP_SLOPE_FORCE_FRAC):
    """Window then fit, in that order — the sequence fit_segments runs, so the
    bounds these checks exercise are the bounds it stores on the segment."""
    win = ramp_slope_window(force, lo, peak_idx, frac)
    if win is None:
        return None, None, None, None, None
    return ramp_loading_slopes(force, piezo, win[0], win[1], rate_hz)

# A constant-velocity pull: 10 kHz sampling, 1000 nm/s piezo, force rising
# linearly to 100 pN.  Chosen so the true rate is exactly representable.
RATE_HZ  = 10_000.0
VEL_NM_S = 1000.0
N_PTS    = 200
F_PEAK   = 100.0


def _linear_ramp() -> tuple[np.ndarray, np.ndarray]:
    """(force pN, piezo nm) for a straight ramp at constant velocity."""
    force = np.linspace(0.0, F_PEAK, N_PTS)
    piezo = np.arange(N_PTS, dtype=float) / RATE_HZ * VEL_NM_S
    return force, piezo


def test_both_presentations_recover_the_true_slope():
    """The fitted slopes are the ramp's actual slopes, not approximations."""
    force, piezo = _linear_ramp()
    rate, _, stiff, _, _ = _slopes(
        force, piezo, 0, N_PTS - 1, RATE_HZ, RAMP_SLOPE_FORCE_FRAC,
    )
    # dF per sample * samples per second, and dF per sample / nm per sample.
    df = F_PEAK / (N_PTS - 1)
    assert np.isclose(rate, df * RATE_HZ), f"dF/dt wrong: {rate}"
    assert np.isclose(stiff, df / (VEL_NM_S / RATE_HZ)), f"dF/dz wrong: {stiff}"


def test_the_two_presentations_agree():
    """rate == stiffness * velocity.

    This identity is the whole reason the displacement slope is taken against
    PIEZO travel.  Against the deflection-corrected extension axis the WLC fits
    use, the same regression measures the molecule's stiffness alone — the
    cantilever having been subtracted out — and the product silently stops
    being the loading rate."""
    force, piezo = _linear_ramp()
    rate, _, stiff, _, _ = _slopes(
        force, piezo, 0, N_PTS - 1, RATE_HZ, RAMP_SLOPE_FORCE_FRAC,
    )
    assert np.isclose(stiff * VEL_NM_S, rate), (
        f"stiffness*v = {stiff * VEL_NM_S} but rate = {rate}"
    )


def test_window_excludes_the_curved_foot_of_the_ramp():
    """The fit takes the straight top, not the whole rise.

    A real ramp is a WLC curve: shallow while the chain has slack, steepening
    as it runs out.  Fitting the whole thing averages the shallow part in and
    reports a rate lower than the one the bond actually felt."""
    # Shallow first half, steep second half — a caricature of WLC curvature.
    force = np.concatenate([
        np.linspace(0.0, F_PEAK / 2.0, N_PTS // 2),        # slope s
        np.linspace(F_PEAK / 2.0, F_PEAK, N_PTS // 2)[1:],  # slope 3s
    ])
    piezo = np.linspace(0.0, 300.0, force.size)
    rate, _, _, _, _ = _slopes(
        force, piezo, 0, force.size - 1, RATE_HZ, RAMP_SLOPE_FORCE_FRAC,
    )
    whole_ramp = np.polyfit(
        np.arange(force.size) / RATE_HZ, force, 1,
    )[0]
    assert rate > whole_ramp, (
        f"windowed slope {rate} should exceed the whole-ramp slope "
        f"{whole_ramp}; the window has leaked into the curved foot"
    )


def test_the_window_is_the_top_of_the_ramp():
    """The bounds cover force from frac*peak up to the peak, and no further."""
    force, piezo = _linear_ramp()
    start, stop = ramp_slope_window(force, 0, N_PTS - 1, RAMP_SLOPE_FORCE_FRAC)
    assert stop == N_PTS - 1, "the window ends on the peak"
    assert force[start] >= RAMP_SLOPE_FORCE_FRAC * F_PEAK
    assert force[start - 1] < RAMP_SLOPE_FORCE_FRAC * F_PEAK, (
        "the sample before the window must be below the floor, or the walk-back "
        "stopped early"
    )


def test_the_stored_bounds_are_the_fitted_bounds():
    """A segment's rate_lo_idx/rate_hi_idx name the samples its rate came from.

    This is what stored bounds buy over recomputed ones: fitting the stored
    range must reproduce the stored slope exactly.  If the two ever disagree,
    anything drawing that window is describing a fit that did not run."""
    force, piezo = _linear_ramp()
    win = ramp_slope_window(force, 0, N_PTS - 1, RAMP_SLOPE_FORCE_FRAC)
    direct = ramp_loading_slopes(force, piezo, win[0], win[1], RATE_HZ)
    assert direct == _slopes(force, piezo, 0, N_PTS - 1, RATE_HZ)


def test_a_short_window_reports_nothing():
    """Under the point floor there is no slope worth quoting."""
    force, piezo = _linear_ramp()
    lo = N_PTS - RAMP_SLOPE_MIN_PTS + 1
    assert _slopes(
        force, piezo, lo, N_PTS - 1, RATE_HZ, RAMP_SLOPE_FORCE_FRAC,
    ) == (None, None, None, None, None)


def test_a_missing_sample_clock_takes_only_the_rate():
    """No time axis means no dF/dt — and no excuse to drop dF/dz, which never
    needed one.  A curve whose wave note stated no sample rate still reports
    its stiffness rather than going blank in both columns."""
    force, piezo = _linear_ramp()
    rate, rate_err, stiff, stiff_err, _ = _slopes(
        force, piezo, 0, N_PTS - 1, 0.0, RAMP_SLOPE_FORCE_FRAC,
    )
    assert rate is None, "a zero sample rate must not produce a time slope"
    assert rate_err is None, "no slope means no error bar on it either"
    assert stiff is not None, "dF/dz does not depend on the sample clock"
    assert stiff_err is not None, "dF/dz reports its own uncertainty"


def test_a_flat_ramp_reports_nothing():
    """Constant piezo and zero force give no slope, not a divide-by-zero."""
    flat = np.zeros(N_PTS)
    assert _slopes(
        flat, flat, 0, N_PTS - 1, RATE_HZ, RAMP_SLOPE_FORCE_FRAC,
    ) == (None, None, None, None, None)


def test_a_clean_ramp_has_an_error_far_below_its_slope():
    """A real ramp's uncertainty is a small fraction of its rate."""
    force, piezo = _linear_ramp()
    rate, rate_err, _, _, _ = _slopes(
        force, piezo, 0, N_PTS - 1, RATE_HZ, RAMP_SLOPE_FORCE_FRAC,
    )
    assert rate_err < 0.01 * rate, (
        f"a straight ramp should be pinned down: {rate} +/- {rate_err}"
    )


def test_a_trendless_window_is_disqualified_by_its_error():
    """The error, not the sign, is what marks an unmeasurable rate.

    A window whose force carries no trend the regression can find yields a
    slope that is noise: its confidence interval straddles zero, and the sign
    it happens to land on is incidental.  Real curves do this — an 8.5 ms
    window whose force rose 34 pN end to end returned -179 pN/s with an error
    of 108, because an oscillation inside it swamped the rise.  Filtering on
    sign catches that one and passes its mirror image at +179, which says
    exactly as little; the error catches both."""
    rng = np.random.default_rng(0)
    n = 400
    force = F_PEAK + rng.normal(0.0, 5.0, n)
    force[-1] = force.max() + 1.0          # keep the peak at the window's end
    piezo = np.arange(n, dtype=float) / RATE_HZ * VEL_NM_S
    rate, rate_err, _, _, _ = _slopes(
        force, piezo, 0, n - 1, RATE_HZ, RAMP_SLOPE_FORCE_FRAC,
    )
    assert abs(rate) < 2.0 * rate_err, (
        f"a trendless window should not look measured: {rate} +/- {rate_err}"
    )


def _main() -> int:
    checks = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in checks:
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {fn.__name__}\n       {e}\n")
        else:
            print(f"[ok]   {fn.__name__}")
    print()
    if failed:
        print(f"{failed} of {len(checks)} loading-rate checks failed.")
    else:
        print(f"All {len(checks)} loading-rate checks pass.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
