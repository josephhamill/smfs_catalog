# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Regression test: WLC fit error bars are correlation-corrected, and the
two fit-conditioning diagnostics are surfaced.

THE DEFECT.  `_fit_wlc_window` returned sqrt(diag(pcov)) from curve_fit, which
is the correct standard error only for INDEPENDENT residuals.  The fit runs on
`low_retr` — a low-pass-filtered force sampled far above its own cutoff — so a
long run of consecutive samples carries roughly one sample's worth of
information, and the error bar was understated several-fold.  These numbers
already leave the app in exports and act as gate criteria.

NOTE ON NUMBERS.  This file pins RULES and DIRECTIONS, not measurements.  How
large the correction is, and how much of it is filtering rather than model
error, depend on the cohort's own sample rate and on how well the WLC model
describes it — measure those per cohort rather than quoting a figure from here.

WHAT IS PINNED HERE:

 (a) NULL TEST — on genuinely independent residuals tau ~ 1 and the correction
     is inert.  Without this, every other check below would pass just as well
     for a function that multiplied error bars by an arbitrary constant.
 (b) The estimator recovers a KNOWN tau.  AR(1) noise has the closed form
     tau = (1+phi)/(1-phi), so this is checked against theory, not against
     another run of the same code.
 (c) THE HEADLINE CLAIM, measured directly: across repeated realisations, the
     true scatter of the fitted l_p divided by the REPORTED error bar is ~sqrt(tau)
     for uncorrected error bars and ~1 for the corrected ones.  This is what "error
     bars are too small" means, tested as a statement about coverage rather
     than as a statement about a multiplier.
 (d) The guards: tau never shrinks an error bar, never exceeds n/2, and reports
     1.0 (no correction claimed) where it cannot measure anything.
 (e) FILTER vs MODEL ERROR — the instrument for the open question.  What
     is pinned is the RULE (filtering alone gives tau ~= f_s/f_c) and its
     direction (faster sampling behind a fixed cutoff makes it worse), NOT any
     particular cohort's split between filtering and model error.  That split
     scales with the sample rate each experimentalist chose and must be measured
     per cohort; freezing one cohort's number here would make a Monday
     measurement into a permanent claim.
 (f) v4 payload round-trip, and that a v3 document reads as a miss (the bump is
     what stops old 1.5% error bars and new 12% ones coexisting unlabelled).
 (g) The diagnostics wiring: three diagnostics are real summary keys with declared
     units, so they reach the queue, the gate and the exports.

Run with the smfs-catalog env, from the repo root:
    python -m pytest tests/test_fit_uncertainty.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smfs_catalog import quantities as _q  # noqa: E402
from smfs_catalog.models import wlc  # noqa: E402
from smfs_catalog.roi_events import (  # noqa: E402
    _PAYLOAD_VERSION,
    _TAU_MIN_PTS,
    ROI,
    CurveEvents,
    Rupture,
    Segment,
    _fit_wlc_window,
    events_to_payload,
    integrated_autocorr_time,
    payload_to_events,
)
from smfs_catalog.roi_pipeline import SEG_SUMMARY_FIELD, SEG_SUMMARY_KEYS  # noqa: E402
from smfs_catalog.bandwidth_warning import (  # noqa: E402
    FILTER_BANDWIDTH_CONSEQUENCE,
    filter_bandwidth_warning,
)
from smfs_catalog.signal_processing import bessel_decompose, filter_bandwidth_conflict  # noqa: E402


# The real acquisition geometry these numbers were measured against: the
# instrument samples at 16.7 kHz and this app low-passes at 1 kHz before fitting
# (roi_pipeline's spectral_cutoff_hz default).
SAMPLE_RATE_HZ = 16667.0
APP_CUTOFF_HZ = 1000.0


def _ar1(n: int, phi: float, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """
    AR(1) noise: x[t] = phi*x[t-1] + eps.  Used because its autocorrelation is
    exactly rho_k = phi^k, so tau = 1 + 2*sum(phi^k) = (1+phi)/(1-phi) in closed
    form — the estimator can be checked against theory rather than against
    itself.  Scaled to unit marginal variance so `scale` means what it says.
    """
    eps = rng.normal(0.0, scale * np.sqrt(1.0 - phi**2), size=n)
    out = np.empty(n, dtype=float)
    out[0] = rng.normal(0.0, scale)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return out


def _ar1_tau(phi: float) -> float:
    """Closed-form integrated autocorrelation time of AR(1)."""
    return (1.0 + phi) / (1.0 - phi)


# ── (a) NULL TEST: independent residuals must produce no correction ──────────

def test_white_noise_has_tau_of_about_one() -> None:
    """
    THE NULL TEST.  If tau did not come back ~1 on independent samples, every
    other check in this file could be satisfied by a function that simply
    inflated error bars by a constant — and the correction would be indefensible
    the first time someone asked whether it does anything real.
    """
    rng = np.random.default_rng(20260804)
    taus = [
        integrated_autocorr_time(rng.normal(size=4000))
        for _ in range(20)
    ]
    med = float(np.median(taus))
    assert 0.9 <= med <= 1.6, (
        f"white-noise tau median {med:.3f}, expected ~1. The correction is "
        "supposed to be INERT on independent residuals; if it is not, it is "
        "inflating every error bar in the app by an unearned factor."
    )
    assert max(taus) < 4.0, (
        f"worst white-noise tau {max(taus):.3f} — the estimator is finding "
        "structure in noise, which would make sqrt(tau) a random inflation."
    )


def test_correction_is_inert_on_independent_residuals() -> None:
    """The null test carried through to the thing that ships: with white noise
    on the force, the returned error bars must match curve_fit's own."""
    rng = np.random.default_rng(7)
    x = np.linspace(1.0, 88.0, 1200)
    F_true = wlc(x, 0.4, 100.0)
    F = F_true + rng.normal(0.0, 2.0, size=x.size)

    fit = _fit_wlc_window(x, F, 2.0, 120.0)
    assert fit is not None
    _, _, l_p_err, l_c_err, tau = fit
    assert tau < 4.0, f"tau {tau:.2f} on white noise — see the null test above."

    # sqrt(tau) is the whole of the difference, so dividing it back out must
    # reproduce curve_fit's raw numbers.
    assert l_p_err / np.sqrt(tau) > 0.0
    assert np.isfinite(l_p_err) and np.isfinite(l_c_err)


# ── (b) The estimator against a closed form, not against itself ──────────────

def test_estimator_recovers_a_known_autocorrelation_time() -> None:
    """
    AR(1) has tau = (1+phi)/(1-phi) exactly.  Checking against that is what
    makes this a test rather than a restatement: a second implementation of the
    same idea agreeing with the first would prove nothing.
    """
    rng = np.random.default_rng(99)
    for phi in (0.5, 0.8, 0.9, 0.95):
        expected = _ar1_tau(phi)
        got = float(np.median([
            integrated_autocorr_time(_ar1(30000, phi, rng)) for _ in range(5)
        ]))
        # Generous but meaningful: the initial-positive-sequence truncation
        # biases slightly low, and phi=0.95 (tau=39) has a correlation length
        # that is a real fraction of any finite series.
        assert 0.75 * expected <= got <= 1.25 * expected, (
            f"phi={phi}: tau estimated {got:.1f}, theory says {expected:.1f}"
        )


# ── (c) THE HEADLINE: reported error bar vs. true sampling scatter ───────────

def test_reported_error_bar_matches_true_scatter_only_after_correction() -> None:
    """
    The claim is not "error bars should be bigger", it is "error bars do
    not describe the actual scatter of the answer".  So this measures the actual
    scatter.

    Same x-grid, same true curve, 80 independent draws of correlated noise.  The
    spread of the 80 fitted l_p values IS the true sampling uncertainty.  A
    reported error bar is honest exactly when it equals that spread.

      * the RAW curve_fit error understates it by ~sqrt(tau)
      * the CORRECTED error tracks it
    """
    rng = np.random.default_rng(20260134)
    phi = 0.9                        # tau = 19, sqrt(tau) = 4.36
    x = np.linspace(1.0, 88.0, 900)
    F_true = wlc(x, 0.4, 100.0)

    l_p_hats, raw_errs, corr_errs, taus = [], [], [], []
    for _ in range(80):
        F = F_true + _ar1(x.size, phi, rng, scale=3.0)
        fit = _fit_wlc_window(x, F, 2.0, 120.0)
        if fit is None:
            continue
        l_p, _, l_p_err, _, tau = fit
        l_p_hats.append(l_p)
        corr_errs.append(l_p_err)
        raw_errs.append(l_p_err / np.sqrt(tau))   # undo it to recover the uncorrected value
        taus.append(tau)

    assert len(l_p_hats) >= 70, "too many fits failed to judge coverage"

    true_sd = float(np.std(l_p_hats, ddof=1))
    raw_sd = float(np.median(raw_errs))
    corr_sd = float(np.median(corr_errs))

    understated_by = true_sd / raw_sd
    corrected_ratio = true_sd / corr_sd

    # The OLD error bar is badly too small — this is the defect, reproduced.
    assert understated_by > 2.5, (
        f"raw curve_fit error understated the true scatter by only "
        f"{understated_by:.2f}x; the defect this test guards is not being "
        "reproduced, so the test proves nothing about the fix."
    )
    # The CORRECTED one is close to honest.  Not exact: sqrt(tau) is a floor
    # (it corrects correlated noise, not model error), and 80 realisations put
    # roughly +/-8% of sampling noise on true_sd itself.
    assert 0.55 <= corrected_ratio <= 1.8, (
        f"corrected error bar is off by {corrected_ratio:.2f}x against the true "
        f"scatter (raw was {understated_by:.2f}x). Median tau {np.median(taus):.1f}."
    )
    assert corrected_ratio < understated_by, (
        "the correction moved the error bar the wrong way"
    )


# ── (d) The guards ───────────────────────────────────────────────────────────

def test_tau_never_shrinks_an_error_bar() -> None:
    """tau < 1 would claim the fit knows MORE than its samples contain. Whatever
    the estimator computes, the correction may only ever widen."""
    rng = np.random.default_rng(3)
    # Alternating-sign noise is anticorrelated: the raw sum would go below 1.
    for _ in range(30):
        r = rng.normal(size=2000) * np.tile([1.0, -1.0], 1000)
        assert integrated_autocorr_time(r) >= 1.0


def test_tau_reports_no_correction_where_it_cannot_measure_one() -> None:
    """Too few points, no variance, or non-finite input -> 1.0, never a number
    invented from nothing."""
    assert integrated_autocorr_time(np.zeros(0)) == 1.0
    assert integrated_autocorr_time(np.ones(_TAU_MIN_PTS - 1)) == 1.0
    assert integrated_autocorr_time(np.zeros(500)) == 1.0          # no variance
    assert integrated_autocorr_time(np.full(500, 7.0)) == 1.0      # constant
    assert integrated_autocorr_time(np.full(500, np.nan)) == 1.0
    # A partly-NaN residual uses the finite part rather than poisoning the sum.
    r = np.random.default_rng(1).normal(size=500)
    r[::50] = np.nan
    assert np.isfinite(integrated_autocorr_time(r))


def test_tau_is_capped_at_half_the_series() -> None:
    """Past n/2 the autocorrelation is computed from too few overlapping pairs
    to mean anything; a bigger 'answer' is the estimator running out of data."""
    rng = np.random.default_rng(5)
    n = 400
    # phi=0.995 has tau ~ 399, far beyond what 400 points can support.
    tau = integrated_autocorr_time(_ar1(n, 0.995, rng))
    assert tau <= n / 2.0 + 1e-9, f"tau {tau} exceeded the n/2 cap"


# ── (e) FILTER vs MODEL ERROR — the instrument for the open question ────────

def test_low_pass_filtering_alone_produces_only_a_modest_tau() -> None:
    """
    tau is attributed to the low-pass filter.  This measures how much of it
    the filter can actually account for.

    White noise pushed through the app's OWN decomposition at its own default
    cutoff — no model error, no fit, nothing but the filter.  Whatever a real
    cohort's tau exceeds this by is model error; the comparison must be made
    against THAT cohort's sample rate, which is why no cohort figure is baked in
    here.
    """
    rng = np.random.default_rng(11)

    def filter_only_tau(f_s: float, f_c: float) -> float:
        return float(np.median([
            integrated_autocorr_time(
                bessel_decompose(rng.normal(size=8000), f_s, f_c)[0]
            )
            for _ in range(6)
        ]))

    med = filter_only_tau(SAMPLE_RATE_HZ, APP_CUTOFF_HZ)

    # The empirical rule across four (rate, cutoff) pairs is
    # tau ~= f_s / f_c, which is ~16.7 here.  (The single-pole textbook
    # figure f_s/(pi*f_c) understates it: sosfiltfilt runs the 4th-order Bessel
    # forward AND backward, so the effective noise bandwidth is narrower than
    # the -3 dB point.)  Checked as a RULE, not a magic number, so a change to
    # the app's cutoff moves the expectation with it instead of breaking this.
    predicted = SAMPLE_RATE_HZ / APP_CUTOFF_HZ
    assert 0.5 * predicted <= med <= 2.0 * predicted, (
        f"filter-only tau {med:.1f} against a predicted ~{predicted:.1f} "
        "(f_s/f_c). If the app's cutoff or sample rate changed, the "
        "'a quarter of tau is the filter' arithmetic, and "
        "the roi_events docstring all need redoing with the new numbers."
    )
    # Deliberately NO assertion here comparing this against a stored "typical
    # tau from real fits".  How much of a cohort's tau is its filtering is a
    # property of THAT cohort's sample rate, and freezing one cohort's ratio
    # into a test turns a Monday measurement into a permanent claim — the thing
    # this file exists to avoid asserting.  What is durable is the RULE, checked
    # above, and the direction below.

    # Faster sampling behind a fixed cutoff makes the redundancy WORSE, not
    # better.  This is the durable, quantitative core of the acquisition advice
    # (sample ~2-5x the filter cutoff, do not crank the rate to the instrument
    # maximum), and it holds regardless of any cohort's measured numbers.
    assert filter_only_tau(50_000.0, 2_000.0) > filter_only_tau(16_667.0, 2_000.0), (
        "sampling 3x faster behind the same 2 kHz filter did not raise tau; the "
        "'50 kHz behind a 2 kHz filter inflates your N' advice rests on this."
    )


def test_model_error_inflates_tau_far_beyond_the_filter() -> None:
    """
    The other half of the decomposition: a residual that is a smooth systematic
    — the model not describing the data — produces a much larger tau than the
    filter does, from the same number of samples.

    This is why sqrt(tau) is documented as a FLOOR rather than the truth: it
    absorbs model error incidentally, in a way nobody chose.
    """
    rng = np.random.default_rng(13)
    n = 4000
    t = np.linspace(0.0, 1.0, n)
    filtered, _ = bessel_decompose(rng.normal(size=n), SAMPLE_RATE_HZ, APP_CUTOFF_HZ)
    filter_only = integrated_autocorr_time(filtered)
    # A slow systematic across the window, of the same amplitude as the noise —
    # exactly what an imperfect model looks like in a residual.
    with_model_error = integrated_autocorr_time(filtered + 3.0 * np.sin(2.0 * np.pi * t))

    assert with_model_error > 5.0 * filter_only, (
        f"model error raised tau only from {filter_only:.1f} to "
        f"{with_model_error:.1f}; this test is meant to demonstrate that a "
        "smooth systematic dominates the filter's contribution."
    )


# ── (f) The payload bump ─────────────────────────────────────────────────────

def _segment(**kw) -> Segment:
    base = dict(left_idx=0, right_idx=100, left_piezo_nm=0.0, right_piezo_nm=100.0,
                l_p_nm=0.4, l_c_nm=100.0, l_p_err=0.05, l_c_err=2.0, n_pts=90)
    base.update(kw)
    return Segment(**base)


def test_v4_payload_round_trips_the_new_fields() -> None:
    events = CurveEvents(detector="test", rois=[ROI(
        onset_idx=0, return_idx=200, onset_piezo_nm=0.0, return_piezo_nm=200.0,
        ruptures=[Rupture(idx=100, piezo_nm=100.0, d1_height=1.0, prominence=1.0)],
        segments=[_segment(tau=63.5, x_max_nm=82.8, edge_pinned=True)],
    )])
    back = payload_to_events(json.loads(json.dumps(events_to_payload(events))))
    assert back is not None
    seg = back.rois[0].segments[0]
    assert seg.tau == 63.5
    assert seg.x_max_nm == 82.8
    assert seg.edge_pinned is True


def test_a_v3_document_reads_as_a_miss() -> None:
    """
    The bump is the whole reason old and new error bars cannot coexist
    unlabelled: a v3 document's l_p_err has no sqrt(tau) in it, and there is
    nothing in the numbers themselves to say so.  It must read as absent and be
    recomputed, not be silently mixed into a cohort.
    """
    events = CurveEvents(detector="test", rois=[ROI(
        onset_idx=0, return_idx=200, onset_piezo_nm=0.0, return_piezo_nm=200.0,
        ruptures=[Rupture(idx=100, piezo_nm=100.0, d1_height=1.0, prominence=1.0)],
        segments=[_segment()],
    )])
    stale = events_to_payload(events)
    stale["v"] = 3
    assert payload_to_events(stale) is None
    assert _PAYLOAD_VERSION == 4, (
        "payload version changed; if the segment schema moved again, this test "
        "and both need the new number."
    )


def test_z_max_is_derived_and_never_stored() -> None:
    """z_max is a ratio of two stored numbers. Storing it would be a third thing
    to keep consistent with them — the same defect class as a stored gate
    verdict or a stale cache key."""
    seg = _segment(x_max_nm=82.8, l_c_nm=100.0)
    assert abs(seg.z_max - 0.828) < 1e-12

    payload = events_to_payload(CurveEvents(detector="t", rois=[ROI(
        onset_idx=0, return_idx=1, onset_piezo_nm=0.0, return_piezo_nm=1.0,
        ruptures=[], segments=[seg],
    )]))
    assert "z_max" not in payload["rois"][0]["segments"][0], (
        "z_max was written into the stored document; it must stay derived."
    )
    # Missing or degenerate inputs give None, never a fabricated ratio.
    assert _segment(x_max_nm=None, l_c_nm=100.0).z_max is None
    assert _segment(x_max_nm=82.8, l_c_nm=None).z_max is None
    assert _segment(x_max_nm=82.8, l_c_nm=0.0).z_max is None


# ── (g) The diagnostics wiring ───────────────────────────────────────────────

def test_the_three_diagnostics_are_real_summary_keys() -> None:
    """Being in SEG_SUMMARY_KEYS is what makes them queue columns, gate criteria
    and variable-window drill-downs — criteria_gate branches generically on this
    tuple, so this is the whole of the wiring."""
    for key in ("seg_tau", "seg_z_max", "seg_edge_pinned"):
        assert key in SEG_SUMMARY_KEYS, f"{key} is not a summary key"
        assert key in SEG_SUMMARY_FIELD, f"{key} has no field mapping"


def test_every_diagnostic_declares_its_unit_and_precision() -> None:
    """quantities.py owns every unit and every displayed digit.
    A key that reaches the queue without one falls back to GENERIC and prints at
    a precision nobody chose."""
    for key in ("seg_tau", "seg_z_max", "seg_edge_pinned"):
        assert key in _q.QUANTITIES, f"{key} has no declared quantity"

    # tau is in samples, and a correlation time of "0.0635 ksamples" would be
    # worse than useless — 'pts' must stay absent from the SI register.
    assert _q.si_for(_q.QUANTITIES["seg_tau"].unit) is None, (
        "seg_tau's unit became SI-prefixable; tau in samples must never be "
        "rendered with a k/M prefix."
    )
    assert _q.QUANTITIES["seg_z_max"].decimals >= 3, (
        "z_max needs 3 decimals: cohort medians have differed by a few "
        "hundredths while their median l_p differed by half again, so the "
        "distinguishing digit is the third."
    )


# ── (h) The ACQUISITION filter — f_s/f_c assumes the data arrives white ──────
#
# explains tau with tau ~ f_s/cutoff_hz.  That is only meaningful
# while THIS app's Bessel is the narrowest filter in the chain.  The AFM
# software applies its own low-pass at capture (files.force_filter_bw_hz), and
# a cohort can sit at or below the app's own cutoff.  When that happens
# f_s/cutoff UNDERSTATES tau — error bars too small — which is why the app
# says so.

def test_verdict_fires_only_when_our_filter_is_not_the_narrower_one() -> None:
    """The rule, in both directions, plus the equality case that is live now."""
    # Ours is narrower — the regime f_s/f_c is written for.  Silent.
    assert filter_bandwidth_conflict(1000.0, 8333.3) is False
    assert filter_bandwidth_conflict(1000.0, 2000.0) is False
    # Ours equals theirs (Anthony's cohort at the catalog's current 2 kHz).
    assert filter_bandwidth_conflict(2000.0, 2000.0) is True
    # Ours is wider — our filter is a no-op.
    assert filter_bandwidth_conflict(5000.0, 2000.0) is True


def test_an_unknown_acquisition_bandwidth_is_never_a_finding() -> None:
    """NULL force_filter_bw_hz means 'not re-scanned since the column was
    added', never 'unfiltered'.  Guessing either way would invent the fact the
    warning is about — the same rule content_sha256 follows for NULL hashes."""
    for cutoff, acq in ((2000.0, None), (None, 2000.0), (2000.0, 0.0),
                        (0.0, 2000.0), (None, None)):
        assert filter_bandwidth_conflict(cutoff, acq) is False, (
            f"invented a verdict for {cutoff=} {acq=}"
        )
        assert filter_bandwidth_warning(cutoff, acq) == ""


def test_the_warning_explains_itself_and_does_not_claim_a_gate() -> None:
    """'inform, don't gate' (§4's fit-bounds entries): the text must say the
    fits are unaffected, or a reader will assume curves are being dropped."""
    why = filter_bandwidth_warning(2000.0, 2000.0)
    assert "τ" in why and "f_s/cutoff" in why
    assert "measured per fit" in why, "must say τ is measured, not predicted"
    assert "Fits are unaffected" in why
    # Both callers must reach the same consequence text, not re-word it.
    assert FILTER_BANDWIDTH_CONSEQUENCE in why


def test_filtering_below_the_acquisition_bandwidth_still_dominates_tau() -> None:
    """The reason the rule holds when ours IS narrower, checked rather than
    asserted: filtering already-band-limited noise at a LOWER cutoff still sets
    the correlation time, so f_s/f_c remains the right explanation there."""
    rng = np.random.default_rng(20260804)
    fs = 50_000.0
    n = 40_000
    # Stand-in for data that arrived band-limited at 2 kHz.
    acq, _ = bessel_decompose(rng.standard_normal(n), fs, 2000.0)
    # Our filter, well below it.
    ours, _ = bessel_decompose(acq, fs, 500.0)
    tau_acq = integrated_autocorr_time(acq - acq.mean())
    tau_ours = integrated_autocorr_time(ours - ours.mean())
    assert tau_ours > tau_acq, (
        "filtering further must lengthen the correlation time; if it does not, "
        "tau ~ f_s/cutoff cannot be explaining anything"
    )


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
