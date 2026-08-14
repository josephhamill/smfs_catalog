"""Direct contracts for the dependency-light numerical preprocessing core."""

from types import SimpleNamespace

import numpy as np
import pytest

from smfs_catalog.signal_processing import (
    DecomposedCurve,
    _ms_to_pts,
    bessel_decompose,
    filter_bandwidth_conflict,
    find_begin_in_contact,
    find_end_in_contact,
    fit_approach_invols,
    fit_retract_baseline,
)


def _curve(defl: np.ndarray, piezo: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(defl_retr=defl, piezo_retr=piezo)


def _decomposed(high_appr: np.ndarray, high_retr: np.ndarray) -> DecomposedCurve:
    return DecomposedCurve(
        low_appr=np.zeros_like(high_appr), high_appr=high_appr,
        low_retr=np.zeros_like(high_retr), high_retr=high_retr,
        sample_rate_hz=1_000.0,
    )


def test_bessel_decomposition_reconstructs_the_input() -> None:
    rng = np.random.default_rng(20260810)
    signal = rng.normal(size=1_000)
    low, high = bessel_decompose(signal, sample_rate_hz=10_000.0, cutoff_hz=1_000.0)
    np.testing.assert_allclose(low + high, signal, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize(
    "signal",
    [np.array([]), np.array([1.0]), np.zeros((2, 2)), np.array([0.0, np.nan])],
)
def test_bessel_decomposition_rejects_malformed_signals(signal: np.ndarray) -> None:
    with pytest.raises(ValueError):
        bessel_decompose(signal, sample_rate_hz=10_000.0, cutoff_hz=1_000.0)


def test_baseline_fit_clamps_anchor_indices_to_short_curve() -> None:
    piezo = np.arange(5.0)
    fit = fit_retract_baseline(_curve(2.0 * piezo + 3.0, piezo))
    assert (fit.fit_lo_idx, fit.fit_hi_idx) == (0, 5)
    assert fit.offset == pytest.approx(7.0)
    assert fit.slope == pytest.approx(2.0)
    assert fit.intercept == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("defl", "piezo"),
    [(np.array([]), np.array([])), (np.arange(3.0), np.arange(2.0)),
     (np.array([0.0, np.nan]), np.arange(2.0))],
)
def test_baseline_fit_rejects_malformed_curves(defl: np.ndarray, piezo: np.ndarray) -> None:
    with pytest.raises(ValueError):
        fit_retract_baseline(_curve(defl, piezo))


def test_invols_fit_and_degenerate_window_have_unambiguous_indices() -> None:
    piezo = np.arange(10.0)
    fit = fit_approach_invols(-2.0 * piezo + 4.0, piezo, offset_pts=1, window_pts=5)
    assert (fit.fit_lo_idx, fit.fit_hi_idx) == (4, 9)
    assert fit.slope == pytest.approx(-2.0)

    failed = fit_approach_invols(piezo, piezo, offset_pts=0, window_pts=20)
    assert np.isnan(failed.slope)
    assert failed.fit_lo_idx == failed.fit_hi_idx == len(piezo)


def test_contact_detectors_find_crossings_and_keep_documented_fallbacks() -> None:
    noisy = np.tile(np.array([-1.0, 1.0]), 10)
    quiet = np.zeros(20)
    dc = _decomposed(np.concatenate((noisy, quiet)), np.concatenate((quiet, noisy)))

    begin, _ = find_begin_in_contact(dc, threshold=0.1, trim_pts=0)
    end, _ = find_end_in_contact(dc, threshold=0.1, trim_pts=0)
    assert 15 <= begin <= 25
    assert 15 <= end <= 25

    quiet_dc = _decomposed(quiet, quiet)
    assert find_begin_in_contact(quiet_dc, threshold=0.1, trim_pts=0)[0] == len(quiet) - 1
    assert find_end_in_contact(quiet_dc, threshold=0.1, trim_pts=0)[0] == len(quiet) - 1


def test_contact_detectors_reject_empty_arrays_and_invalid_parameters() -> None:
    empty = _decomposed(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        find_begin_in_contact(empty)
    with pytest.raises(ValueError):
        find_end_in_contact(empty)

    dc = _decomposed(np.zeros(10), np.zeros(10))
    with pytest.raises(ValueError):
        find_begin_in_contact(dc, trim_pts=-1)
    with pytest.raises(ValueError):
        find_end_in_contact(dc, threshold=-1.0)


def test_duration_conversion_and_bandwidth_unknowns_are_explicit() -> None:
    assert _ms_to_pts(1.0, 10_000.0) == 11
    for invalid in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            _ms_to_pts(invalid, 10_000.0)
        assert filter_bandwidth_conflict(invalid, 2_000.0) is False
        assert filter_bandwidth_conflict(2_000.0, invalid) is False
