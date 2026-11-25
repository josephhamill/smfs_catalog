# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
The synthetic curves in curve_factory are fit to characterise the pass-1
analysis code on: deterministic, and each variant does what its name says when
run through the real functions (analyse_curve, compute_curve_events_coords,
fit_segments inside it) under default parameters and all three rupture
algorithms.  A snapshot taken on curves where nothing is detected would pin
nothing, so these checks pin that contact is found, events are detected and
segments are fitted.
"""
from functools import lru_cache

import numpy as np
import pytest

import curve_factory as cf
from smfs_catalog import curve_analysis as ca
from smfs_catalog import roi_pipeline as rp
from smfs_catalog.analysis_params import AnalysisParams
from smfs_catalog.rupture_algorithms import RUPTURE_ALGORITHMS

ALGORITHMS = list(RUPTURE_ALGORITHMS)
# variant -> number of ruptures it is built to have
EXPECTED_RUPTURES = {"one_rupture": 1, "two_rupture": 2, "flat": 0, "noisy_two_rupture": 2,
                     "two_outer_events": 2}
# variant -> number of ROIs (outer events) the detector finds; the others have one
# ROI when they have ruptures at all
EXPECTED_ROIS = {"one_rupture": 1, "two_rupture": 1, "flat": 0, "noisy_two_rupture": 1,
                 "two_outer_events": 2}
ARRAY_FIELDS = ("piezo_appr", "defl_appr", "piezo_retr", "defl_retr")


@lru_cache(maxsize=None)
def _analyse(variant: str, algorithm: str):
    """(CurveResult, CurveEventsResult) from the real code, default parameters."""
    params = AnalysisParams().with_update(
        "rupture_algorithm_idx", ALGORITHMS.index(algorithm))
    curve = cf.make_curve(variant)
    verdict, _ = ca.analyse_curve(curve, ca.pipeline_params_from(params))
    coords = rp.compute_curve_events_coords(curve, rp.event_params_from(params))
    return verdict, coords


def _ruptures(coords):
    return [r for roi in coords.events.rois for r in roi.ruptures]


def _segments(coords):
    return [s for roi in coords.events.rois for s in roi.segments]


def test_the_variants_are_the_plans_four_plus_the_two_outer_event_curve():
    assert set(cf.VARIANTS) == set(EXPECTED_RUPTURES)


@pytest.mark.parametrize("variant", list(EXPECTED_RUPTURES))
def test_two_calls_give_array_equal_curves(variant):
    first, second = cf.make_curve(variant), cf.make_curve(variant)
    for name in ARRAY_FIELDS:
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert (first.spring_constant, first.sample_rate_hz) == (
        second.spring_constant, second.sample_rate_hz)


def test_noise_is_seeded_not_global():
    np.random.seed(123)
    before = cf.make_curve("noisy_two_rupture")
    np.random.seed(456)
    after = cf.make_curve("noisy_two_rupture")
    np.testing.assert_array_equal(before.defl_retr, after.defl_retr)


def test_noisy_variant_really_is_noisier_than_the_clean_one():
    clean, noisy = cf.make_curve("two_rupture"), cf.make_curve("noisy_two_rupture")
    far_clean, far_noisy = clean.defl_retr[-2000:], noisy.defl_retr[-2000:]
    assert np.std(far_noisy) > 2.0 * np.std(far_clean)


@pytest.mark.parametrize("variant", list(EXPECTED_RUPTURES))
def test_curve_has_both_halves_a_spring_constant_and_a_sample_rate(variant):
    curve = cf.make_curve(variant)
    assert curve.piezo_appr.size == curve.defl_appr.size > 1000
    assert curve.piezo_retr.size == curve.defl_retr.size > 1000
    assert curve.spring_constant > 0
    assert curve.sample_rate_hz > 0
    assert np.all(np.isfinite(curve.defl_appr)) and np.all(np.isfinite(curve.defl_retr))


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("variant", list(EXPECTED_RUPTURES))
def test_contact_and_snapoff_are_found(variant, algorithm):
    verdict, _ = _analyse(variant, algorithm)
    assert np.isfinite(verdict.contact_z) and np.isfinite(verdict.snapoff_z)
    assert np.isfinite(verdict.offset)
    assert verdict.invols_slope == pytest.approx(cf.INVOLS, rel=0.05)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("variant", list(EXPECTED_RUPTURES))
def test_variant_has_the_number_of_ruptures_its_name_says(variant, algorithm):
    verdict, coords = _analyse(variant, algorithm)
    want = EXPECTED_RUPTURES[variant]
    assert len(_ruptures(coords)) == want
    assert verdict.event is (want > 0)
    assert len(coords.events.rois) == EXPECTED_ROIS[variant]


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize(
    "variant", ["one_rupture", "two_rupture", "noisy_two_rupture", "two_outer_events"])
def test_event_variants_get_every_segment_fitted_with_a_force(variant, algorithm):
    _, coords = _analyse(variant, algorithm)
    segments = _segments(coords)
    assert len(segments) == EXPECTED_RUPTURES[variant]
    for seg in segments:
        assert seg.fit_status in ("fit_available", "review")
        assert seg.l_c_nm is not None and np.isfinite(seg.l_c_nm)
    for rupture in _ruptures(coords):
        assert rupture.force_pN is not None and rupture.force_pN > 50.0


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_fitted_contour_lengths_match_the_chains_the_curve_was_built_from(algorithm):
    _, coords = _analyse("two_rupture", algorithm)
    fitted = [s.l_c_nm for s in _segments(coords)]
    assert fitted == pytest.approx(list(cf.VARIANTS["two_rupture"].contours_nm), rel=0.05)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_flat_variant_has_no_segments_to_fit(algorithm):
    _, coords = _analyse("flat", algorithm)
    assert _segments(coords) == []


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_two_outer_events_variant_has_one_rupture_in_each_of_two_rois(algorithm):
    _, coords = _analyse("two_outer_events", algorithm)
    assert [len(roi.ruptures) for roi in coords.events.rois] == [1, 1]
    first, second = (roi.ruptures[0] for roi in coords.events.rois)
    assert first.idx < second.idx


def test_the_lever_relaxes_between_the_two_outer_events():
    """The gap is real: the retract between the breaks is back near zero force."""
    spec = cf.VARIANTS["two_outer_events"]
    curve = cf.make_curve("two_outer_events")
    z = curve.piezo_retr
    first_break = spec.start_nm + 0.88 * spec.contours_nm[0]
    gap = (z > first_break + 40.0) & (z < first_break + spec.gap_nm)
    assert gap.any()
    assert np.max(np.abs(curve.defl_retr[gap])) < 5.0 * spec.free_sigma_nm
