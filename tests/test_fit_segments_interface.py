# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Pass-1 step c5: fit_segments(curve, events, calib, settings) with two frozen
parameter objects (SegmentCalibration, FitSettings) in roi_events, and split
into named stages.  Also pins the fewer-than-5-points-of-positive-extension
guard, which no other test reaches.

The plan does not say whether the parameter objects are frozen; frozen is
chosen (immutable values, safe to share, like CacheContext).
"""
import ast
import dataclasses
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from smfs_catalog import roi_events
from smfs_catalog.roi_events import CurveEvents, ROI, Rupture, Segment, fit_segments
from size_limits import violations

# The seven fit fields, with the defaults fit_segments has today.
FIT_FIELDS = {
    "chain_model": "wlc",
    "rupture_algorithm": "low_pass_max",
    "walk_back_fit_widths": 3,
    "walk_back_noise_multiple": 5.0,
    "muggeo_fit_window_multiplier": 3,
    "muggeo_tolerance": 1e-3,
    "muggeo_restarts": True,
}
# Pinned from today's dataclasses.asdict(EventParams) (the event-map cache key).
EVENT_PARAMS_KEYS = {
    "analysis_revision", "anchor_nm", "cutoff_hz", "trim_pts", "var_window_ms",
    "thresh_retr", "window_pts", "d1_threshold", "inner_threshold", "post_mask_nm",
    "onset_thr", "detector", "prominence", "distance_pts", "chain_model",
    "rupture_algorithm", "walk_back_fit_widths", "walk_back_noise_multiple",
    "muggeo_fit_window_multiplier", "muggeo_tolerance", "muggeo_restarts",
    "invols_offset_pts", "invols_window_pts",
}


# ── interface ────────────────────────────────────────────────────────────────

def test_fit_segments_takes_curve_events_calib_settings():
    assert list(inspect.signature(fit_segments).parameters) == [
        "curve", "events", "calib", "settings"]


def test_segment_calibration_fields_and_defaults():
    cal = roi_events.SegmentCalibration(1.5, 2.5, 3.5)
    assert (cal.offset_retr, cal.invols_slope, cal.snapoff_piezo_nm) == (1.5, 2.5, 3.5)
    assert cal.low_retr is None
    assert [f.name for f in dataclasses.fields(roi_events.SegmentCalibration)] == [
        "offset_retr", "invols_slope", "snapoff_piezo_nm", "low_retr"]
    arr = np.zeros(3)
    assert roi_events.SegmentCalibration(0.0, 1.0, 0.0, low_retr=arr).low_retr is arr


def test_fit_settings_holds_exactly_the_seven_fit_fields_with_todays_defaults():
    got = {f.name: f.default for f in dataclasses.fields(roi_events.FitSettings)}
    assert got == FIT_FIELDS
    assert dataclasses.asdict(roi_events.FitSettings()) == FIT_FIELDS


def test_segment_calibration_is_a_frozen_dataclass():
    cal = roi_events.SegmentCalibration(0.0, 1.0, 0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cal.offset_retr = 5.0


def test_fit_settings_is_a_frozen_dataclass():
    fs = roi_events.FitSettings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        fs.chain_model = "other"


def test_event_params_fit_settings_matches_its_fields():
    from smfs_catalog.analysis_params import AnalysisParams
    from smfs_catalog.roi_pipeline import event_params_from
    ep = dataclasses.replace(
        event_params_from(AnalysisParams()), rupture_algorithm="muggeo_segmented",
        walk_back_fit_widths=7, walk_back_noise_multiple=2.5,
        muggeo_fit_window_multiplier=4, muggeo_tolerance=1e-2, muggeo_restarts=False)
    fs = ep.fit_settings()
    assert isinstance(fs, roi_events.FitSettings)
    assert dataclasses.asdict(fs) == {f: getattr(ep, f) for f in FIT_FIELDS}
    assert fs.rupture_algorithm == "muggeo_segmented" and fs.muggeo_restarts is False
    assert fs.walk_back_fit_widths == 7 and fs.muggeo_fit_window_multiplier == 4


def test_asdict_event_params_keys_are_unchanged():
    from smfs_catalog.analysis_params import AnalysisParams
    from smfs_catalog.roi_pipeline import event_params_from
    assert set(dataclasses.asdict(event_params_from(AnalysisParams()))) == EVENT_PARAMS_KEYS


def test_guess_and_ramp_values_are_module_constants_with_todays_values():
    # The names are the coder's to choose (any case); the values are not.
    lowered = {n.lower(): v for n, v in vars(roi_events).items()}
    assert lowered.get("guess_l_p") == 2.0
    assert lowered.get("guess_l_c") == 120.0
    assert roi_events.RAMP_SLOPE_FORCE_FRAC == 0.5


def test_guess_values_and_calibration_are_not_loose_fit_segments_parameters():
    params = inspect.signature(fit_segments).parameters
    for gone in ("guess_l_p", "guess_l_c", "ramp_force_frac", "low_retr", "chain_model"):
        assert gone not in params


# ── the fewer-than-5-points guard (pins today's behaviour) ───────────────────

N = 60


def _guard_case():
    """Two segments.  Segment 1 climbs to a force peak but sits at NEGATIVE
    extension (piezo = defl - 1), so fewer than 5 points have positive
    extension.  Segment 2 is a normal ramp that climbs past segment 1's force."""
    defl = np.zeros(N)
    defl[0:21] = np.linspace(0.0, 10.0, 21)      # segment 1 ramp, peak 10 at idx 20
    defl[25:46] = np.linspace(0.0, 14.0, 21)     # segment 2 ramp, peak 14 at idx 45
    piezo = np.arange(N, dtype=float)
    piezo[0:21] = defl[0:21] - 1.0               # extension = -1 throughout
    piezo[25:46] = defl[25:46] + 5.0 + np.arange(21) * 0.1   # extension > 0
    curve = SimpleNamespace(defl_retr=defl, piezo_retr=piezo,
                            spring_constant=1.0, sample_rate_hz=1000.0)
    ruptures = [Rupture(idx=20, piezo_nm=0.0, d1_height=1.0),
                Rupture(idx=45, piezo_nm=0.0, d1_height=1.0)]
    segments = [Segment(left_idx=0, right_idx=22, left_piezo_nm=0.0, right_piezo_nm=0.0),
                Segment(left_idx=25, right_idx=47, left_piezo_nm=0.0, right_piezo_nm=0.0)]
    events = CurveEvents(detector="test", rois=[ROI(
        onset_idx=0, return_idx=N - 1, onset_piezo_nm=0.0, return_piezo_nm=0.0,
        ruptures=ruptures, segments=segments)])
    return curve, events


def _run_fit(curve, events) -> None:
    """The one place this file calls fit_segments."""
    fit_segments(curve, events, roi_events.SegmentCalibration(
        0.0, 1.0, 0.0, low_retr=curve.defl_retr))


def test_fewer_than_five_positive_points_is_refused_and_the_next_segment_sees_its_force():
    curve, events = _guard_case()
    _run_fit(curve, events)
    seg1, seg2 = events.rois[0].segments
    rup1, rup2 = events.rois[0].ruptures
    assert seg1.fit_status == "no_fit"
    assert seg1.fit_detail == "insufficient positive extension"
    assert seg1.l_p_nm is None and seg1.x_max_nm is None
    # The refused segment still has its rupture force (located before the guard).
    assert rup1.force_pN == 10.0 and rup1.force_idx == 20
    # prev_rup = rup on this guard: segment 2 finds its isoforce crossing against
    # segment 1's force (10 pN).  With prev_rup reset to None this would be None.
    assert seg2.isoforce_x_nm is not None
    assert rup2.force_pN == 14.0


# ── source-size guard (CLAUDE.md limits), scoped to what c5 touches ──────────

# Every top-level function roi_events.py had before c5.  Any other top-level
# function is a helper c5 introduced and is held to the same limits.
PRE_C5_FUNCTIONS = {
    "detect_ruptures_threshold", "detect_ruptures_findpeaks",
    "segment_baseline_excursions", "find_outer_events", "build_curve_events",
    "_segments_from_ruptures", "ramp_force_peak", "ramp_peak_is_edge_pinned",
    "ramp_slope_window", "ramp_loading_slopes", "fit_segments",
    "_isoforce_crossing_x", "integrated_autocorr_time", "_fit_wlc_window",
    "_fit_chain_model", "events_to_payload", "payload_to_events",
}


def _functions() -> dict:
    tree = ast.parse(Path(roi_events.__file__).read_text(encoding="utf-8"))
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _new_helpers() -> dict:
    return {n: f for n, f in _functions().items() if n not in PRE_C5_FUNCTIONS}


def test_fit_segments_is_split_into_named_stages():
    assert len(_new_helpers()) >= 4


def test_fit_segments_is_within_the_size_limits():
    assert violations("fit_segments", _functions()["fit_segments"]) == []


def test_every_new_helper_is_within_the_size_limits():
    found = [v for n, f in _new_helpers().items() for v in violations(n, f)]
    assert found == []
