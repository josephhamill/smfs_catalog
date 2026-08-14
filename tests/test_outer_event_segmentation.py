import numpy as np

from smfs_catalog.roi_events import build_curve_events, find_outer_events
from smfs_catalog.roi_pipeline import (
    event_geometry_identity,
    resolve_reported_segment,
    resolve_segment_override_state,
)


def _signals():
    piezo = np.arange(120, dtype=float)
    d1 = np.zeros(120, dtype=float)
    mean_dev = np.zeros(120, dtype=float)
    # One validated event: onset boundary at 39, loaded interval 40..90,
    # inner rupture at 60 and outer/terminal rupture at 90.
    mean_dev[40:91] = -1.0
    d1[60:62] = 0.35
    d1[90:93] = 0.8
    return d1, mean_dev, piezo


def test_outer_search_marks_event_without_deciding_its_segments():
    d1, mean_dev, piezo = _signals()
    boundaries = find_outer_events(
        d1, mean_dev, piezo, lo=10, hi=110,
        onset_thr=-0.2, outer_threshold=0.5,
    )
    assert len(boundaries) == 1
    assert boundaries[0].onset_idx == 39
    assert boundaries[0].terminal_idx == 90


def test_inner_threshold_turns_one_outer_event_into_two_segments():
    d1, mean_dev, piezo = _signals()
    events = build_curve_events(
        d1, mean_dev, piezo, lo=10, hi=110,
        onset_thr=-0.2, outer_threshold=0.5, inner_threshold=0.2,
    )
    assert events.n_rois == 1
    assert [r.idx for r in events.rois[0].ruptures] == [60, 90]
    assert len(events.rois[0].segments) == 2


def test_no_inner_crossing_still_produces_one_segment():
    d1, mean_dev, piezo = _signals()
    events = build_curve_events(
        d1, mean_dev, piezo, lo=10, hi=110,
        onset_thr=-0.2, outer_threshold=0.5, inner_threshold=0.5,
    )
    assert events.n_rois == 1
    assert [r.idx for r in events.rois[0].ruptures] == [90]
    assert len(events.rois[0].segments) == 1


def test_outer_crossing_without_baseline_return_is_rejected():
    d1, mean_dev, piezo = _signals()
    mean_dev[:93] = -1.0
    boundaries = find_outer_events(
        d1, mean_dev, piezo, lo=10, hi=110,
        onset_thr=-0.2, outer_threshold=0.5,
    )
    assert boundaries == []


def test_reported_segment_policy_is_separate_and_explicit():
    assert resolve_reported_segment(3, "ultimate").segment_idx == 2
    assert resolve_reported_segment(3, "penultimate").segment_idx == 1
    one = resolve_reported_segment(1, "penultimate")
    assert (one.segment_idx, one.source) == (0, "ultimate")
    manual = resolve_reported_segment(3, "penultimate", primary_idx=0)
    assert (manual.segment_idx, manual.source) == (0, "manual_primary")


def test_unknown_reported_segment_policy_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="unknown segment selection policy"):
        resolve_reported_segment(3, "typo")


def test_manual_pick_survives_param_change_when_geometry_is_identical():
    d1, mean_dev, piezo = _signals()
    events = build_curve_events(
        d1, mean_dev, piezo, lo=10, hi=110,
        onset_thr=-0.2, outer_threshold=0.5, inner_threshold=0.2,
    )
    identity = event_geometry_identity(events)
    override = {
        "primary_segment_idx": 0,
        "secondary_segment_idx": 1,
        "params_json": identity,
    }
    state = resolve_segment_override_state(
        override, '{"different":"parameters"}', 2, identity)
    assert (state.primary_idx, state.secondary_idx, state.status) == (0, 1, "current")


def test_changed_geometry_retains_visible_review_state():
    d1, mean_dev, piezo = _signals()
    original = build_curve_events(
        d1, mean_dev, piezo, lo=10, hi=110,
        onset_thr=-0.2, outer_threshold=0.5, inner_threshold=0.2,
    )
    changed = build_curve_events(
        d1, mean_dev, piezo, lo=10, hi=110,
        onset_thr=-0.2, outer_threshold=0.5, inner_threshold=0.5,
    )
    override = {
        "primary_segment_idx": 0,
        "secondary_segment_idx": None,
        "params_json": event_geometry_identity(original),
    }
    state = resolve_segment_override_state(
        override, None, 1, event_geometry_identity(changed))
    assert state.primary_idx is None
    assert state.status == "needs_review"
