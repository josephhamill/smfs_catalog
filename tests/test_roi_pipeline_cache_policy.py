from types import SimpleNamespace

import numpy as np
import pytest

from smfs_catalog import roi_detection, roi_events, roi_pipeline, signal_processing
from smfs_catalog.analysis_params import AnalysisParams
from smfs_catalog.curve_analysis import CurveAnalysisError


def _arr():
    return np.arange(20, dtype=float)


@pytest.fixture()
def pipeline_stubs(monkeypatch):
    curve = SimpleNamespace(
        piezo_retr=_arr(), piezo_appr=_arr(), defl_retr=_arr(),
        spring_constant=1.0,
    )
    dc = SimpleNamespace(low_retr=_arr(), low_appr=_arr())
    sigs = SimpleNamespace(piezo=_arr(), d1=_arr(), mean_dev=_arr(), low=_arr())
    events = SimpleNamespace(rois=[])
    calls = {"build": 0, "fit": 0, "write": 0}

    monkeypatch.setattr(signal_processing, "fit_retract_baseline", lambda *_: SimpleNamespace(
        offset=0.0, slope=0.0, intercept=0.0, rms=0.0, fit_lo_idx=0, fit_hi_idx=1))
    monkeypatch.setattr(signal_processing, "decompose_curve", lambda *_args, **_kwargs: dc)
    monkeypatch.setattr(signal_processing, "find_end_in_contact", lambda *_args, **_kwargs: (2, None))
    monkeypatch.setattr(signal_processing, "fit_approach_invols", lambda *_args, **_kwargs: SimpleNamespace(
        slope=-1.0, intercept=0.0, rms=0.0, fit_lo_idx=0, fit_hi_idx=10))
    monkeypatch.setattr(roi_detection, "compute_detection_signals", lambda **_kwargs: sigs)
    monkeypatch.setattr(roi_detection, "rupture_search_bounds", lambda *_args: (3, 15))

    def build(*_args, **_kwargs):
        calls["build"] += 1
        return events

    def fit(*_args, **_kwargs):
        calls["fit"] += 1

    monkeypatch.setattr(roi_events, "build_curve_events", build)
    monkeypatch.setattr(roi_events, "fit_segments", fit)
    monkeypatch.setattr(roi_events, "payload_to_events", lambda _payload: events)
    monkeypatch.setattr(roi_events, "events_to_payload", lambda _events: {"v": 1, "rois": []})
    monkeypatch.setattr(roi_pipeline._db, "get_analysis_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(roi_pipeline._db, "get_event_map", lambda *_args, **_kwargs: "{}")

    def write(*_args, **_kwargs):
        calls["write"] += 1

    monkeypatch.setattr(roi_pipeline._db, "write_event_map", write)
    return curve, calls


def test_force_bypasses_event_document_and_persists_once(pipeline_stubs):
    curve, calls = pipeline_stubs
    roi_pipeline.compute_curve_events(
        curve, roi_pipeline.event_params_from(AnalysisParams()),
        db_path="ignored", code_ver="method", file_id=1, force=True,
    )
    assert calls == {"build": 1, "fit": 1, "write": 1}


def test_normal_cache_hit_skips_build_fit_and_write(pipeline_stubs):
    curve, calls = pipeline_stubs
    roi_pipeline.compute_curve_events(
        curve, roi_pipeline.event_params_from(AnalysisParams()),
        db_path="ignored", code_ver="method", file_id=1,
    )
    assert calls == {"build": 0, "fit": 0, "write": 0}


def test_invols_failure_is_not_replaced_with_one(pipeline_stubs, monkeypatch):
    curve, calls = pipeline_stubs

    def fail(*_args, **_kwargs):
        raise ValueError("bad calibration window")

    monkeypatch.setattr(signal_processing, "fit_approach_invols", fail)
    with pytest.raises(CurveAnalysisError, match="approach invOLS failed"):
        roi_pipeline.compute_curve_events(
            curve, roi_pipeline.event_params_from(AnalysisParams()), force=True,
        )
    assert calls["fit"] == 0
    assert calls["write"] == 0
