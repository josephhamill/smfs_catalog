# Copyright (C) 2026 Joseph Hamill

from types import SimpleNamespace

import pytest

from smfs_catalog import curve_analysis as _ca
from smfs_catalog import signal_processing as _sp
from smfs_catalog.analysis_params import AnalysisParams


def test_one_parameter_snapshot_reaches_multi_event_persistence(monkeypatch):
    """A profile edit during one curve cannot split its two analysis stages."""
    snapshot = AnalysisParams()
    loads = []
    captured = []

    def load_params(_db_path):
        loads.append(True)
        return snapshot

    result = _ca.CurveResult(
        event=True, offset=0.0, flatness=0.0, contact_z=1.0,
        snapoff_z=2.0, rupture_z=3.0, onset_z=2.5, invols_slope=1.0,
    )
    monkeypatch.setattr(_ca._db, "load_analysis_params", load_params)
    monkeypatch.setattr(_ca._db, "get_analysis_result", lambda *a, **k: None)
    monkeypatch.setattr(_ca._db, "get_curve_type", lambda *a, **k: "continuous_stretch")
    monkeypatch.setattr(_ca._db, "write_analysis_result", lambda *a, **k: None)
    monkeypatch.setattr(_ca._db, "get_or_store_deflection_histogram",
                        lambda *a, **k: None)
    monkeypatch.setattr(_ca, "load_force_curve",
                        lambda _path: SimpleNamespace(defl_retr=None))
    monkeypatch.setattr(_ca, "cache_version", lambda: "test-build")
    monkeypatch.setattr(_ca, "analyse_curve", lambda *a, **k: (result, _ca.Stage1Search()))
    monkeypatch.setattr(
        _ca, "_persist_multi_event_roi",
        lambda *a, **k: captured.append(k["param_set"]),
    )

    verdict, cached = _ca.analyse_file(1, "curve.ibw", "catalog.db")

    assert (verdict, cached) == ("event", False)
    assert len(loads) == 3  # acquire once; validate before both publications
    assert captured == [snapshot]


def test_calculation_exception_is_not_a_non_event(monkeypatch):
    def fail_baseline(*_args, **_kwargs):
        raise ValueError("baseline window is invalid")

    monkeypatch.setattr(_sp, "fit_retract_baseline", fail_baseline)
    params = SimpleNamespace(anchor_nm=150.0, params_bl="baseline-test")

    with pytest.raises(_ca.CurveAnalysisError, match="retract baseline failed"):
        _ca.analyse_curve(object(), params)


def test_edit_during_curve_reruns_before_current_result_is_published(monkeypatch):
    old = AnalysisParams()
    new = old.with_update("spectral_cutoff_hz", 1234.0)
    snapshots = iter((old, new, new, new, new))
    analysed = []
    persisted = []
    result = _ca.CurveResult(
        event=True, offset=0.0, flatness=0.0, contact_z=1.0,
        snapoff_z=2.0, rupture_z=3.0, onset_z=2.5, invols_slope=1.0,
    )

    monkeypatch.setattr(_ca._db, "load_analysis_params", lambda _p: next(snapshots))
    monkeypatch.setattr(_ca._db, "get_analysis_result", lambda *a, **k: None)
    monkeypatch.setattr(_ca._db, "get_curve_type", lambda *a, **k: "continuous_stretch")
    monkeypatch.setattr(_ca._db, "write_analysis_result", lambda *a, **k: None)
    monkeypatch.setattr(_ca._db, "get_or_store_deflection_histogram",
                        lambda *a, **k: None)
    monkeypatch.setattr(_ca, "load_force_curve",
                        lambda _path: SimpleNamespace(defl_retr=None))
    monkeypatch.setattr(_ca, "cache_version", lambda: "test-build")
    monkeypatch.setattr(
        _ca, "analyse_curve",
        lambda _curve, pipeline, **_kw: (
            analysed.append(pipeline.cutoff_hz) or result,
            _ca.Stage1Search(),
        ),
    )
    monkeypatch.setattr(
        _ca, "_persist_multi_event_roi",
        lambda *a, **k: persisted.append(k["param_set"]),
    )

    verdict, cached = _ca.analyse_file(1, "curve.ibw", "catalog.db")

    assert (verdict, cached) == ("event", False)
    assert analysed == [old.spectral_cutoff_hz, new.spectral_cutoff_hz]
    assert persisted == [new]


def test_the_slow_path_stores_the_loaded_curves_histogram_and_the_fast_path_does_not(
        monkeypatch):
    """One more calculation where the curve is already in memory, and none
    where it is not: a cached verdict never opens the file, so it never bins."""
    snapshot = AnalysisParams()
    stored = []
    curve = SimpleNamespace(defl_retr="retract-deflection")
    result = _ca.CurveResult(
        event=False, offset=0.0, flatness=0.0, contact_z=1.0,
        snapoff_z=2.0, rupture_z=float("nan"), onset_z=float("nan"),
        invols_slope=1.0,
    )
    cached_verdict = [None]

    monkeypatch.setattr(_ca._db, "load_analysis_params", lambda _p: snapshot)
    monkeypatch.setattr(_ca._db, "get_analysis_result",
                        lambda *a, **k: cached_verdict[0])
    monkeypatch.setattr(_ca._db, "get_curve_type", lambda *a, **k: "continuous_stretch")
    monkeypatch.setattr(_ca._db, "write_analysis_result", lambda *a, **k: None)
    monkeypatch.setattr(_ca._db, "delete_event_map", lambda *a, **k: None)
    monkeypatch.setattr(
        _ca._db, "get_or_store_deflection_histogram",
        lambda file_id, defl, db_path, conn=None: stored.append((file_id, defl)))
    monkeypatch.setattr(_ca, "load_force_curve", lambda _path: curve)
    monkeypatch.setattr(_ca, "cache_version", lambda: "test-build")
    monkeypatch.setattr(_ca, "analyse_curve",
                        lambda *a, **k: (result, _ca.Stage1Search()))

    assert _ca.analyse_file(7, "curve.ibw", "catalog.db") == ("non_event", False)
    assert stored == [(7, "retract-deflection")]

    stored.clear()
    cached_verdict[0] = 0.0
    monkeypatch.setattr(_ca, "load_force_curve", lambda _path: pytest.fail(
        "the fast path loaded the curve"))
    assert _ca.analyse_file(7, "curve.ibw", "catalog.db") == ("non_event", True)
    assert stored == []
