"""b3b: broad `except Exception` handlers narrowed to what the callee raises,
or logged through crashlog.log_handled. Logging is observed by replacing
`smfs_catalog.crashlog.log_handled` (looked up at call time by the code)."""
import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from smfs_catalog import analysis_worker as worker_module
from smfs_catalog import clustering, crashlog, db, models, roi_events, variables
from smfs_catalog.analysis_worker import AnalysisWorker
from smfs_catalog.models import CHAIN_MODELS, ChainFit
from smfs_catalog.physical_2dh_window import Physical2DHWindow


@pytest.fixture
def logged(monkeypatch):
    """List of (context, exception) passed to crashlog.log_handled."""
    calls = []
    monkeypatch.setattr(crashlog, "log_handled", lambda ctx, exc: calls.append((ctx, exc)))
    return calls


def _raiser(exc):
    def fn(*a, **k):
        raise exc
    return fn


XF = (np.linspace(1.0, 40.0, 30), np.linspace(1.0, 30.0, 30))


# -- roi_events: the two curve_fit call sites ---------------------------------

FIT_SITES = [
    ("_fit_wlc_window", lambda: roi_events._fit_wlc_window(*XF, 0.4, 50.0)),
    ("_fit_chain_model",
     lambda: roi_events._fit_chain_model(CHAIN_MODELS["fjc"], *XF, 0.4, 50.0)),
]


@pytest.mark.parametrize("name,call", FIT_SITES, ids=[s[0] for s in FIT_SITES])
@pytest.mark.parametrize("exc", [RuntimeError("no convergence"), ValueError("bad bounds")])
def test_fit_failure_from_optimiser_gives_none(monkeypatch, name, call, exc):
    monkeypatch.setattr(roi_events, "fit_model", _raiser(exc))
    assert call() is None


@pytest.mark.parametrize("name,call", FIT_SITES, ids=[s[0] for s in FIT_SITES])
@pytest.mark.parametrize("exc", [TypeError("bug"), ZeroDivisionError("bug")])
def test_fit_other_exception_propagates(monkeypatch, name, call, exc):
    monkeypatch.setattr(roi_events, "fit_model", _raiser(exc))
    with pytest.raises(type(exc)):
        call()


# -- models.ChainFit.x_at_force -----------------------------------------------

def _wlc_fit():
    return ChainFit("wlc", (0.4, 50.0))


def test_x_at_force_unreachable_force_is_none():
    assert _wlc_fit().x_at_force(1e9) is None


def test_x_at_force_brentq_value_error_is_none():
    # F(x) is positive everywhere, so a negative target gives no sign change:
    # the real brentq raises ValueError.
    assert _wlc_fit().x_at_force(-1.0) is None


def test_x_at_force_reachable_force_found():
    x = _wlc_fit().x_at_force(5.0)
    assert x is not None and abs(float(_wlc_fit().force(x)) - 5.0) < 1e-6


def test_x_at_force_brentq_runtime_error_is_none(monkeypatch):
    monkeypatch.setattr(models, "_brentq", _raiser(RuntimeError("max iter")))
    assert _wlc_fit().x_at_force(5.0) is None


def test_x_at_force_unexpected_exception_propagates(monkeypatch):
    # b7: ZeroDivisionError is now an expected outcome (l_p == 0 -> None, see
    # tests/test_b7_hardening.py); other bugs such as TypeError still propagate.
    monkeypatch.setattr(models, "_brentq", _raiser(TypeError("bug")))
    with pytest.raises(TypeError):
        _wlc_fit().x_at_force(5.0)


# -- physical_2dh_window._apply_align_visuals (setFormat) ---------------------

def _align_self(set_format):
    hline = SimpleNamespace(setVisible=lambda v: None)
    vline = SimpleNamespace(label=SimpleNamespace(setFormat=set_format))
    return SimpleNamespace(_align_mode="fstar", _vline=vline, _hline=hline)


def test_align_visuals_attribute_error_is_swallowed():
    Physical2DHWindow._apply_align_visuals(_align_self(_raiser(AttributeError("no label"))))


def test_align_visuals_other_exception_propagates():
    with pytest.raises(RuntimeError):
        Physical2DHWindow._apply_align_visuals(_align_self(_raiser(RuntimeError("bug"))))


# -- analysis_worker._fail ----------------------------------------------------

def test_fail_status_write_failure_is_logged_and_error_still_signalled(monkeypatch, logged):
    worker = AnalysisWorker("catalog.db")
    worker._conn = object()
    boom = OSError("disk full")
    monkeypatch.setattr(worker_module._db, "set_queue_status", _raiser(boom))
    errors = []
    worker.file_error.connect(lambda *a: errors.append(a))

    assert worker._fail(7, "original problem") == "error"
    assert errors == [(7, "original problem")]
    assert [exc for _, exc in logged] == [boom]


def test_fail_without_write_failure_logs_nothing(monkeypatch, logged):
    worker = AnalysisWorker("catalog.db")
    worker._conn = object()
    monkeypatch.setattr(worker_module._db, "set_queue_status", lambda *a, **k: None)
    assert worker._fail(7, "x") == "error"
    assert logged == []


# -- clustering._notify -------------------------------------------------------

@pytest.fixture
def subscribers(monkeypatch):
    monkeypatch.setattr(clustering, "_subscribers", [])
    return clustering._subscribers


def test_notify_logs_crashing_subscriber_and_runs_the_others(subscribers, logged):
    boom = RuntimeError("deleted widget")
    seen = []
    subscribers.append(lambda: seen.append("a"))
    subscribers.append(_raiser(boom))
    subscribers.append(lambda: seen.append("c"))

    clustering._notify()

    assert seen == ["a", "c"]
    assert [exc for _, exc in logged] == [boom]


def test_notify_healthy_subscribers_log_nothing(subscribers, logged):
    subscribers.append(lambda: None)
    clustering._notify()
    assert logged == []


# -- variables.available ------------------------------------------------------

def test_available_logs_queue_types_failure_and_returns_fallback(
    monkeypatch, logged, catalog
):
    boom = sqlite3.OperationalError("no such table")
    monkeypatch.setattr(variables._db, "get_queue_analysis_types", _raiser(boom))

    out = variables.available([], catalog)

    assert [exc for _, exc in logged] == [boom]
    keys = {v.key for v in out}
    assert set(variables._ANALYSIS_LABELS) - variables.EXCLUDED_VARIABLE_KEYS <= keys


def test_available_logs_nothing_when_queue_types_succeed(monkeypatch, logged, catalog):
    monkeypatch.setattr(variables._db, "get_queue_analysis_types", lambda p: [])
    variables.available([], catalog)
    assert logged == []


# -- db.get_experimentalist_profile -------------------------------------------

def test_profile_corrupt_json_is_logged_and_none(catalog, logged):
    db.set_experimentalist_profile("alice", "{bad", catalog)
    assert db.get_experimentalist_profile("alice", catalog) is None
    assert len(logged) == 1 and isinstance(logged[0][1], json.JSONDecodeError)


def test_profile_type_error_is_logged_and_none(catalog, logged, monkeypatch):
    db.set_experimentalist_profile("alice", "{}", catalog)
    monkeypatch.setattr(json, "loads", _raiser(TypeError("not str")))
    assert db.get_experimentalist_profile("alice", catalog) is None
    assert len(logged) == 1 and isinstance(logged[0][1], TypeError)


def test_profile_valid_json_returned_and_not_logged(catalog, logged):
    db.set_experimentalist_profile("alice", '{"a": 1}', catalog)
    assert db.get_experimentalist_profile("alice", catalog) == {"a": 1}
    assert logged == []


def test_profile_missing_is_none_and_not_logged(catalog, logged):
    assert db.get_experimentalist_profile("nobody", catalog) is None
    assert logged == []


def test_profile_unexpected_error_while_parsing_propagates(catalog, logged, monkeypatch):
    db.set_experimentalist_profile("alice", "{}", catalog)
    monkeypatch.setattr(json, "loads", _raiser(ZeroDivisionError("bug")))
    with pytest.raises(ZeroDivisionError):
        db.get_experimentalist_profile("alice", catalog)


def test_profile_sqlite_error_propagates(monkeypatch, logged, catalog):
    closed = sqlite3.connect(":memory:")
    closed.close()
    monkeypatch.setattr(db, "get_connection", lambda p: closed)
    with pytest.raises(sqlite3.ProgrammingError):
        db.get_experimentalist_profile("alice", catalog)
