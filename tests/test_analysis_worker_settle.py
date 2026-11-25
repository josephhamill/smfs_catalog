"""Pins the 'mark done, emit file_done, return done' tail of _process_one."""

import pytest

from smfs_catalog.analysis_worker import AnalysisWorker
from smfs_catalog import analysis_worker as worker_module


def _worker(monkeypatch, *, fail_done=False):
    worker = AnalysisWorker("catalog.db")
    worker._conn = object()
    statuses, saved_events = [], []

    def set_status(fid, status, db_path, conn=None):
        if fail_done and status == "done":
            raise OSError("disk full")
        statuses.append(status)

    monkeypatch.setattr(worker_module._db, "set_queue_status", set_status)
    monkeypatch.setattr(
        worker_module._db, "get_path", lambda fid, db_path, conn=None: "X:/c.ibw"
    )
    monkeypatch.setattr(
        worker_module._db, "set_event",
        lambda fid, event, db_path, conn=None: saved_events.append(event),
    )
    done, errors = [], []
    worker.file_done.connect(lambda *a: done.append(a))
    worker.file_error.connect(lambda *a: errors.append(a))
    return worker, statuses, saved_events, done, errors


def _analyse_returns(monkeypatch, event, cached):
    monkeypatch.setattr(
        worker_module, "analyse_file", lambda *a, **k: (event, cached)
    )


# (event, was_cached, event written to files.event by the worker)
PATHS = [
    ("unanalysed", True, []),
    ("unanalysed", False, []),
    ("unusable", True, []),
    ("unusable", False, []),
    ("event", True, ["event"]),
    ("event", False, ["event"]),
]


@pytest.mark.parametrize("event,cached,written", PATHS)
def test_process_one_settles_done_and_emits_once(monkeypatch, event, cached, written):
    worker, statuses, saved, done, errors = _worker(monkeypatch)
    _analyse_returns(monkeypatch, event, cached)

    assert worker._process_one(5) == "done"
    assert statuses == ["running", "done"]
    assert saved == written
    assert done == [(5, event, cached)]
    assert errors == []


@pytest.mark.parametrize("event,cached,written", PATHS)
def test_process_one_status_write_failure_is_one_error(
    monkeypatch, event, cached, written
):
    worker, statuses, saved, done, errors = _worker(monkeypatch, fail_done=True)
    _analyse_returns(monkeypatch, event, cached)

    assert worker._process_one(6) == "error"
    assert statuses[0] == "running"
    assert statuses[1].startswith("error: saving completion status failed:")
    assert saved == written
    assert done == []
    assert len(errors) == 1 and errors[0][0] == 6


@pytest.mark.parametrize("event,cached", [("unanalysed", True), ("unusable", False),
                                          ("event", True), ("event", False)])
def test_settle_done_marks_done_emits_and_returns_done(monkeypatch, event, cached):
    worker, statuses, saved, done, errors = _worker(monkeypatch)

    assert worker._settle_done(9, event, cached) == "done"
    assert statuses == ["done"]
    assert saved == []
    assert done == [(9, event, cached)]
    assert errors == []


def test_settle_done_status_failure_is_one_error_not_done(monkeypatch):
    worker, statuses, saved, done, errors = _worker(monkeypatch, fail_done=True)

    assert worker._settle_done(9, "event", False) == "error"
    assert statuses[0].startswith("error: saving completion status failed:")
    assert done == []
    assert len(errors) == 1 and errors[0][0] == 9
