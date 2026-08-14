"""Terminal-state and transient-data regression tests for AnalysisWorker."""

import threading
import time

from smfs_catalog.analysis_worker import AnalysisWorker
from smfs_catalog import analysis_worker as worker_module


def _worker_with_fake_db(monkeypatch, *, path="X:/data/curve.ibw"):
    worker = AnalysisWorker("catalog.db")
    worker._conn = object()
    statuses = []
    saved_events = []
    monkeypatch.setattr(
        worker_module._db, "set_queue_status",
        lambda fid, status, db_path, conn=None: statuses.append(status),
    )
    monkeypatch.setattr(
        worker_module._db, "get_path",
        lambda fid, db_path, conn=None: path,
    )
    monkeypatch.setattr(
        worker_module._db, "set_event",
        lambda fid, event, db_path, conn=None: saved_events.append(event),
    )
    return worker, statuses, saved_events


def test_unavailable_data_stays_pending_and_is_not_a_classification(monkeypatch):
    worker, statuses, saved_events = _worker_with_fake_db(monkeypatch)
    unavailable = []
    done = []
    errors = []
    worker.data_unavailable.connect(lambda *args: unavailable.append(args))
    worker.file_done.connect(lambda *args: done.append(args))
    worker.file_error.connect(lambda *args: errors.append(args))
    monkeypatch.setattr(
        worker_module, "analyse_and_classify",
        lambda *args, **kwargs: ("unavailable", False),
    )

    assert worker._process_one(7) == "unavailable"
    assert statuses == ["running", "pending"]
    assert saved_events == []
    assert len(unavailable) == 1
    assert unavailable[0][0:2] == (7, "X:/data/curve.ibw")
    assert done == []
    assert errors == []


def test_analysis_failure_has_one_error_outcome_and_persists_it(monkeypatch):
    worker, statuses, saved_events = _worker_with_fake_db(monkeypatch)
    done = []
    errors = []
    worker.file_done.connect(lambda *args: done.append(args))
    worker.file_error.connect(lambda *args: errors.append(args))

    def fail(*args, **kwargs):
        raise RuntimeError("broken pipeline")

    monkeypatch.setattr(worker_module, "analyse_and_classify", fail)

    assert worker._process_one(8) == "error"
    assert statuses[0] == "running"
    assert statuses[1].startswith("error: analysis failed:")
    assert saved_events == []
    assert len(errors) == 1
    assert done == []


def test_event_write_failure_is_not_also_reported_done(monkeypatch):
    worker, statuses, _saved_events = _worker_with_fake_db(monkeypatch)
    done = []
    errors = []
    worker.file_done.connect(lambda *args: done.append(args))
    worker.file_error.connect(lambda *args: errors.append(args))
    monkeypatch.setattr(
        worker_module, "analyse_and_classify",
        lambda *args, **kwargs: ("event", False),
    )
    monkeypatch.setattr(
        worker_module._db, "set_event",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("read-only")),
    )

    assert worker._process_one(9) == "error"
    assert statuses[-1].startswith("error: saving classification failed:")
    assert len(errors) == 1
    assert done == []


def test_unusable_is_published_without_rewriting_deleted_queue_row(monkeypatch):
    worker, statuses, saved_events = _worker_with_fake_db(monkeypatch)
    done = []
    queue_changes = []
    worker.file_done.connect(lambda *args: done.append(args))
    worker.queue_changed.connect(lambda: queue_changes.append(True))
    monkeypatch.setattr(
        worker_module, "analyse_and_classify",
        lambda *args, **kwargs: ("unusable", False),
    )

    assert worker._process_one(10) == "done"
    assert statuses == ["running"]
    assert saved_events == []
    assert done == [(10, "unusable", False)]
    assert queue_changes == [True]


def test_stop_interrupts_long_throttle_wait():
    worker = AnalysisWorker("catalog.db")
    finished = threading.Event()

    def wait_for_throttle():
        worker._wait_for_throttle(10_000)
        finished.set()

    thread = threading.Thread(target=wait_for_throttle)
    thread.start()
    time.sleep(0.02)
    started = time.monotonic()
    worker.stop()
    thread.join(timeout=1)

    assert finished.is_set()
    assert time.monotonic() - started < 0.5
