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
        worker_module, "analyse_file",
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

    monkeypatch.setattr(worker_module, "analyse_file", fail)

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
        worker_module, "analyse_file",
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


def test_damaged_data_settles_its_row_and_keeps_it_in_the_queue(monkeypatch):
    """The queue is the timeline the navigator walks, so analysing a file never
    removes it: a damaged curve is still one the user can step to and plot."""
    worker, statuses, saved_events = _worker_with_fake_db(monkeypatch)
    done = []
    queue_changes = []
    worker.file_done.connect(lambda *args: done.append(args))
    worker.queue_changed.connect(lambda: queue_changes.append(True))
    monkeypatch.setattr(
        worker_module, "analyse_file",
        lambda *args, **kwargs: ("unusable", False),
    )

    assert worker._process_one(10) == "done"
    assert statuses == ["running", "done"]
    assert saved_events == []
    assert done == [(10, "unusable", False)]
    assert queue_changes == []


def test_a_modality_with_no_pipeline_settles_without_a_verdict(monkeypatch):
    """A force-clamp or held trace is intact and browsable; there is simply no
    analysis for it yet, so no event is written and the row stays queued."""
    worker, statuses, saved_events = _worker_with_fake_db(monkeypatch)
    done = []
    queue_changes = []
    worker.file_done.connect(lambda *args: done.append(args))
    worker.queue_changed.connect(lambda: queue_changes.append(True))
    monkeypatch.setattr(
        worker_module, "analyse_file",
        lambda *args, **kwargs: ("unanalysed", True),
    )

    assert worker._process_one(11) == "done"
    assert statuses == ["running", "done"]
    assert saved_events == []
    assert done == [(11, "unanalysed", True)]
    assert queue_changes == []


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


def _worker_over_queue(monkeypatch, ids):
    """A worker whose queue is `ids`, mutable by the caller."""
    worker = AnalysisWorker("catalog.db")
    monkeypatch.setattr(
        worker_module._db, "list_queue",
        lambda db_path: [{"file_id": i} for i in ids],
    )
    return worker


def test_a_step_from_a_vacated_position_moves_one_place(monkeypatch):
    """Removing the playhead's file must cost one step, not throw the user to
    the far end of the queue."""
    ids = [101, 102, 103, 104, 105]
    worker = _worker_over_queue(monkeypatch, ids)

    worker._set_playhead(103)
    ids.remove(103)
    worker.invalidate_queue_cache()

    assert worker._neighbour_file_id(103, -1) == 102
    assert worker._neighbour_file_id(103, +1) == 104


def test_stepping_back_from_the_first_position_stays_put(monkeypatch):
    """Position one has nothing behind it, whether or not its file is still in
    the queue."""
    ids = [101, 102, 103]
    worker = _worker_over_queue(monkeypatch, ids)

    worker._set_playhead(101)
    assert worker._neighbour_file_id(101, -1) is None

    ids.remove(101)
    worker.invalidate_queue_cache()
    assert worker._neighbour_file_id(101, -1) is None
    assert worker._neighbour_file_id(101, +1) == 102


def test_a_playhead_never_seen_in_the_queue_restarts_at_the_edge(monkeypatch):
    """With no remembered position there is nothing to step from, so the edge
    is the only honest answer."""
    ids = [101, 102, 103]
    worker = _worker_over_queue(monkeypatch, ids)

    assert worker._neighbour_file_id(999, -1) == 103
    assert worker._neighbour_file_id(999, +1) == 101
