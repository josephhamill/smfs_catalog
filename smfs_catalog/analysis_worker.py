# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/analysis_worker.py
#
# Background analysis worker.
#
# Drains analysis_queue one file at a time on a QThread, runs the curve
# analysis pipeline, writes the resulting event to files.event,
# and emits signals so the UI can update without blocking.
#
# Anti-freeze strategy:
#   - Pipeline runs on a worker QThread, NEVER on the GUI thread.
#   - Worker exposes pause/resume/stop via thread-safe primitives.
#   - Dashboard updates are emitted as signals and painted in batches.
#
# Lifecycle:
#   worker = AnalysisWorker(db_path=...)
#   worker.file_done.connect(slot)
#   worker.start()         # begins draining the queue
#   worker.set_paused(True/False)
#   worker.stop()          # graceful shutdown; wait()s on the thread

from __future__ import annotations

from collections import deque
import time

from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker, QWaitCondition

from . import db as _db
from .curve_analysis import analyse_file


# Default inter-file delay (ms).  Can be tuned at runtime via set_throttle_ms().
#
# Default 0 = churn at full speed.  The delay is an msleep on the WORKER thread,
# so it does NOT help GUI responsiveness (that comes from the separate thread +
# the dashboard's batched 150 ms flush).  Its only real purpose is to slow
# autoplay down so a human can watch curves scroll past in the viewer; raise it
# from the viewer's speed slider when you want that.  For headless batch
# processing it should stay 0: at ~44 ms compute/curve a non-zero delay
# dominates, and 1000 ms here turns 9k traces into ~3 h.
THROTTLE_MS = 0


class AnalysisWorker(QThread):
    """
    QThread that drains analysis_queue.

    Signals (all emitted from the worker thread; connect with default
    AutoConnection so Qt marshals to the GUI thread):
        file_started(int)               file_id about to be processed
        file_done(int, str, bool)       file_id, verdict
                                        ('event'|'non_event'|'unusable'), was_cached
        file_error(int, str)            file_id, exception message
        data_unavailable(int, str, str) file_id, path, explanation; processing
                                        paused and the item remains pending
        fatal_error(str)                worker could not start or continue
        queue_empty()                   reached the queue edge in the current
                                        direction; worker is idling.  NOT a claim
                                        that every file has been analysed — files
                                        behind the playhead are simply behind it.
        paused_changed(bool)            new paused state
        throttle_changed(int)           new inter-file delay (ms)
        queue_changed()                 queue membership changed
        playhead_changed(int)           file selected for display/processing
        direction_changed(int)          new queue direction (+1 or -1)
    """

    file_started    = pyqtSignal(int)
    file_done       = pyqtSignal(int, str, bool)
    file_error      = pyqtSignal(int, str)
    data_unavailable = pyqtSignal(int, str, str)
    fatal_error = pyqtSignal(str)
    queue_empty     = pyqtSignal()
    paused_changed  = pyqtSignal(bool)
    # Emitted so every control surface showing the rate limit stays in step —
    # the worker is the one source of truth for it, not whichever window was
    # last used to set it.
    throttle_changed = pyqtSignal(int)
    # Queue membership changed — scrubber positions now mean different curves.
    queue_changed = pyqtSignal()
    # Emitted whenever the playhead lands on a new file — viewers plot it.
    playhead_changed = pyqtSignal(int)
    direction_changed = pyqtSignal(int)

    def __init__(self, db_path: str, parent=None) -> None:
        super().__init__(parent)
        self._db_path = db_path
        self._stop    = False
        self._paused  = True                 # start paused — viewer/dashboard explicitly unpauses
        self._throttle_ms = THROTTLE_MS
        self._direction = +1                 # +1 = forward, -1 = backward
        self._playhead: int | None = None    # current/last-processed file_id
        self._playhead_index: int | None = None  # its position in queue order
        self._step_requests: deque[int] = deque() # one-shot manual steps (Prev/Next)
        self._mutex   = QMutex()
        self._cond    = QWaitCondition()    # signalled when paused → resumed, or new work arrives
        self._idle_emitted = False          # avoid spamming queue_empty
        # Persistent DB connection for per-file status/lookup/event
        # writes — opened in run() (sqlite connections are thread-affine) so the
        # worker stops opening/closing a connection for every file.
        self._conn = None
        # Cached queue id-list for reverse/edge navigation.  Rebuilt lazily; the
        # dashboard calls invalidate_queue_cache() when queue membership changes.
        # Avoids a full 9k-row JOIN (list_queue) on every reverse step (the O(n²)
        # reverse-navigation cost).
        self._queue_ids_cache: list[int] | None = None
        self._queue_cache_generation = 0
        # A transient read failure is retried before advancing after the user
        # reconnects the data and presses Play.
        self._retry_file_id: int | None = None

    # ── Public control ────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Request graceful shutdown and block until the thread exits."""
        with QMutexLocker(self._mutex):
            self._stop = True
            self._cond.wakeAll()
        self.wait()

    def set_paused(self, paused: bool) -> None:
        with QMutexLocker(self._mutex):
            if paused == self._paused:
                return
            self._paused = paused
            if not paused:
                self._cond.wakeAll()
        self.paused_changed.emit(paused)

    def is_paused(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._paused

    def throttle_ms(self) -> int:
        """Current inter-file delay (ms).  Lets a viewer open its speed slider
        matching the worker instead of imposing a slow default."""
        with QMutexLocker(self._mutex):
            return self._throttle_ms

    def set_throttle_ms(self, ms: int) -> None:
        """Adjust the inter-file delay.  Safe to call from the GUI thread.

        Emits throttle_changed so every navigator bar repaints — a value set in
        one window must show up in the other, since both are views onto this one
        number rather than copies of it.  Emitted outside the lock, and only on a
        real change, so two bars can't ping-pong each other.
        """
        ms = max(0, int(ms))
        with QMutexLocker(self._mutex):
            if ms == self._throttle_ms:
                return
            self._throttle_ms = ms
        self.throttle_changed.emit(ms)

    def direction(self) -> int:
        """+1 = forward, -1 = backward.  Read by the navigator bars."""
        with QMutexLocker(self._mutex):
            return self._direction

    def set_direction(self, d: int) -> None:
        """+1 = play forward, -1 = play backward through queue order."""
        d = +1 if d >= 0 else -1
        with QMutexLocker(self._mutex):
            if d == self._direction:
                return
            self._direction = d
        self.direction_changed.emit(d)

    def step_to(self, file_id: int) -> None:
        """
        One-shot: process this file_id next, regardless of play/pause state.
        Used by Prev/Next buttons.  Does NOT change paused state.
        """
        with QMutexLocker(self._mutex):
            self._step_requests.append(int(file_id))
            self._cond.wakeAll()

    def step_relative(self, delta: int) -> None:
        """Step ±N in queue order from the current playhead.  ±1 is typical."""
        target = self._neighbour_file_id(self.playhead(), delta)
        if target is not None:
            self.step_to(target)

    def playhead(self) -> int | None:
        with QMutexLocker(self._mutex):
            return self._playhead

    def notify_work_available(self) -> None:
        """
        Wake the worker after queue membership changes.

        This deliberately does not change the paused state. Adding files while
        browsing a paused queue must not unexpectedly start analysis.
        """
        with QMutexLocker(self._mutex):
            self._cond.wakeAll()

    def queue_ids(self) -> list[int]:
        """
        Queue order, as file_ids.  The navigator bars read positions from here
        rather than querying the DB themselves: a bar is open for the whole of
        every batch, and a per-file list_queue() on a 9k queue is exactly the
        O(n²) cost the cache below avoids.

        Returns the live cached list — treat it as read-only.
        """
        return self._queue_ids()

    def invalidate_queue_cache(self) -> None:
        """
        Drop the cached queue id-list.  Call after any change to queue membership
        (enqueue/dequeue/clear) so reverse/edge navigation rebuilds it once on the
        next step instead of seeing stale ids.  Cheap and thread-safe: a plain
        reference assignment that the worker re-reads on its next navigation.

        Emits queue_changed so every navigator bar re-ranges its scrubber —
        position 400 means a different curve after the queue is repopulated.
        Broadcasting beats asking each caller to remember: there are five call
        sites today and the sixth would be the one that forgets.
        """
        with QMutexLocker(self._mutex):
            self._queue_ids_cache = None
            self._queue_cache_generation += 1
        self.queue_changed.emit()

    # ── Thread body ──────────────────────────────────────────────────────────

    def run(self) -> None:
        # Open one connection on THIS (worker) thread and reuse it for every
        # per-file status/lookup/event write.  Closed on exit.
        try:
            self._conn = _db.get_connection(self._db_path)
            self._run_loop()
        except Exception as exc:
            self.fatal_error.emit(f"Analysis worker stopped: {exc!r}")
        finally:
            try:
                if self._conn is not None:
                    self._conn.close()
            except Exception:
                pass
            self._conn = None

    def _run_loop(self) -> None:
        while True:
            if self._check_stop():
                return

            # 1. Manual steps trump everything — process even when paused.
            fid = self._pop_step_request()
            if fid is not None:
                self._set_playhead(fid)
                self.playhead_changed.emit(fid)
                outcome = self._process_one(fid)
                if outcome == "unavailable":
                    with QMutexLocker(self._mutex):
                        self._retry_file_id = fid
                    self._set_paused_internal(True)
                # Short, fixed delay so rapid Prev/Next scrubbing stays snappy.
                self.msleep(20)
                continue

            # 2. Pause check (after step queue, so user can scrub while paused).
            if self._is_paused_locked():
                self._wait_while_paused()
                # After waking, restart the loop so step requests are picked up
                # before we fall through to autonomous advance.
                continue
            if self._check_stop():
                return

            # 3. Autonomous advance — the next file in queue order, in the
            #    current direction.  One meaning, both directions: from a clean
            #    start (no playhead) that is the queue's first file forward /
            #    last file backward; otherwise it is the neighbour of wherever
            #    the playhead currently is, including where the user just put it.
            with QMutexLocker(self._mutex):
                retry_id = self._retry_file_id
                current = self._playhead
                direction = self._direction
            next_id = None
            if retry_id is not None:
                if retry_id in self._queue_ids():
                    next_id = retry_id
                else:
                    # The user removed/cleared the failed row while paused.
                    with QMutexLocker(self._mutex):
                        if self._retry_file_id == retry_id:
                            self._retry_file_id = None
            if next_id is None:
                next_id = self._neighbour_file_id(current, direction)

            if next_id is None:
                # Reached the edge of the queue in this direction.  Pause and idle.
                if not self._idle_emitted:
                    self.queue_empty.emit()
                    self._idle_emitted = True
                self._set_paused_internal(True)
                self._wait_for_work()
                continue

            self._idle_emitted = False
            self._set_playhead(next_id)
            self.playhead_changed.emit(next_id)
            outcome = self._process_one(next_id)
            with QMutexLocker(self._mutex):
                if outcome == "unavailable":
                    self._retry_file_id = next_id
                elif self._retry_file_id == next_id:
                    self._retry_file_id = None
            if outcome == "unavailable":
                self._set_paused_internal(True)
                continue

            with QMutexLocker(self._mutex):
                delay = self._throttle_ms
            if delay and self._wait_for_throttle(delay):
                return

    # ── Internals ────────────────────────────────────────────────────────────

    def _process_one(self, file_id: int) -> str:
        """Process one queue row and emit exactly one terminal outcome."""
        try:
            _db.set_queue_status(file_id, "running", self._db_path, conn=self._conn)
        except Exception as exc:
            self.file_error.emit(file_id, f"could not mark running: {exc!r}")
            return "error"
        self.file_started.emit(file_id)

        try:
            path = self._lookup_path(file_id)
        except Exception as exc:
            return self._fail(file_id, f"path lookup failed: {exc!r}")
        if path is None:
            return self._fail(file_id, "file_id not found")

        # ONE parameter set is in force for a run: the queue owner's
        # (db.active_param_owner, derived from the queue itself).  Never
        # per-file — that would have the worker and any open tuning window
        # writing the same global `settings` row, last writer winning.

        try:
            # Hand down the worker-thread connection opened in run().  Without
            # it every cache read and result write inside the pipeline opened
            # and closed its own connection, and closing the last connection
            # to a WAL database checkpoints it.
            event, was_cached = analyse_file(
                file_id, path, self._db_path, conn=self._conn
            )
        except Exception as exc:
            return self._fail(file_id, f"analysis failed: {exc!r}")

        if event == "unanalysed":
            # No pipeline for this modality.  The file is intact and stays in
            # the queue to be navigated to and plotted; there is just no
            # analysis to record, so files.event is left alone.
            try:
                _db.set_queue_status(file_id, "done", self._db_path, conn=self._conn)
            except Exception as exc:
                return self._fail(file_id, f"saving completion status failed: {exc!r}")
            self.file_done.emit(file_id, event, was_cached)
            return "done"

        if event == "unusable":
            # The pipeline stored the reason and the verdict on the file, and
            # left the row in the queue.  Settle the row rather than repeating
            # those writes.
            try:
                _db.set_queue_status(file_id, "done", self._db_path, conn=self._conn)
            except Exception as exc:
                return self._fail(file_id, f"saving completion status failed: {exc!r}")
            self.file_done.emit(file_id, event, was_cached)
            return "done"

        if event == "unavailable":
            # Environmental, not scientific: do not write files.event and do
            # not call this row done. One notification pauses the batch before
            # thousands of paths on the same disconnected drive are attempted.
            try:
                _db.set_queue_status(file_id, "pending", self._db_path, conn=self._conn)
            except Exception as exc:
                self.file_error.emit(file_id, f"could not restore pending status: {exc!r}")
            self.data_unavailable.emit(
                file_id, path,
                "The raw curve could not be read. Its drive may be disconnected "
                "or the file may no longer be accessible.",
            )
            return "unavailable"

        try:
            _db.set_event(file_id, event, self._db_path, conn=self._conn)
        except Exception as exc:
            return self._fail(file_id, f"saving classification failed: {exc!r}")

        try:
            _db.set_queue_status(file_id, "done", self._db_path, conn=self._conn)
        except Exception as exc:
            return self._fail(file_id, f"saving completion status failed: {exc!r}")
        self.file_done.emit(file_id, event, was_cached)
        return "done"

    def _fail(self, file_id: int, message: str) -> str:
        """Persist a genuine queue error and emit its sole terminal signal."""
        message = str(message)[:500]
        try:
            _db.set_queue_status(
                file_id, f"error: {message}", self._db_path, conn=self._conn
            )
        except Exception:
            pass
        self.file_error.emit(file_id, message)
        return "error"

    def _lookup_path(self, file_id: int) -> str | None:
        return _db.get_path(file_id, self._db_path, conn=self._conn)

    def _check_stop(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._stop

    def _wait_while_paused(self) -> None:
        # Wake on: resume, stop, or a manual step request arriving.  The step
        # path runs even while paused, so users can scrub with Prev/Next while
        # autonomous play stays off.
        self._mutex.lock()
        while self._paused and not self._stop and not self._step_requests:
            self._cond.wait(self._mutex)
        self._mutex.unlock()

    def _wait_for_work(self) -> None:
        """Block until notify_work_available(), stop(), or 1 s poll."""
        self._mutex.lock()
        if not self._stop:
            self._cond.wait(self._mutex, 1000)
        self._mutex.unlock()

    def _wait_for_throttle(self, delay_ms: int) -> bool:
        """Wait out the rate limit, returning True when shutdown was requested.

        Unlike ``msleep``, the condition wait is woken by :meth:`stop`, so a
        slow playback setting cannot hold the GUI in ``stop()`` for seconds.
        Other condition notifications may arrive during the delay; retain the
        original deadline so they do not accidentally shorten the rate limit.
        """
        deadline = time.monotonic() + max(0, int(delay_ms)) / 1000.0
        self._mutex.lock()
        try:
            while not self._stop:
                remaining_ms = int((deadline - time.monotonic()) * 1000)
                if remaining_ms <= 0:
                    break
                self._cond.wait(self._mutex, remaining_ms)
            return self._stop
        finally:
            self._mutex.unlock()

    def _pop_step_request(self) -> int | None:
        with QMutexLocker(self._mutex):
            if self._step_requests:
                return self._step_requests.popleft()
            return None

    def _is_paused_locked(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._paused

    def _set_paused_internal(self, paused: bool) -> None:
        with QMutexLocker(self._mutex):
            if paused == self._paused:
                return
            self._paused = paused
        self.paused_changed.emit(paused)

    # ── Queue navigation helpers ─────────────────────────────────────────────

    def _queue_ids(self) -> list[int]:
        # Served from a cache rebuilt only when queue membership changes (via
        # invalidate_queue_cache()).  Read the attribute once so a concurrent
        # invalidate (None) can't trip us mid-use.
        while True:
            with QMutexLocker(self._mutex):
                cache = self._queue_ids_cache
                generation = self._queue_cache_generation
            if cache is not None:
                return cache
            cache = [int(r["file_id"]) for r in _db.list_queue(self._db_path)]
            with QMutexLocker(self._mutex):
                # Do not overwrite an invalidation that arrived during the DB
                # query; its caller knows the membership changed.
                if self._queue_cache_generation == generation:
                    self._queue_ids_cache = cache
                    return cache

    def _set_playhead(self, file_id: int) -> None:
        """Move the playhead, remembering the queue position it lands on.

        The position is what navigation needs if the file is later removed from
        the queue: an id that is gone says nothing about where the user was
        standing.
        """
        ids = self._queue_ids()
        try:
            index = ids.index(int(file_id))
        except ValueError:
            index = None
        with QMutexLocker(self._mutex):
            self._playhead = file_id
            self._playhead_index = index

    # Navigation always follows queue order from the user's playhead.

    def _neighbour_file_id(self, current: int | None, delta: int) -> int | None:
        ids = self._queue_ids()
        if not ids:
            return None
        if current is None:
            return ids[0] if delta >= 0 else ids[-1]
        try:
            i = ids.index(int(current))
        except ValueError:
            with QMutexLocker(self._mutex):
                i = self._playhead_index if current == self._playhead else None
            if i is None:
                return ids[0] if delta >= 0 else ids[-1]
            # Rows below the vacated position shifted up one, so that position
            # now holds the successor: forward starts there, backward steps off
            # it.
            j = i + delta - 1 if delta > 0 else i + delta
            return ids[j] if 0 <= j < len(ids) else None
        j = i + delta
        if 0 <= j < len(ids):
            return ids[j]
        return None
