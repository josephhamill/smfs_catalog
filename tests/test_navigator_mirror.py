# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Regression test: ONE navigator, mirrored — the worker is the single source of
truth for playback (#126, 2026-08-03).

The contract under test:
(a) The worker owns the rate limit.  A NavigatorBar READS it when built and
    never writes a default into it — the trap #126 exists to close is a window
    silently changing a value the user chose.
(b) Two bars over one worker mirror each other: a throttle set on one appears
    on the other, in both directions, with no direct link between them.
(c) Closing a window does NOT reset the throttle.  Only a new worker (i.e. a
    new app launch) starts from the default.
(d) The transport reflects worker state rather than its own: pausing the worker
    from anywhere un-checks both bars' play buttons.
(e) The scrubber's meaning follows the queue.  Its range re-derives when queue
    membership changes, and moving it drives the worker's playhead — it does
    not hold a position of its own.
(f) Throttle ↔ slider round-trips, and "no delay" is the far-right end (this
    inverts: max slider = min delay), so the two windows cannot disagree about
    what full speed looks like.
(g) There is no second play/pause control anywhere: the dashboard's old
    _play_btn is gone, and nothing outside navigator_bar drives the worker's
    transport.
(h) Playing forward honours where the playhead was PUT (2026-08-03).  Forward
    means the next file in queue order and nothing else — it never jumps back
    to the oldest still-pending file, and it idles at the queue edge rather
    than wrapping round to pending work behind it.  A fresh worker (no
    playhead) still starts at the top.

Run from the repo root:
    conda run -n smfs-catalog python -m pytest tests/test_navigator_mirror.py
"""
import inspect
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
app = QApplication.instance() or QApplication(sys.argv)

from smfs_catalog import db as _db
from smfs_catalog.analysis_worker import AnalysisWorker, THROTTLE_MS
from smfs_catalog import navigator_bar as nb
from smfs_catalog.navigator_bar import NavigatorBar, WorkerNavBar

tmp = tempfile.mkdtemp(prefix="navbar_")
DB = os.path.join(tmp, "test.sqlite")
_db.initialise(DB)

PATHS = [
    _db.normalize_path(f"/tank/testdata/nav/Image{i:04d}.ibw")
    for i in (1, 2, 3, 4)
]

conn = _db.get_connection(DB)
with conn:
    conn.execute(
        "INSERT INTO watched_directories (id, path, added_at)"
        " VALUES (1, '/tank/testdata/nav', datetime('now'))")
    for p in PATHS:
        conn.execute(
            "INSERT INTO files (path, directory_id, filename, first_seen, last_seen,"
            " experimentalist) VALUES (?, 1, ?, datetime('now'), datetime('now'), 'nav')",
            (p, os.path.basename(p)))
conn.close()

IDS = [_db.get_file_id(p, DB) for p in PATHS]
_db.enqueue_files(IDS, DB)

# One shared idiom for these procedural guards — see checkstyle.py.
import checkstyle                                          # noqa: E402

check = checkstyle.CheckRunner()


# The worker is never start()ed here: this is about the control surfaces and the
# state they read, not about the analysis thread.  A started worker would drain
# the queue and try to load .ibw files that don't exist.
worker = AnalysisWorker(DB)

# ── (f) slider ↔ throttle mapping ────────────────────────────────────────────
check("full-right slider means NO delay",
      nb.slider_to_throttle_ms(nb.SLIDER_MAX) == 0)
check("no delay maps back to the full-right slider",
      nb.throttle_to_slider(0) == nb.SLIDER_MAX)
check("a real delay never reads as full speed",
      nb.throttle_to_slider(200) < nb.SLIDER_MAX)
_mid = (nb.SLIDER_MIN + nb.SLIDER_MAX) // 2
check("throttle round-trips through the slider",
      nb.throttle_to_slider(nb.slider_to_throttle_ms(_mid)) == _mid)
check("no limit is named, not shown as a number",
      nb.throttle_label(0) == "no limit" and "≤" in nb.throttle_label(200))

# ── (a) construction reads, never writes ─────────────────────────────────────
worker.set_throttle_ms(250)
bar_a = NavigatorBar(worker)
check("a new bar leaves the worker's throttle alone",
      worker.throttle_ms() == 250)
check("a new bar shows the worker's value, not its own default",
      bar_a._slider.value() == nb.throttle_to_slider(250))
check("the rate limit is stated in words too", "≤" in bar_a._rate_label.text())

# A bar built while the worker is at its default must not nudge it either.
w2 = AnalysisWorker(DB)
_bar_default = NavigatorBar(w2)
check("a bar over a fresh worker leaves the default throttle untouched",
      w2.throttle_ms() == THROTTLE_MS)

# ── (b) two bars mirror each other ───────────────────────────────────────────
bar_b = NavigatorBar(worker)
check("a second bar opens on the same value", bar_b._slider.value() == bar_a._slider.value())

# The slider has 19 log-spaced positions, so it quantizes: what the worker ends
# up holding is what the CHOSEN POSITION means, not the ms figure asked for.
_pos = nb.throttle_to_slider(1000)
bar_a._slider.setValue(_pos)
check("setting the limit on bar A moves the worker",
      worker.throttle_ms() == nb.slider_to_throttle_ms(_pos) > 0)
check("...and bar B follows without being told",
      bar_b._slider.value() == _pos)

bar_b._btn_clear.click()
check("clearing the limit from bar B clears it on the worker",
      worker.throttle_ms() == 0)
check("...and bar A follows", bar_a._slider.value() == nb.SLIDER_MAX)
check("the clear button is dead once there is no limit",
      not bar_a._btn_clear.isEnabled() and not bar_b._btn_clear.isEnabled())

# ── (c) closing a window does not reset the rate limit ───────────────────────
worker.set_throttle_ms(400)
bar_b.detach()
bar_b.deleteLater()
app.processEvents()
check("a closed window leaves the chosen limit in place", worker.throttle_ms() == 400)
bar_c = NavigatorBar(worker)
check("reopening shows the limit that was still in force",
      bar_c._slider.value() == nb.throttle_to_slider(400))
check("only a new worker (a new launch) starts from the default",
      AnalysisWorker(DB).throttle_ms() == THROTTLE_MS)

# ── (d) transport reflects the worker, not the button that was clicked ───────
bar_a._btn_fwd.click()
app.processEvents()
check("pressing play un-pauses the worker", not worker.is_paused())
check("...and the other bar's play button lights up", bar_c._btn_fwd.isChecked())
check("direction is recorded as forward", worker.direction() > 0)

bar_c._btn_pause.click()
app.processEvents()
check("pausing from another window pauses the worker", worker.is_paused())
check("...and un-checks the first bar's play button", not bar_a._btn_fwd.isChecked())

bar_a._btn_rev.click()
app.processEvents()
check("play-backwards sets the worker's direction", worker.direction() < 0)
check("...and shows as reverse on the other bar",
      bar_c._btn_rev.isChecked() and not bar_c._btn_fwd.isChecked())
worker.set_paused(True)
app.processEvents()

# ── (e) the scrubber follows the queue, and drives the playhead ──────────────
check("scrubber spans the queue", bar_a._scrubber.maximum() == len(IDS) - 1)

bar_a._scrubber.setValue(2)
app.processEvents()
check("moving the scrubber queues a step to that file",
      worker._step_requests and worker._step_requests[-1] == IDS[2])

_db.dequeue_files(IDS[2:], DB)
worker.invalidate_queue_cache()
app.processEvents()
check("shrinking the queue re-ranges every bar's scrubber",
      bar_a._scrubber.maximum() == 1 and bar_c._scrubber.maximum() == 1)

_db.clear_analysis_queue(DB)
worker.invalidate_queue_cache()
app.processEvents()
check("an empty queue is said so, not shown as position 1 of 0",
      bar_a._pos_label.text() == "queue empty")
check("...and its scrubber is not draggable", not bar_a._scrubber.isEnabled())

# The compact diagnostic navigator follows the same cached queue contract. It
# must not issue a full list_queue() JOIN for every playhead signal.
compact = WorkerNavBar(worker, DB)
check("compact navigator starts on the worker's current queue",
      not compact._scrubber.isEnabled() and compact._label.text() == "— / 0")
compact_src = inspect.getsource(WorkerNavBar)
check("compact navigator uses the worker queue cache",
      "_db.list_queue" not in compact_src)

# ── (g) no second transport anywhere ─────────────────────────────────────────
from smfs_catalog import dashboard_window as _dash
from smfs_catalog import rawcurve_window as _raw

_dash_src = inspect.getsource(_dash)
check("the dashboard's separate Play button is gone",
      "_play_btn" not in _dash_src and "_on_play_toggled" not in _dash_src)

_raw_src = inspect.getsource(_raw)
check("the viewer has no navigation loop of its own",
      not any(t in _raw_src for t in ("_nav_timer", "_start_auto", "_toggle_auto")))
check("the viewer no longer defines its own speed-slider maths",
      "_slider_to_rate_hz" not in _raw_src)

# The throttle is the worker's to set, from the navigator only.  Any other
# module calling set_throttle_ms is a second writer of the value #126 is about.
import pathlib
pkg = pathlib.Path(_dash.__file__).parent
writers = sorted(
    p.name for p in pkg.glob("*.py")
    if "set_throttle_ms" in p.read_text()
    and p.name not in ("navigator_bar.py", "analysis_worker.py")
)
check(f"navigator_bar is the only window setting the throttle (found: {writers})",
      not writers)

# ── (h) playing forward honours where the playhead was PUT ───────────────────
# The bug this closes: scrub to the last 100 curves, press play, and the
# playhead snapped back to wherever the batch had stopped — the forward advance
# used to pick whichever came first in queue order out of {in-order neighbour,
# oldest still-pending file}, so a deliberate jump was overruled every time.
_db.enqueue_files(IDS, DB)
worker.invalidate_queue_cache()
app.processEvents()

# Position 2, with 0 and 1 left PENDING behind it — the exact shape the old
# "fresh-work floor" hijacked.
worker._playhead = IDS[2]
check("forward from a scrubbed position goes FORWARD, never back to pending work",
      worker._neighbour_file_id(worker._playhead, +1) == IDS[3])
check("...and idles at the queue edge rather than wrapping back to them",
      worker._neighbour_file_id(IDS[3], +1) is None)
check("reverse from there is still just the previous file",
      worker._neighbour_file_id(worker._playhead, -1) == IDS[1])

# A fresh worker has no playhead, so a new launch still starts at the top —
# removing the floor must not turn "press play" into "resume nowhere".
_w3 = AnalysisWorker(DB)
check("a fresh worker plays forward from the first file",
      _w3._neighbour_file_id(None, +1) == IDS[0])
check("...and backward from the last",
      _w3._neighbour_file_id(None, -1) == IDS[-1])

# Playback walks queue ORDER, never pending STATUS.  A floor reinstated anywhere
# can only fire by overruling a deliberate scrub — see analysis_worker's comment
# above _neighbour_file_id.
_floor = []
for _p in sorted(pkg.glob("*.py")):
    _src = _p.read_text()
    if "def next_pending_file_id" in _src or "db.next_pending_file_id" in _src:
        _floor.append(_p.name)
check(f"nothing steers playback by pending status (found: {_floor})", not _floor)

# Membership changes preserve the current file identity, rather than preserving
# an obsolete slider index that may now point at a different curve.
compact.sync_now()
check("compact navigator expands when files are added",
      compact._scrubber.maximum() == len(IDS) - 1)
worker._playhead = IDS[2]
worker.playhead_changed.emit(IDS[2])
app.processEvents()

_db.dequeue_files([IDS[0]], DB)
worker.invalidate_queue_cache()
app.processEvents()
check("full navigator stays anchored after an earlier row is removed",
      bar_a._scrubber.value() == 1 and bar_a._pos_label.text() == "2 / 3")
check("compact navigator stays anchored after an earlier row is removed",
      compact._scrubber.value() == 1 and compact._label.text() == "2 / 3")

_db.dequeue_files([IDS[2]], DB)
worker.invalidate_queue_cache()
app.processEvents()
check("removing the current file does not select another file implicitly",
      bar_a._pos_label.text() == "— / 2" and compact._label.text() == "— / 2")

print()


# Every check above becomes its own named pytest case.  Must be last:
# pytest_cases reads what the module body recorded.
test_check = checkstyle.pytest_cases(check)
