# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Regression test: closing the dashboard quits, and offers to keep the queue
.

The contract under test:
(a) An empty queue asks NOTHING.  The prompt is not a nag; it appears only when
    there is something to lose.
(b) Cancel leaves the session exactly as it was — window open, worker running,
    queue untouched.  Nothing may be closed before the queue question is
    answered.
(c) Save writes a file Load Queue can read back (a `path` column), and only
    then quits.
(d) Don't save quits and writes nothing.
(e) Child windows close with the dashboard, and the app quits — the dashboard
    is the hub, and a surviving child window would hold the process open with
    no way to get the dashboard back.

Run with the smfs-catalog env, from the repo root:
    python tests/test_dashboard_close.py
"""
import csv
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QMessageBox, QWidget
app = QApplication.instance() or QApplication(sys.argv)

from smfs_catalog import db as _db
from smfs_catalog import dashboard_window as _dash
from smfs_catalog.dashboard_window import DashboardWindow

tmp = tempfile.mkdtemp(prefix="dashclose_")
DB = os.path.join(tmp, "test.sqlite")
_db.initialise(DB)

PATHS = [
    _db.normalize_path(f"/tank/testdata/close/Image{i:04d}.ibw")
    for i in (1, 2, 3)
]

conn = _db.get_connection(DB)
with conn:
    for p in PATHS:
        conn.execute(
            "INSERT INTO files (path, filename, first_seen, last_seen,"
            " experimentalist) VALUES (?, ?, datetime('now'), datetime('now'), 'close')",
            (p, os.path.basename(p)))
conn.close()

IDS = [_db.get_file_id(p, DB) for p in PATHS]

# One shared idiom for these procedural guards — see checkstyle.py.
import checkstyle                                          # noqa: E402

check = checkstyle.CheckRunner()


# ── Test rig ─────────────────────────────────────────────────────────────────
# The prompt is a real modal QMessageBox, so answering it means clicking a
# button from a timer while exec() spins its nested loop.  `answers` is drained
# in order, which also lets us prove a dialog appeared at all (an unconsumed
# answer means it never opened) and that none appeared when none should.

answers: list[str] = []
seen: list[str] = []

def _pump():
    w = app.activeModalWidget()
    if isinstance(w, QMessageBox):
        seen.append(w.windowTitle())
        want = answers.pop(0) if answers else None
        for b in w.buttons():
            if want is not None and b.text().replace("&", "") == want:
                b.click()
                return
        w.reject()      # unexpected dialog — dismiss so the test can't hang

_timer = QTimer()
_timer.timeout.connect(_pump)
_timer.start(5)

# Closing the dashboard calls QApplication.quit(), which would kill the test's
# own event loop.  It resolves through dashboard_window's module global, so a
# stand-in records the call instead — this IS the assertion for "closing the
# dashboard quits the app".
quits: list[int] = []
class _QuitSpy:
    @staticmethod
    def quit():
        quits.append(1)
_real_qapp = _dash.QApplication
_dash.QApplication = _QuitSpy


def new_dashboard(queue: bool):
    """A shown dashboard.  Its __init__ clears the queue, so fill it after."""
    win = DashboardWindow(DB)
    win.show()
    if queue:
        _db.enqueue_files(IDS, DB)
        win._worker.invalidate_queue_cache()
    app.processEvents()
    return win


def exports() -> set:
    # Exports land in the DB's own directory (no override set).  The DB and its
    # transient -wal/-shm sidecars, which come and go as connections open, are
    # not exports.
    stem = os.path.basename(DB)
    return {p for p in os.listdir(tmp) if not p.startswith(stem)}


# ── (b) Cancel changes nothing ───────────────────────────────────────────────
win = new_dashboard(queue=True)
answers[:] = ["Cancel"]
seen.clear()
quits.clear()
win.close()
app.processEvents()
check("a non-empty queue is asked about on the way out", len(seen) == 1)
check("Cancel leaves the window open", win.isVisible())
check("Cancel leaves the worker running", win._worker.isRunning())
check("Cancel does not quit the app", not quits)
check("Cancel leaves the queue alone", len(_db.queue_paths(DB)) == len(IDS))

# ── (c) Save writes a loadable queue file, then quits ────────────────────────
before = exports()
answers[:] = ["Save", "OK"]       # the prompt, then _on_save_queue's confirmation
seen.clear()
win.close()
app.processEvents()
written = sorted(exports() - before)
csvs = [f for f in written if f.endswith(".csv")]
check("Save writes a queue file", len(csvs) == 1)
saved_paths = []
if csvs:
    with open(os.path.join(tmp, csvs[0]), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    saved_paths = [r.get("path") for r in rows]
check("...with the `path` column Load Queue needs, holding every queued file",
      saved_paths == PATHS)
check("...and a manifest beside it, as every export has",
      any(f.endswith(".json") for f in written))
check("Save then quits", bool(quits))
check("...having stopped the worker", not win._worker.isRunning())
check("...and closed the window", not win.isVisible())

# The saved file is the round trip: it must repopulate the queue it came from.
_db.clear_analysis_queue(DB)
n_enq, n_missing = _db.import_queue_from_paths(saved_paths, DB)
check("the saved queue loads straight back in", n_enq == len(IDS) and n_missing == 0)

# ── (d) Don't save quits and writes nothing ──────────────────────────────────
win = new_dashboard(queue=True)
before = exports()
answers[:] = ["Don't save"]
seen.clear()
quits.clear()
win.close()
app.processEvents()
check("Don't save asks once", len(seen) == 1)
check("Don't save writes no file", exports() == before)
check("Don't save quits", bool(quits))

# ── (a) an empty queue asks nothing, and (e) children go down with the hub ───
win = new_dashboard(queue=False)
child = QWidget()
child.setWindowTitle("pretend child window")
child.show()
win._children.append(child)
app.processEvents()

answers.clear()
seen.clear()
quits.clear()
win.close()
app.processEvents()
check("an empty queue is not asked about at all", not seen)
check("an empty queue closes without a prompt", not win.isVisible())
check("closing the dashboard closes the windows it opened", not child.isVisible())
check("closing the dashboard quits the app", bool(quits))
check("the worker is stopped on the way out", not win._worker.isRunning())

_dash.QApplication = _real_qapp
_timer.stop()
print()


# Every check above becomes its own named pytest case.  Must be last:
# pytest_cases reads what the module body recorded.
test_check = checkstyle.pytest_cases(check)
