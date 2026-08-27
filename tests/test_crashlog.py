# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Regression test: the crash log actually catches crashes.

The bug this guards against is a logger that looks installed and writes
nothing when it matters.  So the three fatal cases are tested by CRASHING A
REAL SUBPROCESS, not by calling the hooks directly — calling _excepthook()
by hand would pass even if sys.excepthook were never assigned, which is
precisely the failure mode.

The contract under test:
(a) a clean run writes SESSION START and then CLEAN EXIT.
(b) a hard fault (segfault) writes a C-level traceback — the case that
    otherwise leaves no traceback, no oomd record, no core and no apport
    report.  This is the only writer that survives the interpreter dying.
(c) an unhandled Python exception writes a Python traceback.
(d) an unhandled exception on a plain thread does too — sys.excepthook never
    sees those, so it needs its own hook, and the analysis work in this app
    runs off the GUI thread.
(e) NONE of (b)-(d) write CLEAN EXIT.  This is the "on sight" property: a
    START with no matching CLEAN EXIT is a process that did not come back.
    A logger that marked every ending clean would be worse than useless.
(f) mark_clean_exit() is idempotent — a shutdown path firing twice cannot
    forge a second, tidier ending.
(g) the log rotates at install, keeping the old generation.

Run with the smfs-catalog env, from the repo root:
    python tests/test_crashlog.py
"""
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from smfs_catalog import crashlog

# One shared idiom for these procedural guards — see checkstyle.py for why
# `sys.exit(1)` at the bottom of a file was aborting the whole pytest run.
import checkstyle                                          # noqa: E402

check = checkstyle.CheckRunner()


_tmp = tempfile.mkdtemp(prefix="smfs_crashlog_test_")


def _db_path(name):
    """A DB path in its own directory — the log lands beside it."""
    d = os.path.join(_tmp, name)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "smfs_catalog.db")


def _run_child(db, body):
    """Run body in a fresh interpreter with crash logging installed; return the log text."""
    script = (
        f"import sys; sys.path.insert(0, {_ROOT!r})\n"
        f"from smfs_catalog import crashlog\n"
        f"crashlog.install({db!r})\n"
        + body
    )
    subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    path = crashlog.log_path(db)
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


# -- (a) clean run ------------------------------------------------------------
db_clean = _db_path("clean")
text = _run_child(db_clean, "crashlog.mark_clean_exit()\n")
check("(a) a clean run writes SESSION START", "=== SESSION START" in text)
check("(a) a clean run writes CLEAN EXIT", "=== CLEAN EXIT" in text)

# -- (b) hard fault -----------------------------------------------------------
# faulthandler._sigsegv() dereferences a null pointer on purpose: the closest
# thing available to what killed the app.
db_segv = _db_path("segv")
text = _run_child(db_segv, "import faulthandler; faulthandler._sigsegv()\n")
check("(b) a segfault writes a fatal-error dump",
      "Fatal Python error" in text or "Segmentation fault" in text)
check("(b) the dump names the crashing thread's stack", "File " in text)
check("(e) a segfault does NOT write CLEAN EXIT", "=== CLEAN EXIT" not in text)

# -- (c) unhandled Python exception -------------------------------------------
db_exc = _db_path("exc")
text = _run_child(db_exc, "raise RuntimeError('boom-from-main')\n")
check("(c) an unhandled exception is logged", "=== UNHANDLED EXCEPTION" in text)
check("(c) the exception's message is in the log", "boom-from-main" in text)
check("(c) the traceback is in the log", "RuntimeError" in text and "Traceback" in text)
check("(e) an unhandled exception does NOT write CLEAN EXIT", "=== CLEAN EXIT" not in text)

# -- (d) unhandled exception on a plain thread --------------------------------
db_thr = _db_path("thread")
text = _run_child(
    db_thr,
    "import threading\n"
    "def _boom():\n"
    "    raise ValueError('boom-from-thread')\n"
    "t = threading.Thread(target=_boom, name='worker-under-test')\n"
    "t.start(); t.join()\n",
)
check("(d) a thread exception is logged", "UNHANDLED EXCEPTION IN THREAD" in text)
check("(d) the thread's name is recorded", "worker-under-test" in text)
check("(d) the thread exception's message is in the log", "boom-from-thread" in text)
check("(e) a thread exception does NOT write CLEAN EXIT", "=== CLEAN EXIT" not in text)

# -- (f) CLEAN EXIT is written once -------------------------------------------
db_twice = _db_path("twice")
text = _run_child(db_twice, "crashlog.mark_clean_exit()\ncrashlog.mark_clean_exit()\n")
check("(f) mark_clean_exit is idempotent", text.count("=== CLEAN EXIT") == 1)

# -- (g) rotation at install --------------------------------------------------
db_rot = _db_path("rotate")
path = crashlog.log_path(db_rot)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text("x" * (crashlog.MAX_BYTES + 1), encoding="utf-8")
_run_child(db_rot, "crashlog.mark_clean_exit()\n")
rolled = path.with_suffix(path.suffix + ".1")
check("(g) an oversized log is rolled to crash.log.1", rolled.exists())
check("(g) the rolled file keeps the old content",
      rolled.exists() and rolled.stat().st_size > crashlog.MAX_BYTES)
check("(g) the new log starts fresh",
      path.exists() and path.stat().st_size < crashlog.MAX_BYTES)


# Every check above becomes its own named pytest case.  Must be last:
# pytest_cases reads what the module body recorded.
test_check = checkstyle.pytest_cases(check)
