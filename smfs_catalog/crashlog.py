# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/crashlog.py
"""
Crash log. Three writers, covering three different ways a process can end:

  faulthandler        a hard fault (segfault, abort, stack overflow) — writes a
                      C-level traceback for every thread straight to the fd.
                      This is the only one that survives the interpreter itself
                      going down, which is the case that left nothing.
  sys.excepthook      an unhandled Python exception on any thread PyQt hands
                      back to us.  PyQt6 routes an exception raised inside a
                      slot through sys.excepthook and then aborts the process,
                      so a GUI-callback crash reaches this hook first and then
                      faulthandler on the abort — both halves get logged.
  threading.excepthook  an unhandled exception on a plain (non-Qt) thread,
                      which sys.excepthook never sees.

Plus the fourth thing, which is what actually makes the file readable: every
run writes a SESSION START line, and a clean shutdown writes a CLEAN EXIT
line.  A session with a START and no matching CLEAN EXIT died, and you can see
that at a glance without understanding anything else in the file.  That
distinction is the whole point — "the app was closed" and "the app vanished"
look identical in the morning otherwise.

WHERE: next to the DB (<db dir>/crash.log), because that directory is already
the app's per-user state and is resolved identically on all three platforms
(db.default_db_path).  install() creates it, so the log covers db.initialise()
— migrations included — rather than starting after it.

ROTATION HAPPENS AT INSTALL ONLY, never while running.  faulthandler holds the
raw file descriptor for the life of the process and writes to it from a fault
handler; renaming the file out from under it mid-run would send the one
traceback we care about into an unlinked inode.  So the roll is done before
faulthandler.enable() and not after.  A single run therefore can exceed
MAX_BYTES — correct trade: a bounded pile of files, and never a lost dump.

Several app instances appending to one file interleave, which is why every
line carries the pid.

Deliberately NOT here, so nobody re-derives the search:
  * faulthandler.register(SIGTERM) — would say who killed us, but the observed
    crash left no signal record at all, and the OOM killer uses SIGKILL, which
    is uncatchable by anything.  Out of scope; not free either,
    since it needs chain=True or the process stops honouring `kill`.
  * Hunting the crash this was written for.  It is unreproducible with
    nothing to bisect from, so a hypothesis-driven rebuild of the 2DH path
    would be optimising against a guess.

v2 replaces the "go and read a file" part of this with a message/log display in
the consolidated shell.  The file stays either way.
"""
import datetime
import faulthandler
import os
import platform
import sys
import threading
import traceback
from pathlib import Path

from . import __version__

LOG_FILENAME = "crash.log"

# Roll when the existing log is already this big AT STARTUP, keeping KEEP
# older generations (crash.log.1 … crash.log.N).  Small: these are text lines
# and occasional tracebacks, not a data stream.
MAX_BYTES = 2_000_000
KEEP = 5

# Module-level, and it must stay that way: faulthandler writes to this file's
# descriptor from a fault handler, so if the object were collected the fd would
# close and the dump would go nowhere.
_handle = None
_installed_path: Path | None = None
_clean_exit_written = False


def log_path(db_path: str) -> Path:
    """The crash log for a given DB — <db dir>/crash.log."""
    return Path(db_path).parent / LOG_FILENAME


def _timestamp() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _rotate(path: Path) -> None:
    """
    Roll path -> path.1 -> path.2 … dropping the oldest.  Called only from
    install(), before faulthandler holds the fd (see module docstring).
    """
    if not path.exists() or path.stat().st_size < MAX_BYTES:
        return
    oldest = path.with_suffix(path.suffix + f".{KEEP}")
    if oldest.exists():
        oldest.unlink()
    for n in range(KEEP - 1, 0, -1):
        src = path.with_suffix(path.suffix + f".{n}")
        if src.exists():
            src.rename(path.with_suffix(path.suffix + f".{n + 1}"))
    path.rename(path.with_suffix(path.suffix + ".1"))


def _write(text: str) -> None:
    """
    Append text to the log, flushed immediately.  Never raises: a logger that
    can take the app down with it is worse than no logger.  faulthandler
    bypasses this and writes to the fd directly, so flushing here keeps our
    lines in order relative to a dump.
    """
    if _handle is None:
        return
    try:
        _handle.write(text if text.endswith("\n") else text + "\n")
        _handle.flush()
    except Exception:
        pass


def _excepthook(exc_type, exc_value, exc_tb) -> None:
    _write(
        f"\n=== UNHANDLED EXCEPTION {_timestamp()} pid={os.getpid()} "
        f"thread={threading.current_thread().name}\n"
        + "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    )
    # Chain to whatever was there before so stderr behaves exactly as it did
    # and PyQt's own post-hook handling is untouched.
    sys.__excepthook__(exc_type, exc_value, exc_tb)


def _thread_excepthook(args) -> None:
    # Plain threading.Thread failures never reach sys.excepthook.
    if args.exc_type is SystemExit:
        return
    _write(
        f"\n=== UNHANDLED EXCEPTION IN THREAD {_timestamp()} pid={os.getpid()} "
        f"thread={getattr(args.thread, 'name', '?')}\n"
        + "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
    )


def install(db_path: str) -> Path:
    """
    Turn on crash logging for this process and write the SESSION START line.

    Call once, as early as possible in the entry point — before QApplication
    and before db.initialise(), so a fault in either is still captured.
    Creates the log's directory if it does not exist yet.  Returns the log
    path so the caller can tell the user where it is.
    """
    global _handle, _installed_path, _clean_exit_written
    if _handle is not None:
        return _installed_path

    path = log_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _rotate(path)

    # Line-buffered append.  faulthandler needs a real fileno(), which this has.
    _handle = open(path, "a", buffering=1, encoding="utf-8", errors="replace")
    _installed_path = path
    _clean_exit_written = False

    faulthandler.enable(file=_handle, all_threads=True)
    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook

    _write(
        f"\n=== SESSION START {_timestamp()} pid={os.getpid()} "
        f"smfs_catalog {__version__} "
        f"python {platform.python_version()} {platform.system()} {platform.release()} "
        f"db={db_path}"
    )
    return path


def connect_clean_exit(app) -> None:
    """
    Write CLEAN EXIT when the Qt event loop shuts down normally.

    Kept separate from install() because install() runs before QApplication
    exists — the log has to cover the startup that constructs it.
    """
    app.aboutToQuit.connect(mark_clean_exit)


def mark_clean_exit() -> None:
    """
    The marker that makes an abnormal exit visible on sight: a SESSION START
    with no CLEAN EXIT after it is a process that did not come back.  Written
    at most once per run, so a shutdown path that fires twice cannot forge a
    second, tidier-looking ending.
    """
    global _clean_exit_written
    if _clean_exit_written:
        return
    _clean_exit_written = True
    _write(f"=== CLEAN EXIT {_timestamp()} pid={os.getpid()}")
