#!/usr/bin/env python3
# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# run_dashboard.py
#
# Launch the SMFS Catalog GUI dashboard.
#
#   python run_dashboard.py
#   python run_dashboard.py /path/to/smfs_catalog.db   ← custom DB path
#
# With no argument the per-user default location is used (see
# smfs_catalog.db.default_db_path); $SMFS_DB_PATH overrides it.
#
# If the shell has no DISPLAY (headless ssh / VSCode terminal), the dashboard
# falls back to the persistent xpra display :100, starting the xpra session
# first if needed. View it remotely with:
#   xpra attach ssh://<user>@<this-host>/100

import atexit
import getpass
from pathlib import Path
import os
import shutil
import socket
import subprocess
import sys
import time

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QMessageBox

from smfs_catalog import crashlog as _crashlog
from smfs_catalog import db as _db
from smfs_catalog import sample_marks as _sample_marks
from smfs_catalog.dashboard_window import DashboardWindow

XPRA_DISPLAY = ":100"

# Set only by the branch of _ensure_display() that actually runs `xpra start`.
# A session we merely FOUND already live belongs to whoever started it — reusing
# it is the whole point of a persistent display, and stopping it on our way out
# would take somebody else's windows down with it.
_xpra_started_by_us = False


def _frozen() -> bool:
    """True inside a PyInstaller bundle (the thing colleagues are handed)."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def resource_path(*parts: str) -> Path:
    """
    Locate a bundled data file (icons, and whatever the manual needs later).

    In a checkout that is simply next to this script.  PyInstaller instead
    unpacks the bundle to a temporary directory and points sys._MEIPASS at it,
    so `Path(__file__).parent` is the wrong answer there — it resolves inside
    the one-file stub, where the assets are not.  The visible symptom was a
    windowless-looking app: no icon in the task bar, no error, because the
    caller checks .exists() and simply skips.
    """
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base.joinpath(*parts)


def _stop_xpra_session() -> None:
    """
    Stop the xpra session this process started, if it started one (#33).

    WHY ATEXIT AND NOT AN XPRA FLAG.  The two flags that look right are both
    wrong here.  `--exit-with-children` requires xpra to launch the dashboard,
    but Qt reads DISPLAY once at QApplication construction, so the display has
    to exist before the app does — the app cannot be xpra's child.
    `--exit-with-client` stops the server whenever the viewer detaches, which
    is precisely what happens when the operator disconnects and goes home
    partway through an unattended run; it would kill the session out from
    under a job that is still working.  The session must outlive a detached
    client and die with the application, and only a hook in this process
    expresses that.

    KNOWN LIMIT: atexit covers a normal exit and an unhandled exception, but
    not SIGKILL or os._exit().  `xpra stop :100` by hand remains the backstop
    for a hard kill.  This is why the orphan is a leak and not a corruption —
    the worst case is a session that outlives us, exactly as before.
    """
    global _xpra_started_by_us
    if not _xpra_started_by_us:
        return
    _xpra_started_by_us = False        # never attempt the stop twice
    stopped = subprocess.run(
        ["xpra", "stop", XPRA_DISPLAY], capture_output=True, text=True)
    if stopped.returncode == 0:
        print(f"Stopped xpra session {XPRA_DISPLAY}.")
    else:
        # Not fatal: we are on the way out, and a session we failed to stop is
        # the status quo ante, not a new problem.  Say so rather than dying in
        # an atexit handler, where the traceback would bury the real exit.
        print(
            f"Warning: could not stop xpra session {XPRA_DISPLAY}: "
            f"{stopped.stderr.strip()}",
            file=sys.stderr)


def _ensure_display() -> None:
    # Must run before QApplication is created: Qt reads DISPLAY exactly once,
    # at construction, and aborts the process if it can't connect.
    #
    # Development-only.  This exists so a headless ssh/VSCode session on the remote machine
    # can put the GUI on the persistent xpra display; on a colleague's
    # machine the same code would shell out to `xpra` — a program they do not
    # have and have no reason to want — or, worse, sys.exit() with a message
    # about installing it, from a double-clicked icon that shows no terminal.
    # A frozen build has a real desktop by definition: if DISPLAY is unset
    # there, that is Qt's error to report, not ours to work around.
    if _frozen():
        return
    if not sys.platform.startswith("linux") or os.environ.get("DISPLAY"):
        return
    if shutil.which("xpra") is None:
        sys.exit("No DISPLAY is set and xpra is not installed; cannot open GUI windows.")
    listing = subprocess.run(["xpra", "list"], capture_output=True, text=True)
    if f"LIVE session at {XPRA_DISPLAY}" not in listing.stdout:
        print(f"No DISPLAY set — starting xpra session {XPRA_DISPLAY} …")
        started = subprocess.run(["xpra", "start", XPRA_DISPLAY], capture_output=True, text=True)
        if started.returncode != 0:
            sys.exit(f"'xpra start {XPRA_DISPLAY}' failed:\n{started.stderr.strip()}")
        # Claim it BEFORE the socket wait below, not after.  That wait can
        # sys.exit() on timeout, and by then the server may well be up — the
        # timeout is on the socket appearing, not on the start succeeding.
        # Registering here means even that exit path takes the session with it.
        global _xpra_started_by_us
        _xpra_started_by_us = True
        atexit.register(_stop_xpra_session)
        # xpra daemonizes immediately; wait for the X socket before letting Qt try.
        socket_path = "/tmp/.X11-unix/X" + XPRA_DISPLAY.lstrip(":")
        deadline = time.monotonic() + 15
        while not os.path.exists(socket_path):
            if time.monotonic() > deadline:
                sys.exit(f"xpra display {XPRA_DISPLAY} did not come up within 15 s.")
            time.sleep(0.25)
    os.environ["DISPLAY"] = XPRA_DISPLAY
    print(
        f"Using xpra display {XPRA_DISPLAY} — view with:\n"
        f'  "C:\\Program Files\\Xpra\\Xpra_cmd.exe" attach '
        f"ssh://{getpass.getuser()}@{socket.gethostname().split('.')[0]}/{XPRA_DISPLAY.lstrip(':')}"
    )


def main(db_path: str = _db.DEFAULT_DB_PATH) -> None:
    # First of everything, so a fault in initialise()'s migrations or in
    # QApplication's own startup is still written down.  A silent death is
    # indistinguishable from a finished overnight batch without this.
    log_file = _crashlog.install(db_path)
    print(f"Crash log: {log_file}")   # no-op under --windowed; sys.stdout is None
    _ensure_display()
    # Check BEFORE initialise() stamps a fresh DB with this machine — otherwise a
    # DB copied from another machine would look native the instant we open it.
    machine_warning = _db.check_db_machine(db_path)
    _db.initialise(db_path)
    # Before any window builds a plot, so the first curve drawn is already in
    # the mode this catalog was left in.
    _sample_marks.load(db_path)
    app = QApplication(sys.argv)
    _crashlog.connect_clean_exit(app)
    icon_path = resource_path("smfs_catalog", "assets", "icons", "icon.png")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    if machine_warning:
        QMessageBox.warning(None, "Catalog built on a different machine", machine_warning)
    win = DashboardWindow(db_path)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else _db.DEFAULT_DB_PATH
    main(db_path)
