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
from smfs_catalog.dashboard_window import DashboardWindow

XPRA_DISPLAY = ":100"


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
