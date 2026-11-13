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
import sqlite3
import subprocess
import sys
import time

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from smfs_catalog import crashlog as _crashlog
from smfs_catalog import db as _db
from smfs_catalog import sample_marks as _sample_marks
from smfs_catalog import scanner as _scanner
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
    Locate a bundled data file, such as the icons.

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
    Stop the xpra session this process started, if it started one.

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


def _size_text(n_bytes: int) -> str:
    if n_bytes >= 1e9:
        return f"{n_bytes / 1e9:.1f} GB"
    return f"{n_bytes / 1e6:.1f} MB"


def _catalog_size(db_path: str) -> int:
    """The catalog's size on disk, its write-ahead log included."""
    return sum(os.path.getsize(p) for p in (db_path, db_path + "-wal")
               if os.path.exists(p))


def open_catalog(db_path: str, confirm_upgrade, refuse, progress=None) -> bool:
    """
    Bring the catalog at `db_path` to this app's version, or say why not.

    Nothing is written before the stored version is read.  A catalog newer
    than this app, or a file that is not a catalog, is refused and left as it
    was.  An older catalog is upgraded only once `confirm_upgrade(text)` has
    returned True and its backup is written; with no room on the disk for the
    backup it is refused unchanged.  A new or current catalog opens as before.

    The dialogs come in as `confirm_upgrade(text) -> bool` and `refuse(text)`,
    so the decision runs the same with or without a display, and in a frozen
    build exactly as from a checkout.  `progress(done, total) -> bool` follows
    the backup; returning False cancels it, and the catalog is left unchanged.
    Returns True when the catalog may open.
    """
    try:
        found = _db.catalog_version(db_path)
    except sqlite3.DatabaseError as exc:
        refuse(f"{db_path} cannot be read as a catalog ({exc}).\n\n"
               f"It has not been changed.")
        return False
    if found is not None and found > _db.CATALOG_VERSION:
        refuse(_db.too_new_message(db_path, found))
        return False

    if found is not None and found < _db.CATALOG_VERSION:
        backup = _db.backup_path(db_path)
        size = _catalog_size(db_path)
        free = shutil.disk_usage(os.path.dirname(os.path.abspath(db_path))).free
        if free < size:
            refuse(f"{db_path} needs upgrading, and its backup needs "
                   f"{_size_text(size)}, but only {_size_text(free)} is free on "
                   f"that disk.\n\nThe catalog has not been changed.")
            return False
        n_outdated = _scanner.count_outdated_curve_types(db_path)
        text = (
            f"{db_path} was written by an older version of SMFS Catalog "
            f"(catalog version {found}; this version uses "
            f"{_db.CATALOG_VERSION}).\n\n"
            f"Upgrade first copies it to\n{backup}\n({_size_text(size)}, "
            f"replacing any backup already there), then brings the catalog up "
            f"to date.  Quit leaves it unchanged."
        )
        if n_outdated:
            text += (
                f"\n\n{n_outdated:,} "
                f"{'curve carries' if n_outdated == 1 else 'curves carry'} a "
                f"curve type from the older "
                f"version.  After the upgrade, the dashboard's \"Re-check "
                f"catalogued files…\" button, then \"Fill in what's missing\", "
                f"re-reads those files and puts their type right."
            )
        if not confirm_upgrade(text):
            return False
        # Read again: while the dialog was open another window may have
        # upgraded it, and its backup must not be replaced by an upgraded copy.
        if _db.catalog_version(db_path) == found:
            try:
                _db.backup_catalog(db_path, backup, progress=progress)
            except _db.BackupCancelled:
                return False

    _db.initialise(db_path)
    return True


class _BackupProgress:
    """The backup's progress window, with a Cancel button, opened at its first step.

    It is shown from the first step, however short the copy, so the person who
    clicked Upgrade always sees what is happening.
    """

    def __init__(self, db_path: str):
        self._text = (f"Copying {os.path.basename(db_path)} to "
                      f"{os.path.basename(_db.backup_path(db_path))}")
        self.box = None

    def __call__(self, done: int, total: int) -> bool:
        if self.box is None:
            self.box = QProgressDialog(self._text, "Cancel", 0, 100)
            self.box.setWindowTitle("Backing up catalog")
            self.box.setWindowModality(Qt.WindowModality.ApplicationModal)
            self.box.setMinimumDuration(0)
        self.box.setValue(round(100 * done / total) if total else 100)
        QApplication.processEvents()
        return not self.box.wasCanceled()

    def close(self) -> None:
        if self.box is not None:
            self.box.close()


def _upgrade_box(text: str):
    """The upgrade question, and its Upgrade button."""
    box = QMessageBox(QMessageBox.Icon.Question, "Upgrade catalog", text)
    upgrade = box.addButton("Upgrade", QMessageBox.ButtonRole.AcceptRole)
    quit_button = box.addButton("Quit", QMessageBox.ButtonRole.RejectRole)
    # Enter and Escape both press Quit: a stray keystroke changes nothing, and
    # the app asks again at the next start.
    box.setDefaultButton(quit_button)
    box.setEscapeButton(quit_button)
    return box, upgrade


def _confirm_upgrade_dialog(text: str) -> bool:
    box, upgrade = _upgrade_box(text)
    box.exec()
    return box.clickedButton() is upgrade


def _refuse_dialog(text: str) -> None:
    QMessageBox.critical(None, "Catalog not opened", text)


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
    # Before the catalog is opened: an older or newer one is decided in a dialog.
    app = QApplication(sys.argv)
    _crashlog.connect_clean_exit(app)
    icon_path = resource_path("smfs_catalog", "assets", "icons", "icon.png")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    backup_progress = _BackupProgress(db_path)
    opened = open_catalog(db_path, _confirm_upgrade_dialog, _refuse_dialog,
                          backup_progress)
    backup_progress.close()
    if not opened:
        # Stopping here is a decision, not a crash; the event loop that would
        # otherwise say so never starts.
        _crashlog.mark_clean_exit()
        return
    # Before any window builds a plot, so the first curve drawn is already in
    # the mode this catalog was left in.
    _sample_marks.load(db_path)
    if machine_warning:
        QMessageBox.warning(None, "Catalog built on a different machine", machine_warning)
    win = DashboardWindow(db_path)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else _db.DEFAULT_DB_PATH
    main(db_path)
