# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/add_data_dialog.py
#
# "Add data" dialog: register a new watched directory + scan it.
#
#   1. db.add_directory(path)
#   2. scanner.scan_directory(path, dir_id) — populates files table
#
# No metadata is entered here (#110, 2026-07-29). Descriptive sample metadata
# (experimentalist/analyte/solvent/instrument/cantilever/technique) is
# file-level now, populated AFTER import via the dashboard's scope-based
# "Define metadata for these files..." dialog — one single place that writes
# these fields, not two that can disagree. The one exception is parent-mode
# below, which still auto-infers experimentalist from folder structure (a
# plain path heuristic, not manual entry) and writes it straight onto each
# scanned file. The old per-directory metadata form and the "Auto-fill from
# notes (.txt)" Claude-extraction button are retired outright — see
# CLAUDE.md's dated entry; the code lives on under legacy/ at the repo root.

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QDialogButtonBox, QFileDialog, QMessageBox,
    QCheckBox, QApplication,
)

from . import db as _db
from . import scanner as _scanner
from . import style
from .qt_utils import CancelableProgress, fit_on_screen


class _ScanProgress:
    """
    Adapts the scanner's Qt-free `(done, total, label) -> cancelled` callback
    to a CancelableProgress dialog (#124).

    Registering 6,007 files took 39.6 minutes with no progress bar, no
    spinner, no status text and not even a busy cursor on this path — the only
    way to tell "working" from "hung" was that hovering a button gave no
    highlight.  The scanner knew the count the whole time and threw it away,
    because the GUI called it with console=None.

    The progress dialog is created LAZILY, on the first tick: the scanner only
    knows how many files there are after it has walked the tree, and a bar
    with a made-up total is worse than none.  The wait cursor covers that
    walk, and is dropped the moment the dialog appears — a wait cursor over a
    button you are meant to be able to click is a lie about whether you can.
    """

    def __init__(self, parent, title: str) -> None:
        self._parent = parent
        self._title = title
        self._dlg: CancelableProgress | None = None
        self.cancelled = False
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._cursor_set = True

    def _drop_cursor(self) -> None:
        if self._cursor_set:
            QApplication.restoreOverrideCursor()
            self._cursor_set = False

    def __call__(self, done: int, total: int, label: str = "") -> bool:
        if self._dlg is None:
            self._drop_cursor()
            self._dlg = CancelableProgress(self._parent, self._title, total)
        self.cancelled = self._dlg.tick(
            done, total, f"{self._title}\n{label}" if label else self._title)
        return self.cancelled

    def close(self) -> None:
        self._drop_cursor()
        if self._dlg is not None:
            self._dlg.close()
            self._dlg = None


class AddDataDialog(QDialog):
    """
    Modal dialog.  Returns Accepted after successful directory registration +
    scan.  The dashboard re-populates options after this returns.
    """

    def __init__(self, db_path: str = _db.DEFAULT_DB_PATH, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add data")
        fit_on_screen(self, 560, 200)
        self._db_path = db_path

        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        hint = QLabel(
            "Register a directory of .ibw files.  After confirming, the "
            "catalog scans it and adds every parseable file to the database. "
            "Sample metadata (analyte, solvent, instrument, cantilever, "
            "experimentalist, ...) is set afterward from the dashboard, via "
            "\"Define metadata for these files...\", over whatever scope you "
            "select."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(style.qss_text())
        outer.addWidget(hint)

        # Directory + browse
        dir_row = QHBoxLayout()
        self._dir_edit = QLineEdit()
        self._dir_edit.setPlaceholderText("e.g. D:/Joseph/240601_titin/")
        self._dir_edit.setToolTip(
            "Every .ibw file below this folder is registered, recursively. "
            "Files already in the catalog are skipped unless their modification "
            "time has changed, so re-running this over the same folder is cheap "
            "and safe.")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._on_browse)
        dir_row.addWidget(self._dir_edit, 1)
        dir_row.addWidget(browse)
        form_lbl = QLabel("Data location")
        form_lbl.setStyleSheet(style.QSS_EMPHASIS)
        outer.addWidget(form_lbl)
        outer.addLayout(dir_row)

        # Parent-of-many mode: register each session subfolder as its own unit,
        # inferring the experimentalist from the folder structure
        # (/<this folder>/<experimentalist>/<session>/…/*.ibw). This is the
        # "point at the drive and read everything" path.
        self._parent_chk = QCheckBox(
            "This is a parent of many session folders — auto-detect experimentalist "
            "from subfolder names and register each session"
        )
        outer.addWidget(self._parent_chk)

        outer.addStretch(1)

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        box.accepted.connect(self._on_accept)
        box.rejected.connect(self.reject)
        outer.addWidget(box)

    def _on_browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select data directory")
        if path:
            self._dir_edit.setText(path)

    def _confirm_no_overlap(self, path: str) -> bool:
        """
        Warn on a directory-scope overlap with an already-registered directory
        (#68) — registering a parent after its children re-walks them
        (silently duplicating rows); registering a child already covered by a
        registered parent is redundant.  Returns False (caller should abort)
        only if the user declines to proceed past the warning.
        """
        overlaps = _db.find_overlapping_directories(path, self._db_path)
        if not overlaps:
            return True
        lines = [
            (f"  - {existing} already covers this directory" if rel == "ancestor"
             else f"  - {existing} is inside this directory and would be re-scanned")
            for existing, rel in overlaps
        ]
        r = QMessageBox.warning(
            self, "Directory overlap",
            "This directory overlaps an already-registered one:\n\n"
            + "\n".join(lines)
            + "\n\nScanning it may re-walk already-registered files. Proceed anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return r == QMessageBox.StandardButton.Yes

    def _on_accept(self) -> None:
        path = self._dir_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Add data", "Pick a directory first.")
            return

        if not self._confirm_no_overlap(path):
            return

        if self._parent_chk.isChecked():
            self._accept_parent_tree(path)
            return

        # 1) Register directory (or note it was already registered)
        added = _db.add_directory(path, self._db_path)
        row = _db.get_directory_by_path(path, self._db_path)
        if row is None:
            QMessageBox.critical(self, "Add data", "Directory registration failed.")
            return
        dir_id = row["id"]

        # 2) Scan — no metadata entry here (see module header); every
        # descriptive field starts NULL until "Define metadata..." is run.
        # scan_directory calls mark_directory_scanned itself, and skips it on
        # cancel; this method must NOT call it too, or a cancelled scan would
        # be recorded as a complete one.
        prog = _ScanProgress(self, f"Scanning {Path(path).name}…")
        try:
            n_found, n_updated, n_errors, cancelled = _scanner.scan_directory(
                path, dir_id, self._db_path, progress_cb=prog,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Scan failed", f"Scan raised: {exc!r}")
            return
        finally:
            prog.close()

        if cancelled:
            QMessageBox.information(
                self, "Scan cancelled",
                f"Stopped after {n_updated:,} of {n_found:,} file(s).\n\n"
                "What was scanned is kept — this directory is left marked "
                "not-fully-scanned, so running Add data on it again picks up "
                "where this stopped.\n\nTo undo the import entirely, use "
                "'Remove these files…' on the dashboard.",
            )
            self.accept()
            return

        msg = (
            ("Registered new directory.\n" if added else "Directory already registered; rescanned.\n")
            + f"Found {n_found} .ibw files\n"
            + f"  added/updated: {n_updated}\n"
            + f"  errors: {n_errors}"
        )
        QMessageBox.information(self, "Scan complete", msg)
        self.accept()

    def _accept_parent_tree(self, path: str) -> None:
        """
        Parent mode: register every session subfolder under `path` as its own
        watched_directory, inferring the experimentalist from the folder names
        and writing it onto each scanned file (#110 — experimentalist is
        file-level, not directory-level). This is the "point at the drive and
        read everything" path — one flat registration would leave the whole
        tree unfilterable (nowhere to hang per-session metadata).
        """
        prog = _ScanProgress(self, "Scanning folders…")
        try:
            summary = _scanner.scan_tree(
                path, self._db_path,
                # Explicit rather than relying on the default: this decides
                # whether folder names touch files.experimentalist at all.
                infer_experimentalist=True,
                progress_cb=prog,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Scan failed", f"Tree scan raised: {exc!r}")
            return
        finally:
            prog.close()

        exps = summary["experimentalists"]
        unmatched = summary.get("unmatched_experimentalists", [])
        head = (
            "Scan cancelled — what was scanned is kept.\n\n"
            if summary["cancelled"] else ""
        )
        body = (
            head
            + f"Registered {summary['n_dirs']} session directories\n"
            + f"Found {summary['n_files']} .ibw files ({summary['n_errors']} errors)\n"
            + f"Assigned to ({len(exps)}): {', '.join(exps) if exps else '—'}"
        )
        if unmatched:
            # Say what was NOT assigned and why. A folder naming nobody in the
            # catalog is the normal case for a new person — and also what a
            # mis-aimed import root looks like ("afm", "260319"), which is
            # exactly why it is shown rather than silently written.
            body += (
                f"\n\nNot assigned — no such experimentalist yet "
                f"({len(unmatched)}): {', '.join(unmatched)}\n"
                "Those files were left unattributed. Set them with "
                "\"Define metadata for these files…\", which is also where a "
                "new person gets added."
            )
        QMessageBox.information(
            self, "Scan cancelled" if summary["cancelled"] else "Scan complete",
            body,
        )
        self.accept()
