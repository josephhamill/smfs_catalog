# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/add_data_dialog.py
#
# "Add data": point the catalog at a folder of .ibw files and scan it.
#
# Sample metadata is not entered here.  It is file-level, and
# BulkMetadataDialog is the single place that writes it.  The one exception
# is parent mode, which infers experimentalist from folder names.

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
    Adapts the scanner's Qt-free `(done, total, label) -> cancelled`
    callback to a CancelableProgress dialog.

    The dialog is created lazily, on the first tick: the total is not known
    until the scanner has walked the tree, and a bar with a made-up total is
    worse than none.  A wait cursor covers that walk and is dropped the
    moment the dialog appears — a wait cursor over a clickable button lies
    about whether it can be clicked.
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
    Modal dialog.  Returns Accepted after a successful scan.  The dashboard
    re-populates its options once this returns.
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

        # Parent mode scans each session subfolder in turn and reads the
        # experimentalist from the layout
        # /<this folder>/<experimentalist>/<session>/…/*.ibw.
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
        Warn when `path` nests with a folder that already holds catalogued
        files, either way round: scanning a parent re-walks its children, and
        scanning a child already covered by a parent is redundant.

        Returns False — caller should abort — only if the user declines.
        """
        overlaps = _db.find_overlapping_directories(path, self._db_path)
        if not overlaps:
            return True
        lines = [
            (f"  - {existing} already covers this folder" if rel == "ancestor"
             else f"  - {existing} is inside this directory and would be re-scanned")
            for existing, rel in overlaps
        ]
        r = QMessageBox.warning(
            self, "Directory overlap",
            "This folder overlaps one already in the catalog:\n\n"
            + "\n".join(lines)
            + "\n\nScanning it may re-walk files already catalogued. Proceed anyway?",
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

        # Every descriptive field starts NULL until "Define metadata…" runs.
        known = bool(_db.list_files(db_path=self._db_path, directory=path))
        prog = _ScanProgress(self, f"Scanning {Path(path).name}…")
        try:
            n_found, n_updated, n_errors, cancelled = _scanner.scan_directory(
                path, self._db_path, progress_cb=prog,
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
                "What was scanned is kept — running Add data on this "
                "folder again picks up where this stopped, skipping what is "
                "already in.\n\nTo undo the import entirely, use "
                "'Remove these files…' on the dashboard.",
            )
            self.accept()
            return

        msg = (
            ("New folder.\n" if not known else "Folder already in the catalog; rescanned.\n")
            + f"Found {n_found} .ibw files\n"
            + f"  added/updated: {n_updated}\n"
            + f"  errors: {n_errors}"
        )
        QMessageBox.information(self, "Scan complete", msg)
        self.accept()

    def _accept_parent_tree(self, path: str) -> None:
        """
        Scan every session subfolder under `path`, reading the
        experimentalist from the folder names and writing it onto each
        scanned file.
        """
        prog = _ScanProgress(self, "Scanning folders…")
        try:
            summary = _scanner.scan_tree(
                path, self._db_path,
                # Explicit: this decides whether folder names touch
                # files.experimentalist at all.
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
            # Shown rather than silently written: a folder naming nobody in
            # the catalog is either a new person or a mis-aimed import root,
            # and only the user can tell which.
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
