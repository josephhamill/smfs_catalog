# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/repoint_dialog.py
#
# "Repoint moved data..." — tell the catalog where the curves went.
#
# A curve's path is true for this machine at this moment.  Its analysis is
# not tied to either, because that keys on files.id — so a move is repaired
# by rewriting the part of the path that changed.  See db.py's "Repointing
# the catalog at data that has moved".
#
# The table is the diagnosis (which folder holds curves the app cannot find);
# the two fields under it are the fix.  They are plain editable text because
# the old location generally does not exist on disk and so cannot be browsed
# for.

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QDialog, QDialogButtonBox, QFileDialog,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from . import db as _db
from . import style
from .qt_utils import fit_on_screen

_COLUMNS = ("Folder", "Curves", "Not found", "Status")


class RepointDataDialog(QDialog):
    """
    Modal.  Rewrites stored paths only — it never reads, moves, copies or
    deletes anything on disk, and it will happily repoint at a drive that is
    not mounted yet (it says how many files it can see and lets the user
    decide).

    `changed` records whether anything was written, so the dashboard knows
    whether it needs to refresh.
    """

    def __init__(self, db_path: str = _db.DEFAULT_DB_PATH, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Repoint moved data")
        fit_on_screen(self, 820, 560)
        self._db_path = db_path
        self.changed = False
        self._info: dict = {}
        self._build_ui()
        self._reload()

    # -- UI ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        hint = QLabel(
            "When curves are moved to another drive or folder, the catalog "
            "still holds their old paths and cannot open them. Repointing "
            "rewrites those paths — every verdict, ROI, fit and queue entry "
            "is kept, because they belong to the catalog entry and not to "
            "where the file happens to sit on disk.\n\n"
            "Pick the folder below that has moved, then say where it is now."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(style.qss_text())
        outer.addWidget(hint)

        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(list(_COLUMNS))
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, len(_COLUMNS)):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self._table.itemSelectionChanged.connect(self._on_row_selected)
        outer.addWidget(self._table, 1)

        self._old = QLineEdit()
        self._old.setPlaceholderText("the path the catalog still holds")
        self._old.editingFinished.connect(self._preview)
        outer.addLayout(self._field_row("Moved from", self._old))

        self._new = QLineEdit()
        self._new.setPlaceholderText("where those curves are now")
        self._new.editingFinished.connect(self._preview)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._on_browse)
        outer.addLayout(self._field_row("Moved to", self._new, browse))

        self._preview_lbl = QLabel("")
        self._preview_lbl.setWordWrap(True)
        self._preview_lbl.setStyleSheet(style.qss_text())
        outer.addWidget(self._preview_lbl)

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Close
        )
        self._ok = box.button(QDialogButtonBox.StandardButton.Ok)
        self._ok.setText("Repoint…")
        self._ok.setEnabled(False)
        box.accepted.connect(self._on_accept)
        box.rejected.connect(self.reject)
        outer.addWidget(box)

    def _field_row(self, label: str, edit: QLineEdit, *extra) -> QHBoxLayout:
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setMinimumWidth(90)
        row.addWidget(lbl)
        row.addWidget(edit, 1)
        for w in extra:
            row.addWidget(w)
        return row

    # -- Table ---------------------------------------------------------------

    def _reload(self) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._dirs = sorted(_db.missing_by_directory(self._db_path),
                                key=lambda d: (-d["n_missing"], d["path"]))
        finally:
            QApplication.restoreOverrideCursor()

        self._table.setRowCount(len(self._dirs))
        for r, d in enumerate(self._dirs):
            if not d["exists"]:
                status, broken = "folder not found", True
            elif d["n_missing"]:
                status, broken = "some curves not found", True
            else:
                status, broken = "found", False
            cells = (d["path"], f"{d['n_files']:,}", f"{d['n_missing']:,}", status)
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if broken and c in (2, 3):
                    item.setForeground(QBrush(QColor(style.STATUS_CRITICAL)))
                self._table.setItem(r, c, item)

        n_broken = sum(d["n_missing"] for d in self._dirs)
        if n_broken:
            self._preview_lbl.setText(
                f"{n_broken:,} catalogued curve(s) cannot be found on disk.")
        elif self._dirs:
            self._preview_lbl.setText(
                "Every catalogued curve is where the catalog expects it.")
        else:
            self._preview_lbl.setText("The catalog is empty.")

    def _on_row_selected(self) -> None:
        rows = {i.row() for i in self._table.selectedIndexes()}
        if not rows:
            return
        self._old.setText(self._dirs[min(rows)]["path"])
        self._preview()

    def _on_browse(self) -> None:
        start = self._new.text().strip() or self._old.text().strip()
        picked = QFileDialog.getExistingDirectory(
            self, "Where are these curves now?", start)
        if picked:
            self._new.setText(picked)
            self._preview()

    # -- Preview -------------------------------------------------------------

    def _preview(self) -> None:
        """Recompute the plan for whatever is typed now.  Reads only — nothing is written until Repoint is pressed."""
        old, new = self._old.text().strip(), self._new.text().strip()
        if not old or not new:
            self._ok.setEnabled(False)
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._info = i = _db.describe_relocation(old, new, self._db_path)
        finally:
            QApplication.restoreOverrideCursor()

        if not i["n_files"] and not i["n_blocked"]:
            self._ok.setEnabled(False)
            self._preview_lbl.setText(
                f"No catalogued curve lives under {i['old_root']} — check the "
                f"'Moved from' path against the table above.")
            return

        lines = [f"{i['n_files']:,} curve(s) would be repointed to "
                 f"{i['new_root']}."]
        if i["n_found"] == i["n_files"]:
            lines.append(f"All {i['n_found']:,} are present there.")
        else:
            lines.append(f"{i['n_found']:,} found there, {i['n_missing']:,} NOT "
                         f"found — repointing is still allowed (the drive may "
                         f"not be mounted yet), but check the path first.")
        if i["n_blocked"]:
            lines.append(f"{i['n_blocked']:,} would be skipped: the catalog "
                         f"already holds a separate entry at that new path.")
        self._preview_lbl.setText("  ".join(lines))
        self._ok.setEnabled(bool(i["n_files"]))

    # -- Action --------------------------------------------------------------

    def _on_accept(self) -> None:
        i = self._info
        if not i.get("n_files"):
            return
        r = QMessageBox.question(
            self, "Repoint moved data",
            f"Repoint {i['n_files']:,} catalogued curve(s)?\n\n"
            f"  from  {i['old_root']}\n"
            f"  to    {i['new_root']}\n\n"
            f"{i['n_found']:,} of them are present at the new location"
            + (f"; {i['n_missing']:,} are not" if i["n_missing"] else "")
            + ".\n\nAll analysis is kept. No file on disk is read, moved or "
              "deleted — only the paths the catalog stores change.\n\n"
              "Proceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if r != QMessageBox.StandardButton.Yes:
            return

        out = _db.relocate_files(i["old_root"], i["new_root"], self._db_path)
        self.changed = True
        msg = f"Repointed {out['n_files']:,} curve(s)."
        if out["n_blocked"]:
            msg += (f"\n\n{out['n_blocked']:,} were skipped: the catalog "
                    f"already holds a separate entry at their new path.")
        QMessageBox.information(self, "Repoint moved data", msg)
        self._reload()
        self._new.clear()
        self._ok.setEnabled(False)
