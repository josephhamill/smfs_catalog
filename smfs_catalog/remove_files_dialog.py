# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/remove_files_dialog.py
#
# "Remove these files..." — the undo for an import or an analysis run, over
# the same scope-selected cohort as "Define metadata for these files...".
# The dashboard's other Remove button clears the analysis QUEUE, not the
# catalog.
#
# Two levels, because those are two different mistakes — see db.py's
# "Removing things from the catalog" section for what each one touches.
# NEITHER touches a file on disk.

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QRadioButton, QButtonGroup,
    QDialogButtonBox, QMessageBox,
)

from . import db as _db
from . import style
from .qt_utils import fit_on_screen

_ERASE = "erase"
_REMOVE = "remove"


class RemoveFilesDialog(QDialog):
    """
    Modal dialog. Operates on the EXACT `paths` list it was constructed with —
    it does not re-query scope live, so changing the scope while this is open
    cannot change what gets removed (same contract as BulkMetadataDialog).

    Defaults to the LESS destructive level (erase analysis), deliberately:
    the default on a destructive dialog should be the one you can most easily
    recover from.
    """

    def __init__(
        self, paths: list[str], db_path: str = _db.DEFAULT_DB_PATH, parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Remove these files")
        fit_on_screen(self, 560, 380)
        self._db_path = db_path
        self._paths = list(paths)
        self._info = _db.describe_removal_scope(self._paths, db_path)
        self.result_summary: dict | None = None
        self._build_ui()

    # -- UI ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        i = self._info

        count = QLabel(f"{i['n_files']:,} file(s) in this cohort")
        count.setStyleSheet(style.QSS_EMPHASIS)
        outer.addWidget(count)

        outer.addWidget(self._muted(
            f"  {i['n_classified']:,} carry an analysis verdict "
            f"({i['n_events']:,} classified as events)\n"
            f"  {i['n_with_fits']:,} hold stored ROI/WLC fits\n"
            f"  {i['n_queued']:,} are currently in the analysis queue"
        ))

        self._group = QButtonGroup(self)

        self._erase_rb = QRadioButton(
            "Erase the analysis, keep the files in the catalog")
        self._erase_rb.setChecked(True)
        self._group.addButton(self._erase_rb)
        outer.addWidget(self._erase_rb)
        outer.addWidget(self._muted(
            "Throws away every computed result — fits, ROIs, event verdicts — "
            "and leaves each file reading as never analysed, ready to be "
            "re-analysed under different settings. Files stay in the catalog, "
            "keep their sample metadata, and stay in the queue. Your manual "
            "Primary/Secondary segment picks are kept.", indent=True))

        self._remove_rb = QRadioButton(
            "Remove the files from the catalog entirely")
        self._group.addButton(self._remove_rb)
        outer.addWidget(self._remove_rb)
        outer.addWidget(self._muted(
            "The above, plus the catalog entries themselves — the app forgets "
            "these curves exist.", indent=True))

        outer.addStretch(1)

        outer.addWidget(self._muted(
            "Neither option deletes, moves or modifies any file on disk. "
            "Removed curves come back by running Add Data over the same "
            "folder again — but their analysis does not, and re-running it "
            "costs the same hours it did the first time."))

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        box.button(QDialogButtonBox.StandardButton.Ok).setText("Remove…")
        box.accepted.connect(self._on_accept)
        box.rejected.connect(self.reject)
        outer.addWidget(box)

    def _muted(self, text: str, indent: bool = False) -> QLabel:
        lbl = QLabel(("      " + text) if indent else text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(style.qss_text())
        return lbl

    # -- Action --------------------------------------------------------------

    def _mode(self) -> str:
        return _ERASE if self._erase_rb.isChecked() else _REMOVE

    def _on_accept(self) -> None:
        i = self._info
        if not i["n_files"]:
            QMessageBox.information(
                self, "Remove these files", "No files matched — nothing to do.")
            return

        mode = self._mode()
        if mode == _ERASE:
            what = (
                f"Erase all analysis for {i['n_files']:,} file(s)?\n\n"
                f"This discards {i['n_with_fits']:,} file(s) worth of stored "
                f"fits and {i['n_classified']:,} verdict(s). The files stay in "
                f"the catalog."
            )
        else:
            what = (
                f"Remove {i['n_files']:,} file(s) from the catalog?\n\n"
                f"This also discards {i['n_with_fits']:,} file(s) worth of "
                f"stored fits and {i['n_classified']:,} verdict(s)."
            )

        r = QMessageBox.question(
            self, "Remove these files",
            what + "\n\nNo file on disk is touched. This cannot be undone from "
                   "inside the app — re-importing restores the entries, not "
                   "the analysis.\n\nProceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if r != QMessageBox.StandardButton.Yes:
            return

        if mode == _ERASE:
            out = _db.erase_analysis_for_files(self._paths, self._db_path)
            msg = (
                f"Erased analysis for {out['n_files']:,} file(s).\n\n"
                f"  fits/ROIs cleared: {out['event_map']:,}\n"
                f"  stored results cleared: {out['analysis_results']:,}\n"
                f"  histograms cleared: {out['event_histograms']:,}"
            )
        else:
            out = _db.remove_files_from_catalog(self._paths, self._db_path)
            msg = (
                f"Removed {out['n_files']:,} file(s) from the catalog.\n\n"
                f"  fits/ROIs cleared: {out['event_map']:,}\n"
                f"  queue entries cleared: {out['analysis_queue']:,}\n\n"
                f"The .ibw files themselves are untouched on disk."
            )

        self.result_summary = out
        QMessageBox.information(self, "Remove these files", msg)
        self.accept()
