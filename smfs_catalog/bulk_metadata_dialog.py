# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/bulk_metadata_dialog.py
#
# "Define metadata for these files..." — bulk-write sample metadata onto
# every file in a pre-resolved path list (#110, 2026-07-29).
#
# Descriptive sample metadata (experimentalist, analyte, solvent, instrument,
# cantilever, technique) is file-level, same grain as the per-file instrument
# columns the scanner already parses. It is populated AFTER import: filter to
# a cohort with the dashboard's existing scope filters, then open this dialog
# to write typed values onto every file the current scope expresses. There is
# no automatic extraction anymore — the .txt-scrape/Claude pipeline that used
# to do this at directory-registration time is retired (see CLAUDE.md's dated
# entry); this dialog is the only way these fields get set now.

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QComboBox,
    QCheckBox, QDialogButtonBox, QMessageBox, QCompleter,
)

from . import db as _db
from . import style
from .qt_utils import fit_on_screen

# (field key, form label)
_FIELDS: list[tuple[str, str]] = [
    ("experimentalist", "Experimentalist"),
    ("analyte",         "Analyte"),
    ("solvent",         "Solvent"),
    ("afm_unit",        "Instrument"),
    ("cantilever",      "Cantilever"),
    ("technique",       "Technique"),
]


class BulkMetadataDialog(QDialog):
    """
    Modal dialog. Operates on the EXACT `paths` list it was constructed with
    — it does not re-query scope live, so if the scope changes while this
    dialog is open, that has no effect on what gets written on accept.

    Only fields whose checkbox is ticked are written; ticking a checkbox
    with a blank value clears that field on every file in the cohort
    (explicit, confirmed before writing — never a silent side effect of an
    unrelated field).
    """

    def __init__(
        self, paths: list[str], db_path: str = _db.DEFAULT_DB_PATH, parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Define metadata for these files")
        fit_on_screen(self, 480, 320)
        self._db_path = db_path
        self._paths = list(paths)
        self._rows: list[tuple[str, QCheckBox, QComboBox]] = []
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        count_lbl = QLabel(f"{len(self._paths):,} file(s) in this cohort")
        count_lbl.setStyleSheet(style.QSS_EMPHASIS)
        outer.addWidget(count_lbl)

        hint = QLabel(
            "Tick a field to apply it to every file above. Untouched fields "
            "are left exactly as they are — this never blanks a field you "
            "didn't tick."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(style.qss_text())
        outer.addWidget(hint)

        form = QFormLayout()
        for key, label in _FIELDS:
            chk = QCheckBox()
            edit = self._make_value_box(key)
            edit.setEnabled(False)
            chk.toggled.connect(edit.setEnabled)
            row = QHBoxLayout()
            row.addWidget(chk)
            row.addWidget(edit, 1)
            form.addRow(label, self._wrap(row))
            self._rows.append((key, chk, edit))
        outer.addLayout(form)

        outer.addStretch(1)

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        box.accepted.connect(self._on_accept)
        box.rejected.connect(self.reject)
        outer.addWidget(box)

    def _make_value_box(self, key: str) -> QComboBox:
        """
        An EDITABLE combo seeded with the values already in the catalog.

        These six fields are open classes — a new experimentalist arrives about
        once a year, a new analyte more often — so the list can never be closed
        and typing a genuinely new value must stay possible.  What the dropdown
        removes is the *accidental* new value: "Anastasia" for "Anastasiia",
        a stray trailing space, a lowercase folder name.  Those do not read as
        errors anywhere — they read as a new person with no profile, no
        thresholds, and a cohort silently split in two.

        This is the only UI that writes these fields.  The other writer is
        scanner.experimentalist_from_path, which takes a folder name verbatim
        and cannot be constrained this way, so this dialog is also where such a
        name gets corrected — which is exactly why picking an existing value has
        to be one click.
        """
        box = QComboBox()
        box.setEditable(True)
        box.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        try:
            existing = _db.get_distinct_values(key, table="files", db_path=self._db_path)
        except Exception:
            existing = []          # never block editing because a lookup failed
        box.addItem("")            # blank first: ticking + leaving blank CLEARS
        box.addItems([v for v in existing if v])
        box.setCurrentIndex(0)
        comp = box.completer()
        if comp is not None:
            comp.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            comp.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        if existing:
            box.setToolTip(
                "Pick an existing value, or type a new one.\n"
                "Already in the catalog: " + ", ".join(existing)
            )
        return box

    def _wrap(self, layout) -> "QWidget":
        from PyQt6.QtWidgets import QWidget
        w = QWidget()
        w.setLayout(layout)
        return w

    def _on_accept(self) -> None:
        fields = {
            key: (edit.currentText().strip() or None)
            for key, chk, edit in self._rows
            if chk.isChecked()
        }
        if not fields:
            QMessageBox.information(
                self, "Define metadata", "Nothing checked — nothing to apply.")
            return

        lines = [f"  {label} → {fields[key]!r}" for key, label in _FIELDS if key in fields]
        r = QMessageBox.question(
            self, "Define metadata",
            f"About to write to {len(self._paths):,} file(s):\n\n"
            + "\n".join(lines)
            + "\n\nThis overwrites the field(s) above on every file in the "
              "cohort (clearing it if left blank). Proceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if r != QMessageBox.StandardButton.Yes:
            return

        n = _db.set_file_descriptors_bulk(self._paths, fields, self._db_path)
        QMessageBox.information(self, "Define metadata", f"Updated {n:,} file(s).")
        self.accept()
