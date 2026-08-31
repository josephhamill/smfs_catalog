# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/date_picker_dialog.py
#
# Date-range picker.  Primary control is a DENSE LIST of the dates that
# actually have data (sparse catalogues have only a few per month), with file
# counts.  Pick one day, or select a range (click first, Shift-click last) and
# BOTH 'from' and 'to' are filled in a single window — no reopen/renavigate.
#
# A read-only calendar sits alongside as a visual availability map.

from __future__ import annotations

from PyQt6.QtCore import Qt, QDate
from PyQt6.QtGui import QTextCharFormat, QColor, QFont
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QCalendarWidget, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QAbstractItemView,
)

from . import db as _db
from . import style
from .qt_utils import fit_on_screen


class DatePickerDialog(QDialog):
    """
    Modal date-range picker.  Result accessors after exec():
        self.date_from — QDate (earliest selected) or None
        self.date_to   — QDate (latest selected)   or None
    Both None if the user cancelled or selected nothing.
    """

    def __init__(
        self,
        db_path: str = _db.DEFAULT_DB_PATH,
        parent=None,
        scope: dict | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select date range")
        fit_on_screen(self, 640, 460)
        self.date_from: QDate | None = None
        self.date_to:   QDate | None = None

        # Dates available for the CURRENT scope (other dimensions), not the
        # whole DB.  Empty scope → every date.
        scope = scope or {}
        date_counts = _db.get_distinct_dates(
            db_path,
            users=scope.get("users") or None,
            analytes=scope.get("analytes") or None,
            solvents=scope.get("solvents") or None,
            techniques=scope.get("techniques") or None,
            substrates=scope.get("substrates") or None,
            sample_preps=scope.get("sample_preps") or None,
            afm_units=scope.get("afm_units") or None,
            curve_types=scope.get("curve_types") or None,
            search=scope.get("search"),
        )
        self._counts = {ds: n for ds, n in date_counts}

        layout = QVBoxLayout(self)
        hint = QLabel(
            "Pick one day, or a range: click the first date then Shift-click the "
            "last.  ‘From’ and ‘To’ fill together — no need to reopen."
        )
        hint.setStyleSheet(style.qss_text())
        hint.setWordWrap(True)
        layout.addWidget(hint)

        body = QHBoxLayout()

        # ── Dense list of available dates (primary control) ───────────────────
        self._list = QListWidget()
        self._list.setStyleSheet(style.LIST_QSS)
        self._list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        for ds, n in date_counts:
            item = QListWidgetItem(f"{ds}    ({n:,} files)")
            item.setData(Qt.ItemDataRole.UserRole, ds)
            self._list.addItem(item)
        self._list.itemSelectionChanged.connect(self._on_sel_changed)
        body.addWidget(self._list, 1)

        # ── Calendar (read-only availability map) ─────────────────────────────
        self._cal = QCalendarWidget()
        self._cal.setGridVisible(True)
        self._cal.setSelectionMode(QCalendarWidget.SelectionMode.NoSelection)
        avail_fmt = QTextCharFormat()
        # The SAME green the queue/database tables use for an event row.
        # Read from style — never hand-copied.
        avail_fmt.setBackground(QColor(style.ROW_TINT["event"]))
        avail_fmt.setFontWeight(QFont.Weight.Bold)
        for ds in self._counts:
            qd = self._qdate(ds)
            if qd is not None:
                self._cal.setDateTextFormat(qd, avail_fmt)
        body.addWidget(self._cal, 1)

        layout.addLayout(body, 1)

        self._summary = QLabel("No dates selected.")
        f = self._summary.font(); f.setBold(True); self._summary.setFont(f)
        layout.addWidget(self._summary)

        btn_row = QHBoxLayout()
        clear = QPushButton("Clear"); clear.clicked.connect(self._list.clearSelection)
        cancel = QPushButton("Cancel"); cancel.clicked.connect(self.reject)
        ok = QPushButton("OK"); ok.setDefault(True); ok.clicked.connect(self._on_accept)
        btn_row.addWidget(clear)
        btn_row.addStretch(1)
        btn_row.addWidget(cancel)
        btn_row.addWidget(ok)
        layout.addLayout(btn_row)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _qdate(ds: str) -> "QDate | None":
        try:
            y, m, d = (int(x) for x in ds.split("-"))
            return QDate(y, m, d)
        except (ValueError, AttributeError):
            return None

    def _selected_dates(self) -> list[str]:
        return sorted(
            it.data(Qt.ItemDataRole.UserRole) for it in self._list.selectedItems()
        )

    def _on_sel_changed(self) -> None:
        ds_list = self._selected_dates()
        if not ds_list:
            self._summary.setText("No dates selected.")
            return
        d_from, d_to = ds_list[0], ds_list[-1]
        n_files = sum(self._counts.get(d, 0) for d in ds_list)
        if d_from == d_to:
            self._summary.setText(f"{d_from}   ·   {n_files:,} files")
        else:
            self._summary.setText(
                f"{d_from}  →  {d_to}   ·   {len(ds_list)} dates   ·   "
                f"{n_files:,} files"
            )

    def _on_accept(self) -> None:
        ds_list = self._selected_dates()
        if ds_list:
            self.date_from = self._qdate(ds_list[0])
            self.date_to   = self._qdate(ds_list[-1])
        self.accept()
