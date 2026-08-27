# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.


from __future__ import annotations

import sqlite3
import time
from collections import deque
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import (
    Qt, QTimer, pyqtSignal, QDate, QAbstractTableModel, QModelIndex,
)
from PyQt6.QtGui import QColor, QBrush
from PyQt6.QtWidgets import (
    QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem,
    QTableView, QHeaderView, QAbstractItemView, QLineEdit, QCheckBox,
    QDateEdit, QSplitter, QGroupBox, QMessageBox, QComboBox, QFrame,
    QFileDialog, QApplication, QSizePolicy,
)

from . import db as _db
from . import criteria_gate as _gate
from . import export_utils as _export
from . import quantities as _quant
from . import signal_processing as _sp
from . import variables as _vars
from . import style
from .bandwidth_warning import FILTER_BANDWIDTH_CONSEQUENCE
from .scope import new_scope, scope_to_query
from .widgets import CollapsibleSection, FlowLayout, LabeledControl
from .analysis_worker import AnalysisWorker
from .navigator_bar import NavigatorBar
from .roi_pipeline import (
    read_segment_select, write_segment_select, segment_summary_bulk,
)
from .qt_utils import fit_on_screen


_SELECTION_QSS = style.TABLE_QSS

# How short a splitter section may be dragged.  Enough to keep its header and
# a row or two of content visible; below this a section is better collapsed.
_MIN_SECTION_H = 90


def _vsep() -> QFrame:
    """Thin vertical rule — groups a toolbar row's button clusters without a caption above each one."""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


def _row_get(row, key):
    """Read one field from a sqlite3.Row or a dict; None if absent."""
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


def _bg_for(event: Optional[str], status: Optional[str]) -> QColor:
    return QColor(style.row_tint(event, status))


_DB_COLUMNS: list[tuple[str, str]] = [
    ("filename",              "Filename"),
    ("dir_path",              "Directory"),
    ("event",        "Event"),
    ("curve_type",            "Type"),
    ("unusable_reason",       "Unusable"),
    ("unusable_detail",       "Unusable detail"),
    ("duplicate_of",          "Duplicate of"),
    ("measured_date",         "Date"),
    ("measured_at",           "Acquired"),
    ("experimentalist",       "Experimentalist"),
    ("analyte",               "Analyte"),
    ("solvent",               "Solvent"),
    ("afm_unit",              "Instrument"),
    ("microscope_model",      "Model"),
    ("cantilever",            "Cantilever"),
    ("technique",             "Technique"),
    ("spring_constant_pn_nm", "k (pN/nm)"),
    ("velocity_nm_s",         "Vel (nm/s)"),
    ("force_dist_nm",         "Force dist (nm)"),
    ("trigger_point_nn",      "Trigger"),
    ("inv_ols_nm_v",          "InvOLS (nm/V)"),
    ("sample_rate_hz",        "Sample (Hz)"),
    ("force_filter_bw_hz",    "Acq filter (Hz)"),
    ("xpos_um",               "X (µm)"),
    ("ypos_um",               "Y (µm)"),
    ("size_bytes",            "Size"),
    ("modified_at",           "Modified"),
    ("first_seen",            "First seen"),
    ("last_seen",             "Last seen"),
    ("parse_ok",              "Parsed"),
    ("parse_error",           "Parse err"),
    ("dwell_setting",         "Dwell"),
    ("indent_mode",           "Indent"),
]

# The queue's derived columns, in display order.  KEYS ONLY: the label comes
# from variables.label, which is the one register the scatter and the
# variable window already ask.  This list used to carry its own labels and six
# of them had drifted from it — seg_dX_ext_nm was "Ext ΔX (nm)" here and
# "Rupture separation (nm)" in the scatter, one number under two names.
_QUEUE_DERIVED_KEYS = (
    "snapoff_piezo_nm",
    "contact_dx_nm",
    "offset_retr",
    "flatness_slope",
    "baseline_rms",
    "invols_slope",
    "invols_rms",
    "onset_dx_nm",
    "rupture_dx_nm",
    "seg_n_segments",
    # The reported rupture's force and its two extensions, adjacent because
    # they are one point: (x from snap-off, x from onset, y).
    "seg_force_pN",
    "seg_x_rupture_nm",
    "seg_x_junction_nm",
    "seg_l_p_nm",
    "seg_l_p_err",
    "seg_l_c_nm",
    "seg_l_c_err",
    "seg_tau",
    "seg_z_max",
    "seg_x_max_nm",
    "seg_edge_pinned",
    "seg_dF_pN",
    "seg_dX_iso_nm",
    "seg_dX_ext_nm",
)

_QUEUE_DERIVED = [(k, _vars.label(k)) for k in _QUEUE_DERIVED_KEYS]


_ETA_COST_SAMPLES = 200
_ETA_MAX_SAMPLE_S = 60.0

_FRESHNESS_LABEL = {
    "fresh": "up to date",
    "stale": "stale params",
    "new":   "not analysed",
}
_QUEUE_COLUMNS_FIXED = [
    ("filename", "Filename"),
    ("status",   "Status"),
    ("event", "Event"),
    ("hit",      "Hit"),
]

_FIXED_COL_TOOLTIPS: dict[str, str] = {
    "filename":
        "The file's name only. Full paths are in the files-in-scope table "
        "above and in every export — the same name recurs across directories, "
        "so never identify a curve by this alone.",
    "status":
        "What analysing this row would COST under the parameter set currently "
        "in force — not whether it has been looked at.\n\n"
        "  up to date   — a stored result matches today's parameters AND "
        "today's code; it will be served from cache in milliseconds.\n"
        "  stale params — a result is stored, but under different parameters "
        "or a different scientific method, so it needs a full recompute.\n"
        "  not analysed — nothing is stored for this file at all.\n\n"
        "Any of the three may carry '· visited', which means only that the "
        "playhead has passed this row during THIS session. That distinction "
        "distinguishes session navigation from analysis state.\n\n"
        "  error        — this row raised an unexpected processing or database "
        "error; hover for the recorded detail.\n\n"
        "This is worked out from the live settings when the queue is loaded "
        "or explicitly inspected. Merely switching windows does not change "
        "the queue's analysis state.",
    "event":
        "The stage-1 classification: did the pipeline find a rupture event on "
        "this curve?\n\n"
        "  event        — baseline fit, snap-off and both ROI landmarks were "
        "found, in a consistent order. Force is NOT part of this test.\n"
        "  non_event    — analysed properly, no event found. A real verdict.\n"
        "  unavailable  — a legacy temporary marker for a file that could not "
        "be read. New read failures pause analysis and remain unclassified; "
        "reconnect the data and press Play to retry.\n"
        "  unusable     — the file read fine but cannot be analysed (no "
        "numbers in a required channel, a channel that never varies, not a "
        "force-extension curve, no spring constant). This is durable, so the "
        "file is dequeued and never retried. The Unusable column says which.\n"
        "  blank        — not analysed yet.\n\n"
        "The last two are deliberately different words: one means come back "
        "later, the other means never. Treating them alike is what once "
        "printed a drive warning for a healthy drive.",
    "hit":
        "The stage-2 verdict: does this curve pass the criteria gate? Only "
        "meaningful for rows classified as 'event'.\n\n"
        "Worked out live from the checked criteria and bounds belonging to "
        "THIS file's own experimentalist — a queue routinely mixes several "
        "people's files, and each is judged against their own. Nothing is "
        "stored, so it always reflects the criteria as they are right now.\n\n"
        "WITH NO CRITERIA CHECKED, EVERY EVENT IS A HIT. That is not a bug "
        "and it is the basis of the hand-built cohort workflow: queue the "
        "curves you want, check nothing, and the population is exactly what "
        "you queued.\n\n"
        "A checked criterion REQUIRES a value: an event with no value for it "
        "becomes a non-hit rather than being ignored.",
}
_QUEUE_COLUMNS = _QUEUE_COLUMNS_FIXED + _QUEUE_DERIVED

_QUEUE_BASE_KEYS = list(_QUEUE_DERIVED_KEYS)

_QUEUE_HIDE = _vars.EXCLUDED_VARIABLE_KEYS

_QUEUE_DERIVED_ORDER = _QUEUE_BASE_KEYS


def _prettify_key(key: str) -> str:
    """The register's name for a column, or a readable form of the raw key.

    An analysis_type present in the queue but not in the register still gets a
    column (see _compute_queue_derived_cols); variables.label hands such a key
    straight back, and the underscore swap is what it used to be shown as.
    """
    lbl = _vars.label(key)
    return key.replace("_", " ") if lbl == key else lbl


class FilesTableModel(QAbstractTableModel):
    """Virtual model for the DB table."""

    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self._columns = columns
        self._col_index = {key: i for i, (key, _) in enumerate(columns)}
        self._rows: list = []
        self._id_to_row: dict[int, int] = {}

    def set_rows(self, rows) -> None:
        self.beginResetModel()
        self._rows = list(rows)
        self._id_to_row = {
            int(_row_get(r, "id")): i
            for i, r in enumerate(self._rows) if _row_get(r, "id") is not None
        }
        self.endResetModel()

    def row_id(self, row: int):
        return _row_get(self._rows[row], "id") if 0 <= row < len(self._rows) else None

    def row_path(self, row: int):
        return _row_get(self._rows[row], "path") if 0 <= row < len(self._rows) else None

    def index_for_id(self, file_id: int) -> int | None:
        return self._id_to_row.get(int(file_id))

    def value_for_id(self, file_id: int, key: str):
        """One field of one row by file_id, or None if the id isn't loaded."""
        idx = self.index_for_id(file_id)
        return _row_get(self._rows[idx], key) if idx is not None else None

    def update_field(self, file_id: int, key: str, value) -> None:
        """Patch one field of one row."""
        idx = self.index_for_id(file_id)
        if idx is None:
            return None
        row = self._rows[idx]
        prev = _row_get(row, key)
        if not isinstance(row, dict):
            row = dict(row)
            self._rows[idx] = row
        row[key] = value
        col = self._col_index.get(key)
        if col is not None:
            qidx_l = self.index(idx, col)
            qidx_r = self.index(idx, col)
            self.dataChanged.emit(qidx_l, qidx_r, [
                Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.BackgroundRole,
            ])
            row_l = self.index(idx, 0)
            row_r = self.index(idx, self.columnCount() - 1)
            self.dataChanged.emit(row_l, row_r, [Qt.ItemDataRole.BackgroundRole])
        return prev

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._columns)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self._columns[section][1]
        return str(section + 1)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        key = self._columns[index.column()][0]
        if role == Qt.ItemDataRole.DisplayRole:
            return _fmt_cell(_row_get(row, key), key)
        if role == Qt.ItemDataRole.BackgroundRole:
            return QBrush(_bg_for(_row_get(row, "event"), None))
        return None

    def sort(self, column: int, order=Qt.SortOrder.AscendingOrder) -> None:
        """Sort in-place by the given column."""
        if not self._rows or column < 0 or column >= len(self._columns):
            return
        key = self._columns[column][0]

        def _k(r):
            v = _row_get(r, key)
            if v is None or v == "":
                return (1, "")
            if isinstance(v, (int, float)):
                return (0, v)
            return (0, str(v))

        self.beginResetModel()
        self._rows.sort(key=_k, reverse=(order == Qt.SortOrder.DescendingOrder))
        self.endResetModel()


class DashboardWindow(QMainWindow):
    """Single-window dashboard."""

    scope_changed = pyqtSignal(dict)


    def __init__(self, db_path: str = _db.DEFAULT_DB_PATH, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("SMFS Catalog — Dashboard")
        fit_on_screen(self, 1400, 900)
        self._db_path = db_path
        self._scope = new_scope()
        self._children: list[QWidget] = []

        _db.clear_analysis_queue(db_path)

        self._count_timer = QTimer(self)
        self._count_timer.setSingleShot(True)
        self._count_timer.setInterval(250)
        self._count_timer.timeout.connect(self._refresh_db_and_counts)

        self._queue_class_counts = {k: 0 for k in _db.EVENT_VERDICTS} | {"unclassified": 0}

        self._queue_total = 0
        self._done_ids: set[int] = set()
        self._rate_times: deque = deque()

        self._queue_row_ids: list[int] = []

        self._freshness: dict[int, str] = {}

        self._cost_samples: deque = deque(maxlen=_ETA_COST_SAMPLES)
        self._last_done_t: float | None = None

        self._loaded_count = 0
        self._cached_count = 0

        self._done_buffer: list[tuple[int, str, bool]] = []
        self._started_buffer: list[int] = []
        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(150)
        self._flush_timer.timeout.connect(self._flush_worker_events)
        self._flush_timer.start()

        self._queue_id_to_row: dict[int, int] = {}

        self._worker = AnalysisWorker(db_path)
        self._worker.set_paused(True)

        self._build_ui()
        self._refresh_db_and_counts()

        self._worker.file_started.connect(self._on_file_started)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.file_error.connect(self._on_file_error)
        self._worker.data_unavailable.connect(self._on_data_unavailable)
        self._worker.fatal_error.connect(self._on_worker_fatal_error)
        self._worker.paused_changed.connect(self._on_worker_paused_changed)
        self._worker.playhead_changed.connect(self._on_worker_playhead_changed)
        self._worker.direction_changed.connect(self._on_worker_direction_changed)
        self._worker.queue_empty.connect(self._on_worker_queue_empty)
        # Queue changes originating inside the analysis pipeline (notably an
        # unusable curve being dequeued) must remove the row from this view too.
        self._worker.queue_changed.connect(self._refresh_queue_table)
        self._worker.start()
        self._update_location_label()

    def _refresh_freshness(self) -> None:
        """Recompute freshness and repaint the Status column + gate summary."""
        if not self._queue_row_ids:
            return
        self._freshness = self._compute_freshness()
        for row, fid in enumerate(self._queue_row_ids):
            if row < self._queue_table.rowCount():
                self._set_status_cell(row, fid, self._raw_status(row))
        self._update_gate_label()
        self._update_location_label()

    def closeEvent(self, event) -> None:
        """Closing the dashboard quits the app."""
        if not self._confirm_queue_saved():
            event.ignore()
            return
        try:
            self._worker.stop()
        except Exception:
            pass
        for win in list(self._children):
            try:
                win.close()
            except Exception:
                pass
        self._children = []
        super().closeEvent(event)
        QApplication.quit()

    def _confirm_queue_saved(self) -> bool:
        """Offer to save a non-empty queue on the way out."""
        n = len(_db.queue_paths(self._db_path))
        if not n:
            return True
        box = QMessageBox(self)
        box.setWindowTitle("Save the queue?")
        box.setText(f"The analysis queue has {n:,} file(s). Save it before quitting?")
        box.setInformativeText(
            "The queue is cleared at the next launch. Saving writes a file that "
            "Load Queue reads back, so you can pick this cohort up again.")
        save   = box.addButton("Save",       QMessageBox.ButtonRole.AcceptRole)
        dont   = box.addButton("Don't save", QMessageBox.ButtonRole.DestructiveRole)
        cancel = box.addButton("Cancel",     QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(save)
        box.exec()
        clicked = box.clickedButton()
        if clicked is dont:
            return True
        if clicked is not save:
            return False
        try:
            self._on_save_queue()
        except Exception as exc:
            QMessageBox.warning(
                self, "Save the queue?",
                f"The queue was NOT saved:\n\n{exc}\n\n"
                "Nothing has been closed — try again, or quit and choose "
                "\"Don't save\".")
            return False
        return True


    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        # FlowLayout: the title plus six long-labelled buttons sum to about
        # 1500 px in one row, which was the dashboard's real minimum width —
        # the queue bar below it was never the binding constraint.
        header = FlowLayout(margin=0, h_spacing=6, v_spacing=4)
        title = QLabel("SMFS Catalog")
        title.setFont(style.font(title.font(), size_pt=style.FONT_TITLE_PT, bold=True))
        header.addWidget(title)

        report_btn = QPushButton("📄 Classification report…")
        report_btn.setToolTip(
            "Write a CSV of every file's event/hit status to the export "
            "folder — the whole database, or the current scope filters' "
            "subset if any are set."
        )
        report_btn.clicked.connect(self._on_export_classification_report)
        header.addWidget(report_btn)

        self._export_dir_lbl = QLabel("")
        self._export_dir_lbl.setStyleSheet(style.qss_text())
        header.addWidget(self._export_dir_lbl)
        self._update_export_dir_label()

        folder_btn = QPushButton("📁 Export folder…")
        folder_btn.setToolTip(
            "Where every export button in the app writes files. Defaults "
            "to the database's own folder. This is ONE setting for the whole "
            "database, not per experimentalist — changing it changes where "
            "everybody's exports go."
        )
        folder_btn.clicked.connect(self._on_set_export_folder)
        header.addWidget(folder_btn)

        define_meta_btn = QPushButton("🏷 Define metadata for these files…")
        define_meta_btn.setToolTip(
            "Bulk-write sample metadata (experimentalist, analyte, solvent, "
            "instrument, cantilever, technique) onto every file currently in "
            "scope — the same cohort 'Database — files in scope' counts."
        )
        define_meta_btn.clicked.connect(self._on_define_metadata)
        header.addWidget(define_meta_btn)

        remove_btn = QPushButton("🗑 Remove these files…")
        remove_btn.setToolTip(
            "Undo an import or an analysis run over every file currently in "
            "scope — the same cohort 'Database — files in scope' counts. "
            "Either erase the analysis and keep the files, or remove the "
            "catalog entries entirely. Never touches a file on disk."
        )
        remove_btn.clicked.connect(self._on_remove_files)
        header.addWidget(remove_btn)

        recheck_btn = QPushButton("🔍 Re-check catalogued files…")
        recheck_btn.setToolTip(
            "Re-read catalogued files to fill in what was not known when they "
            "were imported: their content fingerprint, which finds duplicate "
            "copies of the same curve, and any newer qualification check.\n\n"
            "An ordinary scan skips files whose timestamp has not changed, so "
            "this is the way to bring old files up to date. Resumable."
        )
        recheck_btn.clicked.connect(self._on_recheck_catalog)
        header.addWidget(recheck_btn)

        repoint_btn = QPushButton("📍 Repoint moved data…")
        repoint_btn.setToolTip(
            "Tell the catalog where curves went after they were moved to "
            "another drive or folder.\n\n"
            "Only the stored paths change — every verdict, ROI, fit and queue "
            "entry is kept, so a move costs seconds instead of a new database "
            "and a full re-analysis. No file on disk is touched."
        )
        repoint_btn.clicked.connect(self._on_repoint_data)
        header.addWidget(repoint_btn)


        self._add_btn = QPushButton("➕ Add data…")
        self._add_btn.clicked.connect(self._on_add_data)
        header.addWidget(self._add_btn)

        _header_bar = QWidget()
        _header_bar.setLayout(header)
        _header_bar.setSizePolicy(QSizePolicy.Policy.Preferred,
                                  QSizePolicy.Policy.Minimum)
        outer.addWidget(_header_bar)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter = splitter
        outer.addWidget(splitter, 1)

        self._sections = [
            self._build_scope_section(),
            self._build_db_section(),
            self._build_queue_section(),
        ]
        self._section_weights = [0, 3, 3]
        for i, sec in enumerate(self._sections):
            # An explicit small minimum, because a vertical QSplitter's own
            # minimum is the SUM of its children's.  Left to their content
            # hints the three sections summed to 935 px and the window's
            # minimum height came to 1200 -- taller than a 1080 screen, so the
            # title bar sat above the desktop and the window could not be
            # dragged back into reach.  Each section's content already scrolls,
            # and the splitter handles are how space is allocated between them.
            sec.setMinimumHeight(_MIN_SECTION_H)
            splitter.addWidget(sec)
            splitter.setStretchFactor(i, self._section_weights[i])
            sec.toggled.connect(self._on_section_toggled)
        # The heights the expanded sections are shared out in.  Read as
        # proportions, not pixels: absolute heights would remember a section as
        # tall as it grew while its siblings were collapsed, and it would come
        # back demanding that, squeezing them to their minimum.
        self._section_ref_h = [max(sec.sizeHint().height(), _MIN_SECTION_H)
                               for sec in self._sections]
        splitter.splitterMoved.connect(lambda *_a: self._snapshot_sections())

    def _snapshot_sections(self) -> None:
        """Record the proportions the visible sections are currently at,
        leaving the collapsed ones' share of the total untouched."""
        sizes = self._splitter.sizes()
        shown = [i for i, sec in enumerate(self._sections)
                 if sizes[i] > sec.header_height()]
        now = sum(sizes[i] for i in shown)
        if not shown or now <= 0:
            return
        was = sum(self._section_ref_h[i] for i in shown)
        for i in shown:
            self._section_ref_h[i] = sizes[i] * was / now

    def _on_section_toggled(self, _expanded: bool) -> None:
        # Qt has not re-laid out yet, so this still sees the heights the
        # sections had before the click.
        self._snapshot_sections()
        # Deferred, because until Qt applies the new height bounds the
        # splitter can still be as short as the collapsed ones held it, and
        # sizes shared out of that height never grow back.
        QTimer.singleShot(0, self._relayout_splitter)

    def _relayout_splitter(self) -> None:
        """Redistribute splitter panes when a section collapses/expands."""
        # Sum of the panes, not the splitter's height: the handles are the
        # difference, and sizes adding up to more than this push the last
        # section off the bottom instead of shrinking the others.
        total = sum(self._splitter.sizes())
        if total <= 0:
            return
        sizes = [0] * len(self._sections)
        expanded = []
        used = 0
        for i, sec in enumerate(self._sections):
            if sec.is_expanded():
                expanded.append(i)
            else:
                sizes[i] = sec.header_height()
                used += sizes[i]
        if expanded:
            remaining = max(0, total - used)
            wsum = sum(self._section_ref_h[i] for i in expanded) or 1
            for i in expanded:
                sizes[i] = int(remaining * self._section_ref_h[i] / wsum)
            sizes[expanded[-1]] += remaining - sum(sizes[i] for i in expanded)
        self._splitter.setSizes(sizes)

    def _build_scope_section(self) -> QWidget:
        sec = CollapsibleSection("Scope filters", expanded=True)
        body = QWidget(); body_l = QVBoxLayout(body); body_l.setContentsMargins(0,0,0,0)

        hint = QLabel(
            "Filters narrow each other — each list shows only values that still "
            "have data, with counts.  Empty = no constraint on that dimension."
        )
        hint.setStyleSheet(style.qss_text())
        hint.setWordWrap(True)
        body_l.addWidget(hint)

        grid = QHBoxLayout()
        self._lists: dict[str, QListWidget] = {}
        for key, label in [
            ("users",       "Experimentalist"),
            ("analytes",    "Analyte"),
            ("solvents",    "Solvent"),
            ("afm_units",   "Instrument"),
            ("curve_types", "Experiment type"),
        ]:
            box = QGroupBox(label); lay = QVBoxLayout(box)
            lst = QListWidget()
            lst.setSelectionMode(QListWidget.SelectionMode.NoSelection)
            lst.setMaximumHeight(120)
            lst.itemChanged.connect(self._on_scope_edit)
            lay.addWidget(lst)
            self._lists[key] = lst
            grid.addWidget(box)
        body_l.addLayout(grid)

        bot = QHBoxLayout()
        date_box = QGroupBox("Date range")
        df = QHBoxLayout(date_box)
        self._date_from_chk = QCheckBox("from")
        self._date_from = QDateEdit(); self._date_from.setDisplayFormat("yyyy-MM-dd")
        self._date_from.setCalendarPopup(True); self._date_from.setDate(QDate.currentDate().addYears(-1))
        self._date_to_chk = QCheckBox("to")
        self._date_to = QDateEdit(); self._date_to.setDisplayFormat("yyyy-MM-dd")
        self._date_to.setCalendarPopup(True); self._date_to.setDate(QDate.currentDate())
        for w in (self._date_from_chk, self._date_to_chk):
            w.toggled.connect(self._on_scope_edit)
        for w in (self._date_from, self._date_to):
            w.dateChanged.connect(self._on_scope_edit)
        df.addWidget(self._date_from_chk); df.addWidget(self._date_from)
        df.addWidget(self._date_to_chk);   df.addWidget(self._date_to)
        pick_btn = QPushButton("📅 Available…")
        pick_btn.setToolTip("Show a calendar of dates that have data")
        pick_btn.clicked.connect(self._on_pick_date)
        df.addWidget(pick_btn)
        bot.addWidget(date_box)

        name_box = QGroupBox("Filename contains")
        nl = QVBoxLayout(name_box)
        self._search = QLineEdit()
        self._search.setPlaceholderText("substring of filename or path…")
        self._search.textChanged.connect(self._on_scope_edit)
        nl.addWidget(self._search)
        bot.addWidget(name_box, 1)

        clear_btn = QPushButton("Clear filters")
        clear_btn.clicked.connect(self._clear_filters)
        bot.addWidget(clear_btn)
        body_l.addLayout(bot)

        self._scope_count_lbl = QLabel("")
        self._scope_count_lbl.setFont(style.font(self._scope_count_lbl.font(), bold=True))
        body_l.addWidget(self._scope_count_lbl)

        sec.body_layout.addWidget(body)
        return sec

    def _build_db_section(self) -> QWidget:
        sec = CollapsibleSection("Database — files in scope", expanded=True)
        body = QWidget(); body_l = QVBoxLayout(body); body_l.setContentsMargins(0,0,0,0)

        bar = QHBoxLayout()
        self._db_count_lbl = QLabel("")
        bar.addWidget(self._db_count_lbl)
        bar.addStretch(1)
        select_all_btn = QPushButton("Select all")
        select_all_btn.clicked.connect(lambda: self._db_view.selectAll())
        bar.addWidget(select_all_btn)
        self._send_btn = QPushButton("Send selection to queue →")
        self._send_btn.clicked.connect(self._on_send_to_queue)
        bar.addWidget(self._send_btn)
        body_l.addLayout(bar)

        self._db_model = FilesTableModel(_DB_COLUMNS, self)
        self._db_view = QTableView()
        self._db_view.setModel(self._db_model)
        self._db_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._db_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._db_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._db_view.setStyleSheet(_SELECTION_QSS)
        self._db_view.setSortingEnabled(True)
        self._db_view.verticalHeader().setVisible(False)
        self._db_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self._db_view.doubleClicked.connect(self._on_db_double_click)
        body_l.addWidget(self._db_view, 1)

        sec.body_layout.addWidget(body)
        return sec

    def _build_queue_section(self) -> QWidget:
        sec = CollapsibleSection("Analysis queue", expanded=True)
        body = QWidget(); body_l = QVBoxLayout(body); body_l.setContentsMargins(0,0,0,0)

        # FlowLayout: as one QHBoxLayout this row of five buttons, the segment
        # combo and the gate cluster set a 1223 px minimum, which is most of
        # why the dashboard could not be narrowed below 1503 px.
        queue_bar = FlowLayout(margin=0, h_spacing=6, v_spacing=4)

        self._open_viewer_btn = QPushButton("👁  View traces")
        self._open_viewer_btn.setToolTip(
            "Opens the raw-curve viewer, showing whichever curve the worker is "
            "on.  It carries the same navigator as the row above — one worker, "
            "two control surfaces."
        )
        self._open_viewer_btn.clicked.connect(self._on_open_viewer)
        queue_bar.addWidget(self._open_viewer_btn)

        queue_bar.addWidget(_vsep())

        self._segment_select_combo = QComboBox()
        self._segment_select_combo.addItems(["Ultimate", "Penultimate"])
        self._segment_select_combo.setToolTip(
            "Which segment supplies the dashboard, scatter, histogram and "
            "summary scalar values. This does not choose 2DH alignment.\n"
            "Ultimate = the last segment of the right-most ROI: the tether, or "
            "the whole pull if there is only one segment.\n"
            "Penultimate = the segment before it when there are two or more. "
            "With only one segment, it falls back to that same segment; it does "
            "not become blank or exclude the curve. To require a distinct "
            "penultimate segment, gate on ROI Segments ≥ 2.\n\n"
            "A manually-picked Primary segment overrides this, for that curve "
            "only."
        )
        _seg_sel = read_segment_select(self._db_path)
        self._segment_select_combo.setCurrentIndex(0 if _seg_sel == "ultimate" else 1)
        self._segment_select_combo.currentIndexChanged.connect(self._on_segment_select_changed)
        queue_bar.addWidget(LabeledControl("Reported segment:", self._segment_select_combo))

        queue_bar.addWidget(_vsep())

        # Added one by one, so these sit in the same line as every other
        # button and wrap with them, rather than as a block of their own.
        for _btn in self._build_gate_cluster():
            queue_bar.addWidget(_btn)

        self._rm_btn = QPushButton("✖  Remove from queue")
        self._rm_btn.clicked.connect(self._on_remove_from_queue)
        queue_bar.addWidget(self._rm_btn)
        self._empty_btn = QPushButton("🗑  Empty queue")
        self._empty_btn.setToolTip(
            "Pause the worker and clear the whole analysis queue, so you can "
            "enqueue a fresh selection and start over.  Analysis results already "
            "cached in the DB are kept."
        )
        self._empty_btn.clicked.connect(self._on_empty_queue)
        queue_bar.addWidget(self._empty_btn)

        self._save_queue_btn = QPushButton("💾 Save Queue As…")
        self._save_queue_btn.setToolTip(
            "Write the current queue's file list to a CSV so it can be "
            "reloaded in a later session."
        )
        self._save_queue_btn.clicked.connect(self._on_save_queue)
        queue_bar.addWidget(self._save_queue_btn)
        self._load_queue_btn = QPushButton("📂 Load Queue…")
        self._load_queue_btn.setToolTip(
            "Repopulate the queue from a saved file list — a dedicated "
            "queue-save, the classification report, or any export with a "
            "'path' column."
        )
        self._load_queue_btn.clicked.connect(self._on_load_queue)
        queue_bar.addWidget(self._load_queue_btn)
        self._export_queue_btn = QPushButton("⬇ Export table…")
        self._export_queue_btn.setToolTip(
            "Write the queue table as it stands — every visible column, raw "
            "values, one row per queued file — plus a manifest recording the "
            "columns, segment selection and file list."
        )
        self._export_queue_btn.clicked.connect(self._on_export_queue_table)
        queue_bar.addWidget(self._export_queue_btn)
        # Wrapped in a QWidget rather than addLayout()ed: height-for-width does
        # not propagate reliably through a nested layout, and a flow that
        # cannot report its wrapped height gets clipped to one row.
        _queue_bar_w = QWidget()
        _queue_bar_w.setLayout(queue_bar)
        _queue_bar_w.setSizePolicy(QSizePolicy.Policy.Preferred,
                                   QSizePolicy.Policy.Minimum)
        body_l.addWidget(_queue_bar_w)

        self._nav = NavigatorBar(self._worker, compact=True)
        body_l.addWidget(self._nav)

        # setWordWrap on both: these carry RUNTIME text whose length depends on
        # the queue, and an unwrapped QLabel reports a minimum width equal to
        # its whole string.  _gate_lbl in particular gets the acquisition-filter
        # warning appended (~330 characters, about 2000 px on one line), which
        # became the dashboard's minimum width the moment a queue was
        # populated — the window then grew off the right of the screen.
        self._worker_status_lbl = QLabel("Worker: ⏸ paused")
        self._worker_status_lbl.setWordWrap(True)
        self._worker_status_lbl.setStyleSheet(style.qss_inset(fill=True))
        body_l.addWidget(self._worker_status_lbl)

        self._gate_lbl = QLabel("")
        self._gate_lbl.setWordWrap(True)
        self._gate_lbl.setStyleSheet(style.qss_inset())
        self._gate_lbl.setTextFormat(Qt.TextFormat.RichText)
        body_l.addWidget(self._gate_lbl)

        self._queue_derived_cols: list[tuple[str, str]] = list(_QUEUE_DERIVED)
        self._queue_table = QTableWidget(0, len(_QUEUE_COLUMNS))
        self._queue_table.setHorizontalHeaderLabels([c[1] for c in _QUEUE_COLUMNS])
        self._queue_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._queue_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._queue_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._queue_table.setStyleSheet(_SELECTION_QSS)
        self._queue_table.setSortingEnabled(False)
        self._queue_table.verticalHeader().setVisible(False)
        _qhdr = self._queue_table.horizontalHeader()
        _qhdr.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        _qhdr.setSectionsClickable(True)
        _qhdr.setToolTip("Click a variable column header to view its distribution & drift over time")
        _qhdr.sectionClicked.connect(self._on_queue_header_clicked)
        self._queue_table.cellDoubleClicked.connect(self._on_queue_double_click)
        body_l.addWidget(self._queue_table, 1)

        sec.body_layout.addWidget(body)
        return sec

    def _build_gate_cluster(self) -> list[QPushButton]:
        """The event-exploration buttons, returned as a plain list.

        A list rather than a layout so the caller's FlowLayout can place each
        button as its own item: wrapped in a QWidget they travelled as one
        indivisible block and read as a separate cluster sitting apart from
        the rest of the row.
        """
        self._events_btn = QPushButton("Explore Events…")
        self._events_btn.clicked.connect(self._open_event_summary)
        blk_btn = QPushButton("View Non-events")
        blk_btn.clicked.connect(self._open_non_events)
        scat_btn = QPushButton("Plot variables…")
        scat_btn.setToolTip(
            "Scatter any per-file variable against any other, over the "
            "queued events.\n"
            "Reports Spearman rho with n, and fits a line with a 95% "
            "confidence band.\n\n"
            "Put acquisition time on X and the slope is the drift rate."
        )
        scat_btn.clicked.connect(self._open_scatter)

        self._sync_gate_buttons()
        return [self._events_btn, blk_btn, scat_btn]

    def _sync_gate_buttons(self) -> None:
        """Describe whether the current cohort has an active bounded criterion."""
        paths = self._queue_event_paths()
        has = any(_gate.has_criteria_checked(paths, self._db_path).values())
        self._events_btn.setToolTip(
            "" if has else
            "No criteria checked yet — every event currently shows as a hit. "
            "Use Filtering… inside the window to narrow it down."
        )

    def _on_export_classification_report(self) -> None:
        """Report on whatever the CURRENT scope filters express — the whole DB if none are set, the same subset "Database — files in scope" counts..."""
        kw = scope_to_query(self._scope)
        select = read_segment_select(self._db_path)
        header, rows = _db.classification_report_rows(
            self._db_path, select=select, **kw)
        if not rows:
            QMessageBox.information(
                self, "Classification report",
                "No files match the current scope — nothing written.",
            )
            return
        with _export.export_group(
            self._db_path, "classification_report", [".csv"],
            kind="classification_report",
        ) as g:
            g.contributing_files(r[0] for r in rows)
            g.note(
                window="dashboard",
                scope=kw or "whole database (no scope filters set)",
                segment_select=select,
                columns=header,
                n_rows=len(rows),
            )
            g.table(".csv", header, rows)
        QMessageBox.information(
            self, "Classification report",
            f"{len(rows)} rows.\n\n{g.message()}",
        )

    def _on_define_metadata(self) -> None:
        """Bulk-write sample metadata onto every file the CURRENT scope expresses — same cohort as "Database — files in scope" and the..."""
        kw = scope_to_query(self._scope)
        rows = _db.list_files(db_path=self._db_path, **kw)
        paths = [row["path"] for row in rows]
        if not paths:
            QMessageBox.information(
                self, "Define metadata", "No files match the current scope.")
            return
        from .bulk_metadata_dialog import BulkMetadataDialog
        dlg = BulkMetadataDialog(paths, self._db_path, self)
        if dlg.exec():
            self._refresh_db_and_counts()

    def _on_remove_files(self) -> None:
        """The undo for Add Data and for an analysis run, over the CURRENT scope — same cohort as "Database — files in scope", "Define metadata..."..."""
        kw = scope_to_query(self._scope)
        rows = _db.list_files(db_path=self._db_path, **kw)
        paths = [row["path"] for row in rows]
        if not paths:
            QMessageBox.information(
                self, "Remove these files", "No files match the current scope.")
            return
        from .remove_files_dialog import RemoveFilesDialog
        dlg = RemoveFilesDialog(paths, self._db_path, self)
        if dlg.exec():
            self._refresh_db_and_counts()
            self._refresh_queue_table()

    def _on_recheck_catalog(self) -> None:
        """Re-read catalogued files to fill in their content fingerprint and any qualification check added since they were imported."""
        from .add_data_dialog import _ScanProgress
        from . import scanner as _scanner

        n_all = len(_db.list_files(db_path=self._db_path))
        if not n_all:
            QMessageBox.information(self, "Re-check catalogued files",
                                    "The catalog is empty.")
            return
        n_missing = len([r for r in _db.list_files(db_path=self._db_path)
                         if r["content_sha256"] is None])

        box = QMessageBox(self)
        box.setWindowTitle("Re-check catalogued files")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText("Re-read catalogued files to fingerprint them and re-run "
                    "today's qualification checks?")
        box.setInformativeText(
            f"Fill in what's missing — {n_missing:,} of {n_all:,} files have no "
            f"fingerprint yet. Free to repeat; skips files already done.\n\n"
            f"Re-check everything — re-reads all {n_all:,}. Use after a "
            f"qualification rule changes, or if a file may have been replaced "
            f"on disk without its timestamp changing.\n\n"
            f"Roughly 13–50 ms per file, mostly waiting on the disk. Cancel at "
            f"any point keeps everything already done. No file on disk is "
            f"modified."
        )
        fill_btn = box.addButton(f"Fill in what's missing ({n_missing:,})",
                                 QMessageBox.ButtonRole.AcceptRole)
        all_btn  = box.addButton(f"Re-check everything ({n_all:,})",
                                 QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(fill_btn)
        box.exec()
        if box.clickedButton() not in (fill_btn, all_btn):
            return
        only_missing = box.clickedButton() is fill_btn

        prog = _ScanProgress(self, "Re-checking catalogued files…")
        try:
            summary = _scanner.requalify_catalog(
                self._db_path, progress_cb=prog, only_missing=only_missing)
        except Exception as exc:                              # noqa: BLE001
            prog.close()
            QMessageBox.critical(self, "Re-check catalogued files", repr(exc))
            return
        finally:
            prog.close()

        dupes = _db.duplicate_groups(self._db_path)
        n_copies = sum(len(g["copies"]) for g in dupes)
        QMessageBox.information(
            self, "Re-check catalogued files",
            f"{'Cancelled after' if summary['cancelled'] else 'Finished —'} "
            f"{summary['hashed']:,} of {summary['seen']:,} files fingerprinted.\n"
            f"{summary['requalified']:,} changed classification.\n"
            f"{summary['unreadable']:,} could not be read (drive disconnected?) "
            f"— nothing was written for those, so they are retried next time.\n\n"
            f"{n_copies:,} redundant copies found across {len(dupes):,} sets of "
            f"identical files. They stay in the catalog, labelled in the "
            f"'Duplicate of' column and left out of scope; tick 'Show ONLY "
            f"redundant copies' in Edit Scope to select them."
        )
        self._refresh_db_and_counts()

    def _on_save_queue(self) -> None:
        paths = _db.queue_paths(self._db_path)
        if not paths:
            QMessageBox.information(self, "Save Queue As…", "Queue is empty — nothing written.")
            return
        with _export.export_group(
            self._db_path, "queue", [".csv"], kind="queue_save",
        ) as g:
            g.contributing_files(paths)
            g.note(window="dashboard",
                   note="Reload with Load Queue. v1 restores the file set only; "
                        "the parameter set active at save time is not recalled.")
            g.table(".csv", ["path"], [[p] for p in paths])
        QMessageBox.information(
            self, "Save Queue As…", f"{len(paths)} path(s).\n\n{g.message()}")

    def _on_export_queue_table(self) -> None:
        """Export the queue table exactly as it stands: one row per queued file, one column per column currently shown, raw values."""
        rows = _db.list_queue(self._db_path)
        if not rows:
            QMessageBox.information(self, "Export queue table",
                                     "Queue is empty — nothing written.")
            return
        paths = [r["path"] for r in rows]
        col_values = self._fetch_queue_column_data(paths)
        hit_set, _reasons = self._gate_hit_and_reasons(
            [r["path"] for r in rows if r["event"] == "event"])

        derived_cols = list(self._queue_derived_cols)
        header = ["path", "filename", "status", "event", "hit"] + [
            label for _key, label in derived_cols]
        out = []
        for row in rows:
            path = row["path"]
            cls  = row["event"] or ""
            verdict = ("hit" if path in hit_set else "non_hit") if cls == "event" else ""
            rec = [path, row["filename"] or "", row["status"] or "", cls, verdict]
            for key, _label in derived_cols:
                v = self._queue_cell_value(key, path, col_values)
                rec.append("" if v is None else v)
            out.append(rec)

        with _export.export_group(
            self._db_path, "queue_table", [".csv"], kind="queue_table",
        ) as g:
            g.contributing_files(paths)
            g.note(
                window="dashboard_queue",
                segment_select=read_segment_select(self._db_path),
                columns=header,
                column_keys=["path", "filename", "status", "event", "hit"]
                            + [k for k, _l in derived_cols],
                n_rows=len(out),
            )
            g.table(".csv", header, out)
        QMessageBox.information(
            self, "Export queue table", f"{len(out)} rows.\n\n{g.message()}")

    def _on_load_queue(self) -> None:
        export_dir = _export.resolve_export_dir(self._db_path)
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Load Queue", str(export_dir), "CSV files (*.csv);;All files (*)",
        )
        if not chosen:
            return
        import csv
        with open(chosen, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "path" not in reader.fieldnames:
                QMessageBox.warning(
                    self, "Load Queue",
                    "This file has no 'path' column — Load Queue needs one. "
                    "A dedicated Save Queue file, the classification report, "
                    "or any export carrying a 'path' column all work; a "
                    "filename-only export doesn't, since it can't be "
                    "resolved back to one file unambiguously.",
                )
                return
            paths = [row["path"] for row in reader if row.get("path")]
        n_enqueued, n_missing = _db.import_queue_from_paths(paths, self._db_path)
        self._worker.invalidate_queue_cache()
        self._worker.notify_work_available()
        self._refresh_queue_table()
        msg = f"Enqueued {n_enqueued} file(s)."
        if n_missing:
            msg += f"\n{n_missing} path(s) not found in this database and were skipped."
        QMessageBox.information(self, "Load Queue", msg)

    def _update_export_dir_label(self) -> None:
        """Show where exports currently go, next to the button that sets it."""
        current = _export.resolve_export_dir(self._db_path)
        is_default = not _db.get_app_setting(
            _db.APP_SETTING_EXPORT_DIR, "", self._db_path)
        shown = str(current)
        if len(shown) > 44:
            shown = "…" + shown[-43:]
        self._export_dir_lbl.setText(
            f"Exports → {shown}" + ("  (default)" if is_default else ""))
        self._export_dir_lbl.setToolTip(
            f"Every export in the app writes to:\n{current}\n\n"
            + ("No folder chosen, so this is the database's own directory."
               if is_default else
               "Chosen with the Export folder button. One setting for the "
               "whole database.")
        )

    def _on_set_export_folder(self) -> None:
        current = _export.resolve_export_dir(self._db_path)
        chosen = QFileDialog.getExistingDirectory(
            self, "Export folder", str(current),
        )
        if chosen:
            _export.set_export_dir_override(chosen, self._db_path)
            self._update_export_dir_label()


    def _refresh_facets(self) -> None:
        """Cascading facets."""
        scope = self._scope
        try:
            facets = _db.get_facet_options(
                self._db_path,
                users=scope.get("users") or None,
                analytes=scope.get("analytes") or None,
                solvents=scope.get("solvents") or None,
                afm_units=scope.get("afm_units") or None,
                curve_types=scope.get("curve_types") or None,
                date_from=scope.get("date_from"),
                date_to=scope.get("date_to"),
                search=scope.get("search"),
            )
        except Exception:
            return

        for key, lst in self._lists.items():
            checked = set(scope.get(key) or [])
            counts = {v: n for v, n in facets.get(key, [])}
            values = sorted(set(counts) | checked, key=lambda s: str(s).lower())
            lst.blockSignals(True)
            lst.clear()
            for v in values:
                n = counts.get(v, 0)
                item = QListWidgetItem(f"{v}  ({n:,})")
                item.setData(Qt.ItemDataRole.UserRole, v)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.CheckState.Checked if v in checked else Qt.CheckState.Unchecked
                )
                lst.addItem(item)
            lst.blockSignals(False)

    def _current_scope(self) -> dict:
        out = new_scope()
        for key, lst in self._lists.items():
            out[key] = [
                lst.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(lst.count())
                if lst.item(i).checkState() == Qt.CheckState.Checked
            ]
        if self._date_from_chk.isChecked():
            out["date_from"] = self._date_from.date().toString("yyyy-MM-dd")
        if self._date_to_chk.isChecked():
            out["date_to"] = self._date_to.date().toString("yyyy-MM-dd")
        s = self._search.text().strip()
        out["search"] = s or None
        return out

    def _on_scope_edit(self, *_args) -> None:
        self._scope = self._current_scope()
        self._count_timer.start()
        self.scope_changed.emit(self._scope)
        self._prune_children()
        for win in self._children:
            if hasattr(win, "apply_scope") and win.isVisible():
                try:
                    win.apply_scope(self._scope)
                except Exception:
                    pass

    def _prune_children(self) -> None:
        """Drop child windows that have been closed or destroyed."""
        alive = []
        for win in self._children:
            try:
                if win.isVisible():
                    alive.append(win)
            except RuntimeError:
                pass
        self._children = alive

    def _clear_filters(self) -> None:
        for lst in self._lists.values():
            lst.blockSignals(True)
            for i in range(lst.count()):
                lst.item(i).setCheckState(Qt.CheckState.Unchecked)
            lst.blockSignals(False)
        self._date_from_chk.setChecked(False)
        self._date_to_chk.setChecked(False)
        self._search.clear()
        self._on_scope_edit()


    def _refresh_db_and_counts(self) -> None:
        self._refresh_facets()
        kw = scope_to_query(self._scope)
        try:
            rows = _db.list_files(db_path=self._db_path, **kw)
        except Exception as exc:
            self._db_count_lbl.setText(f"(query failed: {exc!r})")
            return
        total = len(rows)
        self._scope_count_lbl.setText(f"{total:,} files match scope")

        self._db_model.set_rows(rows)
        self._db_count_lbl.setText(f"{total:,} files in scope")

        self._refresh_queue_table()

    def _on_db_double_click(self, index) -> None:
        path = self._db_model.row_path(index.row())
        if path:
            self._open_raw_viewer(path)

    def _on_queue_double_click(self, row: int, _col: int) -> None:
        item = self._queue_table.item(row, 0)
        path = item.data(Qt.ItemDataRole.UserRole + 1) if item else None
        if path:
            self._open_raw_viewer(path)

    def _selected_db_ids(self) -> list[int]:
        """File ids of every selected row in the DB view."""
        ids = []
        for idx in self._db_view.selectionModel().selectedRows():
            fid = self._db_model.row_id(idx.row())
            if fid is not None:
                ids.append(int(fid))
        return ids


    def _compute_queue_derived_cols(self) -> list[tuple[str, str]]:
        """Build the derived-column (key, label) list from the analysis_type keys actually present for queued files."""
        present = set(_db.get_queue_analysis_types(self._db_path)) - _QUEUE_HIDE
        keep = set(_QUEUE_BASE_KEYS) | present
        cols: list[tuple[str, str]] = []
        seen: set[str] = set()
        for key in _QUEUE_DERIVED_ORDER:
            if key in keep:
                cols.append((key, _prettify_key(key)))
                seen.add(key)
        for key in sorted(present - seen):
            cols.append((key, _prettify_key(key)))
        return cols

    def _fetch_queue_column_data(
        self, paths: list[str],
    ) -> tuple[dict, dict]:
        """THE one place that fetches queue-derived-column source data for a set of paths — {path: {key: value}}."""
        keys = [k for k, _ in self._queue_derived_cols]
        return _vars.values(paths, keys, self._db_path)

    @staticmethod
    def _queue_cell_value(key: str, path: str, values: dict):
        """RAW value for one (column key, file path) queue cell, or None. _queue_cell_text formats this for the table; the queue export writes it..."""
        return values.get(_db.normalize_path(path), {}).get(key)

    @classmethod
    def _queue_cell_text(cls, key: str, path: str, values: dict) -> str:
        """Display text for one (column key, file path) queue cell, given the dict _fetch_queue_column_data returned."""
        raw = cls._queue_cell_value(key, path, values)
        return "" if raw is None else _fmt_cell(raw, key)

    def _refresh_queue_table(self) -> None:
        rows = _db.list_queue(self._db_path)

        self._queue_derived_cols = self._compute_queue_derived_cols()
        self._queue_table.setColumnCount(len(_QUEUE_COLUMNS_FIXED) + len(self._queue_derived_cols))
        self._queue_table.setHorizontalHeaderLabels(
            [c[1] for c in _QUEUE_COLUMNS_FIXED]
            + [c[1] for c in self._queue_derived_cols]
        )
        for c, (key, _label) in enumerate(_QUEUE_COLUMNS_FIXED):
            tip = _FIXED_COL_TOOLTIPS.get(key)
            if tip:
                self._queue_table.horizontalHeaderItem(c).setToolTip(tip)
        for c, (key, _label) in enumerate(self._queue_derived_cols, start=len(_QUEUE_COLUMNS_FIXED)):
            tip = _vars.describe(key)
            if tip:
                self._queue_table.horizontalHeaderItem(c).setToolTip(tip)

        paths = [r["path"] for r in rows]
        col_values = self._fetch_queue_column_data(paths)

        nb_paths = [r["path"] for r in rows if r["event"] == "event"]
        hit_set, gate_reasons = self._gate_hit_and_reasons(nb_paths)

        self._freshness = self._compute_freshness()

        self._queue_table.setRowCount(len(rows))
        self._queue_id_to_row = {}
        self._queue_row_ids = [int(row["file_id"]) for row in rows]
        start_derived = len(_QUEUE_COLUMNS_FIXED)
        for r, row in enumerate(rows):
            self._queue_id_to_row[int(row["file_id"])] = r
            self._queue_table.setItem(r, 0, QTableWidgetItem(row["filename"] or ""))
            self._set_status_cell(r, int(row["file_id"]), row["status"])
            cls = row["event"] or ""
            self._queue_table.setItem(r, 2, QTableWidgetItem(cls))
            self._set_hit_cell(r, cls, row["path"], hit_set, gate_reasons)
            self._queue_table.item(r, 0).setData(Qt.ItemDataRole.UserRole, row["file_id"])
            self._queue_table.item(r, 0).setData(Qt.ItemDataRole.UserRole + 1, row["path"])
            for c, (key, _label) in enumerate(self._queue_derived_cols, start=start_derived):
                text = self._queue_cell_text(key, row["path"], col_values)
                self._queue_table.setItem(r, c, QTableWidgetItem(text))
            bg = _bg_for(cls if cls else None, row["status"])
            for c in range(self._queue_table.columnCount()):
                self._queue_table.item(r, c).setBackground(QBrush(bg))
        self._queue_table.resizeColumnsToContents()

        self._queue_class_counts = {k: 0 for k in _db.EVENT_VERDICTS} | {"unclassified": 0}
        for r in rows:
            k = r["event"] if r["event"] else "unclassified"
            self._queue_class_counts[k] = self._queue_class_counts.get(k, 0) + 1

        self._queue_total = len(rows)
        self._done_ids = {int(r["file_id"]) for r in rows if r["status"] == "done"}
        self._rate_times.clear()
        self._loaded_count = 0
        self._cached_count = 0
        self._last_done_t = None
        self._update_location_label()
        self._update_gate_label()

        self._refresh_population_children()

    def _selected_ids(self, table: QTableWidget) -> list[int]:
        ids = []
        for idx in table.selectionModel().selectedRows():
            item = table.item(idx.row(), 0)
            if item is not None:
                fid = item.data(Qt.ItemDataRole.UserRole)
                if fid is not None:
                    ids.append(int(fid))
        return ids

    def _on_send_to_queue(self) -> None:
        ids = self._selected_db_ids()
        if not ids:
            return
        _db.enqueue_files(ids, self._db_path)
        self._worker.invalidate_queue_cache()
        self._refresh_queue_table()
        self._worker.notify_work_available()

    def _on_segment_select_changed(self, idx: int) -> None:
        write_segment_select("ultimate" if idx == 0 else "penultimate", self._db_path)
        self._refresh_queue_table()
        self._on_criteria_changed()

    def _on_remove_from_queue(self) -> None:
        ids = self._selected_ids(self._queue_table)
        if not ids:
            return
        _db.dequeue_files(ids, self._db_path)
        self._worker.invalidate_queue_cache()
        self._refresh_queue_table()

    def _on_empty_queue(self) -> None:
        n = self._queue_table.rowCount()
        if n == 0:
            return
        if QMessageBox.question(
            self, "Empty queue",
            f"Remove all {n:,} files from the analysis queue?\n\n"
            "Cached analysis results are kept; this only clears the queue so "
            "you can enqueue a fresh selection.",
        ) != QMessageBox.StandardButton.Yes:
            return
        self._worker.set_paused(True)
        _db.clear_analysis_queue(self._db_path)
        self._worker.invalidate_queue_cache()
        self._refresh_queue_table()


    def _on_worker_paused_changed(self, paused: bool) -> None:
        self._update_location_label()

    def _on_worker_playhead_changed(self, file_id: int) -> None:
        self._current_playhead_id = file_id
        self._update_location_label()

    def _on_worker_direction_changed(self, d: int) -> None:
        self._current_direction = d
        self._update_location_label()

    def _on_worker_queue_empty(self) -> None:
        self._update_location_label(extra=" — reached the end of the queue")

    def _compute_freshness(self) -> dict[int, str]:
        """{file_id: 'fresh'|'stale'|'new'} for the queue, against live params."""
        try:
            from .curve_analysis import current_signature
            params_json, code_ver = current_signature(self._db_path)
            return _db.queue_freshness(params_json, code_ver, self._db_path)
        except Exception:
            return {}

    def _status_class(self, file_id: int, status: str | None) -> str:
        """The CLASS the Status column reports: what this row will cost."""
        if status == "running":
            return "running"
        if status and status.startswith("error: "):
            return "error"
        return _FRESHNESS_LABEL.get(self._freshness.get(int(file_id)), "not analysed")

    def _set_status_cell(self, row: int, file_id: int, status: str | None) -> None:
        """Status cell: what this row NEEDS, plus whether it has been visited."""
        text = self._status_class(file_id, status)
        if status == "running":
            tip = "The worker is analysing this file now."
        elif status and status.startswith("error: "):
            tip = status
        else:
            fresh = self._freshness.get(int(file_id))
            if fresh == "fresh":
                tip = ("A verdict is stored under the parameter set and code "
                       "version in force now, so the worker will visit this "
                       "file and skip it without reading the curve.")
            elif fresh == "stale":
                tip = ("A verdict is stored, but under different parameters or "
                       "a different scientific-method version, so this file will be fully "
                       "re-analysed when the playhead reaches it.")
            else:
                tip = "Nothing stored for this file yet — a full analysis."
            if status == "done":
                text += " · visited"
                tip += "\n\nThe playhead has passed this row this session."
        item = QTableWidgetItem(text)
        item.setToolTip(tip)
        item.setData(Qt.ItemDataRole.UserRole, status or "")
        self._queue_table.setItem(row, 1, item)

    def _raw_status(self, row: int) -> str | None:
        item = self._queue_table.item(row, 1)
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole) or None

    def _freshness_line(self) -> str:
        """`3,796 up to date · 1,204 stale · 0 not analysed` for the queue."""
        counts = {k: 0 for k in _db.QUEUE_FRESHNESS}
        for fid in self._queue_row_ids:
            k = self._freshness.get(fid)
            if k in counts:
                counts[k] += 1
        parts = [f"{counts[k]:,} {_FRESHNESS_LABEL[k]}"
                 for k in _db.QUEUE_FRESHNESS if counts[k]]
        return "  ·  ".join(parts)

    def _current_rate(self) -> float:
        """Throughput (files/s) over a trailing 15 s window; 0 if too few."""
        now = time.monotonic()
        while self._rate_times and now - self._rate_times[0] > 15.0:
            self._rate_times.popleft()
        if len(self._rate_times) >= 2:
            span = self._rate_times[-1] - self._rate_times[0]
            if span > 0:
                return (len(self._rate_times) - 1) / span
        return 0.0

    def _mean_cost(self, cached: bool) -> float | None:
        """Mean seconds per file for one cost class; None if never seen one."""
        vals = [dt for dt, was_cached in self._cost_samples if was_cached is cached]
        return (sum(vals) / len(vals)) if vals else None

    def _files_ahead(self) -> list[int]:
        """File ids the playhead will reach, in the direction it is travelling."""
        ids = self._queue_row_ids
        if not ids:
            return []
        forward = getattr(self, "_current_direction", +1) >= 0
        head_id = getattr(self, "_current_playhead_id", None)
        row = self._queue_id_to_row.get(int(head_id)) if head_id is not None else None
        if row is None:
            return list(ids) if forward else list(reversed(ids))
        return ids[row + 1:] if forward else ids[:row][::-1]

    def _eta_text(self) -> str | None:
        """`ETA 4m 20s (1,204 to analyse, 3,796 cached)`, or None if unknowable."""
        ahead = self._files_ahead()
        if not ahead:
            return None
        n_fresh = sum(1 for fid in ahead if self._freshness.get(fid) == "fresh")
        n_work  = len(ahead) - n_fresh

        cost_fresh = self._mean_cost(True)
        cost_work  = self._mean_cost(False)
        total = 0.0
        floor = False
        for n, cost in ((n_fresh, cost_fresh), (n_work, cost_work)):
            if not n:
                continue
            if cost is None:
                floor = True
            else:
                total += n * cost
        if total <= 0:
            return None

        detail = []
        if n_work:
            detail.append(f"{n_work:,} to analyse")
        if n_fresh:
            detail.append(f"{n_fresh:,} cached")
        prefix = "ETA ≥ " if floor else "ETA "
        return prefix + self._fmt_eta(total) + (f" ({', '.join(detail)})" if detail else "")

    def _update_location_label(self, extra: str = "") -> None:
        """Line 1 — where the worker is *right now*: play state, direction, playhead position in queue order, current file, live throughput, and..."""
        worker = getattr(self, "_worker", None)
        paused = worker.is_paused() if worker is not None else True
        d_attr = getattr(self, "_current_direction", +1)
        direction = "forward" if d_attr >= 0 else "reverse"
        parts = ["⏸ paused" if paused else f"▶ running ({direction})"]

        head_id = getattr(self, "_current_playhead_id", None)
        if head_id is not None:
            r = self._queue_id_to_row.get(int(head_id))
            if r is not None and self._queue_total:
                parts.append(f"at {r + 1:,} / {self._queue_total:,}")
            fname = self._filename_for_playhead(head_id)
            if fname:
                parts.append(f"current: {fname}")

        rate = self._current_rate()
        if rate > 0:
            parts.append(f"{rate:.1f}/s")
        eta = self._eta_text()
        if eta:
            parts.append(eta)

        if self._loaded_count or self._cached_count:
            parts.append(f"loaded {self._loaded_count:,} · cached {self._cached_count:,}")

        self._worker_status_lbl.setText("Worker: " + "  ·  ".join(parts) + extra)

    def _acq_filter_line(self, queue_rows) -> str:
        """Warn when the ACQUISITION filter, not ours, is setting the bandwidth."""
        cutoff = _db.load_analysis_params(self._db_path).spectral_cutoff_hz
        affected: dict[float, int] = {}
        for r in queue_rows:
            if "force_filter_bw_hz" not in r.keys():
                continue
            bw = r["force_filter_bw_hz"]
            if _sp.filter_bandwidth_conflict(cutoff, bw):
                affected[bw] = affected.get(bw, 0) + 1
        if not affected:
            return ""
        n = sum(affected.values())
        bws = " / ".join(f"{b:,.0f}" for b in sorted(affected))
        return "<br>" + style.html_text(
            f"⚠ {n:,} queued curve{'s' if n != 1 else ''} acquired with a "
            f"{bws} Hz filter, at or below our {cutoff:,.0f} Hz cutoff. "
            f"{FILTER_BANDWIDTH_CONSEQUENCE}",
            style.TEXT_WARNING, bold=True)

    def _update_gate_label(self) -> None:
        """Line 2 — the Level-3 gate."""
        c = self._queue_class_counts
        total = self._queue_total
        owner = _db.active_param_owner(self._db_path)
        queue_rows = _db.list_queue(self._db_path)
        owners = {r["experimentalist"] for r in queue_rows
                  if "experimentalist" in r.keys() and r["experimentalist"]}
        mixed = (f" — queue holds {len(owners)} experimentalists, "
                 f"one set applies to all" if len(owners) > 1 else "")
        params_line = (
            "<br>" + style.html_text(
                f"Parameters: <b>{owner}</b>'s set{mixed}")
        )
        fresh_line = self._freshness_line()
        if fresh_line:
            params_line += "<br>" + style.html_text(f"Under those: {fresh_line}")
        params_line += self._acq_filter_line(queue_rows)
        if total == 0:
            self._gate_lbl.setText(
                style.html_text("Queue empty — send files from the "
                                "database list above.", style.UI_FAINT)
                + params_line
            )
            return
        analysed = c["event"] + c["non_event"]
        unavailable = c.get("unavailable", 0)
        unusable = c.get("unusable", 0)
        todo = c.get("unclassified", 0) + unavailable
        non_hit = self._count_non_hit()
        hit = max(c["event"] - non_hit, 0)
        breakdown = (
            f"{analysed:,} analysed "
            f"({c['event']:,} events → {hit:,} hit · {non_hit:,} non-hit; "
            f"{c['non_event']:,} non-events)"
        )
        warn = ""
        if unavailable > 0:
            warn += " · " + style.html_text(
                f"{unavailable:,} unavailable (check data drive)",
                style.TEXT_WARNING, bold=True)
        if unusable > 0:
            warn += " · " + style.html_text(
                f"{unusable:,} unusable — see the Unusable column",
                style.UI_MUTED)
        if todo > 0:
            tail = style.html_text(f"{todo:,} not yet analysed",
                                  style.TEXT_BAD, bold=True)
        else:
            tail = style.html_text("✓ ready for View Events",
                                  style.TEXT_GOOD, bold=True)
        self._gate_lbl.setText(f"{total:,} in queue · {breakdown} · {tail}{warn}" + params_line)

    def _filename_for_playhead(self, head_id: int) -> Optional[str]:
        """Filename for the worker's current file, without touching the DB."""
        name = self._db_model.value_for_id(head_id, "filename")
        if name:
            return name
        r = self._queue_id_to_row.get(int(head_id))
        if r is not None:
            item = self._queue_table.item(r, 0)
            if item is not None:
                return item.text()
        return None

    def _on_open_viewer(self) -> None:
        """Open the raw curve window as the worker's playback controller (singleton)."""
        from .rawcurve_window import RawCurveWindow
        viewer = getattr(self, "_viewer", None)
        if viewer is None or not viewer.isVisible():
            self._prune_children()
            self._viewer = RawCurveWindow(
                paths=[],
                db_path=self._db_path,
                worker=self._worker,
            )
            self._children.append(self._viewer)
            self._viewer.show()
        else:
            self._viewer.raise_()
            self._viewer.activateWindow()

    def _open_raw_viewer(self, path: str) -> None:
        """Open the worker-following viewer and seed it with this specific file."""
        fid = _db.get_file_id(path, self._db_path)
        self._on_open_viewer()
        if fid is not None:
            # A request from an inspection window means "stay on this curve",
            # not "flash it briefly and let active playback move straight on".
            self._worker.set_paused(True)
            if fid not in self._queue_id_to_row:
                _db.enqueue_files([fid], self._db_path)
                self._worker.invalidate_queue_cache()
                self._refresh_queue_table()
            self._worker.step_to(fid)

    def reveal_raw_at(self, path: str) -> None:
        """Public entry point for other windows (e.g. the WLC fit window's 'Raw' button): open — or reuse — the singleton raw viewer and navigate..."""
        self._open_raw_viewer(path)

    def reveal_roi_at(self, path: str) -> None:
        """Like reveal_raw_at, then reveal the viewer's ROI detection window on the same curve (it follows the worker's playhead)."""
        self._open_raw_viewer(path)
        viewer = getattr(self, "_viewer", None)
        if viewer is not None:
            opener = getattr(viewer, "open_roi_window", None)
            if callable(opener):
                opener()


    def _on_file_started(self, file_id: int) -> None:
        self._started_buffer.append(file_id)

    def _on_file_done(self, file_id: int, event: str, was_cached: bool) -> None:
        self._done_buffer.append((file_id, event, was_cached))

    def _on_file_error(self, file_id: int, msg: str) -> None:
        # file_started precedes every terminal signal. Remove its buffered UI
        # update so the next 150 ms flush cannot repaint this row as running.
        self._started_buffer = [fid for fid in self._started_buffer if fid != file_id]
        self._update_queue_row(file_id, status=f"error: {msg[:40]}")

    def _on_data_unavailable(self, file_id: int, path: str, detail: str) -> None:
        self._started_buffer = [fid for fid in self._started_buffer if fid != file_id]
        QMessageBox.warning(
            self,
            "Raw data unavailable",
            f"Analysis has been paused because this file could not be opened:\n\n"
            f"{path}\n\n{detail}\n\n"
            "The queue and existing analysis results have not been changed. "
            "Reconnect the data source, then press Play to retry this file. "
            "You can leave analysis paused and continue browsing catalog data "
            "that is already stored in the database.",
        )
        # A modal dialog runs an event loop, including the buffer timer. Restore
        # the authoritative pending state after it closes as well as in the DB.
        self._update_queue_row(file_id, status="pending")

    def _on_worker_fatal_error(self, message: str) -> None:
        QMessageBox.critical(
            self, "Analysis worker stopped",
            f"{message}\n\nThe dashboard is still available, but analysis "
            "cannot continue in this session.",
        )

    def _flush_worker_events(self) -> None:
        """Drain buffered worker events at a fixed cadence (150 ms)."""
        started = self._started_buffer
        done    = self._done_buffer
        if not started and not done:
            return
        self._started_buffer = []
        self._done_buffer = []

        for fid in started:
            self._update_queue_row(fid, status="running")

        if not done:
            return

        paths_by_fid: dict[int, str] = {}
        for fid, _cls, _was_cached in done:
            row_idx = self._queue_id_to_row.get(int(fid))
            if row_idx is None:
                continue
            item = self._queue_table.item(row_idx, 0)
            if item is None:
                continue
            p = item.data(Qt.ItemDataRole.UserRole + 1)
            if p:
                paths_by_fid[int(fid)] = p
        col_values = self._fetch_queue_column_data(list(paths_by_fid.values()))

        done_nb = [paths_by_fid[int(fid)] for fid, cls, _was_cached in done
                   if cls == "event" and int(fid) in paths_by_fid]
        done_hit, done_reasons = self._gate_hit_and_reasons(done_nb)

        gate_dirty = False
        now = time.monotonic()
        for fid, cls, was_cached in done:
            row_idx = self._queue_id_to_row.get(int(fid))
            old_key = "unclassified"
            if row_idx is not None:
                cell = self._queue_table.item(row_idx, 2)
                if cell is not None and cell.text():
                    old_key = cell.text()

            # A successful terminal result was either recomputed under the
            # live signature or accepted from that signature's cache.  Update
            # the derived state before painting the row; otherwise a stale row
            # briefly (and, while this window stays open, indefinitely) reads
            # "stale params · visited" until some later full refresh happens.
            if cls in ("event", "non_event"):
                self._freshness[int(fid)] = "fresh"
            self._update_queue_row(fid, status="done", event=cls)
            self._done_ids.add(int(fid))
            self._rate_times.append(now)
            if was_cached:
                self._cached_count += 1
            else:
                self._loaded_count += 1

            if self._last_done_t is not None:
                dt = now - self._last_done_t
                if 0 < dt <= _ETA_MAX_SAMPLE_S:
                    self._cost_samples.append((dt, bool(was_cached)))
            self._last_done_t = now

            new_key = cls or "unclassified"
            if old_key != new_key:
                self._queue_class_counts[old_key] = max(
                    0, self._queue_class_counts.get(old_key, 0) - 1)
                self._queue_class_counts[new_key] = self._queue_class_counts.get(new_key, 0) + 1
                gate_dirty = True

            path = paths_by_fid.get(int(fid))
            if row_idx is not None and path:
                self._set_hit_cell(row_idx, cls, path, done_hit, done_reasons)
                for c, (key, _label) in enumerate(self._queue_derived_cols, start=len(_QUEUE_COLUMNS_FIXED)):
                    text = self._queue_cell_text(key, path, col_values)
                    item = self._queue_table.item(row_idx, c)
                    if item is None:
                        self._queue_table.setItem(row_idx, c, QTableWidgetItem(text))
                    else:
                        item.setText(text)

            if self._db_model.index_for_id(fid) is not None:
                self._db_model.update_field(fid, "event", cls)

        if gate_dirty:
            self._update_gate_label()

        self._update_location_label()

        self._refresh_population_children()

    @staticmethod
    def _fmt_eta(seconds: float) -> str:
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds}s"
        m, s = divmod(seconds, 60)
        if m < 60:
            return f"{m}m {s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h {m:02d}m"


    def _update_queue_row(
        self,
        file_id: int,
        status: Optional[str] = None,
        event: Optional[str] = None,
    ) -> None:
        r = self._queue_id_to_row.get(int(file_id))
        if r is None:
            return
        if status is not None:
            self._set_status_cell(r, int(file_id), status)
        if event is not None:
            self._queue_table.item(r, 2).setText(event)
        cls_text = self._queue_table.item(r, 2).text() or None
        st_text  = self._raw_status(r)
        bg = _bg_for(cls_text, st_text)
        for c in range(self._queue_table.columnCount()):
            self._queue_table.item(r, c).setBackground(QBrush(bg))


    def _on_pick_date(self) -> None:
        from .date_picker_dialog import DatePickerDialog
        dlg = DatePickerDialog(self._db_path, self, scope=self._scope)
        if not dlg.exec():
            return
        if dlg.date_from is not None:
            self._date_from_chk.setChecked(True)
            self._date_from.setDate(dlg.date_from)
        if dlg.date_to is not None:
            self._date_to_chk.setChecked(True)
            self._date_to.setDate(dlg.date_to)


    def _on_add_data(self) -> None:
        from .add_data_dialog import AddDataDialog
        dlg = AddDataDialog(self._db_path, self)
        if dlg.exec():
            self._refresh_db_and_counts()


    def _on_repoint_data(self) -> None:
        """Rewrite the stored paths of curves that have moved to another drive, keeping the analysis attached to them."""
        from .repoint_dialog import RepointDataDialog
        dlg = RepointDataDialog(self._db_path, self)
        dlg.exec()
        if dlg.changed:
            self._refresh_db_and_counts()
            self._refresh_queue_table()


    def _spawn(self, win: QWidget) -> None:
        self._prune_children()
        self._children.append(win)
        win.show()
        win.raise_()
        win.activateWindow()

    _CATEGORICAL_COLS = {2: ("event", "Event")}

    def _on_queue_header_clicked(self, section: int) -> None:
        """Open a per-column stats view for a clicked queue header."""
        rows = _db.list_queue(self._db_path)
        if not rows:
            return

        if section == 1:
            self._refresh_freshness()
            pairs = [(r["path"], self._status_class(int(r["file_id"]), r["status"]))
                     for r in rows]
            from .categorical_window import CategoricalStatsWindow
            win = CategoricalStatsWindow(
                "status", "Status (work outstanding)", pairs, self._db_path,
                session_info=None)
            win.view_file_requested.connect(self._open_raw_viewer)
            self._spawn(win)
            return

        if section == 3:
            nb_paths = [r["path"] for r in rows if r["event"] == "event"]
            hit_set = set(_gate.evaluate(nb_paths, self._db_path)[0]) if nb_paths else set()
            pairs = [
                (r["path"], self._hit_text(r["event"] or "", r["path"], hit_set) or "—")
                for r in rows
            ]
            from .categorical_window import CategoricalStatsWindow
            win = CategoricalStatsWindow("hit", "Hit", pairs, self._db_path,
                                         session_info=None)
            win.view_file_requested.connect(self._open_raw_viewer)
            self._spawn(win)
            return

        if section in self._CATEGORICAL_COLS:
            key, label = self._CATEGORICAL_COLS[section]
            pairs = [(r["path"], r[key]) for r in rows]
            from .categorical_window import CategoricalStatsWindow
            win = CategoricalStatsWindow(key, label, pairs, self._db_path,
                                         session_info=None)
            win.view_file_requested.connect(self._open_raw_viewer)
            self._spawn(win)
            return

        base = len(_QUEUE_COLUMNS_FIXED)
        di = section - base
        if not (0 <= di < len(self._queue_derived_cols)):
            return
        key, label = self._queue_derived_cols[di]
        paths = [r["path"] for r in rows]

        from .variable_window import VariableStatsWindow
        win = VariableStatsWindow(key, label, paths, self._db_path, session_info=None)
        win.view_file_requested.connect(self._open_raw_viewer)
        win.thresholds_changed.connect(self._on_criteria_changed)
        self._spawn(win)

    def _gate_hit_and_reasons(self, nb_paths: list[str]) -> tuple[set, dict]:
        """Run the gate once and return (hit_set, reasons)."""
        if not nb_paths:
            return set(), {}
        reasons = _gate.explain(nb_paths, self._db_path)
        hit_set = set(nb_paths) - set(reasons)
        return hit_set, reasons

    def _set_hit_cell(self, r_idx: int, cls: str, path: str,
                      hit_set: set, reasons: dict) -> None:
        """Write the Hit cell text + a 'why non-hit' tooltip in one place."""
        text = self._hit_text(cls, path, hit_set)
        cell = self._queue_table.item(r_idx, 3)
        if cell is None:
            cell = QTableWidgetItem(text)
            self._queue_table.setItem(r_idx, 3, cell)
        else:
            cell.setText(text)
        cell.setToolTip(self._hit_tooltip(text, path, reasons))

    def _hit_tooltip(self, text: str, path: str, reasons: dict) -> str:
        """Human-readable list of failing criteria for a non-hit curve."""
        if text != "non_hit":
            return ""
        rs = reasons.get(path) or []
        if not rs:
            return "Did not pass the criteria gate."
        label_of = {k: lbl for k, lbl in self._queue_derived_cols}
        lines = []
        for key, value, kind, bound in rs:
            lbl = label_of.get(key, key)
            if kind == "missing":
                lines.append(f"{lbl}: missing / NaN")
            else:
                shown = _quant.format_value(key, value, with_unit=True)
                limit = _quant.format_value(key, bound, with_unit=True)
                lines.append(f"{lbl}: {shown} ({kind} {limit})")
        return "Failed criteria:\n" + "\n".join(lines)

    @staticmethod
    def _hit_text(cls: str, path: str, hit_set: set) -> str:
        """hit-column cell — three honest states: event -> 'hit'/'non_hit' (the gate ruled) non_event -> '—' (analysed, but no event, so hit is not..."""
        if cls == "event":
            return "hit" if path in hit_set else "non_hit"
        if cls == "non_event":
            return "—"
        return ""

    def _count_non_hit(self) -> int:
        """Number of queued events the gate currently calls non-hit."""
        rows = _db.list_queue(self._db_path)
        nb_paths = [r["path"] for r in rows if r["event"] == "event"]
        if not nb_paths:
            return 0
        has_crit = _gate.has_criteria_checked(nb_paths, self._db_path)
        _hits, non_hits = _gate.evaluate(nb_paths, self._db_path)
        return sum(1 for p in non_hits if has_crit.get(p, False))

    def _refresh_hit_column(self) -> None:
        """Recompute the Hit column from the criteria gate (e.g. after a criterion changes) without rebuilding the whole queue table."""
        rows = _db.list_queue(self._db_path)
        nb_paths = [r["path"] for r in rows if r["event"] == "event"]
        hit_set, gate_reasons = self._gate_hit_and_reasons(nb_paths)
        for row in rows:
            r_idx = self._queue_id_to_row.get(int(row["file_id"]))
            if r_idx is None:
                continue
            self._set_hit_cell(r_idx, row["event"] or "",
                               row["path"], hit_set, gate_reasons)

    def _queue_event_paths(self) -> list[str]:
        """Paths of queue files currently classified as event (events)."""
        return [r["path"] for r in _db.list_queue(self._db_path)
                if r["event"] == "event"]

    def _open_event_summary(self) -> None:
        win = getattr(self, "_event_summary_win", None)
        if win is not None and win.isVisible():
            win.raise_(); win.activateWindow()
            return
        from .event_summary_window import EventSummaryWindow
        results = [{"path": p} for p in self._queue_event_paths()]
        win = EventSummaryWindow(results, db_path=self._db_path, session_info=None)
        win.set_criteria_opener(self._open_criteria)
        self._attach_raw(win)
        self._event_summary_win = win
        self._spawn(win)


    def _open_scatter(self) -> None:
        """Any-vs-any scatter over the queued events."""
        win = getattr(self, "_scatter_win", None)
        if win is not None and win.isVisible():
            win.raise_(); win.activateWindow()
            return
        paths = self._queue_event_paths()
        if not paths:
            QMessageBox.information(
                self, "Plot variables",
                "No events in the queue yet — analyse some curves first.")
            return
        from .scatter_window import ScatterWindow
        win = ScatterWindow(paths, db_path=self._db_path,
                            caption="queued events", session_info=None)
        win.view_file_requested.connect(self._open_raw_viewer)
        self._scatter_win = win
        self._spawn(win)

    def _attach_raw(self, win) -> None:
        """Wire a event-summary window to the singleton viewer for WLC navigation."""
        viewer = getattr(self, "_viewer", None)
        if viewer is not None:
            win.set_raw_window(viewer)

    def _open_criteria(self) -> None:
        dlg = getattr(self, "_criteria_dlg", None)
        if dlg is not None and dlg.isVisible():
            dlg.set_event_paths(self._queue_event_paths())
            dlg.raise_(); dlg.activateWindow()
            return
        from .criteria_dialog import CriteriaDialog
        dlg = CriteriaDialog(
            self._compute_queue_derived_cols(),
            self._queue_event_paths(),
            self._db_path,
        )
        dlg.view_file_requested.connect(self._open_raw_viewer)
        dlg.criteria_changed.connect(self._on_criteria_changed)
        self._criteria_dlg = dlg
        self._spawn(dlg)

    def _on_criteria_changed(self) -> None:
        """A criterion (participation or bound) changed — re-derive everything."""
        self._sync_gate_buttons()
        self._refresh_hit_column()
        self._update_gate_label()
        self._refresh_population_children()

    def _open_non_events(self) -> None:
        win = getattr(self, "_non_events_win", None)
        if win is not None and win.isVisible():
            win.raise_(); win.activateWindow()
            return
        from .class_lineplot_window import ClassLinePlotWindow
        win = ClassLinePlotWindow("non_event", self._db_path, session_info=None)
        win.view_file_requested.connect(self._open_raw_viewer)
        self._non_events_win = win
        self._spawn(win)

    def _refresh_population_children(self) -> None:
        """Refresh any open Events/Blanks windows so they update LIVE as the worker classifies curves."""
        self._prune_children()
        event_paths = None
        for win in self._children:
            try:
                if hasattr(win, "reload_paths"):
                    if event_paths is None:
                        event_paths = self._queue_event_paths()
                    win.reload_paths(event_paths)
                elif hasattr(win, "set_event_paths"):
                    if event_paths is None:
                        event_paths = self._queue_event_paths()
                    win.set_event_paths(event_paths)
                elif hasattr(win, "refresh"):
                    win.refresh()
            except Exception:
                pass


def _fmt_cell(val, key: str = "") -> str:
    """Display text for one table cell, at the column's declared precision."""
    if val is None:
        return ""
    if isinstance(val, float):
        if val != val:
            return ""
        return _quant.format_value(key, val)
    return str(val)
