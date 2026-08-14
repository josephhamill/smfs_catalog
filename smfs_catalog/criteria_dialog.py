# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/criteria_dialog.py
#
# CriteriaDialog — the gate's control surface.
#
# Class-scoped to EVENTS: a flat, data-driven list of analysis variables, each
# with ONE persistent participation checkbox.  Checking a variable makes it
# gate the hit; its bounds come from the thresholds table (set in the
# variable window — the how/whether split), shown read-only here with an
# "Edit bounds…" shortcut.  A live "N of M events survive" readout recomputes on
# every toggle so the effect of the gate is immediately visible.
#
# This window decides WHETHER a variable participates; it never edits bounds.
# Hits/non-hit membership is derived by criteria_gate.evaluate — never stored.

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QScrollArea,
    QVBoxLayout, QWidget,
)

from . import criteria_gate as _gate
from . import db as _db
from . import quantities as _quant
from . import style
from . import variables as _vars
from .qt_utils import fit_on_screen


def _bounds_text(row, key: str = "") -> str:
    """Human-readable bounds for one variable, from its thresholds row.

    Digits come from quantities.py so this line and the editing spin box agree.
    Units are included because the text is read while deciding a bound rather
    than beside an axis label.
    """
    if row is None:
        return "no bounds — passes all"
    lo = row["lower_bound"]
    hi = row["upper_bound"]
    if lo is None and hi is None:
        return "no bounds — passes all"
    f = lambda v: _quant.format_value(key, v, with_unit=True)
    if lo is not None and hi is not None:
        return f"{f(lo)} ≤ x ≤ {f(hi)}"
    if lo is not None:
        return f"x ≥ {f(lo)}"
    return f"x ≤ {f(hi)}"


_TICKING_MEANS = (
    "Ticking selects this variable for the gate. It becomes active once at "
    "least one bound is set; without a bound it does not constrain the hit.\n\n"
    "When active, the variable is REQUIRED: a curve with no finite value for "
    "it becomes a non-hit even if everything else passes. That matters most "
    "for variables only some curves have — the reload and rupture-separation "
    "distances exist only where an ROI has two or more ruptures."
)


def _criterion_tooltip(key: str) -> str:
    """
    What the variable means, then what checking it does.

    The first half is variables.describe() — the same sentence the queue
    header and the scatter axes show, so there is one place to edit it.  The
    second half is about the GATE rather than the variable, which is why it
    is written here and not in the register.
    """
    desc = _vars.describe(key)
    return f"{desc}\n\n———\n\n{_TICKING_MEANS}" if desc else _TICKING_MEANS


class CriteriaDialog(QMainWindow):
    """Per-variable participation checkboxes + live hit survival count."""

    # Re-emitted from a spawned variable window so the dashboard can route a
    # double-clicked file to its singleton worker viewer.
    view_file_requested = pyqtSignal(str)
    # Emitted whenever participation changes, so the dashboard updates the
    # grey-out state of the result buttons and refreshes open hit/non-hit
    # windows.
    criteria_changed = pyqtSignal()

    def __init__(
        self,
        variables: list[tuple[str, str]],
        event_paths: list[str],
        db_path:   str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.Window)
        fit_on_screen(self, 560, 520)
        self._db_path   = db_path
        self._variables = variables
        self._event_paths = list(event_paths)
        self._var_wins: list = []   # spawned variable windows, kept from GC
        self._rows: list[tuple[str, QCheckBox, QLabel]] = []
        # The queue has one active profile owner, selected by its first row.
        # Use the gate's resolver so the controls and live verdict can never
        # describe different profiles when the queue contains several owners.
        self._experimentalist = _gate.active_owner(db_path)
        self._update_title()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        intro = QLabel(
            "Tick the variables that should gate the hit.  A ticked variable "
            "filters by its bounds (set via “Edit bounds…”); an unticked one "
            "doesn’t constrain.  Events that fail any active, bounded variable "
            "— or have no finite value for it — become non-hits."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(style.qss_text())
        root.addWidget(intro)

        self._context_label = QLabel("")
        self._context_label.setStyleSheet(style.qss_text(style.UI_TEXT))
        root.addWidget(self._context_label)
        self._update_title()

        # Scrollable variable list (auto-grows with the data-driven set).
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        self._list_l = QVBoxLayout(inner)
        self._list_l.setContentsMargins(0, 0, 0, 0)
        self._list_l.setSpacing(2)
        scroll.setWidget(inner)
        root.addWidget(scroll, 1)

        checked = _gate.get_criteria(self._experimentalist, db_path)
        for key, label in variables:
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(2, 1, 2, 1)
            chk = QCheckBox(label)
            chk.setChecked(key in checked)
            chk.toggled.connect(lambda on, k=key: self._on_toggle(k, on))
            # What the variable IS (the one register, same text the queue
            # header and the scatter axes show) followed by what ticking it
            # DOES.  This is the highest-stakes place a variable is named in
            # the app: a box ticked without understanding silently changes
            # which curves count as hits for this whole cohort.
            chk.setToolTip(_criterion_tooltip(key))
            bounds_lbl = QLabel("")
            bounds_lbl.setStyleSheet(style.qss_text(style.UI_FAINT))
            edit_btn = QPushButton("Edit bounds…")
            edit_btn.setToolTip(
                "Set the low and high bounds for this variable.\n\n"
                "Bounds belong to ONE experimentalist, named above the list, so "
                "editing them changes the hit set for their data and nobody "
                "else's."
            )
            edit_btn.clicked.connect(lambda _=False, k=key, lb=label: self._edit_bounds(k, lb))
            row_l.addWidget(chk, 1)
            row_l.addWidget(bounds_lbl)
            row_l.addWidget(edit_btn)
            self._list_l.addWidget(row_w)
            self._rows.append((key, chk, bounds_lbl))
        self._list_l.addStretch(1)

        self._count_lbl = QLabel("")
        f = self._count_lbl.font(); f.setBold(True); self._count_lbl.setFont(f)
        root.addWidget(self._count_lbl)

        self.refresh()

    # ── Live state ───────────────────────────────────────────────────────────

    def _update_title(self) -> None:
        """Keep ownership explicit in the dialog body, where it cannot truncate."""
        if self._experimentalist is not None:
            who = self._experimentalist
        else:
            who = _db.DEFAULT_EXPERIMENTALIST
        self.setWindowTitle("SMFS — event criteria")
        if hasattr(self, "_context_label"):
            self._context_label.setText(f"Criteria owner: {who}")

    def set_event_paths(self, event_paths: list[str]) -> None:
        """Update the input cohort (dashboard calls this on queue changes)."""
        self._event_paths = list(event_paths)
        self._experimentalist = _gate.active_owner(self._db_path)
        self._update_title()
        self.refresh()

    def refresh(self) -> None:
        """
        Resync checkboxes + bounds text to the CURRENT experimentalist, then
        recompute the live hit survival count.  The checkbox resync matters
        because this dialog is a reused singleton (dashboard_window's
        _open_criteria calls set_event_paths on an already-open instance) —
        without it, reopening on a different owner's queue would keep
        showing the PREVIOUS owner's checked state, the same staleness bug
        this whole per-experimentalist change exists to close.
        """
        checked = _gate.get_criteria(self._experimentalist, self._db_path)
        for key, chk, bounds_lbl in self._rows:
            chk.blockSignals(True)
            chk.setChecked(key in checked)
            chk.blockSignals(False)
            bounds_lbl.setText(_bounds_text(
                _db.get_threshold(key, self._experimentalist, self._db_path), key))
        hits, non_hits = _gate.evaluate(self._event_paths, self._db_path)
        active = _gate.get_active_criteria(
            self._experimentalist, self._db_path)
        total = len(self._event_paths)
        if not active:
            self._count_lbl.setText(
                f"No active criteria — hit undefined ({total:,} events)."
            )
        else:
            self._count_lbl.setText(
                f"{len(hits):,} of {total:,} events are hits  ·  "
                f"{len(non_hits):,} non-hits"
            )

    # ── Interaction ──────────────────────────────────────────────────────────

    def _on_toggle(self, key: str, on: bool) -> None:
        _gate.set_criterion(key, on, self._experimentalist, self._db_path)
        self.refresh()
        self.criteria_changed.emit()

    def _edit_bounds(self, key: str, label: str) -> None:
        from .variable_window import VariableStatsWindow
        win = VariableStatsWindow(
            key, label, self._event_paths, self._db_path, session_info=None,
            experimentalist=self._experimentalist)
        win.thresholds_changed.connect(self.refresh)
        win.thresholds_changed.connect(self.criteria_changed)
        win.view_file_requested.connect(self.view_file_requested)
        self._var_wins.append(win)
        win.show()
        win.raise_()
        win.activateWindow()
