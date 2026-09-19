# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""The dashboard shows what defines a hit, and offers the control that sets it.

The Hit column reports the gate's verdict. Until now the only way to reach
the gate was inside Explore Events, so the dashboard stated a consequence and
offered no route to its cause — and nothing anywhere named the criteria in
force, which is what made a changed gate something you discovered rather than
something you were told.

Both halves matter for the same reason: a criterion that cuts a cohort without
announcing itself turns every count on screen into an unverifiable claim.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QPushButton

from smfs_catalog import criteria_gate as _gate
from smfs_catalog import db as _db
from smfs_catalog.dashboard_window import DashboardWindow


_app = QApplication.instance() or QApplication([])


def _bound(db_path: str, lo, hi) -> None:
    """Give the default owner a bound, which is what makes it a criterion.

    The default owner, not a named one: opening the dashboard clears the
    analysis queue, so there is no first queue row for the gate to resolve an
    owner from and it falls back to the shared default. Whose bounds these are
    is criteria_gate's question, tested there.
    """
    _db.set_threshold("baseline_rms", lo, hi, "Baseline RMS",
                      _db.DEFAULT_EXPERIMENTALIST, db_path)


def test_the_dashboard_offers_the_criteria_control(tmp_path):
    db_path = str(tmp_path / "dash.db")
    _db.initialise(db_path)
    win = DashboardWindow(db_path)
    try:
        buttons = [b.text() for b in win.findChildren(QPushButton)]
        assert "Hit criteria…" in buttons
        # The same singleton the Explore Events button raises, not a copy.
        assert callable(win._open_criteria)
    finally:
        win.close()


def test_the_dashboard_names_the_criteria_in_force(tmp_path):
    db_path = str(tmp_path / "dash.db")
    _db.initialise(db_path)
    _bound(db_path, 0.001, 0.4)

    win = DashboardWindow(db_path)
    try:
        win._update_gate_label()
        line = win._gate_lbl.text()
        assert "Hit criteria:" in line
        assert "Baseline RMS" in line
        # The bounds are on hover: a real gate runs to a dozen criteria and
        # spelling every bound out would turn the line into a paragraph.
        assert "0.4" not in line
        assert "0.4" in win._gate_lbl.toolTip()
    finally:
        win.close()


def test_the_line_says_so_when_nothing_gates(tmp_path):
    """No criteria means every event is a hit — the most surprising state the
    gate has, and the one most worth stating outright."""
    db_path = str(tmp_path / "dash.db")
    _db.initialise(db_path)

    win = DashboardWindow(db_path)
    try:
        win._update_gate_label()
        assert "every event is a hit" in win._gate_lbl.text()
        assert win._gate_lbl.toolTip() == ""
    finally:
        win.close()


def test_the_line_follows_the_gate(tmp_path):
    """Clearing a criterion has to leave the line describing what is left.

    A stale line is worse than no line: it is a claim about the population
    that the Hit column beside it no longer agrees with.
    """
    db_path = str(tmp_path / "dash.db")
    _db.initialise(db_path)
    _bound(db_path, 0.001, 0.4)

    win = DashboardWindow(db_path)
    try:
        win._update_gate_label()
        assert "Baseline RMS" in win._gate_lbl.text()

        _bound(db_path, None, None)
        win._on_criteria_changed()
        assert "Baseline RMS" not in win._gate_lbl.text()
        assert "every event is a hit" in win._gate_lbl.text()
        assert not _gate.gate(None, db_path)
    finally:
        win.close()
