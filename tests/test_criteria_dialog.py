# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""The criteria window: the cut is made on the distribution it cuts.

What this guards is the claim the window makes by existing — that dragging an
interval IS setting the criterion. If a drag did not reach the database, the
window would be a picture of a gate rather than the gate, and the user would
be back to wondering whether what they did took effect.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from smfs_catalog import criteria_gate as gate
from smfs_catalog import db
from smfs_catalog.criteria_dialog import CriteriaDialog


_app = QApplication.instance() or QApplication([])

VARS = [("baseline_rms", "Baseline RMS"), ("invols_rms", "InvOLS RMS")]


def _cohort(db_path: str, tmp_path: Path, n: int = 12) -> list[str]:
    """`n` queued events owned by A. Only every third has an invols_rms, so
    one variable has a real missing-value cost and the other does not."""
    paths, ids = [], []
    conn = db.get_connection(db_path)
    with conn:
        for i in range(n):
            path = db.normalize_path(str(tmp_path / f"{i}.ibw"))
            paths.append(path)
            cur = conn.execute(
                "INSERT INTO files (path, filename, first_seen, last_seen,"
                " event, experimentalist) VALUES (?,?,datetime('now'),"
                "datetime('now'),'event','A')", (path, f"{i}.ibw"))
            ids.append(cur.lastrowid)
            conn.execute(
                "INSERT INTO analysis_results (file_id, analysis_type, value,"
                " params_json, code_version, computed_at)"
                " VALUES (?,'baseline_rms',?,'{}','t',datetime('now'))",
                (cur.lastrowid, 0.1 + i * 0.05))
            if i % 3 == 0:
                conn.execute(
                    "INSERT INTO analysis_results (file_id, analysis_type,"
                    " value, params_json, code_version, computed_at)"
                    " VALUES (?,'invols_rms',?,'{}','t',datetime('now'))",
                    (cur.lastrowid, 0.2 + i * 0.01))
    conn.close()
    db.enqueue_files(ids, db_path)
    return paths


def test_dragging_an_interval_sets_the_criterion(tmp_path):
    """No Apply step: letting go of the handle is the write.

    This is the whole of #7. A bound that needed a second confirmation is a
    bound that can be left un-applied, which is the state that made the old
    window lie about what was gating.
    """
    db_path = str(tmp_path / "c.db")
    db.initialise(db_path)
    paths = _cohort(db_path, tmp_path)
    db.set_threshold("baseline_rms", 0.2, 0.5, "Baseline RMS", "A", db_path)

    win = CriteriaDialog(VARS, paths, db_path)
    try:
        card = win._widgets["baseline_rms"]
        card._region.setRegion((0.3, 0.9))
        card._on_region_settled()

        row = db.get_threshold("baseline_rms", "A", db_path)
        assert (row["lower_bound"], row["upper_bound"]) == (0.3, 0.9)
    finally:
        win.close()


def test_the_handle_and_the_number_never_disagree(tmp_path):
    """Both are snapped, so the bound stored is the bound on screen."""
    db_path = str(tmp_path / "c.db")
    db.initialise(db_path)
    paths = _cohort(db_path, tmp_path)
    db.set_threshold("seg_n_segments", 1.0, 2.0, "Segments", "A", db_path)

    win = CriteriaDialog([("seg_n_segments", "ROI Segments")], paths, db_path)
    try:
        card = win._widgets["seg_n_segments"]
        card._region.setRegion((0.8, 2.59))     # as if dragged there
        card._on_region_settled()

        assert card._spin_lo.value() == 1.0
        assert card._spin_hi.value() == 3.0
        row = db.get_threshold("seg_n_segments", "A", db_path)
        assert (row["lower_bound"], row["upper_bound"]) == (1.0, 3.0)
    finally:
        win.close()


def test_a_card_states_what_its_criterion_would_cost(tmp_path):
    """How many curves cannot answer this variable, shown BEFORE it is set.

    Gating on a quantity most of the cohort lacks is a decision, and the
    moment to see it is while choosing, not afterwards when the gate looks
    inexplicably harsh.
    """
    db_path = str(tmp_path / "c.db")
    db.initialise(db_path)
    paths = _cohort(db_path, tmp_path)

    win = CriteriaDialog(VARS, paths, db_path)
    try:
        # 4 of 12 have an invols_rms; the other 8 cannot pass any bound on it.
        assert "8 have no value" in win._widgets["invols_rms"]._cost.text()
        # Every curve has a baseline_rms, so that card has no cost to warn of.
        assert "no value" not in win._widgets["baseline_rms"]._cost.text()
    finally:
        win.close()


def test_bounding_an_unaskable_variable_costs_the_curves_that_lack_it(tmp_path):
    """The footer is the gate's own answer, so the cost actually lands."""
    db_path = str(tmp_path / "c.db")
    db.initialise(db_path)
    paths = _cohort(db_path, tmp_path)

    win = CriteriaDialog(VARS, paths, db_path)
    try:
        assert "every event is a hit" in win._count_lbl.text()
        win._widgets["invols_rms"]._chk_hi.setChecked(True)
        # 8 of the 12 have no invols_rms at all, so they cannot be hits.
        cls = gate.classify(paths, db_path)
        assert len(cls.population(gate.HIT)) <= 4
        assert str(len(cls.population(gate.HIT))) in win._count_lbl.text()
    finally:
        win.close()


def test_clear_removes_the_criterion(tmp_path):
    db_path = str(tmp_path / "c.db")
    db.initialise(db_path)
    paths = _cohort(db_path, tmp_path)
    db.set_threshold("baseline_rms", 0.2, 0.5, "Baseline RMS", "A", db_path)

    win = CriteriaDialog(VARS, paths, db_path)
    try:
        assert gate.gate("A", db_path).criteria
        win._widgets["baseline_rms"]._clear_btn.click()
        assert not gate.gate("A", db_path)
        assert "every event is a hit" in win._count_lbl.text()
    finally:
        win.close()


def test_the_window_uses_the_queues_gate_owner(tmp_path):
    """One queue, one governing profile — the cards and the footer cannot
    describe different people's bounds."""
    db_path = str(tmp_path / "c.db")
    db.initialise(db_path)
    paths = _cohort(db_path, tmp_path)

    win = CriteriaDialog(VARS, paths, db_path)
    try:
        assert win._experimentalist == gate.active_owner(db_path) == "A"
        assert win._context_label.text() == "Criteria owner: A"
    finally:
        win.close()


def test_in_force_variables_come_first(tmp_path):
    """"What is cutting my data" is why the window was opened."""
    db_path = str(tmp_path / "c.db")
    db.initialise(db_path)
    paths = _cohort(db_path, tmp_path)
    db.set_threshold("invols_rms", 0.2, None, "InvOLS RMS", "A", db_path)

    from smfs_catalog import criteria_cards as cards
    built = cards.cohort(VARS, paths, "A", db_path)
    assert built[0].key == "invols_rms" and built[0].in_force
    assert not built[1].in_force
