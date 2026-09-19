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


def test_committing_a_bound_does_not_tear_down_the_card_that_set_it(tmp_path):
    """The crash this guards was a segfault, not an exception.

    Releasing a handle commits, which emits criteria_changed, which the
    dashboard answers by fanning out over its children — and that fan-out
    calls set_event_paths on this window. Rebuilding there destroyed the
    PlotWidget whose mouse-release was still on the stack, and pyqtgraph's
    GraphicsScene then received an event for an object C++ had already freed:

        GraphicsScene.mouseReleaseEvent
        RuntimeError: wrapped C/C++ object of type GraphicsScene has been
        deleted
        Fatal Python error: Segmentation fault

    So the same cohort must never rebuild. Identity is the assertion, because
    "the widget still exists" is precisely what was untrue.
    """
    db_path = str(tmp_path / "c.db")
    db.initialise(db_path)
    paths = _cohort(db_path, tmp_path)
    db.set_threshold("baseline_rms", 0.2, 0.5, "Baseline RMS", "A", db_path)

    win = CriteriaDialog(VARS, paths, db_path)
    try:
        card = win._widgets["baseline_rms"]
        rebuilt = []
        win.criteria_changed.connect(
            lambda: win.set_event_paths(paths))        # what the dashboard does

        card._region.setRegion((0.3, 0.9))
        card._on_region_settled()

        assert win._widgets["baseline_rms"] is card, \
            "the card that set the bound was replaced mid-event"
        assert card._plot.scene() is not None
        assert (db.get_threshold("baseline_rms", "A", db_path)["lower_bound"]
                == 0.3)
    finally:
        win.close()


def test_a_changed_cohort_does_rebuild(tmp_path):
    """The guard is "same cohort", not "never" — a new queue needs new cards."""
    db_path = str(tmp_path / "c.db")
    db.initialise(db_path)
    paths = _cohort(db_path, tmp_path)

    win = CriteriaDialog(VARS, paths, db_path)
    try:
        before = win._widgets["baseline_rms"]
        win.set_event_paths(paths[:4])
        assert win._widgets["baseline_rms"] is not before
    finally:
        win.close()


def test_dragging_an_unbounded_card_starts_a_criterion(tmp_path):
    """A card you cannot grab gives you no way to begin.

    The interval on a variable with no bounds used to be hidden, so the only
    route into a new criterion was to find the end ticks first — the
    two-step this window exists to remove.
    """
    db_path = str(tmp_path / "c.db")
    db.initialise(db_path)
    paths = _cohort(db_path, tmp_path)

    win = CriteriaDialog(VARS, paths, db_path)
    try:
        card = win._widgets["baseline_rms"]
        assert not card._card.in_force
        assert card._region.isVisible(), "nothing to grab"
        assert "not gating" in card._cost.text()

        card._region.setRegion((0.2, 0.5))
        card._on_region_settled()

        row = db.get_threshold("baseline_rms", "A", db_path)
        assert (row["lower_bound"], row["upper_bound"]) == (0.2, 0.5)
        assert card._chk_lo.isChecked() and card._chk_hi.isChecked()
    finally:
        win.close()


def test_a_newly_bounded_card_moves_into_the_in_force_section(tmp_path):
    """A criterion doing the cutting must appear among the ones that cut.

    The segfault fix stopped rebuilding on commit, and re-sectioning was
    collateral damage: a card could be the single biggest influence on the
    hit set while still sitting under "Available". Cards are moved rather
    than rebuilt, so the fix holds and the section is still true.
    """
    from PyQt6.QtWidgets import QApplication

    db_path = str(tmp_path / "c.db")
    db.initialise(db_path)
    paths = _cohort(db_path, tmp_path)

    win = CriteriaDialog(VARS, paths, db_path)
    try:
        card = win._widgets["baseline_rms"]
        assert card.parentWidget() is win._available_host

        card._region.setRegion((0.2, 0.5))
        card._on_region_settled()
        QApplication.processEvents()          # the deferred relayout

        assert card.parentWidget() is win._in_force_host
        assert win._widgets["baseline_rms"] is card, "the card was rebuilt"
        assert card._plot.scene() is not None
    finally:
        win.close()


def test_a_cleared_card_moves_back_out(tmp_path):
    from PyQt6.QtWidgets import QApplication

    db_path = str(tmp_path / "c.db")
    db.initialise(db_path)
    paths = _cohort(db_path, tmp_path)
    db.set_threshold("baseline_rms", 0.2, 0.5, "Baseline RMS", "A", db_path)

    win = CriteriaDialog(VARS, paths, db_path)
    try:
        card = win._widgets["baseline_rms"]
        assert card.parentWidget() is win._in_force_host
        card._clear_btn.click()
        QApplication.processEvents()
        assert card.parentWidget() is win._available_host
    finally:
        win.close()


def test_moving_a_bound_within_a_section_does_not_relayout(tmp_path):
    """Only crossing the boundary moves anything. Dragging an already-active
    criterion must leave the grid alone — a card jumping under the cursor
    mid-adjustment is its own bug."""
    from PyQt6.QtWidgets import QApplication

    db_path = str(tmp_path / "c.db")
    db.initialise(db_path)
    paths = _cohort(db_path, tmp_path)
    db.set_threshold("baseline_rms", 0.2, 0.5, "Baseline RMS", "A", db_path)
    db.set_threshold("invols_rms", 0.2, 0.5, "InvOLS RMS", "A", db_path)

    win = CriteriaDialog(VARS, paths, db_path)
    try:
        card = win._widgets["baseline_rms"]
        grid = win._in_force_host.layout()
        before = [grid.itemAt(i).widget() for i in range(grid.count())]

        card._region.setRegion((0.25, 0.45))
        card._on_region_settled()
        QApplication.processEvents()

        after = [grid.itemAt(i).widget() for i in range(grid.count())]
        assert after == before
    finally:
        win.close()
