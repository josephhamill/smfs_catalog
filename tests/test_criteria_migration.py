# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""Carrying an existing catalog onto "a bound is the criterion".

The one thing the migration owes: every experimentalist's EFFECTIVE gate is
the same before and after.  A bound that was gating still gates, a bound that
was stored but never ticked does not start gating, and an owner who inherited
the shared default's bounds keeps inheriting exactly what they were getting.

The five situations below are the ones that interact, and the inheritance
ones are why this is not a one-line UPDATE: ticks were per-owner with a
whole-profile fallback, while bounds merge per-row.
"""

from __future__ import annotations

import json

from smfs_catalog import criteria_gate as _gate
from smfs_catalog import criteria_migration as _mig
from smfs_catalog import db as _db


DEFAULT = _db.DEFAULT_EXPERIMENTALIST


def _tick(owner: str, key: str, on: bool, db_path: str) -> None:
    """Write a legacy participation tick, the way the old code did."""
    _db.merge_experimentalist_profile(
        owner, {"criteria_use:" + key: 1.0 if on else 0.0}, db_path)


def _legacy_gate(owner: str, db_path: str) -> dict:
    """What `owner`'s gate WAS: ticked and bounded, ticks falling back whole.

    Written out rather than imported, so it keeps describing the old rule
    after the live code has moved on.
    """
    profile = _db.get_experimentalist_profile(owner, db_path)
    if not profile and owner != DEFAULT:
        profile = _db.get_experimentalist_profile(DEFAULT, db_path)
    ticked = {k[len("criteria_use:"):]
              for k, on in (profile or {}).items()
              if k.startswith("criteria_use:") and on}
    return {
        row["analysis_type"]: (row["lower_bound"], row["upper_bound"])
        for row in _db.get_thresholds(owner, db_path)
        if row["analysis_type"] in ticked
        and (row["lower_bound"], row["upper_bound"]) != (None, None)
    }


def _current_gate(owner: str, db_path: str) -> dict:
    """What `owner`'s gate IS: every bound that is set."""
    return {c.key: (c.lower, c.upper)
            for c in _gate.gate(owner, db_path).criteria}


def _unstamp(db_path: str) -> None:
    """Put the catalog back to never-reconciled, as an old one would be."""
    conn = _db.get_connection(db_path)
    with conn:
        conn.execute("DELETE FROM meta WHERE key LIKE 'criteria_%'")
    conn.close()


def _old_catalog(tmp_path) -> str:
    """A catalog holding all five owner situations, pre-migration."""
    db_path = str(tmp_path / "old.db")
    _db.initialise(db_path)

    # (a) shared default: bounded AND ticked — everyone inheriting this gated.
    _db.set_threshold("shared_on", 0.5, None, "Shared on", DEFAULT, db_path)
    _tick(DEFAULT, "shared_on", True, db_path)
    # (b) shared default: bounded but NEVER ticked — inert, must stay inert.
    _db.set_threshold("shared_off", 2.0, None, "Shared off", DEFAULT, db_path)

    # (c) an owner with their own bounds and their own ticks.
    _db.set_threshold("own", 1.0, 9.0, "Own", "amy", db_path)
    _tick("amy", "own", True, db_path)
    # ...who also has a bound they parked without ticking.
    _db.set_threshold("parked", 3.0, None, "Parked", "amy", db_path)
    # ...and who never ticked the shared one, so it did not gate for her.
    _tick("amy", "shared_on", False, db_path)

    # (d) an owner with a profile but no ticks at all: nothing gated for them,
    #     because a profile that exists does not fall back to the default's.
    _db.merge_experimentalist_profile("ben", {"some_param": 1.0}, db_path)

    # (e) an owner with no profile and no bounds of their own: falls back
    #     wholesale to the default's ticks, so the shared criterion DID gate
    #     for them. Carl reaches the migration through `files` alone — the
    #     governing profile is chosen from the queue's first file, so an owner
    #     with nothing else stored can still be the one whose gate applies,
    #     and leaving him out of the owner sweep would leave him unreconciled.
    conn = _db.get_connection(db_path)
    with conn:
        conn.execute(
            "INSERT INTO files (path, filename, first_seen, last_seen, event,"
            " experimentalist) VALUES (?, ?, datetime('now'), datetime('now'),"
            " 'event', ?)",
            (_db.normalize_path(str(tmp_path / "carl.ibw")), "carl.ibw", "carl"))
    conn.close()

    _unstamp(db_path)
    return db_path


OWNERS = (DEFAULT, "amy", "ben", "carl")


def test_every_owners_effective_gate_survives_the_migration(tmp_path):
    db_path = _old_catalog(tmp_path)
    before = {o: _legacy_gate(o, db_path) for o in OWNERS}

    # The situations are only worth testing if they actually differ.
    assert before[DEFAULT] == {"shared_on": (0.5, None)}
    assert before["amy"] == {"own": (1.0, 9.0)}
    assert before["ben"] == {}
    assert before["carl"] == {"shared_on": (0.5, None)}

    _mig.reconcile(db_path)

    for owner in OWNERS:
        assert _current_gate(owner, db_path) == before[owner], owner


def test_reconciling_twice_changes_nothing(tmp_path):
    """The marker, not the data, is what stops a second pass.

    Guarded on the data, a re-run would compute an empty legacy gate for
    everyone — the ticks having been stripped — and clear every bound in the
    catalog.
    """
    db_path = _old_catalog(tmp_path)
    _mig.reconcile(db_path)
    after_first = {o: _current_gate(o, db_path) for o in OWNERS}

    _mig.reconcile(db_path)
    _db.initialise(db_path)
    _db.initialise(db_path)

    assert {o: _current_gate(o, db_path) for o in OWNERS} == after_first
    assert after_first[DEFAULT] == {"shared_on": (0.5, None)}


def test_an_owner_who_agrees_with_the_default_gets_no_rows_of_their_own(tmp_path):
    """Materialising everybody would quietly end the lab-default mechanism.

    carl inherits exactly what he was already getting, so the migration must
    leave him inheriting it rather than freezing a copy under his name.
    """
    db_path = _old_catalog(tmp_path)
    _mig.reconcile(db_path)

    conn = _db.get_connection(db_path)
    carl_rows = conn.execute(
        "SELECT analysis_type FROM thresholds WHERE experimentalist = ?",
        ("carl",)).fetchall()
    conn.close()
    assert carl_rows == []


def test_an_inherited_bound_that_did_not_gate_is_shadowed(tmp_path):
    """ben gated on nothing, and must go on gating on nothing.

    He has no bounds of his own, so under the new rule he would inherit the
    default's `shared_on` and start gating on a criterion he never set. An
    own row with both bounds NULL is what stops that.
    """
    db_path = _old_catalog(tmp_path)
    _mig.reconcile(db_path)

    assert _current_gate("ben", db_path) == {}
    row = _db.get_threshold("shared_on", "ben", db_path)
    assert row is not None and row["experimentalist"] == "ben"
    assert (row["lower_bound"], row["upper_bound"]) == (None, None)


def test_parked_bounds_are_archived_before_they_are_cleared(tmp_path):
    """The numbers go, so they are written down on the way out.

    Nothing reads this back — a restore path would rebuild the two-switch
    system — but "I had 3.0 in there" deserves an answer.
    """
    db_path = _old_catalog(tmp_path)
    _mig.reconcile(db_path)

    conn = _db.get_connection(db_path)
    row = conn.execute(
        "SELECT value FROM meta WHERE key = 'criteria_parked_bounds_v1'"
    ).fetchone()
    conn.close()
    archived = {(r["experimentalist"], r["analysis_type"]): r["lower_bound"]
                for r in json.loads(row[0])}
    assert archived[("amy", "parked")] == 3.0
    assert archived[(DEFAULT, "shared_off")] == 2.0


def test_the_participation_ticks_are_gone(tmp_path):
    """Nothing reads them, so leaving them would be a second source of truth."""
    db_path = _old_catalog(tmp_path)
    _mig.reconcile(db_path)

    for owner in OWNERS:
        profile = _db.get_experimentalist_profile(owner, db_path) or {}
        assert not any(k.startswith("criteria_use:") for k in profile), owner
    # Non-criteria settings in the same profile are untouched.
    assert _db.get_experimentalist_profile("ben", db_path)["some_param"] == 1.0


def test_a_fresh_catalog_is_stamped_and_unchanged(tmp_path):
    """New catalogs have nothing to reconcile and must not be touched."""
    db_path = str(tmp_path / "fresh.db")
    _db.initialise(db_path)
    _db.set_threshold("metric", 0.0, 1.0, "Metric", DEFAULT, db_path)

    _db.initialise(db_path)
    assert _current_gate(DEFAULT, db_path) == {"metric": (0.0, 1.0)}


def test_a_counts_bound_becomes_whole_without_moving_anybody(tmp_path):
    """The display said 3 segments passed; the gate rejected them.

    Over integers `<= 2.590865` and `<= 2` admit the same curves, so the
    verdict cannot move — which is the whole point of ceil/floor here rather
    than quantize's round-to-nearest.
    """
    db_path = str(tmp_path / "whole.db")
    _db.initialise(db_path)
    conn = _db.get_connection(db_path)
    with conn:                       # raw, so set_threshold's quantize is bypassed
        conn.execute(
            "INSERT INTO thresholds (experimentalist, analysis_type, lower_bound,"
            " upper_bound, label, created_at) VALUES (?,?,?,?,'',datetime('now'))",
            (DEFAULT, "seg_n_segments", 0.797854, 2.590865))
        conn.execute("DELETE FROM meta WHERE key = 'criteria_integer_bounds_whole_v1'")
    conn.close()

    _mig.whole_integer_bounds(db_path)

    row = _db.get_threshold("seg_n_segments", DEFAULT, db_path)
    assert (row["lower_bound"], row["upper_bound"]) == (1.0, 2.0)
    # Rounding to nearest would have given 3.0 and started admitting the
    # 3-segment curves the old bound rejected.
    assert row["upper_bound"] != 3.0


def test_a_non_integer_quantity_is_left_alone(tmp_path):
    """Only counts. Rounding a continuous bound to its displayed decimals
    WOULD move curves across it, so that is not a display fix."""
    db_path = str(tmp_path / "whole.db")
    _db.initialise(db_path)
    conn = _db.get_connection(db_path)
    with conn:
        conn.execute(
            "INSERT INTO thresholds (experimentalist, analysis_type, lower_bound,"
            " upper_bound, label, created_at) VALUES (?,?,?,?,'',datetime('now'))",
            (DEFAULT, "baseline_rms", 0.001, 0.416959942408))
        conn.execute("DELETE FROM meta WHERE key = 'criteria_integer_bounds_whole_v1'")
    conn.close()

    _mig.whole_integer_bounds(db_path)

    row = _db.get_threshold("baseline_rms", DEFAULT, db_path)
    assert row["upper_bound"] == 0.416959942408


def test_set_threshold_quantizes_a_count_on_the_way_in(tmp_path):
    """The write path is what stops this recurring; the migration only
    catches bounds nobody re-applies."""
    db_path = str(tmp_path / "whole.db")
    _db.initialise(db_path)
    _db.set_threshold("seg_n_segments", 0.8, 2.6, "Segments", DEFAULT, db_path)

    row = _db.get_threshold("seg_n_segments", DEFAULT, db_path)
    assert (row["lower_bound"], row["upper_bound"]) == (1.0, 3.0)
