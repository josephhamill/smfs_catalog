# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""Carry existing catalogs onto the rule that a bound IS the criterion.

A criterion used to need two things to agree: a `criteria_use:<key>` tick in
the owner's profile AND a bound in the thresholds table.  Only the bound
survives, so on an untouched catalog every bound that was stored but never
ticked would quietly start gating, and every hit set in the catalog would
move under the user without a word.

The reconciliation makes each owner's EFFECTIVE gate identical across the
upgrade: a bound that was gating still gates, a bound that was not is
cleared.  What is lost is a parked bound — a number typed and deliberately
left unticked.  Those are archived to `meta` before they go, so "I had 0.95
in there" has an answer; nothing reads the archive back, because a restore
path would rebuild the two-switch system in a shed out back.

THE LEGACY RULE IS WRITTEN OUT HERE, not imported from criteria_gate.  A
migration that asks live code what the old behaviour was stops describing the
old behaviour the moment that code is refactored again.

Inheritance is the subtle half.  db.get_thresholds merges the shared default
owner's rows under each owner's own, but ticks were per-owner with a
whole-profile fallback.  So an owner could inherit a bound they never ticked
and must not start gating on it — which is expressed by giving them their own
row with both bounds NULL, shadowing the inherited one.  Only owners who would
otherwise differ get a row: materialising everybody would quietly end the
lab-default mechanism nobody asked to lose.
"""

from __future__ import annotations

import json

from . import db as _db


_MARKER = "criteria_bounds_reconciled_v1"
_ARCHIVE = "criteria_parked_bounds_v1"
_LEGACY_PREFIX = "criteria_use:"


def _legacy_ticked(owner: str, db_path: str) -> set[str]:
    """The variable keys `owner` had ticked, under the rule as it stood.

    Reproduces criteria_gate.get_criteria as it was: the owner's own profile
    when they have one, otherwise the shared default owner's.
    """
    profile = _db.get_experimentalist_profile(owner, db_path)
    if not profile and owner != _db.DEFAULT_EXPERIMENTALIST:
        profile = _db.get_experimentalist_profile(
            _db.DEFAULT_EXPERIMENTALIST, db_path)
    return {
        key[len(_LEGACY_PREFIX):]
        for key, on in (profile or {}).items()
        if key.startswith(_LEGACY_PREFIX) and on
    }


def _legacy_effective(owner: str, db_path: str) -> dict[str, tuple]:
    """`owner`'s gate under the old rule: ticked AND bounded."""
    ticked = _legacy_ticked(owner, db_path)
    out = {}
    for row in _db.get_thresholds(owner, db_path):
        key = row["analysis_type"]
        bounds = (row["lower_bound"], row["upper_bound"])
        if key in ticked and bounds != (None, None):
            out[key] = bounds
    return out


def _owners(conn) -> set[str]:
    """Every experimentalist whose gate could ever govern a queue.

    `files` is in here because the governing profile is chosen from the
    queue's first file, so an owner who has neither thresholds nor a profile
    can still become the one whose criteria apply.
    """
    found = {_db.DEFAULT_EXPERIMENTALIST}
    for table in ("thresholds", "experimentalist_profiles", "files"):
        try:
            rows = conn.execute(
                f"SELECT DISTINCT experimentalist FROM {table}").fetchall()
        except Exception:
            continue
        found.update(r[0] for r in rows if r[0])
    return found


def _upsert(conn, owner: str, key: str, bounds: tuple) -> None:
    """Write one owner's bounds on the caller's connection.

    Not db.set_threshold: that opens its own connection, and this runs inside
    a held write transaction, where a second writer finds the database locked.
    """
    conn.execute("""
        INSERT INTO thresholds
            (experimentalist, analysis_type, lower_bound, upper_bound, label, created_at)
        VALUES (?, ?, ?, ?, '', ?)
        ON CONFLICT(experimentalist, analysis_type) DO UPDATE SET
            lower_bound = excluded.lower_bound,
            upper_bound = excluded.upper_bound
    """, (owner, key, bounds[0], bounds[1], _db._now()))


def _needs_run(conn) -> bool:
    """Whether this catalog has not been reconciled yet.

    A `meta` marker, not a condition read off the data: after the profile
    keys are stripped every owner's legacy gate computes as empty, so a
    second pass guarded on the data would clear every bound in the catalog.
    """
    row = conn.execute(
        "SELECT value FROM meta WHERE key = ?", (_MARKER,)).fetchone()
    return row is None


def reconcile(db_path: str = _db.DEFAULT_DB_PATH) -> None:
    """Make bounds-alone reproduce each owner's old effective gate. Idempotent."""
    conn = _db.get_connection(db_path)
    try:
        if not _needs_run(conn):
            return
        owners = _owners(conn)
        effective = {o: _legacy_effective(o, db_path) for o in owners}
        default = _db.DEFAULT_EXPERIMENTALIST
        parked: list[dict] = []

        with conn:
            # The shared rows first: everything else is expressed as a
            # difference from what they will inherit.
            for row in conn.execute(
                "SELECT analysis_type, lower_bound, upper_bound FROM thresholds "
                "WHERE experimentalist = ?", (default,)
            ).fetchall():
                key, lo, hi = row[0], row[1], row[2]
                if (lo, hi) == (None, None) or key in effective[default]:
                    continue
                parked.append({"experimentalist": default, "analysis_type": key,
                               "lower_bound": lo, "upper_bound": hi})
                conn.execute(
                    "UPDATE thresholds SET lower_bound = NULL, upper_bound = NULL "
                    "WHERE experimentalist = ? AND analysis_type = ?", (default, key))

            inherited = effective[default]
            for owner in sorted(owners - {default}):
                own_rows = {
                    r[0]: (r[1], r[2]) for r in conn.execute(
                        "SELECT analysis_type, lower_bound, upper_bound FROM "
                        "thresholds WHERE experimentalist = ?", (owner,)).fetchall()
                }
                wanted = effective[owner]
                for key in set(wanted) | set(inherited) | set(own_rows):
                    target = wanted.get(key, (None, None))
                    if target == inherited.get(key, (None, None)) \
                            and key not in own_rows:
                        # Inheritance already says the right thing.
                        continue
                    if key in own_rows and own_rows[key] not in ((None, None), target):
                        parked.append({
                            "experimentalist": owner, "analysis_type": key,
                            "lower_bound": own_rows[key][0],
                            "upper_bound": own_rows[key][1]})
                    _upsert(conn, owner, key, target)

            if parked:
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                    (_ARCHIVE, json.dumps(parked)))
            _strip_ticks(conn, owners)
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (_MARKER, _db._now()))
    finally:
        conn.close()


_WHOLE_MARKER = "criteria_integer_bounds_whole_v1"


def whole_integer_bounds(db_path: str = _db.DEFAULT_DB_PATH) -> None:
    """Make a count's stored bound a whole number, without moving anybody.

    A bound seeded from a percentile of an integer-valued distribution stored
    2.590865 for a segment COUNT and displayed it as 3 — so six people read a
    criterion that admitted 3 segments while it rejected them.  db.set_threshold
    quantizes now, but a bound nobody re-applies stays as it was.

    CEIL the lower and FLOOR the upper, NOT quantize's round-to-nearest. Over
    integers `>= 0.798` admits exactly what `>= 1` admits and `<= 2.591`
    exactly what `<= 2` admits, so every verdict in the catalog is unchanged
    and only the number on screen becomes true. Rounding 2.590865 to 3 would
    instead start admitting the 3-segment curves it had been rejecting, which
    is a change to results dressed up as a display fix.
    """
    import math

    from . import quantities as _quant

    conn = _db.get_connection(db_path)
    try:
        if conn.execute("SELECT 1 FROM meta WHERE key = ?",
                        (_WHOLE_MARKER,)).fetchone():
            return
        rows = conn.execute(
            "SELECT experimentalist, analysis_type, lower_bound, upper_bound"
            " FROM thresholds").fetchall()
        with conn:
            for r in rows:
                if not _quant.get(r["analysis_type"]).integer:
                    continue
                lo, hi = r["lower_bound"], r["upper_bound"]
                new_lo = None if lo is None else float(math.ceil(lo))
                new_hi = None if hi is None else float(math.floor(hi))
                if (new_lo, new_hi) == (lo, hi):
                    continue
                conn.execute(
                    "UPDATE thresholds SET lower_bound = ?, upper_bound = ?"
                    " WHERE experimentalist = ? AND analysis_type = ?",
                    (new_lo, new_hi, r["experimentalist"], r["analysis_type"]))
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (_WHOLE_MARKER, _db._now()))
    finally:
        conn.close()


def _strip_ticks(conn, owners: set[str]) -> None:
    """Remove every `criteria_use:` key — nothing reads them any more."""
    for owner in owners:
        row = conn.execute(
            "SELECT params_json FROM experimentalist_profiles WHERE "
            "experimentalist = ?", (owner,)).fetchone()
        if row is None or not row[0]:
            continue
        params = json.loads(row[0])
        kept = {k: v for k, v in params.items()
                if not k.startswith(_LEGACY_PREFIX)}
        if len(kept) != len(params):
            conn.execute(
                "UPDATE experimentalist_profiles SET params_json = ? WHERE "
                "experimentalist = ?", (json.dumps(kept), owner))
