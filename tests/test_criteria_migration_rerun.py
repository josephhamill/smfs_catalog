"""reconcile's retry and failure behaviour (PR b5 / D8).

The guarantee: the done-marker is written in the same transaction as every
change, so a reconcile that fails leaves nothing behind and the next launch
runs the whole migration again; a reconcile that has finished is never run
twice.  Only a missing table is tolerated while finding owners; a corrupt
profile is logged and skipped, not fatal.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from smfs_catalog import criteria_migration as _mig
from smfs_catalog import crashlog
from smfs_catalog import db as _db

from conftest import add_files

DEFAULT = _db.DEFAULT_EXPERIMENTALIST


def _snapshot(db_path: str) -> dict:
    """Every row of every table, the done-marker included."""
    conn = _db.get_connection(db_path)
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%'")]
        return {t: sorted((tuple(r) for r in conn.execute(f"SELECT * FROM {t}")),
                          key=repr)
                for t in tables}
    finally:
        conn.close()


def _marker(db_path: str):
    conn = _db.get_connection(db_path)
    try:
        return conn.execute("SELECT value FROM meta WHERE key = ?",
                            (_mig._MARKER,)).fetchone()
    finally:
        conn.close()


def _bounds(db_path: str, owner: str, key: str):
    for r in _db.get_thresholds(owner, db_path):
        if r["analysis_type"] == key and r["experimentalist"] == owner:
            return (r["lower_bound"], r["upper_bound"])
    return None


def _alice_catalog(catalog) -> str:
    """Pre-migration catalog: alice owns files, ticked one bound, parked another."""
    add_files(catalog, 2)
    conn = _db.get_connection(catalog)
    with conn:
        conn.execute("UPDATE files SET experimentalist = 'alice'")
    conn.close()
    _db.set_threshold("gating", 1.0, 9.0, "Gating", "alice", catalog)
    _db.set_threshold("parked", 3.0, None, "Parked", "alice", catalog)
    _db.merge_experimentalist_profile(
        "alice", {"criteria_use:gating": 1.0, "other_param": 2.0}, catalog)
    conn = _db.get_connection(catalog)
    with conn:
        conn.execute("DELETE FROM meta WHERE key LIKE 'criteria_%'")
    conn.close()
    return catalog


class _FailingRead:
    """A connection that raises `error` on the owner-list read of `table`.

    Wraps the real connection; everything else passes straight through.
    """

    def __init__(self, conn, table, error, armed):
        self._conn, self._table, self._error, self._armed = conn, table, error, armed

    def execute(self, sql, *args):
        if self._armed["on"] and f"FROM {self._table}" in sql \
                and "DISTINCT experimentalist" in sql:
            raise self._error
        return self._conn.execute(sql, *args)

    def __enter__(self):
        self._conn.__enter__()
        return self

    def __exit__(self, *exc):
        return self._conn.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def _fail_owner_read(monkeypatch, table, error) -> dict:
    """Patch db.get_connection so `_owners`' SELECT DISTINCT experimentalist
    FROM <table> raises `error`.  Returns a switch: armed["on"] = False clears it.
    """
    armed = {"on": True}
    real = _db.get_connection
    monkeypatch.setattr(
        _db, "get_connection",
        lambda *a, **k: _FailingRead(real(*a, **k), table, error, armed))
    return armed


def test_reconcile_with_marker_present_changes_nothing(catalog):
    """With the marker present a second reconcile is a no-op on every table.

    A rerun WITHOUT the marker is unsafe, which is why the marker gates it:
    the first pass strips the `criteria_use:` ticks, so a second pass computes
    an empty legacy gate and would wipe live bounds.  See `_needs_run`'s
    docstring.
    """
    db_path = _alice_catalog(catalog)
    _mig.reconcile(db_path)
    first = _snapshot(db_path)
    assert _marker(db_path) is not None

    _mig.reconcile(db_path)

    assert _snapshot(db_path) == first
    assert _bounds(db_path, "alice", "gating") == (1.0, 9.0)


def test_missing_table_in_owner_search_is_tolerated(catalog, monkeypatch):
    db_path = _alice_catalog(catalog)
    _fail_owner_read(monkeypatch, "experimentalist_profiles",
                     sqlite3.OperationalError("no such table: experimentalist_profiles"))

    _mig.reconcile(db_path)

    assert _marker(db_path) is not None
    assert _bounds(db_path, "alice", "gating") == (1.0, 9.0)
    assert _bounds(db_path, "alice", "parked") == (None, None)


@pytest.mark.parametrize("error", [
    sqlite3.OperationalError("database is locked"),
    RuntimeError("disk exploded"),
], ids=["locked", "other"])
def test_other_owner_search_errors_propagate_and_leave_catalog_untouched(
        catalog, monkeypatch, error):
    db_path = _alice_catalog(catalog)
    before = _snapshot(db_path)
    armed = _fail_owner_read(monkeypatch, "thresholds", error)

    with pytest.raises(type(error)):
        _mig.reconcile(db_path)

    assert _snapshot(db_path) == before
    assert _marker(db_path) is None

    armed["on"] = False                      # next launch, error gone
    _mig.reconcile(db_path)
    assert _marker(db_path) is not None
    assert _bounds(db_path, "alice", "gating") == (1.0, 9.0)
    assert _bounds(db_path, "alice", "parked") == (None, None)


def _corrupt_bob(db_path: str) -> None:
    """Give bob a bound and a profile that is not valid JSON."""
    _db.set_threshold("gating", 2.0, 4.0, "Gating", "bob", db_path)
    conn = _db.get_connection(db_path)
    with conn:
        conn.execute(
            "INSERT INTO experimentalist_profiles "
            "(experimentalist, params_json, updated_at) VALUES ('bob', '{bad', 'now')")
    conn.close()


def test_corrupt_profile_is_logged_and_migration_still_completes(
        catalog, monkeypatch):
    db_path = _alice_catalog(catalog)
    _corrupt_bob(db_path)
    logged = []
    monkeypatch.setattr(crashlog, "log_handled",
                        lambda ctx, exc: logged.append((ctx, exc)))

    _mig.reconcile(db_path)

    assert _marker(db_path) is not None
    contexts = [ctx for ctx, _ in logged]
    assert "Strip legacy criteria ticks" in contexts
    assert _raw_profile(db_path, "bob") == "{bad"
    assert _bounds(db_path, "alice", "gating") == (1.0, 9.0)
    assert _bounds(db_path, "alice", "parked") == (None, None)
    assert "criteria_use:gating" not in json.loads(_raw_profile(db_path, "alice"))


def _raw_profile(db_path: str, owner: str) -> str:
    conn = _db.get_connection(db_path)
    try:
        return conn.execute(
            "SELECT params_json FROM experimentalist_profiles "
            "WHERE experimentalist = ?", (owner,)).fetchone()[0]
    finally:
        conn.close()


def test_no_parked_d8_marker_left_in_criteria_migration():
    source = Path(_mig.__file__).read_text(encoding="utf-8")
    assert "parked: D8" not in source
