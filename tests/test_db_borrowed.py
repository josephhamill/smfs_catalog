"""db._borrowed: use the caller's connection or open (and close) our own."""
import sqlite3

import pytest

from smfs_catalog import db

from conftest import add_files


class Boom(Exception):
    pass


def _insert(conn, name):
    conn.execute(
        "INSERT INTO files (path, filename, first_seen, last_seen) VALUES (?, ?, ?, ?)",
        (f"/data/{name}", name, "2026-01-01T00:00:00", "2026-01-01T00:00:00"),
    )


def _count(catalog):
    """Rows in files, seen through a separate connection (only committed data)."""
    c = db.get_connection(catalog)
    try:
        return c.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    finally:
        c.close()


def _assert_closed(conn):
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_borrowed_connection_is_yielded_and_stays_usable(catalog):
    conn = db.get_connection(catalog)
    try:
        with db._borrowed(conn, catalog) as c:
            assert c is conn
        assert conn.execute("SELECT 1").fetchone()[0] == 1
    finally:
        conn.close()


def test_borrowed_connection_stays_usable_after_error(catalog):
    conn = db.get_connection(catalog)
    try:
        with pytest.raises(Boom):
            with db._borrowed(conn, catalog):
                raise Boom
        assert conn.execute("SELECT 1").fetchone()[0] == 1
    finally:
        conn.close()


def test_own_connection_is_closed_after_block(catalog):
    with db._borrowed(None, catalog) as c:
        assert c.execute("SELECT 1").fetchone()[0] == 1
    _assert_closed(c)


def test_own_connection_is_closed_after_error(catalog):
    holder = []
    with pytest.raises(Boom):
        with db._borrowed(None, catalog) as c:
            holder.append(c)
            raise Boom
    _assert_closed(holder[0])


def test_commit_true_persists_insert_on_success(catalog):
    with db._borrowed(None, catalog, commit=True) as c:
        _insert(c, "a.ibw")
    assert _count(catalog) == 1


def test_commit_true_drops_insert_on_error(catalog):
    with pytest.raises(Boom):
        with db._borrowed(None, catalog, commit=True) as c:
            _insert(c, "a.ibw")
            raise Boom
    assert _count(catalog) == 0


def test_commit_true_on_borrowed_connection_commits_and_keeps_it_open(catalog):
    conn = db.get_connection(catalog)
    try:
        with db._borrowed(conn, catalog, commit=True) as c:
            _insert(c, "a.ibw")
        assert _count(catalog) == 1
        assert conn.execute("SELECT 1").fetchone()[0] == 1
    finally:
        conn.close()


def test_commit_true_on_borrowed_connection_rolls_back_on_error(catalog):
    conn = db.get_connection(catalog)
    try:
        with pytest.raises(Boom):
            with db._borrowed(conn, catalog, commit=True) as c:
                _insert(c, "a.ibw")
                raise Boom
        assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 0
        assert _count(catalog) == 0
    finally:
        conn.close()


def test_commit_false_does_not_commit(catalog):
    conn = db.get_connection(catalog)
    try:
        with db._borrowed(conn, catalog) as c:
            _insert(c, "a.ibw")
        assert _count(catalog) == 0
    finally:
        conn.close()


# --- one real converted site: set_event (worked example in the plan) ---

def _event(catalog, file_id):
    c = db.get_connection(catalog)
    try:
        return c.execute("SELECT event FROM files WHERE id = ?", (file_id,)).fetchone()[0]
    finally:
        c.close()


def _first_file_id(catalog):
    add_files(catalog, 1)
    c = db.get_connection(catalog)
    try:
        return c.execute("SELECT id FROM files").fetchone()[0]
    finally:
        c.close()


def test_set_event_with_callers_connection_commits_and_leaves_it_open(catalog):
    file_id = _first_file_id(catalog)
    verdict = sorted(db.EVENT_VERDICTS)[0]
    conn = db.get_connection(catalog)
    try:
        db.set_event(file_id, verdict, catalog, conn=conn)
        assert _event(catalog, file_id) == verdict  # committed, seen from another connection
        assert conn.execute("SELECT 1").fetchone()[0] == 1  # still open
    finally:
        conn.close()


def test_set_event_own_connection_error_mid_update_leaves_row_unchanged(catalog):
    file_id = _first_file_id(catalog)
    verdict = sorted(db.EVENT_VERDICTS)[0]
    setup = db.get_connection(catalog)
    try:
        with setup:
            setup.execute(
                "CREATE TRIGGER fail_event BEFORE UPDATE OF event ON files "
                "BEGIN SELECT RAISE(ABORT, 'boom'); END"
            )
    finally:
        setup.close()
    with pytest.raises(sqlite3.Error):
        db.set_event(file_id, verdict, catalog)
    assert _event(catalog, file_id) is None
