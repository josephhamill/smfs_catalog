"""The shared mock catalog gives a real-schema database with known rows."""
import sqlite3

from conftest import add_files
from smfs_catalog import db


def _count(path, table="files"):
    con = sqlite3.connect(path)
    try:
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        con.close()


def test_catalog_is_initialised_and_empty(catalog):
    assert db.catalog_version(catalog) == db.CATALOG_VERSION
    assert _count(catalog) == 0


def test_add_files_leaves_exactly_that_many_rows(catalog):
    paths = add_files(catalog, 3)
    assert _count(catalog) == 3
    assert len(set(paths)) == 3
    con = sqlite3.connect(catalog)
    try:
        stored = [r[0] for r in con.execute("SELECT path FROM files")]
    finally:
        con.close()
    assert sorted(paths) == sorted(stored)
