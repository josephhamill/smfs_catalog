"""Shared test fixtures.

`catalog` is the shared mock catalog: a real-schema database in a temp folder,
built by db.initialise.  `add_files` puts a known number of file rows in it.
"""
import pytest

from smfs_catalog import db


@pytest.fixture
def catalog(tmp_path):
    """Path to a freshly initialised, empty catalog database."""
    path = str(tmp_path / "catalog.db")
    db.initialise(path)
    return path


def add_files(catalog, n):
    """Insert n file rows (file_000.ibw ... ) and return their paths as stored (normalised)."""
    paths = []
    conn = db.get_connection(catalog)
    try:
        with conn:  # one connection, one commit for all rows
            for i in range(n):
                path = f"/data/session/file_{i:03d}.ibw"
                db.upsert_file(
                    {"path": path, "filename": f"file_{i:03d}.ibw",
                     "first_seen": "2026-01-01T00:00:00",
                     "last_seen": "2026-01-01T00:00:00"},
                    catalog, conn=conn,
                )
                paths.append(db.normalize_path(path))
    finally:
        conn.close()
    return paths
