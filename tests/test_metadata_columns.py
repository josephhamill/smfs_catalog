# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""Descriptive metadata columns must reach catalogs that already exist.

`initialise()` creates the `files` table only when it is absent, so a column
added to that declaration lands in new catalogs and nowhere else.  Every
catalog in the field was made by an older build, which makes the migration --
not the declaration -- the thing that decides whether a new metadata field
exists for the person who has been collecting data all year.
"""

from smfs_catalog import db


def _columns(path: str) -> set[str]:
    conn = db.get_connection(path)
    try:
        return {row["name"] for row in conn.execute("PRAGMA table_info(files)")}
    finally:
        conn.close()


def test_a_catalog_made_before_a_column_existed_gains_it(tmp_path):
    path = str(tmp_path / "older.sqlite")
    db.initialise(path)

    conn = db.get_connection(path)
    with conn:
        conn.execute("ALTER TABLE files DROP COLUMN sample_prep")
    conn.close()
    assert "sample_prep" not in _columns(path)

    db.initialise(path)

    assert "sample_prep" in _columns(path)


def test_reopening_a_current_catalog_leaves_its_columns_alone(tmp_path):
    path = str(tmp_path / "current.sqlite")
    db.initialise(path)
    before = _columns(path)

    db.initialise(path)

    assert _columns(path) == before


def test_the_descriptive_columns_are_writable_and_readable_back(tmp_path):
    path = str(tmp_path / "write.sqlite")
    db.initialise(path)
    db.upsert_file(
        {"path": "/data/x.ibw", "filename": "x.ibw",
         "first_seen": "2026-01-01", "last_seen": "2026-01-01"},
        db_path=path,
    )

    db.set_file_descriptors_bulk(
        ["/data/x.ibw"], {"substrate": "mica", "sample_prep": "drop-cast"}, path,
    )

    row = db.list_files(db_path=path)[0]
    assert row["substrate"] == "mica"
    assert row["sample_prep"] == "drop-cast"
