"""Per-trace 2DH cache preserves the integer-count source representation."""

from __future__ import annotations

import numpy as np

from smfs_catalog import db


def test_event_histogram_round_trips_uint32_counts(tmp_path):
    db_path = str(tmp_path / "catalog.db")
    db.initialise(db_path)
    conn = db.get_connection(db_path)
    with conn:
        file_id = conn.execute(
            """INSERT INTO files (path, filename, first_seen, last_seen)
               VALUES (?, ?, ?, ?)""",
            (str(tmp_path / "trace.ibw"), "trace.ibw", "now", "now"),
        ).lastrowid
    conn.close()

    expected = np.array([[0, 1], [2, 70000]], dtype=np.uint32)
    db.write_event_histogram(file_id, expected, "v-test", db_path)
    actual = db.get_event_histogram(file_id, "v-test", db_path)

    assert actual.dtype == np.uint32
    np.testing.assert_array_equal(actual, expected)
