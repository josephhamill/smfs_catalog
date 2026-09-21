"""Per-trace 2DH cache preserves the integer-count source representation."""

from __future__ import annotations

import numpy as np

from smfs_catalog import db


def _catalog_with_one_file(tmp_path):
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
    return db_path, file_id


def test_event_histogram_round_trips_uint32_counts(tmp_path):
    db_path, file_id = _catalog_with_one_file(tmp_path)
    expected = np.array([[0, 1], [2, 70000]], dtype=np.uint32)
    db.write_event_histogram(file_id, expected, "v-test", db_path)
    actual = db.get_event_histogram(file_id, "v-test", db_path)

    assert actual.dtype == np.uint32
    np.testing.assert_array_equal(actual.toarray(), expected)
    bulk = db.get_event_histograms_bulk([file_id], "v-test", db_path)
    np.testing.assert_array_equal(bulk[file_id].toarray(), expected)


def test_dense_event_histogram_rows_still_read(tmp_path):
    db_path, file_id = _catalog_with_one_file(tmp_path)
    expected = np.array([[0, 1], [2, 70000]], dtype=np.uint32)
    conn = db.get_connection(db_path)
    with conn:
        conn.execute(
            "INSERT INTO event_histograms (file_id, histogram, x_bins, f_bins,"
            " params_json, computed_at) VALUES (?, ?, 2, 2, 'v-test', 'now')",
            (file_id, expected.tobytes()))
    conn.close()

    actual = db.get_event_histogram(file_id, "v-test", db_path)
    np.testing.assert_array_equal(actual.toarray(), expected)
