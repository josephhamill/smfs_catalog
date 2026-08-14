from pathlib import Path

from smfs_catalog import criteria_gate as gate
from smfs_catalog import db


def test_first_paths_profile_applies_to_the_cohort(tmp_path, monkeypatch):
    db_path = str(tmp_path / "gate.db")
    db.initialise(db_path)
    paths = [db.normalize_path(str(tmp_path / name)) for name in ("a.ibw", "b.ibw")]
    conn = db.get_connection(db_path)
    with conn:
        conn.execute(
            "INSERT INTO watched_directories(path, added_at) VALUES (?, ?)",
            (db.normalize_path(str(tmp_path)), "now"),
        )
        directory_id = conn.execute(
            "SELECT id FROM watched_directories"
        ).fetchone()[0]
        for path, owner in zip(paths, ("A", "B")):
            conn.execute(
                "INSERT INTO files(path, directory_id, filename, first_seen, "
                "last_seen, experimentalist, event) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (path, directory_id, Path(path).name, "now", "now", owner, "event"),
            )
        file_ids = [
            conn.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()[0]
            for path in paths
        ]
    conn.close()

    values = {paths[0]: {"metric": 0.5}, paths[1]: {"metric": 5.0}}
    monkeypatch.setattr(gate, "_values", lambda paths, checked, db_path: values)
    for owner, upper in (("A", 1.0), ("B", 10.0)):
        gate.set_criterion("metric", True, owner, db_path)
        db.set_threshold(
            "metric", 0.0, upper, experimentalist=owner, db_path=db_path
        )

    db.enqueue_files(file_ids, db_path)
    assert gate.evaluate(paths, db_path) == ([paths[0]], [paths[1]])
    db.clear_analysis_queue(db_path)
    db.enqueue_files([file_ids[1]], db_path)
    assert gate.evaluate(paths, db_path) == (paths, [])
    db.clear_analysis_queue(db_path)
    db.enqueue_files(file_ids, db_path)
    assert set(gate.explain(paths, db_path)) == {paths[1]}


def test_no_bounded_criterion_passes_everything(tmp_path):
    db_path = str(tmp_path / "gate.db")
    db.initialise(db_path)
    paths = [db.normalize_path(str(tmp_path / "a.ibw"))]
    gate.set_criterion("unbounded", True, "Default", db_path)

    assert gate.evaluate(paths, db_path) == (paths, [])
    assert gate.explain(paths, db_path) == {}
    assert gate.has_criteria_checked(paths, db_path) == {paths[0]: False}
