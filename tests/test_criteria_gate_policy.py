from pathlib import Path

from smfs_catalog import criteria_gate as gate
from smfs_catalog import db


def test_first_paths_profile_applies_to_the_cohort(tmp_path, monkeypatch):
    db_path = str(tmp_path / "gate.db")
    db.initialise(db_path)
    paths = [db.normalize_path(str(tmp_path / name)) for name in ("a.ibw", "b.ibw")]
    conn = db.get_connection(db_path)
    with conn:
        for path, owner in zip(paths, ("A", "B")):
            conn.execute(
                "INSERT INTO files(path, filename, first_seen, "
                "last_seen, experimentalist, event) VALUES (?, ?, ?, ?, ?, ?)",
                (path, Path(path).name, "now", "now", owner, "event"),
            )
        file_ids = [
            conn.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()[0]
            for path in paths
        ]
    conn.close()

    values = {paths[0]: {"metric": 0.5}, paths[1]: {"metric": 5.0}}
    monkeypatch.setattr(gate, "_values", lambda paths, keys, db_path: values)
    for owner, upper in (("A", 1.0), ("B", 10.0)):
        db.set_threshold(
            "metric", 0.0, upper, experimentalist=owner, db_path=db_path
        )

    db.enqueue_files(file_ids, db_path)
    cls = gate.classify(paths, db_path)
    assert cls.population(gate.HIT) == [paths[0]]
    assert cls.population(gate.NON_HIT) == [paths[1]]

    db.clear_analysis_queue(db_path)
    db.enqueue_files([file_ids[1]], db_path)
    assert gate.classify(paths, db_path).population(gate.HIT) == paths

    db.clear_analysis_queue(db_path)
    db.enqueue_files(file_ids, db_path)
    assert set(gate.classify(paths, db_path).reasons) == {paths[1]}


def test_a_row_with_no_bounds_is_not_a_criterion(tmp_path):
    """Clearing both bounds is how a criterion is removed, so an all-NULL row
    must not gate — and it must not count as a gate being in force either."""
    db_path = str(tmp_path / "gate.db")
    db.initialise(db_path)
    paths = [db.normalize_path(str(tmp_path / "a.ibw"))]
    db.set_threshold("unbounded", None, None, experimentalist="Default",
                     db_path=db_path)

    assert not gate.gate(None, db_path)
    cls = gate.classify(paths, db_path)
    assert cls.population(gate.HIT) == paths
    assert cls.reasons == {}


def test_either_bound_alone_makes_a_criterion(tmp_path, monkeypatch):
    """A one-sided bound gates. The other end simply does not constrain."""
    db_path = str(tmp_path / "gate.db")
    db.initialise(db_path)
    paths = [db.normalize_path(str(tmp_path / n)) for n in ("a.ibw", "b.ibw")]
    monkeypatch.setattr(gate, "_values", lambda p, k, d: {
        paths[0]: {"metric": 0.5}, paths[1]: {"metric": 5.0}})
    db.set_threshold("metric", 1.0, None, experimentalist="Default",
                     db_path=db_path)

    g = gate.gate(None, db_path)
    assert bool(g) and len(g.criteria) == 1
    cls = gate.classify(paths, db_path)
    assert cls.population(gate.NON_HIT) == [paths[0]]
    assert cls.why(paths[0])[0][2] == "below"


def test_a_missing_value_fails_a_bounded_criterion(tmp_path, monkeypatch):
    """A curve with no value for a gating variable cannot pass it.

    The reason is tagged "missing" rather than a violation, because the
    question could not be asked of this curve — the Hit tooltip tells the
    two apart.
    """
    db_path = str(tmp_path / "gate.db")
    db.initialise(db_path)
    paths = [db.normalize_path(str(tmp_path / "a.ibw"))]
    monkeypatch.setattr(gate, "_values", lambda p, k, d: {paths[0]: {}})
    db.set_threshold("metric", 0.0, 1.0, experimentalist="Default",
                     db_path=db_path)

    cls = gate.classify(paths, db_path)
    assert cls.of(paths[0]) == gate.NON_HIT
    assert cls.why(paths[0]) == [("metric", None, "missing", None)]
