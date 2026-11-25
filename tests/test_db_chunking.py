"""Bulk db functions give the right answer when their id/path list spans several
800-item chunks, and db.py has exactly one chunker (_sql_chunks)."""
import ast
import shutil
from pathlib import Path

import pytest

from smfs_catalog import db

from conftest import add_files

N = 1700  # three chunks at size 800: 800 + 800 + 100
EDGES = (0, 1, 799, 800, 801, 1599, 1600, N - 1)  # either side of each boundary


@pytest.fixture(scope="module")
def _template(tmp_path_factory):
    """Build the N-file catalog once (slow); each test gets its own copy."""
    path = str(tmp_path_factory.mktemp("tpl") / "catalog.db")
    db.initialise(path)
    paths = add_files(path, N)
    ids = [db.get_file_id(p, path) for p in paths]
    assert None not in ids
    return path, paths, ids


@pytest.fixture
def many(_template, tmp_path):
    """A private copy of the N-file catalog; returns (catalog, paths, file ids in path order)."""
    src, paths, ids = _template
    dest = str(tmp_path / "catalog.db")
    shutil.copyfile(src, dest)
    return dest, paths, ids


def _read(catalog, sql):
    c = db.get_connection(catalog)
    try:
        return c.execute(sql).fetchall()
    finally:
        c.close()


def test_sql_chunks_splits_into_800s_in_order():
    chunks = list(db._sql_chunks(list(range(N))))
    assert [len(c) for c in chunks] == [800, 800, 100]
    assert [x for c in chunks for x in c] == list(range(N))


def test_set_file_descriptors_bulk_updates_every_path_across_chunks(many):
    catalog, paths, _ = many
    n = db.set_file_descriptors_bulk(paths, {"analyte": "BSA"}, catalog)
    assert n == N
    rows = _read(catalog, "SELECT COUNT(*) FROM files WHERE analyte = 'BSA'")
    assert rows[0][0] == N


def test_set_file_descriptors_bulk_leaves_unlisted_files_alone(many):
    catalog, paths, _ = many
    n = db.set_file_descriptors_bulk(paths[:-1], {"analyte": "BSA"}, catalog)
    assert n == N - 1
    left = _read(catalog, "SELECT path FROM files WHERE analyte IS NULL")
    assert [r[0] for r in left] == [paths[-1]]


def test_get_derived_results_bulk_latest_returns_all_across_chunks(many):
    catalog, paths, ids = many
    for fid in (ids[i] for i in EDGES):
        db.write_analysis_result(fid, "thing", float(fid), "{}", "v1", catalog)
    got = db.get_derived_results_bulk_latest(paths, ["thing"], catalog)
    assert set(got) == {paths[i] for i in EDGES}
    for i in EDGES:
        assert got[paths[i]] == {"thing": (float(ids[i]), "{}")}


def test_get_derived_results_bulk_latest_with_no_results_is_empty(many):
    catalog, paths, _ = many
    assert db.get_derived_results_bulk_latest(paths, ["thing"], catalog) == {}


def test_get_segment_overrides_bulk_returns_every_id_across_chunks(many):
    catalog, _, ids = many
    for i in EDGES:
        db.set_primary_segment_idx(ids[i], i, f'{{"i": {i}}}', catalog)
    got = db.get_segment_overrides_bulk(ids, catalog)
    assert set(got) == set(ids)
    for i in EDGES:
        assert got[ids[i]]["primary_segment_idx"] == i
        assert got[ids[i]]["params_json"] == f'{{"i": {i}}}'
    untouched = [k for j, k in enumerate(ids) if j not in EDGES]
    assert all(got[k]["primary_segment_idx"] is None for k in untouched)


def test_import_queue_from_paths_enqueues_known_and_counts_unknown(many):
    catalog, paths, ids = many
    unknown = [f"/data/session/missing_{i}.ibw" for i in range(3)]
    n_enqueued, n_missing = db.import_queue_from_paths(paths + unknown, catalog)
    assert (n_enqueued, n_missing) == (N, 3)
    queued = _read(catalog, "SELECT file_id FROM analysis_queue")
    assert sorted(r[0] for r in queued) == sorted(ids)


def _db_tree():
    src = Path(db.__file__).read_text(encoding="utf-8")
    return ast.parse(src)


def test_db_has_no_local_chunks_function():
    found = [n.lineno for n in ast.walk(_db_tree())
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name == "_chunks"]
    assert found == [], f"local _chunks defined at db.py lines {found}; use _sql_chunks"


def test_db_has_no_hand_written_800_step_loop():
    found = []
    for n in ast.walk(_db_tree()):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "range" and len(n.args) == 3
                and isinstance(n.args[2], ast.Constant) and n.args[2].value == 800):
            found.append(n.lineno)
    assert found == [], f"range(..., 800) loops at db.py lines {found}; use _sql_chunks"
