# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Tests for scripts/build_catalog.py — building a catalog with a version's own
code.

One build test per era of the code, because the eras differ in what the
analysis entry point is called, and a build that silently analysed nothing
would still produce a file.  Each runs on two seed curves so the suite stays
fast; the full 316-curve build is a run of the tool, not a test of it.

The two build tests need the live catalog and /afm_data, and skip without
them.  Everything else runs anywhere: the guards are what stop a build
producing something other than what was asked for.

Run with the smfs-catalog-dev env, from the repo root:
    python -m pytest tests/test_build_catalog.py -q
"""
import csv
import os
import sqlite3
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import build_catalog as bc                                   # noqa: E402

LIVE = bc.DEFAULT_LIVE
HAVE_DATA = os.path.exists(LIVE) and os.path.isdir("/afm_data")
needs_data = pytest.mark.skipif(
    not HAVE_DATA, reason="needs the live catalog and /afm_data")


# ── Which entry point each era is asked for ───────────────────────────────────

def _scanner_with_tree():
    return types.SimpleNamespace(scan_tree=lambda *a, **k: None)


def test_current_era_uses_analyse_file():
    ca = types.SimpleNamespace(analyse_file=lambda *a: None)
    name, fn = bc._entry_points(_scanner_with_tree(), ca)
    assert name == "analyse_file"
    assert fn is ca.analyse_file


def test_v130_era_uses_analyse_and_classify():
    ca = types.SimpleNamespace(analyse_and_classify=lambda *a: None)
    name, fn = bc._entry_points(_scanner_with_tree(), ca)
    assert name == "analyse_and_classify"
    assert fn is ca.analyse_and_classify


def test_unknown_era_stops_and_names_what_is_missing():
    with pytest.raises(SystemExit) as exc:
        bc._entry_points(_scanner_with_tree(), types.SimpleNamespace())
    assert "analyse_file" in str(exc.value)
    assert "analyse_and_classify" in str(exc.value)


def test_a_version_without_scan_tree_stops():
    with pytest.raises(SystemExit) as exc:
        bc._entry_points(types.SimpleNamespace(),
                         types.SimpleNamespace(analyse_file=lambda *a: None))
    assert "scan_tree" in str(exc.value)


# ── The output is never overwritten by accident ───────────────────────────────

def test_existing_output_is_refused_without_force(tmp_path):
    out = tmp_path / "test_main.db"
    out.write_bytes(b"not to be lost")
    with pytest.raises(SystemExit) as exc:
        bc._clear_output(str(out), force=False)
    assert "--force" in str(exc.value)
    assert out.read_bytes() == b"not to be lost"


def test_force_replaces_the_output_and_its_sidecars(tmp_path):
    out = tmp_path / "test_main.db"
    for suffix in ("", "-wal", "-shm"):
        (tmp_path / ("test_main.db" + suffix)).write_bytes(b"old")
    bc._clear_output(str(out), force=True)
    for suffix in ("", "-wal", "-shm"):
        assert not (tmp_path / ("test_main.db" + suffix)).exists()


# ── The seed list names curves by id, and checks it still means that curve ────

def _fake_catalog(path, rows):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE files (id INTEGER PRIMARY KEY, path TEXT, "
                "experimentalist TEXT, segment_override_params_json TEXT)")
    con.executemany("INSERT INTO files VALUES (?, ?, ?, ?)", rows)
    con.commit()
    con.close()


def test_an_id_that_now_means_another_file_stops_the_build(tmp_path):
    cat = tmp_path / "live.db"
    _fake_catalog(str(cat), [(7, "/afm_data/Alexandre/260318/Image9999.ibw",
                              "Alexandre", None)])
    rows = [{"file_id": "7", "experimentalist": "Alexandre",
             "basename": "Image0682.ibw", "known": "event"}]
    with pytest.raises(SystemExit) as exc:
        bc.resolve_curves(rows, str(cat))
    assert "Image9999.ibw" in str(exc.value)
    assert "Image0682.ibw" in str(exc.value)


def test_an_id_missing_from_the_catalog_stops_the_build(tmp_path):
    cat = tmp_path / "live.db"
    _fake_catalog(str(cat), [])
    rows = [{"file_id": "7", "experimentalist": "Alexandre",
             "basename": "Image0682.ibw", "known": "event"}]
    with pytest.raises(SystemExit) as exc:
        bc.resolve_curves(rows, str(cat))
    assert "file id 7" in str(exc.value)


def test_an_empty_list_stops_the_build(tmp_path):
    empty = tmp_path / "none.csv"
    empty.write_text("file_id,experimentalist,basename,known\n")
    with pytest.raises(SystemExit) as exc:
        bc.read_curve_list(str(empty))
    assert "no curves" in str(exc.value)


def test_the_path_comes_from_the_catalog_not_the_list(tmp_path):
    """The list names curves by id; where each one lives is the catalog's answer."""
    cat = tmp_path / "live.db"
    _fake_catalog(str(cat), [
        (7, "/afm_data/Anastasiia/80% ethanol/260115/Image2907.ibw",
         "Anastasiia", None)])
    rows = [{"file_id": "7", "experimentalist": "Anastasiia",
             "basename": "Image2907.ibw", "known": "event"}]
    got = bc.resolve_curves(rows, str(cat))
    assert got[0]["path"] == \
        "/afm_data/Anastasiia/80% ethanol/260115/Image2907.ibw"
    assert got[0]["experimentalist"] == "Anastasiia"


def test_the_shipped_list_is_the_agreed_test_set():
    rows = bc.read_curve_list(bc.DEFAULT_CURVES)
    assert len(rows) == 316
    assert sum(1 for r in rows if r["known"] == "event") == 64
    assert len({r["file_id"] for r in rows}) == 316


# ── The staging tree ──────────────────────────────────────────────────────────

def test_staging_mirrors_each_path_and_copies_no_data(tmp_path):
    real = tmp_path / "afm" / "Alexandre" / "260318"
    real.mkdir(parents=True)
    curve = real / "Image0682.ibw"
    curve.write_bytes(b"\x00wave")
    stage_root = str(tmp_path / "stage")
    bc.stage({str(curve): "Alexandre"}, stage_root)
    staged = os.path.join(stage_root, str(curve).lstrip(os.sep))
    assert os.path.islink(staged)
    assert os.path.realpath(staged) == os.path.realpath(str(curve))


# ── A real build, once per era ────────────────────────────────────────────────

def _short_list(out_dir, n=3):
    """The first n rows of the shipped test set — all Alexandre's."""
    rows = bc.read_curve_list(bc.DEFAULT_CURVES)[:n]
    path = os.path.join(out_dir, "curves.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path, rows


def _built(version, out_dir):
    curves, _rows = _short_list(out_dir)
    return bc.build(version=version, curves_path=curves, live=LIVE,
                    out=os.path.join(out_dir, "test_%s.db" % version),
                    force=False, python=sys.executable, keep=False)


def _assert_a_real_catalog(report, out):
    assert os.path.exists(out), "no catalog was written"
    # Exactly what the list named, nothing more: the test set is a list.
    assert report["n_files"] == 3, report["n_files"]
    assert report["failures"] == [], report["failures"]
    assert report["verdicts"], "nothing was analysed"
    assert sum(report["verdicts"].values()) == report["n_files"]
    assert report["parameters"]["copied"]["experimentalist_profiles"] > 0

    # Every curve measured under its own owner's set, as the app does.
    assert report["owners_in_force"] == {"Alexandre": "Alexandre"}, \
        report["owners_in_force"]

    con = sqlite3.connect(out)
    try:
        owners = {r[0] for r in con.execute(
            "SELECT DISTINCT experimentalist FROM files")}
        n_profiles = con.execute(
            "SELECT COUNT(*) FROM experimentalist_profiles").fetchone()[0]
        # The verdict reaches the file itself, not only analysis_results: a
        # catalog whose files.event is NULL reads as never analysed.
        n_published = con.execute(
            "SELECT COUNT(*) FROM files WHERE event IS NOT NULL").fetchone()[0]
        queued = con.execute("SELECT COUNT(*) FROM analysis_queue").fetchone()[0]
    finally:
        con.close()
    assert owners == {"Alexandre"}, owners
    assert n_profiles > 0
    assert n_published == report["published"] > 0
    assert queued == 0, "the build left rows in the analysis queue"


@needs_data
def test_build_with_current_code(tmp_path):
    report = _built("main", str(tmp_path))
    assert report["entry_point"] == "analyse_file"
    _assert_a_real_catalog(report, os.path.join(str(tmp_path), "test_main.db"))


@needs_data
def test_build_with_v130_code(tmp_path):
    report = _built("v1.3.0", str(tmp_path))
    assert report["entry_point"] == "analyse_and_classify"
    _assert_a_real_catalog(report, os.path.join(str(tmp_path), "test_v1.3.0.db"))
