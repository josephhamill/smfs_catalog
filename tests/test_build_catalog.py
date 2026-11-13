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
import json
import os
import shutil
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


# ── The copied profiles, written in the version's form ────────────────────────

def _profiles_catalog(path, profiles, thresholds):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE experimentalist_profiles (experimentalist TEXT "
                "PRIMARY KEY, params_json TEXT, updated_at TEXT)")
    con.execute("CREATE TABLE thresholds (experimentalist TEXT, analysis_type "
                "TEXT, lower_bound REAL, upper_bound REAL)")
    con.executemany("INSERT INTO experimentalist_profiles VALUES (?, ?, '')",
                    [(o, json.dumps(p)) for o, p in profiles.items()])
    con.executemany("INSERT INTO thresholds VALUES (?, ?, ?, ?)", thresholds)
    con.commit()
    con.close()


def _a_version(declared, ticks=None):
    """A stand-in for a version's db and criteria_gate modules.

    `ticks` is the list its set_criterion records into; None means the version
    has no set_criterion, as every version since a bound became the criterion.
    """
    def get_thresholds(owner, db_path):
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        try:
            return con.execute("SELECT * FROM thresholds WHERE experimentalist = ?",
                               (owner,)).fetchall()
        finally:
            con.close()
    db = types.SimpleNamespace(profile_defaults=lambda: dict.fromkeys(declared, 0.0),
                               get_thresholds=get_thresholds)
    if ticks is None:
        return db, types.SimpleNamespace()
    return db, types.SimpleNamespace(
        set_criterion=lambda key, on, owner, db_path: ticks.append((owner, key, on)))


def test_profile_keys_the_version_does_not_declare_are_removed(tmp_path):
    cat = str(tmp_path / "built.db")
    _profiles_catalog(cat, {"Alexandre": {"seg_window": 5.0, "added_later": 1.0}}, [])
    db, gate = _a_version(declared=["seg_window"])
    report = bc.fit_profiles_to_version(db, gate, cat)
    con = sqlite3.connect(cat)
    stored = json.loads(con.execute("SELECT params_json FROM experimentalist_profiles"
                                    ).fetchone()[0])
    con.close()
    assert stored == {"seg_window": 5.0}
    assert report["removed"] == {"Alexandre": ["added_later"]}


def test_a_version_that_gates_by_ticks_gets_a_tick_for_every_bound(tmp_path):
    cat = str(tmp_path / "built.db")
    _profiles_catalog(cat, {"Alexandre": {"seg_window": 5.0}}, [
        ("Alexandre", "baseline_rms", 0.001, 0.4),
        ("Alexandre", "seg_l_c_err", None, 10.0),
        ("Alexandre", "invols_rms", None, None),     # no bound: not a criterion
    ])
    ticks = []
    db, gate = _a_version(declared=["seg_window"], ticks=ticks)
    report = bc.fit_profiles_to_version(db, gate, cat)
    assert sorted(ticks) == [("Alexandre", "baseline_rms", True),
                             ("Alexandre", "seg_l_c_err", True)]
    assert report["ticked"] == {"Alexandre": ["baseline_rms", "seg_l_c_err"]}


def test_a_version_where_the_bound_is_the_criterion_gets_no_ticks(tmp_path):
    cat = str(tmp_path / "built.db")
    _profiles_catalog(cat, {"Alexandre": {"seg_window": 5.0}},
                      [("Alexandre", "baseline_rms", 0.001, 0.4)])
    db, gate = _a_version(declared=["seg_window"])
    report = bc.fit_profiles_to_version(db, gate, cat)
    assert report["ticked"] == {}


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


def _n_bounds(path):
    con = sqlite3.connect(path)
    try:
        return con.execute("SELECT COUNT(*) FROM thresholds WHERE lower_bound IS "
                           "NOT NULL OR upper_bound IS NOT NULL").fetchone()[0]
    finally:
        con.close()


def _bounded_and_ticked(path, default_owner):
    """Per owner: the criteria with a bound in force, and the criteria ticked.

    The bound in force is the owner's own row over the default owner's, as
    db.get_thresholds merges them.
    """
    con = sqlite3.connect(path)
    try:
        rows = con.execute("SELECT experimentalist, analysis_type, lower_bound, "
                           "upper_bound FROM thresholds").fetchall()
        profiles = dict(con.execute(
            "SELECT experimentalist, params_json FROM experimentalist_profiles"))
    finally:
        con.close()
    out = {}
    for owner, params_json in profiles.items():
        merged = {key: (lo, hi) for who, key, lo, hi in rows if who == default_owner}
        merged.update({key: (lo, hi) for who, key, lo, hi in rows if who == owner})
        bounded = {key for key, b in merged.items() if b != (None, None)}
        ticked = {key[len("criteria_use:"):]
                  for key, on in json.loads(params_json).items()
                  if key.startswith("criteria_use:") and on}
        out[owner] = (bounded, ticked)
    return out


@needs_data
def test_build_with_current_code(tmp_path):
    report = _built("main", str(tmp_path))
    assert report["entry_point"] == "analyse_file"
    _assert_a_real_catalog(report, os.path.join(str(tmp_path), "test_main.db"))
    # A bound is the criterion in this era: nothing to tick.
    assert report["profiles"]["ticked"] == {}


@needs_data
def test_build_with_v130_code(tmp_path):
    from smfs_catalog import db as today

    report = _built("v1.3.0", str(tmp_path))
    out = os.path.join(str(tmp_path), "test_v1.3.0.db")
    assert report["entry_point"] == "analyse_and_classify"
    _assert_a_real_catalog(report, out)

    # v1.3.0 gated a criterion only when ticked: every bound in force is ticked.
    per_owner = _bounded_and_ticked(out, today.DEFAULT_EXPERIMENTALIST)
    assert any(bounded for bounded, _ in per_owner.values())
    for owner, (bounded, ticked) in per_owner.items():
        assert bounded == ticked, owner

    # Every profile key is one v1.3.0 declares, or one of its ticks.
    con = sqlite3.connect(out)
    keys = set()
    for (params_json,) in con.execute("SELECT params_json FROM experimentalist_profiles"):
        keys |= set(json.loads(params_json))
    con.close()
    declared = set(report["profiles"]["declared"])
    assert {k for k in keys if not k.startswith("criteria_use:")} <= declared

    # Today's setup, run on a copy, keeps every bound: the gate carries over.
    copy = os.path.join(str(tmp_path), "upgraded.db")
    shutil.copy(out, copy)
    before = _n_bounds(copy)
    today.initialise(copy)
    assert before > 0
    assert _n_bounds(copy) == before


@needs_data
def test_build_with_labograph_baseline_code(tmp_path):
    from smfs_catalog import db as today

    report = _built("labograph-baseline", str(tmp_path))
    out = os.path.join(str(tmp_path), "test_labograph-baseline.db")
    assert report["entry_point"] == "analyse_file"
    _assert_a_real_catalog(report, out)
    # A bound is the criterion in this era: nothing to tick.
    assert report["profiles"]["ticked"] == {}

    # Every profile key is one this version declares.
    con = sqlite3.connect(out)
    keys = set()
    for (params_json,) in con.execute("SELECT params_json FROM experimentalist_profiles"):
        keys |= set(json.loads(params_json))
    con.close()
    assert keys <= set(report["profiles"]["declared"])

    # Today's setup, run on a copy, keeps every bound.
    copy = os.path.join(str(tmp_path), "upgraded.db")
    shutil.copy(out, copy)
    before = _n_bounds(copy)
    today.initialise(copy)
    assert before > 0
    assert _n_bounds(copy) == before
