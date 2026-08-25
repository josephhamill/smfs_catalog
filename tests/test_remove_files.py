# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Regression test: removing things from the catalog — the two-level, scope-
selected undo for Add Data and for an analysis run.

Before this landed there was no way to remove anything from the catalog in
the dashboard at all, and the one removal API (db.remove_directory) deleted
a directory row while explicitly leaving its file records behind — orphaning
every one of them, and nothing called it.

The contract under test:
(a) level 2 (erase_analysis_for_files) deletes every computed result for the
    cohort — analysis_results / event_histograms / event_map — and clears
    files.event, while KEEPING the file rows, their sample metadata, their
    file_metadata (parsed from the wave note, not computed) and their queue
    entries, so the cohort can simply be re-analysed.
(b) level 2 keeps the human's manual Primary/Secondary segment picks (#107):
    they are a person's curation, not the app's output, and they already
    self-invalidate via resolve_segment_override's params check.
(c) level 1 (remove_files_from_catalog) deletes the file rows themselves plus
    all five dependent tables.  There is no ON DELETE CASCADE in this schema
    and foreign_keys is ON, so this only works if children go first — a
    regression here shows up as an IntegrityError, not as silent orphaning.
(d) BOTH levels touch only files in the given cohort: an out-of-scope file in
    the same directory keeps its rows, its analysis and its queue entry.
(e) a folder is not a catalog object: it is read off files.path, so
    emptying one simply stops it appearing, and a folder that still holds an
    out-of-scope file keeps appearing.
(f) describe_removal_scope reports what WOULD happen without changing
    anything — it is what the confirmation dialog puts in front of the user.
(g) neither level touches the filesystem: a real file on disk survives both.

Run with the smfs-catalog env, from the repo root:
    python tests/test_remove_files.py
"""
import os
import sys
import sqlite3
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smfs_catalog import db as _db

tmp = tempfile.mkdtemp(prefix="remove_files_")
DB = os.path.join(tmp, "test.sqlite")
_db.initialise(DB)

# A real directory with a real file in it, so (g) can check the disk.
DATA_DIR = os.path.join(tmp, "cohort")
OTHER_DIR = os.path.join(tmp, "empty_dir")
os.makedirs(DATA_DIR)
os.makedirs(OTHER_DIR)

IN_SCOPE_A = os.path.join(DATA_DIR, "Image0001.ibw")
IN_SCOPE_B = os.path.join(DATA_DIR, "Image0002.ibw")
OUT_OF_SCOPE = os.path.join(DATA_DIR, "Image0003.ibw")   # same dir, NOT in cohort
for p in (IN_SCOPE_A, IN_SCOPE_B, OUT_OF_SCOPE):
    with open(p, "wb") as fh:
        fh.write(b"not really an ibw, just has to exist on disk")

COHORT = [IN_SCOPE_A, IN_SCOPE_B]

# One shared idiom for these procedural guards — see checkstyle.py for why
# `sys.exit(1)` at the bottom of a file was aborting the whole pytest run.
import checkstyle                                          # noqa: E402

check = checkstyle.CheckRunner()


def seed():
    """Rebuild the whole fixture — two dirs, three files, full analysis."""
    conn = _db.get_connection(DB)
    with conn:
        for tbl in ("event_map", "event_histograms", "analysis_results",
                    "file_metadata", "analysis_queue", "files"):
            conn.execute(f"DELETE FROM {tbl}")
        for path in (IN_SCOPE_A, IN_SCOPE_B, OUT_OF_SCOPE):
            conn.execute(
                "INSERT INTO files (path, filename, first_seen,"
                " last_seen, solvent, event, primary_segment_idx,"
                " secondary_segment_idx, segment_override_params_json)"
                " VALUES (?, ?, datetime('now'), datetime('now'),"
                " 'PBS', 'event', 2, 1, '{\"p\": 1}')",
                (_db.normalize_path(path), os.path.basename(path)))
        for path in (IN_SCOPE_A, IN_SCOPE_B, OUT_OF_SCOPE):
            fid = conn.execute(
                "SELECT id FROM files WHERE path = ?",
                (_db.normalize_path(path),)).fetchone()["id"]
            conn.execute(
                "INSERT INTO analysis_results (file_id, analysis_type, value,"
                " params_json, code_version, computed_at)"
                " VALUES (?, 'seg_force_pN', 42.0, '{}', 'v1', datetime('now'))",
                (fid,))
            conn.execute(
                "INSERT INTO event_histograms (file_id, histogram, x_bins,"
                " f_bins, params_json, computed_at)"
                " VALUES (?, ?, 10, 10, '{}', datetime('now'))",
                (fid, b"hist"))
            conn.execute(
                "INSERT INTO event_map (file_id, params_json, code_version,"
                " payload_json, computed_at)"
                " VALUES (?, '{}', 'v1', '{}', datetime('now'))",
                (fid,))
            conn.execute(
                "INSERT INTO file_metadata (file_id, key, value_text)"
                " VALUES (?, 'SpringConstant', '0.04')", (fid,))
            conn.execute(
                "INSERT INTO analysis_queue (file_id, enqueued_at, status)"
                " VALUES (?, datetime('now'), 'pending')", (fid,))
    conn.close()


def rows(sql, params=()):
    conn = _db.get_connection(DB)
    out = conn.execute(sql, params).fetchall()
    conn.close()
    return out


def n_rows(table, path):
    r = rows(
        f"SELECT COUNT(*) AS n FROM {table} WHERE file_id ="
        " (SELECT id FROM files WHERE path = ?)", (_db.normalize_path(path),))
    return r[0]["n"]


def file_exists_in_db(path):
    return bool(rows("SELECT 1 FROM files WHERE path = ?",
                     (_db.normalize_path(path),)))


# -- (f) describe_removal_scope is read-only and honest -----------------------
seed()
info = _db.describe_removal_scope(COHORT, DB)
check("(f) describe reports the cohort size", info["n_files"] == 2)
check("(f) describe counts classified files", info["n_classified"] == 2)
check("(f) describe counts events", info["n_events"] == 2)
check("(f) describe counts files holding stored fits", info["n_with_fits"] == 2)
check("(f) describe counts queued files", info["n_queued"] == 2)
check("(f) describe changed nothing — all 3 files still present",
      len(rows("SELECT id FROM files")) == 3)
check("(f) describe changed nothing — event_map intact",
      len(rows("SELECT file_id FROM event_map")) == 3)

# -- (a)(b)(d) level 2: erase analysis ---------------------------------------
seed()
out = _db.erase_analysis_for_files(COHORT, DB)
check("(a) erase reports the cohort size", out["n_files"] == 2)
check("(a) erase deleted both cohort event_map rows", out["event_map"] == 2)

check("(a) erase KEEPS the file rows", file_exists_in_db(IN_SCOPE_A))
check("(a) erase cleared files.event",
      rows("SELECT event FROM files WHERE path = ?",
           (_db.normalize_path(IN_SCOPE_A),))[0]["event"] is None)
check("(a) files.hit no longer exists to go stale",
      "hit" not in {r["name"] for r in
                    _db.get_connection(DB).execute("PRAGMA table_info(files)")})

seg = rows("SELECT primary_segment_idx, secondary_segment_idx FROM files"
           " WHERE path = ?", (_db.normalize_path(IN_SCOPE_A),))[0]
check("(b) erase KEEPS the manual Primary pick (human curation, not output)",
      seg["primary_segment_idx"] == 2)
check("(b) erase KEEPS the manual Secondary pick",
      seg["secondary_segment_idx"] == 1)

check("(d) erase left the out-of-scope file's verdict alone",
      rows("SELECT event FROM files WHERE path = ?",
           (_db.normalize_path(OUT_OF_SCOPE),))[0]["event"] == "event")
check("(d) erase left the out-of-scope file's fits alone",
      n_rows("event_map", OUT_OF_SCOPE) == 1)

# -- (c)(d)(e)(g) level 1: remove from catalog --------------------------------
seed()
try:
    out = _db.remove_files_from_catalog(COHORT, DB)
    integrity_ok = True
except sqlite3.IntegrityError as exc:
    integrity_ok = False
    out = {}
    print(f"      IntegrityError: {exc}")
check("(c) removal does not trip a foreign-key constraint "
      "(children deleted before parents)", integrity_ok)
check("(c) removal reports the cohort size", out.get("n_files") == 2)
check("(c) the file rows are gone", not file_exists_in_db(IN_SCOPE_A))
check("(c) their event_map rows are gone",
      len(rows("SELECT file_id FROM event_map")) == 1)
check("(c) their file_metadata rows are gone",
      len(rows("SELECT file_id FROM file_metadata")) == 1)
check("(c) their queue entries are gone",
      len(rows("SELECT file_id FROM analysis_queue")) == 1)

check("(d) the out-of-scope file survives intact", file_exists_in_db(OUT_OF_SCOPE))
check("(d) so does its analysis", n_rows("event_map", OUT_OF_SCOPE) == 1)
check("(d) so does its queue entry", n_rows("analysis_queue", OUT_OF_SCOPE) == 1)

check("(e) the folder still lists while it holds the out-of-scope file",
      _db.normalize_path(DATA_DIR) in
      {r["path"] for r in _db.list_directories(DB)})

check("(g) every .ibw is still on disk after a catalog removal",
      all(os.path.exists(p) for p in (IN_SCOPE_A, IN_SCOPE_B, OUT_OF_SCOPE)))

# Take the last file too — the folder simply stops appearing.
out = _db.remove_files_from_catalog([OUT_OF_SCOPE], DB)
check("(e) a folder holding no catalogued files stops listing",
      not _db.list_directories(DB))
check("(g) the files are still on disk after removing every catalog entry",
      all(os.path.exists(p) for p in (IN_SCOPE_A, IN_SCOPE_B, OUT_OF_SCOPE)))

# -- empty input is a no-op, not a wipe ---------------------------------------
seed()
before = len(rows("SELECT id FROM files"))
_db.erase_analysis_for_files([], DB)
_db.remove_files_from_catalog([], DB)
check("an empty cohort removes NOTHING (never a whole-table delete)",
      len(rows("SELECT id FROM files")) == before
      and len(rows("SELECT file_id FROM event_map")) == 3)


# Every check above becomes its own named pytest case.  Must be last:
# pytest_cases reads what the module body recorded.
test_check = checkstyle.pytest_cases(check)
