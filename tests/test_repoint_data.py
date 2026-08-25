# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Regression test: repointing the catalog at data that has moved (#30).

Curves get moved to a bigger drive.  Every path in the catalog is then
wrong at once, and without a repair the only way back is a new database and
a re-run of every analysis.

The whole point is that ONLY the path is stale.  Every verdict, ROI, fit,
histogram, queue entry and hand-set segment pick is keyed on files.id, so
repointing must rewrite paths and touch nothing else.

The contract under test:
(a) relocate_files rewrites every path under the old root, keeping the row
    id — and with it every dependent row.
(b) files OUTSIDE the old root are untouched, so a partial move is safe.
(c) the prefix is matched with a separator: repointing /curves must not drag
    /curves2 along with it.
(d) describe_relocation reports what WOULD happen — including how many of
    the new paths actually exist on disk — and changes nothing.
(e) a file whose new path is already held by a different catalog entry (the
    same data imported twice) is left alone and reported, never merged: the
    two entries carry different analysis and the app must not pick one.
(f) re-nesting (/data -> /data/2026) works, though every new path collides
    with some moving row's old path.
(g) nothing on disk is created, moved or deleted, and a root that is not
    mounted yet still repoints — the drive may be plugged in afterwards.
(h) old_root == new_root is a no-op, and so is a root nothing lives under.
(i) missing_by_directory counts orphans exactly — that is how the user finds
    out a move happened at all.

Run with the smfs-catalog env, from the repo root:
    python tests/test_repoint_data.py
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smfs_catalog import db as _db

import checkstyle                                          # noqa: E402

check = checkstyle.CheckRunner()

tmp = tempfile.mkdtemp(prefix="repoint_")
DB = os.path.join(tmp, "test.sqlite")
_db.initialise(DB)

OLD = os.path.join(tmp, "old_drive", "curves")
NEW = os.path.join(tmp, "new_drive", "curves")
SUB = os.path.join(OLD, "day2")
DECOY = os.path.join(tmp, "old_drive", "curves2")     # (c): must NOT move
ELSEWHERE = os.path.join(tmp, "other", "curves")

N = _db.normalize_path

LAYOUT = [
    (1, os.path.join(OLD, "Force0001.ibw")),
    (2, os.path.join(SUB, "Force0002.ibw")),
    (3, os.path.join(ELSEWHERE, "Force0003.ibw")),
    (4, os.path.join(DECOY, "Force0004.ibw")),
]


def seed(new_on_disk: bool = True):
    """Four analysed curves: two under OLD (one nested), one elsewhere, one in the look-alike folder."""
    for _fid, p in LAYOUT:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(b"stand-in for an ibw")
    if new_on_disk:
        os.makedirs(os.path.join(NEW, "day2"), exist_ok=True)
        for rel in ("Force0001.ibw", os.path.join("day2", "Force0002.ibw")):
            with open(os.path.join(NEW, rel), "wb") as fh:
                fh.write(b"moved")

    conn = _db.get_connection(DB)
    with conn:
        for tbl in ("event_map", "event_histograms", "analysis_results",
                    "file_metadata", "analysis_queue", "files"):
            conn.execute(f"DELETE FROM {tbl}")
        for fid, p in LAYOUT:
            conn.execute(
                "INSERT INTO files (id, path, filename, first_seen, last_seen,"
                " event, primary_segment_idx)"
                " VALUES (?, ?, ?, datetime('now'), datetime('now'), 'event', 2)",
                (fid, N(p), os.path.basename(p)))
            conn.execute(
                "INSERT INTO analysis_results (file_id, analysis_type, value,"
                " params_json, code_version, computed_at)"
                " VALUES (?, 'seg_force_pN', 42.0, '{}', 'v1', datetime('now'))",
                (fid,))
            conn.execute(
                "INSERT INTO event_map (file_id, params_json, code_version,"
                " payload_json, computed_at)"
                " VALUES (?, '{}', 'v1', '{}', datetime('now'))", (fid,))
            conn.execute(
                "INSERT INTO analysis_queue (file_id, enqueued_at, status)"
                " VALUES (?, datetime('now'), 'pending')", (fid,))
    conn.close()


def rows(sql, params=()):
    conn = _db.get_connection(DB)
    out = conn.execute(sql, params).fetchall()
    conn.close()
    return out


def path_of(fid):
    r = rows("SELECT path FROM files WHERE id = ?", (fid,))
    return r[0]["path"] if r else None


def n(sql, params=()):
    return rows(sql, params)[0][0]


# -- (d) describe first, and confirm it is read-only --------------------------
seed()
before = [path_of(i) for i in (1, 2, 3, 4)]
info = _db.describe_relocation(OLD, NEW, DB)
check("(d) describe counts every curve under the old root", info["n_files"] == 2)
check("(d) it reports how many of the new paths it can see on disk",
      info["n_found"] == 2 and info["n_missing"] == 0)
check("(d) it shows worked examples of the rewrite",
      info["examples"] and all(p.startswith(N(NEW)) for p in info["examples"]))
check("(d) describing changes NOTHING",
      [path_of(i) for i in (1, 2, 3, 4)] == before)

# -- (a)(b)(c) the move itself ------------------------------------------------
out = _db.relocate_files(OLD, NEW, DB)
check("(a) every curve under the old root is repointed",
      out["n_files"] == 2
      and path_of(1) == N(os.path.join(NEW, "Force0001.ibw"))
      and path_of(2) == N(os.path.join(NEW, "day2", "Force0002.ibw")))
check("(a) the file KEEPS its id, so its analysis is still attached",
      n("SELECT COUNT(*) FROM analysis_results WHERE file_id IN (1,2)") == 2
      and n("SELECT COUNT(*) FROM event_map WHERE file_id IN (1,2)") == 2
      and n("SELECT COUNT(*) FROM analysis_queue WHERE file_id IN (1,2)") == 2)
check("(a) verdicts and hand-set segment picks survive the move",
      n("SELECT COUNT(*) FROM files WHERE id IN (1,2) AND event = 'event'"
        " AND primary_segment_idx = 2") == 2)
check("(b) a curve outside the old root is untouched",
      path_of(3) == N(os.path.join(ELSEWHERE, "Force0003.ibw")))
check("(c) the look-alike folder /curves2 is NOT dragged along by /curves",
      path_of(4) == N(os.path.join(DECOY, "Force0004.ibw")))

# -- (c) the same separator rule, on the filter that scopes a query ----------
# The catalog is filtered by folder in the scanner and the dashboard, and that
# filter is a prefix test too.  A wildcard-based one would over-match on both
# a look-alike sibling and on any root containing _ — which every root here
# does, since they all sit under a *_drive folder.
seed()
LOOKALIKE = os.path.join(tmp, "oldXdrive", "curves")
os.makedirs(LOOKALIKE, exist_ok=True)
conn = _db.get_connection(DB)
with conn:
    conn.execute(
        "INSERT INTO files (id, path, filename, first_seen, last_seen)"
        " VALUES (8, ?, 'Force0008.ibw', datetime('now'), datetime('now'))",
        (N(os.path.join(LOOKALIKE, "Force0008.ibw")),))
conn.close()
scoped = {r["path"] for r in _db.list_files(db_path=DB, directory=OLD)}
check("(c) scoping to a folder returns its files, nested ones included",
      scoped == {N(os.path.join(OLD, "Force0001.ibw")),
                 N(os.path.join(SUB, "Force0002.ibw"))})
check("(c) scoping to /curves does not pull in the sibling /curves2",
      not any("curves2" in p for p in scoped))
check("(c) _ in a folder name is a literal, not a wildcard",
      not any("oldXdrive" in p for p in scoped))
check("(g) no file on disk was created, moved or removed by the repoint",
      os.path.exists(os.path.join(OLD, "Force0001.ibw"))
      and os.path.exists(os.path.join(NEW, "Force0001.ibw")))

# -- (g) repointing at somewhere not mounted yet ------------------------------
seed()
GONE = os.path.join(tmp, "not_mounted", "curves")
info = _db.describe_relocation(OLD, GONE, DB)
check("(g) describe says plainly that none of the new paths can be seen",
      info["n_files"] == 2 and info["n_found"] == 0 and info["n_missing"] == 2)
out = _db.relocate_files(OLD, GONE, DB)
check("(g) the repoint is still allowed — the drive may be mounted later",
      out["n_files"] == 2 and path_of(1) == N(os.path.join(GONE, "Force0001.ibw")))

# -- (e) the same data imported twice -----------------------------------------
seed()
conn = _db.get_connection(DB)
with conn:
    conn.execute(
        "INSERT INTO files (id, path, filename, first_seen, last_seen)"
        " VALUES (9, ?, 'Force0001.ibw', datetime('now'), datetime('now'))",
        (N(os.path.join(NEW, "Force0001.ibw")),))
conn.close()
info = _db.describe_relocation(OLD, NEW, DB)
check("(e) describe warns that one destination is already catalogued",
      info["n_blocked"] == 1 and info["n_files"] == 1)
out = _db.relocate_files(OLD, NEW, DB)
check("(e) the blocked curve keeps its old path — neither entry is discarded",
      out["n_blocked"] == 1
      and path_of(1) == N(os.path.join(OLD, "Force0001.ibw"))
      and path_of(9) == N(os.path.join(NEW, "Force0001.ibw")))
check("(e) the unblocked curve still moves",
      path_of(2) == N(os.path.join(NEW, "day2", "Force0002.ibw")))

# -- (f) re-nesting, where every new path collides with a moving old one ------
seed()
NESTED = os.path.join(OLD, "2026")
out = _db.relocate_files(OLD, NESTED, DB)
check("(f) re-nesting under the old root succeeds",
      out["n_files"] == 2
      and path_of(1) == N(os.path.join(NESTED, "Force0001.ibw"))
      and path_of(2) == N(os.path.join(NESTED, "day2", "Force0002.ibw")))
check("(f) no row is left parked under the collision-avoidance prefix",
      n("SELECT COUNT(*) FROM files WHERE path LIKE char(0) || '%'") == 0)

# -- (h) no-ops ----------------------------------------------------------------
seed()
before = [path_of(i) for i in (1, 2, 3, 4)]
same = _db.relocate_files(OLD, OLD, DB)
none = _db.relocate_files(os.path.join(tmp, "nowhere"), NEW, DB)
check("(h) repointing a root at itself changes nothing",
      same["n_files"] == 0 and [path_of(i) for i in (1, 2, 3, 4)] == before)
check("(h) a root no catalogued curve lives under changes nothing",
      none["n_files"] == 0 and [path_of(i) for i in (1, 2, 3, 4)] == before)

# -- (i) finding out a move happened ------------------------------------------
seed()
os.rename(OLD, OLD + "_moved")
by_dir = {d["path"]: d for d in _db.missing_by_directory(DB)}
check("(i) a folder whose data has moved reports all of its curves missing",
      by_dir[N(OLD)]["n_files"] == 1
      and by_dir[N(OLD)]["n_missing"] == 1
      and by_dir[N(OLD)]["exists"] is False)
check("(i) a folder whose data is still there reports none missing",
      by_dir[N(ELSEWHERE)]["n_missing"] == 0
      and by_dir[N(ELSEWHERE)]["exists"] is True)
check("(i) folders are read off files.path, so every one holding curves lists",
      {N(OLD), N(SUB), N(ELSEWHERE), N(DECOY)} <= set(by_dir))
os.rename(OLD + "_moved", OLD)


# Every check above becomes its own named pytest case.  Must be last:
# pytest_cases reads what the module body recorded.
test_check = checkstyle.pytest_cases(check)
