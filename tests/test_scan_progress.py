# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Regression test: scan progress reporting and cancellation (#124).

A large registration runs for a long time.  With no progress bar, spinner,
status text or busy cursor there is no way to tell "working" from "hung", and
the scanner has the count available the whole time.

The contract under test:
(a) scan_directory reports progress through a plain callable — the scanner must
    stay Qt-free, so the GUI adapts it on its side.  One call per file, `done`
    counting COMPLETED files (never ahead of the work), `total` the real count,
    and a final tick landing exactly on total.
(b) returning True from the callback cancels: the remaining files are NOT
    scanned, and the call reports cancelled=True.
(c) a cancelled scan KEEPS what it already scanned — those rows are real — but
    are kept, and the next scan finishes the job by skipping them.
    (Undoing the import entirely is the removal dialog's job, #145.)
(d) re-running a completed scan is idempotent — it adds nothing.
(e) cancelling takes effect before the file is touched, so the file count in
    the DB matches the `done` value the callback last saw.
(f) scan_tree reports ONE bar across the whole tree — a total equal to every
    file under it, monotonically increasing, not restarting per folder — and
    propagates cancellation, leaving later folders untouched.
(g) progress_cb is optional everywhere: omitting it changes nothing.

Run with the smfs-catalog env, from the repo root:
    python tests/test_scan_progress.py
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smfs_catalog import db as _db
from smfs_catalog import scanner as _scanner

tmp = tempfile.mkdtemp(prefix="scan_progress_")
DB = os.path.join(tmp, "test.sqlite")
_db.initialise(DB)

# One shared idiom for these procedural guards — see checkstyle.py for why
# `sys.exit(1)` at the bottom of a file was aborting the whole pytest run.
import checkstyle                                          # noqa: E402

check = checkstyle.CheckRunner()


def make_dir(name, n):
    """A directory of n files that look like .ibw but will not parse.

    Parse failure is fine and deliberate here: this test is about the loop's
    progress/cancel bookkeeping, which runs identically either way, and it
    keeps the fixture free of binary blobs.
    """
    d = os.path.join(tmp, name)
    os.makedirs(d, exist_ok=True)
    for i in range(n):
        with open(os.path.join(d, f"Image{i:04d}.ibw"), "wb") as fh:
            fh.write(b"\x00not-a-real-wave")
    return d


def n_files_in_db(directory):
    """How many catalogued files live under `directory` — read off files.path, the one place a location is stored."""
    return len(_db.list_files(db_path=DB, directory=directory))


# -- (a)(d)(g) a full, uncancelled scan ---------------------------------------
D1 = make_dir("full", 10)

seen = []
n_found, n_updated, n_errors, cancelled = _scanner.scan_directory(
    D1, DB, progress_cb=lambda done, total, label="": seen.append(
        (done, total, label)) or False)

check("(a) the callback fired once per file, plus a final tick",
      len(seen) == 11)
check("(a) total is the real file count on every call",
      all(t == 10 for _d, t, _l in seen))
check("(a) done counts COMPLETED files, starting at 0",
      [d for d, _t, _l in seen] == list(range(11)))
check("(a) done never runs ahead of total", all(d <= t for d, t, _l in seen))
check("(a) the last tick lands exactly on total", seen[-1][0] == 10)
check("(a) the label names the file being scanned",
      seen[0][2] == "Image0000.ibw")
check("(a) a completed scan reports cancelled=False", cancelled is False)
check("(a) it found every file", n_found == 10 and n_files_in_db(D1) == 10)
_again = _scanner.scan_directory(D1, DB)
check("(d) re-running a completed scan adds nothing",
      _again[3] is False and n_files_in_db(D1) == 10)

D1B = make_dir("full_nocb", 4)
res = _scanner.scan_directory(D1B, DB)
check("(g) progress_cb is optional — scanning without one still works",
      res[0] == 4 and res[3] is False and n_files_in_db(D1B) == 4)

# -- (b)(c)(e) cancelling mid-scan --------------------------------------------
D2 = make_dir("cancelled", 10)

STOP_AFTER = 4
seen2 = []


def cancel_cb(done, total, label=""):
    seen2.append(done)
    return done >= STOP_AFTER


n_found2, n_updated2, _e2, cancelled2 = _scanner.scan_directory(
    D2, DB, progress_cb=cancel_cb)

check("(b) cancelling reports cancelled=True", cancelled2 is True)
check("(b) the scan stopped early — no further callbacks",
      seen2[-1] == STOP_AFTER)
check("(b) it did NOT scan the remaining files", n_updated2 == STOP_AFTER)
check("(c) what was already scanned is KEPT, not rolled back",
      n_files_in_db(D2) == STOP_AFTER)
check("(e) the DB row count matches the last `done` the callback saw",
      n_files_in_db(D2) == seen2[-1])

# Re-running finishes the job — the point of keeping the partial result.
_f, _u, _e, again_cancelled = _scanner.scan_directory(D2, DB)
check("(c) re-scanning after a cancel completes the directory",
      again_cancelled is False and n_files_in_db(D2) == 10)

# -- (f) scan_tree: one bar across the whole tree -----------------------------
TREE = os.path.join(tmp, "tree")
os.makedirs(TREE, exist_ok=True)
for name, n in (("s1", 5), ("s2", 7), ("s3", 3)):
    d = os.path.join(TREE, name)
    os.makedirs(d, exist_ok=True)
    for i in range(n):
        with open(os.path.join(d, f"Image{i:04d}.ibw"), "wb") as fh:
            fh.write(b"\x00not-a-real-wave")

tree_seen = []
summary = _scanner.scan_tree(
    TREE, DB, progress_cb=lambda done, total, label="": tree_seen.append(
        (done, total, label)) or False)

totals = {t for _d, t, _l in tree_seen}
dones = [d for d, _t, _l in tree_seen]
check("(f) scan_tree reports ONE total for the whole tree, not per folder",
      totals == {15})
check("(f) progress increases monotonically across folders (no restart)",
      all(b >= a for a, b in zip(dones, dones[1:])))
check("(f) progress reaches the grand total", max(dones) == 15)
check("(f) the label names which folder of how many",
      any("Folder 1/3" in l for _d, _t, l in tree_seen))
check("(f) the tree scan completed", summary["cancelled"] is False
      and summary["n_files"] == 15 and summary["n_dirs"] == 3)

# Cancel partway through the tree.
TREE2 = os.path.join(tmp, "tree2")
os.makedirs(TREE2, exist_ok=True)
for name, n in (("a1", 5), ("a2", 5), ("a3", 5)):
    d = os.path.join(TREE2, name)
    os.makedirs(d, exist_ok=True)
    for i in range(n):
        with open(os.path.join(d, f"Image{i:04d}.ibw"), "wb") as fh:
            fh.write(b"\x00not-a-real-wave")

summary2 = _scanner.scan_tree(
    TREE2, DB, progress_cb=lambda done, total, label="": done >= 7)
check("(f) cancelling inside a tree scan propagates out",
      summary2["cancelled"] is True)
check("(f) folders after the cancel were never scanned",
      n_files_in_db(os.path.join(TREE2, "a3")) == 0)


# Every check above becomes its own named pytest case.  Must be last:
# pytest_cases reads what the module body recorded.
test_check = checkstyle.pytest_cases(check)
