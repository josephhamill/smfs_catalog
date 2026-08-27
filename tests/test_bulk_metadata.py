# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Regression test for file-level sample metadata and bulk editing.

Descriptive sample metadata (experimentalist/analyte/solvent/afm_unit/
cantilever/technique) moved from watched_directories (one value per whole
registered directory) to file-level columns on `files`, populated by
db.set_file_descriptors_bulk over a scope-selected path list.

The contract under test:
(a) only files whose path is in the given list get written — an
    out-of-scope file (same directory, not in the cohort) is left untouched
    on every field, checked or not.
(b) a field NOT present in the `fields` dict is left untouched on in-scope
    files too — proves "only write checked fields," not "write everything,
    blank included, unconditionally."
(c) an explicit None/empty value for a key that IS present clears that
    field on every in-scope file — proves omission != blankness.
(d) the return value equals the count of files actually matched, including
    across the 800-row chunk boundary.
(e) experimentalist is file-level now: two files under the SAME directory
    can legitimately carry different experimentalist values, and the bulk owner
    accessors preserve
    those distinct values rather than collapsing them per directory.

Run with the smfs-catalog env, from the repo root:
    python tests/test_bulk_metadata.py
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smfs_catalog import db as _db

tmp = tempfile.mkdtemp(prefix="bulk_metadata_")
DB = os.path.join(tmp, "test.sqlite")
_db.initialise(DB)

IN_SCOPE_A = _db.normalize_path("/tank/testdata/bulkmeta/Image0001.ibw")
IN_SCOPE_B = _db.normalize_path("/tank/testdata/bulkmeta/Image0002.ibw")
OUT_OF_SCOPE = _db.normalize_path(
    "/tank/testdata/bulkmeta/Image0003.ibw"
)  # same directory, NOT in cohort

conn = _db.get_connection(DB)
with conn:
    for path in (IN_SCOPE_A, IN_SCOPE_B, OUT_OF_SCOPE):
        conn.execute(
            "INSERT INTO files (path, filename, first_seen, last_seen, solvent)"
            " VALUES (?, ?, datetime('now'), datetime('now'), 'pre-existing-solvent')",
            (path, os.path.basename(path)))
conn.close()

# One shared idiom for these procedural guards — see checkstyle.py for why
# `sys.exit(1)` at the bottom of a file was aborting the whole pytest run.
import checkstyle                                          # noqa: E402

check = checkstyle.CheckRunner()


def _row(path: str) -> dict:
    c = _db.get_connection(DB)
    r = c.execute(
        "SELECT experimentalist, analyte, solvent, afm_unit, cantilever, technique "
        "FROM files WHERE path = ?", (_db.normalize_path(path),)
    ).fetchone()
    c.close()
    return dict(r)

# ── (a)/(b): bulk-write only touches in-scope paths, only touches given keys ─
n = _db.set_file_descriptors_bulk(
    [IN_SCOPE_A, IN_SCOPE_B], {"analyte": "titin", "experimentalist": "anastasiia"}, DB)
check("(d) set_file_descriptors_bulk returns the number of rows matched (2)", n == 2)

a = _row(IN_SCOPE_A)
b = _row(IN_SCOPE_B)
out = _row(OUT_OF_SCOPE)

check("(a) in-scope file A got analyte written", a["analyte"] == "titin")
check("(a) in-scope file B got analyte written", b["analyte"] == "titin")
check("(a) out-of-scope file's analyte is untouched (still NULL)", out["analyte"] is None)
check("(a) out-of-scope file's solvent is untouched (still the pre-existing value)",
      out["solvent"] == "pre-existing-solvent")

check("(b) analyte write did NOT touch solvent on in-scope files (still pre-existing)",
      a["solvent"] == "pre-existing-solvent" and b["solvent"] == "pre-existing-solvent")
check("(b) analyte write did NOT touch cantilever/technique (still NULL, never set)",
      a["cantilever"] is None and a["technique"] is None)

# ── (c): explicit blank value for a PRESENT key clears it ──────────────────
n2 = _db.set_file_descriptors_bulk([IN_SCOPE_A], {"solvent": None}, DB)
check("(c) explicit None for a present key returns 1 row matched", n2 == 1)
a2 = _row(IN_SCOPE_A)
check("(c) explicit None for a present key CLEARS the field", a2["solvent"] is None)
b2 = _row(IN_SCOPE_B)
check("(c) clearing file A's solvent does not affect file B's",
      b2["solvent"] == "pre-existing-solvent")

# ── omitted-fields dict / empty paths: no-ops, not errors ──────────────────
check("empty fields dict is a no-op (returns 0)",
      _db.set_file_descriptors_bulk([IN_SCOPE_A], {}, DB) == 0)
check("empty paths list is a no-op (returns 0)",
      _db.set_file_descriptors_bulk([], {"analyte": "x"}, DB) == 0)
check("unknown keys are silently dropped, not written (returns 0)",
      _db.set_file_descriptors_bulk([IN_SCOPE_A], {"not_a_real_column": "x"}, DB) == 0)

# ── (d) chunk-boundary: > 800 paths in one call ────────────────────────────
many_paths = []
conn = _db.get_connection(DB)
with conn:
    for i in range(1000):
        p = _db.normalize_path(f"/tank/testdata/bulkmeta_chunk/f{i:05d}.ibw")
        conn.execute(
            "INSERT INTO files (path, filename, first_seen, last_seen)"
            " VALUES (?, ?, datetime('now'), datetime('now'))",
            (p, os.path.basename(p)))
        many_paths.append(p)
conn.close()
n_many = _db.set_file_descriptors_bulk(many_paths, {"technique": "SMFS"}, DB)
check("(d) chunked bulk-write over 1000 paths (>800 chunk size) matches all 1000",
      n_many == 1000)

# ── (e): experimentalist is file-level — two files, same directory, different owners ─
_db.set_file_descriptors_bulk([IN_SCOPE_A], {"experimentalist": "anastasiia"}, DB)
_db.set_file_descriptors_bulk([IN_SCOPE_B], {"experimentalist": "anthony"}, DB)

owners = _db.get_experimentalists_for_files([IN_SCOPE_A, IN_SCOPE_B], DB)
check("(e) file A's owner resolves to anastasiia despite sharing a directory with file B",
      owners[_db.normalize_path(IN_SCOPE_A)] == "anastasiia")
check("(e) file B's owner resolves to anthony despite sharing a directory with file A",
      owners[_db.normalize_path(IN_SCOPE_B)] == "anthony")

common = _db.resolve_common_experimentalist([IN_SCOPE_A, IN_SCOPE_B], DB)
check("(e) resolve_common_experimentalist returns None for a genuinely mixed-owner cohort",
      common is None)

# Every check above becomes its own named pytest case.  Must be last:
# pytest_cases reads what the module body recorded.
test_check = checkstyle.pytest_cases(check)
