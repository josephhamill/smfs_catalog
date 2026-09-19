# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Regression test: per-experimentalist criteria/threshold isolation.

The contract under test:
(a) each experimentalist's set of criteria is independent — and since a bound
    IS the criterion, that is the same fact as (b),
(b) each experimentalist's threshold bounds are independent,
(c) classify() applies the queue's one active owner's bounds to the complete
    cohort, including a mixed-owner queue,
(d) a never-configured experimentalist falls back to the shared
    DEFAULT_EXPERIMENTALIST bucket, not another experimentalist's settings,
(e) "no criteria -> no verdict" is decided once from that active queue
    profile and reported consistently for every path,
(f) the verdict is never persisted — it is derived, so it is
    computed on demand and nothing writes files.hit.

Run with the smfs-catalog env:
    python tests/test_criteria_isolation.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tmpdirs                                              # noqa: E402

from smfs_catalog import db as _db
from smfs_catalog import criteria_gate as _gate

tmp = tmpdirs.mkdtemp(prefix="criteria_iso_")
DB = os.path.join(tmp, "test.sqlite")
_db.initialise(DB)

ALEX_FILE = _db.normalize_path("/tank/testdata/alexandre/Image0001.ibw")
ANA_FILE = _db.normalize_path("/tank/testdata/anastasiia/Image0002.ibw")
DANA_FILE = _db.normalize_path("/tank/testdata/dana/Image0003.ibw")  # never configured

conn = _db.get_connection(DB)
with conn:
    # Experimentalist is file-level, and a file's folder is read off its own
    # path — there is no directory registry to seed.
    for path, who in [
        (ALEX_FILE, "alexandre"), (ANA_FILE, "anastasiia"), (DANA_FILE, "dana"),
    ]:
        conn.execute(
            "INSERT INTO files (path, filename, first_seen, last_seen, event, experimentalist)"
            " VALUES (?, ?, datetime('now'), datetime('now'), 'event', ?)",
            (path, os.path.basename(path), who))
    # test_metric values (a stand-in gate criterion, any ordinary gateable
    # key would do): Alexandre's is a GOOD value (0.99), Anastasiia's a BAD
    # one (0.40), Dana's middling (0.70) — chosen so each experimentalist's
    # OWN bound (below) accepts their own file and nobody else's.
    for path, r2 in [(ALEX_FILE, 0.99), (ANA_FILE, 0.40), (DANA_FILE, 0.70)]:
        fid = conn.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()[0]
        conn.execute(
            "INSERT INTO analysis_results (file_id, analysis_type, value, params_json, code_version, computed_at)"
            " VALUES (?, 'test_metric', ?, '{}', 'test', datetime('now'))",
            (fid, r2))
conn.close()

# Alexandre is the first queue row, so his profile is the one active profile
# governing analysis parameters and criteria for this mixed cohort.
conn = _db.get_connection(DB)
queue_ids = [
    conn.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()[0]
    for path in (ALEX_FILE, ANA_FILE, DANA_FILE)
]
conn.close()
_db.enqueue_files(queue_ids, DB)

# One shared idiom for these procedural guards — see checkstyle.py for why
# `sys.exit(1)` at the bottom of a file was aborting the whole pytest run.
import checkstyle                                          # noqa: E402

check = checkstyle.CheckRunner()


# ── (a)/(b): independent criteria, which are their bounds ───────────────────
_db.set_threshold("test_metric", 0.95, None, "Test Metric", "alexandre", DB)   # his: R2 >= 0.95
_db.set_threshold("test_metric", 0.30, None, "Test Metric", "anastasiia", DB)  # hers: R2 >= 0.30

check("alexandre's criterion is in force, independent of anastasiia's",
      "test_metric" in {c.key for c in _gate.gate("alexandre", DB).criteria})
check("anastasiia's criterion is in force, independent of alexandre's",
      "test_metric" in {c.key for c in _gate.gate("anastasiia", DB).criteria})

alex_bound = _db.get_threshold("test_metric", "alexandre", DB)
ana_bound  = _db.get_threshold("test_metric", "anastasiia", DB)
check("alexandre's own bound is 0.95, not anastasiia's 0.30",
      abs(alex_bound["lower_bound"] - 0.95) < 1e-9)
check("anastasiia's own bound is 0.30, not alexandre's 0.95",
      abs(ana_bound["lower_bound"] - 0.30) < 1e-9)

# Clearing HERS is how her criterion is removed, and it must not touch his.
_db.set_threshold("test_metric", None, None, "Test Metric", "anastasiia", DB)
check("clearing her bounds removed her criterion",
      not _gate.gate("anastasiia", DB))
check("clearing her bounds did not remove his criterion",
      "test_metric" in {c.key for c in _gate.gate("alexandre", DB).criteria})
check("clearing her bounds did not change his bound",
      abs(_db.get_threshold("test_metric", "alexandre", DB)["lower_bound"] - 0.95) < 1e-9)
_db.set_threshold("test_metric", 0.30, None, "Test Metric", "anastasiia", DB)  # restore

# ── (d): never-configured experimentalist falls back to __default__ ─────────
check("dana (never configured) has no criteria of her own -> falls back to empty default",
      _gate.gate("dana", DB).criteria == ())
dana_bound = _db.get_threshold("test_metric", "dana", DB)
check("dana's bound lookup falls back to __default__ (none set there either -> None)",
      dana_bound is None)

# ── (c): one active queue profile governs the complete cohort ───────────────
mixed_paths = [ALEX_FILE, ANA_FILE, DANA_FILE]
cls = _gate.classify(mixed_paths, DB)
hits, non_hits = cls.population(_gate.HIT), cls.population(_gate.NON_HIT)
check("alexandre's file (R2=0.99 >= his 0.95) is a hit",
      ALEX_FILE in hits)
check("anastasiia's file is judged by active owner alexandre's 0.95 bound",
      ANA_FILE in non_hits)
check("dana's file is also judged by active owner alexandre's 0.95 bound",
      DANA_FILE in non_hits)

# Changing an inactive owner's bound cannot change this cohort's verdict.
_db.set_threshold("test_metric", 0.50, None, "Test Metric", "anastasiia", DB)
cls2 = _gate.classify(mixed_paths, DB)
check("changing an inactive owner's bound leaves the complete split unchanged",
      cls2.population(_gate.HIT) == hits
      and cls2.population(_gate.NON_HIT) == non_hits)

reasons = cls2.reasons
check("failure explanations use the active owner's bound for every path",
      set(reasons) == {ANA_FILE, DANA_FILE}
      and all(rs[0][3] == 0.95 for rs in reasons.values()))
_db.set_threshold("test_metric", 0.30, None, "Test Metric", "anastasiia", DB)  # restore

# ── (e): the gate is cohort-wide ────────────────────────────────────────────
# Whether a gate is in force and what it decides must come from the same
# active profile for the complete cohort.
cls3 = _gate.classify(mixed_paths, DB)

check("the active profile puts a gate in force for the whole cohort",
      bool(cls3.gate))
check("the gate in force and the verdict come from one profile",
      cls3.gate.owner == _gate.active_owner(DB))
check("gate availability and evaluation use the same active profile",
      cls3.population(_gate.HIT) == hits
      and cls3.population(_gate.NON_HIT) == non_hits)

# ── (f): the verdict is not persisted ───────────────────────────────────────────
check("restamping is not part of criteria_gate's surface",
      not hasattr(_gate, "restamp"))
check("bulk hit-stamping is not part of db's surface",
      not hasattr(_db, "set_hit_bulk"))

# Column absence guarantees there is nowhere to persist the derived verdict.
conn = _db.get_connection(DB)
cols = {r["name"] for r in conn.execute("PRAGMA table_info(files)")}
conn.close()
check("files.hit does not exist — a stored verdict has nowhere to live",
      "hit" not in cols)


# Every check above becomes its own named pytest case.  Must be last:
# pytest_cases reads what the module body recorded.
test_check = checkstyle.pytest_cases(check)
