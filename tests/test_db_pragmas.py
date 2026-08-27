# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Regression test: the per-connection PRAGMAs get_connection must set.

Two PRAGMAs are per-connection in SQLite — they are NOT stored in the database
file header and reset to their defaults on every new connection, unlike
journal_mode=WAL which initialise() sets once and which persists.  So both have
to be re-issued by get_connection, and a refactor that "tidies" either one out
of the hot path would silently revert it with no error anywhere.

The contract under test:
(a) journal_mode is WAL — persistent, set by initialise(), and the precondition
    that makes (b) safe.
(b) synchronous is NORMAL (1), not FULL (2).  FULL forces a platter flush on
    every commit, which dominates the cost of a scan.  NORMAL is durable
    against an application crash in WAL
    mode; only an OS crash or power cut can lose recent transactions, and this
    is a re-runnable catalogue scan.
(c) foreign_keys is ON — the pre-existing per-connection PRAGMA, asserted here
    so it is covered by the same guard rather than only by whatever happens to
    depend on it.
(d) all three hold on a SECOND, independently-opened connection.  This is the
    real point: a per-connection PRAGMA set once by accident (e.g. only inside
    initialise()) would satisfy (a)-(c) on the first connection and silently
    fail on every subsequent one — which is every connection the app actually
    uses.

Run with the smfs-catalog env, from the repo root:
    python tests/test_db_pragmas.py
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smfs_catalog import db as _db

# One shared idiom for these procedural guards — see checkstyle.py for why
# `sys.exit(1)` at the bottom of a file was aborting the whole pytest run.
import checkstyle                                          # noqa: E402

check = checkstyle.CheckRunner()


_tmp = tempfile.mkdtemp(prefix="smfs_pragma_test_")
DB   = os.path.join(_tmp, "pragmas.db")
_db.initialise(DB)


def _pragma(conn, name):
    return conn.execute(f"PRAGMA {name}").fetchone()[0]


# -- first connection ---------------------------------------------------------
conn = _db.get_connection(DB)
jm   = _pragma(conn, "journal_mode")
sync = _pragma(conn, "synchronous")
fk   = _pragma(conn, "foreign_keys")
conn.close()

check("(a) journal_mode is WAL (persistent; set by initialise)",
      str(jm).lower() == "wal")
check(f"(b) synchronous is NORMAL (1), not FULL (2) — got {sync}",
      sync == 1)
check("(c) foreign_keys is ON",
      fk == 1)

# -- second, independent connection -------------------------------------------
# The one that matters: per-connection PRAGMAs must be re-issued every time.
conn2 = _db.get_connection(DB)
jm2   = _pragma(conn2, "journal_mode")
sync2 = _pragma(conn2, "synchronous")
fk2   = _pragma(conn2, "foreign_keys")
conn2.close()

check("(d) journal_mode still WAL on a second connection",
      str(jm2).lower() == "wal")
check(f"(d) synchronous still NORMAL on a second connection — got {sync2}",
      sync2 == 1)
check("(d) foreign_keys still ON on a second connection",
      fk2 == 1)


# Every check above becomes its own named pytest case.  Must be last:
# pytest_cases reads what the module body recorded.
test_check = checkstyle.pytest_cases(check)
