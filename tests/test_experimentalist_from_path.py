# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Regression test: the scanner may REUSE an experimentalist, never MINT one
(2026-08-05).

`scanner.experimentalist_from_path` reads the depth-1 folder BELOW WHEREVER
ADD DATA WAS AIMED, so the same file yields a different answer per import root:
point at /tank instead of /tank/afm and everybody becomes "afm"; point one
level too deep and a date folder ("260319") becomes a person.  It then wrote
that string onto every file in the leaf.

An invented name is not an error anywhere downstream.  It is a NEW PERSON with
no profile row and no thresholds row, who falls back to Default, and whose
curves are therefore gated differently from the rest of that cohort — one
cohort silently split in two by a folder someone named in lowercase.

The contract under test:
(a) with a `known` map, an exact folder name resolves to that person.
(b) case and surrounding whitespace are folded to the CANONICAL spelling, so
    "anastasiia" and "  ANASTASIIA  " both become the one existing person
    rather than two more.
(c) a name matching nobody returns None — a misspelling, a mis-aimed import
    root, and a genuinely new person are indistinguishable to the scanner, and
    all three are the human's call, so none of them is written.
(d) unset means NULL, never the string "Default": "nobody has said yet" has to
    stay queryable and distinct from "deliberately assigned to Default".  The
    fallback happens when the active criteria profile is selected.
(e) scan_tree writes only matched names, reports the unmatched ones in its
    summary, and leaves those files' experimentalist unset.
(f) known=None preserves the old mint-anything behaviour for direct callers.
"""

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smfs_catalog import db as _db
from smfs_catalog import scanner as _scanner

tmp = tempfile.mkdtemp(prefix="exp_from_path_")
DB = os.path.join(tmp, "test.sqlite")
_db.initialise(DB)

# One shared idiom for these procedural guards — see checkstyle.py for why
# `sys.exit(1)` at the bottom of a file was aborting the whole pytest run.
import checkstyle                                          # noqa: E402

check = checkstyle.CheckRunner()


KNOWN = {"anastasiia": "Anastasiia", "anthony": "Anthony"}
ROOT = "/data/afm"


# ── (a) exact match ──────────────────────────────────────────────────────────
check("(a) exact folder name resolves to that experimentalist",
      _scanner.experimentalist_from_path(
          "/data/afm/Anastasiia/260319", ROOT, KNOWN) == "Anastasiia")

# ── (b) case / whitespace fold to the CANONICAL spelling ─────────────────────
for folder in ("anastasiia", "ANASTASIIA", "AnAsTaSiIa", "  Anastasiia  "):
    check(f"(b) {folder!r} folds to the existing 'Anastasiia', not a new person",
          _scanner.experimentalist_from_path(
              f"/data/afm/{folder}/260319", ROOT, KNOWN) == "Anastasiia")

# ── (c) anything matching nobody is NOT written ──────────────────────────────
# A misspelling, a mis-aimed root and a real new person are indistinguishable
# here on purpose — all three are a human's call.
check("(c) a misspelling mints nothing",
      _scanner.experimentalist_from_path(
          "/data/afm/Anastasia/260319", ROOT, KNOWN) is None)
check("(c) root aimed too SHALLOW does not turn 'afm' into a person",
      _scanner.experimentalist_from_path(
          "/data/afm/Anastasiia/260319", "/data", KNOWN) is None)
check("(c) root aimed too DEEP does not turn a date folder into a person",
      _scanner.experimentalist_from_path(
          "/data/afm/Anastasiia/260319/x", "/data/afm/Anastasiia", KNOWN) is None)
check("(c) a genuinely new person is left for the human",
      _scanner.experimentalist_from_path(
          "/data/afm/Newperson/260319", ROOT, KNOWN) is None)

# ── (f) known=None keeps the old behaviour for direct callers ────────────────
check("(f) known=None still returns the raw folder name",
      _scanner.experimentalist_from_path(
          "/data/afm/Whoever/260319", ROOT) == "Whoever")

# ── known_experimentalists reads the catalog, casefolded ─────────────────────
check("known_experimentalists is empty on a fresh catalog",
      _scanner.known_experimentalists(DB) == {})


# ── (d)+(e) end to end through scan_tree ─────────────────────────────────────
TREE = os.path.join(tmp, "afm")


def make_leaf(*parts, n=2):
    d = os.path.join(TREE, *parts)
    os.makedirs(d, exist_ok=True)
    for i in range(n):
        with open(os.path.join(d, f"Image{i:04d}.ibw"), "wb") as fh:
            fh.write(b"not a real ibw")   # parse failure is irrelevant here
    return d


make_leaf("Celia", "260319")
make_leaf("celia", "260320")            # same person, lowercase folder
make_leaf("Celiaa", "260321")           # typo -> nobody
make_leaf("260322", "session1")         # a DATE where a person should be
                                        # (what a mis-aimed root looks like)
make_leaf("seed", n=1)                  # directly under root: no depth-1 person
                                        # folder exists, so nothing to infer and
                                        # nothing to report

# Seed ONE known person the way a real catalog already would have them.
_scanner.scan_tree(TREE, DB, infer_experimentalist=False)
seed_paths = [r["path"] for r in _db.list_files(db_path=DB) if "seed" in r["path"]]
_db.set_file_descriptors_bulk(seed_paths, {"experimentalist": "Celia"}, DB)

check("known_experimentalists now sees the seeded person",
      _scanner.known_experimentalists(DB) == {"celia": "Celia"})

summary = _scanner.scan_tree(TREE, DB, infer_experimentalist=True, force_rescan=True)

check("(e) summary lists only the experimentalist actually assigned",
      summary["experimentalists"] == ["Celia"])
check("(e) summary reports the folder names it refused to invent",
      summary["unmatched_experimentalists"] == ["260322", "Celiaa"])


def owners_under(folder):
    rows = _db.list_files(db_path=DB)
    sep = os.sep
    return {
        r["experimentalist"]
        for r in rows
        if f"{sep}afm{sep}{folder}{sep}" in r["path"]
    }


def owners_in_session(session):
    rows = _db.list_files(db_path=DB)
    sep = os.sep
    return {
        r["experimentalist"]
        for r in rows
        if f"{sep}{session}{sep}" in r["path"]
    }


check("(e) exact-match folder was assigned",
      owners_under("Celia") == {"Celia"})
check("(e) lowercase folder was assigned the CANONICAL spelling",
      owners_in_session("260320") == {"Celia"})
check("(d) a typo folder leaves experimentalist NULL",
      owners_under("Celiaa") == {None})
check("(d) a date folder leaves experimentalist NULL",
      owners_under("260322") == {None})

# The whole point of NULL over the literal string: the two states stay
# distinguishable, and "what still needs attributing" stays answerable.
all_owners = {r["experimentalist"] for r in _db.list_files(db_path=DB)}
check("(d) unset is NULL, never the string 'Default'",
      None in all_owners and "Default" not in all_owners)


# Every check above becomes its own named pytest case.  Must be last:
# pytest_cases reads what the module body recorded.
test_check = checkstyle.pytest_cases(check)
