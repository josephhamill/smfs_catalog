#!/usr/bin/env python
# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Rebuild a CLEAN catalog DB — keep only the irreplaceable source-of-truth, drop
all derived/recomputable data and the accumulated bloat.

Why: the derived tables (event_histograms 2DH BLOBs, dead analysis_results
code_versions) can grow to many GB and are all recomputable.  Only the scan
result (file registry + wave-note metadata) and the saved settings/profiles are
expensive/irreplaceable.  So we copy those into a fresh file with the SAME
schema and leave every derived table empty, then VACUUM.

NOTE: This is for reclaiming space WITHOUT a full rescan.  For a clean start
(the schema-9 wipe), you don't need this at all — just delete the DB file and
re-scan; smfs_catalog.db.initialise() builds the correct schema from scratch.

This writes a NEW file and does NOT touch the original.  Verify the result, then
swap it in (rename) yourself.

    python rebuild_clean_db.py                 # default DB -> <default>_clean.db
    python rebuild_clean_db.py OLD.db NEW.db   # explicit paths
"""

import os
import sys
import sqlite3

from smfs_catalog import db as _db

# Source-of-truth tables to carry over (expensive to regenerate).
KEEP = [
    "watched_directories",
    "files",
    "file_metadata",
    "settings",
    "thresholds",
    "experimentalist_profiles",
    "meta",
]
# Derived/session tables recreated by the schema copy but left EMPTY:
#   analysis_results, event_histograms, distribution_fits, gmm_fits, analysis_queue


def main(old: str, new: str) -> None:
    if not os.path.exists(old):
        raise SystemExit(f"Source DB not found: {old}")
    if os.path.exists(new):
        raise SystemExit(f"{new} already exists — remove it first.")

    src = sqlite3.connect(old)
    src.row_factory = sqlite3.Row

    # 1. Copy the full schema (tables + indexes) verbatim from the source, so the
    #    new DB is structurally identical — nothing about the layout changes.
    schema = [
        r[0] for r in src.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'"
        )
    ]
    dst = sqlite3.connect(new)
    for stmt in schema:
        dst.execute(stmt)
    dst.commit()

    # 2. Copy only the keep-tables' rows.
    dst.execute("ATTACH DATABASE ? AS srcdb", (old,))
    src_tables = {r[0] for r in src.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    for t in KEEP:
        if t not in src_tables:
            print(f"  skip {t}: not in source DB")
            continue
        n = dst.execute(f"SELECT COUNT(*) FROM srcdb.{t}").fetchone()[0]
        dst.execute(f"INSERT INTO main.{t} SELECT * FROM srcdb.{t}")
        print(f"  copied {t}: {n} rows")
    dst.commit()

    # 3. Reset the stage-1 verdict (event) and stage-2 gate (hit) to NULL so every
    #    file reads as "not yet analysed" in the clean DB.  Reprocessing repopulates.
    reset = dst.execute("UPDATE main.files SET event = NULL, hit = NULL").rowcount
    print(f"  reset event/hit on {reset} files")
    dst.commit()

    dst.execute("DETACH DATABASE srcdb")
    print("  VACUUM …")
    dst.execute("VACUUM")
    dst.close()
    src.close()

    old_mb = os.path.getsize(old) / 1e6
    new_mb = os.path.getsize(new) / 1e6
    print(f"\nDone.  {old}: {old_mb:,.0f} MB  ->  {new}: {new_mb:,.0f} MB")
    print("Derived tables (analysis_results, event_histograms, distribution_fits,")
    print("gmm_fits, analysis_queue) are EMPTY in the new DB.")
    print("Verify, then rename it over the original, then reprocess.")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        old_path, new_path = sys.argv[1], sys.argv[2]
    else:
        old_path = sys.argv[1] if len(sys.argv) > 1 else _db.DEFAULT_DB_PATH
        base, ext = os.path.splitext(old_path)
        new_path = f"{base}_clean{ext}"
    main(old_path, new_path)
