#!/usr/bin/env python
# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Build a catalog with a chosen version's own code.

A problem in an older catalog, or in older code, cannot be reproduced without a
catalog that version built.  This runs a version's own `db.initialise`, scanner
and analysis over a named set of curves, so any tag or commit can be reproduced
on demand instead of a test catalog being assembled by hand with its parameter
sets left out.

What it does, in order:

 1. Reads the test set — scripts/test_catalog_curves.csv, one row per file.
    Each row's path is resolved from the live catalog by its file id and the
    basename checked, because a path written down elsewhere goes stale.
 2. Extracts the chosen version's code from git into a temporary folder.
 3. Runs the build in a child process with that code first on its path:
    create the catalog, copy the parameter sets, register the curves, analyse
    them.
 4. Writes the result to ~/.local/share/smfs_catalog/test_<version>.db and
    deletes the temporary folder.

The test set is a list, not a rule: what lands in the catalog is what is in the
file, so two builds of the same version hold the same curves whatever has since
appeared on disk.  The `known` column records which curves the rupture study
identified as events.  The build never reads it; it is there for whoever reads
the catalog afterwards.

The live catalog is opened read-only and is never copied.

Usage, from the repo root:

    python scripts/build_catalog.py main
    python scripts/build_catalog.py v1.3.0
    python scripts/build_catalog.py main --force             # replace the output
    python scripts/build_catalog.py main --curves other.csv  # a different set
    python scripts/build_catalog.py main --python /path/to/that/versions/python
"""
import argparse
import csv
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_CURVES = os.path.join(HERE, "test_catalog_curves.csv")
OUT_DIR = os.path.expanduser("~/.local/share/smfs_catalog")
DEFAULT_LIVE = os.path.join(OUT_DIR, "smfs_catalog.db")

# The parameter sets: what a method uses to produce a result.  A catalog
# without them analyses under defaults, which is a different experiment.
PARAM_TABLES = ("experimentalist_profiles", "thresholds", "settings",
                "app_settings")

# How the child hands its summary back, on a line of its own.
RESULT = "BUILD-RESULT "


def _ro(path):
    """Connect to `path` read-only.  The live catalog is read in place."""
    con = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
    con.row_factory = sqlite3.Row
    return con


# ── The test set ──────────────────────────────────────────────────────────────

def read_curve_list(curves_path):
    """The test set: one row per file — file id, experimentalist, basename."""
    with open(curves_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("%s lists no curves" % curves_path)
    return rows


def resolve_curves(rows, live_db):
    """Each listed curve's path and owner, read from the live catalog by file id.

    The id is resolved here and the basename checked, so an id that has come to
    mean a different file stops the build rather than quietly building
    something else.  This is why the list stores ids: folders move, and a path
    written down months ago points at nothing.
    """
    con = _ro(live_db)
    try:
        out = []
        for r in rows:
            got = con.execute(
                "SELECT path, experimentalist, segment_override_params_json "
                "FROM files WHERE id = ?", (int(r["file_id"]),)).fetchone()
            if got is None:
                raise SystemExit("file id %s (%s) is not in %s"
                                 % (r["file_id"], r["basename"], live_db))
            if os.path.basename(got["path"]) != r["basename"]:
                raise SystemExit(
                    "file id %s is now %s, not %s"
                    % (r["file_id"], os.path.basename(got["path"]), r["basename"]))
            out.append({
                "file_id":         int(r["file_id"]),
                "experimentalist": got["experimentalist"],
                "path":            got["path"],
                "override":        got["segment_override_params_json"],
            })
        return out
    finally:
        con.close()


def stage(curves, stage_root):
    """A tree of symlinks mirroring each curve's own path under `stage_root`.

    Symlinked, not copied: the data stays where it is.  Mirrored rather than
    flattened so one scan of the tree reaches every folder exactly once.

    The scan is aimed here rather than at the real folders because the scanner
    registers whole folders — ours hold up to 14,354 curves each — and this
    build wants only the curves it chose.
    """
    for path in curves:
        dest = os.path.join(stage_root, path.lstrip(os.sep))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if not os.path.lexists(dest):
            os.symlink(path, dest)
    return stage_root


# ── The chosen version's code ─────────────────────────────────────────────────

def checkout(version, dest):
    """That version's own code, extracted from git into `dest`."""
    os.makedirs(dest, exist_ok=True)
    archive = subprocess.run(["git", "-C", REPO, "archive", version],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if archive.returncode != 0:
        raise SystemExit("git archive %s failed: %s"
                         % (version, archive.stderr.decode().strip()))
    untar = subprocess.run(["tar", "-x", "-C", dest], input=archive.stdout,
                           stderr=subprocess.PIPE)
    if untar.returncode != 0:
        raise SystemExit("extracting %s failed: %s"
                         % (version, untar.stderr.decode().strip()))
    return dest


def _entry_points(scanner, curve_analysis):
    """This version's scan and analyse entry points.

    Registration needs no adapter: both eras expose the same `scan_tree`, and
    v1.3.0's adds its own watched_directories row inside.  Only the analysis
    entry point was renamed.
    """
    if not hasattr(scanner, "scan_tree"):
        raise SystemExit("this version has no scanner.scan_tree; no adapter fits it")
    for name in ("analyse_file", "analyse_and_classify"):
        fn = getattr(curve_analysis, name, None)
        if fn is not None:
            return name, fn
    raise SystemExit(
        "this version has neither curve_analysis.analyse_file nor "
        "curve_analysis.analyse_and_classify; no adapter fits it")


# ── The build, run with that version's code on the path ───────────────────────

def copy_param_tables(live_db, out_db):
    """The parameter sets, copied column-wise from the live catalog.

    Column-wise because the tables are not the same shape in every version:
    a column the target has no room for is reported, not forced in.

    Before the scan, not after: a curve's analysis reads its owner's profile,
    and the owner has to be here before anything is analysed under it.
    """
    src = _ro(live_db)
    dst = sqlite3.connect(out_db)
    report = {"copied": {}, "dropped_columns": {}, "absent_tables": []}
    try:
        have = {r[0] for r in dst.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")}
        for table in PARAM_TABLES:
            if table not in have:
                report["absent_tables"].append(table)
                continue
            src_cols = [r[1] for r in src.execute("PRAGMA table_info(%s)" % table)]
            dst_cols = [r[1] for r in dst.execute("PRAGMA table_info(%s)" % table)]
            use = [c for c in src_cols if c in dst_cols]
            dropped = [c for c in src_cols if c not in dst_cols]
            if dropped:
                report["dropped_columns"][table] = dropped
            rows = src.execute("SELECT %s FROM %s" % (", ".join(use), table)).fetchall()
            dst.executemany(
                "INSERT OR REPLACE INTO %s (%s) VALUES (%s)"
                % (table, ", ".join(use), ", ".join("?" * len(use))),
                [tuple(r) for r in rows])
            report["copied"][table] = len(rows)
        dst.commit()
    finally:
        src.close()
        dst.close()
    return report


def copy_overrides(overrides, out_db):
    """Each chosen curve's own per-file parameter overrides."""
    if not overrides:
        return 0
    con = sqlite3.connect(out_db)
    try:
        n = 0
        for path, payload in overrides.items():
            n += con.execute(
                "UPDATE files SET segment_override_params_json = ? WHERE path = ?",
                (payload, path)).rowcount
        con.commit()
        return n
    finally:
        con.close()


# Three verdicts the app deliberately does not write onto the file:
# "unanalysed" (no pipeline for this modality), "unusable" (the pipeline
# already stored the verdict and its reason) and "unavailable" (the curve could
# not be read — environmental, not a classification).  See
# analysis_worker._analyse_one.
NOT_PUBLISHED = ("unanalysed", "unusable", "unavailable")


def analyse_by_owner(_db, analyse, out_db):
    """Analyse every registered curve under its own owner's parameter set.

    One person at a time, because the set in force is the queue owner's —
    `db.active_param_owner` reads the first queued row — so analysing a mixed
    queue would measure everybody's curves with whoever happened to come first.

    The verdict is published exactly as the app's worker publishes it.
    """
    con = sqlite3.connect(out_db)
    con.row_factory = sqlite3.Row
    try:
        people = {}
        for r in con.execute("SELECT id, path, experimentalist FROM files "
                             "ORDER BY experimentalist, path"):
            people.setdefault(r["experimentalist"], []).append((r["id"], r["path"]))
    finally:
        con.close()

    verdicts = {}
    failures = []
    owners = {}
    published = 0
    for person, rows in sorted(people.items(), key=lambda kv: str(kv[0])):
        _db.clear_analysis_queue(out_db)
        _db.enqueue_files([file_id for file_id, _ in rows], out_db)
        owners[str(person)] = _db.active_param_owner(out_db)
        for file_id, path in rows:
            try:
                verdict, _cached = analyse(file_id, path, out_db)
            except Exception as exc:                            # noqa: BLE001
                failures.append({"path": path,
                                 "error": "%s: %s" % (type(exc).__name__, exc)})
                continue
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
            if verdict not in NOT_PUBLISHED:
                _db.set_event(file_id, verdict, out_db)
                published += 1
    _db.clear_analysis_queue(out_db)
    return {"verdicts": verdicts, "failures": failures, "owners_in_force": owners,
            "published": published,
            "n_files": sum(len(rows) for rows in people.values())}


def _runner(payload_path):
    """Build the catalog, using the code the payload points at.

    Runs as a child process so the version under test owns `smfs_catalog` for
    the whole build; the parent's own copy is never imported here.
    """
    with open(payload_path) as fh:
        job = json.load(fh)
    sys.path.insert(0, job["src"])

    from smfs_catalog import db as _db
    from smfs_catalog import scanner as _scanner
    from smfs_catalog import curve_analysis as _ca

    where = os.path.dirname(os.path.abspath(_db.__file__))
    if not where.startswith(job["src"]):
        raise SystemExit("imported %s, not the checkout at %s" % (where, job["src"]))

    entry_name, analyse = _entry_points(_scanner, _ca)
    result = {"entry_point": entry_name, "code_from": where, "seconds": {}}

    t0 = time.time()
    _db.initialise(job["out"])
    result["seconds"]["create"] = round(time.time() - t0, 1)

    t0 = time.time()
    result["parameters"] = copy_param_tables(job["live"], job["out"])
    result["seconds"]["parameters"] = round(time.time() - t0, 1)

    t0 = time.time()
    result["scan"] = _scanner.scan_tree(job["stage_root"], job["out"],
                                        infer_experimentalist=False)
    result["seconds"]["register"] = round(time.time() - t0, 1)

    owned = 0
    for person, paths in job["people"].items():
        owned += _db.set_file_descriptors_bulk(paths, {"experimentalist": person},
                                               job["out"])
    result["owned"] = owned
    result["overrides"] = copy_overrides(job["overrides"], job["out"])

    t0 = time.time()
    result.update(analyse_by_owner(_db, analyse, job["out"]))
    result["seconds"]["analyse"] = round(time.time() - t0, 1)
    if result["n_files"]:
        result["seconds"]["per_curve"] = round(
            result["seconds"]["analyse"] / result["n_files"], 2)

    print(RESULT + json.dumps(result))


# ── Driving it ────────────────────────────────────────────────────────────────

def _default_out(version):
    return os.path.join(OUT_DIR, "test_%s.db" % re.sub(r"[^A-Za-z0-9._-]", "_", version))


def _clear_output(out, force):
    """Refuse an output that already exists; `--force` replaces it."""
    siblings = [out, out + "-wal", out + "-shm"]
    if os.path.exists(out) and not force:
        raise SystemExit("%s already exists — pass --force to replace it" % out)
    if force:
        for path in siblings:
            if os.path.exists(path):
                os.remove(path)


def build(version, curves_path, live, out, force, python, keep):
    curves = resolve_curves(read_curve_list(curves_path), live)
    paths = [c["path"] for c in curves]
    people = {}
    for c in curves:
        if c["experimentalist"]:
            people.setdefault(c["experimentalist"], []).append(c["path"])
    overrides = {c["path"]: c["override"] for c in curves if c["override"]}

    print("%d curves across %d folders, %d people"
          % (len(curves), len({os.path.dirname(p) for p in paths}), len(people)))

    _clear_output(out, force)
    work = tempfile.mkdtemp(prefix="build_catalog_")
    try:
        stage_root = stage(paths, os.path.join(work, "stage"))
        src = checkout(version, os.path.join(work, "src"))
        payload_path = os.path.join(work, "job.json")
        with open(payload_path, "w") as fh:
            json.dump({"src": src, "out": out, "live": live,
                       "stage_root": stage_root, "people": people,
                       "overrides": overrides}, fh)

        print("building %s with its own code (%s) …" % (version, python))
        proc = subprocess.run([python, "-s", os.path.abspath(__file__),
                               "--runner", payload_path],
                              stdout=subprocess.PIPE)
        text = proc.stdout.decode()
        line = [ln for ln in text.splitlines() if ln.startswith(RESULT)]
        if proc.returncode != 0 or not line:
            sys.stdout.write(text)
            raise SystemExit(
                "the build failed under %s. If this version cannot run on this "
                "interpreter, create its own environment from the checkout's "
                "environment.yml and pass it:\n"
                "    conda env create -p /tmp/env-%s -f <checkout>/environment.yml\n"
                "    python scripts/build_catalog.py %s --python /tmp/env-%s/bin/python"
                % (python, version, version, version))
        report = json.loads(line[-1][len(RESULT):])
    finally:
        if keep:
            print("left in place: %s" % work)
        else:
            shutil.rmtree(work, ignore_errors=True)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("version", nargs="?",
                    help="tag or commit whose code builds the catalog")
    ap.add_argument("--runner", metavar="PAYLOAD",
                    help=argparse.SUPPRESS)
    ap.add_argument("--curves", default=DEFAULT_CURVES,
                    help="the test set, one row per file (default: %(default)s)")
    ap.add_argument("--live", default=DEFAULT_LIVE,
                    help="the live catalog, read-only (default: %(default)s)")
    ap.add_argument("--out", help="output catalog (default: <OUT_DIR>/test_<version>.db)")
    ap.add_argument("--force", action="store_true", help="replace an existing output")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter for the build (default: this one)")
    ap.add_argument("--keep-work", action="store_true",
                    help="keep the temporary checkout and staging tree")
    args = ap.parse_args(argv)

    if args.runner:
        _runner(args.runner)
        return 0
    if not args.version:
        ap.error("a version is required")
    if not os.path.exists(args.live):
        raise SystemExit("no live catalog at %s" % args.live)

    out = args.out or _default_out(args.version)
    report = build(args.version, args.curves, args.live, out, args.force,
                   args.python, args.keep_work)

    print("\n%s" % out)
    print("  built by      %s, analysed with %s"
          % (args.version, report["entry_point"]))
    print("  parameters    %s" % ", ".join(
        "%s %d" % (t, n) for t, n in sorted(report["parameters"]["copied"].items())))
    for table, cols in sorted(report["parameters"]["dropped_columns"].items()):
        print("  no column for %s.%s" % (table, ", ".join(cols)))
    for table in report["parameters"]["absent_tables"]:
        print("  no table      %s in this version" % table)
    print("  curves        %d registered, %d owned, %d with overrides"
          % (report["n_files"], report["owned"], report["overrides"]))
    print("  verdicts      %s  (%d written onto the file)" % (", ".join(
        "%s %d" % (v, n) for v, n in sorted(report["verdicts"].items())),
        report["published"]))
    off = {p: o for p, o in report["owners_in_force"].items() if p != o}
    print("  measured under each curve's own owner: %s"
          % ("yes" if not off else "NO — %s" % off))
    if report["failures"]:
        print("  failed        %d" % len(report["failures"]))
        for f in report["failures"][:5]:
            print("                %s — %s" % (os.path.basename(f["path"]), f["error"]))
    print("  time          %s" % ", ".join(
        "%s %ss" % (k, v) for k, v in report["seconds"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
