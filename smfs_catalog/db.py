# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.


import sqlite3
import os
import sys
import platform
from functools import lru_cache
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np

from .analysis_params import (
    ANALYSIS_PARAM_DEFAULTS,
    ANALYSIS_PARAM_KEYS,
    AnalysisParams,
)

APP_NAME = "smfs_catalog"
DB_FILENAME = "smfs_catalog.db"


def default_db_path() -> str:
    """Resolve the per-user catalog DB location in a cross-platform, distributable way."""
    override = os.environ.get("SMFS_DB_PATH")
    if override:
        return override

    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.path.expanduser("~/.local/share")

    return str(Path(base) / APP_NAME / DB_FILENAME)


DEFAULT_DB_PATH = default_db_path()


@lru_cache(maxsize=None)
def normalize_path(path: str) -> str:
    """Canonical form used as the files.path identity EVERYWHERE (both when the scanner writes a row and when anything later looks a file up or..."""
    return str(Path(path).resolve())


def _this_machine() -> tuple[str, str]:
    """(hostname, os) identifying the machine that owns a DB file."""
    return platform.node() or "unknown", platform.system() or "unknown"


def check_db_machine(db_path: str = DEFAULT_DB_PATH) -> Optional[str]:
    """Return a human-readable warning if this DB was built on a different machine than the one now opening it, else None."""
    if not os.path.exists(db_path):
        return None
    host, os_name = _this_machine()
    try:
        conn = get_connection(db_path)
        rows = dict(conn.execute(
            "SELECT key, value FROM meta WHERE key IN ('built_host','built_os')"
        ).fetchall())
        conn.close()
    except Exception:
        return None
    built_host = rows.get("built_host")
    built_os = rows.get("built_os")
    if built_host and built_host != host:
        return (
            f"This catalog was built on '{built_host}' ({built_os or '?'}) but "
            f"you are on '{host}' ({os_name}).  Stored file paths are per-machine "
            f"— curves will fail to open until you rescan on this machine.  "
            f"Do NOT share a DB between machines; each machine keeps its own."
        )
    return None


def get_connection(db_path: str = DEFAULT_DB_PATH, timeout: float = 30.0) -> sqlite3.Connection:
    """Open (or create) the catalog database."""
    conn = sqlite3.connect(db_path, timeout=timeout)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


_ANALYSIS_RESULTS_SQL = """
    CREATE TABLE IF NOT EXISTS analysis_results (
        file_id        INTEGER NOT NULL REFERENCES files(id),
        analysis_type  TEXT    NOT NULL,
        value          REAL,
        params_json    TEXT    NOT NULL DEFAULT '{}',
        code_version   TEXT,
        computed_at    TEXT    NOT NULL,
        PRIMARY KEY (file_id, analysis_type)
    )
"""
_EVENT_HISTOGRAMS_SQL = """
    CREATE TABLE IF NOT EXISTS event_histograms (
        file_id     INTEGER NOT NULL REFERENCES files(id),
        histogram   BLOB    NOT NULL,
        x_bins      INTEGER NOT NULL,
        f_bins      INTEGER NOT NULL,
        params_json TEXT    NOT NULL DEFAULT '{}',
        computed_at TEXT    NOT NULL,
        PRIMARY KEY (file_id)
    )
"""

_THRESHOLDS_SQL = """
    CREATE TABLE IF NOT EXISTS thresholds (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        experimentalist TEXT    NOT NULL,
        analysis_type   TEXT    NOT NULL,
        lower_bound     REAL,
        upper_bound     REAL,
        label           TEXT,
        created_at      TEXT    NOT NULL,
        UNIQUE (experimentalist, analysis_type)
    )
"""


def initialise(db_path: str = DEFAULT_DB_PATH) -> None:
    """Create all tables if they do not already exist."""
    import json
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    conn = get_connection(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,

                -- The ONE place a file's location is stored.  A path already
                -- contains its parent folders; storing any prefix of it a
                -- second time is storing the same fact twice.
                path                TEXT    NOT NULL UNIQUE,
                filename            TEXT    NOT NULL,

                size_bytes          INTEGER,
                modified_at         TEXT,
                first_seen          TEXT    NOT NULL,
                last_seen           TEXT    NOT NULL,

                spring_constant_pn_nm   REAL,
                velocity_nm_s           REAL,
                force_dist_nm           REAL,
                trigger_point_nn        REAL,
                xpos_um                 REAL,
                ypos_um                 REAL,
                sample_rate_hz          REAL,
                force_filter_bw_hz      REAL,
                inv_ols_nm_v            REAL,
                microscope_model        TEXT,
                measured_date           TEXT,
                measured_at             TEXT,
                dwell_setting           INTEGER,
                indent_mode             INTEGER,

                curve_type          TEXT,

                unusable_reason     TEXT,
                unusable_detail     TEXT,

                content_sha256      TEXT,

                parse_ok            INTEGER NOT NULL DEFAULT 0,
                parse_error         TEXT,

                analyte             TEXT,
                technique           TEXT,
                solvent             TEXT,
                afm_unit            TEXT,
                cantilever          TEXT,
                experimentalist     TEXT,
                notes               TEXT,

                n_traces            INTEGER,
                quality             TEXT,
                analysis_at         TEXT,

                event               TEXT,


                primary_segment_idx           INTEGER,
                secondary_segment_idx         INTEGER,
                segment_override_params_json  TEXT,

                tags                TEXT    DEFAULT '{}'
            )
        """)

        # The column references the table, so it must go first.
        if "directory_id" in {r["name"] for r in
                              conn.execute("PRAGMA table_info(files)")}:
            conn.execute("DROP INDEX IF EXISTS idx_files_directory_id")
            conn.execute("ALTER TABLE files DROP COLUMN directory_id")
        conn.execute("DROP TABLE IF EXISTS watched_directories")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                key     TEXT PRIMARY KEY,
                value   TEXT
            )
        """)
        _host, _os = _this_machine()
        conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('built_host', ?)", (_host,))
        conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('built_os', ?)", (_os,))


        conn.execute(_THRESHOLDS_SQL)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS file_metadata (
                file_id     INTEGER NOT NULL REFERENCES files(id),
                key         TEXT    NOT NULL,
                value_real  REAL,
                value_text  TEXT,
                PRIMARY KEY (file_id, key)
            )
        """)


        conn.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key         TEXT PRIMARY KEY,
                value_real  REAL NOT NULL,
                updated_at  TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS app_settings (
                key         TEXT PRIMARY KEY,
                value_text  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS experimentalist_profiles (
                experimentalist TEXT PRIMARY KEY,
                params_json     TEXT NOT NULL,
                updated_at      TEXT NOT NULL
            )
        """)
        conn.execute("""
            INSERT INTO experimentalist_profiles (experimentalist, params_json, updated_at)
            VALUES (?, json(?), ?)
            ON CONFLICT(experimentalist) DO UPDATE SET
                params_json = json_patch(excluded.params_json,
                                         experimentalist_profiles.params_json),
                updated_at  = excluded.updated_at
        """, (DEFAULT_EXPERIMENTALIST, json.dumps(profile_defaults()), _now()))

        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_queue (
                file_id      INTEGER PRIMARY KEY REFERENCES files(id),
                enqueued_at  TEXT    NOT NULL,
                status       TEXT    NOT NULL DEFAULT 'pending'
            )
        """)

        conn.execute(_ANALYSIS_RESULTS_SQL)

        conn.execute(_EVENT_HISTOGRAMS_SQL)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS event_map (
                file_id      INTEGER NOT NULL REFERENCES files(id),
                params_json  TEXT    NOT NULL DEFAULT '{}',
                code_version TEXT,
                payload_json TEXT    NOT NULL,
                computed_at  TEXT    NOT NULL,
                PRIMARY KEY (file_id, params_json)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS distribution_fits (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                variable        TEXT    NOT NULL,
                units           TEXT    NOT NULL DEFAULT '',
                n_values        INTEGER NOT NULL,
                n_peaks         INTEGER NOT NULL,
                model_label     TEXT    NOT NULL,
                params_json     TEXT    NOT NULL,
                gof_json        TEXT    NOT NULL,
                fit_config_json TEXT    NOT NULL DEFAULT '{}',
                created_at      TEXT    NOT NULL,
                experimentalist TEXT    NOT NULL DEFAULT ''
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS gmm_fits (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                x_variable      TEXT    NOT NULL,
                y_variable      TEXT    NOT NULL,
                n_values        INTEGER NOT NULL,
                k_components    INTEGER NOT NULL,
                cov_type        TEXT    NOT NULL,
                means_json      TEXT    NOT NULL,
                covs_json       TEXT    NOT NULL,
                weights_json    TEXT    NOT NULL,
                gof_json        TEXT    NOT NULL,
                fit_config_json TEXT    NOT NULL DEFAULT '{}',
                created_at      TEXT    NOT NULL,
                experimentalist TEXT    NOT NULL DEFAULT ''
            )
        """)


        for stmt in (
            "CREATE INDEX IF NOT EXISTS idx_files_event          ON files(event)",
            "CREATE INDEX IF NOT EXISTS idx_files_curve_type     ON files(curve_type)",
            "CREATE INDEX IF NOT EXISTS idx_files_measured_date  ON files(measured_date)",
            "CREATE INDEX IF NOT EXISTS idx_files_content_sha256 ON files(content_sha256)",
        ):
            conn.execute(stmt)


    conn.close()


def directory_of(path: str) -> str:
    """The folder holding `path`.  Derived, never stored — see the files.path comment in initialise()."""
    return str(Path(normalize_path(path)).parent)


def find_overlapping_directories(
    path: str, db_path: str = DEFAULT_DB_PATH,
) -> list[tuple[str, str]]:
    """Folders already holding catalogued files that nest with `path` either way — scanning a parent after its children re-walks them."""
    new_p = Path(normalize_path(path))
    out: list[tuple[str, str]] = []
    for row in list_directories(db_path):
        existing = Path(row["path"])
        if existing == new_p:
            continue
        if new_p.is_relative_to(existing):
            out.append((str(existing), "ancestor"))
        elif existing.is_relative_to(new_p):
            out.append((str(existing), "descendant"))
    return out


def list_directories(db_path: str = DEFAULT_DB_PATH) -> list[sqlite3.Row]:
    """Every folder the catalog holds files in, with how many, read off files.path itself."""
    conn = get_connection(db_path)
    rows = conn.execute("""
        SELECT substr(f.path, 1, length(f.path) - length(f.filename) - 1) AS path,
               COUNT(*) AS n_files
        FROM   files f
        GROUP  BY path
        ORDER  BY path
    """).fetchall()
    conn.close()
    return rows


def upsert_file(
    record: dict, db_path: str = DEFAULT_DB_PATH,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    """Insert or update a file record. 'record' is a plain dict with keys matching the files table columns."""
    if "path" in record:
        record = {**record, "path": normalize_path(record["path"])}
    allowed = {
        "path", "filename",
        "size_bytes", "modified_at", "first_seen", "last_seen",
        "spring_constant_pn_nm", "velocity_nm_s", "force_dist_nm",
        "trigger_point_nn", "xpos_um", "ypos_um", "sample_rate_hz",
        "force_filter_bw_hz",
        "inv_ols_nm_v", "microscope_model", "measured_date", "measured_at", "dwell_setting", "indent_mode",
        "curve_type", "unusable_reason", "unusable_detail", "content_sha256",
        "parse_ok", "parse_error",
        "n_traces", "quality", "analyte", "technique", "notes",
        "analysis_at", "tags",
    }
    rec = {k: v for k, v in record.items() if k in allowed}

    columns = ", ".join(rec.keys())
    placeholders = ", ".join("?" for _ in rec)
    updates = ", ".join(
        f"{k} = excluded.{k}"
        for k in rec
        if k not in ("path", "first_seen")
    )

    sql = f"""
        INSERT INTO files ({columns})
        VALUES ({placeholders})
        ON CONFLICT(path) DO UPDATE SET {updates}
    """
    if conn is not None:
        conn.execute(sql, list(rec.values()))
        return
    own = get_connection(db_path)
    try:
        with own:
            own.execute(sql, list(rec.values()))
    finally:
        own.close()


def set_file_descriptors_bulk(
    paths: "list[str]",
    fields: dict,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Apply validated descriptive metadata fields to every supplied path."""
    allowed = {"analyte", "technique", "solvent", "afm_unit", "cantilever", "experimentalist"}
    rec = {k: v for k, v in fields.items() if k in allowed}
    if not rec or not paths:
        return 0

    resolved = [normalize_path(p) for p in paths]
    sets = ", ".join(f"{k} = ?" for k in rec)
    value_params = list(rec.values())

    def _chunks(seq, size=800):
        for i in range(0, len(seq), size):
            yield seq[i:i + size]

    n = 0
    conn = get_connection(db_path)
    try:
        with conn:
            for chunk in _chunks(resolved):
                ph = ",".join("?" * len(chunk))
                cur = conn.execute(
                    f"UPDATE files SET {sets} WHERE path IN ({ph})",
                    value_params + chunk,
                )
                n += cur.rowcount
    finally:
        conn.close()
    return n


_ANALYSIS_TABLES: tuple[str, ...] = (
    "analysis_results", "event_histograms", "event_map",
)
_FILE_CHILD_TABLES: tuple[str, ...] = ("file_metadata", "analysis_queue")


def _sql_chunks(seq: list, size: int = 800):
    """SQLite caps host parameters per statement; chunk any IN (...) list."""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _file_ids_for_paths(
    conn: sqlite3.Connection, paths: list[str],
) -> list[int]:
    """The file ids for `paths`."""
    ids: list[int] = []
    for chunk in _sql_chunks(paths):
        ph = ",".join("?" * len(chunk))
        for row in conn.execute(
            f"SELECT id FROM files WHERE path IN ({ph})", chunk
        ):
            ids.append(row["id"])
    return ids


def _count_by_id(conn: sqlite3.Connection, sql: str, ids: list[int]) -> int:
    """Run a COUNT query whose only placeholder group is `{ph}`, chunked."""
    n = 0
    for chunk in _sql_chunks(ids):
        ph = ",".join("?" * len(chunk))
        n += conn.execute(sql.format(ph=ph), chunk).fetchone()[0]
    return n


def _exec_by_id(conn: sqlite3.Connection, sql: str, ids: list[int]) -> int:
    """Run a DELETE/UPDATE whose only placeholder group is `{ph}`, chunked over `ids`; returns the total rowcount across chunks."""
    n = 0
    for chunk in _sql_chunks(ids):
        ph = ",".join("?" * len(chunk))
        n += conn.execute(sql.format(ph=ph), chunk).rowcount
    return n


def describe_removal_scope(
    paths: list[str],
    db_path: str = DEFAULT_DB_PATH,
) -> dict:
    """What a removal WOULD affect — the numbers the confirmation dialog puts in front of the user before anything is deleted."""
    resolved = [normalize_path(p) for p in paths]
    out = {
        "n_files": 0, "n_classified": 0, "n_events": 0,
        "n_with_fits": 0, "n_queued": 0,
    }
    if not resolved:
        return out

    conn = get_connection(db_path)
    try:
        ids = _file_ids_for_paths(conn, resolved)
        if not ids:
            return out
        out["n_files"] = len(ids)
        out["n_classified"] = _count_by_id(
            conn,
            "SELECT COUNT(*) FROM files WHERE id IN ({ph}) AND event IS NOT NULL",
            ids)
        out["n_events"] = _count_by_id(
            conn,
            "SELECT COUNT(*) FROM files WHERE id IN ({ph}) AND event = 'event'",
            ids)
        out["n_with_fits"] = _count_by_id(
            conn,
            "SELECT COUNT(DISTINCT file_id) FROM event_map WHERE file_id IN ({ph})",
            ids)
        out["n_queued"] = _count_by_id(
            conn,
            "SELECT COUNT(*) FROM analysis_queue WHERE file_id IN ({ph})",
            ids)
    finally:
        conn.close()
    return out


def erase_analysis_for_files(
    paths: list[str],
    db_path: str = DEFAULT_DB_PATH,
) -> dict:
    """Level 2: throw away every computed result for these files, keeping the catalog entries themselves."""
    resolved = [normalize_path(p) for p in paths]
    out: dict = {"n_files": 0}
    out.update({t: 0 for t in _ANALYSIS_TABLES})
    if not resolved:
        return out

    conn = get_connection(db_path)
    try:
        with conn:
            ids = _file_ids_for_paths(conn, resolved)
            if not ids:
                return out
            out["n_files"] = len(ids)
            for table in _ANALYSIS_TABLES:
                out[table] = _exec_by_id(
                    conn, f"DELETE FROM {table} WHERE file_id IN ({{ph}})", ids)
            _exec_by_id(
                conn,
                "UPDATE files SET event = NULL WHERE id IN ({ph})",
                ids)
    finally:
        conn.close()
    return out


def remove_files_from_catalog(
    paths: list[str],
    db_path: str = DEFAULT_DB_PATH,
) -> dict:
    """Level 1: erase these files' analysis AND delete the file rows themselves."""
    resolved = [normalize_path(p) for p in paths]
    out: dict = {"n_files": 0}
    out.update({t: 0 for t in _ANALYSIS_TABLES + _FILE_CHILD_TABLES})
    if not resolved:
        return out

    conn = get_connection(db_path)
    try:
        with conn:
            ids = _file_ids_for_paths(conn, resolved)
            if not ids:
                return out
            out["n_files"] = len(ids)

            for table in _ANALYSIS_TABLES + _FILE_CHILD_TABLES:
                out[table] = _exec_by_id(
                    conn, f"DELETE FROM {table} WHERE file_id IN ({{ph}})", ids)
            _exec_by_id(conn, "DELETE FROM files WHERE id IN ({ph})", ids)
    finally:
        conn.close()
    return out


# ── Repointing the catalog at data that has moved ─────────────────────────────
#
# A curve's path is true for this machine at this moment, and moving the data
# to another drive makes every one of them wrong at once.  Nothing else goes
# wrong: verdicts, ROIs, WLC fits, histograms, queue entries and hand-set
# segment picks all key on files.id.  So the whole repair is to rewrite the
# part of the path that changed, which is why this is two short functions and
# not a subsystem.  describe_relocation says what would happen; only
# relocate_files writes.


def _repointed(path: str, old_root: str, new_root: str) -> Optional[str]:
    """`path` re-expressed under `new_root`, or None when it is not under `old_root`.

    Compared with a separator appended, so /a/b never captures /a/b2 — the
    one way a prefix test quietly does the wrong thing.
    """
    if path == old_root:
        return new_root
    if not path.startswith(old_root + os.sep):
        return None
    return new_root + path[len(old_root):]


def _relocation_plan(
    conn: sqlite3.Connection, old_root: str, new_root: str,
) -> tuple[list[tuple[int, str]], int]:
    """([(file id, new path)], n_blocked) for the move.

    A file whose new path is already held by a DIFFERENT catalog entry — the
    same data imported twice, once at each location — is excluded and counted
    instead.  Merging them would mean choosing which entry's analysis to keep,
    and that is not a choice this should make silently.
    """
    rows = [(r["id"], r["path"]) for r in conn.execute("SELECT id, path FROM files")]
    held = {p: i for i, p in rows}
    plan, blocked = [], 0
    moving = set()
    for fid, path in rows:
        new = _repointed(path, old_root, new_root)
        if new is not None and new != path:
            plan.append((fid, new))
            moving.add(fid)
    for fid, new in plan:
        owner = held.get(new)
        if owner is not None and owner != fid and owner not in moving:
            blocked += 1
    keep = [(fid, new) for fid, new in plan
            if held.get(new) in (None, fid) or held[new] in moving]
    return keep, blocked


def describe_relocation(
    old_root: str,
    new_root: str,
    db_path: str = DEFAULT_DB_PATH,
) -> dict:
    """What repointing WOULD do — the numbers the dialog puts in front of the user before a single row is rewritten."""
    old_root = normalize_path(old_root)
    new_root = normalize_path(new_root)
    out = {
        "old_root": old_root, "new_root": new_root,
        "n_files": 0, "n_found": 0, "n_missing": 0, "n_blocked": 0,
        "examples": [],
    }
    if old_root == new_root:
        return out
    conn = get_connection(db_path)
    try:
        plan, out["n_blocked"] = _relocation_plan(conn, old_root, new_root)
        out["n_files"] = len(plan)
        out["examples"] = [new for _fid, new in plan[:3]]
        for _fid, new in plan:
            if os.path.exists(new):
                out["n_found"] += 1
            else:
                out["n_missing"] += 1
    finally:
        conn.close()
    return out


def relocate_files(
    old_root: str,
    new_root: str,
    db_path: str = DEFAULT_DB_PATH,
) -> dict:
    """Repoint every catalogued curve under `old_root` at `new_root`, keeping each file's id and therefore all of its analysis.

    Nothing on disk is read, moved or deleted — this edits stored paths only.
    Repointing at a drive that is not mounted yet is deliberately allowed;
    describe_relocation is what tells the user how many of the new paths it
    could actually see.
    """
    old_root = normalize_path(old_root)
    new_root = normalize_path(new_root)
    out = {"old_root": old_root, "new_root": new_root, "n_files": 0, "n_blocked": 0}
    if old_root == new_root:
        return out
    conn = get_connection(db_path)
    try:
        with conn:
            plan, out["n_blocked"] = _relocation_plan(conn, old_root, new_root)
            # Two passes, because path is UNIQUE and re-nesting a root
            # under itself makes one moving row's new path another moving
            # row's old one.  A NUL prefix cannot collide with a real path or
            # with a row that is staying put.  Both passes are in the one
            # transaction, so an abort leaves nothing parked.
            for fid, _new in plan:
                conn.execute(
                    "UPDATE files SET path = char(0) || path WHERE id = ?", (fid,))
            for fid, new in plan:
                conn.execute("UPDATE files SET path = ? WHERE id = ?", (new, fid))
            out["n_files"] = len(plan)
    finally:
        conn.close()
    return out


def missing_by_directory(db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """Every folder the catalog holds curves in, and how many of them are not on disk.

    Stats each catalogued path — one syscall per file, and cheaper still when
    the drive is gone, since the folder itself is missing and the walk stops
    there.  An exact count is the point: "1,904 of 6,007 curves cannot be
    found" is actionable where "some are missing" is not.
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute("""
            SELECT substr(f.path, 1, length(f.path) - length(f.filename) - 1) AS dir,
                   f.path
            FROM   files f
            ORDER  BY dir, f.path
        """).fetchall()
    finally:
        conn.close()
    by_dir: dict[str, list[str]] = {}
    for row in rows:
        by_dir.setdefault(row["dir"], []).append(row["path"])
    out = []
    for d, paths in by_dir.items():
        here = os.path.isdir(d)
        out.append({
            "path": d,
            "n_files": len(paths),
            "n_missing": (len(paths) if not here
                          else sum(1 for p in paths if not os.path.exists(p))),
            "exists": here,
        })
    return out


def get_distinct_values(
    column: str,
    table: str = "files",
    db_path: str = DEFAULT_DB_PATH,
    *,
    users: Optional[list] = None,
    techniques: Optional[list] = None,
    analytes: Optional[list] = None,
    solvents: Optional[list] = None,
    afm_units: Optional[list] = None,
) -> list[str]:
    """Return sorted list of distinct non-null values for a given column."""
    _allowed = {
        "files": {
            "curve_type", "experimentalist", "technique", "analyte",
            "solvent", "afm_unit", "cantilever",
        },
    }
    if table not in _allowed or column not in _allowed[table]:
        return []

    clauses = [f"{column} IS NOT NULL AND {column} != ''"]
    params: list = []

    if users:
        ph = ",".join("?" * len(users))
        clauses.append(f"experimentalist IN ({ph})")
        params.extend(users)
    if techniques:
        ph = ",".join("?" * len(techniques))
        clauses.append(f"technique IN ({ph})")
        params.extend(techniques)
    if analytes:
        ph = ",".join("?" * len(analytes))
        clauses.append(f"analyte IN ({ph})")
        params.extend(analytes)
    if solvents:
        ph = ",".join("?" * len(solvents))
        clauses.append(f"solvent IN ({ph})")
        params.extend(solvents)
    if afm_units:
        ph = ",".join("?" * len(afm_units))
        clauses.append(f"afm_unit IN ({ph})")
        params.extend(afm_units)

    where = " AND ".join(clauses)
    conn = get_connection(db_path)
    rows = conn.execute(
        f"SELECT DISTINCT {column} FROM {table} WHERE {where} ORDER BY {column}",
        params,
    ).fetchall()
    conn.close()
    return [row[0] for row in rows]


_FACET_COLUMNS = {
    "users":       "f.experimentalist",
    "analytes":    "f.analyte",
    "solvents":    "f.solvent",
    "afm_units":   "f.afm_unit",
    "curve_types": "f.curve_type",
}


_CANONICAL_IDS_SQL = """
    SELECT id FROM files
     WHERE content_sha256 IS NULL
        OR id = (SELECT MIN(id) FROM files d WHERE d.content_sha256 = files.content_sha256)
"""


def _file_filter_clauses(
    *,
    directory=None, directories=None,
    curve_type=None, curve_types=None, parse_ok=None, quality=None,
    search=None, users=None, analytes=None, solvents=None, afm_units=None,
    date_from=None, date_to=None, k_min=None, k_max=None,
    usable=None, unique=None, exclude_facet=None,
) -> tuple[list[str], list]:
    """Build the one authoritative set of predicates over ``files f``."""
    clauses: list[str] = []
    params: list = []

    def add_many(column: str, values) -> None:
        if values:
            placeholders = ",".join("?" * len(values))
            clauses.append(f"{column} IN ({placeholders})")
            params.extend(values)

    for d in ([directory] if directory else []) + list(directories or []):
        # Compared with a trailing separator, so /a/b never captures
        # /a/b2.  substr rather than LIKE: LIKE reads _ and % in the root as
        # wildcards, and a real folder name may contain either.
        root = normalize_path(d)
        under = root + os.sep
        clauses.append("(f.path = ? OR substr(f.path, 1, ?) = ?)")
        params.extend([root, len(under), under])
    if usable is not None:
        clauses.append(
            "f.unusable_reason IS NULL" if usable
            else "f.unusable_reason IS NOT NULL"
        )
    if unique is not None:
        clauses.append(f"f.id {'' if unique else 'NOT '}IN ({_CANONICAL_IDS_SQL})")
    if curve_type is not None:
        clauses.append("f.curve_type = ?")
        params.append(curve_type)
    if exclude_facet != "curve_types":
        add_many("f.curve_type", curve_types)
    if parse_ok is not None:
        clauses.append("f.parse_ok = ?")
        params.append(1 if parse_ok else 0)
    if quality is not None:
        clauses.append("f.quality = ?")
        params.append(quality)
    if search:
        clauses.append("f.path LIKE ?")
        params.append(f"%{search}%")
    for facet, column, values in (
        ("users", "f.experimentalist", users),
        ("analytes", "f.analyte", analytes),
        ("solvents", "f.solvent", solvents),
        ("afm_units", "f.afm_unit", afm_units),
    ):
        if exclude_facet != facet:
            add_many(column, values)
    if date_from:
        clauses.append("f.measured_date >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("f.measured_date <= ?")
        params.append(date_to)
    if k_min is not None:
        clauses.append("f.spring_constant_pn_nm >= ?")
        params.append(k_min)
    if k_max is not None:
        clauses.append("f.spring_constant_pn_nm <= ?")
        params.append(k_max)
    return clauses, params


def get_facet_options(
    db_path: str = DEFAULT_DB_PATH,
    *,
    users:       Optional[list] = None,
    analytes:    Optional[list] = None,
    solvents:    Optional[list] = None,
    afm_units:   Optional[list] = None,
    curve_types: Optional[list] = None,
    date_from:   Optional[str]  = None,
    date_to:     Optional[str]  = None,
    search:      Optional[str]  = None,
) -> dict[str, list[tuple[str, int]]]:
    """Faceted-search options: for each filter dimension, the distinct values that still have data given the OTHER selected dimensions, each..."""
    out: dict[str, list[tuple[str, int]]] = {}
    conn = get_connection(db_path)
    try:
        for dim, col in _FACET_COLUMNS.items():
            clauses, params = _file_filter_clauses(
                users=users, analytes=analytes, solvents=solvents,
                afm_units=afm_units, curve_types=curve_types,
                date_from=date_from, date_to=date_to, search=search,
                parse_ok=True, usable=True, unique=True,
                exclude_facet=dim,
            )
            clauses.append(f"{col} IS NOT NULL AND {col} != ''")
            where = " AND ".join(clauses)
            rows = conn.execute(f"""
                SELECT {col} AS v, COUNT(*) AS n
                FROM   files f
                WHERE  {where}
                GROUP BY {col}
                ORDER BY {col}
            """, params).fetchall()
            out[dim] = [(r["v"], r["n"]) for r in rows]
    finally:
        conn.close()
    return out


def list_files(
    db_path: str = DEFAULT_DB_PATH,
    directory: Optional[str] = None,
    directories: Optional[list] = None,
    curve_type: Optional[str] = None,
    curve_types: Optional[list] = None,
    parse_ok: Optional[bool] = None,
    quality: Optional[str] = None,
    search: Optional[str] = None,
    users: Optional[list] = None,
    analytes: Optional[list] = None,
    solvents: Optional[list] = None,
    afm_units: Optional[list] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    k_min: Optional[float] = None,
    k_max: Optional[float] = None,
    usable: Optional[bool] = None,
    unique: Optional[bool] = None,
) -> list[sqlite3.Row]:
    """Fetch file records with optional filters."""
    clauses, params = _file_filter_clauses(
        directory=directory, directories=directories,
        curve_type=curve_type, curve_types=curve_types, parse_ok=parse_ok,
        quality=quality, search=search, users=users, analytes=analytes,
        solvents=solvents, afm_units=afm_units,
        date_from=date_from, date_to=date_to, k_min=k_min, k_max=k_max,
        usable=usable, unique=unique,
    )

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    conn = get_connection(db_path)
    rows = conn.execute(f"""
        SELECT f.*,
               substr(f.path, 1, length(f.path) - length(f.filename) - 1) AS dir_path,
               -- Derived, never stored: which OTHER path holds these same
               -- bytes. NULL for a unique file and for the canonical copy of a
               -- duplicated one, so a non-empty cell always means "this row is
               -- the redundant one, and here is the original".
               (SELECT o.path FROM files o
                 WHERE o.content_sha256 = f.content_sha256
                   AND f.content_sha256 IS NOT NULL
                   AND o.id < f.id
                 ORDER BY o.id LIMIT 1) AS duplicate_of
        FROM   files f
        {where}
        ORDER BY f.path
    """, params).fetchall()
    conn.close()
    return rows
def duplicate_groups(db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """Every set of catalogued files sharing one content hash, largest first."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute("""
            SELECT content_sha256 AS sha, id, path FROM files
             WHERE content_sha256 IS NOT NULL
               AND content_sha256 IN (SELECT content_sha256 FROM files
                                       WHERE content_sha256 IS NOT NULL
                                       GROUP BY content_sha256 HAVING COUNT(*) > 1)
             ORDER BY content_sha256, id
        """).fetchall()
    finally:
        conn.close()

    groups: dict[str, list[sqlite3.Row]] = {}
    for r in rows:
        groups.setdefault(r["sha"], []).append(r)
    out = [
        {"sha256": sha, "n": len(rs),
         "canonical": rs[0]["path"],
         "copies": [r["path"] for r in rs[1:]]}
        for sha, rs in groups.items()
    ]
    out.sort(key=lambda g: -g["n"])
    return out


_FILE_REPORT_COLUMNS = (
    "path", "filename", "curve_type", "unusable_reason",
    "experimentalist", "analyte", "solvent", "afm_unit", "cantilever",
    "technique", "measured_date", "measured_at", "microscope_model",
    "spring_constant_pn_nm", "velocity_nm_s", "inv_ols_nm_v",
    "trigger_point_nn", "force_dist_nm", "sample_rate_hz",
    "force_filter_bw_hz",
)
def _classification_report_columns() -> tuple[str, ...]:
    """The report's full header."""
    from .roi_pipeline import SEG_SUMMARY_KEYS

    return _FILE_REPORT_COLUMNS + ("event", "hit") + tuple(SEG_SUMMARY_KEYS)


def classification_report_rows(
    db_path: str = DEFAULT_DB_PATH,
    *,
    select: str = "ultimate",
    **scope_kwargs,
) -> tuple[list[str], list[list]]:
    """Build a classification report: one row per file matching `scope_kwargs` (any list_files() filter — users/analytes/solvents/afm_units/..."""
    rows = list_files(db_path=db_path, **scope_kwargs)
    if not rows:
        return list(_classification_report_columns()), []

    from . import criteria_gate as _gate
    event_paths = [r["path"] for r in rows if r["event"] == "event"]
    hit_by_path: dict[str, str] = {}
    if event_paths:
        has_crit = _gate.has_criteria_checked(event_paths, db_path)
        hits, _non_hits = _gate.evaluate(event_paths, db_path)
        hit_set = set(hits)
        for p in event_paths:
            if has_crit.get(p, False):
                hit_by_path[p] = "hit" if p in hit_set else "non_hit"

    from .roi_pipeline import SEG_SUMMARY_FIELD, SEG_SUMMARY_KEYS, segment_summary_bulk
    seg_by_path: dict[str, dict] = (
        segment_summary_bulk(event_paths, select, db_path) if event_paths else {})

    header = list(_classification_report_columns())
    out: list[list] = []
    for row in rows:
        path = row["path"]
        seg  = seg_by_path.get(path) or {}
        record = [row[c] if c in row.keys() else "" for c in _FILE_REPORT_COLUMNS]
        record.append(row["event"])
        record.append(hit_by_path.get(path, ""))
        record.extend(seg.get(SEG_SUMMARY_FIELD[k]) for k in SEG_SUMMARY_KEYS)
        out.append(["" if v is None else v for v in record])
    return header, out


def get_distinct_dates(
    db_path: str = DEFAULT_DB_PATH,
    *,
    users:       Optional[list] = None,
    analytes:    Optional[list] = None,
    solvents:    Optional[list] = None,
    afm_units:   Optional[list] = None,
    curve_types: Optional[list] = None,
    search:      Optional[str]  = None,
) -> list[tuple[str, int]]:
    """Return [(measured_date, n_files), ...] for dates that have at least one parseable file."""
    clauses = [
        "f.measured_date IS NOT NULL", "f.measured_date != ''", "f.parse_ok = 1",
    ]
    params: list = []
    for col, vals in (
        ("f.experimentalist", users), ("f.analyte", analytes), ("f.solvent", solvents),
        ("f.afm_unit", afm_units), ("f.curve_type", curve_types),
    ):
        if vals:
            ph = ",".join("?" * len(vals))
            clauses.append(f"{col} IN ({ph})")
            params.extend(vals)
    if search:
        clauses.append("f.path LIKE ?")
        params.append(f"%{search}%")
    where = " AND ".join(clauses)

    conn = get_connection(db_path)
    try:
        rows = conn.execute(f"""
            SELECT f.measured_date AS d, COUNT(*) AS n
            FROM   files f
            WHERE  {where}
            GROUP BY f.measured_date
            ORDER BY f.measured_date
        """, params).fetchall()
    finally:
        conn.close()
    return [(row["d"], row["n"]) for row in rows]


def get_measured_dates(
    paths: list[str], db_path: str = DEFAULT_DB_PATH
) -> dict[str, str | None]:
    """Return {path: measured_date} for a list of paths."""
    if not paths:
        return {}
    placeholders = ",".join("?" for _ in paths)
    conn = get_connection(db_path)
    rows = conn.execute(
        f"SELECT path, measured_date FROM files WHERE path IN ({placeholders})",
        paths,
    ).fetchall()
    conn.close()
    return {row["path"]: row["measured_date"] for row in rows}


def get_measured_datetimes(
    paths: list[str], db_path: str = DEFAULT_DB_PATH
) -> dict[str, str | None]:
    """Return {path: best-available acquisition datetime string}."""
    if not paths:
        return {}
    placeholders = ",".join("?" for _ in paths)
    conn = get_connection(db_path)
    rows = conn.execute(
        f"""SELECT path, COALESCE(measured_at, measured_date) AS t
            FROM files WHERE path IN ({placeholders})""",
        paths,
    ).fetchall()
    conn.close()
    return {row["path"]: row["t"] for row in rows}


def get_file_id(
    path: str, db_path: str = DEFAULT_DB_PATH,
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[int]:
    """Return files.id for the given path, or None if the file is not in the DB."""
    path = normalize_path(path)
    c = conn or get_connection(db_path)
    try:
        row = c.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()
    finally:
        if conn is None:
            c.close()
    return row["id"] if row else None


def get_path(
    file_id: int,
    db_path: str = DEFAULT_DB_PATH,
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[str]:
    """Return files.path for the given id, or None if not present."""
    c = conn or get_connection(db_path)
    try:
        row = c.execute(
            "SELECT path FROM files WHERE id = ?", (file_id,)
        ).fetchone()
    finally:
        if conn is None:
            c.close()
    return row["path"] if row else None


def get_analysis_result(
    file_id: int,
    analysis_type: str,
    params_json: str,
    code_version: Optional[str],
    db_path: str = DEFAULT_DB_PATH,
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[float]:
    """Return a cached value only when its legacy code_version column matches the supplied scientific-method identity."""
    if code_version is None:
        return None
    c = conn or get_connection(db_path)
    try:
        row = c.execute("""
            SELECT value, code_version
            FROM   analysis_results
            WHERE  file_id = ? AND analysis_type = ? AND params_json = ?
        """, (file_id, analysis_type, params_json)).fetchone()
    finally:
        if conn is None:
            c.close()
    if row is None:
        return None
    if row["code_version"] != code_version:
        return None
    return row["value"]


def write_analysis_result(
    file_id: int,
    analysis_type: str,
    value: float,
    params_json: str,
    code_version: Optional[str],
    db_path: str = DEFAULT_DB_PATH,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    """Store a derived value for a file."""
    if code_version is None:
        return
    sql = """
        INSERT OR REPLACE INTO analysis_results
            (file_id, analysis_type, value, params_json, code_version, computed_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    args = (file_id, analysis_type, value, params_json, code_version, _now())
    if conn is not None:
        with conn:
            conn.execute(sql, args)
        return
    own = get_connection(db_path)
    try:
        with own:
            own.execute(sql, args)
    finally:
        own.close()


def get_analysis_results_multi(
    file_id: int,
    analysis_types: "list[str]",
    params_json: str,
    code_version: Optional[str],
    db_path: str = DEFAULT_DB_PATH,
    conn: Optional[sqlite3.Connection] = None,
) -> "dict[str, float]":
    """Batched get_analysis_result for several analysis_types under the same (file_id, params_json, code_version) — one connection/query..."""
    if not analysis_types or code_version is None:
        return {}
    c = conn or get_connection(db_path)
    placeholders = ",".join("?" for _ in analysis_types)
    try:
        rows = c.execute(f"""
            SELECT analysis_type, value, code_version
            FROM   analysis_results
            WHERE  file_id = ? AND params_json = ? AND analysis_type IN ({placeholders})
        """, (file_id, params_json, *analysis_types)).fetchall()
    finally:
        if conn is None:
            c.close()
    return {
        r["analysis_type"]: r["value"]
        for r in rows
        if r["code_version"] == code_version
    }


def write_analysis_results_multi(
    file_id: int,
    values: "dict[str, float]",
    params_json: str,
    code_version: Optional[str],
    db_path: str = DEFAULT_DB_PATH,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    """Batched write_analysis_result for several analysis_types under the same (file_id, params_json, code_version) — one..."""
    if not values or code_version is None:
        return
    now  = _now()
    sql  = """
        INSERT OR REPLACE INTO analysis_results
            (file_id, analysis_type, value, params_json, code_version, computed_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    args = [
        (file_id, atype, value, params_json, code_version, now)
        for atype, value in values.items()
    ]
    if conn is not None:
        with conn:
            conn.executemany(sql, args)
        return
    own = get_connection(db_path)
    try:
        with own:
            own.executemany(sql, args)
    finally:
        own.close()


def get_derived_results_bulk_latest(
    paths:          list[str],
    analysis_types: list[str],
    db_path:        str = DEFAULT_DB_PATH,
) -> dict[str, dict[str, tuple[float, str]]]:
    """Bulk-fetch the current derived result per (file, analysis_type), regardless of params_json or code_version."""
    if not paths or not analysis_types:
        return {}

    paths_resolved = [normalize_path(p) for p in paths]
    type_ph = ",".join("?" * len(analysis_types))

    def _chunks(seq, size=800):
        for i in range(0, len(seq), size):
            yield seq[i:i + size]

    conn = get_connection(db_path)
    try:
        path_by_fid: dict[int, str] = {}
        for chunk in _chunks(paths_resolved):
            ph = ",".join("?" * len(chunk))
            for r in conn.execute(
                f"SELECT id, path FROM files WHERE path IN ({ph})", chunk
            ).fetchall():
                path_by_fid[r["id"]] = r["path"]
        if not path_by_fid:
            return {}

        rows = []
        fids = list(path_by_fid.keys())
        for chunk in _chunks(fids):
            fid_ph = ",".join("?" * len(chunk))
            rows += conn.execute(f"""
                SELECT ar.file_id, ar.analysis_type, ar.value, ar.params_json
                FROM   analysis_results ar
                WHERE  ar.analysis_type IN ({type_ph})
                  AND  ar.file_id IN ({fid_ph})
            """, (*analysis_types, *chunk)).fetchall()
    finally:
        conn.close()

    result: dict[str, dict[str, tuple[float, str]]] = {}
    for row in rows:
        p = path_by_fid.get(row["file_id"])
        if p is None:
            continue
        result.setdefault(p, {})[row["analysis_type"]] = (row["value"], row["params_json"])
    return result


def get_queue_analysis_types(db_path: str = DEFAULT_DB_PATH) -> list[str]:
    """Distinct analysis_type keys that have a stored result for any file currently in the analysis queue, sorted alphabetically."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute("""
            SELECT DISTINCT ar.analysis_type
            FROM   analysis_results ar
            JOIN   analysis_queue   q ON q.file_id = ar.file_id
            ORDER  BY ar.analysis_type
        """).fetchall()
    finally:
        conn.close()
    return [r["analysis_type"] for r in rows]


def set_threshold(
    analysis_type: str,
    lower_bound: Optional[float],
    upper_bound: Optional[float],
    label: str = "",
    experimentalist: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Set (or replace) one experimentalist's threshold for one analysis_type."""
    key = experimentalist or DEFAULT_EXPERIMENTALIST
    conn = get_connection(db_path)
    with conn:
        conn.execute("""
            INSERT INTO thresholds
                (experimentalist, analysis_type, lower_bound, upper_bound, label, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(experimentalist, analysis_type) DO UPDATE SET
                lower_bound = excluded.lower_bound,
                upper_bound = excluded.upper_bound,
                label       = excluded.label,
                created_at  = excluded.created_at
        """, (key, analysis_type, lower_bound, upper_bound, label, _now()))
    conn.close()


def get_thresholds(
    experimentalist: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> list[sqlite3.Row]:
    """Effective threshold rows for one experimentalist, ordered by analysis_type: their own bound where they've set one, DEFAULT_PROFILE_..."""
    key = experimentalist or DEFAULT_EXPERIMENTALIST
    conn = get_connection(db_path)
    default_rows = conn.execute(
        "SELECT * FROM thresholds WHERE experimentalist = ?", (DEFAULT_EXPERIMENTALIST,)
    ).fetchall()
    own_rows = [] if key == DEFAULT_EXPERIMENTALIST else conn.execute(
        "SELECT * FROM thresholds WHERE experimentalist = ?", (key,)
    ).fetchall()
    conn.close()
    by_type: dict[str, sqlite3.Row] = {r["analysis_type"]: r for r in default_rows}
    for r in own_rows:
        by_type[r["analysis_type"]] = r
    return sorted(by_type.values(), key=lambda r: r["analysis_type"])


def get_threshold(
    analysis_type: str,
    experimentalist: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[sqlite3.Row]:
    """Return one experimentalist's threshold row for one analysis_type — their own if set, else DEFAULT_EXPERIMENTALIST's shared row, else None."""
    key = experimentalist or DEFAULT_EXPERIMENTALIST
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT * FROM thresholds WHERE experimentalist = ? AND analysis_type = ?",
        (key, analysis_type),
    ).fetchone()
    if row is None and key != DEFAULT_EXPERIMENTALIST:
        row = conn.execute(
            "SELECT * FROM thresholds WHERE experimentalist = ? AND analysis_type = ?",
            (DEFAULT_EXPERIMENTALIST, analysis_type),
        ).fetchone()
    conn.close()
    return row


def get_experimentalists_for_files(
    paths: list[str],
    db_path: str = DEFAULT_DB_PATH,
) -> dict[str, Optional[str]]:
    """Bulk version of get_experimentalist_for_file: {normalized_path: experimentalist_or_None} for every path, in ONE query rather than one..."""
    if not paths:
        return {}
    resolved = [normalize_path(p) for p in paths]
    conn = get_connection(db_path)
    placeholders = ",".join("?" * len(resolved))
    rows = conn.execute(f"""
        SELECT path, experimentalist
        FROM   files
        WHERE  path IN ({placeholders})
    """, resolved).fetchall()
    conn.close()
    out: dict[str, Optional[str]] = {rp: None for rp in resolved}
    for row in rows:
        out[row["path"]] = row["experimentalist"] or None
    return out


def resolve_common_experimentalist(
    paths: list[str],
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[str]:
    """The single experimentalist who owns ALL of `paths`, or None if they span more than one (or none can be resolved) — for UI windows..."""
    owners = set(get_experimentalists_for_files(paths, db_path).values())
    owners.discard(None)
    if len(owners) == 1:
        return next(iter(owners))
    return None


_settings_cache: dict[str, dict[str, float]] = {}


def _invalidate_settings_cache(db_path: str = DEFAULT_DB_PATH) -> None:
    """Drop the cached settings dict for db_path; next read rebuilds it."""
    _settings_cache.pop(db_path, None)


def get_all_settings(db_path: str = DEFAULT_DB_PATH) -> dict[str, float]:
    """Return the entire settings table as {key: value_real}, served from an in-process cache."""
    cached = _settings_cache.get(db_path)
    if cached is not None:
        return cached
    conn = get_connection(db_path)
    rows = conn.execute("SELECT key, value_real FROM settings").fetchall()
    conn.close()
    settings = {row["key"]: float(row["value_real"]) for row in rows}
    _settings_cache[db_path] = settings
    return settings


def get_setting(
    key: str,
    default: float,
    db_path: str = DEFAULT_DB_PATH,
) -> float:
    """Return the stored value for key, or default if the key is not present."""
    return get_all_settings(db_path).get(key, default)


# Compatibility names for the UI/quantity registry.  Both are derived from the
# canonical dataclass, never maintained as parallel declarations.
PARAM_KEYS = ANALYSIS_PARAM_KEYS
PARAM_DEFAULTS = ANALYSIS_PARAM_DEFAULTS

DEFAULT_EXPERIMENTALIST = "Default"


def active_param_owner(db_path: str = DEFAULT_DB_PATH) -> str:
    """Whose parameter set is in force."""
    conn = get_connection(db_path)
    try:
        row = conn.execute("""
            SELECT f.experimentalist
            FROM   analysis_queue q
            JOIN   files f ON f.id = q.file_id
            ORDER  BY q.enqueued_at, q.file_id
            LIMIT  1
        """).fetchone()
    finally:
        conn.close()
    if row and row[0]:
        return str(row[0])
    return DEFAULT_EXPERIMENTALIST


def view_defaults() -> dict[str, float | str]:
    """The 2DH grid settings' seed values — the other half of a profile."""
    from . import event_processor as _ep
    return {
        "wlc_x_bins": float(_ep.WLC_X_BINS),   "wlc_f_bins": float(_ep.WLC_F_BINS),
        "wlc_x_min":  _ep.WLC_X_RANGE[0],      "wlc_x_max":  _ep.WLC_X_RANGE[1],
        "wlc_f_min":  _ep.WLC_F_RANGE[0],      "wlc_f_max":  _ep.WLC_F_RANGE[1],
        "wlc_align_segment": _ep.ALIGN_SEG_DEFAULT,
        "phys_x_bins": float(_ep.PHYS_X_BINS), "phys_f_bins": float(_ep.PHYS_F_BINS),
        "phys_x_min": _ep.PHYS_X_RANGE[0],     "phys_x_max": _ep.PHYS_X_RANGE[1],
        "phys_f_min": _ep.PHYS_F_RANGE[0],     "phys_f_max": _ep.PHYS_F_RANGE[1],
        "phys_f_star": _ep.PHYS_F_STAR_DEFAULT,
        "phys_align_mode":    _ep.PHYS_ALIGN_DEFAULT,
        "phys_align_segment": _ep.ALIGN_SEG_DEFAULT,
    }


def profile_defaults() -> dict[str, float | str]:
    """Everything a brand-new profile starts from: the analysis parameters and the view settings, one dict."""
    return {**PARAM_DEFAULTS, **view_defaults()}


def _materialized_param_set(
    db_path: str,
    updates: Optional[dict[str, float]] = None,
) -> AnalysisParams:
    """Resolve and durably store the active owner's complete analysis set."""
    import json

    conn = get_connection(db_path)
    try:
        # Completion and an optional edit are one write transaction. No other
        # parameter writer can be lost between our read and persistence.
        conn.execute("BEGIN IMMEDIATE")
        owner_row = conn.execute("""
            SELECT f.experimentalist
            FROM   analysis_queue q
            JOIN   files f ON f.id = q.file_id
            ORDER  BY q.enqueued_at, q.file_id
            LIMIT  1
        """).fetchone()
        owner = (str(owner_row[0]) if owner_row and owner_row[0]
                 else DEFAULT_EXPERIMENTALIST)

        def _profile(name: str) -> dict:
            row = conn.execute(
                "SELECT params_json FROM experimentalist_profiles "
                "WHERE experimentalist = ?", (name,),
            ).fetchone()
            if row is None:
                return {}
            try:
                value = json.loads(row[0])
                return value if isinstance(value, dict) else {}
            except (TypeError, ValueError):
                return {}

        profile = _profile(owner)
        shared = {} if owner == DEFAULT_EXPERIMENTALIST else _profile(DEFAULT_EXPERIMENTALIST)
        resolved: dict[str, float] = {}
        for key, dflt in PARAM_DEFAULTS.items():
            for src in (profile, shared):
                try:
                    resolved[key] = float(src[key])
                    break
                except (KeyError, TypeError, ValueError):
                    continue
            else:
                resolved[key] = float(dflt)
        if not any("roi_inner_threshold_nm_per_nm" in src for src in (profile, shared)):
            resolved["roi_inner_threshold_nm_per_nm"] = resolved["roi_threshold_nm_per_nm"]
        for key, value in (updates or {}).items():
            if key not in PARAM_KEYS:
                raise KeyError(f"unknown analysis parameter: {key}")
            resolved[key] = float(value)
        params = AnalysisParams.from_mapping(resolved)

        # Preserve the profile's view settings and any future non-analysis
        # fields while filling/replacing only analysis keys.
        stored = dict(profile)
        changed = False
        for key, value in params.items():
            try:
                same = float(stored[key]) == value
            except (KeyError, TypeError, ValueError):
                same = False
            if not same:
                stored[key] = value
                changed = True
        if changed or not profile:
            conn.execute("""
                INSERT INTO experimentalist_profiles
                    (experimentalist, params_json, updated_at)
                VALUES (?, json(?), ?)
                ON CONFLICT(experimentalist) DO UPDATE SET
                    params_json = excluded.params_json,
                    updated_at  = excluded.updated_at
            """, (owner, json.dumps(stored), _now()))
        conn.commit()
        return params
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def load_analysis_params(db_path: str = DEFAULT_DB_PATH) -> AnalysisParams:
    """THE complete immutable analysis snapshot currently in force."""
    return _materialized_param_set(db_path)


def update_analysis_param(
    key: str,
    value: float,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Write one analysis parameter to the queue owner's set — the one place it lives."""
    _materialized_param_set(db_path, {key: float(value)})


def get_param_set(db_path: str = DEFAULT_DB_PATH) -> dict:
    """The whole parameter set in force, as {key: value} — for display, export and stamping onto results, so the numbers that produced a result..."""
    return load_analysis_params(db_path).as_dict()


def set_setting(
    key: str,
    value: float,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Store (or replace) a setting value."""
    conn = get_connection(db_path)
    with conn:
        conn.execute("""
            INSERT OR REPLACE INTO settings (key, value_real, updated_at)
            VALUES (?, ?, ?)
        """, (key, float(value), _now()))
    conn.close()
    _invalidate_settings_cache(db_path)


APP_SETTING_EXPORT_DIR = "export_dir"


def get_app_setting(
    key: str,
    default: str = "",
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """The stored text value for key, or `default` if the key is absent."""
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT value_text FROM app_settings WHERE key = ?", (key,)
    ).fetchone()
    conn.close()
    return row["value_text"] if row is not None else default


def set_app_setting(
    key: str,
    value: str,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Store (or replace) a DB-wide text setting."""
    conn = get_connection(db_path)
    with conn:
        conn.execute("""
            INSERT OR REPLACE INTO app_settings (key, value_text, updated_at)
            VALUES (?, ?, ?)
        """, (key, str(value), _now()))
    conn.close()


def write_event_histogram(
    file_id:     int,
    histogram:   np.ndarray,
    params_json: str,
    db_path:     str = DEFAULT_DB_PATH,
) -> None:
    """Cache one event's raw uint32 2DH bin counts as a binary blob."""
    x_bins, f_bins = histogram.shape
    blob = histogram.astype(np.uint32).tobytes()
    conn = get_connection(db_path)
    with conn:
        conn.execute("""
            INSERT OR REPLACE INTO event_histograms
                (file_id, histogram, x_bins, f_bins, params_json, computed_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (file_id, blob, x_bins, f_bins, params_json, _now()))
    conn.close()


def write_event_histograms_bulk(
    items:   list,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Write multiple per-event histograms in a single transaction."""
    now = _now()
    rows = []
    for file_id, histogram, params_json in items:
        x_bins, f_bins = histogram.shape
        blob = histogram.astype(np.uint32).tobytes()
        rows.append((file_id, blob, x_bins, f_bins, params_json, now))
    conn = get_connection(db_path)
    with conn:
        conn.executemany("""
            INSERT OR REPLACE INTO event_histograms
                (file_id, histogram, x_bins, f_bins, params_json, computed_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, rows)
    conn.close()


def get_event_histogram(
    file_id:     int,
    params_json: str,
    db_path:     str = DEFAULT_DB_PATH,
    conn:        Optional[sqlite3.Connection] = None,
) -> Optional[np.ndarray]:
    """Return cached histogram array, or None if not yet computed."""
    c = conn or get_connection(db_path)
    try:
        row = c.execute(
            "SELECT histogram, x_bins, f_bins FROM event_histograms "
            "WHERE file_id = ? AND params_json = ?",
            (file_id, params_json),
        ).fetchone()
    finally:
        if conn is None:
            c.close()
    if row is None:
        return None
    return np.frombuffer(row["histogram"], dtype=np.uint32).reshape(
        row["x_bins"], row["f_bins"]
    ).copy()


def write_event_map(
    file_id:      int,
    payload_json: str,
    params_json:  str,
    code_version: str,
    db_path:      str = DEFAULT_DB_PATH,
    conn:         Optional[sqlite3.Connection] = None,
) -> None:
    """Store the multi-event ROI document for one file — as its ONE current registration."""
    if code_version is None:
        return
    c = conn or get_connection(db_path)
    try:
        with c:
            c.execute("DELETE FROM event_map WHERE file_id = ?", (file_id,))
            c.execute("""
                INSERT OR REPLACE INTO event_map
                    (file_id, params_json, code_version, payload_json, computed_at)
                VALUES (?, ?, ?, ?, ?)
            """, (file_id, params_json, code_version, payload_json, _now()))
    finally:
        if conn is None:
            c.close()


def get_event_map(
    file_id:      int,
    params_json:  str,
    code_version: str,
    db_path:      str = DEFAULT_DB_PATH,
    conn:         Optional[sqlite3.Connection] = None,
) -> Optional[str]:
    """Return a cached payload only when its legacy code_version column matches the supplied scientific-method identity."""
    if code_version is None:
        return None
    c = conn or get_connection(db_path)
    try:
        row = c.execute(
            "SELECT payload_json, code_version FROM event_map "
            "WHERE file_id = ? AND params_json = ?",
            (file_id, params_json),
        ).fetchone()
    finally:
        if conn is None:
            c.close()
    if row is None or row["code_version"] != code_version:
        return None
    return row["payload_json"]


def get_latest_event_map(
    file_id: int,
    db_path: str = DEFAULT_DB_PATH,
    conn:    Optional[sqlite3.Connection] = None,
) -> Optional[str]:
    """Return the payload_json of the MOST RECENTLY computed event_map document for this file, across ALL param sets and code versions — or..."""
    c = conn or get_connection(db_path)
    try:
        row = c.execute(
            "SELECT payload_json FROM event_map WHERE file_id = ? "
            "ORDER BY computed_at DESC LIMIT 1",
            (file_id,),
        ).fetchone()
    finally:
        if conn is None:
            c.close()
    return row["payload_json"] if row is not None else None


def get_latest_event_map_params(
    file_id: int,
    db_path: str = DEFAULT_DB_PATH,
    conn:    Optional[sqlite3.Connection] = None,
) -> Optional[str]:
    """params_json of the file's current event_map row, or None if it has none. write_event_map deletes a file's prior row before inserting, so..."""
    c = conn or get_connection(db_path)
    try:
        row = c.execute(
            "SELECT params_json FROM event_map WHERE file_id = ?", (file_id,),
        ).fetchone()
    finally:
        if conn is None:
            c.close()
    return row["params_json"] if row is not None else None


def get_event_map_provenance_bulk(
    paths: list[str], db_path: str = DEFAULT_DB_PATH,
) -> dict[str, dict[str, Optional[str]]]:
    """Producing parameter/code/time identity for each stored event map."""
    if not paths:
        return {}
    resolved = [normalize_path(path) for path in paths]
    out: dict[str, dict[str, Optional[str]]] = {}
    conn = get_connection(db_path)
    try:
        for i in range(0, len(resolved), 500):
            chunk = resolved[i:i + 500]
            marks = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT f.path, e.params_json, e.code_version, e.computed_at "
                f"FROM files f JOIN event_map e ON e.file_id = f.id "
                f"WHERE f.path IN ({marks})",
                chunk,
            ).fetchall()
            for row in rows:
                out[row["path"]] = {
                    "params_json": row["params_json"],
                    "code_version": row["code_version"],
                    "computed_at": row["computed_at"],
                }
    finally:
        conn.close()
    return out


def get_segment_override(
    file_id: int,
    db_path: str = DEFAULT_DB_PATH,
    conn:    Optional[sqlite3.Connection] = None,
) -> dict:
    """{"primary_segment_idx", "secondary_segment_idx", "params_json"} — the raw stored manual pick, or all-None fields if never set."""
    c = conn or get_connection(db_path)
    try:
        row = c.execute(
            "SELECT primary_segment_idx, secondary_segment_idx, "
            "segment_override_params_json FROM files WHERE id = ?",
            (file_id,),
        ).fetchone()
    finally:
        if conn is None:
            c.close()
    if row is None:
        return {"primary_segment_idx": None, "secondary_segment_idx": None, "params_json": None}
    return {
        "primary_segment_idx":   row["primary_segment_idx"],
        "secondary_segment_idx": row["secondary_segment_idx"],
        "params_json":           row["segment_override_params_json"],
    }


def get_segment_overrides_bulk(
    file_ids: "list[int]",
    db_path:  str = DEFAULT_DB_PATH,
    conn:     Optional[sqlite3.Connection] = None,
) -> "dict[int, dict]":
    """Bulk form of get_segment_override — one query, not one per file_id."""
    out = {
        fid: {"primary_segment_idx": None, "secondary_segment_idx": None, "params_json": None}
        for fid in file_ids
    }
    if not file_ids:
        return out
    c = conn or get_connection(db_path)
    try:
        for i in range(0, len(file_ids), 800):
            chunk = file_ids[i:i + 800]
            ph = ",".join("?" * len(chunk))
            for r in c.execute(
                f"SELECT id, primary_segment_idx, secondary_segment_idx, "
                f"segment_override_params_json FROM files WHERE id IN ({ph})",
                chunk,
            ).fetchall():
                out[r["id"]] = {
                    "primary_segment_idx":   r["primary_segment_idx"],
                    "secondary_segment_idx": r["secondary_segment_idx"],
                    "params_json":           r["segment_override_params_json"],
                }
    finally:
        if conn is None:
            c.close()
    return out


def set_primary_segment_idx(
    file_id:     int,
    segment_idx: Optional[int],
    params_json: Optional[str],
    db_path:     str = DEFAULT_DB_PATH,
) -> None:
    """Set (or clear, with segment_idx=None) this file's manually-picked Primary segment."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                "UPDATE files SET primary_segment_idx = ?, "
                "segment_override_params_json = ? WHERE id = ?",
                (segment_idx, params_json, file_id),
            )
    finally:
        conn.close()


def set_secondary_segment_idx(
    file_id:     int,
    segment_idx: Optional[int],
    params_json: Optional[str],
    db_path:     str = DEFAULT_DB_PATH,
) -> None:
    """Set (or clear) this file's manually-picked Secondary segment — see set_primary_segment_idx."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                "UPDATE files SET secondary_segment_idx = ?, "
                "segment_override_params_json = ? WHERE id = ?",
                (segment_idx, params_json, file_id),
            )
    finally:
        conn.close()


def delete_event_map(
    file_id: int,
    db_path: str = DEFAULT_DB_PATH,
    conn:    Optional[sqlite3.Connection] = None,
) -> None:
    """Drop every event_map document (all param sets) for one file."""
    c = conn or get_connection(db_path)
    try:
        with c:
            c.execute("DELETE FROM event_map WHERE file_id = ?", (file_id,))
    finally:
        if conn is None:
            c.close()


def write_file_metadata(
    file_id:  int,
    metadata: dict,
    db_path:  str = DEFAULT_DB_PATH,
    conn:     Optional[sqlite3.Connection] = None,
) -> None:
    """Store key/value pairs from the wave note for one file."""
    if not metadata:
        return

    def _write(c: sqlite3.Connection) -> None:
        for key, value in metadata.items():
            if isinstance(value, str):
                c.execute("""
                    INSERT OR REPLACE INTO file_metadata
                        (file_id, key, value_real, value_text)
                    VALUES (?, ?, NULL, ?)
                """, (file_id, key, value))
            else:
                c.execute("""
                    INSERT OR REPLACE INTO file_metadata
                        (file_id, key, value_real, value_text)
                    VALUES (?, ?, ?, NULL)
                """, (file_id, key, float(value)))

    if conn is not None:
        _write(conn)
        return
    own = get_connection(db_path)
    try:
        with own:
            _write(own)
    finally:
        own.close()


def get_file_metadata(
    file_id: int,
    db_path: str = DEFAULT_DB_PATH,
) -> dict:
    """Return the wave-note metadata stored for one file."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute("""
            SELECT key, value_real, value_text
            FROM file_metadata
            WHERE file_id = ?
        """, (file_id,)).fetchall()
        return {
            row["key"]: (
                row["value_real"]
                if row["value_real"] is not None
                else row["value_text"]
            )
            for row in rows
        }
    finally:
        conn.close()


def get_file_columns(
    paths:   list[str],
    columns: list[str],
    db_path: str = DEFAULT_DB_PATH,
) -> dict[str, dict]:
    """{resolved_path: {column: value}} for arbitrary `files` columns, in bulk."""
    if not paths or not columns:
        return {}
    conn = get_connection(db_path)
    try:
        known = {r["name"] for r in conn.execute("PRAGMA table_info(files)")}
        unknown = [c for c in columns if c not in known]
        if unknown:
            raise ValueError(f"not columns of `files`: {', '.join(sorted(unknown))}")
        resolved = [normalize_path(p) for p in paths]
        out: dict[str, dict] = {}
        select = ", ".join(f'"{c}"' for c in columns)
        for i in range(0, len(resolved), 500):
            chunk = resolved[i:i + 500]
            marks = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT path, {select} FROM files WHERE path IN ({marks})", chunk
            ).fetchall()
            for row in rows:
                out[row["path"]] = {c: row[c] for c in columns}
        return out
    finally:
        conn.close()


def get_experimentalist_for_file(
    file_path: str,
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[str]:
    """Experimentalist who owns file_path, or None when the file is not in the DB or has no experimentalist assigned yet."""
    resolved = normalize_path(file_path)
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT experimentalist FROM files WHERE path = ?", (resolved,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return row["experimentalist"] or None


def get_experimentalist_profile(
    experimentalist: str,
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[dict]:
    """Return the stored parameter profile for an experimentalist as a plain dict, or None if no profile exists."""
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT params_json FROM experimentalist_profiles WHERE experimentalist = ?",
        (experimentalist,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    import json
    try:
        return json.loads(row["params_json"])
    except Exception:
        return None


def set_experimentalist_profile(
    experimentalist: str,
    params_json: str,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Store (or replace) the parameter profile for an experimentalist. params_json is a JSON string whose keys are analysis setting names and..."""
    conn = get_connection(db_path)
    with conn:
        conn.execute("""
            INSERT OR REPLACE INTO experimentalist_profiles (experimentalist, params_json, updated_at)
            VALUES (?, ?, ?)
        """, (experimentalist, params_json, _now()))
    conn.close()


def merge_experimentalist_profile(
    experimentalist: str,
    updates: dict,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Atomically merge `updates` into experimentalist_profiles[experimentalist] in ONE SQL statement (json_patch inside INSERT ..."""
    import json
    conn = get_connection(db_path)
    with conn:
        conn.execute("""
            INSERT INTO experimentalist_profiles (experimentalist, params_json, updated_at)
            VALUES (?, json(?), ?)
            ON CONFLICT(experimentalist) DO UPDATE SET
                params_json = json_patch(experimentalist_profiles.params_json, excluded.params_json),
                updated_at  = excluded.updated_at
        """, (experimentalist, json.dumps(updates), _now()))
    conn.close()


def save_distribution_fit(
    variable:        str,
    units:           str,
    n_values:        int,
    n_peaks:         int,
    model_label:     str,
    params_json:     str,
    gof_json:        str,
    fit_config_json: str,
    db_path:         str = DEFAULT_DB_PATH,
    experimentalist: str = "",
) -> None:
    """Insert one committed distribution fit."""
    conn = get_connection(db_path)
    with conn:
        conn.execute("""
            INSERT INTO distribution_fits
                (variable, units, n_values, n_peaks, model_label,
                 params_json, gof_json, fit_config_json, created_at, experimentalist)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (variable, units, n_values, n_peaks, model_label,
              params_json, gof_json, fit_config_json, _now(), experimentalist))
    conn.close()


def get_distribution_fits(
    variable: str,
    db_path:  str = DEFAULT_DB_PATH,
) -> list[sqlite3.Row]:
    """Return all saved fits for a variable, newest first."""
    conn = get_connection(db_path)
    rows = conn.execute("""
        SELECT * FROM distribution_fits
        WHERE variable = ?
        ORDER BY created_at DESC
    """, (variable,)).fetchall()
    conn.close()
    return rows


def save_gmm_fit(
    x_variable:      str,
    y_variable:      str,
    n_values:        int,
    k_components:    int,
    cov_type:        str,
    means_json:      str,
    covs_json:       str,
    weights_json:    str,
    gof_json:        str,
    fit_config_json: str = "{}",
    db_path:         str = DEFAULT_DB_PATH,
    experimentalist: str = "",
) -> None:
    """Insert one committed 2D GMM fit."""
    conn = get_connection(db_path)
    with conn:
        conn.execute("""
            INSERT INTO gmm_fits
                (x_variable, y_variable, n_values, k_components, cov_type,
                 means_json, covs_json, weights_json, gof_json,
                 fit_config_json, created_at, experimentalist)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (x_variable, y_variable, n_values, k_components, cov_type,
              means_json, covs_json, weights_json, gof_json,
              fit_config_json, _now(), experimentalist))
    conn.close()


def get_gmm_fits(
    x_variable: str,
    y_variable: str,
    db_path:    str = DEFAULT_DB_PATH,
) -> list[sqlite3.Row]:
    """Return all saved GMM fits for the given variable pair, newest first."""
    conn = get_connection(db_path)
    rows = conn.execute("""
        SELECT * FROM gmm_fits
        WHERE x_variable = ? AND y_variable = ?
        ORDER BY created_at DESC
    """, (x_variable, y_variable)).fetchall()
    conn.close()
    return rows


EVENT_VERDICTS = ("event", "non_event", "unavailable", "unusable")


def set_event(
    file_id:        int,
    event: Optional[str],
    db_path:        str = DEFAULT_DB_PATH,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    """Write the stage-1 verdict for one file."""
    if event is not None and event not in EVENT_VERDICTS:
        raise ValueError(f"invalid event verdict: {event!r}")
    c = conn or get_connection(db_path)
    try:
        with c:
            c.execute(
                "UPDATE files SET event = ? WHERE id = ?",
                (event, file_id),
            )
    finally:
        if conn is None:
            c.close()


def set_unusable_reason(
    file_id: int,
    reason:  Optional[str],
    detail:  Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    """Record WHY a file cannot be analysed — or clear it (reason=None)."""
    c = conn or get_connection(db_path)
    try:
        with c:
            c.execute(
                "UPDATE files SET unusable_reason = ?, unusable_detail = ? WHERE id = ?",
                (reason, detail, file_id),
            )
    finally:
        if conn is None:
            c.close()


def clear_analysis_queue(db_path: str = DEFAULT_DB_PATH) -> None:
    """Empty the queue."""
    conn = get_connection(db_path)
    with conn:
        conn.execute("DELETE FROM analysis_queue")
    conn.close()


def enqueue_files(file_ids: list[int], db_path: str = DEFAULT_DB_PATH) -> int:
    """Add files to the queue."""
    if not file_ids:
        return 0
    now = datetime.now().isoformat(timespec="microseconds")
    rows = [(fid, now, "pending") for fid in file_ids]
    conn = get_connection(db_path)
    with conn:
        cur = conn.executemany(
            "INSERT OR IGNORE INTO analysis_queue (file_id, enqueued_at, status) "
            "VALUES (?, ?, ?)",
            rows,
        )
        n = cur.rowcount
    conn.close()
    return n


def dequeue_files(file_ids: list[int], db_path: str = DEFAULT_DB_PATH) -> int:
    """Remove files from the queue."""
    if not file_ids:
        return 0
    placeholders = ",".join("?" * len(file_ids))
    conn = get_connection(db_path)
    with conn:
        cur = conn.execute(
            f"DELETE FROM analysis_queue WHERE file_id IN ({placeholders})",
            file_ids,
        )
        n = cur.rowcount
    conn.close()
    return n


def set_queue_status(
    file_id: int,
    status:  str,
    db_path: str = DEFAULT_DB_PATH,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    """Update the status of one queue entry."""
    if status not in ("pending", "running", "done") and not status.startswith("error: "):
        raise ValueError(f"invalid queue status: {status!r}")
    c = conn or get_connection(db_path)
    try:
        with c:
            c.execute(
                "UPDATE analysis_queue SET status = ? WHERE file_id = ?",
                (status, file_id),
            )
    finally:
        if conn is None:
            c.close()


def queue_paths(db_path: str = DEFAULT_DB_PATH) -> list[str]:
    """The CURRENT queue's file paths, for "Save Queue As…"."""
    return [row["path"] for row in list_queue(db_path)]


def import_queue_from_paths(paths: "list[str]", db_path: str = DEFAULT_DB_PATH) -> "tuple[int, int]":
    """Repopulate the queue from a plain list of paths."""
    conn = get_connection(db_path)
    try:
        resolved = [normalize_path(p) for p in paths]
        found: list[int] = []
        for i in range(0, len(resolved), 800):
            chunk = resolved[i:i + 800]
            ph = ",".join("?" * len(chunk))
            found.extend(
                r["id"] for r in conn.execute(
                    f"SELECT id FROM files WHERE path IN ({ph})", chunk,
                ).fetchall()
            )
    finally:
        conn.close()
    n_enqueued = enqueue_files(found, db_path)
    return n_enqueued, len(paths) - len(found)


def list_queue(db_path: str = DEFAULT_DB_PATH) -> list[sqlite3.Row]:
    """Return queue rows joined with file info, in enqueue order."""
    conn = get_connection(db_path)
    rows = conn.execute("""
        SELECT q.file_id, q.enqueued_at, q.status,
               f.path, f.filename, f.event, f.curve_type,
               f.experimentalist, f.force_filter_bw_hz,
               substr(f.path, 1, length(f.path) - length(f.filename) - 1) AS dir_path
        FROM   analysis_queue q
        JOIN   files f ON f.id = q.file_id
        ORDER BY q.enqueued_at, q.file_id
    """).fetchall()
    conn.close()
    return rows


QUEUE_FRESHNESS = ("fresh", "stale", "new")


def queue_freshness(
    params_json:  str,
    code_version: Optional[str],
    db_path:      str = DEFAULT_DB_PATH,
) -> "dict[int, str]":
    """{file_id: 'fresh'|'stale'|'new'} for every row currently in the queue."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute("""
            SELECT q.file_id            AS file_id,
                   r.params_json        AS stored_params,
                   r.code_version       AS stored_code
            FROM   analysis_queue q
            LEFT JOIN analysis_results r
                   ON r.file_id = q.file_id AND r.analysis_type = 'event'
        """).fetchall()
    finally:
        conn.close()

    out: dict[int, str] = {}
    for r in rows:
        if r["stored_params"] is None:
            out[int(r["file_id"])] = "new"
        elif r["stored_params"] == params_json and r["stored_code"] == code_version:
            out[int(r["file_id"])] = "fresh"
        else:
            out[int(r["file_id"])] = "stale"
    return out


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
