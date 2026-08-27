# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/scanner.py
#
# Walks watched directories, parses .ibw files, and writes records to the DB.
# Designed to be importable by the app and also runnable standalone.
#
# Deliberately does NOT import from pysmfs — it reads the .ibw wave note
# directly using igor2, so it works even when pysmfs is not on sys.path.

import hashlib
import io
import os
import re
from pathlib import Path
from datetime import datetime

from igor2.binarywave import load as load_ibw

from . import db
from .curve_loader import _hold_z_sensor, _spring_constant, qualify_wave

# Files per scan transaction.  Commit is what makes a batch durable, so
# this is also the most work an interrupted scan can lose — and that work is
# re-derived by re-reading the same files on the next scan.  200 keeps the
# fsync cost negligible without holding a write transaction open for long.
SCAN_BATCH_SIZE = 200


# ── IBW parsing ───────────────────────────────────────────────────────────────
# These mirror get_params.py but return None on failure instead of raising,
# because in a batch scanner we want to record partial results, not crash.

def _safe_float(pattern: bytes, note: bytes, scale: float = 1.0) -> float | None:
    m = re.search(pattern, note)
    if m is None:
        return None
    try:
        return float(m.group(1)) * scale
    except (ValueError, IndexError):
        return None


def _safe_bool(pattern: bytes, note: bytes) -> int | None:
    m = re.search(pattern, note)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except (ValueError, IndexError):
        return None


def _safe_str(pattern: bytes, note: bytes) -> str | None:
    m = re.search(pattern, note)
    if m is None:
        return None
    try:
        return m.group(1).decode("latin-1", errors="replace").strip()
    except (AttributeError, IndexError):
        return None


# Asylum wave-note Date/Time formats actually observed in this catalogue:
# Image files write ISO ('2026-03-17'); ForceClamp files write the long US form
# ('Thu, Jan 29, 2026').  Time is uniformly 12-hour with AM/PM ('11:14:22 AM',
# also non-padded '9:51:07 AM' — %I accepts both).  Add a format here only when
# real data needs it.
_DATE_FMTS = ("%Y-%m-%d", "%a, %b %d, %Y")
_TIME_FMTS = ("%I:%M:%S %p",)


def _normalize_date_str(date_s: str | None) -> str | None:
    """
    Normalize a wave-note Date field to ISO 'YYYY-MM-DD', trying every format
    in _DATE_FMTS (Image files already write ISO; ForceClamp files write the
    long US form, e.g. 'Thu, Jan 29, 2026'). Falls back to the raw string if
    none match, so an unrecognized format is still recorded, not dropped.
    """
    if not date_s:
        return None
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(date_s, fmt).date().isoformat()
        except ValueError:
            continue
    return date_s


def _measured_at(note: bytes) -> str | None:
    """
    Full acquisition datetime as ISO 'YYYY-MM-DD HH:MM:SS' from the wave note's
    Date + Time fields (e.g. 'Date:Thu, Jan 29, 2026' + 'Time:11:14:22 AM').
    Tries several Date/Time formats; returns None if either field is missing or
    unparseable (the day-only measured_date remains the fallback).
    """
    date_s = _safe_str(rb"\rDate:([^\r]+)\r", note)
    time_s = _safe_str(rb"\rTime:([^\r]+)\r", note)
    if not date_s or not time_s:
        return None
    d = t = None
    for fmt in _DATE_FMTS:
        try:
            d = datetime.strptime(date_s, fmt).date(); break
        except ValueError:
            continue
    for fmt in _TIME_FMTS:
        try:
            t = datetime.strptime(time_s, fmt).time(); break
        except ValueError:
            continue
    if d is None or t is None:
        return None
    return datetime.combine(d, t).strftime("%Y-%m-%d %H:%M:%S")


def _note_float(key: str, note: bytes, scale: float = 1.0) -> float | None:
    """
    Extract a numeric value by key from the wave note.
    Allows unit suffixes (e.g. ' °C') between the number and the line end.
    """
    pattern = (rb"\r" + key.encode() +
               rb": ?(-?[0-9]*\.?[0-9]*(?:[eE][+-]?[0-9]+)?)[^\r]*\r")
    return _safe_float(pattern, note, scale)


def _note_str(key: str, note: bytes) -> str | None:
    """Extract a string value by key from the wave note."""
    pattern = rb"\r" + key.encode() + rb":([^\r]+)\r"
    return _safe_str(pattern, note)


# Keys to extract into file_metadata.
# Format: key → (type, scale)
#   type  : "float" or "str"
#   scale : multiply raw SI value by this (ignored for str)
#
# Units after scaling:
#   velocities → nm/s    distances → nm    sensitivities → nm/V
#   scan size  → µm      temperatures → °C (stripped from wave note)
#   frequencies, Q, dimensionless → as-is
_METADATA_KEYS: dict[str, tuple[str, float]] = {
    # ── Force curve acquisition ────────────────────────────────────────────
    "ApproachVelocity":   ("float", 1e9),    # m/s  → nm/s
    "RetractVelocity":    ("float", 1e9),    # m/s  → nm/s
    "ForceScanRate":      ("float", 1.0),    # Hz
    "DwellTime":          ("float", 1.0),    # s
    "DwellRate":          ("float", 1.0),    # Hz (sampling during dwell)
    "NumPtsPerSec":       ("float", 1.0),    # Hz
    "NumPtsPerWave":      ("float", 1.0),    # points
    "ForceFilterBW":      ("float", 1.0),    # Hz
    "ExtendZ":            ("float", 1e9),    # m   → nm
    "StartDist":          ("float", 1e9),    # m   → nm
    "TriggerChannel":     ("str",   1.0),    # e.g. "Force"
    # ── Cantilever thermal calibration ────────────────────────────────────
    "ThermalFrequency":   ("float", 1.0),    # Hz
    "ThermalQ":           ("float", 1.0),
    "ThermalDC":          ("float", 1.0),    # m²/Hz
    "ThermalWhiteNoise":  ("float", 1.0),    # m²/Hz
    "ThermalTemperature": ("float", 1.0),    # K
    "AmpInvOLS":          ("float", 1e9),    # m/V → nm/V
    "TuneFreqResult":     ("float", 1.0),    # Hz
    "TuneQResult":        ("float", 1.0),
    "TunePeakResult":     ("float", 1.0),
    # ── Environmental ─────────────────────────────────────────────────────
    "StartHeadTemp":      ("float", 1.0),    # °C (suffix stripped by regex)
    "StartScannerTemp":   ("float", 1.0),    # °C
    "FreeAirDeflection":  ("float", 1.0),
    # ── Piezo / LVDT calibration ──────────────────────────────────────────
    "ZLVDTSens":          ("float", 1e9),    # m/V → nm/V
    "ZPiezoSens":         ("float", 1e9),    # m/V → nm/V
    # ── Image-specific ────────────────────────────────────────────────────
    "ScanSize":           ("float", 1e6),    # m   → µm
    "ScanRate":           ("float", 1.0),    # Hz
    "ImagingMode":        ("str",   1.0),    # e.g. "Contact"
}


def _parse_ibw(path: str) -> dict:
    """
    Parse a single .ibw file and return a dict of metadata fields.
    Never raises — errors are recorded in parse_ok / parse_error.
    """
    result = {
        "path":       path,
        "filename":   Path(path).name,
        "parse_ok":   0,
        "parse_error": None,
        # header fields all start as None
        "spring_constant_pn_nm": None,
        "velocity_nm_s":         None,
        "force_dist_nm":         None,
        "trigger_point_nn":      None,
        "xpos_um":               None,
        "ypos_um":               None,
        "sample_rate_hz":        None,
        "force_filter_bw_hz":    None,
        "curve_type":            "unknown",
        "unusable_reason":       None,
        "unusable_detail":       None,
        "content_sha256":        None,
        "inv_ols_nm_v":          None,
        "microscope_model":      None,
        "measured_date":         None,
        "measured_at":           None,
        "dwell_setting":         None,
        "indent_mode":           None,
    }

    try:
        # ONE read of the bytes, used for both the content hash and the parse.
        # igor2 accepts a file-like object, so handing it the buffer we already
        # have costs nothing and avoids reopening the file solely for hashing.
        raw = Path(path).read_bytes()
        result["content_sha256"] = hashlib.sha256(raw).hexdigest()

        wave     = load_ibw(io.BytesIO(raw))
        note     = wave["wave"]["note"]         # bytes
        wdata    = wave["wave"]["wData"]
        header   = wave["wave"]["wave_header"]

        # Spring constant in pN/nm, via the ONE parser curve_loader owns — the
        # column below and the force_extension/unknown call in qualify_wave
        # must be the same number, not two regexes that resemble each other.
        result["spring_constant_pn_nm"] = _spring_constant(note)
        result["velocity_nm_s"] = _safe_float(
            rb"Velocity: ([0-9]*\.?[0-9]*e?[+-]?[0-9]*)\r", note, scale=1e9)
        result["force_dist_nm"] = _safe_float(
            rb"ForceDist: ([0-9]*\.?[0-9]*e?[+-]?[0-9]*)\r", note, scale=1e9)
        # TriggerPoint is stored in Newtons (SI) in the Asylum Research wave note.
        # scale=1e9 converts N → nN.  The column is trigger_point_nn — it is a
        # FORCE (nN), not a distance.  Confirmed: trigger(nN) × (1/k) = max_deflection(nm).
        # E.g. 1 nN / 79.3 pN/nm = 12.6 nm, 4 nN / 79.3 pN/nm = 50.4 nm — both match observed data.
        result["trigger_point_nn"] = _safe_float(
            rb"TriggerPoint: ([0-9]*\.?[0-9]*e?[+-]?[0-9]*)\r", note, scale=1e9)
        result["xpos_um"] = _safe_float(
            rb"XLVDT: ?(-?[0-9]*\.?[0-9]*e?-?[0-9]*)\r", note, scale=1e6)
        result["ypos_um"] = _safe_float(
            rb"YLVDT: ?(-?[0-9]*\.?[0-9]*e?-?[0-9]*)\r", note, scale=1e6)
        result["inv_ols_nm_v"] = _safe_float(
            rb"InvOLS: ?([0-9]*\.?[0-9]*e?[+-]?[0-9]*)\r", note, scale=1e9)
        # The ACQUISITION low-pass the experimentalist set in the AFM software,
        # applied to the deflection channel at capture — the data reaches this
        # app already band-limited to it. It determines whether
        # spectral_cutoff_hz is the narrower filter. Also written to
        # file_metadata via _METADATA_KEYS; the column is the copy
        # that can be a queue column, a scope filter and an export field.
        result["force_filter_bw_hz"] = _note_float("ForceFilterBW", note)
        # Asylum instrument model (e.g. 'MFP3D', 'Infinity') — auto-parsed fact,
        # distinct from the manually-entered afm_unit ("Instrument": which
        # physical unit/color, since several units share one model).
        result["microscope_model"] = _safe_str(
            rb"MicroscopeModel:([^\r]+)\r", note)
        result["dwell_setting"] = _safe_bool(
            rb"DwellSetting: ([0-1])\r", note)
        result["indent_mode"] = _safe_bool(
            rb"IndentMode: ([0-1])\r", note)
        result["measured_date"] = _normalize_date_str(_safe_str(
            rb"\rDate:([^\r]+)\r", note))
        result["measured_at"] = _measured_at(note)

        # Sample rate from wave header sfA[0] (seconds per point)
        try:
            sfa = header["sfA"][0]
            if sfa and sfa > 0:
                result["sample_rate_hz"] = float(1.0 / sfa)
        except (KeyError, IndexError, ZeroDivisionError):
            pass

        # ── Qualification ────────────────────────────────────────────────────
        # Import is where "can we use this file?" gets asked and answered, once
        # — wData is already in memory, so it costs nothing here, and a file
        # that fails never reaches the queue.  curve_type records what the file
        # IS; unusable_reason records why it cannot be analysed (NULL = fine).
        # The two are separate because an aborted force-extension acquisition
        # is still a force-extension acquisition.
        #
        # qualify_wave is shared with curve_loader so import and analysis apply
        # the same content qualification rules.
        q = qualify_wave(
            wdata,
            labels=wave["wave"]["labels"],
            indent_mode=result["indent_mode"],
            hold_z=_hold_z_sensor(note),
            spring_constant=result["spring_constant_pn_nm"],
        )
        result["curve_type"]      = q.curve_type
        result["unusable_reason"] = q.reason
        result["unusable_detail"] = q.detail

        # Extended metadata — stored separately in file_metadata table
        meta = {}
        for key, (kind, scale) in _METADATA_KEYS.items():
            if kind == "float":
                v = _note_float(key, note, scale)
            else:
                v = _note_str(key, note)
            if v is not None:
                meta[key] = v
        result["_file_metadata"] = meta

        result["parse_ok"] = 1

    except Exception as exc:
        result["parse_error"] = str(exc)[:500]

    return result


# ── Filesystem helpers ────────────────────────────────────────────────────────

def _file_meta(path: str) -> dict:
    stat = os.stat(path)
    return {
        "size_bytes":  stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }


# Directory names are not used to suppress suspected duplicates. Duplicate
# content is identified by content hash (see db.duplicate_groups).


def _find_ibw_files(directory: str) -> list[str]:
    """Recursively find every .ibw file under `directory`."""
    found = []
    for root, dirs, files in os.walk(directory):
        for fname in files:
            if fname.lower().endswith(".ibw"):
                # Normalize to the SAME canonical form the DB stores/looks up by
                # (db.normalize_path), so `fpath in existing` and get_file_id()
                # agree on every OS — not just Linux, where '/' happens to match.
                found.append(db.normalize_path(os.path.join(root, fname)))
    return sorted(found)


# ── Tree / parent-scan helpers ────────────────────────────────────────────────
# A "leaf" here is any directory that DIRECTLY contains ≥1 .ibw file.  When you
# point the catalog at a parent (e.g. /tank/afm), each leaf becomes its own
# attributed, filterable unit — because the experimentalist and session live in
# the folder structure (/<root>/<experimentalist>/<session>/…/*.ibw), not in a
# single flat registration.

def leaf_ibw_dirs(root: str) -> list[str]:
    """Every directory at/under `root` that DIRECTLY contains at least one .ibw."""
    leaves: set[str] = set()
    for r, _dirs, files in os.walk(root):
        if any(f.lower().endswith(".ibw") for f in files):
            leaves.add(db.normalize_path(r))
    return sorted(leaves)


def experimentalist_from_path(
    path: str, root: str, known: "dict[str, str] | None" = None,
) -> str | None:
    """
    Infer the experimentalist (data recorder) from where `path` sits under
    `root`: our convention is the depth-1 folder, i.e. /<root>/<experimentalist>/…
    Returns None if `path` isn't under `root` or has no sub-structure (nothing to
    infer from).

    **The scanner may REUSE an existing name; it may never MINT a new one.**
    Pass `known` — {casefolded name: canonical spelling}, from
    known_experimentalists() — and the depth-1 folder is only accepted when it
    already names somebody in the catalog, adopting that person's own spelling
    (so a folder called "anastasiia" resolves to "Anastasiia" rather than
    becoming a second person). Anything else returns None, leaving
    files.experimentalist unset for BulkMetadataDialog to fill in with a human
    looking at it.

    Why: depth-1 is measured from wherever the user aimed Add Data, so the same
    file yields a different answer per import root — point at /tank instead of
    /tank/afm and every experimentalist becomes "afm"; point one level too deep
    and a date folder ("260319") becomes a person. Nothing downstream can tell a
    guessed name from a confirmed one, and an unrecognised name is not an error
    anywhere: it is a new person with no profile and no thresholds, silently
    splitting one cohort in two.

    Unset is deliberately NOT the string "Default". NULL means "nobody has said
    yet" and is queryable as such; writing "Default" would make un-attributed
    files indistinguishable from files deliberately assigned to it. The fallback
    happens at read time when the active criteria profile is selected.
    unknown owner to DEFAULT_EXPERIMENTALIST — so leaving it unset gets that
    behaviour without the ambiguity.

    `known=None` mints from the raw folder name, for callers that genuinely
    want it (and for the existing tests).
    """
    root_n = db.normalize_path(root)
    path_n = db.normalize_path(path)
    try:
        rel = os.path.relpath(path_n, root_n)
    except ValueError:
        return None                      # different drive (Windows) → not under root
    parts = [p for p in rel.split(os.sep) if p not in ("", ".")]
    if len(parts) < 2 or parts[0] == "..":
        return None
    candidate = parts[0]
    if known is None:
        return candidate
    return known.get(candidate.strip().casefold())


def known_experimentalists(db_path: str = db.DEFAULT_DB_PATH) -> "dict[str, str]":
    """
    {casefolded name: canonical spelling} for every experimentalist already in
    the catalog — the set the scanner is allowed to reuse.

    Built fresh per scan rather than cached: a scan can register several trees
    in one session, and a name confirmed via BulkMetadataDialog between them
    should be reusable immediately.
    """
    out: dict[str, str] = {}
    for name in db.get_distinct_values("experimentalist", table="files", db_path=db_path):
        if name and name.strip():
            out.setdefault(name.strip().casefold(), name)
    return out


def scan_tree(
    root:         str,
    db_path:      str  = db.DEFAULT_DB_PATH,
    infer_experimentalist: bool = True,
    force_rescan: bool = False,
    progress_cb=None,
) -> dict:
    """
    Register + scan a whole tree: every leaf directory (see leaf_ibw_dirs) becomes
    its own watched_directory, with the experimentalist inferred from the path
    and written onto every file scanned under it. Experimentalist identity is
    stored at file level.

    progress_cb is the same `(done, total, label) -> cancelled` callable
    scan_directory takes, but reported across the WHOLE tree rather than per
    session folder — one bar that fills once, not one that restarts N times.
    That costs an extra directory walk up front to learn the real total, which
    is seconds against a scan that parses every file it counts.

    Returns a summary dict: n_dirs, n_files, n_errors, experimentalists
    (sorted), cancelled.
    """
    root = db.normalize_path(root)
    leaves = leaf_ibw_dirs(root)
    n_dirs = n_files = n_errors = 0
    experimentalists: set[str] = set()
    # Folder names that looked like an experimentalist but match nobody in the
    # catalog. Reported, never written — see experimentalist_from_path.
    unmatched: set[str] = set()
    known = known_experimentalists(db_path) if infer_experimentalist else {}
    cancelled = False

    # Pre-walk for a real grand total; only when someone is watching.
    grand_total = (
        sum(len(_find_ibw_files(leaf)) for leaf in leaves)
        if progress_cb is not None else 0
    )
    done_before = 0

    for idx, leaf in enumerate(leaves, 1):
        leaf_cb = None
        if progress_cb is not None:
            prefix = f"Folder {idx}/{len(leaves)}: {Path(leaf).name}"

            def leaf_cb(done, total, label="", _o=done_before, _p=prefix):
                return progress_cb(
                    _o + done, grand_total,
                    f"{_p} — {label}" if label else _p)

        nf, _nu, ne, leaf_cancelled = scan_directory(
            leaf, db_path, force_rescan=force_rescan, progress_cb=leaf_cb)
        done_before += nf

        if infer_experimentalist:
            exp = experimentalist_from_path(leaf, root, known)
            if exp:
                experimentalists.add(exp)
                paths = [
                    r["path"] for r in
                    db.list_files(db_path=db_path, directory=leaf)
                ]
                if paths:
                    db.set_file_descriptors_bulk(paths, {"experimentalist": exp}, db_path)
            else:
                # Nobody by that name. Leave files.experimentalist unset and say
                # so — never invent a person from a folder string.
                raw = experimentalist_from_path(leaf, root)
                if raw:
                    unmatched.add(raw)

        n_dirs += 1
        n_files += nf
        n_errors += ne

        if leaf_cancelled:
            # Everything scanned so far is committed and kept; this folder is
            # left unmarked, and the folders after it were never touched.
            cancelled = True
            break

    return {
        "n_dirs": n_dirs,
        "n_files": n_files,
        "n_errors": n_errors,
        "experimentalists": sorted(experimentalists),
        "unmatched_experimentalists": sorted(unmatched),
        "cancelled": cancelled,
    }


# ── Main scan entry points ────────────────────────────────────────────────────

def scan_directory(
    directory_path: str,
    db_path:        str  = db.DEFAULT_DB_PATH,
    force_rescan:   bool = False,
    progress_cb=None,
) -> tuple[int, int, int, bool]:
    """
    Scan one watched directory and upsert all .ibw files into the DB.

    Parameters
    ----------
    directory_path : str
        Absolute path to the directory.
    db_path : str
        Path to the catalog database.
    force_rescan : bool
        If False, skip files whose modified_at timestamp hasn't changed.
    progress_cb : callable or None
        `progress_cb(done, total, label) -> bool`, called once per file with
        the number completed so far; return True to cancel.  This is a
        PLAIN CALLABLE on purpose — the scanner must not import Qt, so the GUI
        adapts it to a progress dialog on its side (see add_data_dialog.
        _ScanProgress) and this module stays usable headless and from tests.
        `label` is the bare filename; composing a fuller line is the caller's
        job, since it is the only one that knows whether this is one directory
        or the Nth session folder of a tree.

    Returns
    -------
    (n_found, n_new_or_updated, n_errors, cancelled)

    On cancel the files already scanned are COMMITTED and kept — they are
    real, and re-deriving them costs another walk.  The next scan finishes the
    job: it skips what is already recorded with an unchanged mtime.  Undoing a
    mistaken import is the removal dialog's job, not something a half-finished
    scan should do silently.
    """
    if not os.path.isdir(directory_path):
        # A disconnected drive is not a fact about the files — report nothing
        # scanned and let the caller decide; the next scan finds them again.
        return 0, 0, 0, False

    ibw_files = _find_ibw_files(directory_path)
    n_found   = len(ibw_files)
    n_updated = 0
    n_errors  = 0
    now       = datetime.now().isoformat(timespec="seconds")

    # Fetch existing records for this directory to check timestamps
    existing = {
        row["path"]: row
        for row in db.list_files(db_path=db_path, directory=directory_path)
    }

    # One connection and one transaction per SCAN_BATCH_SIZE files, not one
    # of each per file.  A commit under synchronous=FULL flushes to the
    # platter, and closing the last connection to a WAL database checkpoints
    # it — both correct SQLite behaviour for a usage pattern it does not
    # expect.  A connection is meant to be long-lived.
    #
    # Crash behaviour is deliberate and unchanged in kind: an interrupted scan
    # loses at most the current batch, which the next scan re-derives by reading
    # the same files.
    conn      = db.get_connection(db_path)
    pending   = 0
    cancelled = False
    try:
        for done, fpath in enumerate(ibw_files):
            # Both checked at the top so the `continue` below cannot skip
            # them.  `done` is the number of files already finished, so
            # cancelling takes effect BEFORE this one is touched and the
            # count reported is never ahead of the work.
            if pending >= SCAN_BATCH_SIZE:
                conn.commit()
                pending = 0

            if progress_cb is not None and progress_cb(
                    done, n_found, Path(fpath).name):
                cancelled = True
                break

            fmeta = _file_meta(fpath)

            # Skip if unchanged and not forcing rescan
            if not force_rescan and fpath in existing:
                if existing[fpath]["modified_at"] == fmeta["modified_at"]:
                    # Still update last_seen so we know the file still exists
                    db.upsert_file({
                        "path":      fpath,
                        "last_seen": now,
                        "filename":  Path(fpath).name,
                        "first_seen": existing[fpath]["first_seen"],
                        **fmeta,
                    }, db_path=db_path, conn=conn)
                    pending += 1
                    continue

            record = _parse_ibw(fpath)
            file_meta = record.pop("_file_metadata", {})
            record.update(fmeta)
            record["last_seen"]    = now
            record["first_seen"]   = (existing[fpath]["first_seen"] if fpath in existing else now) or now

            db.upsert_file(record, db_path=db_path, conn=conn)

            if file_meta and record["parse_ok"]:
                # Visible on this connection even though the batch is not
                # committed yet — a connection always sees its own writes.
                file_id = db.get_file_id(fpath, db_path, conn=conn)
                if file_id is not None:
                    db.write_file_metadata(file_id, file_meta, db_path, conn=conn)

            pending   += 1
            n_updated += 1
            if not record["parse_ok"]:
                n_errors += 1

        # Committed on the cancel path too: those files really were scanned,
        # and throwing the work away would be a worse answer to "Cancel" than
        # keeping it — the directory stays unmarked below, so the next scan
        # picks up where this one stopped.
        conn.commit()          # final, partial batch
    except BaseException:
        conn.rollback()        # drop the incomplete batch, keep the DB consistent
        raise
    finally:
        conn.close()

    if cancelled:
        return n_found, n_updated, n_errors, True

    if progress_cb is not None:
        progress_cb(n_found, n_found, "")
    return n_found, n_updated, n_errors, False


def requalify_catalog(
    db_path:    str  = db.DEFAULT_DB_PATH,
    progress_cb=None,
    only_missing: bool = True,
) -> dict:
    """
    Re-read catalogued files to fill in what the scanner did not know when they
    were first imported: their content hash, and any qualification check added
    since.

    Needed because scan_directory deliberately does NOT re-open a file whose
    mtime is unchanged — the right call for an ordinary scan, but it means a
    new check reaches old files only when they happen to be re-analysed. This
    is the explicit, once-off alternative to making every scan slow.

    only_missing=True (the default) visits just the files with no hash yet, so
    running it twice costs nothing and an interrupted run resumes where it
    stopped: progress IS the hash column, not a separate bookmark. Pass False
    to re-read everything, e.g. after changing a qualification rule.

    Reads at ~40 ms/file, dominated by disk; 140k files is roughly two hours.
    progress_cb(done, total, label) -> cancelled, exactly as scan_directory's.
    Cancelling keeps every file already done — they are real results.

    Returns {'seen','hashed','requalified','unreadable','cancelled'}.
    """
    conn = db.get_connection(db_path)
    try:
        sql = "SELECT id, path, curve_type, unusable_reason, content_sha256 FROM files"
        if only_missing:
            sql += " WHERE content_sha256 IS NULL"
        rows = conn.execute(sql + " ORDER BY id").fetchall()
    finally:
        conn.close()

    total = len(rows)
    seen = hashed = requalified = unreadable = 0
    cancelled = False
    conn = db.get_connection(db_path)
    pending = 0
    try:
        for done, row in enumerate(rows):
            if pending >= SCAN_BATCH_SIZE:
                conn.commit()
                pending = 0
            if progress_cb is not None and progress_cb(
                    done, total, Path(row["path"]).name):
                cancelled = True
                break

            seen += 1
            try:
                raw = Path(row["path"]).read_bytes()
            except OSError:
                # The file is catalogued but not readable right now — a
                # disconnected drive, most likely. That is not a fact about the
                # file, so nothing is written and the next run tries again.
                unreadable += 1
                continue

            sha = hashlib.sha256(raw).hexdigest()
            try:
                wave = load_ibw(io.BytesIO(raw))
                q = qualify_wave(
                    wave["wave"]["wData"],
                    labels=wave["wave"]["labels"],
                    indent_mode=_safe_bool(rb"IndentMode: ([0-1])\r", wave["wave"]["note"]),
                    hold_z=_hold_z_sensor(wave["wave"]["note"]),
                    spring_constant=_spring_constant(wave["wave"]["note"]),
                )
            except Exception as exc:                       # noqa: BLE001
                # Readable bytes that igor2 cannot parse: record the hash we
                # did compute (it is true) and mark the parse failure, rather
                # than dropping both facts on the floor.
                conn.execute(
                    "UPDATE files SET content_sha256=?, parse_ok=0, parse_error=? WHERE id=?",
                    (sha, str(exc)[:500], row["id"]))
                hashed += 1
                pending += 1
                continue

            changed = (q.curve_type != row["curve_type"]
                       or q.reason != row["unusable_reason"])
            conn.execute(
                "UPDATE files SET content_sha256=?, curve_type=?, "
                "unusable_reason=?, unusable_detail=? WHERE id=?",
                (sha, q.curve_type, q.reason, q.detail, row["id"]))
            hashed += 1
            requalified += int(changed)
            pending += 1

        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()

    if progress_cb is not None and not cancelled:
        progress_cb(total, total, "")
    return {"seen": seen, "hashed": hashed, "requalified": requalified,
            "unreadable": unreadable, "cancelled": cancelled}


