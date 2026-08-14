# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""Shared storage, file-format, and manifest conventions for all exports.

Windows decide what data and provenance to export. This module deliberately
knows nothing about UI state: it chooses the database-wide destination,
allocates a fresh basename, writes standard formats, and records universal
provenance. Every successful export is a complete group of declared data files
plus a manifest; an interrupted or structurally incomplete group has no
manifest and therefore cannot be mistaken for a finished result.

This module contains no Qt code. New export formats belong on ``ExportGroup``
so every window uses the same writing convention.
"""

from __future__ import annotations

import csv
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from . import db as _db
from . import quantities as _quant
from .provenance import app_version, code_version

MANIFEST_SUFFIX = "_manifest.json"


def resolve_export_dir(db_path: str) -> Path:
    """Where every export in this app writes: the chosen folder if one is set,
    else the DB's own directory.

    The export destination is one database-wide setting with no
    experimentalist owner. Per-cohort resolution would make the destination
    depend on the data currently in view rather than the database being
    exported."""
    override = _db.get_app_setting(_db.APP_SETTING_EXPORT_DIR, "", db_path)
    if override:
        return Path(override)
    return Path(db_path).resolve().parent


def set_export_dir_override(folder: str | None, db_path: str) -> None:
    """Set the export folder, or clear it (folder=None) to fall back to the DB
    directory again.

    Clearing writes an EMPTY value rather than deleting the row, preserving an
    explicit "no override" choice."""
    _db.set_app_setting(_db.APP_SETTING_EXPORT_DIR, folder or "", db_path)


def new_export_path(export_dir: Path, stem: str, suffix: str) -> Path:
    """A fresh timestamped path inside export_dir, creating the folder if
    this is the first export of the session. Disambiguates rather than
    overwrites if two exports land in the same second."""
    export_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = export_dir / f"{stem}_{ts}.{suffix}"
    n = 2
    while path.exists():
        path = export_dir / f"{stem}_{ts}_{n}.{suffix}"
        n += 1
    return path


def slug(text: str) -> str:
    """A filename/column-safe version of a UI label.

    Window labels ("Rupture force (selected segment)", "Peak 1 (Gaussian)")
    become part of export filenames and CSV column headers; spaces, slashes
    and unicode in either are a portability problem for whoever opens the
    file next. Collapses to lowercase words joined by underscores."""
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch.lower())
        elif out and out[-1] != "_":
            out.append("_")
    return "".join(out).strip("_") or "unnamed"


def new_export_group_stem(export_dir: Path, stem: str, suffixes: list[str]) -> str:
    """A shared timestamped stem for a group of files that must travel
    together as one unit (e.g. a 2DH's bare matrix + its paired 1DH edge
    files + a provenance manifest — none of them is self-describing alone).
    `suffixes` are the exact filename tails the caller is about to write
    (e.g. "_matrix.csv", "_manifest.json"); disambiguates the WHOLE group at
    once if any of them already exists, so the four files can never end up
    mismatched (e.g. a matrix from one export paired with a manifest from a
    later one)."""
    export_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{stem}_{ts}"
    n = 2
    while any((export_dir / f"{base}{suf}").exists() for suf in suffixes):
        base = f"{stem}_{ts}_{n}"
        n += 1
    return base


# ── The writing convention ────────────────────────────────────────────────────

class ExportGroup:
    """One export: a set of data files sharing a timestamped basename, plus
    the manifest that says what they are.

    Never constructed directly — use the `export_group()` context manager,
    which allocates the basename and guarantees the manifest gets written.

    The data-writing methods (`table`, `histogram`, `matrix`, `text`) exist so
    that the app's file *formats* are defined once too, not just its folder.
    A histogram exported from Explore Events and one exported from a 2DH's
    marginal projection are the same kind of object and must be readable by
    the same script; before this they were two hand-rolled writers that
    happened to agree.
    """

    def __init__(self, export_dir: Path, base: str, kind: str,
                 parts: Sequence[str],
                 db_path: str = _db.DEFAULT_DB_PATH) -> None:
        self.dir     = export_dir
        self.base    = base
        self._kind   = kind
        self._db_path = db_path
        self._expected = tuple(parts)
        self._written: list[str] = []
        self._files:   list[str] = []
        self._notes:   dict[str, Any] = {}

    # ── Declaring what this export is ────────────────────────────────────────

    def contributing_files(self, paths: Iterable[str]) -> None:
        """The curve files whose data is in this export.

        **Full paths, never basenames** — the same filename recurs across
        directories in a real catalog, so a basename cannot identify a curve.
        Callers may pass duplicates or None-ish entries; both are cleaned up
        here rather than at each of the dozen call sites.
        """
        self._files = sorted({str(p) for p in paths if p})

    def note(self, **fields: Any) -> None:
        """Add provenance fields to the manifest (settings, parameters,
        seeds, counts). Call as often as convenient; later calls win."""
        self._notes.update(fields)

    def note_dict(self, fields: dict[str, Any]) -> None:
        """Same as note(), for a dict built elsewhere — typically a window's
        own `export_provenance()`."""
        self._notes.update(fields)

    # ── Writing data ─────────────────────────────────────────────────────────

    def path(self, suffix: str) -> Path:
        return self.dir / f"{self.base}{suffix}"

    def table(
        self, suffix: str, header: Sequence[str], rows: Iterable[Sequence],
    ) -> Path:
        """A plain CSV table: one header row, then data rows."""
        out = self.path(suffix)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(list(header))
            w.writerows(rows)
        return self._record(suffix, out)

    def dict_table(
        self, suffix: str, header: Sequence[str], keys: Sequence[str],
        rows: Iterable[dict],
    ) -> Path:
        """A CSV table built from dicts: `header` is what the columns are
        called in the file, `keys` how to look each one up in a row. Missing
        keys are written blank rather than raising — a row lacking an optional
        value is normal here, not an error."""
        return self.table(
            suffix, header,
            ([r.get(k, "") if r.get(k) is not None else "" for k in keys]
             for r in rows),
        )

    def histogram(
        self, suffix: str, edges: Sequence[float], counts: Sequence[float],
        *, value_column: str = "count",
    ) -> Path:
        """A 1-D histogram as `bin_left, bin_right, <value_column>`.

        `value_column` is normally "count". Derived histograms may contain
        floats such as counts/trace; the manifest records that quantity while
        this column convention stays uniform.
        """
        if len(edges) != len(counts) + 1:
            raise ValueError(
                "histogram edges must contain exactly one more value than "
                f"counts (got {len(edges)} edges and {len(counts)} counts)"
            )
        rows = [
            (float(edges[i]), float(edges[i + 1]), float(counts[i]))
            for i in range(len(counts))
        ]
        return self.table(suffix, ["bin_left", "bin_right", value_column], rows)

    def matrix(self, suffix: str, array, fmt: str = "%.17g") -> Path:
        """A round-trip-precision numeric grid with no headers or index.

        It pastes straight into
        Excel/Origin as a matrix. Its axis definition lives in the paired
        1-D files and/or the manifest (#62's triplet convention), which is
        exactly why this file can stay bare."""
        out = self.path(suffix)
        np.savetxt(out, np.asarray(array), delimiter=",", fmt=fmt)
        return self._record(suffix, out)

    def text(self, suffix: str, content: str) -> Path:
        out = self.path(suffix)
        with open(out, "w", encoding="utf-8") as f:
            f.write(content)
        return self._record(suffix, out)

    def _record(self, suffix: str, out: Path) -> Path:
        name = f"{self.base}{suffix}"
        if name not in self._written:
            self._written.append(name)
        return out

    # ── Finishing ────────────────────────────────────────────────────────────

    def _validate_complete(self) -> None:
        """Verify that the caller wrote exactly the parts it declared."""
        expected = {f"{self.base}{suffix}" for suffix in self._expected}
        written = set(self._written)
        if written == expected and len(self._expected) == len(expected):
            return

        details = []
        missing = sorted(expected - written)
        unexpected = sorted(written - expected)
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"undeclared: {', '.join(unexpected)}")
        if len(self._expected) != len(expected):
            details.append("duplicate declared parts")
        raise RuntimeError("incomplete export group (" + "; ".join(details) + ")")

    def _write_manifest(self) -> Path:
        param_set = _db.get_param_set(self._db_path)
        manifest: dict[str, Any] = {
            "export":         self._kind,
            "basename":       self.base,
            "data_files":     list(self._written),
            "n_files":        len(self._files),
            "files":          self._files,
            "generated_at":   datetime.now().isoformat(timespec="seconds"),
            "app_version":    app_version(),
            "app_build_identity": code_version(),
            "schema_version": _db.SCHEMA_VERSION,
            "database_path":  str(Path(self._db_path).resolve()),
            # These are the settings in force WHEN THE EXPORT WAS MADE. They
            # are useful context, but may not be the settings that produced
            # every stored row in a lazily updated or mixed-history cohort.
            # Per-row/window provenance may add exact producer identities;
            # never relabel this live snapshot as historical provenance.
            "active_param_owner": _db.active_param_owner(self._db_path),
            "active_param_set": param_set,
            "active_param_scope": "export-time analysis context; not necessarily every stored row's producer",
            # Stored units for exactly the values above. An empty string means
            # deliberately dimensionless; an absent key means undeclared.
            "active_param_units": _quant.units_for(param_set),
        }
        # Window provenance last: a window may legitimately override
        # n_files/files semantics for its own export (e.g. an export whose
        # rows are fits rather than curves), and its own words should win.
        manifest.update(self._notes)
        out = self.path(MANIFEST_SUFFIX)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
        self._record(MANIFEST_SUFFIX, out)
        return out

    def message(self, intro: str = "Wrote") -> str:
        """Plain-text summary for the caller's message box. Built here so the
        wording is identical everywhere; the QMessageBox itself stays in the
        window, since this module holds no Qt."""
        listed = "\n".join(f"  {n}" for n in self._written)
        return f"{intro}:\n{listed}\nto:\n{self.dir}"


@contextmanager
def export_group(
    db_path: str, stem: str, parts: Sequence[str], *, kind: str,
) -> Iterator[ExportGroup]:
    """Open an export.

        with _export.export_group(db, "hist_force_hit", [".csv"],
                                  kind="histogram_force") as g:
            g.contributing_files(paths)
            g.note(variable="force_pN", n_bins=len(counts))
            g.histogram(".csv", edges, counts)
        QMessageBox.information(self, "Export", g.message())

    `parts` are the data-file suffixes about to be written. They are declared
    up front so the WHOLE group — manifest included — is disambiguated as one
    unit against anything already on disk; that is what stops a matrix from
    one export ending up beside a manifest from a later one.

    The manifest is written only when the body exits cleanly and every declared
    part was written, with no undeclared extras. A failed or partial group has
    no manifest, so it cannot be mistaken for a complete export.

    The export folder is one DB-wide setting, never per-cohort or per-person —
    see resolve_export_dir. Do not add a per-window identity parameter here.
    """
    export_dir = resolve_export_dir(db_path)
    base = new_export_group_stem(
        export_dir, stem, list(parts) + [MANIFEST_SUFFIX])
    group = ExportGroup(export_dir, base, kind, parts, db_path)
    yield group
    group._validate_complete()
    group._write_manifest()
