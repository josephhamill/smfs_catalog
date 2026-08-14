# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Guard: every export in this app goes through export_utils, and every export
writes a manifest.

WHY THIS EXISTS (2026-07-31)
    Exports are how results leave this app. They become report figures and
    publication figures, read months later by someone with no access to the
    window that produced them. A CSV of bare numbers with no record of which
    curves and which settings produced it cannot be reproduced, and an
    un-reproducible result is a dead end for whoever has to publish it.

    Before this convention there were three hand-rolled copies of "write a
    manifest" and three of "write a CSV", drifting apart in what they
    recorded, plus one export (roi_explorer) that skipped both and wrote
    wherever a file dialog landed. This test stops that recurring.

WHAT IT CHECKS
    (a) No module outside export_utils opens a file for writing or calls
        np.savetxt / csv.writer directly. Those are how a fork starts.
    (b) db.py — which cannot import export_utils (circular) — returns data
        rather than writing export files.
    (c) ExportGroup always writes a manifest, and the manifest carries the
        file list, settings, app version and timestamp.
    (d) A group's filenames are allocated together, so a data file can never
        be paired with a manifest from a different export. A manifest is
        written only when all declared parts, and no undeclared parts, exist.
    (e) Every window class that has an export button implements
        export_provenance().
    (f) A drawn CI band exports the full covariance, not just its diagonal.
    (g) The export folder is one DB-wide setting with no owner and is shown
        on screen. Retired per-profile values cannot override that setting.

HOW TO READ A FAILURE
    A failure means an export was added that bypasses the shared convention.
    The fix is to route it through export_utils.export_group(), NOT to relax
    this test. If a genuinely new kind of writing is needed, add it as a
    method on ExportGroup so every caller gets it.

Run with the smfs-catalog environment, from the repo root:
    python tests/test_export_convention.py
"""
import ast
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG  = ROOT / "smfs_catalog"
sys.path.insert(0, str(ROOT))

from smfs_catalog import db as _db                     # noqa: E402
from smfs_catalog import export_utils as _export       # noqa: E402

# One shared idiom for these procedural guards — see checkstyle.py for why
# `sys.exit(1)` at the bottom of a file was aborting the whole pytest run.
import checkstyle                                          # noqa: E402

check = checkstyle.CheckRunner()


SRC = {p.name: p.read_text(encoding="utf-8", errors="replace") for p in PKG.glob("*.py")}

# ── (a) Nothing outside export_utils writes export files itself ───────────────
#
# `open(..., "w")`, csv.writer and np.savetxt are the three ways an export file
# gets written. export_utils owns all three. Anywhere else, they are either a
# new fork of the convention or a non-export write that should say so.
#
# Allowed exception, deliberately narrow: export_utils owns the convention.
_WRITE_ALLOWED = {"export_utils.py"}
_WRITE_RE = re.compile(
    r"""open\s*\([^)]*["']w["']|csv\.writer|csv\.DictWriter|np\.savetxt|numpy\.savetxt""")

offenders = []
for name, src in sorted(SRC.items()):
    if name in _WRITE_ALLOWED:
        continue
    for i, line in enumerate(src.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        if _WRITE_RE.search(line):
            offenders.append(f"{name}:{i}: {line.strip()}")
check(
    "(a) no module outside export_utils writes export files directly",
    not offenders,
    "\n      ".join(offenders),
)

# ── (b) db.py returns data, it does not write export files ───────────────────
#
# db.py cannot import export_utils (export_utils imports db — a module-level
# cycle would break both), so the rule for db is stronger: it must not have
# export-writing functions at all. It supplies rows; a window writes them.
db_src = SRC["db.py"]
db_tree = ast.parse(db_src)
db_funcs = {n.name for n in ast.walk(db_tree) if isinstance(n, ast.FunctionDef)}
check(
    "(b) db.export_csv is gone (was dead code — its only caller, app.py, was deleted)",
    "export_csv" not in db_funcs,
)
check(
    "(b) db.export_classification_report no longer writes a file itself",
    "export_classification_report" not in db_funcs,
    "replaced by classification_report_rows(), which returns (header, rows)",
)
check(
    "(b) db.export_queue_paths no longer writes a file itself",
    "export_queue_paths" not in db_funcs,
    "replaced by queue_paths(), which returns a list",
)
check(
    "(b) db supplies the report as data",
    "classification_report_rows" in db_funcs and "queue_paths" in db_funcs,
)

# ── (c)/(d) ExportGroup behaviour, against a real temp DB ────────────────────
tmp = tempfile.mkdtemp()
DB = str(Path(tmp) / "t.db")
_db.initialise(DB)
_export.set_export_dir_override(tmp, DB)

with _export.export_group(DB, "guard", [".csv", "_m.csv"], kind="guard_test") as g:
    g.contributing_files(["/data/b.ibw", "/data/a.ibw", "/data/a.ibw", None])
    g.note(setting_one="x", k=3)
    g.histogram(".csv", [0.0, 1.0, 2.0], [4, 5])
    g.matrix("_m.csv", [[1, 2], [3, 4]])

manifest_path = g.path("_manifest.json")
check("(c) a manifest is written without the caller asking", manifest_path.exists())

man = json.loads(manifest_path.read_text())
check("(c) manifest names the kind of export", man.get("export") == "guard_test")
check("(c) manifest lists the contributing files, deduped and sorted",
      man.get("files") == ["/data/a.ibw", "/data/b.ibw"], str(man.get("files")))
check("(c) manifest counts the files", man.get("n_files") == 2)
check("(c) manifest carries the window's settings",
      man.get("setting_one") == "x" and man.get("k") == 3)
check("(c) manifest records the app version", bool(man.get("app_version")))

# smfs_catalog.__version__ is declared in the package (so it works from a
# checkout, an install, and a PyInstaller bundle alike) rather than read from
# pyproject.toml at runtime. That means two declarations, so check they agree —
# a manifest stating the wrong version is worse than one stating none.
import tomllib                                          # noqa: E402
with open(ROOT / "pyproject.toml", "rb") as _f:
    _pyproject_version = tomllib.load(_f)["project"]["version"]
check("(c) __version__ matches pyproject.toml",
      man.get("app_version") == _pyproject_version,
      f"manifest says {man.get('app_version')}, pyproject says {_pyproject_version}")
check("(c) manifest records the schema version",
      man.get("schema_version") == _db.SCHEMA_VERSION)
check("(c) manifest records the exact database",
      man.get("database_path") == str(Path(DB).resolve()))
check("(c) manifest records the cache/provenance build identity",
      bool(man.get("app_build_identity")) and
      "current_scientific_method" not in man)
check("(c) live parameters are honestly labelled as export-time context",
      "active_param_set" in man and "active_param_owner" in man and
      "not necessarily" in man.get("active_param_scope", "") and
      "param_set" not in man and "param_owner" not in man)
check("(c) manifest records when it was generated", bool(man.get("generated_at")))
check("(c) manifest lists the data files it describes",
      sorted(man.get("data_files", [])) == sorted(
          [f"{g.base}.csv", f"{g.base}_m.csv"]),
      str(man.get("data_files")))

# A path of None or "" must never reach the file list.
check("(c) blank/None paths are dropped, never written as empty entries",
      all(p for p in man["files"]))

# The histogram convention is one shared format, not per-window.
hist_lines = g.path(".csv").read_text().strip().splitlines()
check("(c) histograms use the shared bin_left/bin_right/count header",
      hist_lines[0] == "bin_left,bin_right,count", hist_lines[0])
check("(c) histogram rows are (left, right, value)",
      hist_lines[1] == "0.0,1.0,4.0", hist_lines[1])

# (d) The whole group shares one basename, allocated together. Writing a second
# group with the same stem in the same second must not collide with the first —
# otherwise a matrix from one export could end up beside another's manifest.
with _export.export_group(DB, "guard", [".csv", "_m.csv"], kind="guard_test") as g2:
    g2.contributing_files(["/data/a.ibw"])
    g2.histogram(".csv", [0.0, 1.0], [1])
    g2.matrix("_m.csv", [[9]])
check("(d) a second export with the same stem gets its own basename",
      g2.base != g.base, f"{g.base} vs {g2.base}")
check("(d) the first export's files were not overwritten",
      json.loads(g.path("_manifest.json").read_text())["n_files"] == 2)
check("(d) every file in a group shares one basename",
      all(n.startswith(g2.base) for n in
          [f"{g2.base}.csv", f"{g2.base}_m.csv"]))

# A failing export must not leave a manifest claiming files that were never
# finished — better an obviously-incomplete group than a lying one.
try:
    with _export.export_group(DB, "guard_fail", [".csv"], kind="guard_test") as g3:
        g3.contributing_files(["/data/a.ibw"])
        raise RuntimeError("simulated failure mid-export")
except RuntimeError:
    pass
check("(d) an export that raises writes no manifest",
      not g3.path("_manifest.json").exists())

# Declaring parts up front is also a completeness contract. A forgotten writer
# call must not produce a manifest that makes a partial export look finished.
try:
    with _export.export_group(
        DB, "guard_missing", [".csv", "_m.csv"], kind="guard_test",
    ) as g4:
        g4.table(".csv", ["value"], [[1]])
except RuntimeError:
    pass
check("(d) a missing declared part prevents the manifest",
      not g4.path("_manifest.json").exists())

try:
    with _export.export_group(DB, "guard_extra", [".csv"], kind="guard_test") as g5:
        g5.table(".csv", ["value"], [[1]])
        g5.table("_extra.csv", ["value"], [[2]])
except RuntimeError:
    pass
check("(d) an undeclared part prevents the manifest",
      not g5.path("_manifest.json").exists())

try:
    with _export.export_group(DB, "guard_hist", [".csv"], kind="guard_test") as g6:
        g6.histogram(".csv", [0.0, 1.0, 2.0], [1])
except ValueError:
    pass
check("(c) malformed histogram geometry is rejected",
      not g6.path("_manifest.json").exists())

# ── (f) A drawn CI band must be reproducible from the export ─────────────────
#
# Both windows that draw a total_fit_ci band export the band's lo/hi columns
# AND the full parameter covariance. The DIAGONAL alone (stderr/ci_lo/ci_hi in
# the params table) cannot regenerate the band — total_fit_ci samples the full
# matrix precisely because correlated parameters make the diagonal overstate
# the spread — so exporting only the diagonal would let someone redraw the
# figure with a visibly wrong band.
from smfs_catalog.dist_fit_core import (                 # noqa: E402
    CI_N_DRAWS, CI_PCT, CI_SEED, ci_manifest_fields, total_fit_ci,
)
import numpy as np                                       # noqa: E402

cov = np.array([[0.04, 0.03], [0.03, 0.09]])             # strongly correlated
fields = ci_manifest_fields(cov, True)
check("(f) CI manifest carries the FULL covariance, not just its diagonal",
      fields["param_covariance"] == cov.tolist(), str(fields["param_covariance"]))
check("(f) CI manifest records the Monte-Carlo settings needed to reproduce it",
      (fields["ci_pct"], fields["ci_n_draws"], fields["ci_seed"])
      == (CI_PCT, CI_N_DRAWS, CI_SEED))
check("(f) CI manifest says whether a band was actually drawn",
      fields["ci_band_drawn"] is True)

# An unusable covariance yields no band and no fabricated matrix.
bad = ci_manifest_fields(np.array([[np.inf, 0.0], [0.0, 1.0]]), False)
check("(f) an unusable covariance exports None, never a fabricated interval",
      bad["param_covariance"] is None and bad["ci_band_drawn"] is False)
check("(f) total_fit_ci itself returns None for an unusable covariance",
      total_fit_ci(lambda x, a, b: a * x + b, np.linspace(0, 1, 5),
                   [1.0, 0.0], np.array([[np.inf, 0.0], [0.0, 1.0]])) is None)

# The seed really does make it reproducible — otherwise recording it is a lie.
fn = lambda x, a, b: a * np.exp(-b * x)                   # noqa: E731
band1 = total_fit_ci(fn, np.linspace(0, 2, 20), [1.0, 0.5], cov)
band2 = total_fit_ci(fn, np.linspace(0, 2, 20), [1.0, 0.5], cov)
check("(f) a seeded CI band is reproducible run to run",
      band1 is not None and np.allclose(band1[0], band2[0])
      and np.allclose(band1[1], band2[1]))

# The exports that write a band must write both edges of it.
for mod, needle in (("dist_fit_window.py", '"ci_lo", "ci_hi"'),
                    ("mean_curve_window.py", '"ci_lo", "ci_hi"')):
    check(f"(f) {mod} exports the band's lo/hi columns",
          needle in SRC[mod])
    check(f"(f) {mod} records the CI provenance in its manifest",
          "ci_manifest_fields(" in SRC[mod])

# ── (g) The export folder is ONE DB-wide setting, with no owner ──────────────
#
# #123 (2026-08-01 user test): the folder was set with the app's own button and
# four hours later the app was writing to the database directory instead, with
# nothing on screen saying so. It was stored per-experimentalist and resolved
# through "whose data is this" — the first file in the catalog carrying a name.
# Adding files changed that answer, so the setting was written under one person
# and read back under another, and ended up recorded under two at once.
#
# The rule now: one setting, no experimentalist dimension anywhere in the
# resolution path. These checks are what stops an owner being reintroduced
# "just for the folder".
import inspect                                            # noqa: E402

for fn in (_export.resolve_export_dir, _export.set_export_dir_override):
    params = list(inspect.signature(fn).parameters)
    check(f"(g) {fn.__name__} takes no experimentalist",
          not any("experimentalist" in p or "owner" in p for p in params),
          str(params))

# Prose about the retired storage is fine and wanted; a CALL into the profile
# tables is the thing that must not come back.
_profile_calls = [
    f"_db.{fn}(" for fn in
    ("get_experimentalist_profile", "merge_experimentalist_profile",
     "set_experimentalist_profile")
    if f"_db.{fn}(" in SRC["export_utils.py"]
]
check("(g) the export folder is not stored on a per-experimentalist profile",
      not _profile_calls, ", ".join(_profile_calls))

# The behaviour those signatures exist to guarantee: the folder a person set
# must survive the catalog gaining files owned by someone else. Under the old
# storage this exact sequence silently changed the answer.
DB2 = str(Path(tempfile.mkdtemp()) / "t2.db")
_db.initialise(DB2)
chosen = tempfile.mkdtemp()
_export.set_export_dir_override(chosen, DB2)
conn = _db.get_connection(DB2)
with conn:
    conn.execute(
        "INSERT INTO files (path, filename, experimentalist, "
        "                   first_seen, last_seen) "
        "VALUES ('/data/z.ibw', 'z.ibw', 'Dylan', '2026-08-03', '2026-08-03')")
conn.close()
check("(g) the chosen folder survives the catalog gaining a new owner",
      str(_export.resolve_export_dir(DB2)) == chosen,
      str(_export.resolve_export_dir(DB2)))

_export.set_export_dir_override(None, DB2)
check("(g) clearing it falls back to the database's own directory",
      _export.resolve_export_dir(DB2) == Path(DB2).resolve().parent)

# Clearing must stay cleared: an explicit empty value means retired profile
# data cannot become active again on the next launch.
_db.merge_experimentalist_profile(
    "Dylan", {"export_dir_override": "/somewhere/retired"}, DB2)
_db.initialise(DB2)
check("(g) an explicit 'no folder' ignores retired profile data",
      _export.resolve_export_dir(DB2) == Path(DB2).resolve().parent,
      str(_export.resolve_export_dir(DB2)))

# The dashboard must SHOW it. The only way to learn the folder used to be to
# perform an export and read the confirmation dialog.
check("(g) the dashboard puts the export folder on screen",
      "_update_export_dir_label" in SRC["dashboard_window.py"]
      and "self._export_dir_lbl" in SRC["dashboard_window.py"])
check("(g) the dashboard refreshes that label when the folder changes",
      SRC["dashboard_window.py"].count("_update_export_dir_label()") >= 2)


# ── (e) Every window with an export button implements export_provenance ──────
#
# The manifest's settings block comes from export_provenance(). A window that
# exports without one writes a manifest that says which FILES but not under
# which SETTINGS — half a record.
_PROV_EXEMPT = {
    # The dashboard's three exports (classification report, queue save, queue
    # table) are DB-scoped, not view-scoped: their settings are the scope
    # filters and segment selection, noted inline at each call site rather
    # than via one window-wide method that would have to mean three things.
    "dashboard_window.py",
}
missing = []
for name, src in sorted(SRC.items()):
    if name in _PROV_EXEMPT or name == "export_utils.py":
        continue
    if "_export.export_group(" not in src:
        continue
    if "def export_provenance(" not in src:
        missing.append(name)
check("(e) every exporting window implements export_provenance()",
      not missing, ", ".join(missing))

# Windows that call export_group must import export_utils as _export — the
# aliasing is uniform so (a)'s scan and this one stay reliable.
bad_import = [
    n for n, s in sorted(SRC.items())
    if n != "export_utils.py"          # its own docstring shows the call
    and "_export.export_group(" in s
    and "from . import export_utils as _export" not in s
]
check("(e) exporting modules import export_utils the same way",
      not bad_import, ", ".join(bad_import))


# Every check above becomes its own named pytest case.  Must be last:
# pytest_cases reads what the module body recorded.
test_check = checkstyle.pytest_cases(check)
