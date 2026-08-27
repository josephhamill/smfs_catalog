# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Regression test: manual Primary/Secondary segment override.

The contract under test:
(a) with no override set, segment_summary_bulk behaves exactly as before
    (Ultimate/Penultimate `select`, dF/isoforce from the last two ruptures).
(b) a Primary override takes precedence over `select` entirely for
    l_p_nm/l_c_nm/l_p_err/l_c_err/force_pN, for THAT curve only.
(c) Primary+Secondary together drive dF_pN/seg_dX_iso_nm/seg_dX_ext_nm
    INSTEAD of the last-two-ruptures default.
(d) dF_pN/seg_dX_ext_nm need no adjacency (plain subtractions of two
    already-known points); seg_dX_iso_nm DOES (isoforce_x_nm is only ever
    stored relative to the immediately preceding rupture) and is None for a
    non-adjacent pair, same as any other missing piece — never fabricated.
    seg_dX_ext_nm is the order-independent, plain-geometry
    counterpart to dF_pN — see roi_events.ROI.dX_ext_pairs — deliberately
    NOT the same thing as isoforce_x_nm, which stays one-directional on
    purpose (see that docstring for why a backward-looking generalisation
    was tried and rejected).
(e) which one was clicked "Primary" vs "Secondary" never flips the sign —
    the pair is resolved by actual rupture index, not by role.
(f) an override tagged against a stale event_map (the curve was
    reanalysed and resegmented since the pick was made) is ignored,
    falling back to (a)'s default behaviour — a pinned index does not
    self-correct the way Ultimate/Penultimate/First/Last do.
(g) Save Queue / Load Queue round-trips a plain file list through the same
    `path`-column convention as every other export in the app.

Run with the smfs-catalog env, from the repo root:
    python tests/test_segment_override.py
"""
import json
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smfs_catalog import db as _db
from smfs_catalog.provenance import cache_version
from smfs_catalog import roi_pipeline as _rp
# Imported, never retyped: a hand-copied schema version in a fixture goes stale
# silently on the next bump and the whole file then fails as a version miss
# rather than on the thing it tests (which is exactly what happened at v3->v4).
from smfs_catalog.roi_events import _PAYLOAD_VERSION

tmp = tempfile.mkdtemp(prefix="segment_override_")
DB = os.path.join(tmp, "test.sqlite")
_db.initialise(DB)

FILE_PATH = _db.normalize_path("/tank/testdata/segoverride/Image0169.ibw")
CODE_VER = cache_version() or "test-build"


def _payload(params_tag: str) -> dict:
    """One ROI, 3 segments/ruptures. Forces [50, 80, 120] pN (ascending, so
    dF/isoforce below are unambiguous about direction). isoforce_x_nm is
    only meaningful relative to the IMMEDIATELY PRECEDING rupture (real
    behaviour, roi_events.Segment docstring), so segments[1] carries a value
    relative to ruptures[0], and segments[2] relative to ruptures[1] — there
    is deliberately no value that would answer "segment 2 vs rupture 0".
    """
    def seg(i, l_p, l_c, iso_x):
        return {
            "left_idx": i * 100, "right_idx": i * 100 + 90,
            "left_piezo_nm": float(i * 100), "right_piezo_nm": float(i * 100 + 90),
            "l_p_nm": l_p, "l_c_nm": l_c, "l_p_err": 0.05, "l_c_err": 2.0,
            "n_pts": 50, "fit_lo_idx": i * 100, "fit_hi_idx": i * 100 + 90,
            "isoforce_x_nm": iso_x,
            # v4 diagnostics. Distinct per segment so the checks
            # below can tell WHICH segment's diagnostics were selected — the
            # property that matters is that they follow the same selection rule
            # as l_p/l_c, so a tau can never describe a different fit than the
            # error bar it explains.
            "tau": 10.0 * (i + 1), "x_max_nm": l_c * 0.8,
            # v5. Only segments[0]'s value is the junction's onset; the others
            # start mid-snapback of the preceding rupture and are deliberately
            # different, so a junction extension computed off the wrong segment
            # would show up in the checks below.
            "left_extension_nm": 4.0 if i == 0 else float(i * 100),
            "edge_pinned": (i == 1),
        }

    def rup(i, force, ext):
        return {
            "idx": i * 100 + 90, "piezo_nm": float(i * 100 + 90),
            "d1_height": 1.0, "prominence": 1.0,
            "force_pN": force, "force_idx": i * 100 + 85,
            "rise_idx": i * 100 + 80, "fall_idx": i * 100 + 92,
            "extension_nm": ext,
        }

    return {
        "v": _PAYLOAD_VERSION, "detector": "test",
        "rois": [{
            "onset_idx": 0, "return_idx": 400,
            "onset_piezo_nm": 0.0, "return_piezo_nm": 400.0,
            "ruptures": [rup(0, 50.0, 10.0), rup(1, 80.0, 20.0), rup(2, 120.0, 30.0)],
            "segments": [
                seg(0, 0.4, 15.0, None),          # first segment: no "previous rupture"
                seg(1, 0.4, 25.0, 20.5),          # relative to ruptures[0] (force=50)
                seg(2, 0.4, 35.0, 30.7),          # relative to ruptures[1] (force=80)
            ],
        }],
        # tag carried in params_json only so the test can simulate a
        # resegmentation by writing a second document under a new tag —
        # write_event_map itself doesn't interpret this string at all.
        "_test_tag": params_tag,
    }


conn = _db.get_connection(DB)
with conn:
    conn.execute(
        "INSERT INTO files (path, filename, first_seen, last_seen, event)"
        " VALUES (?, ?, datetime('now'), datetime('now'), 'event')",
        (FILE_PATH, os.path.basename(FILE_PATH)))
fid = conn.execute("SELECT id FROM files WHERE path=?", (FILE_PATH,)).fetchone()[0]
conn.close()

PARAMS_V1 = json.dumps({"tag": "v1"})
_db.write_event_map(fid, json.dumps(_payload("v1")), PARAMS_V1, CODE_VER, DB)

# One shared idiom for these procedural guards — see checkstyle.py for why
# `sys.exit(1)` at the bottom of a file was aborting the whole pytest run.
import checkstyle                                          # noqa: E402

check = checkstyle.CheckRunner()


# ── (a) no override: default Ultimate/Penultimate + last-two dF/isoforce ────
summ = _rp.segment_summary_bulk([FILE_PATH], "ultimate", DB)[_db.normalize_path(FILE_PATH)]
check("(a) no override, ultimate: force_pN is the LAST segment's rupture (120)",
      summ["force_pN"] == 120.0)
check("(a) no override: dF_pN defaults to last-two ruptures (120-80=40)",
      summ["dF_pN"] == 40.0)
check("(a) no override: seg_dX_iso_nm defaults to last-two (segments[2].isoforce_x_nm(30.7) - ruptures[1].extension_nm(20.0) = 10.7)",
      abs(summ["dX_iso_nm"] - 10.7) < 1e-9)
check("(a) no override: seg_dX_ext_nm defaults to last-two, X1-X2 (ruptures[1].extension_nm(20.0) - ruptures[2].extension_nm(30.0) = -10.0)",
      abs(summ["dX_ext_nm"] - (-10.0)) < 1e-9)

summ_pen = _rp.segment_summary_bulk([FILE_PATH], "penultimate", DB)[_db.normalize_path(FILE_PATH)]
check("(a) no override, penultimate: force_pN is the SECOND-TO-LAST rupture (80)",
      summ_pen["force_pN"] == 80.0)

# ── (a2) the reported rupture's own (x, y) ─────────────────────────────────
# force_pN and both extensions are read off ONE Rupture. A row that paired a
# force with an extension from a different rupture is the defect these exist
# to remove, so the check is that they move together, not that each is right
# on its own.
check("(a2) ultimate: x_rupture_nm is ruptures[2].extension_nm (30)",
      summ["x_rupture_nm"] == 30.0)
check("(a2) ultimate: x_junction_nm is that rupture minus the ONSET segment's "
      "left_extension_nm (30 - 4 = 26), not minus its own segment's (100)",
      abs(summ["x_junction_nm"] - 26.0) < 1e-9)
check("(a2) penultimate: force AND extension both step back to ruptures[1] "
      "(80 pN, x=20, junction 20-4=16)",
      summ_pen["force_pN"] == 80.0
      and summ_pen["x_rupture_nm"] == 20.0
      and abs(summ_pen["x_junction_nm"] - 16.0) < 1e-9)
check("(a2) the two extensions differ by the onset offset alone, the same "
      "constant under either select",
      abs((summ["x_rupture_nm"] - summ["x_junction_nm"])
          - (summ_pen["x_rupture_nm"] - summ_pen["x_junction_nm"])) < 1e-9)

# ── (b) Primary override takes precedence over `select` entirely ───────────
_db.set_primary_segment_idx(fid, 0, PARAMS_V1, DB)
summ_p0 = _rp.segment_summary_bulk([FILE_PATH], "ultimate", DB)[_db.normalize_path(FILE_PATH)]
check("(b) Primary=0 overrides 'ultimate': force_pN is segment 0's rupture (50)",
      summ_p0["force_pN"] == 50.0)
summ_p0_pen = _rp.segment_summary_bulk([FILE_PATH], "penultimate", DB)[_db.normalize_path(FILE_PATH)]
check("(b) Primary=0 overrides 'penultimate' too — same absolute pick regardless of select",
      summ_p0_pen["force_pN"] == 50.0)
check("(b) Primary set alone (no Secondary): dF/isoforce still fall back to the last-two default",
      summ_p0["dF_pN"] == 40.0)
check("(b) Primary=0: the extensions follow the override with the force — "
      "ruptures[0] (x=10), junction 10-4=6",
      summ_p0["x_rupture_nm"] == 10.0
      and abs(summ_p0["x_junction_nm"] - 6.0) < 1e-9)

# ── (c)/(d) Primary+Secondary, non-adjacent: dF works, isoforce doesn't ────
_db.set_secondary_segment_idx(fid, 2, PARAMS_V1, DB)   # primary=0, secondary=2 (non-adjacent)
summ_pair = _rp.segment_summary_bulk([FILE_PATH], "ultimate", DB)[_db.normalize_path(FILE_PATH)]
check("(c) non-adjacent pair (0,2): dF_pN is a plain difference (120-50=70), adjacency-agnostic",
      summ_pair["dF_pN"] == 70.0)
check("(d) non-adjacent pair (0,2): seg_dX_iso_nm is None — isoforce_x_nm has no meaning here",
      summ_pair["dX_iso_nm"] is None)
check("(d) the shared display/summary resolver rejects the same non-adjacent pair",
      _rp.resolve_isoforce_pair(3, 0, 2) is None)
check("(d) non-adjacent pair (0,2): seg_dX_ext_nm IS computed (10.0-30.0=-20.0) — plain "
      "subtraction needs no adjacency, unlike seg_dX_iso_nm",
      summ_pair["dX_ext_nm"] is not None and abs(summ_pair["dX_ext_nm"] - (-20.0)) < 1e-9)

# ── (e) click-order independence: swap which one is tagged Primary/Secondary ─
_db.set_primary_segment_idx(fid, 2, PARAMS_V1, DB)     # now primary=2, secondary=0 (reversed)
_db.set_secondary_segment_idx(fid, 0, PARAMS_V1, DB)
summ_swapped = _rp.segment_summary_bulk([FILE_PATH], "ultimate", DB)[_db.normalize_path(FILE_PATH)]
check("(e) swapping which segment is tagged Primary vs Secondary gives the SAME dF_pN (70, not -70)",
      summ_swapped["dF_pN"] == 70.0)
check("(e) swapped pair: seg_dX_iso_nm is still None (still non-adjacent, order doesn't matter)",
      summ_swapped["dX_iso_nm"] is None)
check("(e) swapping Primary/Secondary gives the SAME seg_dX_ext_nm (-20.0, not +20.0) — "
      "resolved by rupture index, not by which one was clicked",
      abs(summ_swapped["dX_ext_nm"] - (-20.0)) < 1e-9)

# ── adjacent pair sanity check: (1,2) matches the (a) default exactly ──────
_db.set_primary_segment_idx(fid, 1, PARAMS_V1, DB)
_db.set_secondary_segment_idx(fid, 2, PARAMS_V1, DB)
summ_adj = _rp.segment_summary_bulk([FILE_PATH], "ultimate", DB)[_db.normalize_path(FILE_PATH)]
check("adjacent pair (1,2): dF_pN matches the (a) default exactly (force[2]-force[1]=40)",
      summ_adj["dF_pN"] == 40.0)
check("adjacent pair (1,2): seg_dX_iso_nm is computed (not None) for an adjacent pick",
      summ_adj["dX_iso_nm"] is not None and abs(summ_adj["dX_iso_nm"] - 10.7) < 1e-9)
check("adjacent pair (1,2): the shared display/summary resolver selects that pair",
      _rp.resolve_isoforce_pair(3, 1, 2) == (1, 2))
check("adjacent pair roles are order-independent in the shared resolver",
      _rp.resolve_isoforce_pair(3, 2, 1) == (1, 2))
check("an incomplete manual pair preserves the last-two default",
      _rp.resolve_isoforce_pair(3, 0, None) == (1, 2))
check("adjacent pair (1,2): seg_dX_ext_nm matches the (a) default exactly (20.0-30.0=-10.0)",
      abs(summ_adj["dX_ext_nm"] - (-10.0)) < 1e-9)

# ── (f) staleness: reanalysis under new params invalidates the old override ─
PARAMS_V2 = json.dumps({"tag": "v2"})
_db.write_event_map(fid, json.dumps(_payload("v2")), PARAMS_V2, CODE_VER, DB)
summ_stale = _rp.segment_summary_bulk([FILE_PATH], "ultimate", DB)[_db.normalize_path(FILE_PATH)]
check("(f) after resegmentation (new params), the old Primary/Secondary pick is IGNORED",
      summ_stale["force_pN"] == 120.0 and summ_stale["dF_pN"] == 40.0)

# A fresh pick made against the CURRENT segmentation works again immediately.
_db.set_primary_segment_idx(fid, 0, PARAMS_V2, DB)
summ_fresh = _rp.segment_summary_bulk([FILE_PATH], "ultimate", DB)[_db.normalize_path(FILE_PATH)]
check("(f) a NEW pick made against the current event_map is honoured",
      summ_fresh["force_pN"] == 50.0)

# ── (g) Save Queue / Load Queue round-trip ──────────────────────────────────
# db.queue_paths supplies the paths; export_utils.ExportGroup writes the file
# (db.py writes no export files itself — one writer, and
# every export carries a manifest). Load Queue must still read it back.
_db.enqueue_files([fid], DB)
from smfs_catalog import export_utils as _export
with _export.export_group(DB, "queue", [".csv"], kind="queue_save") as _g:
    saved_paths = _db.queue_paths(DB)
    _g.contributing_files(saved_paths)
    _g.table(".csv", ["path"], [[p] for p in saved_paths])
csv_path = str(_g.path(".csv"))
check("(g) queue save writes one row for the one queued file", len(saved_paths) == 1)
check("(g) queue save wrote a manifest beside the CSV",
      _g.path("_manifest.json").exists())

_db.clear_analysis_queue(DB)
check("(g) queue is empty after clear_analysis_queue", len(_db.list_queue(DB)) == 0)

with open(csv_path, newline="", encoding="utf-8") as f:
    import csv as _csv
    paths = [row["path"] for row in _csv.DictReader(f)]
n_enqueued, n_missing = _db.import_queue_from_paths(paths, DB)
check("(g) import_queue_from_paths re-enqueues the one saved file", n_enqueued == 1)
check("(g) import_queue_from_paths reports zero missing (file still in this DB)", n_missing == 0)
restored = _db.list_queue(DB)
check("(g) the restored queue contains exactly the original file",
      len(restored) == 1 and restored[0]["path"] == _db.normalize_path(FILE_PATH))


# Every check above becomes its own named pytest case.  Must be last:
# pytest_cases reads what the module body recorded.
test_check = checkstyle.pytest_cases(check)
