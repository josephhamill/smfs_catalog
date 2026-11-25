# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Quality pass 1, step c7: segment_summary_bulk -> named stages, same dict.

Two halves.  The behaviour pins cover the branches neither the characterization
snapshot nor test_segment_override.py reach (they pass before and after the
split).  The size guard fails until segment_summary_bulk is split.
"""
import ast
import inspect
import json
import os
from pathlib import Path

import pytest

from smfs_catalog import db as _db
from smfs_catalog import roi_pipeline
from smfs_catalog.provenance import cache_version
from smfs_catalog.roi_events import _PAYLOAD_VERSION, payload_to_events
from smfs_catalog.roi_selection import event_geometry_identity
from size_limits import violations

summary = roi_pipeline.segment_summary_bulk
CODE_VER = cache_version() or "test-build"
PARAMS = json.dumps({"tag": "p"})

ROW_KEYS = {
    "l_p_nm", "l_c_nm", "l_p_err", "l_c_err", "b_nm", "b_err", "k_pN", "k_err",
    "chain_model", "force_pN", "x_rupture_nm", "x_junction_nm", "dF_pN",
    "dX_iso_nm", "dX_ext_nm", "n_segments", "tau", "z_max", "x_max_nm",
    "edge_pinned", "loading_rate_pN_s", "loading_stiffness_pN_nm",
    "loading_rate_err_pN_s", "loading_stiffness_err_pN_nm", "rate_tau",
    "fit_status", "fit_detail",
}


# -- fixtures / builders ------------------------------------------------------

def _seg(i, l_c, **extra):
    """A segment dict; extra keys override fields.  Shorthands: iso_x, edge, onset."""
    d = {
        "left_idx": i * 100, "right_idx": i * 100 + 90,
        "left_piezo_nm": float(i * 100), "right_piezo_nm": float(i * 100 + 90),
        "l_p_nm": 0.4, "l_c_nm": l_c, "l_p_err": 0.05, "l_c_err": 2.0,
        "isoforce_x_nm": extra.pop("iso_x", None), "tau": 10.0 * (i + 1),
        "x_max_nm": None if l_c is None else l_c * 0.8,
        "left_extension_nm": extra.pop("onset", None),
        "edge_pinned": extra.pop("edge", None),
    }
    d.update(extra)
    return d


def _rup(i, force, ext):
    return {
        "idx": i * 100 + 90, "piezo_nm": float(i * 100 + 90), "d1_height": 1.0,
        "prominence": 1.0, "force_pN": force, "force_idx": i * 100 + 85,
        "rise_idx": i * 100 + 80, "fall_idx": i * 100 + 92, "extension_nm": ext,
    }


def _roi(rups, segs, base=0):
    return {
        "onset_idx": base, "return_idx": base + 400,
        "onset_piezo_nm": float(base), "return_piezo_nm": float(base + 400),
        "ruptures": rups, "segments": segs,
    }


def _payload(*rois, v=_PAYLOAD_VERSION):
    return {"v": v, "detector": "test", "rois": list(rois)}


def _three_roi():
    """Forces [50, 80, 120], extensions [10, 20, 30]; onset 4."""
    return _roi(
        [_rup(0, 50.0, 10.0), _rup(1, 80.0, 20.0), _rup(2, 120.0, 30.0)],
        [_seg(0, 15.0, onset=4.0), _seg(1, 25.0, iso_x=20.5, edge=True, onset=100.0),
         _seg(2, 35.0, iso_x=30.7, edge=False, onset=200.0)],
    )


class Catalog:
    def __init__(self, db):
        self.db = db
        _db.initialise(db)
        self.fid = {}

    def add(self, path, payload=None):
        path = _db.normalize_path(path)
        conn = _db.get_connection(self.db)
        with conn:
            conn.execute(
                "INSERT INTO files (path, filename, first_seen, last_seen, event)"
                " VALUES (?, ?, datetime('now'), datetime('now'), 'event')",
                (path, os.path.basename(path)))
        self.fid[path] = conn.execute(
            "SELECT id FROM files WHERE path=?", (path,)).fetchone()[0]
        conn.close()
        if payload is not None:
            _db.write_event_map(self.fid[path], json.dumps(payload), PARAMS,
                                CODE_VER, self.db)
        return path


@pytest.fixture
def cat(tmp_path):
    return Catalog(str(tmp_path / "t.sqlite"))


def _none_row():
    return {k: None for k in ROW_KEYS}


# -- signature ----------------------------------------------------------------

def test_signature_is_paths_select_db_path():
    params = list(inspect.signature(summary).parameters.values())
    assert [p.name for p in params] == ["paths", "select", "db_path"]
    assert all(p.default is p.empty and p.kind is p.POSITIONAL_OR_KEYWORD
               for p in params)


# -- empty / absent inputs ----------------------------------------------------

def test_empty_input_gives_empty_dict(cat):
    assert summary([], "ultimate", cat.db) == {}


def test_empty_input_does_not_validate_select(cat):
    assert summary([], "bogus", cat.db) == {}


def test_path_missing_from_catalog_gives_all_none_row(cat):
    out = summary(["/tank/nowhere/a.ibw"], "ultimate", cat.db)
    assert out == {_db.normalize_path("/tank/nowhere/a.ibw"): _none_row()}


def test_missing_path_does_not_validate_select(cat):
    out = summary(["/tank/nowhere/a.ibw"], "bogus", cat.db)
    assert list(out.values()) == [_none_row()]


def test_file_without_event_map_gives_all_none_row_and_no_select_check(cat):
    p = cat.add("/tank/x/noev.ibw")
    assert summary([p], "bogus", cat.db) == {p: _none_row()}


def test_wrong_schema_version_gives_all_none_row(cat):
    p = cat.add("/tank/x/old.ibw", _payload(_three_roi(), v=_PAYLOAD_VERSION - 1))
    assert summary([p], "ultimate", cat.db) == {p: _none_row()}


def test_roi_without_ruptures_gives_all_none_row_and_no_select_check(cat):
    p = cat.add("/tank/x/norup.ibw", _payload(_roi([], [])))
    assert summary([p], "bogus", cat.db) == {p: _none_row()}


def test_no_rois_at_all_gives_all_none_row(cat):
    p = cat.add("/tank/x/norois.ibw", _payload())
    assert summary([p], "ultimate", cat.db) == {p: _none_row()}


# -- select validated lazily, only once a curve reaches the choice -----------

def test_bogus_select_raises_value_error_when_a_curve_has_ruptures(cat):
    p = cat.add("/tank/x/a.ibw", _payload(_three_roi()))
    with pytest.raises(ValueError):
        summary([p], "bogus", cat.db)


def test_bogus_select_raises_when_mixed_with_empty_curves(cat):
    empty = cat.add("/tank/x/e.ibw")
    full = cat.add("/tank/x/f.ibw", _payload(_three_roi()))
    with pytest.raises(ValueError):
        summary([empty, full], "bogus", cat.db)


# -- a fully populated curve, both selects ------------------------------------

def test_full_row_ultimate_exact_values(cat):
    p = cat.add("/tank/x/a.ibw", _payload(_three_roi()))
    row = summary([p], "ultimate", cat.db)[p]
    assert set(row) == ROW_KEYS
    expected = _none_row()
    expected.update(
        l_p_nm=0.4, l_c_nm=35.0, l_p_err=0.05, l_c_err=2.0, chain_model="wlc",
        force_pN=120.0, x_rupture_nm=30.0, x_junction_nm=26.0,
        dF_pN=40.0, dX_ext_nm=10.0, n_segments=3, tau=30.0,
        x_max_nm=pytest.approx(28.0), edge_pinned=0.0,
        fit_status="fit_available",
        dX_iso_nm=pytest.approx(10.7), z_max=pytest.approx(28.0 / 35.0),
    )
    # z_max is derived from x_max/l_c by the Segment, never stored.
    assert row == expected
    assert isinstance(row["edge_pinned"], float)


def test_penultimate_moves_segment_and_rupture_but_not_pair_metrics(cat):
    p = cat.add("/tank/x/a.ibw", _payload(_three_roi()))
    u = summary([p], "ultimate", cat.db)[p]
    pen = summary([p], "penultimate", cat.db)[p]
    assert (pen["l_c_nm"], pen["force_pN"], pen["tau"]) == (25.0, 80.0, 20.0)
    assert (pen["x_rupture_nm"], pen["x_junction_nm"]) == (20.0, 16.0)
    assert pen["edge_pinned"] == 1.0
    for k in ("dF_pN", "dX_iso_nm", "dX_ext_nm", "n_segments"):
        assert pen[k] == u[k]


# -- ROI choice ---------------------------------------------------------------

def test_rightmost_roi_with_ruptures_is_used_and_trailing_empty_roi_skipped(cat):
    early = _roi([_rup(0, 5.0, 1.0)], [_seg(0, 11.0, onset=0.0)])
    late = _three_roi()
    trailing = _roi([], [], base=900)
    p = cat.add("/tank/x/r.ibw", _payload(early, late, trailing))
    row = summary([p], "ultimate", cat.db)[p]
    assert row["n_segments"] == 3 and row["force_pN"] == 120.0


def test_earlier_roi_is_ignored_when_a_later_one_has_ruptures(cat):
    early = _three_roi()
    late = _roi([_rup(0, 7.0, 2.0)], [_seg(0, 12.0, onset=1.0)])
    p = cat.add("/tank/x/r.ibw", _payload(early, late))
    row = summary([p], "ultimate", cat.db)[p]
    assert row["n_segments"] == 1 and row["force_pN"] == 7.0


# -- single segment -----------------------------------------------------------

def test_single_segment_penultimate_falls_back_to_the_same_segment(cat):
    roi = _roi([_rup(0, 33.0, 9.0)], [_seg(0, 12.0, edge=None, onset=2.0)])
    p = cat.add("/tank/x/s.ibw", _payload(roi))
    u = summary([p], "ultimate", cat.db)[p]
    pen = summary([p], "penultimate", cat.db)[p]
    assert u == pen
    assert (u["force_pN"], u["n_segments"], u["x_junction_nm"]) == (33.0, 1, 7.0)


def test_single_segment_has_no_pair_metrics_and_none_edge_pinned(cat):
    roi = _roi([_rup(0, 33.0, 9.0)], [_seg(0, 12.0, edge=None, onset=2.0)])
    p = cat.add("/tank/x/s.ibw", _payload(roi))
    row = summary([p], "ultimate", cat.db)[p]
    assert row["dF_pN"] is None and row["dX_iso_nm"] is None
    assert row["dX_ext_nm"] is None
    assert row["edge_pinned"] is None


def test_two_segments_penultimate_picks_the_first_segment(cat):
    roi = _roi([_rup(0, 10.0, 1.0), _rup(1, 20.0, 2.0)],
               [_seg(0, 11.0, onset=0.5), _seg(1, 22.0, iso_x=2.5, onset=50.0)])
    p = cat.add("/tank/x/two.ibw", _payload(roi))
    pen = summary([p], "penultimate", cat.db)[p]
    assert (pen["l_c_nm"], pen["force_pN"], pen["dF_pN"]) == (11.0, 10.0, 10.0)


# -- None pieces inside an otherwise present curve ---------------------------

def test_missing_rupture_extension_gives_none_x_fields_and_keeps_dF(cat):
    roi = _roi([_rup(0, 50.0, 10.0), _rup(1, 80.0, None)],
               [_seg(0, 15.0, onset=4.0), _seg(1, 25.0, iso_x=20.5, onset=9.0)])
    p = cat.add("/tank/x/m.ibw", _payload(roi))
    row = summary([p], "ultimate", cat.db)[p]
    assert row["force_pN"] == 80.0
    assert row["x_rupture_nm"] is None and row["x_junction_nm"] is None
    assert row["dF_pN"] == 30.0


def test_missing_onset_extension_gives_none_junction_but_keeps_rupture_x(cat):
    roi = _roi([_rup(0, 50.0, 10.0)], [_seg(0, 15.0, onset=None)])
    p = cat.add("/tank/x/m.ibw", _payload(roi))
    row = summary([p], "ultimate", cat.db)[p]
    assert row["x_rupture_nm"] == 10.0 and row["x_junction_nm"] is None


def test_unfitted_segment_reports_status_and_none_fit_values(cat):
    roi = _roi([_rup(0, 50.0, 10.0)],
               [_seg(0, None, onset=4.0, l_p_nm=None, fit_status="fit_failed",
                     fit_detail="why", tau=None)])
    p = cat.add("/tank/x/u.ibw", _payload(roi))
    row = summary([p], "ultimate", cat.db)[p]
    assert row["l_p_nm"] is None and row["l_c_nm"] is None
    assert row["fit_status"] == "fit_failed" and row["fit_detail"] == "why"
    assert row["force_pN"] == 50.0


def test_non_default_chain_model_and_extra_fit_fields_are_passed_through(cat):
    seg = _seg(0, 15.0, onset=4.0, chain_model="efjc", b_nm=0.3, b_err=0.01,
               k_pN=900.0, k_err=5.0, loading_rate_pN_s=100.0,
               loading_stiffness_pN_nm=2.0, loading_rate_err_pN_s=3.0,
               loading_stiffness_err_pN_nm=0.1, rate_tau=4.0)
    p = cat.add("/tank/x/c.ibw", _payload(_roi([_rup(0, 50.0, 10.0)], [seg])))
    row = summary([p], "ultimate", cat.db)[p]
    assert (row["chain_model"], row["b_nm"], row["b_err"]) == ("efjc", 0.3, 0.01)
    assert (row["k_pN"], row["k_err"]) == (900.0, 5.0)
    assert (row["loading_rate_pN_s"], row["loading_stiffness_pN_nm"]) == (100.0, 2.0)
    assert (row["loading_rate_err_pN_s"], row["loading_stiffness_err_pN_nm"]) == (3.0, 0.1)
    assert row["rate_tau"] == 4.0


# -- overrides: branches test_segment_override.py does not reach -------------

def _pair_cat(cat, roi, primary, secondary):
    p = cat.add("/tank/x/o.ibw", _payload(roi))
    _db.set_primary_segment_idx(cat.fid[p], primary, PARAMS, cat.db)
    _db.set_secondary_segment_idx(cat.fid[p], secondary, PARAMS, cat.db)
    return p


def test_pair_with_a_missing_force_gives_none_dF_but_keeps_dX_ext(cat):
    roi = _three_roi()
    roi["ruptures"][0]["force_pN"] = None
    p = _pair_cat(cat, roi, 0, 2)
    row = summary([p], "ultimate", cat.db)[p]
    assert row["dF_pN"] is None and row["dX_ext_nm"] == 20.0


def test_pair_with_a_missing_extension_gives_none_dX_ext_but_keeps_dF(cat):
    roi = _three_roi()
    roi["ruptures"][2]["extension_nm"] = None
    p = _pair_cat(cat, roi, 0, 2)
    row = summary([p], "ultimate", cat.db)[p]
    assert row["dX_ext_nm"] is None and row["dF_pN"] == 70.0


def test_adjacent_pair_with_missing_isoforce_gives_none_dX_iso(cat):
    roi = _three_roi()
    roi["segments"][1]["isoforce_x_nm"] = None
    p = _pair_cat(cat, roi, 0, 1)
    row = summary([p], "ultimate", cat.db)[p]
    assert row["dX_iso_nm"] is None and row["dF_pN"] == 30.0


def test_adjacent_pair_with_missing_rupture_extension_gives_none_dX_iso(cat):
    roi = _three_roi()
    roi["ruptures"][0]["extension_nm"] = None
    p = _pair_cat(cat, roi, 0, 1)
    row = summary([p], "ultimate", cat.db)[p]
    assert row["dX_iso_nm"] is None and row["dX_ext_nm"] is None


def test_secondary_alone_leaves_the_default_pair_and_the_select_choice(cat):
    p = cat.add("/tank/x/o.ibw", _payload(_three_roi()))
    _db.set_secondary_segment_idx(cat.fid[p], 0, PARAMS, cat.db)
    row = summary([p], "ultimate", cat.db)[p]
    assert row["force_pN"] == 120.0 and row["dF_pN"] == 40.0


def test_out_of_range_primary_is_ignored(cat):
    p = cat.add("/tank/x/o.ibw", _payload(_three_roi()))
    _db.set_primary_segment_idx(cat.fid[p], 7, PARAMS, cat.db)
    row = summary([p], "ultimate", cat.db)[p]
    assert row["force_pN"] == 120.0


def test_geometry_tagged_pick_is_honoured_for_current_geometry(cat):
    payload = _payload(_three_roi())
    p = cat.add("/tank/x/g.ibw", payload)
    geom = event_geometry_identity(payload_to_events(payload))
    _db.set_primary_segment_idx(cat.fid[p], 0, geom, cat.db)
    assert summary([p], "ultimate", cat.db)[p]["force_pN"] == 50.0


def test_geometry_tagged_pick_is_ignored_for_other_geometry(cat):
    p = cat.add("/tank/x/g.ibw", _payload(_three_roi()))
    _db.set_primary_segment_idx(cat.fid[p], 0, "geometry:v1:deadbeef", cat.db)
    assert summary([p], "ultimate", cat.db)[p]["force_pN"] == 120.0


def test_override_applies_only_to_its_own_curve(cat):
    a = cat.add("/tank/x/a.ibw", _payload(_three_roi()))
    b = cat.add("/tank/x/b.ibw", _payload(_three_roi()))
    _db.set_primary_segment_idx(cat.fid[a], 0, PARAMS, cat.db)
    out = summary([a, b], "ultimate", cat.db)
    assert (out[a]["force_pN"], out[b]["force_pN"]) == (50.0, 120.0)


# -- keys, order, duplicates, chunking ----------------------------------------

def test_output_is_keyed_by_normalized_path_in_input_order(cat):
    a = cat.add("/tank/x/a.ibw", _payload(_three_roi()))
    b = cat.add("/tank/x/b.ibw")
    c = cat.add("/tank/x/c.ibw", _payload(_three_roi()))
    missing = _db.normalize_path("/tank/x/zz.ibw")
    out = summary([c, missing, a, b], "ultimate", cat.db)
    assert list(out) == [c, missing, a, b]


def test_unnormalised_input_path_is_keyed_by_its_normalised_form(cat):
    a = cat.add("/tank/x/a.ibw", _payload(_three_roi()))
    raw = "/tank/x/../x/a.ibw"
    out = summary([raw], "ultimate", cat.db)
    assert list(out) == [_db.normalize_path(raw)]
    assert out[_db.normalize_path(raw)] == summary([a], "ultimate", cat.db)[a]


def test_duplicate_input_paths_collapse_to_one_entry_at_first_position(cat):
    a = cat.add("/tank/x/a.ibw", _payload(_three_roi()))
    b = cat.add("/tank/x/b.ibw", _payload(_three_roi()))
    out = summary([a, b, a], "ultimate", cat.db)
    assert list(out) == [a, b]
    assert out[a]["force_pN"] == 120.0


def test_rows_are_independent_dicts(cat):
    a = cat.add("/tank/x/a.ibw")
    b = cat.add("/tank/x/b.ibw")
    out = summary([a, b], "ultimate", cat.db)
    out[a]["force_pN"] = 1.0
    assert out[b]["force_pN"] is None


def test_catalog_larger_than_one_sql_chunk_is_fully_resolved(cat):
    conn = _db.get_connection(cat.db)
    paths = [_db.normalize_path(f"/tank/many/f{i:04d}.ibw") for i in range(850)]
    with conn:
        conn.executemany(
            "INSERT INTO files (path, filename, first_seen, last_seen, event)"
            " VALUES (?, ?, datetime('now'), datetime('now'), 'event')",
            [(p, os.path.basename(p)) for p in paths])
    fid = {r["path"]: r["id"] for r in conn.execute("SELECT id, path FROM files")}
    conn.close()
    for p in (paths[0], paths[849]):
        _db.write_event_map(fid[p], json.dumps(_payload(_three_roi())), PARAMS,
                            CODE_VER, cat.db)
    out = summary(paths, "ultimate", cat.db)
    assert list(out) == paths
    hits = [p for p, r in out.items() if r["force_pN"] is not None]
    assert hits == [paths[0], paths[849]]


# -- source-size guard (CLAUDE.md limits), scoped to what c7 touches ----------

# Every top-level function roi_pipeline.py had at HEAD d939860 (before c7).
# Helpers are found by call-reachability from segment_summary_bulk among the
# functions NOT in this set, so helpers c6 adds elsewhere never count.
PRE_C7_FUNCTIONS = {
    "event_params_from", "event_map_params_json", "compute_curve_events",
    "_can_read_cache", "_scalar_keys", "_cached_scalar", "_store_scalars",
    "_cached_events", "_offset_for", "_snapoff_for", "_band_for",
    "_built_events", "_fresh_invols", "_invols_for", "_fit_and_store",
    "_decomposed_and_signals", "compute_curve_events_coords",
    "populate_event_map", "assemble_rows", "segment_summary_bulk",
    "roi_count_histogram", "coverage",
}


def _functions() -> dict:
    tree = ast.parse(Path(roi_pipeline.__file__).read_text(encoding="utf-8"))
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _called_names(fn) -> set:
    return {n.func.id for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def _c7_helpers() -> dict:
    funcs = _functions()
    new = set(funcs) - PRE_C7_FUNCTIONS
    found, todo = {}, [funcs["segment_summary_bulk"]]
    while todo:
        for name in _called_names(todo.pop()) & new:
            if name not in found:
                found[name] = funcs[name]
                todo.append(funcs[name])
    return found


def test_segment_summary_bulk_is_split_into_named_stages():
    assert len(_c7_helpers()) >= 4


def test_segment_summary_bulk_is_within_the_size_limits():
    assert violations("segment_summary_bulk", _functions()["segment_summary_bulk"]) == []


def test_every_c7_helper_is_within_the_size_limits():
    found = [v for n, f in _c7_helpers().items() for v in violations(n, f)]
    assert found == []
