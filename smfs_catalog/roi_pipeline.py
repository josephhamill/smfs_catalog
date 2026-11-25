# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/roi_pipeline.py
#
# Computation, cache orchestration, persistence, and stored-result queries for
# multi-event ROIs. Presentation policy and manual segment-pick resolution live
# in roi_selection; names are re-exported here for callers that import them.
#
# ONE place computes a curve's full events-within-events structure (so the
# viewer and the batch populate step can never drift):
#
#   compute_curve_events(curve, ep) -> CurveEvents   (detect + segment + fit)
#
# populate_event_map(paths, db_path) loads each curve, computes, and writes the
# readable-JSON document to the event_map table — keyed by (params, code_ver) so
# a settings or code change re-derives lazily instead of mass-invalidating.
#
# This module reads DB settings (unlike the pure roi_events / roi_assembly) but
# has no Qt dependency, so it is safe from a worker thread or a script.

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict

from . import db as _db
from .analysis_params import AnalysisParams
from .curve_analysis import CacheContext, cache_get, cache_put_multi, can_use_cache
from .curve_loader import LoadError, load_force_curve
from .models import CHAIN_MODEL_BY_IDX
from .muggeo_segmented import RESTARTS as MUGGEO_RESTARTS
from .muggeo_segmented import TOLERANCES as MUGGEO_TOLERANCES
from .rupture_algorithms import RUPTURE_ALGORITHM_BY_IDX
from .provenance import cache_version
from .roi_selection import (
    event_geometry_identity,
    read_segment_select,  # noqa: F401  re-export
    resolve_reported_segment,
    resolve_isoforce_pair,
    resolve_segment_override,
    resolve_segment_override_state,  # noqa: F401  re-export
    write_segment_select,  # noqa: F401  re-export
)


# ── Parameters ────────────────────────────────────────────────────────────────

# THE ONE map from the on-disk roi_detector_mode_idx integer to a detector name.
# The ROI viewer (display_roi.py) imports this rather than keeping its own copy,
# so the worker and the viewer can never decode the same stored value two
# different ways.
#
# Index 0 is retained as a read-only compatibility alias for "threshold".
# Indices 1 and 2 are the canonical stored detector modes.
DETECTOR_BY_IDX = {0: "threshold", 1: "threshold", 2: "find_peaks"}

# UI-visible options (label, mode) — the viewer's combo box is built from this.
DETECTOR_MODE_LABELS = [
    ("multi: threshold",  "threshold"),
    ("multi: find_peaks", "find_peaks"),
]

# Canonical index to WRITE for a given mode, using the same on-disk numbering as
# DETECTOR_BY_IDX above. Index 0 is read for compatibility but never written.
MODE_TO_STORED_IDX = {"threshold": 1, "find_peaks": 2}


@dataclass(frozen=True)
class EventParams:
    """Every setting the multi-event computation reads."""
    analysis_revision: str
    anchor_nm:         float
    cutoff_hz:         float
    trim_pts:          int
    var_window_ms:     float
    thresh_retr:       float
    window_pts:        int
    d1_threshold:      float     # outer/terminal threshold (ROI-validating)
    inner_threshold:   float     # inner sub-event threshold (<= d1_threshold)
    post_mask_nm:      float
    onset_thr:         float
    detector:          str
    prominence:        float
    distance_pts:      int
    chain_model:       str       # a models.CHAIN_MODELS key
    rupture_algorithm: str       # a rupture_algorithms.RUPTURE_ALGORITHMS key
    walk_back_fit_widths:     int
    walk_back_noise_multiple: float
    muggeo_fit_window_multiplier: int
    muggeo_tolerance:  float     # a muggeo_segmented.TOLERANCES value
    muggeo_restarts:   bool
    invols_offset_pts: int
    invols_window_pts: int

    def fit_settings(self):
        from .roi_events import FitSettings
        return FitSettings(
            chain_model=self.chain_model,
            rupture_algorithm=self.rupture_algorithm,
            walk_back_fit_widths=self.walk_back_fit_widths,
            walk_back_noise_multiple=self.walk_back_noise_multiple,
            muggeo_fit_window_multiplier=self.muggeo_fit_window_multiplier,
            muggeo_tolerance=self.muggeo_tolerance,
            muggeo_restarts=self.muggeo_restarts,
        )


def event_params_from(
    params: AnalysisParams,
    detector: str | None = None,
) -> EventParams:
    """
    THE parameter set in force, as EventParams.  `detector` overrides the
    stored mode; otherwise the stored roi_detector_mode_idx is used.

    There is no owner or DB argument: orchestration resolved one snapshot.
    The inner detector is the more sensitive tier, so it is clamped to the
    outer threshold here as a final numerical-boundary safeguard.
    """
    det = detector or DETECTOR_BY_IDX.get(params.roi_detector_mode_idx, "threshold")
    # Inner (sub-event) threshold defaults to the outer one → single-tier unless
    # the user sets a smaller inner value.
    return EventParams(
        analysis_revision=params.revision,
        anchor_nm=params.baseline_anchor_nm,
        cutoff_hz=params.spectral_cutoff_hz,
        trim_pts=params.turnaround_trim_pts,
        var_window_ms=params.var_window_ms,
        thresh_retr=params.detection_threshold_retr,
        window_pts=params.roi_window_pts,
        d1_threshold=params.roi_threshold_nm_per_nm,
        inner_threshold=min(
            params.roi_inner_threshold_nm_per_nm,
            params.roi_threshold_nm_per_nm,
        ),
        post_mask_nm=params.roi_post_snapoff_mask_nm,
        onset_thr=params.roi_onset_threshold_nm,
        detector          = det,
        prominence=params.roi_prominence,
        distance_pts=params.roi_min_distance_pts,
        chain_model=CHAIN_MODEL_BY_IDX[params.chain_model_idx],
        rupture_algorithm=RUPTURE_ALGORITHM_BY_IDX[params.rupture_algorithm_idx],
        walk_back_fit_widths=params.walk_back_fit_widths,
        walk_back_noise_multiple=params.walk_back_noise_multiple,
        muggeo_fit_window_multiplier=params.muggeo_fit_window_multiplier,
        muggeo_tolerance=MUGGEO_TOLERANCES[params.muggeo_tolerance_idx],
        muggeo_restarts=MUGGEO_RESTARTS[params.muggeo_restarts_idx],
        invols_offset_pts=params.invols_offset_pts,
        invols_window_pts=params.invols_window_pts,
    )


@dataclass(frozen=True)
class CurveEventsResult:
    """
    Full output of compute_curve_events_coords, as named fields rather than a
    positional tuple — a positional tuple that grows over time is exactly the
    kind of silent-breakage risk this session's refactor exists to remove (see
    wlc_view_window.py's `events, offset, inv, snap = compute_curve_events_coords(...)`,
    which would have broken the moment this gained a 5th field).

    sigs/dc/snapoff_idx/mask_* are exposed so a diagnostic viewer that needs the
    raw detection-signal arrays for plotting (display_roi.py) can read them off
    THIS call instead of recomputing them a second time.
    """
    events:            object   # roi_events.CurveEvents
    offset:            float
    invols:            float
    snap_piezo:        float
    sigs:              object   # roi_detection.DetectionSignals
    dc:                object   # signal_processing.DecomposedCurve
    snapoff_idx:       int
    mask_postsnap_idx: int
    mask_anchor_idx:   int


def event_map_params_json(ep: EventParams) -> str:
    """Stable cache key for event_map — every field that changes the result."""
    return json.dumps(asdict(ep), sort_keys=True, separators=(",", ":"))


# ── Single-curve computation (the shared source of truth) ─────────────────────

def compute_curve_events(
    curve, ep: EventParams,
    *,
    cache: CacheContext | None = None,
    stage1=None,
    force: bool = False,
):
    """
    Full multi-event structure for one loaded curve: baseline, detection signals,
    snap-off, search band, ROI segmentation, and per-segment WLC fit.  Returns a
    fitted CurveEvents (forces + l_p/l_c filled where fittable).

    See compute_curve_events_coords for the stage1/cache reuse contract.
    """
    return compute_curve_events_coords(
        curve, ep, cache=cache, stage1=stage1, force=force,
    ).events


def _scalar_keys(cache: CacheContext | None):
    # THE canonical key-builder for these rows (curve_analysis.pipeline_params_from)
    # — reused rather than hand-rebuilt, so this can never drift from the keys
    # analyse_curve actually wrote under.
    if can_use_cache(cache) and cache.param_set is not None:
        from .curve_analysis import pipeline_params_from
        return pipeline_params_from(cache.param_set)
    return None


def _cached_scalar(cache: CacheContext | None, atype: str, key_field: str):
    keys = _scalar_keys(cache)
    if keys is None:
        return None
    return cache_get(cache, atype, getattr(keys, key_field))


def _store_scalars(cache: CacheContext | None, key_field: str, values: dict) -> None:
    keys = _scalar_keys(cache)
    if keys is not None:
        cache_put_multi(cache, values, getattr(keys, key_field))


def _cached_events(ep: EventParams, cache: CacheContext | None, force: bool):
    """The stored event_map document under the current signature, or None.

    The signature is asdict(ep) — EVERY parameter the finder reads — so a match
    means the parameters have not changed.  That contract holds only while `ep`
    comes from ONE owner (see event_params_from/db.load_analysis_params).
    """
    from .roi_events import payload_to_events

    if force or not can_use_cache(cache):
        return None
    doc = _db.get_event_map(
        cache.file_id, event_map_params_json(ep), cache.code_ver, cache.db_path,
        conn=cache.conn,
    )
    return None if doc is None else payload_to_events(json.loads(doc))


def _offset_for(curve, ep: EventParams, stage1, cache: CacheContext | None) -> float:
    from .signal_processing import fit_retract_baseline

    if stage1 is not None and stage1.offset is not None:
        return stage1.offset
    cv = _cached_scalar(cache, "offset_retr", "params_bl")
    if cv is not None:
        return float(cv)
    # Fresh computation neither stage1 nor the DB cache had — same
    # persist-what-you-compute treatment as the invOLS stage.
    bl = fit_retract_baseline(curve, ep.anchor_nm)
    if bl.offset == bl.offset:   # not NaN
        _store_scalars(cache, "params_bl", {
            "offset_retr":         float(bl.offset),
            "flatness_slope":      float(bl.slope),
            "baseline_intercept":  float(bl.intercept),
            "baseline_rms":        float(bl.rms),
            "baseline_fit_lo_idx": float(bl.fit_lo_idx),
            "baseline_fit_hi_idx": float(bl.fit_hi_idx),
        })
    return bl.offset


def _snapoff_for(dc, ep: EventParams, stage1, cache: CacheContext | None) -> int:
    from .signal_processing import find_end_in_contact

    if stage1 is not None and stage1.snapoff_idx is not None:
        return stage1.snapoff_idx
    cv = _cached_scalar(cache, "snapoff_idx", "params_cd")
    if cv is not None:
        return int(cv)
    si, _ = find_end_in_contact(
        dc, var_window_ms=ep.var_window_ms, threshold=ep.thresh_retr,
        trim_pts=ep.trim_pts,
    )
    return si


def _band_for(sigs, si: int, ep: EventParams, stage1):
    """Always computed, even on an event-map hit: display_roi's mask overlay needs a valid index."""
    from .roi_detection import rupture_search_bounds

    if (stage1 is not None and stage1.mask_postsnap_idx is not None
            and stage1.mask_anchor_idx is not None):
        return stage1.mask_postsnap_idx, stage1.mask_anchor_idx
    return rupture_search_bounds(sigs.piezo, si, ep.anchor_nm, ep.post_mask_nm)


def _built_events(sigs, band, ep: EventParams, stage1):
    from .roi_events import DetectorSettings, build_curve_events

    settings = DetectorSettings(
        onset_thr=ep.onset_thr, detector=ep.detector,
        outer_threshold=ep.d1_threshold, inner_threshold=ep.inner_threshold,
        prominence=ep.prominence, distance_pts=ep.distance_pts,
    )
    return build_curve_events(
        sigs, band, settings,
        outer_events=(stage1.outer_events if stage1 is not None else None),
    )


def _fresh_invols(curve, dc, ep: EventParams, cache: CacheContext | None):
    from .signal_processing import fit_approach_invols
    from .curve_analysis import CurveAnalysisError

    try:
        fit = fit_approach_invols(
            dc.low_appr, curve.piezo_appr, ep.invols_offset_pts, ep.invols_window_pts,
        )
        inv = fit.slope
        # Persist what you compute: write it and its QC diagnostics now, the
        # same as curve_analysis.analyse_curve does, so the next caller (same
        # params) gets a cache hit instead of re-fitting.
        if inv == inv:   # not NaN
            _store_scalars(cache, "params_invols", {
                "invols_slope":      float(inv),
                "invols_intercept":  float(fit.intercept),
                "invols_rms":        float(fit.rms),
                "invols_fit_lo_idx": float(fit.fit_lo_idx),
                "invols_fit_hi_idx": float(fit.fit_hi_idx),
            })
        return inv
    except Exception as exc:
        raise CurveAnalysisError("approach invOLS", exc) from exc


def _invols_for(curve, dc, ep: EventParams, stage1, cache: CacheContext | None) -> float:
    from .curve_analysis import CurveAnalysisError

    if stage1 is not None and stage1.invols_slope is not None:
        inv = stage1.invols_slope
    else:
        cv = _cached_scalar(cache, "invols_slope", "params_invols")
        inv = float(cv) if cv is not None else _fresh_invols(curve, dc, ep, cache)
    if inv is None or not isinstance(inv, (int, float)) or not math.isfinite(inv) or inv == 0.0:
        raise CurveAnalysisError(
            "approach invOLS",
            ValueError("no finite, non-zero calibration slope was produced"),
        )
    return inv


def _fit_and_store(curve, events, ep: EventParams, calib, cache: CacheContext | None) -> None:
    """Fit the freshly built events in place and persist them (calib is a roi_events.SegmentCalibration).

    This is the only event_map write: _persist_multi_event_roi does no separate write.
    """
    from .roi_events import fit_segments, events_to_payload

    fit_segments(curve, events, calib, ep.fit_settings())
    if can_use_cache(cache):
        _db.write_event_map(
            cache.file_id, json.dumps(events_to_payload(events)),
            event_map_params_json(ep), cache.code_ver, cache.db_path,
            conn=cache.conn,
        )


def _decomposed_and_signals(curve, ep: EventParams, stage1, offset: float):
    """Never stored (large per-sample arrays, cheap to regenerate); computed even on an event-map hit because callers draw them (the ROI window plots dc/sigs)."""
    from .signal_processing import decompose_curve
    from .roi_detection import compute_detection_signals

    dc = stage1.dc if (stage1 is not None and stage1.dc is not None) \
        else decompose_curve(curve, cutoff_hz=ep.cutoff_hz)
    sigs = stage1.sigs if (stage1 is not None and stage1.sigs is not None) \
        else compute_detection_signals(
            low_retr=dc.low_retr - offset, piezo_retr=curve.piezo_retr,
            window_pts=ep.window_pts,
        )
    return dc, sigs


def compute_curve_events_coords(
    curve, ep: EventParams,
    *,
    cache: CacheContext | None = None,
    stage1=None,   # curve_analysis.Stage1Search | None
    force: bool = False,
):
    """
    Like compute_curve_events, but also returns the (offset, invols, snap_piezo)
    transform so callers can draw force-extension overlays that align exactly with
    the fitted segments:
        defl_corr = (defl_retr - offset) / invols
        force     = k * defl_corr
        extension = (piezo_retr - snap_piezo) - defl_corr
    Returns a CurveEventsResult (named fields - see its docstring).

    Reuse order for baseline, snap-off index, search band and invOLS: (1) `stage1`
    (same pass as analyse_curve, zero recompute); (2) the analysis_results cache
    when `cache` carries db_path/code_ver/file_id and param_set; (3) fresh.  No
    cache means always fresh.  The decomposed curve and detection signals are
    never stored, only reused via `stage1`.

    A stored event_map document under the current signature is checked first for
    every caller (the WLC viewer included, without an outer guard); on
    a hit the event build and WLC fit are skipped (dc, sigs, band and invOLS are
    still resolved).  A fresh computation is always written to event_map when
    cache identity is available.  ``force=True`` skips only that document; stage-1
    values and scalar caches stay reusable.  ``cache.conn`` keeps every read and
    write on the caller's connection.
    """
    stored = _cached_events(ep, cache, force)
    offset = _offset_for(curve, ep, stage1, cache)
    dc, sigs = _decomposed_and_signals(curve, ep, stage1, offset)
    si = _snapoff_for(dc, ep, stage1, cache)
    band = _band_for(sigs, si, ep, stage1)
    events = stored if stored is not None else _built_events(sigs, band, ep, stage1)
    inv = _invols_for(curve, dc, ep, stage1, cache)
    snap = float(curve.piezo_retr[si]) if 0 <= si < len(curve.piezo_retr) else 0.0
    if stored is None:
        from .roi_events import SegmentCalibration
        _fit_and_store(curve, events, ep,
                       SegmentCalibration(offset, inv, snap, dc.low_retr), cache)
    return CurveEventsResult(
        events=events, offset=offset, invols=inv, snap_piezo=snap,
        sigs=sigs, dc=dc, snapoff_idx=si,
        mask_postsnap_idx=band[0], mask_anchor_idx=band[1],
    )


# ── Batch populate ────────────────────────────────────────────────────────────

def populate_event_map(
    paths:     list[str],
    db_path:   str,
    detector:  str | None = None,
    progress=None,
    force:     bool = False,
) -> dict:
    """
    Compute + persist multi-event documents for `paths` into event_map.

    Skips curves already cached under the current (params, code_ver) unless
    force=True. Forced runs bypass the event-map document but may reuse valid
    scalar calibration caches. The computation function is the sole writer;
    this batch wrapper supplies its existing connection and does not write a
    second copy. Loads each curve once; per-curve outcomes are tallied, not
    fatal. progress(i, n, path) is called after each curve if given.

    Outcome buckets (kept distinct so nothing fails silently):
      ok         — computed and written
      skipped    — already cached under these (params, code_ver)
      not_curve  — LoadError (e.g. a scan image / force-clamp / truncated file)
      error      — an unexpected failure; the first few reasons are returned
                   in 'errors' as (path, repr) so problems are visible.

    Returns {'ok','skipped','not_curve','error','errors','params_json'}.
    """
    param_set = _db.load_analysis_params(db_path)
    ep = event_params_from(param_set, detector=detector)
    pj = event_map_params_json(ep)
    cv = cache_version()

    n_ok = n_skip = n_notcurve = n_err = 0
    errors: list[tuple[str, str]] = []
    conn = _db.get_connection(db_path)
    try:
        total = len(paths)
        for i, path in enumerate(paths):
            fid = _db.get_file_id(path, db_path, conn=conn)
            if fid is None:
                n_err += 1
                if len(errors) < 20:
                    errors.append((path, "file_id not found"))
            elif not force and _db.get_event_map(fid, pj, cv, db_path, conn=conn) is not None:
                n_skip += 1
            else:
                try:
                    curve = load_force_curve(path)
                except LoadError:
                    n_notcurve += 1
                except Exception as e:                       # noqa: BLE001
                    n_err += 1
                    if len(errors) < 20:
                        errors.append((path, repr(e)))
                else:
                    try:
                        # db_path/code_ver/file_id let this reuse whatever the
                        # worker's own analyse_curve pass already cached for this
                        # file (offset, snap-off index, invOLS) instead of always
                        # recomputing from the raw curve — see
                        # compute_curve_events_coords's reuse-order docstring.
                        compute_curve_events(
                            curve, ep,
                            cache=CacheContext(db_path, cv, fid, conn, param_set=param_set),
                            force=force,
                        )
                        n_ok += 1
                    except Exception as e:                   # noqa: BLE001
                        n_err += 1
                        if len(errors) < 20:
                            errors.append((path, repr(e)))
            if progress is not None:
                progress(i + 1, total, path)
    finally:
        conn.close()

    return {"ok": n_ok, "skipped": n_skip, "not_curve": n_notcurve,
            "error": n_err, "errors": errors, "params_json": pj}


# ── Read side: assemble projected rows from stored documents ──────────────────

def assemble_rows(
    paths:        list[str],
    db_path:      str,
    *,
    mode:         str,
    min_ruptures: int = 1,
    max_ruptures: int | None = None,
    detector:     str | None = None,
) -> list[dict]:
    """
    Read event_map documents for `paths` (under the current params/code) and
    project them into flat rows under `mode` with the rupture-count filter.
    Curves without a stored document are silently skipped (populate them first).
    """
    from .roi_events import payload_to_events
    from .roi_assembly import project_curve_events

    ep = event_params_from(_db.load_analysis_params(db_path), detector=detector)
    pj = event_map_params_json(ep)
    cv = cache_version()

    rows: list[dict] = []
    conn = _db.get_connection(db_path)
    try:
        for path in paths:
            fid = _db.get_file_id(path, db_path, conn=conn)
            if fid is None:
                continue
            doc = _db.get_event_map(fid, pj, cv, db_path, conn=conn)
            if not doc:
                continue
            events = payload_to_events(json.loads(doc))
            if events is None:
                continue
            rows.extend(project_curve_events(
                events, mode=mode, file_id=fid, path=path,
                min_ruptures=min_ruptures, max_ruptures=max_ruptures,
            ))
    finally:
        conn.close()
    return rows


# The one map from a dashboard/variable-window column key to the field
# segment_summary_bulk returns it under.  Both dashboard_window.py (the queue
# table) and variable_window.py (its per-column drill-down) import this rather
# than each keeping their own copy, which would fork the moment one of them
# gained a column.
SEG_SUMMARY_KEYS = (
    "seg_l_p_nm", "seg_l_c_nm", "seg_l_p_err", "seg_l_c_err",
    "seg_b_nm", "seg_b_err", "seg_k_pN", "seg_k_err",
    # force and the two extensions are the reported rupture's own (x, y) —
    # kept adjacent because they are read off one Rupture and must stay so.
    "seg_force_pN", "seg_x_rupture_nm", "seg_x_junction_nm",
    "seg_dF_pN", "seg_dX_iso_nm", "seg_dX_ext_nm",
    "seg_n_segments",
    # Fit-conditioning diagnostics.  Adding a key here makes
    # it a queue column, a variable_window drill-down AND a criteria-gate
    # criterion automatically, because criteria_gate branches generically on
    # this tuple.  That is the point: these become criteria a user may CHOOSE to
    # check — exactly like seg_n_segments — never something the app applies on
    # their behalf.
    #
    # Checking a variable REQUIRES it: the gate ANDs across checked
    # variables, so a missing value is an automatic non-hit and checking
    # seg_z_max silently drops every curve whose fit failed.
    "seg_tau", "seg_z_max", "seg_x_max_nm", "seg_edge_pinned",
    # How fast the reported rupture was loaded.  Kept adjacent because they are
    # one regression's two outputs: rate = stiffness * pulling velocity.
    "seg_loading_rate_pN_s", "seg_loading_stiffness_pN_nm",
    "seg_loading_rate_err_pN_s", "seg_loading_stiffness_err_pN_nm",
    "seg_rate_tau",
)
SEG_SUMMARY_FIELD = {
    "seg_l_p_nm": "l_p_nm", "seg_l_c_nm": "l_c_nm",
    "seg_l_p_err": "l_p_err", "seg_l_c_err": "l_c_err",
    "seg_b_nm": "b_nm", "seg_b_err": "b_err",
    "seg_k_pN": "k_pN", "seg_k_err": "k_err",
    "seg_force_pN": "force_pN",
    "seg_x_rupture_nm": "x_rupture_nm", "seg_x_junction_nm": "x_junction_nm",
    "seg_dF_pN": "dF_pN", "seg_dX_iso_nm": "dX_iso_nm",
    "seg_dX_ext_nm": "dX_ext_nm",
    "seg_n_segments": "n_segments",
    "seg_tau": "tau", "seg_z_max": "z_max", "seg_x_max_nm": "x_max_nm",
    "seg_edge_pinned": "edge_pinned",
    "seg_loading_rate_pN_s": "loading_rate_pN_s",
    "seg_loading_stiffness_pN_nm": "loading_stiffness_pN_nm",
    "seg_loading_rate_err_pN_s": "loading_rate_err_pN_s",
    "seg_loading_stiffness_err_pN_nm": "loading_stiffness_err_pN_nm",
    "seg_rate_tau": "rate_tau",
}


def _empty_summary_row() -> dict[str, "float | None"]:
    return {
        "l_p_nm": None, "l_c_nm": None, "l_p_err": None, "l_c_err": None,
        "b_nm": None, "b_err": None, "k_pN": None, "k_err": None,
        "chain_model": None,
        "force_pN": None, "x_rupture_nm": None, "x_junction_nm": None,
        "dF_pN": None, "dX_iso_nm": None, "dX_ext_nm": None,
        "n_segments": None,
        "tau": None, "z_max": None, "x_max_nm": None, "edge_pinned": None,
        "loading_rate_pN_s": None, "loading_stiffness_pN_nm": None,
        "loading_rate_err_pN_s": None,
        "loading_stiffness_err_pN_nm": None, "rate_tau": None,
        "fit_status": None, "fit_detail": None,
    }


def _fetch_rows(paths_resolved: list[str], db_path: str):
    """
    (path_by_fid, payload_by_fid, params_by_fid, override_by_fid) for the
    resolved paths.

    Bulk by construction.  A per-path db.get_file_id opens and closes its own
    sqlite connection each time, and the queue-table rebuild behind every
    raw-viewer double-click pays that cost again for the whole queue.  So
    path->file_id and the event_map fetch both go through chunked bulk SQL,
    the same way get_derived_results_bulk_latest does.
    """
    conn = _db.get_connection(db_path)
    try:
        path_by_fid: dict[int, str] = {}
        for chunk in _db._sql_chunks(paths_resolved):
            ph = ",".join("?" * len(chunk))
            for r in conn.execute(
                f"SELECT id, path FROM files WHERE path IN ({ph})", chunk
            ).fetchall():
                path_by_fid[r["id"]] = r["path"]
        if not path_by_fid:
            return path_by_fid, {}, {}, {}

        # event_map holds exactly one row per file_id (write_event_map deletes
        # any prior row before inserting), so this is already "the latest" —
        # no need for a per-file MAX(computed_at) subquery.
        fids = list(path_by_fid.keys())
        payload_by_fid: dict[int, str] = {}
        params_by_fid: dict[int, str] = {}
        for chunk in _db._sql_chunks(fids):
            fid_ph = ",".join("?" * len(chunk))
            for r in conn.execute(
                f"SELECT file_id, payload_json, params_json FROM event_map "
                f"WHERE file_id IN ({fid_ph})",
                chunk,
            ).fetchall():
                payload_by_fid[r["file_id"]] = r["payload_json"]
                params_by_fid[r["file_id"]]  = r["params_json"]

        # Manual segment overrides — bulk, for the same reason as above: one
        # query, not one per file.
        override_by_fid = _db.get_segment_overrides_bulk(fids, db_path, conn=conn)
    finally:
        conn.close()
    return path_by_fid, payload_by_fid, params_by_fid, override_by_fid


def _pick_roi(events):
    """
    The right-most outer ROI with ruptures, or None.  Every selected-segment
    field below (l_p/l_c/force_pN, n_segments) comes from this ROI, so an
    earlier ROI is ignored once a later one has ruptures.
    """
    return next((r for r in reversed(events.rois) if r.ruptures), None)


def _picked_pair_metrics(roi, primary_idx, secondary_idx, pair) -> dict:
    """
    dF_pN/dX_ext_nm/dX_iso_nm for a usable Primary+Secondary pick.

    Sorted by actual rupture index, never by which one was tagged Primary vs
    Secondary, so pick order never flips the sign.  dF_pN/dX_ext_nm need no
    adjacency (plain subtractions); dX_iso_nm does, and is None for a
    non-adjacent pair, same as any other missing piece.
    """
    out = {"dF_pN": None, "dX_iso_nm": None, "dX_ext_nm": None}
    lo, hi = sorted((primary_idx, secondary_idx))
    f_lo, f_hi = roi.ruptures[lo].force_pN, roi.ruptures[hi].force_pN
    out["dF_pN"] = None if f_lo is None or f_hi is None else f_hi - f_lo
    # dX_ext_nm is a plain extension-point subtraction — needs no
    # adjacency, same as dF_pN, unlike dX_iso_nm below.
    x_lo, x_hi = roi.ruptures[lo].extension_nm, roi.ruptures[hi].extension_nm
    out["dX_ext_nm"] = None if x_lo is None or x_hi is None else x_hi - x_lo
    # isoforce_x_nm is only ever stored relative to the IMMEDIATELY
    # PRECEDING rupture (roi_events.Segment docstring) — a non-
    # adjacent pair legitimately has no such value to read; None,
    # same as any other missing piece, never fabricated.
    if pair is not None:
        pair_lo, pair_hi = pair
        rup_lo, seg_hi = roi.ruptures[pair_lo], roi.segments[pair_hi]
        out["dX_iso_nm"] = (
            None if rup_lo.extension_nm is None or seg_hi.isoforce_x_nm is None
            else seg_hi.isoforce_x_nm - rup_lo.extension_nm
        )
    return out


def _pair_metrics(roi, primary_idx, secondary_idx) -> dict:
    """
    dF_pN/dX_iso_nm/dX_ext_nm: the gap between two ruptures.

    By default the last two (ultimate minus penultimate, by definition), so
    they are NOT selection-dependent (`select`) — flipping `select` moves only
    l_p_nm/l_c_nm/force_pN.  If Primary AND Secondary are both usable (see
    resolve_segment_override) they are computed from that pair instead.

    dX_ext_nm is the plain ruptures[hi].extension_nm -
    ruptures[lo].extension_nm gap — no crossing search, defined regardless of
    which rupture is stronger, unlike dX_iso_nm (see roi_events.ROI.
    dX_ext_pairs). It exists so dF_pN has an order-independent extension-side
    counterpart to compare against; dX_iso_nm remains the deliberately
    one-directional, force-matched "reload distance" — see roi_events.ROI.
    isoforce_dX_pairs for why that one is NOT made symmetric the same way.
    """
    out = {"dF_pN": None, "dX_iso_nm": None, "dX_ext_nm": None}
    pair = resolve_isoforce_pair(
        len(roi.segments), primary_idx, secondary_idx,
    )
    if primary_idx is not None and secondary_idx is not None:
        out = _picked_pair_metrics(roi, primary_idx, secondary_idx, pair)
    elif pair is not None:
        lo, hi = pair
        out["dF_pN"]     = roi.dF_pairs[hi - 1]
        out["dX_iso_nm"] = roi.isoforce_dX_pairs[hi - 1]
        out["dX_ext_nm"] = roi.dX_ext_pairs[hi - 1]
    return out


def _segment_diagnostics(seg) -> dict:
    """
    tau/z_max/edge_pinned are the SELECTED segment's fit-conditioning
    diagnostics, following the same selection rule as l_p/l_c so that a value
    and its explanation always describe the same fit:
      * tau     — integrated autocorrelation time of that fit's residual, in
                  samples.  l_p_err/l_c_err ALREADY include sqrt(tau);
                  this is here so the size of an error bar can be explained.
      * z_max   — x_max/l_c, how close the fit window got to the WLC pole.  The
                  fit's conditioning number in all but name; below ~0.8 only the
                  product l_p*l_c is well determined.  Derived, never stored.
      * edge_pinned — 1.0 when the force peak sat on the window's right edge, so
                  the rupture force is an underestimate and the fit stopped
                  early.  0.0 otherwise, None if no peak was found.
    All three inform; none of them gates anything by itself.
    """
    return {
        # The three diagnostics describe the SELECTED segment's own fit, so
        # they follow the same Ultimate/Penultimate/override rule as the
        # values they explain — an error bar and the tau behind it must
        # never come from different segments.
        "tau": seg.tau,
        "z_max": seg.z_max,
        # z_max's own numerator. Offered beside it so a low ratio can be
        # read as a short pull or an overlong contour length, rather than
        # only as a number that is too small.
        "x_max_nm": seg.x_max_nm,
        # The selected segment's own loading ramp, following the same rule
        # as the force it explains — a rupture force and the rate it was
        # reached at must never come from different segments.
        "loading_rate_pN_s": seg.loading_rate_pN_s,
        "loading_stiffness_pN_nm": seg.loading_stiffness_pN_nm,
        # Each slope's own uncertainty, following the value it belongs to.
        "loading_rate_err_pN_s": seg.loading_rate_err_pN_s,
        "loading_stiffness_err_pN_nm": seg.loading_stiffness_err_pN_nm,
        "rate_tau": seg.rate_tau,
        # Numeric 0/1, not a bool: this reaches criteria_gate (which bounds
        # numbers) and quantities.format_value (which formats them).  The
        # dataclass keeps the bool; only this projection flattens it.
        "edge_pinned": (
            None if seg.edge_pinned is None else float(bool(seg.edge_pinned))
        ),
    }


def _segment_fields(seg) -> dict:
    """
    The SELECTED segment's fit values, plus its diagnostics.

    l_p_err/l_c_err are the WLC fit's own uncertainty on l_p/l_c; large values
    flag an under-constrained fit.  They inform; they gate nothing by
    themselves.

    fit_status/fit_detail are that same segment's stored fit outcome (see
    roi_events.Segment), so a missing l_c or force can be explained by the
    pipeline's own reason. Strings, so they stay out of SEG_SUMMARY_KEYS and
    are never offered as criteria.
    """
    return {
        "l_p_nm": seg.l_p_nm,
        "l_c_nm": seg.l_c_nm,
        "l_p_err": seg.l_p_err,
        "l_c_err": seg.l_c_err,
        "b_nm": seg.b_nm, "b_err": seg.b_err,
        "k_pN": seg.k_pN, "k_err": seg.k_err,
        # Which model the values above are from, so a view that needs one
        # model's equation can tell.
        "chain_model": seg.chain_model,
        **_segment_diagnostics(seg),
        "fit_status": seg.fit_status,
        "fit_detail": seg.fit_detail,
    }


def _rupture_fields(roi, rup) -> dict:
    """
    force_pN, x_rupture_nm, x_junction_nm of the reported rupture.

    x_rupture_nm/x_junction_nm are the reported rupture's position on the
    extension coordinate — (piezo - snapoff) - deflection, the axis every WLC
    fit runs against — under its two useful zeros:
      * x_rupture_nm  — from snap-off. The junction's end-to-end LENGTH when it
                        broke, and the number commensurable with l_c (z_max is
                        x_max/l_c, so the app already treats this coordinate as
                        the chain's extension).
      * x_junction_nm — from the junction's onset (segments[0].left_extension_nm).
                        How far it STRETCHED under load. Not a length, and not
                        comparable with l_c.
    Both are read off the SAME Rupture as force_pN and therefore follow the
    same select/override rule, so a row's force and extension always describe
    one point.  NOT raw piezo: onset_dx_nm/rupture_dx_nm are that, on a
    different axis and (for the rupture) a different sample.  Pinning
    x_junction_nm to the terminal rupture instead would put one row's force
    and extension on two different ruptures whenever "penultimate" is in force.
    """
    # The x of the SAME point force_pN is the y of — fit_segments sets
    # both from one sample (roi_events: rup.force_pN / rup.extension_nm
    # at peak_rel).  Read off the same `rup` here so a row can never
    # pair a force with an extension from a different rupture.
    # x_junction_nm measures that rupture from the junction's onset instead
    # of from snap-off: segments[0] is the onset segment, so its left edge is
    # the junction's onset whichever segment is reported.
    onset_x = roi.segments[0].left_extension_nm
    return {
        "force_pN": rup.force_pN,
        "x_rupture_nm": rup.extension_nm,
        "x_junction_nm": (
            None if rup.extension_nm is None or onset_x is None
            else rup.extension_nm - onset_x
        ),
    }


def _fill_row(row, roi, select, primary_idx, secondary_idx) -> None:
    """
    Fill one curve's row from its chosen ROI.

    l_p_nm/l_c_nm/l_p_err/l_c_err/force_pN come from the SELECTED segment (and
    its terminating rupture): "ultimate" = its last segment (the tether/final
    rupture, or the whole pull if there's only one segment); "penultimate" =
    the one before it when the ROI has >= 2 segments, else the SAME segment
    ultimate would use — matching base_2dh_window._stored_segment_fit.
    force_pN is the characterization value for "rupture force": whichever
    segment is selected here IS the rupture force, by definition — one
    user-facing force number, not a second parallel one alongside a stage-1
    scalar.  A usable (non-stale) Primary override replaces that choice for
    this curve outright — an absolute per-curve override, not another value
    of `select`.

    n_segments is the segment count of the ROI — NOT selection-dependent (it
    describes the ROI, not the pick within it).  It exists so a caller who
    wants "only curves with a real, distinct penultimate segment" can filter
    on it explicitly (seg_n_segments >= 2) instead of relying on `select` to
    narrow the population silently.
    """
    row["n_segments"] = len(roi.segments)
    row.update(_pair_metrics(roi, primary_idx, secondary_idx))
    # segments/ruptures are always the same length, one segment per
    # terminating rupture (ROI docstring: "Always len(ruptures) segments"),
    # so one length check picks both the segment and its rupture.
    reported = resolve_reported_segment(
        len(roi.segments), select, primary_idx,
    )
    if reported is not None:
        idx = reported.segment_idx
        row.update(_segment_fields(roi.segments[idx]))
        row.update(_rupture_fields(roi, roi.ruptures[idx]))


def segment_summary_bulk(
    paths: list[str], select: str, db_path: str,
) -> dict[str, dict[str, "float | None"]]:
    """
    Per-path dict of the keys in _empty_summary_row (the SEG_SUMMARY_FIELD
    values plus chain_model, fit_status, fit_detail), read from each curve's
    latest event_map (whatever params/code produced it — like the 2DH windows,
    this never silently drops a curve pending a recompute).

    Any missing piece (no file, no document, wrong schema version) yields
    None fields — never a fabricated value.  A present-but-single-segment
    ROI under "penultimate" is not one of those: it is the documented
    fallback, not an absence.  Keyed by resolved path (db.normalize_path),
    matching get_derived_results_bulk_latest.  The per-field rules are in
    _fetch_rows, _pair_metrics, _segment_fields, _rupture_fields, _fill_row.
    """
    from .roi_events import payload_to_events

    out = {_db.normalize_path(p): _empty_summary_row() for p in paths}
    if not paths:
        return out
    path_by_fid, payload_by_fid, params_by_fid, override_by_fid = _fetch_rows(
        list(out.keys()), db_path,
    )
    for fid, doc in payload_by_fid.items():
        events = payload_to_events(json.loads(doc))
        roi = None if events is None else _pick_roi(events)
        if roi is None:
            continue
        primary_idx, secondary_idx = resolve_segment_override(
            override_by_fid.get(fid, {}), params_by_fid.get(fid), len(roi.segments),
            event_geometry_identity(events),
        )
        _fill_row(out[path_by_fid[fid]], roi, select, primary_idx, secondary_idx)
    return out


def roi_count_histogram(
    paths:    list[str],
    db_path:  str,
    detector: str | None = None,
) -> dict[int, int]:
    """{n_ruptures: n_ROIs} across the stored documents for `paths`."""
    from .roi_events import payload_to_events

    ep = event_params_from(_db.load_analysis_params(db_path), detector=detector)
    pj = event_map_params_json(ep)
    cv = cache_version()

    hist: dict[int, int] = {}
    conn = _db.get_connection(db_path)
    try:
        for path in paths:
            fid = _db.get_file_id(path, db_path, conn=conn)
            if fid is None:
                continue
            doc = _db.get_event_map(fid, pj, cv, db_path, conn=conn)
            if not doc:
                continue
            events = payload_to_events(json.loads(doc))
            if events is None:
                continue
            for roi in events.rois:
                hist[roi.n_ruptures] = hist.get(roi.n_ruptures, 0) + 1
    finally:
        conn.close()
    return dict(sorted(hist.items()))


def coverage(paths: list[str], db_path: str, detector: str | None = None) -> tuple[int, int]:
    """Return (n_with_document, n_total) so the UI can show how much is populated."""
    ep = event_params_from(_db.load_analysis_params(db_path), detector=detector)
    pj = event_map_params_json(ep)
    cv = cache_version()
    n = 0
    conn = _db.get_connection(db_path)
    try:
        for path in paths:
            fid = _db.get_file_id(path, db_path, conn=conn)
            if fid is not None and _db.get_event_map(fid, pj, cv, db_path, conn=conn) is not None:
                n += 1
    finally:
        conn.close()
    return n, len(paths)
