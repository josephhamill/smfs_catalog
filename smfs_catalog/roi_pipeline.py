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
from .curve_loader import LoadError, load_force_curve
from .provenance import cache_version
from .roi_selection import (
    ReportedSegmentChoice,
    SegmentOverrideResolution,
    event_geometry_identity,
    read_segment_select,
    resolve_reported_segment,
    resolve_isoforce_pair,
    resolve_segment_override,
    resolve_segment_override_state,
    write_segment_select,
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
    invols_offset_pts: int
    invols_window_pts: int


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
    db_path: str | None = None, code_ver: str | None = None,
    file_id: int | None = None, stage1=None,
    param_set: AnalysisParams | None = None,
    force: bool = False,
    conn=None,
):
    """
    Full multi-event structure for one loaded curve: baseline, detection signals,
    snap-off, search band, ROI segmentation, and per-segment WLC fit.  Returns a
    fitted CurveEvents (forces + l_p/l_c filled where fittable).

    See compute_curve_events_coords for the stage1/cache reuse contract.
    """
    return compute_curve_events_coords(
        curve, ep, db_path=db_path, code_ver=code_ver, file_id=file_id,
        stage1=stage1, param_set=param_set, force=force, conn=conn,
    ).events


def compute_curve_events_coords(
    curve, ep: EventParams,
    *,
    db_path:  str | None = None,
    code_ver: str | None = None,
    file_id:  int | None = None,
    stage1=None,   # curve_analysis.Stage1Search | None
    param_set: AnalysisParams | None = None,
    force: bool = False,
    conn=None,
):
    """
    Like compute_curve_events, but also returns the (offset, invols, snap_piezo)
    transform so callers can draw force-extension overlays that align exactly with
    the fitted segments:
        defl_corr = (defl_retr − offset) / invols
        force     = k · defl_corr
        extension = (piezo_retr − snap_piezo) − defl_corr
    Returns a CurveEventsResult (named fields — see its docstring).

    Reuse order for every stage below — baseline, snap-off index, the outer
    rupture's search band, invOLS — is: (1) `stage1`, when the caller is in the
    SAME pass as a fresh curve_analysis.analyse_curve call and already has these
    in memory — zero recompute, zero DB round-trip; (2) the analysis_results
    cache, when `db_path`/`code_ver`/`file_id` are given but stage1 doesn't have
    a value (a cold viewer open, a batch backfill) — one DB read, no numpy; (3)
    fresh computation, the only option when neither is available.  A caller that
    passes none of this (the historical call shape) behaves exactly as before —
    always fresh — so this is purely additive.

    The decomposed curve and the detection signals (d1/mean_dev) are never
    persisted (large per-sample arrays, cheap to regenerate) — they're only ever
    reused via `stage1`, never via the DB cache.

    Before doing ANY of the above, this also checks for a stored event_map
    document under the current signature — if one already matches, the whole
    decompose/detect/fit chain is skipped entirely and the stored document is
    deserialised back into events instead (`_persist_multi_event_roi` already
    does this check before ever calling in; this makes it apply to every
    caller, including ones — like the WLC fit viewer — that call in directly
    without that outer guard).

    Always writes the freshly-computed result to event_map when this call
    actually had to compute (i.e. no matching document already existed) and
    cache identity (db_path/code_ver/file_id) is available — there is no way
    for a caller to trigger a fresh computation here and have it silently not
    saved.  `_persist_multi_event_roi` performs no separate write: it skips
    out before reaching here on a cache hit, and this function saves whenever
    it actually computes.

    ``force=True`` bypasses the complete event-map document and reruns event
    detection, segmentation, fitting, and persistence. Stage-1 values and
    scalar analysis caches remain reusable calibrations. A supplied ``conn``
    keeps those reads and the single event-map write on the caller's existing
    database connection.
    """
    from .signal_processing import (
        decompose_curve, fit_retract_baseline, find_end_in_contact,
        fit_approach_invols,
    )
    from .roi_detection import compute_detection_signals, rupture_search_bounds
    from .roi_events import build_curve_events, fit_segments, payload_to_events, events_to_payload
    from .curve_analysis import CurveAnalysisError

    can_read_cache = db_path is not None and code_ver is not None and file_id is not None
    _p1 = None
    if can_read_cache and param_set is not None:
        # THE canonical key-builder for these rows (curve_analysis.pipeline_params_from)
        # — reused rather than hand-rebuilt, so this can never drift from the keys
        # analyse_curve actually wrote under (copy the asdict() pattern, don't
        # hand-maintain key lists).
        from .curve_analysis import pipeline_params_from
        _p1 = pipeline_params_from(param_set)

    def _cached(atype: str, params_json: str):
        return _db.get_analysis_result(file_id, atype, params_json, code_ver, db_path, conn=conn) \
            if can_read_cache else None

    # ── Fast path: reuse a stored event_map document if it already matches
    # the current signature — skips the event-building and WLC fit (the
    # expensive stages) entirely.  `dc`/`sigs` are still computed normally
    # below regardless of this hit (they're the "cheap to regenerate" stage
    # per this function's own docstring, and some callers — e.g. the ROI
    # detection window — draw them directly, so they must never come back
    # None). The rupture search band (below) is also still computed on a
    # cache hit: it's a cheap scan over the already-computed d1 signal, not
    # part of what the cache hit is skipping, and display_roi's mask overlay
    # requires it to always be a valid index.
    #
    # The signature is asdict(ep) — EVERY parameter the finder reads — so a
    # match means the parameters have not changed and the stored marks are
    # the marks these numbers produce.  A miss means they HAVE changed, and
    # the finder runs again.  That contract holds only while `ep` comes from
    # ONE owner — assembled from two profiles, a "match" certifies a parameter
    # set nobody chose.  See event_params_from/db.load_analysis_params.
    _pj = None
    _cached_events = None
    if can_read_cache and not force:
        _pj = event_map_params_json(ep)
        _doc = _db.get_event_map(file_id, _pj, code_ver, db_path, conn=conn)
        if _doc is not None:
            _cached_events = payload_to_events(json.loads(_doc))
    elif can_read_cache:
        _pj = event_map_params_json(ep)

    # ── offset (baseline) ────────────────────────────────────────────────────
    if stage1 is not None and stage1.offset is not None:
        offset = stage1.offset
    else:
        _cv = _cached("offset_retr", _p1.params_bl) if _p1 else None
        if _cv is not None:
            offset = float(_cv)
        else:
            # Fresh computation neither stage1 nor the DB cache had — same
            # persist-what-you-compute treatment as the invOLS block below.
            _bl = fit_retract_baseline(curve, ep.anchor_nm)
            offset = _bl.offset
            if _p1 is not None and offset == offset:   # not NaN
                _db.write_analysis_results_multi(file_id, {
                    "offset_retr":         float(_bl.offset),
                    "flatness_slope":      float(_bl.slope),
                    "baseline_intercept":  float(_bl.intercept),
                    "baseline_rms":        float(_bl.rms),
                    "baseline_fit_lo_idx": float(_bl.fit_lo_idx),
                    "baseline_fit_hi_idx": float(_bl.fit_hi_idx),
                }, _p1.params_bl, code_ver, db_path, conn=conn)

    # ── decomposed curve — never DB-cached, only ever reused via stage1 ─────
    dc = stage1.dc if (stage1 is not None and stage1.dc is not None) \
        else decompose_curve(curve, cutoff_hz=ep.cutoff_hz)

    # ── detection signals — never DB-cached, only ever reused via stage1 ────
    sigs = stage1.sigs if (stage1 is not None and stage1.sigs is not None) \
        else compute_detection_signals(
            low_retr=dc.low_retr - offset, piezo_retr=curve.piezo_retr,
            window_pts=ep.window_pts,
        )

    # ── snap-off index ───────────────────────────────────────────────────────
    if stage1 is not None and stage1.snapoff_idx is not None:
        si = stage1.snapoff_idx
    else:
        _cv = _cached("snapoff_idx", _p1.params_cd) if _p1 else None
        if _cv is not None:
            si = int(_cv)
        else:
            si, _ = find_end_in_contact(
                dc, var_window_ms=ep.var_window_ms, threshold=ep.thresh_retr,
                trim_pts=ep.trim_pts,
            )

    # ── outer-rupture search band — always computed, cache hit or not (see
    # the fast-path comment above: display_roi's mask overlay needs a real
    # index either way).
    if (stage1 is not None and stage1.mask_postsnap_idx is not None
            and stage1.mask_anchor_idx is not None):
        lo_band, hi_band = stage1.mask_postsnap_idx, stage1.mask_anchor_idx
    else:
        lo_band, hi_band = rupture_search_bounds(
            sigs.piezo, si, ep.anchor_nm, ep.post_mask_nm,
        )

    # ── event build — the expensive stage the cache hit actually skips.
    if _cached_events is not None:
        events = _cached_events
    else:
        events = build_curve_events(
            sigs.d1, sigs.mean_dev, sigs.piezo,
            lo=lo_band, hi=hi_band,
            onset_thr=ep.onset_thr, detector=ep.detector,
            outer_threshold=ep.d1_threshold, inner_threshold=ep.inner_threshold,
            prominence=ep.prominence, distance_pts=ep.distance_pts,
            outer_events=(stage1.outer_events if stage1 is not None else None),
        )

    # ── invOLS ───────────────────────────────────────────────────────────────
    inv = None
    if stage1 is not None and stage1.invols_slope is not None:
        inv = stage1.invols_slope
    else:
        _cv = _cached("invols_slope", _p1.params_invols) if _p1 else None
        if _cv is not None:
            inv = float(_cv)
        else:
            try:
                _fit = fit_approach_invols(
                    dc.low_appr, curve.piezo_appr, ep.invols_offset_pts, ep.invols_window_pts,
                )
                inv = _fit.slope
                # This is a real fresh computation (neither stage1 nor the DB
                # cache had it).  Persist what you compute: write it and
                # its QC diagnostics now, the same as
                # curve_analysis.analyse_curve does, so the next caller (same
                # params) gets a cache hit instead of re-fitting.
                if _p1 is not None and inv == inv:   # not NaN
                    _db.write_analysis_results_multi(file_id, {
                        "invols_slope":      float(inv),
                        "invols_intercept":  float(_fit.intercept),
                        "invols_rms":        float(_fit.rms),
                        "invols_fit_lo_idx": float(_fit.fit_lo_idx),
                        "invols_fit_hi_idx": float(_fit.fit_hi_idx),
                    }, _p1.params_invols, code_ver, db_path, conn=conn)
            except Exception as exc:
                raise CurveAnalysisError("approach invOLS", exc) from exc
    if inv is None or not isinstance(inv, (int, float)) or not math.isfinite(inv) or inv == 0.0:
        raise CurveAnalysisError(
            "approach invOLS",
            ValueError("no finite, non-zero calibration slope was produced"),
        )

    snap = float(curve.piezo_retr[si]) if 0 <= si < len(curve.piezo_retr) else 0.0

    if _cached_events is None:
        fit_segments(curve, events, offset, inv, snap, low_retr=dc.low_retr)

        if can_read_cache:
            _db.write_event_map(
                file_id, json.dumps(events_to_payload(events)), _pj, code_ver,
                db_path, conn=conn,
            )

    return CurveEventsResult(
        events=events, offset=offset, invols=inv, snap_piezo=snap,
        sigs=sigs, dc=dc, snapoff_idx=si,
        mask_postsnap_idx=lo_band, mask_anchor_idx=hi_band,
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
                            curve, ep, db_path=db_path, code_ver=cv, file_id=fid,
                            param_set=param_set, force=force, conn=conn,
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
    "seg_tau", "seg_z_max", "seg_edge_pinned",
)
SEG_SUMMARY_FIELD = {
    "seg_l_p_nm": "l_p_nm", "seg_l_c_nm": "l_c_nm",
    "seg_l_p_err": "l_p_err", "seg_l_c_err": "l_c_err",
    "seg_force_pN": "force_pN",
    "seg_x_rupture_nm": "x_rupture_nm", "seg_x_junction_nm": "x_junction_nm",
    "seg_dF_pN": "dF_pN", "seg_dX_iso_nm": "dX_iso_nm",
    "seg_dX_ext_nm": "dX_ext_nm",
    "seg_n_segments": "n_segments",
    "seg_tau": "tau", "seg_z_max": "z_max", "seg_edge_pinned": "edge_pinned",
}


def segment_summary_bulk(
    paths: list[str], select: str, db_path: str,
) -> dict[str, dict[str, "float | None"]]:
    """
    Per-path {"l_p_nm", "l_c_nm", "l_p_err", "l_c_err", "force_pN",
    "x_rupture_nm", "x_junction_nm", "dF_pN", "dX_iso_nm", "dX_ext_nm",
    "n_segments", "tau", "z_max", "edge_pinned"},
    read from each curve's latest event_map (whatever params/code produced it —
    like the 2DH windows, this never silently drops a curve pending a
    recompute).

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
    different axis and (for the rupture) a different sample.

    tau/z_max/edge_pinned are the SELECTED segment's fit-
    conditioning diagnostics, following the same selection rule as l_p/l_c so
    that a value and its explanation always describe the same fit:
      * tau     — integrated autocorrelation time of that fit's residual, in
                  samples.  l_p_err/l_c_err above ALREADY include sqrt(tau);
                  this is here so the size of an error bar can be explained.
      * z_max   — x_max/l_c, how close the fit window got to the WLC pole.  The
                  fit's conditioning number in all but name; below ~0.8 only the
                  product l_p*l_c is well determined.  Derived, never stored.
      * edge_pinned — 1.0 when the force peak sat on the window's right edge, so
                  the rupture force is an underestimate and the fit stopped
                  early.  0.0 otherwise, None if no peak was found.
    All three inform; none of them gates anything by itself.

    l_p_nm/l_c_nm/l_p_err/l_c_err/force_pN come from the SELECTED segment (and
    its terminating rupture) of the right-most outer ROI with ruptures:
    "ultimate" = its last segment (the tether/final rupture, or the whole pull
    if there's only one segment); "penultimate" = the one before it when that
    ROI has >= 2 segments, else the SAME segment ultimate would use — matching
    base_2dh_window._stored_segment_fit.  force_pN is the characterization
    value for "rupture force": whichever segment is selected here IS the
    rupture force, by definition — one user-facing force number, not a second
    parallel one alongside a stage-1 scalar.  l_p_err/l_c_err are the WLC
    fit's own uncertainty on l_p/l_c; large values flag an under-constrained
    fit.  They inform; they gate nothing by themselves.

    n_segments is the segment count of that SAME right-most outer ROI —
    NOT selection-dependent (it describes the ROI, not the pick within it).
    It exists so a caller who wants "only curves with a real, distinct
    penultimate segment" can filter on it explicitly (seg_n_segments >= 2)
    instead of relying on `select` to narrow the population silently.

    x_rupture_nm/x_junction_nm ARE selection-dependent, for the reason above:
    they belong to the reported rupture, so flipping `select` moves them
    exactly as it moves force_pN.  Pinning x_junction_nm to the terminal
    rupture instead would put one row's force and extension on two different
    ruptures whenever "penultimate" is in force.

    dF_pN/dX_iso_nm/dX_ext_nm are NOT selection-dependent (`select`) either by
    default — they are the gap between the last two ruptures (ultimate minus
    penultimate, by definition), so they don't change when `select` is
    flipped; only l_p_nm/l_c_nm/force_pN do.

    dX_ext_nm is the plain ruptures[lo].extension_nm -
    ruptures[hi].extension_nm gap — no crossing search, defined regardless of
    which rupture is stronger, unlike dX_iso_nm (see roi_events.ROI.
    dX_ext_pairs). It exists so dF_pN has an order-independent extension-side
    counterpart to compare against; dX_iso_nm remains the deliberately
    one-directional, force-matched "reload distance" — see roi_events.ROI.
    isoforce_dX_pairs for why that one is NOT made symmetric the same way.

    Manual segment override: a curve with a usable (non-stale, see
    resolve_segment_override) Primary pick uses THAT segment for
    l_p_nm/l_c_nm/l_p_err/l_c_err/force_pN instead of the `select`-driven
    Ultimate/Penultimate choice — an absolute per-curve override, not another
    value of `select`. If Primary AND Secondary are both usable, dF_pN/
    dX_iso_nm/dX_ext_nm are computed from that pair instead of the
    last-two-ruptures default — sorted by actual rupture index, never by
    which one was tagged Primary vs Secondary, so pick order never flips the
    sign. dF_pN/dX_ext_nm need no adjacency (plain subtractions); dX_iso_nm
    does (isoforce_x_nm is only ever stored relative to the immediately
    preceding rupture) and is None for a non-adjacent pair, same as any other
    missing piece below.

    Any missing piece (no file, no document, wrong schema version) yields
    None fields — never a fabricated value.  A present-but-single-segment
    ROI under "penultimate" is not one of those: it is the documented
    fallback above, not an absence.  Keyed by resolved path
    (db.normalize_path), matching get_derived_results_bulk_latest.

    Bulk by construction.  A per-path db.get_file_id opens and closes its own
    sqlite connection each time, and the queue-table rebuild behind every
    raw-viewer double-click pays that cost again for the whole queue.  So
    path->file_id and the event_map fetch both go through chunked bulk SQL,
    the same way get_derived_results_bulk_latest does.
    """
    from .roi_events import payload_to_events

    out: dict[str, dict[str, float | None]] = {
        _db.normalize_path(p): {
            "l_p_nm": None, "l_c_nm": None, "l_p_err": None, "l_c_err": None,
            "force_pN": None, "x_rupture_nm": None, "x_junction_nm": None,
            "dF_pN": None, "dX_iso_nm": None, "dX_ext_nm": None,
            "n_segments": None,
            "tau": None, "z_max": None, "edge_pinned": None,
        }
        for p in paths
    }
    if not paths:
        return out

    def _chunks(seq, size=800):
        for i in range(0, len(seq), size):
            yield seq[i:i + size]

    conn = _db.get_connection(db_path)
    try:
        paths_resolved = list(out.keys())
        path_by_fid: dict[int, str] = {}
        for chunk in _chunks(paths_resolved):
            ph = ",".join("?" * len(chunk))
            for r in conn.execute(
                f"SELECT id, path FROM files WHERE path IN ({ph})", chunk
            ).fetchall():
                path_by_fid[r["id"]] = r["path"]
        if not path_by_fid:
            return out

        # event_map holds exactly one row per file_id (write_event_map deletes
        # any prior row before inserting), so this is already "the latest" —
        # no need for a per-file MAX(computed_at) subquery.
        fids = list(path_by_fid.keys())
        payload_by_fid: dict[int, str] = {}
        params_by_fid: dict[int, str] = {}
        for chunk in _chunks(fids):
            fid_ph = ",".join("?" * len(chunk))
            for r in conn.execute(
                f"SELECT file_id, payload_json, params_json FROM event_map "
                f"WHERE file_id IN ({fid_ph})",
                chunk,
            ).fetchall():
                payload_by_fid[r["file_id"]] = r["payload_json"]
                params_by_fid[r["file_id"]]  = r["params_json"]

        # Manual segment overrides — bulk, for the same reason this function's
        # own docstring gives: one query, not one per file.
        override_by_fid = _db.get_segment_overrides_bulk(fids, db_path, conn=conn)
    finally:
        conn.close()

    for fid, doc in payload_by_fid.items():
        path = path_by_fid[fid]
        events = payload_to_events(json.loads(doc))
        if events is None:
            continue
        roi = next((r for r in reversed(events.rois) if r.ruptures), None)
        if roi is None:
            continue
        row = out[path]
        row["n_segments"] = len(roi.segments)

        primary_idx, secondary_idx = resolve_segment_override(
            override_by_fid.get(fid, {}), params_by_fid.get(fid), len(roi.segments),
            event_geometry_identity(events),
        )

        pair = resolve_isoforce_pair(
            len(roi.segments), primary_idx, secondary_idx,
        )
        if primary_idx is not None and secondary_idx is not None:
            lo, hi = sorted((primary_idx, secondary_idx))
            f_lo, f_hi = roi.ruptures[lo].force_pN, roi.ruptures[hi].force_pN
            row["dF_pN"] = None if f_lo is None or f_hi is None else f_hi - f_lo
            # dX_ext_nm is a plain extension-point subtraction — needs no
            # adjacency, same as dF_pN, unlike dX_iso_nm below.
            x_lo, x_hi = roi.ruptures[lo].extension_nm, roi.ruptures[hi].extension_nm
            row["dX_ext_nm"] = None if x_lo is None or x_hi is None else x_lo - x_hi
            # isoforce_x_nm is only ever stored relative to the IMMEDIATELY
            # PRECEDING rupture (roi_events.Segment docstring) — a non-
            # adjacent pair legitimately has no such value to read; None,
            # same as any other missing piece, never fabricated.
            if pair is not None:
                pair_lo, pair_hi = pair
                rup_lo, seg_hi = roi.ruptures[pair_lo], roi.segments[pair_hi]
                row["dX_iso_nm"] = (
                    None if rup_lo.extension_nm is None or seg_hi.isoforce_x_nm is None
                    else seg_hi.isoforce_x_nm - rup_lo.extension_nm
                )
        elif pair is not None:
            lo, hi = pair
            row["dF_pN"]     = roi.dF_pairs[hi - 1]
            row["dX_iso_nm"] = roi.isoforce_dX_pairs[hi - 1]
            row["dX_ext_nm"] = roi.dX_ext_pairs[hi - 1]

        # segments/ruptures are always the same length, one segment per
        # terminating rupture (ROI docstring: "Always len(ruptures) segments"),
        # so one length check picks both the segment and its rupture.
        seg = None
        rup = None
        reported = resolve_reported_segment(
            len(roi.segments), select, primary_idx,
        )
        if reported is not None:
            seg = roi.segments[reported.segment_idx]
            rup = roi.ruptures[reported.segment_idx]
        if seg is not None:
            row["l_p_nm"] = seg.l_p_nm
            row["l_c_nm"] = seg.l_c_nm
            row["l_p_err"] = seg.l_p_err
            row["l_c_err"] = seg.l_c_err
            # The three diagnostics describe the SELECTED segment's own fit, so
            # they follow the same Ultimate/Penultimate/override rule as the
            # values they explain — an error bar and the tau behind it must
            # never come from different segments.
            row["tau"]   = seg.tau
            row["z_max"] = seg.z_max
            # Numeric 0/1, not a bool: this reaches criteria_gate (which bounds
            # numbers) and quantities.format_value (which formats them).  The
            # dataclass keeps the bool; only this projection flattens it.
            row["edge_pinned"] = (
                None if seg.edge_pinned is None else float(bool(seg.edge_pinned))
            )
        if rup is not None:
            row["force_pN"] = rup.force_pN
            # The x of the SAME point force_pN is the y of — fit_segments sets
            # both from one sample (roi_events: rup.force_pN / rup.extension_nm
            # at peak_rel).  Read off the same `rup` here so a row can never
            # pair a force with an extension from a different rupture.
            row["x_rupture_nm"] = rup.extension_nm
            # The same rupture measured from the junction's onset instead of
            # from snap-off: how far the junction stretched under load, rather
            # than how long it was.  segments[0] is the onset segment, so its
            # left edge is the junction's onset whichever segment is reported.
            _onset_x = roi.segments[0].left_extension_nm
            row["x_junction_nm"] = (
                None if rup.extension_nm is None or _onset_x is None
                else rup.extension_nm - _onset_x
            )
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
