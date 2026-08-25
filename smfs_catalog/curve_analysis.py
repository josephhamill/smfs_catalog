# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/curve_analysis.py
#
# Non-GUI curve-analysis physics and persistence orchestration.
#
# One AnalysisParams snapshot is acquired by analyse_and_classify and projected by
# pipeline_params_from. Every stage in one curve pass receives that immutable
# snapshot. If the stored revision changes while a curve is running, the wrapper
# reruns it before publishing the verdict/current event map.
#
# analyse_curve(curve, params, ...) → (CurveResult, Stage1Search)
#     The single landmark-and-verdict implementation. It may reuse and persist
#     stage caches when given DB identity, but it never loads parameter state.
#     An event is defined by validated landmarks; rupture force is downstream
#     per-segment characterization in roi_events.fit_segments. Stage1Search
#     carries same-pass arrays and indices into that characterization step.
#
# analyse_and_classify(file_id, path, db_path) → (verdict, was_cached)
#     Loads one AnalysisParams snapshot, checks the verdict cache, loads and
#     analyses on a miss, persists the event map/verdict, and rejects publication
#     if the snapshot revision changed during the pass. Read/qualification failure
#     remains distinct from a scientifically verified non-event.
#
# This module has no Qt dependencies and is safe on the analysis worker thread.

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from . import db as _db
from .analysis_params import AnalysisParams
from .curve_loader import LoadError, UnusableCurveError, load_force_curve
from .provenance import cache_version


class CurveAnalysisError(RuntimeError):
    """A calculation stage failed; this is not a scientific non-event."""

    def __init__(self, stage: str, cause: Exception):
        self.stage = stage
        self.cause = cause
        super().__init__(f"{stage} failed: {type(cause).__name__}: {cause}")


@dataclass(frozen=True)
class _PipelineParams:
    """Stage inputs projected from AnalysisParams, plus canonical cache keys."""
    anchor_nm: float
    cutoff_hz: float
    trim_pts: int
    var_window_ms: float
    thresh_appr: float
    thresh_retr: float
    roi_win_pts: int
    roi_d1_thr: float
    roi_post_mask: float
    roi_onset_thr: float
    invols_offset_pts: int
    invols_window_pts: int
    params_bl: str
    params_cd: str
    params_roi: str
    params_invols: str
    all_params: str          # full signature — verdict-level cache key


def pipeline_params_from(
    param_set: AnalysisParams,
) -> _PipelineParams:
    """
    Pure stage/cache projection of one immutable analysis snapshot. Interactive
    previews create a replaced AnalysisParams first; this function never mixes
    overrides into a stored snapshot.
    """
    _ps = param_set
    anchor_nm = _ps.baseline_anchor_nm
    cutoff_hz = _ps.spectral_cutoff_hz
    trim_pts = _ps.turnaround_trim_pts
    var_window_ms = _ps.var_window_ms
    thresh_appr = _ps.detection_threshold_appr
    thresh_retr = _ps.detection_threshold_retr
    invols_offset_pts = _ps.invols_offset_pts
    invols_window_pts = _ps.invols_window_pts
    roi_win_pts = _ps.roi_window_pts
    roi_d1_thr = _ps.roi_threshold_nm_per_nm
    roi_post_mask = _ps.roi_post_snapoff_mask_nm
    roi_onset_thr = _ps.roi_onset_threshold_nm
    # #80 key 1: these six are read by roi_pipeline.event_params_from (the
    # multi-event finder _persist_multi_event_roi runs downstream) but were
    # absent from params_roi, so changing one didn't invalidate the verdict
    # fast-path cache — the fast path would return early and the finder would
    # never even run against the new setting.  Defaults/keys match
    # event projection exactly, including inner defaulting to outer.
    roi_inner_thr = _ps.roi_inner_threshold_nm_per_nm
    roi_detector_idx = _ps.roi_detector_mode_idx
    roi_prominence = _ps.roi_prominence
    roi_min_dist_pts = _ps.roi_min_distance_pts

    params_bl = json.dumps({"anchor_nm": float(anchor_nm)}, sort_keys=True,
                           separators=(",", ":"))
    params_cd = json.dumps({
        "cutoff_hz":     float(cutoff_hz),
        "trim_pts":      trim_pts,
        "var_window_ms": float(var_window_ms),
        "thresh_appr":   float(thresh_appr),
        "thresh_retr":   float(thresh_retr),
    }, sort_keys=True, separators=(",", ":"))
    params_roi = json.dumps({
        "anchor_nm":                float(anchor_nm),
        "cutoff_hz":                float(cutoff_hz),
        "trim_pts":                 int(trim_pts),
        "var_window_ms":            float(var_window_ms),
        "thresh_retr":              float(thresh_retr),
        "roi_window_pts":           int(roi_win_pts),
        "roi_threshold_nm_per_nm":  float(roi_d1_thr),
        "roi_post_snapoff_mask_nm": float(roi_post_mask),
        "roi_onset_threshold_nm":   float(roi_onset_thr),
        "roi_inner_threshold_nm_per_nm": float(roi_inner_thr),
        "roi_detector_mode_idx":    roi_detector_idx,
        "roi_prominence":           float(roi_prominence),
        "roi_min_distance_pts":     roi_min_dist_pts,
    }, sort_keys=True, separators=(",", ":"))
    params_invols = json.dumps({
        "cutoff_hz":  float(cutoff_hz),
        "offset_pts": invols_offset_pts,
        "window_pts": invols_window_pts,
    }, sort_keys=True, separators=(",", ":"))
    # Any parameter change changes this half of the cache identity. The
    # registered scientific-method fingerprint is the other half.
    all_params = param_set.revision
    return _PipelineParams(
        anchor_nm, cutoff_hz, trim_pts, var_window_ms, thresh_appr, thresh_retr,
        roi_win_pts, roi_d1_thr, roi_post_mask, roi_onset_thr,
        invols_offset_pts, invols_window_pts,
        params_bl, params_cd, params_roi, params_invols, all_params,
    )


@dataclass(frozen=True)
class Stage1Search:
    """
    Intermediate ROI-search state from analyse_curve, returned alongside
    CurveResult so a caller still holding the loaded curve in the same pass
    (the multi-event enrichment step, _persist_multi_event_roi)
    can reuse it instead of re-decomposing / re-searching from scratch.

    Every field is best-effort: populated only as far as THIS call actually
    computed fresh (a cache hit on a stage means that stage's numpy work never
    ran, so its fields here stay None).  A caller that finds a field None
    should fall back to the persisted analysis_results row for that stage
    (now written by analyse_curve — see the *_idx keys), not silently skip it.

    invols_intercept/r2/rms/fit_lo_idx/fit_hi_idx are diagnostics from the
    invOLS fit — used for QC on the fit quality (r2/rms) and for drawing the
    fit line (fit_lo_idx/fit_hi_idx, decomposition_window.py). Like
    invols_slope, they are persisted (params_invols) and populated here on
    EITHER a fresh compute or a cache hit — unlike the rest of this dataclass,
    which is fresh-compute-only.

    baseline_intercept/r2/fit_lo_idx/fit_hi_idx are the same idea for the
    retract-baseline fit (params_bl) — `offset`/`flatness` above already carry
    the values that matter physically (what's subtracted, and the slope
    diagnostic); these four are purely for QC (r2) and drawing the fit line
    (fit_lo_idx/fit_hi_idx, decomposition_window.py), and likewise populated on
    either a fresh compute or a cache hit.
    """
    dc:                   object | None = None   # signal_processing.DecomposedCurve
    offset:               float  | None = None
    snapoff_idx:          int    | None = None
    sigs:                 object | None = None   # roi_detection.DetectionSignals
    mask_postsnap_idx:    int    | None = None
    mask_anchor_idx:      int    | None = None
    rupture_idx:          int    | None = None
    onset_idx:            int    | None = None
    outer_events:         object | None = None  # list[roi_events.OuterEventBoundary]
    invols_slope:         float  | None = None
    invols_intercept:     float  | None = None
    invols_rms:           float  | None = None
    invols_fit_lo_idx:    int    | None = None
    invols_fit_hi_idx:    int    | None = None
    baseline_intercept:   float  | None = None
    baseline_rms:         float  | None = None
    baseline_fit_lo_idx:  int    | None = None
    baseline_fit_hi_idx:  int    | None = None


@dataclass(frozen=True)
class CurveResult:
    """
    The complete outcome of one curve analysis.

    Always present (NaN only when that stage could not be computed):
        offset, flatness       — retract baseline
        contact_z, snapoff_z   — contact / snap-off piezo landmarks (nm)

    Event landmarks — present ONLY when `event` is True (NaN otherwise):
        rupture_z, onset_z     — ROI landmarks (nm)

    Calibration:
        invols_slope           — approach invOLS slope; independent of verdict

    Rupture force is NOT part of this result — an event is defined by landmarks
    alone. Force is a characterization quantity computed downstream, per
    segment, by the segmentation stage (roi_events.fit_segments).
    """
    event:         bool
    offset:        float
    flatness:      float
    contact_z:     float
    snapoff_z:     float
    rupture_z:     float
    onset_z:       float
    invols_slope:  float

    @property
    def verdict(self) -> str:
        return "event" if self.event else "non_event"


def analyse_curve(
    curve,
    p:        _PipelineParams,
    *,
    db_path:  str | None = None,
    code_ver: str | None = None,
    file_id:  int | None = None,
    conn=None,
) -> tuple[CurveResult, Stage1Search]:
    """
    THE single implementation of the landmark + verdict physics.

    Runs baseline fit, invOLS fit, contact/snap-off detection, and ROI search
    on an already-loaded `curve`, using the parameters in `p`.  Returns
    (CurveResult, Stage1Search); rupture/onset are NaN unless the verdict is an
    event — invols_slope is not gated this way. Stage1Search
    carries whatever intermediate arrays/indices this call actually computed
    fresh, so a same-pass caller (the multi-event enrichment step) can reuse
    them instead of re-deriving from the raw curve — see Stage1Search's
    docstring for the cache-miss fallback contract.

    No force is computed here. An event is defined by landmarks alone:
    baseline fit succeeds, snap-off found, both ROI landmarks found, and
    rupture after snap-off. Rupture force is a characterization quantity,
    computed per segment downstream by the segmentation stage
    (roi_events.fit_segments, reached via _persist_multi_event_roi).

    Caching (optional — only when db_path, code_ver and file_id are all given):
      - offset/flatness (params_bl) and invols_slope (params_invols) are always
        read/written for every curve that reaches this far, event or not.
        invOLS is an approach-only calibration fit — independent of whether the
        retract ever shows a validated rupture — so it's computed and stored
        right alongside baseline rather than deferred until an event verdict.
      - contact/snapoff (params_cd) are likewise always read/written.
      - rupture/onset (params_roi) are read back to skip recompute, but only
        WRITTEN for a validated event. invols_slope is not part of that gate —
        it's a per-curve calibration number that exists whether or not this
        curve turns out to hold an event.

    Always writes whatever it computes fresh, whenever db_path/code_ver/file_id
    are all given — there is no way for a caller to compute a value here and
    have it silently not saved. If two legitimate analysis producers compute
    the same value around the same time, the write is a replace rather than an
    append, so a redundant write remains safe. Display-only windows must read
    these results instead of becoming additional analysis producers.
    """
    can_cache = db_path is not None and code_ver is not None and file_id is not None

    def _get(atype: str, params: str):
        if not can_cache:
            return None
        return _db.get_analysis_result(file_id, atype, params, code_ver, db_path,
                                      conn=conn)

    def _put(atype: str, value: float, params: str) -> None:
        if can_cache:
            _db.write_analysis_result(file_id, atype, value, params, code_ver,
                                      db_path, conn=conn)

    def _get_multi(atypes, params: str) -> dict:
        if not can_cache:
            return {}
        return _db.get_analysis_results_multi(file_id, atypes, params, code_ver,
                                             db_path, conn=conn)

    def _put_multi(values: dict, params: str) -> None:
        if can_cache:
            _db.write_analysis_results_multi(file_id, values, params, code_ver,
                                            db_path, conn=conn)

    from .signal_processing import (
        decompose_curve, find_begin_in_contact, find_end_in_contact,
        fit_approach_invols,
    )
    from .roi_detection import compute_detection_signals, rupture_search_bounds
    from .roi_events import find_outer_events

    _nan = float("nan")

    def _non_event(offset=_nan, flatness=_nan, contact_z=_nan, snapoff_z=_nan,
                    invols_slope=_nan) -> CurveResult:
        # rupture/onset stay NaN — a non_event has no ROI landmarks by
        # definition. invols_slope is NOT part of that gate: it's an
        # approach-only calibration fit, known regardless of event status once
        # we've reached this point, so it's passed through rather than blanked.
        return CurveResult(
            event=False, offset=offset, flatness=flatness,
            contact_z=contact_z, snapoff_z=snapoff_z,
            rupture_z=_nan, onset_z=_nan, invols_slope=invols_slope,
        )

    # Populated as far as this call actually computes fresh; see Stage1Search.
    _dc                = None
    snap_idx_cached: int | None = None
    sigs               = None
    rsearch             = None
    outer_events        = None
    rup_idx: int | None = None
    ons_idx: int | None = None
    invols_slope_val:        float | None = None
    invols_intercept_val:    float | None = None
    invols_rms_val:          float | None = None
    invols_fit_lo_idx_val:   int   | None = None
    invols_fit_hi_idx_val:   int   | None = None
    baseline_intercept_val:  float | None = None
    baseline_rms_val:        float | None = None
    baseline_fit_lo_idx_val: int   | None = None
    baseline_fit_hi_idx_val: int   | None = None

    def _stage1() -> Stage1Search:
        return Stage1Search(
            dc=_dc, offset=(offset if not np.isnan(offset) else None),
            snapoff_idx=snap_idx_cached, sigs=sigs,
            mask_postsnap_idx=(rsearch.mask_postsnap_idx if rsearch is not None else None),
            mask_anchor_idx=(rsearch.mask_anchor_idx if rsearch is not None else None),
            rupture_idx=rup_idx, onset_idx=ons_idx, invols_slope=invols_slope_val,
            outer_events=outer_events,
            invols_intercept=invols_intercept_val,
            invols_rms=invols_rms_val,
            baseline_intercept=baseline_intercept_val,
            baseline_rms=baseline_rms_val,
            baseline_fit_lo_idx=baseline_fit_lo_idx_val,
            baseline_fit_hi_idx=baseline_fit_hi_idx_val,
            invols_fit_lo_idx=invols_fit_lo_idx_val,
            invols_fit_hi_idx=invols_fit_hi_idx_val,
        )

    # ── Baseline ─────────────────────────────────────────────────────────────
    # baseline_r2 is deliberately absent. R² reads near 0 on a good flat,
    # undriven baseline because a
    # flat anchor region has little variance for a line to explain, so the
    # usual "near 1 is good" reading is backwards.  baseline_rms is the real
    # fit-quality signal.
    _BL_KEYS = ("offset_retr", "flatness_slope", "baseline_intercept",
                "baseline_rms", "baseline_fit_lo_idx",
                "baseline_fit_hi_idx")
    _bl_cached = _get_multi(_BL_KEYS, p.params_bl)   # one round-trip, not six
    # Require ALL SIX rows, not just offset/flatness. A row cached back when
    # only offset/flatness existed would
    # otherwise register as a permanent cache hit and never backfill missing
    # diagnostic fields. Recomputing here is cheap: a small polyfit over the anchor
    # window, not a curve reload/decompose.
    if all(k in _bl_cached for k in _BL_KEYS):
        offset, flatness        = float(_bl_cached["offset_retr"]), float(_bl_cached["flatness_slope"])
        baseline_intercept_val  = float(_bl_cached["baseline_intercept"])
        baseline_rms_val        = float(_bl_cached["baseline_rms"])
        baseline_fit_lo_idx_val = int(_bl_cached["baseline_fit_lo_idx"])
        baseline_fit_hi_idx_val = int(_bl_cached["baseline_fit_hi_idx"])
    else:
        try:
            from .signal_processing import fit_retract_baseline
            bl = fit_retract_baseline(curve, float(p.anchor_nm))
            offset, flatness = bl.offset, bl.slope
            baseline_intercept_val  = bl.intercept
            baseline_rms_val        = bl.rms
            baseline_fit_lo_idx_val = bl.fit_lo_idx
            baseline_fit_hi_idx_val = bl.fit_hi_idx
        except Exception as exc:
            raise CurveAnalysisError("retract baseline", exc) from exc
        if not np.isnan(offset):
            _put_multi({
                "offset_retr":         offset,
                "flatness_slope":      flatness,
                "baseline_intercept":  float(baseline_intercept_val),
                "baseline_rms":        float(baseline_rms_val),
                "baseline_fit_lo_idx": float(baseline_fit_lo_idx_val),
                "baseline_fit_hi_idx": float(baseline_fit_hi_idx_val),
            }, p.params_bl)
    if np.isnan(offset):
        return _non_event(), _stage1()

    # ── invOLS fit ───────────────────────────────────────────────────────────
    # Approach-only (curve.piezo_appr / the decomposed low-frequency approach
    # channel) — independent of the retract baseline above and of whether this
    # curve turns out to hold a validated rupture. Computed and stored for
    # every curve that gets this far, event or not: it is a calibration fit,
    # unlike the event-only landmarks below, which are gated on a validated
    # event.
    # invols_r2 deliberately absent — same reason as baseline_r2 above.
    _IOL_KEYS = ("invols_slope", "invols_intercept",
                 "invols_rms", "invols_fit_lo_idx", "invols_fit_hi_idx")
    _iol_cached = _get_multi(_IOL_KEYS, p.params_invols)   # one round-trip, not six
    # Same completeness requirement as the baseline block above: a bare
    # invols_slope row cached before the diagnostic fields existed would
    # otherwise be a permanent cache hit that never backfills them.
    if all(k in _iol_cached for k in _IOL_KEYS):
        invols_slope           = float(_iol_cached["invols_slope"])
        invols_intercept_val   = float(_iol_cached["invols_intercept"])
        invols_rms_val         = float(_iol_cached["invols_rms"])
        invols_fit_lo_idx_val  = int(_iol_cached["invols_fit_lo_idx"])
        invols_fit_hi_idx_val  = int(_iol_cached["invols_fit_hi_idx"])
    else:
        try:
            if _dc is None:
                _dc = decompose_curve(curve, cutoff_hz=float(p.cutoff_hz))
            fit = fit_approach_invols(
                _dc.low_appr, curve.piezo_appr, p.invols_offset_pts, p.invols_window_pts,
            )
            invols_slope = fit.slope
            invols_intercept_val  = fit.intercept
            invols_rms_val        = fit.rms
            invols_fit_lo_idx_val = fit.fit_lo_idx
            invols_fit_hi_idx_val = fit.fit_hi_idx
        except Exception as exc:
            raise CurveAnalysisError("approach invOLS", exc) from exc
        if np.isnan(invols_slope):
            raise CurveAnalysisError(
                "approach invOLS",
                ValueError("fit window is degenerate; no calibration slope was produced"),
            )
        if not np.isnan(invols_slope):
            _put_multi({
                "invols_slope":      float(invols_slope),
                "invols_intercept":  float(invols_intercept_val),
                "invols_rms":        float(invols_rms_val),
                "invols_fit_lo_idx": float(invols_fit_lo_idx_val),
                "invols_fit_hi_idx": float(invols_fit_hi_idx_val),
            }, p.params_invols)
    if not np.isnan(invols_slope):
        invols_slope_val = invols_slope

    # ── Contact detection ────────────────────────────────────────────────────
    # NOTE: _dc is NOT reset here. It may already hold a real DecomposedCurve
    # from the invOLS section above (a cache miss there computes it) — resetting
    # it here would silently force a second, redundant decompose_curve() call.
    # _dc starts as None (see the "Populated as far as..." block above) and is
    # only ever set, never cleared, from this point through the rest of the
    # function — every "if _dc is None: _dc = decompose_curve(...)" guard below
    # still does the right thing whichever section reaches it first.

    contact_z = _get("contact_piezo_nm", p.params_cd)
    snapoff_z = _get("snapoff_piezo_nm", p.params_cd)
    if contact_z is None or snapoff_z is None:
        try:
            _dc = decompose_curve(curve, cutoff_hz=float(p.cutoff_hz))
            ci, _ = find_begin_in_contact(
                _dc, var_window_ms=float(p.var_window_ms),
                threshold=float(p.thresh_appr), trim_pts=p.trim_pts,
            )
            si, _ = find_end_in_contact(
                _dc, var_window_ms=float(p.var_window_ms),
                threshold=float(p.thresh_retr), trim_pts=p.trim_pts,
            )
            snap_idx_cached = si
            contact_z = float(curve.piezo_appr[ci])
            snapoff_z = float(curve.piezo_retr[si])
        except Exception as exc:
            raise CurveAnalysisError("contact/snap-off detection", exc) from exc
        if not np.isnan(contact_z):
            _put("contact_piezo_nm", contact_z, p.params_cd)
            _put("snapoff_piezo_nm", snapoff_z, p.params_cd)
            if snap_idx_cached is not None:
                _put("snapoff_idx", float(snap_idx_cached), p.params_cd)
    else:
        contact_z, snapoff_z = float(contact_z), float(snapoff_z)
        # Values hit cache, but the INDEX wasn't necessarily remembered from a
        # prior pass — recover it so the ROI section below doesn't have to
        # re-run find_end_in_contact just to get back an index we already have.
        _cached_snap_idx = _get("snapoff_idx", p.params_cd)
        if _cached_snap_idx is not None:
            snap_idx_cached = int(_cached_snap_idx)

    if np.isnan(snapoff_z):
        return _non_event(offset, flatness, contact_z, snapoff_z, invols_slope), _stage1()

    # ── ROI search ───────────────────────────────────────────────────────────
    rupture_z = _get("rupture_piezo_nm", p.params_roi)
    onset_z   = _get("onset_piezo_nm",   p.params_roi)
    if rupture_z is None or onset_z is None:
        try:
            if _dc is None:
                _dc = decompose_curve(curve, cutoff_hz=float(p.cutoff_hz))
            if snap_idx_cached is not None:
                si_roi = snap_idx_cached
            else:
                si_roi, _ = find_end_in_contact(
                    _dc, var_window_ms=float(p.var_window_ms),
                    threshold=float(p.thresh_retr), trim_pts=p.trim_pts,
                )
            low_corr = _dc.low_retr - offset
            sigs = compute_detection_signals(
                low_retr   = low_corr,
                piezo_retr = curve.piezo_retr,
                window_pts = p.roi_win_pts,
            )
            lo_band, hi_band = rupture_search_bounds(
                sigs.piezo, si_roi, float(p.anchor_nm), p.roi_post_mask,
            )
            outer_events = find_outer_events(
                sigs.d1, sigs.mean_dev, sigs.piezo,
                lo=lo_band, hi=hi_band, onset_thr=p.roi_onset_thr,
                outer_threshold=p.roi_d1_thr,
            )
            # Scalar compatibility fields describe the right-most validated
            # outer event. Inner segmentation later consumes this exact list.
            if outer_events:
                primary_outer = outer_events[-1]
                rup_idx = primary_outer.terminal_idx
                ons_idx = primary_outer.onset_idx
                rupture_z = float(sigs.piezo[rup_idx])
                onset_z = float(sigs.piezo[ons_idx])
            else:
                rupture_z = onset_z = _nan

            rsearch = SimpleNamespace(
                mask_postsnap_idx=lo_band, mask_anchor_idx=hi_band,
            )
        except Exception as exc:
            raise CurveAnalysisError("ROI landmark detection", exc) from exc
        # Rupture/onset are not persisted until the landmark validity gate below.
        # invOLS was persisted independently above; force belongs to downstream
        # per-segment characterization and is never computed here.
    else:
        rupture_z, onset_z = float(rupture_z), float(onset_z)
        # Same recovery as snap-off above: the piezo-nm values hit cache, but
        # pull the indices back too so same-pass multi-event characterization
        # does not have to re-derive them from the curve.
        _cached_rup_idx = _get("rupture_idx", p.params_roi)
        _cached_ons_idx = _get("onset_idx",   p.params_roi)
        if _cached_rup_idx is not None:
            rup_idx = int(_cached_rup_idx)
        if _cached_ons_idx is not None:
            ons_idx = int(_cached_ons_idx)

    # Validate the event landmarks: rupture and onset must both exist, and the
    # rupture must lie after snap-off in piezo space. invOLS was resolved above
    # independently of the verdict; rupture force is not computed in this module.
    if np.isnan(rupture_z) or np.isnan(onset_z) or (rupture_z - snapoff_z) <= 0:
        return _non_event(offset, flatness, contact_z, snapoff_z, invols_slope), _stage1()

    # invOLS is already resolved above (computed alongside baseline, before the
    # event gate) — nothing to compute here, just use it.

    # ── Verdict ──────────────────────────────────────────────────────────────
    # Event is defined by landmarks alone (see the gate above): baseline OK,
    # snap-off found, both ROI landmarks found, rupture after snap-off. All of
    # that is already guaranteed true by the time we reach here — no separate
    # force condition. Rupture force is NOT computed in this module; it's a
    # characterization quantity owned by the segmentation stage downstream.

    # rup_idx/ons_idx may still be None here if rupture_z/onset_z hit cache but
    # their index counterparts predate index caching — recover them from the
    # piezo-nm values so the backfill below still runs.
    if rup_idx is None:
        rup_idx = int(np.argmin(np.abs(curve.piezo_retr - rupture_z)))
    if ons_idx is None:
        ons_idx = int(np.argmin(np.abs(curve.piezo_retr - onset_z)))

    # ── Persist validated event landmarks ────────────────
    # invols_slope is NOT written here — it's already been written above,
    # unconditionally, alongside baseline (it isn't gated by event status).
    _put("rupture_piezo_nm", float(rupture_z), p.params_roi)
    _put("onset_piezo_nm",   float(onset_z),   p.params_roi)
    # Indices alongside their piezo-nm counterparts, so a later pass (this
    # curve's own multi-event enrichment, the ROI viewer, a batch backfill) can
    # read the landmark position back without re-searching for it.
    _put("rupture_idx", float(rup_idx), p.params_roi)
    _put("onset_idx",   float(ons_idx), p.params_roi)

    return CurveResult(
        event=True, offset=offset, flatness=flatness,
        contact_z=contact_z, snapoff_z=snapoff_z,
        rupture_z=float(rupture_z), onset_z=float(onset_z),
        invols_slope=float(invols_slope),
    ), _stage1()


def current_signature(db_path: str) -> tuple[str, str | None]:
    """
    (all_params, scientific-method version) — the exact pair the FAST PATH tests a
    stored verdict against.  Nothing else; this is a read of the live
    settings, not an analysis.

    Exists so a caller can ask "which queued files would the fast path serve?"
    WITHOUT running the pipeline (issue #96).  The dashboard needs that to
    tell up-to-date rows from stale ones and to give the ETA an honest
    denominator: an up-to-date file still gets VISITED — the worker steers by
    queue order and nothing overrules a scrub — just in milliseconds rather
    than seconds.

    Deliberately returns the signature rather than a verdict, so the freshness
    question stays DERIVED (db.queue_freshness) instead of becoming a stored
    per-file class that every parameter or numerical-method edit would have to
    invalidate.  A stored class is a cache key that nothing updates.
    """
    return (pipeline_params_from(_db.load_analysis_params(db_path)).all_params,
            cache_version())


def analyse_and_classify(
    file_id: int,
    path:    str,
    db_path: str,
    conn=None,
) -> tuple[str, bool]:
    """
    Thin persistence wrapper around analyse_curve.  Caches the verdict (and,
    via analyse_curve, all derived values) in analysis_results and returns the
    verdict string plus whether the FAST PATH served it (issue #37 — makes the
    fast-path observable end-to-end, up to the dashboard's session counter).

    Returns (verdict, was_cached):
        verdict:
            'event'       — pipeline located a valid event (an ROI excursion)
            'non_event'   — no ROI / event could be identified
            'unavailable' — the curve could not be READ at all (issue #69,
                             e.g. a disconnected drive). Distinct from
                             'non_event' on purpose: a load failure is not a
                             classification, and must not be able to
                             masquerade as one. No verdict is cached for this
                             outcome, so the next pass retries.
            'unusable'    — the curve read fine but does not qualify (issue
                             #122, e.g. a channel full of NaN). Also not a
                             classification. Unlike 'unavailable' this will
                             never fix itself, so the file is dequeued and
                             labelled with its reason instead of retried.
        was_cached: True iff the fast path served this verdict (no disk read,
            no recompute, no writes) rather than the slow path actually
            loading and analysing the curve.
    """
    code_ver = cache_version()
    param_set = _db.load_analysis_params(db_path)
    p = pipeline_params_from(param_set)

    # ── FAST PATH ─────────────────────────────────────────────────────────────
    # If a verdict was already cached under this exact parameter signature and
    # scientific-method version, the file is up to date. Return it WITHOUT loading the curve
    # — no disk read, no recompute, no writes.  (Stored as 1.0=event / 0.0=non_event.)
    cached_verdict = _db.get_analysis_result(
        file_id, "event", p.all_params, code_ver, db_path, conn=conn
    )
    if cached_verdict is not None:
        if _db.load_analysis_params(db_path).revision != param_set.revision:
            return analyse_and_classify(file_id, path, db_path, conn=conn)
        return ("event" if cached_verdict >= 0.5 else "non_event"), True

    # ── SLOW PATH ─────────────────────────────────────────────────────────────
    # Verdict not cached for these params → load the curve and run the pipeline.
    try:
        curve = load_force_curve(path)
    except UnusableCurveError as exc:
        # The file read fine but does not qualify (curve_loader.qualify_wave):
        # a dead channel, a constant channel, no turnaround, or an all-zero
        # retract. That is a durable fact about the file, so:
        #   - record WHY, on the file, where the user can see it. The file is
        #     kept in the catalog rather than dropped — a curve that silently
        #     vanishes is one the user goes looking for (#122).
        #   - drop it from the queue so the worker never loads it again.
        #   - return 'unusable', NOT 'non_event'. It was never classified;
        #     calling it a non_event would be a verdict we did not reach, the
        #     same disguise #69 removed for 'unavailable'.
        # No verdict is cached, but nothing retries it either: unlike
        # 'unavailable' this will not fix itself, and re-qualifying it on every
        # pass is work with a known answer.
        _db.set_unusable_reason(file_id, exc.reason, str(exc), db_path, conn=conn)
        _db.set_event(file_id, "unusable", db_path, conn=conn)
        _db.dequeue_files([file_id], db_path)
        return "unusable", False
    except LoadError:
        # A generic LoadError is a READ failure and may be transient (e.g. a
        # disconnected drive). Issue #69: this is NOT a classification, so it
        # must not return "non_event" — that string is indistinguishable, to
        # any caller, from a real "no rupture found" verdict.
        # "unavailable" is its own outcome; no verdict is cached,
        # so the file IS retried on the next pass — the drive may come back.
        return "unavailable", False

    result, stage1 = analyse_curve(
        curve, p, db_path=db_path, code_ver=code_ver, file_id=file_id, conn=conn,
    )

    # A UI edit is immediately visible to new work, but never injected into a
    # calculation already in flight.  If this curve finished under an older
    # snapshot, rerun before publishing its verdict/current event map.
    if _db.load_analysis_params(db_path).revision != param_set.revision:
        return analyse_and_classify(file_id, path, db_path, conn=conn)

    # A non_event still purges the legacy rupture_force_pn row, if any — see
    # db._EVENT_BUNDLE_TYPES. rupture/onset/invOLS are NOT purged here: they
    # persist across an event -> non_event flip like any other computed
    # value. The multi-event ROI document is dropped regardless
    # so a vanished ROI does not linger in the fit window or the 2DH windows.
    if not result.event:
        _db.delete_event_map(file_id, db_path, conn=conn)
    else:
        _persist_multi_event_roi(
            file_id, curve, db_path, code_ver, stage1=stage1,
            param_set=param_set,
        )

    if _db.load_analysis_params(db_path).revision != param_set.revision:
        return analyse_and_classify(file_id, path, db_path, conn=conn)

    # Cache the verdict so the next up-to-date pass short-circuits before load.
    _db.write_analysis_result(
        file_id, "event",
        1.0 if result.event else 0.0,
        p.all_params, code_ver, db_path, conn=conn,
    )
    return result.verdict, False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _persist_multi_event_roi(
    file_id: int, curve, db_path: str, code_ver: str,
    param_set: AnalysisParams,
    stage1: Stage1Search | None = None,
) -> None:
    """Store the multi-event ROI finder's full document (ROIs → ruptures →
    segments + per-segment WLC fits) to event_map, so the finder's result is
    persisted the moment analysis runs on a trace — available to the fit window,
    the 2DH windows, and any retrieval, without anyone re-running the finder
    downstream.

    `stage1`, when given, is analyse_curve's own intermediate state (decomposed
    curve, offset, invOLS, landmark indices) from THIS SAME pass over `curve` —
    passed straight through to compute_curve_events_coords so it doesn't
    re-decompose / re-fit-baseline / re-fit-invOLS / re-search for the outer
    rupture that analyse_curve just found seconds ago.  None is a safe default
    (falls back to computing/DB-cache-reading from scratch) for any future
    caller that doesn't have a fresh Stage1Search in hand.

    Keyed by the EventParams signature + scientific-method version. A curve already carrying
    a document under the current signature is left untouched (idempotent); a
    later inner/outer threshold change is a cache miss, so this file's next
    analysis/visit overwrites its one document (lazy per-file update — the DB
    never accumulates stale marks). This is now entirely compute_curve_events_
    coords's own doing: it checks the same signature and does the same write
    internally whenever it actually has to compute, so this function
    duplicates neither the cache check nor the write — it calls through.

    A characterization exception is raised as CurveAnalysisError. It is not
    converted into a non-event, and the caller does not publish the verdict as
    successfully analysed; failed calculation and scientific non-event remain
    distinct outcomes.
    """
    try:
        from .roi_pipeline import compute_curve_events_coords, event_params_from

        ep = event_params_from(param_set)
        compute_curve_events_coords(
            curve, ep, db_path=db_path, code_ver=code_ver, file_id=file_id,
            stage1=stage1, param_set=param_set,
        )
    except CurveAnalysisError:
        raise
    except Exception as exc:
        raise CurveAnalysisError("multi-event characterization", exc) from exc


