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
# One AnalysisParams snapshot is acquired by the force-extension lane and
# projected by pipeline_params_from. Every stage in one curve pass receives that immutable
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
# analyse_file(file_id, path, db_path) → (verdict, was_cached)
#     Sends a file down the lane its stored modality names. MODALITY_PIPELINES
#     holds one lane per modality; the force-extension lane loads one
#     AnalysisParams snapshot, checks the verdict cache, loads and analyses on a
#     miss, persists the event map/verdict, and rejects publication if the
#     snapshot revision changed during the pass. Read/qualification failure
#     remains distinct from a scientifically verified non-event.
#
# This module has no Qt dependencies and is safe on the analysis worker thread.

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import NamedTuple

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
    # These are read by roi_pipeline.event_params_from (the multi-event
    # finder _persist_multi_event_roi runs downstream), so they belong in
    # params_roi: changing one must invalidate the verdict fast-path cache, or
    # the fast path returns early and the finder never runs against the new
    # setting.  Defaults/keys match event projection exactly, including inner
    # defaulting to outer.
    roi_inner_thr = _ps.roi_inner_threshold_nm_per_nm
    roi_detector_idx = _ps.roi_detector_mode_idx
    roi_prominence = _ps.roi_prominence
    roi_min_dist_pts = _ps.roi_min_distance_pts
    chain_model_idx = _ps.chain_model_idx
    rupture_algorithm_idx = _ps.rupture_algorithm_idx
    walk_back_fit_widths = _ps.walk_back_fit_widths
    walk_back_noise_multiple = _ps.walk_back_noise_multiple
    muggeo_fit_window_multiplier = _ps.muggeo_fit_window_multiplier
    muggeo_tolerance_idx = _ps.muggeo_tolerance_idx
    muggeo_restarts_idx = _ps.muggeo_restarts_idx

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
        "chain_model_idx":          chain_model_idx,
        "rupture_algorithm_idx":    rupture_algorithm_idx,
        "walk_back_fit_widths":     walk_back_fit_widths,
        "walk_back_noise_multiple": float(walk_back_noise_multiple),
        "muggeo_fit_window_multiplier": muggeo_fit_window_multiplier,
        "muggeo_tolerance_idx":     muggeo_tolerance_idx,
        "muggeo_restarts_idx":      muggeo_restarts_idx,
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
class CacheContext:
    """Cache identity for analyse_curve / compute_curve_events; param_set only gates scalar-cache reads and writes."""
    db_path: str | None
    code_ver: str | None
    file_id: int | None
    conn: object
    param_set: AnalysisParams | None = None


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


_NAN = float("nan")

# baseline_r2 is deliberately absent. R² reads near 0 on a good flat,
# undriven baseline because a flat anchor region has little variance for a
# line to explain, so the usual "near 1 is good" reading is backwards.
# baseline_rms is the real fit-quality signal.
_BL_KEYS = ("offset_retr", "flatness_slope", "baseline_intercept",
            "baseline_rms", "baseline_fit_lo_idx", "baseline_fit_hi_idx")
# invols_r2 deliberately absent — same reason as baseline_r2.
_IOL_KEYS = ("invols_slope", "invols_intercept",
             "invols_rms", "invols_fit_lo_idx", "invols_fit_hi_idx")


class _Baseline(NamedTuple):
    offset: float
    flatness: float
    intercept: float
    rms: float
    fit_lo_idx: int
    fit_hi_idx: int


class _Invols(NamedTuple):
    slope: float
    intercept: float
    rms: float
    fit_lo_idx: int
    fit_hi_idx: int
    dc: object | None


class _Contact(NamedTuple):
    contact_z: float
    snapoff_z: float
    snap_idx: int | None
    dc: object | None


class _Roi(NamedTuple):
    rupture_z: float
    onset_z: float
    rup_idx: int | None
    ons_idx: int | None
    sigs: object | None
    mask_postsnap_idx: int | None
    mask_anchor_idx: int | None
    outer_events: object | None
    dc: object | None


# Cache write policy for analyse_curve (optional — only when the CacheContext
# carries db_path, code_ver and file_id): each stage reads its values back and
# writes whatever it computes fresh (see the stage helpers for what is gated
# on an event verdict). There is no way for a caller to compute a value there
# and have it silently not saved. If two legitimate analysis producers compute
# the same value around the same time, the write is a replace rather than an
# append, so a redundant write remains safe. Display-only windows must read
# these results instead of becoming additional analysis producers.
def can_use_cache(cache: CacheContext | None) -> bool:
    """The one rule for when a CacheContext may be read or written: it exists and db_path, code_ver and file_id are all set."""
    return (cache is not None and cache.db_path is not None
            and cache.code_ver is not None and cache.file_id is not None)


def cache_get(cache: CacheContext | None, atype: str, params: str):
    """Read one stored scalar analysis result for the cache's file; returns None when can_use_cache is false."""
    if not can_use_cache(cache):
        return None
    return _db.get_analysis_result(cache.file_id, atype, params, cache.code_ver,
                                  cache.db_path, conn=cache.conn)


def _cache_put(cache: CacheContext | None, atype: str, value: float, params: str) -> None:
    if can_use_cache(cache):
        _db.write_analysis_result(cache.file_id, atype, value, params, cache.code_ver,
                                  cache.db_path, conn=cache.conn)


def _cache_get_multi(cache: CacheContext | None, atypes, params: str) -> dict:
    if not can_use_cache(cache):
        return {}
    return _db.get_analysis_results_multi(cache.file_id, atypes, params, cache.code_ver,
                                         cache.db_path, conn=cache.conn)


def cache_put_multi(cache: CacheContext | None, values: dict, params: str) -> None:
    """Write several scalar analysis results for the cache's file in one call; does nothing when can_use_cache is false."""
    if can_use_cache(cache):
        _db.write_analysis_results_multi(cache.file_id, values, params, cache.code_ver,
                                        cache.db_path, conn=cache.conn)


def _non_event(offset=_NAN, flatness=_NAN, contact_z=_NAN, snapoff_z=_NAN,
               invols_slope=_NAN) -> CurveResult:
    # rupture/onset stay NaN — a non_event has no ROI landmarks by
    # definition. invols_slope is NOT part of that gate: it's an
    # approach-only calibration fit, known regardless of event status once
    # we've reached this point, so it's passed through rather than blanked.
    return CurveResult(
        event=False, offset=offset, flatness=flatness,
        contact_z=contact_z, snapoff_z=snapoff_z,
        rupture_z=_NAN, onset_z=_NAN, invols_slope=invols_slope,
    )


def _attr(stage, name: str):
    return getattr(stage, name) if stage is not None else None


def _stage1_search(bl: _Baseline, iol, ct, roi, dc) -> Stage1Search:
    """
    Stage1Search from the final stage results; a stage the call never reached
    is None and leaves its fields None. `dc` is whichever DecomposedCurve the
    call last held (see _contact_stage for why it can be replaced).
    """
    slope = _attr(iol, "slope")
    return Stage1Search(
        dc=dc, offset=(bl.offset if not np.isnan(bl.offset) else None),
        snapoff_idx=_attr(ct, "snap_idx"), sigs=_attr(roi, "sigs"),
        mask_postsnap_idx=_attr(roi, "mask_postsnap_idx"),
        mask_anchor_idx=_attr(roi, "mask_anchor_idx"),
        rupture_idx=_attr(roi, "rup_idx"), onset_idx=_attr(roi, "ons_idx"),
        invols_slope=(None if slope is None or np.isnan(slope) else slope),
        outer_events=_attr(roi, "outer_events"),
        invols_intercept=_attr(iol, "intercept"),
        invols_rms=_attr(iol, "rms"),
        baseline_intercept=bl.intercept,
        baseline_rms=bl.rms,
        baseline_fit_lo_idx=bl.fit_lo_idx,
        baseline_fit_hi_idx=bl.fit_hi_idx,
        invols_fit_lo_idx=_attr(iol, "fit_lo_idx"),
        invols_fit_hi_idx=_attr(iol, "fit_hi_idx"),
    )


def _fit_baseline(curve, p) -> _Baseline:
    try:
        from .signal_processing import fit_retract_baseline
        bl = fit_retract_baseline(curve, float(p.anchor_nm))
        return _Baseline(bl.offset, bl.slope, bl.intercept, bl.rms,
                         bl.fit_lo_idx, bl.fit_hi_idx)
    except Exception as exc:
        raise CurveAnalysisError("retract baseline", exc) from exc


def _baseline_stage(curve, p, cache: CacheContext | None) -> _Baseline:
    """
    Retract baseline (params_bl): offset/flatness plus the QC diagnostics,
    read from the cache or fitted and, when the offset is finite, written back.
    Always read/written for every curve, event or not.
    """
    cached = _cache_get_multi(cache, _BL_KEYS, p.params_bl)   # one round-trip, not six
    # Require ALL SIX rows, not just offset/flatness: a row holding only those
    # two would register as a permanent cache hit and never backfill the
    # diagnostic fields. Recomputing here is cheap — a small polyfit over the
    # anchor window, not a curve reload/decompose.
    if all(k in cached for k in _BL_KEYS):
        return _Baseline(
            float(cached["offset_retr"]), float(cached["flatness_slope"]),
            float(cached["baseline_intercept"]), float(cached["baseline_rms"]),
            int(cached["baseline_fit_lo_idx"]), int(cached["baseline_fit_hi_idx"]),
        )
    bl = _fit_baseline(curve, p)
    if not np.isnan(bl.offset):
        cache_put_multi(cache, {
            "offset_retr":         bl.offset,
            "flatness_slope":      bl.flatness,
            "baseline_intercept":  float(bl.intercept),
            "baseline_rms":        float(bl.rms),
            "baseline_fit_lo_idx": float(bl.fit_lo_idx),
            "baseline_fit_hi_idx": float(bl.fit_hi_idx),
        }, p.params_bl)
    return bl


def _fit_invols(curve, p) -> _Invols:
    from .signal_processing import decompose_curve, fit_approach_invols
    try:
        dc = decompose_curve(curve, cutoff_hz=float(p.cutoff_hz))
        fit = fit_approach_invols(
            dc.low_appr, curve.piezo_appr, p.invols_offset_pts, p.invols_window_pts,
        )
        out = _Invols(fit.slope, fit.intercept, fit.rms,
                      fit.fit_lo_idx, fit.fit_hi_idx, dc)
    except Exception as exc:
        raise CurveAnalysisError("approach invOLS", exc) from exc
    if np.isnan(out.slope):
        raise CurveAnalysisError(
            "approach invOLS",
            ValueError("fit window is degenerate; no calibration slope was produced"),
        )
    return out


def _invols_stage(curve, p, cache: CacheContext | None) -> _Invols:
    """
    invOLS fit (params_invols). Approach-only (curve.piezo_appr / the
    decomposed low-frequency approach channel) — independent of the retract
    baseline and of whether this curve turns out to hold a validated rupture.
    Computed and stored for every curve that gets this far, event or not: it
    is a calibration fit, unlike the event-only landmarks, which are gated on
    a validated event. `dc` is set only when this call decomposed the curve.
    """
    cached = _cache_get_multi(cache, _IOL_KEYS, p.params_invols)   # one round-trip
    # Same completeness requirement as the baseline: a bare invols_slope row
    # cached before the diagnostic fields existed would otherwise be a
    # permanent cache hit that never backfills them.
    if all(k in cached for k in _IOL_KEYS):
        return _Invols(
            float(cached["invols_slope"]), float(cached["invols_intercept"]),
            float(cached["invols_rms"]), int(cached["invols_fit_lo_idx"]),
            int(cached["invols_fit_hi_idx"]), None,
        )
    iol = _fit_invols(curve, p)
    cache_put_multi(cache, {
        "invols_slope":      float(iol.slope),
        "invols_intercept":  float(iol.intercept),
        "invols_rms":        float(iol.rms),
        "invols_fit_lo_idx": float(iol.fit_lo_idx),
        "invols_fit_hi_idx": float(iol.fit_hi_idx),
    }, p.params_invols)
    return iol


def _fit_contact(curve, p) -> _Contact:
    from .signal_processing import (
        decompose_curve, find_begin_in_contact, find_end_in_contact,
    )
    try:
        dc = decompose_curve(curve, cutoff_hz=float(p.cutoff_hz))
        ci, _ = find_begin_in_contact(
            dc, var_window_ms=float(p.var_window_ms),
            threshold=float(p.thresh_appr), trim_pts=p.trim_pts,
        )
        si, _ = find_end_in_contact(
            dc, var_window_ms=float(p.var_window_ms),
            threshold=float(p.thresh_retr), trim_pts=p.trim_pts,
        )
        return _Contact(float(curve.piezo_appr[ci]), float(curve.piezo_retr[si]), si, dc)
    except Exception as exc:
        raise CurveAnalysisError("contact/snap-off detection", exc) from exc


def _contact_stage(curve, p, cache: CacheContext | None, dc) -> _Contact:
    """
    Contact/snap-off (params_cd), likewise always read/written. `dc` is the
    DecomposedCurve the invOLS stage may already hold; it is passed through on
    a cache hit.

    A cache miss re-decomposes even when `dc` is already set, so the returned
    DecomposedCurve is a new object (Stage1Search.dc identity changes). This
    is long-standing behaviour and is deliberately preserved.
    """
    contact_z = cache_get(cache, "contact_piezo_nm", p.params_cd)
    snapoff_z = cache_get(cache, "snapoff_piezo_nm", p.params_cd)
    if contact_z is None or snapoff_z is None:
        ct = _fit_contact(curve, p)
        if not np.isnan(ct.contact_z):
            _cache_put(cache, "contact_piezo_nm", ct.contact_z, p.params_cd)
            _cache_put(cache, "snapoff_piezo_nm", ct.snapoff_z, p.params_cd)
            if ct.snap_idx is not None:
                _cache_put(cache, "snapoff_idx", float(ct.snap_idx), p.params_cd)
        return ct
    # Values hit cache, but the INDEX wasn't necessarily remembered from a
    # prior pass — recover it so the ROI stage doesn't have to re-run
    # find_end_in_contact just to get back an index we already have.
    snap_idx = cache_get(cache, "snapoff_idx", p.params_cd)
    return _Contact(float(contact_z), float(snapoff_z),
                    None if snap_idx is None else int(snap_idx), dc)


def _fresh_roi(curve, p, offset: float, snap_idx: int | None, dc) -> _Roi:
    from .signal_processing import decompose_curve, find_end_in_contact
    from .roi_detection import compute_detection_signals, rupture_search_bounds
    from .roi_events import find_outer_events
    try:
        if dc is None:
            dc = decompose_curve(curve, cutoff_hz=float(p.cutoff_hz))
        if snap_idx is not None:
            si_roi = snap_idx
        else:
            si_roi, _ = find_end_in_contact(
                dc, var_window_ms=float(p.var_window_ms),
                threshold=float(p.thresh_retr), trim_pts=p.trim_pts,
            )
        low_corr = dc.low_retr - offset
        sigs = compute_detection_signals(
            low_retr   = low_corr,
            piezo_retr = curve.piezo_retr,
            window_pts = p.roi_win_pts,
        )
        lo_band, hi_band = rupture_search_bounds(
            sigs.piezo, si_roi, float(p.anchor_nm), p.roi_post_mask,
        )
        outer_events = find_outer_events(
            sigs, (lo_band, hi_band), p.roi_onset_thr, p.roi_d1_thr,
        )
        rup_idx = ons_idx = None
        rupture_z = onset_z = _NAN
        # Scalar compatibility fields describe the right-most validated
        # outer event. Inner segmentation later consumes this exact list.
        if outer_events:
            primary_outer = outer_events[-1]
            rup_idx = primary_outer.terminal_idx
            ons_idx = primary_outer.onset_idx
            rupture_z = float(sigs.piezo[rup_idx])
            onset_z = float(sigs.piezo[ons_idx])
        return _Roi(rupture_z, onset_z, rup_idx, ons_idx, sigs,
                    lo_band, hi_band, outer_events, dc)
    except Exception as exc:
        raise CurveAnalysisError("ROI landmark detection", exc) from exc


def _roi_stage(curve, p, cache: CacheContext | None, offset: float, ct: _Contact) -> _Roi:
    """
    ROI search (params_roi): rupture/onset are read back to skip recompute, but
    only WRITTEN for a validated event (see _landmark_stage). Fresh rupture/
    onset are not persisted here; invOLS was persisted independently, and
    force belongs to downstream per-segment characterization and is never
    computed here.
    """
    rupture_z = cache_get(cache, "rupture_piezo_nm", p.params_roi)
    onset_z   = cache_get(cache, "onset_piezo_nm",   p.params_roi)
    if rupture_z is None or onset_z is None:
        return _fresh_roi(curve, p, offset, ct.snap_idx, ct.dc)
    # Same recovery as snap-off: the piezo-nm values hit cache, but pull the
    # indices back too so same-pass multi-event characterization does not have
    # to re-derive them from the curve.
    rup_idx = cache_get(cache, "rupture_idx", p.params_roi)
    ons_idx = cache_get(cache, "onset_idx",   p.params_roi)
    return _Roi(float(rupture_z), float(onset_z),
                None if rup_idx is None else int(rup_idx),
                None if ons_idx is None else int(ons_idx),
                None, None, None, None, ct.dc)


def _landmarks_valid(roi: _Roi, snapoff_z: float) -> bool:
    # Rupture and onset must both exist, and the rupture must lie after
    # snap-off in piezo space.
    return not (np.isnan(roi.rupture_z) or np.isnan(roi.onset_z)
                or (roi.rupture_z - snapoff_z) <= 0)


def _landmark_stage(curve, p, cache: CacheContext | None, roi: _Roi) -> _Roi:
    """
    Persist a validated event's landmarks and return `roi` with both indices
    filled in. An event is defined by landmarks alone: baseline OK, snap-off
    found, both ROI landmarks found, rupture after snap-off (_landmarks_valid);
    there is no separate force condition. invols_slope is NOT written here — it
    was already written by the invOLS stage, unconditionally, because it isn't
    gated by event status.
    """
    # rup_idx/ons_idx may still be None if rupture_z/onset_z hit cache but
    # their index counterparts predate index caching — recover them from the
    # piezo-nm values so the backfill below still runs.
    rup_idx, ons_idx = roi.rup_idx, roi.ons_idx
    if rup_idx is None:
        rup_idx = int(np.argmin(np.abs(curve.piezo_retr - roi.rupture_z)))
    if ons_idx is None:
        ons_idx = int(np.argmin(np.abs(curve.piezo_retr - roi.onset_z)))
    _cache_put(cache, "rupture_piezo_nm", float(roi.rupture_z), p.params_roi)
    _cache_put(cache, "onset_piezo_nm",   float(roi.onset_z),   p.params_roi)
    # Indices alongside their piezo-nm counterparts, so a later pass (this
    # curve's own multi-event enrichment, the ROI viewer, a batch backfill) can
    # read the landmark position back without re-searching for it.
    _cache_put(cache, "rupture_idx", float(rup_idx), p.params_roi)
    _cache_put(cache, "onset_idx",   float(ons_idx), p.params_roi)
    return roi._replace(rup_idx=rup_idx, ons_idx=ons_idx)


def analyse_curve(
    curve,
    p:     _PipelineParams,
    *,
    cache: CacheContext | None = None,
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

    Caching is optional and follows the write policy above can_use_cache.
    """
    # Only p.anchor_nm / p.params_bl are read before the baseline (checked by
    # test_curve_analysis_contract).
    bl = _baseline_stage(curve, p, cache)
    if np.isnan(bl.offset):
        return _non_event(), _stage1_search(bl, None, None, None, None)
    iol = _invols_stage(curve, p, cache)
    ct = _contact_stage(curve, p, cache, iol.dc)
    if np.isnan(ct.snapoff_z):
        return (_non_event(bl.offset, bl.flatness, ct.contact_z, ct.snapoff_z, iol.slope),
                _stage1_search(bl, iol, ct, None, ct.dc))
    roi = _roi_stage(curve, p, cache, bl.offset, ct)
    if not _landmarks_valid(roi, ct.snapoff_z):
        return (_non_event(bl.offset, bl.flatness, ct.contact_z, ct.snapoff_z, iol.slope),
                _stage1_search(bl, iol, ct, roi, roi.dc))
    roi = _landmark_stage(curve, p, cache, roi)
    return CurveResult(
        event=True, offset=bl.offset, flatness=bl.flatness,
        contact_z=ct.contact_z, snapoff_z=ct.snapoff_z,
        rupture_z=float(roi.rupture_z), onset_z=float(roi.onset_z),
        invols_slope=float(iol.slope),
    ), _stage1_search(bl, iol, ct, roi, roi.dc)


def current_signature(db_path: str) -> tuple[str, str | None]:
    """
    (all_params, scientific-method version) — the exact pair the FAST PATH tests a
    stored verdict against.  Nothing else; this is a read of the live
    settings, not an analysis.

    Exists so a caller can ask "which queued files would the fast path serve?"
    WITHOUT running the pipeline.  The dashboard needs that to
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


def _lane_no_pipeline_yet(
    file_id: int, path: str, db_path: str, conn=None,
) -> tuple[str, bool]:
    """The lane for a modality whose analysis is not written yet."""
    return "unanalysed", True


def _lane_force_extension(
    file_id: int,
    path:    str,
    db_path: str,
    conn=None,
) -> tuple[str, bool]:
    """
    The force-extension lane: persistence wrapper around analyse_curve, caching
    the verdict and every derived value in analysis_results.

    Returns (verdict, was_cached):
        verdict:
            'event'       — a valid event (an ROI excursion) was located
            'non_event'   — no ROI / event could be identified
            'unavailable' — the curve could not be READ at all (e.g. a
                             disconnected drive). Not a classification, and it
                             must not masquerade as one. No verdict is cached,
                             so the next pass retries.
            'unusable'    — the curve's own data is damaged (e.g. a channel
                             full of NaN). Also not a classification. Unlike
                             'unavailable' it will not fix itself, so the
                             reason is recorded on the file.
        was_cached: True iff the answer came without a disk read.
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
            return _lane_force_extension(file_id, path, db_path, conn=conn)
        return ("event" if cached_verdict >= 0.5 else "non_event"), True

    # ── SLOW PATH ─────────────────────────────────────────────────────────────
    # Verdict not cached for these params → load the curve and run the pipeline.
    try:
        curve = load_force_curve(path)
    except UnusableCurveError as exc:
        # The file read fine but its data is damaged (curve_loader.qualify_wave):
        # a dead channel, a constant channel, no turnaround, or an all-zero
        # retract.  Record WHY on the file, where the user can see it, and
        # return 'unusable' rather than 'non_event' — nothing was classified,
        # and calling it a non_event would report a verdict never reached.
        #
        # Queue membership is untouched.  The queue is the timeline the
        # navigator walks, so removing a row here would delete curves out from
        # under someone paging through them.
        _db.set_unusable_reason(file_id, exc.reason, str(exc), db_path, conn=conn)
        _db.set_event(file_id, "unusable", db_path, conn=conn)
        return "unusable", False
    except LoadError:
        # A generic LoadError is a READ failure and may be transient (e.g. a
        # disconnected drive). This is NOT a classification, so it
        # must not return "non_event" — that string is indistinguishable, to
        # any caller, from a real "no rupture found" verdict.
        # "unavailable" is its own outcome; no verdict is cached,
        # so the file IS retried on the next pass — the drive may come back.
        return "unavailable", False

    result, stage1 = analyse_curve(
        curve, p, cache=CacheContext(db_path, code_ver, file_id, conn),
    )

    # A UI edit is immediately visible to new work, but never injected into a
    # calculation already in flight.  If this curve finished under an older
    # snapshot, rerun before publishing its verdict/current event map.
    if _db.load_analysis_params(db_path).revision != param_set.revision:
        return _lane_force_extension(file_id, path, db_path, conn=conn)

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
        return _lane_force_extension(file_id, path, db_path, conn=conn)

    # The curve is already in memory on this path, so its deflection histogram
    # is one more small calculation. It takes no parameters, so it is stored
    # once and only computed here when no row exists yet. The fast path above
    # never loads the curve and does not do this.
    _db.get_or_store_deflection_histogram(
        file_id, curve.defl_retr, db_path, conn=conn)

    # Cache the verdict so the next up-to-date pass short-circuits before load.
    _db.write_analysis_result(
        file_id, "event",
        1.0 if result.event else 0.0,
        p.all_params, code_ver, db_path, conn=conn,
    )
    return result.verdict, False


# One lane per acquisition modality, covering every value curve_loader._modality
# can return.  Indexed, not searched: a modality with no lane raises here rather
# than being quietly treated as something else.
MODALITY_PIPELINES = {
    "continuous_stretch": _lane_force_extension,
    "stretch_hold":       _lane_no_pipeline_yet,
    "force_clamp":        _lane_no_pipeline_yet,
    "indentation":        _lane_no_pipeline_yet,
    "image_contact":      _lane_no_pipeline_yet,
    "image_ac":           _lane_no_pipeline_yet,
    "unknown":            _lane_no_pipeline_yet,
}


def analyse_file(
    file_id: int,
    path:    str,
    db_path: str,
    conn=None,
) -> tuple[str, bool]:
    """Send one file down the lane its modality names.

    The scanner reads the modality from the file's own wave note at import and
    stores it, so this is a lookup.  Nothing else takes part in the decision:
    what a file IS settles where it goes, and what is wrong with its data is
    the lane's business once it gets there.
    """
    lane = MODALITY_PIPELINES[_db.get_curve_type(file_id, db_path, conn=conn)]
    return lane(file_id, path, db_path, conn=conn)


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
            curve, ep,
            cache=CacheContext(db_path, code_ver, file_id, None, param_set=param_set),
            stage1=stage1,
        )
    except CurveAnalysisError:
        raise
    except Exception as exc:
        raise CurveAnalysisError("multi-event characterization", exc) from exc


