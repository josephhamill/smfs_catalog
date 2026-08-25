# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/roi_assembly.py
#
# Projection layer: turn the multi-event ROI structure (CurveEvents) into flat,
# tabular "data points" for ensembles / distributions — under a SELECTABLE mode,
# because what a curve means depends on the experiment.
#
# Modes (see project_curve_events):
#   'all'         — naive: every segment is a point, tagged by position.  No
#                   assumption about what the curve means or where it came from.
#   'first_of_2'  — tethered-molecule: ROIs with EXACTLY 2 ruptures.  Peak 1 is
#                   the MEASUREMENT (bond: force + WLC fit); peak 2 and the dX/dF
#                   to it are QUALITY CONTROL (the tether should break at a known
#                   force with a fixed dX/dF).  One row per qualifying ROI.
#   'last'        — single-molecule-expected: the TERMINAL peak is the measurement
#                   (the lone molecule).  n_ruptures flags contamination
#                   (penultimate=2 molecules, antepenultimate=3, …).  One row per
#                   ROI (or per curve — caller's choice via one_per_curve).
#
# Every row also carries identity (file_id/path), the ROI index, its rupture
# count and ordering, so the caller can filter (e.g. n_ruptures in {2,3,4}) and
# summarise dX/dF without re-deriving anything.
#
# Pure: no Qt, no DB, no I/O.  Operates on CurveEvents produced by roi_events.

from __future__ import annotations

from typing import Optional

from .roi_events import CurveEvents, ROI


MODES = ("all", "first_of_2", "last")


def _position(i: int, n: int) -> str:
    """Tag a rupture/segment by its position within an n-rupture ROI."""
    if n == 1:
        return "single"
    if i == 0:
        return "first"
    if i == n - 1:
        return "terminal"
    return "inner"


def _seg_row(roi: ROI, i: int, *, file_id, path, roi_index: int) -> dict:
    """One flat row for segment i (which ends on rupture i) of an ROI."""
    seg = roi.segments[i]
    rup = roi.ruptures[i]
    n   = roi.n_ruptures
    return {
        "file_id":     file_id,
        "path":        path,
        "roi_index":   roi_index,
        "n_ruptures":  n,
        "ordering":    roi.ordering,
        "seg_index":   i,
        "position":    _position(i, n),
        "l_p_nm":      seg.l_p_nm,
        "l_c_nm":      seg.l_c_nm,
        "l_p_err":     seg.l_p_err,
        "l_c_err":     seg.l_c_err,
        # Fit-conditioning diagnostics.  These travel WITH the error bars,
        # in the same row: uncertainty travels with the value it belongs to,
        # and an uncertainty that leaves the app without the number
        # explaining it cannot be read — l_p_err already has sqrt(tau) folded
        # in, and a
        # colleague reading the CSV months later cannot recover tau from it.
        "tau":         seg.tau,
        "z_max":       seg.z_max,
        "x_max_nm":    seg.x_max_nm,
        # 0/1 rather than True/False so the column is machine-readable in a CSV
        # alongside every other numeric column here.
        "edge_pinned": None if seg.edge_pinned is None else int(bool(seg.edge_pinned)),
        "n_fit_pts":   seg.n_pts,
        "rupture_force_pN": rup.force_pN,
        # dX/dF from the PREVIOUS rupture (None for the first).
        "dX_from_prev_nm": roi.dX_pairs[i - 1] if i > 0 else None,
        "dF_from_prev_pN": roi.dF_pairs[i - 1] if i > 0 else None,
    }


def project_curve_events(
    events:       CurveEvents,
    *,
    mode:         str,
    file_id=None,
    path:         Optional[str] = None,
    min_ruptures: int = 1,
    max_ruptures: Optional[int] = None,
) -> list[dict]:
    """
    Project one curve's CurveEvents into flat data-point rows under `mode`.

    ROIs are first filtered by rupture count: min_ruptures <= n <= max_ruptures
    (max_ruptures=None means no upper bound).  The mode then decides what rows
    each surviving ROI yields.  Returns [] if nothing qualifies.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")

    rows: list[dict] = []
    for ri, roi in enumerate(events.rois):
        n = roi.n_ruptures
        if n < min_ruptures or (max_ruptures is not None and n > max_ruptures):
            continue

        if mode == "all":
            for i in range(n):
                rows.append(_seg_row(roi, i, file_id=file_id, path=path, roi_index=ri))

        elif mode == "first_of_2":
            if n != 2:
                continue
            # Measurement = peak 1; QC = peak 2 + dX/dF.
            m = _seg_row(roi, 0, file_id=file_id, path=path, roi_index=ri)
            qc = _seg_row(roi, 1, file_id=file_id, path=path, roi_index=ri)
            m.update({
                "qc_rupture_force_pN": qc["rupture_force_pN"],
                "qc_l_c_nm":           qc["l_c_nm"],
                "qc_l_p_nm":           qc["l_p_nm"],
                "dX_1to2_nm":          roi.dX_pairs[0],
                "dF_1to2_pN":          roi.dF_pairs[0],
            })
            m["position"] = "first_of_2"
            rows.append(m)

        elif mode == "last":
            # Terminal segment = the measurement; n_ruptures is the contamination
            # flag (how many peaks precede the single-molecule one).
            row = _seg_row(roi, n - 1, file_id=file_id, path=path, roi_index=ri)
            row["position"] = "terminal"
            row["n_preceding"] = n - 1
            rows.append(row)

    return rows


def summarise_deltas(rows: list[dict]) -> dict:
    """
    Summary stats for dX/dF across projected rows.  Reads whichever delta keys
    the mode produced ('dX_from_prev_nm'/'dF_from_prev_pN' for 'all', or
    'dX_1to2_nm'/'dF_1to2_pN' for 'first_of_2').  Returns count/mean/std/min/max
    per available delta, skipping None.
    """
    import numpy as np

    out: dict = {}
    for xkey, fkey in (("dX_from_prev_nm", "dF_from_prev_pN"),
                       ("dX_1to2_nm",      "dF_1to2_pN")):
        for key in (xkey, fkey):
            vals = [r[key] for r in rows if r.get(key) is not None]
            if vals:
                a = np.asarray(vals, dtype=float)
                out[key] = {
                    "n":    int(a.size),
                    "mean": float(a.mean()),
                    "std":  float(a.std()),
                    "min":  float(a.min()),
                    "max":  float(a.max()),
                }
    return out
