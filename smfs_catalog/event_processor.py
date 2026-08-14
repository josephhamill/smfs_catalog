# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/event_processor.py
#
# 2DH grid constants and histogram builders shared by normalized_2dh_window.py
# and physical_2dh_window.py — both windows read stored per-segment WLC fits
# directly and call the compute_*_histogram functions here. This module has no
# curve-loading or WLC-fitting side effects.
#
# Grid constants (WLC_*) are the defaults imported by normalized_2dh_window.py.
# _wlc_grid_params() is the cache-key builder; bump its version whenever the
# histogram calculation changes without changing one of its explicit inputs.

from __future__ import annotations

import json

import numpy as np

from .models import normalize_wlc

# ── WLC-normalised 2DH grid ───────────────────────────────────────────────────
WLC_X_BINS  = 128
WLC_F_BINS  = 128
WLC_X_RANGE = (-0.1, 1.2)
WLC_F_RANGE = (-5.0,  45.0)

# ── Physical (force-registered) 2DH grid ─────────────────────────────────────
PHYS_X_BINS  = 128
PHYS_F_BINS  = 128
PHYS_X_RANGE = (-200.0, 400.0)   # nm, relative to the selected anchor
PHYS_F_RANGE = (  -200.0, 600.0)  # pN


# ── The 2DH view defaults, in ONE place ───────────────────────────────────────
# The seed values for a profile that has never been tuned.  They live HERE, in
# the Qt-free module both 2DH windows already import, so that db.py can seed them
# into the DEFAULT_EXPERIMENTALIST row at initialise() without importing a
# window. Keeping them here ensures both windows share one default policy.
#
# Rupture alignment is fit-free, so curves without a valid WLC fit can still be
# registered in physical coordinates.
PHYS_ALIGN_DEFAULT = "rupture"

# "last" selects the final segment of the target ROI. One constant governs
# both 2DH windows.
ALIGN_SEG_DEFAULT = "last"

# This reference applies only to `fstar` alignment mode.
PHYS_F_STAR_DEFAULT = 50.0


def _wlc_grid_params(
    x_bins:  int   = WLC_X_BINS,
    f_bins:  int   = WLC_F_BINS,
    x_range: tuple = WLC_X_RANGE,
    f_range: tuple = WLC_F_RANGE,
    align_segment: str = ALIGN_SEG_DEFAULT,
) -> str:
    """Cache key for normalized 2DH, including every histogram-changing choice."""
    return json.dumps(
        {"x_bins": x_bins, "f_bins": f_bins,
         "x_min": x_range[0], "x_max": x_range[1],
         "f_min": f_range[0], "f_max": f_range[1],
         "seg": align_segment, "v": 6},  # v6: uint32 count storage
        separators=(",", ":"),
    )


def _phys_grid_params(
    F_star:  float,
    x_bins:  int   = PHYS_X_BINS,
    f_bins:  int   = PHYS_F_BINS,
    x_range: tuple = PHYS_X_RANGE,
    f_range: tuple = PHYS_F_RANGE,
    align_mode:    str = PHYS_ALIGN_DEFAULT,
    align_segment: str = ALIGN_SEG_DEFAULT,
) -> str:
    """Cache key for physical 2DH — every choice that changes the histogram is
    part of the key, so a different alignment or fit-segment recomputes."""
    return json.dumps(
        {"type": "phys", "x_bins": x_bins, "f_bins": f_bins,
         "x_min": x_range[0], "x_max": x_range[1],
         "f_min": f_range[0], "f_max": f_range[1],
         "F_star_pN": float(F_star),
         "align": align_mode, "seg": align_segment, "v": 4},   # v4: uint32 count storage
        separators=(",", ":"),
    )


def _wlc_x_at_force(F_target: float, l_p: float, l_c: float) -> float | None:
    """
    Return extension x (nm) where WLC(x, l_p, l_c) = F_target (pN).
    Brent's method on the monotone WLC curve in (0, l_c).
    Returns None if F_target is unreachable for these parameters.
    """
    from scipy.optimize import brentq
    from .models import wlc as _wlc_model
    try:
        x_lo, x_hi = 1e-6 * l_c, 0.9999 * l_c
        if float(_wlc_model(x_hi, l_p, l_c)) < F_target:
            return None   # F_target above WLC singularity — unreachable
        return float(
            brentq(lambda x: float(_wlc_model(x, l_p, l_c)) - F_target, x_lo, x_hi)
        )
    except Exception:
        return None


# ── Physical-2DH alignment: one anchor subroutine per registration mode ───────
# Each returns the extension x (nm) that maps to Δx = 0 for one curve; the
# histogram then plots (x − anchor) vs F.  Kept as separate, named functions so
# the grid-box "Align at" menu maps one-to-one onto them — adding a mode = add a
# function + a dict entry + a menu label.  Uniform signature; each uses only what
# it needs.  Return None to drop the curve (anchor undefined for it).
#
# rupture_x is the chosen segment's terminating rupture, converted from a stored curve index
# to an x (nm) position by the caller — this module has no curve loaded, so
# it can't do that conversion itself, only consume the result.

def phys_anchor_onset(x, F, l_p, l_c, F_star, rupture_x=None):
    """Δx=0 at the loading-ramp start (first sample of the ROI slice = onset).
    Fit-free — immune to a bad WLC fit."""
    return float(x[0]) if len(x) else None

def phys_anchor_snapoff(x, F, l_p, l_c, F_star, rupture_x=None):
    """Δx=0 at tip–surface contact — x is already measured from snap-off."""
    return 0.0

def phys_anchor_fstar(x, F, l_p, l_c, F_star, rupture_x=None):
    """Δx=0 where the chosen segment's WLC fit reaches force F*."""
    return _wlc_x_at_force(F_star, l_p, l_c)

def phys_anchor_lc(x, F, l_p, l_c, F_star, rupture_x=None):
    """Δx=0 at the chosen segment's fitted contour length l_c."""
    return float(l_c) if l_c else None

def phys_anchor_rupture(x, F, l_p, l_c, F_star, rupture_x=None):
    """Δx=0 at the CHOSEN segment's own terminating rupture (first/penult/
    last — same segment choice as fstar/lc). Fit-free — a real data point,
    not a model extrapolation, unlike fstar/lc."""
    return float(rupture_x) if rupture_x is not None else None

PHYS_ALIGN_ANCHORS = {
    "onset":   phys_anchor_onset,
    "fstar":   phys_anchor_fstar,
    "snapoff": phys_anchor_snapoff,
    "lc":      phys_anchor_lc,
    "rupture": phys_anchor_rupture,
}
# Declared with the other anchors so the menu and its default sit together; the
# value itself is set with the rest of the 2DH view defaults above.


def compute_physical_histogram_at(
    x: np.ndarray, F: np.ndarray, anchor: float | None,
    x_bins:  int   = PHYS_X_BINS,
    f_bins:  int   = PHYS_F_BINS,
    x_range: tuple = PHYS_X_RANGE,
    f_range: tuple = PHYS_F_RANGE,
) -> np.ndarray | None:
    """(PHYS_X_BINS × PHYS_F_BINS) uint32 count histogram of (x − anchor) vs F, where
    `anchor` (nm) is the extension mapped to Δx=0 by one of the phys_anchor_*
    subroutines.  Returns None when anchor is None (the curve is dropped)."""
    if anchor is None:
        return None
    H, _, _ = np.histogram2d(
        x - anchor, F,
        bins=[x_bins, f_bins],
        range=[x_range, f_range],
    )
    return H.astype(np.uint32)


def compute_physical_histogram(
    x: np.ndarray, F: np.ndarray,
    l_p: float, l_c: float,
    F_star: float,
    x_bins:  int   = PHYS_X_BINS,
    f_bins:  int   = PHYS_F_BINS,
    x_range: tuple = PHYS_X_RANGE,
    f_range: tuple = PHYS_F_RANGE,
) -> np.ndarray | None:
    """Back-compatible F*-registered histogram (Δx = x − x*(F*)): the F* anchor
    plus compute_physical_histogram_at.  Returns None if x*(F*) is unreachable."""
    return compute_physical_histogram_at(
        x, F, _wlc_x_at_force(F_star, l_p, l_c),
        x_bins=x_bins, f_bins=f_bins, x_range=x_range, f_range=f_range,
    )


def compute_wlc_histogram(
    x: np.ndarray, F: np.ndarray, l_p: float, l_c: float,
    x_bins:  int   = WLC_X_BINS,
    f_bins:  int   = WLC_F_BINS,
    x_range: tuple = WLC_X_RANGE,
    f_range: tuple = WLC_F_RANGE,
) -> np.ndarray:
    """Return a (x_bins × f_bins) uint32 count histogram in normalised WLC coords."""
    x_n, F_n = normalize_wlc(x, F, l_p, l_c)
    H, _, _ = np.histogram2d(
        x_n, F_n,
        bins=[x_bins, f_bins],
        range=[x_range, f_range],
    )
    return H.astype(np.uint32)
