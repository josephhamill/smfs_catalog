# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/roi_detection.py
#
# Detection signals for SMFS ROI (region-of-interest) search.
#
# All functions operate on the LOW-frequency channel of the retract trace in
# its natural forward orientation (index 0 = in-contact end, last index = far
# baseline).  Signals are returned in the same orientation so they can be
# plotted directly against piezo_retr without axis gymnastics.
#
# The search direction (back-to-front) is a separate concern and lives in
# find_rupture() — do not conflate orientation of the metric with direction
# of the scan.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter


# ── Individual detection signal functions ─────────────────────────────────────

def signal_d1(
    low_retr:   np.ndarray,
    piezo_retr: np.ndarray,
    window_pts: int,
) -> np.ndarray:
    """
    First derivative of the forward low channel via Savitzky-Golay, in physical
    units of nm deflection per nm piezo travel (dimensionless slope).

    Signed: uses the mean piezo step WITH sign (not absolute value) so d1 is
    a proper d(defl)/d(piezo).  At a rupture the defl jumps sharply over a
    small piezo range — |d1| peaks; the sign of the peak depends on whether
    the stored piezo samples run low→high or high→low through the retract.
    """
    low_retr = np.asarray(low_retr, dtype=float)
    piezo_retr = np.asarray(piezo_retr, dtype=float)
    if low_retr.ndim != 1 or piezo_retr.ndim != 1:
        raise ValueError("ROI detection signals must be one-dimensional")
    if low_retr.size != piezo_retr.size:
        raise ValueError("low_retr and piezo_retr must have the same length")
    if low_retr.size < 3:
        raise ValueError("ROI derivative requires at least 3 samples")
    if not np.all(np.isfinite(low_retr)) or not np.all(np.isfinite(piezo_retr)):
        raise ValueError("ROI detection signals must contain only finite values")
    wp = _bounded_odd(window_pts, low_retr.size)
    d1_per_sample = savgol_filter(low_retr, wp, polyorder=2, deriv=1)
    d_piezo = float(np.mean(np.diff(piezo_retr)))
    if d_piezo == 0:
        d_piezo = 1.0
    return d1_per_sample / d_piezo


def signal_mean_dev(
    low_retr:   np.ndarray,
    window_pts: int,
) -> np.ndarray:
    """
    Sliding window mean, in nm deflection.

    `low_retr` is required to already be baseline-subtracted by the caller
    (see compute_detection_signals) — so its baseline region should already
    sit at zero, and this is a plain sliding mean, not a deviation from a
    second, independently-remeasured reference.  Baseline → near zero; event
    region → departs.  Not normalised.
    """
    low_retr = np.asarray(low_retr, dtype=float)
    if low_retr.ndim != 1 or low_retr.size == 0:
        raise ValueError("low_retr must be a non-empty one-dimensional signal")
    if not np.all(np.isfinite(low_retr)):
        raise ValueError("low_retr must contain only finite values")
    wp = _bounded_odd(window_pts, low_retr.size, minimum=1)
    return uniform_filter1d(low_retr, size=wp, mode='nearest')


# ── Orchestrator ──────────────────────────────────────────────────────────────

@dataclass
class DetectionSignals:
    """
    Detection signals for one curve, in forward (natural) retract orientation.

    All arrays share the same length and are indexed 0 → in-contact end,
    last → far baseline.  Plot directly against `piezo`.

    Only d1 and mean_dev are computed.
    """
    piezo:    np.ndarray   # forward piezo (nm)
    low:      np.ndarray   # forward low channel (nm), baseline-subtracted
    d1:       np.ndarray   # first derivative (nm deflection / nm piezo)
    mean_dev: np.ndarray   # sliding mean deviation from baseline (nm)


def compute_detection_signals(
    low_retr:   np.ndarray,
    piezo_retr: np.ndarray,
    window_pts: int = 51,
) -> DetectionSignals:
    """
    Compute detection signals on the forward retract low channel.

    Caller is expected to have already baseline-subtracted `low_retr` using
    the ONE stored offset characterization (signal_processing.fit_retract_
    baseline, persisted as offset_retr/flatness_slope) — this function does
    NOT independently re-measure that region. It used to, via a second,
    independent baseline measurement of the same far-retract stretch on the
    smoothed signal; checked against three real curves (2026-07-24), the two
    agreed to within ~1e-4 nm, so there was no real difference being
    measured, just the same answer computed twice (removed 2026-07-29).
    mean_dev now reads directly off the caller's already-corrected signal
    instead of re-deriving its own reference.

    A requested window longer than the trace is reduced to the largest supported
    odd window. Fewer than three samples cannot support the quadratic derivative
    and raises a clear ValueError.
    """
    window_pts = _ensure_odd(window_pts)

    low   = np.asarray(low_retr,   dtype=float)
    piezo = np.asarray(piezo_retr, dtype=float)

    d1   = signal_d1(low, piezo, window_pts)
    mdev = signal_mean_dev(low, window_pts)

    return DetectionSignals(
        piezo    = piezo,
        low      = low,
        d1       = d1,
        mean_dev = mdev,
    )


# ── ROI rupture search ────────────────────────────────────────────────────────

@dataclass
class RuptureSearchResult:
    """
    Outcome of a back-to-front threshold scan on |d1|.

    rupture_idx      : index into the forward d1 / piezo arrays.  -1 if no
                       crossing was found in the searchable region.
    rupture_piezo_nm : piezo value at that index (nan if not found).
    mask_anchor_idx  : first forward index inside the far-retract baseline
                       characterization region (exclusive search upper bound).
    mask_postsnap_idx: last forward index that is INSIDE the post-snap-off
                       mask (exclusive lower bound of the search).
    """
    rupture_idx:       int
    rupture_piezo_nm:  float
    mask_anchor_idx:   int
    mask_postsnap_idx: int


def rupture_search_bounds(
    piezo: np.ndarray,
    snapoff_idx: int,
    anchor_nm: float,
    post_snapoff_mask_nm: float,
) -> tuple[int, int]:
    """Return the shared [post-snap-off, baseline-anchor) search band."""
    n = len(piezo)
    if n == 0 or snapoff_idx < 0 or snapoff_idx >= n:
        return 0, n
    total_range = abs(float(piezo[-1]) - float(piezo[0]))
    frac_anchor = min(anchor_nm / total_range, 1.0) if total_range > 0 else 0.15
    n_anchor = max(10, int(round(frac_anchor * n)))
    mask_anchor_idx = n - n_anchor
    frac_post = min(post_snapoff_mask_nm / total_range, 1.0) if total_range > 0 else 0.0
    n_post = max(0, int(round(frac_post * n)))
    mask_postsnap_idx = min(n, snapoff_idx + n_post)
    return mask_postsnap_idx, mask_anchor_idx


def find_rupture(
    d1:                    np.ndarray,
    piezo:                 np.ndarray,
    snapoff_idx:           int,
    anchor_nm:             float,
    post_snapoff_mask_nm:  float,
    threshold_nm_per_nm:   float,
) -> RuptureSearchResult:
    """
    Scan d1 back-to-front (from the far-baseline side toward the surface)
    and return the first index whose SIGNED value exceeds threshold_nm_per_nm.

    A WLC rupture gives a known-sign pulse (positive d(defl)/d(piezo) in the
    conventional orientation) — testing the signed value rejects the opposite
    polarity pulses that |d1| would spuriously catch.

    The search region EXCLUDES:
      - the baseline characterization region (last anchor_nm of piezo travel), and
      - the first post_snapoff_mask_nm of piezo travel past snapoff_idx.

    Returns rupture_idx = -1 if no crossing is found.  Callers can still
    draw the mask extents even on a miss.
    """
    n = len(d1)
    if n == 0 or n != len(piezo) or snapoff_idx < 0 or snapoff_idx >= n:
        return RuptureSearchResult(-1, float('nan'), n, 0)

    mask_postsnap_idx, mask_anchor_idx = rupture_search_bounds(
        piezo, snapoff_idx, anchor_nm, post_snapoff_mask_nm,
    )

    # Searchable region: [mask_postsnap_idx, mask_anchor_idx)
    search_hi = mask_anchor_idx
    search_lo = mask_postsnap_idx
    if search_lo >= search_hi:
        return RuptureSearchResult(-1, float('nan'), mask_anchor_idx, mask_postsnap_idx)

    # Back-to-front scan: from search_hi - 1 down to search_lo.
    for i in range(search_hi - 1, search_lo - 1, -1):
        if d1[i] > threshold_nm_per_nm:
            return RuptureSearchResult(
                rupture_idx       = i,
                rupture_piezo_nm  = float(piezo[i]),
                mask_anchor_idx   = mask_anchor_idx,
                mask_postsnap_idx = mask_postsnap_idx,
            )

    return RuptureSearchResult(-1, float('nan'), mask_anchor_idx, mask_postsnap_idx)


@dataclass
class OnsetSearchResult:
    """
    Outcome of the onset search on mean_dev.

    onset_idx      : index into the forward mean_dev / piezo arrays.  -1 if
                     no qualifying crossing was found.
    onset_piezo_nm : piezo value at that index (nan if not found).
    """
    onset_idx:      int
    onset_piezo_nm: float


def find_onset(
    mean_dev:              np.ndarray,
    piezo:                 np.ndarray,
    rupture_idx:           int,
    post_snapoff_mask_idx: int,
    threshold_nm:          float,
) -> OnsetSearchResult:
    """
    Locate the molecular-event onset by walking mean_dev back-to-front
    (leftward in forward index) starting from rupture_idx.

    Two-state scan:
      1. Walk leftward from rupture_idx.  While mean_dev[i] is still above
         threshold_nm, we are in the post-rupture / near-baseline region —
         keep walking.
      2. Once mean_dev[i] drops BELOW threshold_nm, state flips to "inside
         dip".
      3. The first index where mean_dev[i] rises back to >= threshold_nm is
         the onset.

    Threshold is signed (typical default -0.2 nm — deflection is pulled
    toward the surface during a WLC event).  The window size used upstream
    provides the hysteresis that keeps us from false-triggering on noise.

    Stops at post_snapoff_mask_idx (inclusive lower bound of the search).
    Returns onset_idx = -1 if no qualifying crossing is found.
    """
    n = len(mean_dev)
    if rupture_idx < 0 or rupture_idx >= n or n != len(piezo):
        return OnsetSearchResult(-1, float('nan'))

    lo = max(0, post_snapoff_mask_idx)
    state_below = False

    for i in range(rupture_idx, lo - 1, -1):
        v = float(mean_dev[i])
        if not state_below:
            if v < threshold_nm:
                state_below = True
        else:
            if v >= threshold_nm:
                return OnsetSearchResult(
                    onset_idx      = i,
                    onset_piezo_nm = float(piezo[i]),
                )

    return OnsetSearchResult(-1, float('nan'))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_odd(n: int) -> int:
    """Return n if odd, n+1 if even.  Minimum 5."""
    n = max(5, int(n))
    return n if n % 2 == 1 else n + 1


def _bounded_odd(n: int, size: int, *, minimum: int = 3) -> int:
    """Return an odd window supported by a signal of ``size`` samples."""
    if size < minimum:
        raise ValueError(f"signal must contain at least {minimum} samples")
    requested = max(minimum, int(n))
    if requested % 2 == 0:
        requested += 1
    largest = size if size % 2 == 1 else size - 1
    return min(requested, largest)
