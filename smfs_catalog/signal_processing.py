# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/signal_processing.py
#
# Numerical preprocessing and contact detection for SMFS force curves.
# No application-layer or pysmfs imports. Algorithms adapted from pysmfs.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import bessel, sosfiltfilt

# ── Tuning knobs ──────────────────────────────────────────────────────────────
# _DEFAULT_CUTOFF_HZ : spectral split frequency (Hz).  Boundary between the
#   structural/thermodynamic channel (low) and thermal/mechanical channel (high).
#   Should be below cantilever resonance and above rupture event bandwidth.
#   Typical range: 500–5000 Hz.  Overridden at runtime by the DB setting
#   'spectral_cutoff_hz' when called from the Qt viewer.
_DEFAULT_CUTOFF_HZ = 1000.0   # Hz

# _TRIM_PTS : number of samples to exclude at the approach/retract turnaround
#   when searching for the variance threshold crossing.  The piezo reversal
#   creates a brief mechanical transient that produces a spurious variance spike;
#   trimming these points prevents false contact detection.
#   Hard-coded for now; exposed as a named constant so it can be made
#   user-adjustable later without hunting through the code.
_TRIM_PTS = 100

# _DEFAULT_VAR_WINDOW_MS : moving-variance window used by both contact
#   (begin) and snap-off (end) detectors.  Sets the time scale over which
#   variance is averaged before the threshold comparison.
_DEFAULT_VAR_WINDOW_MS = 1.0    # ms

# _DEFAULT_VAR_THRESHOLD : variance threshold (nm²) used by both contact and
#   snap-off detectors.  The first sample whose moving variance exceeds this
#   value is taken as the event.  Kept identical on both sides so begin/end
#   detectors cannot silently drift apart.
_DEFAULT_VAR_THRESHOLD = 0.05   # nm²

# _BESSEL_ORDER : order of the zero-phase Bessel low-pass used by
#   bessel_decompose and (by default) decompose_curve.  Kept as a single
#   constant so the filter response is identical wherever decomposition runs.
_BESSEL_ORDER = 4


# ── Acquisition/application filter relationship ──────────────────────────────

def filter_bandwidth_conflict(
    cutoff_hz: float | None,
    acq_bw_hz: float | None,
) -> bool:
    """
    Return whether the application cutoff is at or above acquisition bandwidth.

    ``None``, non-finite values, and non-positive values mean "unknown" and
    return False. UI wording belongs to the presentation layer.
    """
    if cutoff_hz is None or acq_bw_hz is None:
        return False
    if not np.isfinite(cutoff_hz) or not np.isfinite(acq_bw_hz):
        return False
    return cutoff_hz > 0 and acq_bw_hz > 0 and cutoff_hz >= acq_bw_hz


def _as_finite_1d(values: np.ndarray, name: str, *, min_size: int = 1) -> np.ndarray:
    """Return ``values`` as a finite float vector, or raise a clear ValueError."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size < min_size:
        raise ValueError(f"{name} must contain at least {min_size} samples")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Return slope, intercept, R², and residual RMS for paired vectors."""
    if np.ptp(x) == 0:
        raise ValueError("linear-fit x values must span a non-zero range")
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rms = float(np.sqrt(np.mean(resid ** 2)))
    return float(slope), float(intercept), r2, rms


# ── 1a. fit_retract_baseline ──────────────────────────────────────────────────

@dataclass
class BaselineFit:
    offset:     float   # nm; constant subtracted from the whole retract trace
                         # — mean of the baseline characterization region
    slope:      float   # nm/nm; characterization-region slope (drift/roll-off
                         # diagnostic only — does not affect `offset`)
    intercept:  float   # nm
    r2:         float   # coefficient of determination of the diagnostic fit —
                         # NOT a goodness-of-fit measure for this window: the
                         # characterization region should be flat, so a good
                         # (low-noise, undriven) baseline has little variance
                         # for a line to explain and r2 reads near 0 by
                         # construction. Use `rms` to judge fit quality.
    rms:        float   # nm; residual RMS of the diagnostic fit — the actual
                         # goodness-of-fit number (see r2 above)
    fit_lo_idx: int     # inclusive — start of characterization region
    fit_hi_idx: int     # exclusive — end of region (== len(defl_retr))


def fit_retract_baseline(
    curve,
    anchor_nm: float = 150.0,
) -> BaselineFit:
    """
    Constant-offset baseline correction for the retract deflection.

    The final ``anchor_nm`` of retract piezo travel is a representative,
    statistically useful subset of the far-from-surface baseline; it is not
    intended to identify every baseline sample. The mean deflection of this
    characterization region (``offset``) is subtracted from the full retract.

    Alongside that, a linear fit through the same region (slope,
    intercept, R²) is a pure diagnostic — large slope or poor R² indicates
    roll-off or residual drift in what's supposed to be a flat, contact-free
    region. It does not change `offset` or anything derived from it.

    Parameters
    ----------
    curve     : ForceCurve from curve_loader.py
    anchor_nm : width of the far-retract baseline characterization region in
                nm of piezo travel (default 150 nm). Clamped to the full
                retract if longer. Its inner boundary also starts ROI search.

    Returns
    -------
    BaselineFit with offset, slope, intercept, R², and the [lo, hi) index range
    of the characterization region (hi is always len(curve.defl_retr)).
    """
    defl = _as_finite_1d(curve.defl_retr, "curve.defl_retr", min_size=2)
    piezo = _as_finite_1d(curve.piezo_retr, "curve.piezo_retr", min_size=2)
    n     = len(defl)

    if len(piezo) != n:
        raise ValueError("curve.defl_retr and curve.piezo_retr must have equal lengths")
    if not np.isfinite(anchor_nm) or anchor_nm <= 0:
        raise ValueError(f"anchor_nm must be positive and finite, got {anchor_nm}")

    total_range = abs(float(piezo[-1]) - float(piezo[0]))
    frac     = min(anchor_nm / total_range, 1.0) if total_range > 0 else 0.15
    n_anchor = min(n, max(10, int(round(frac * n))))

    lo, hi = n - n_anchor, n
    anchor_defl  = defl[lo:hi]
    anchor_piezo = piezo[lo:hi]

    offset = float(np.mean(anchor_defl))

    slope, intercept, r2, rms = _linear_fit(anchor_piezo, anchor_defl)

    return BaselineFit(
        offset     = offset,
        slope      = float(slope),
        intercept  = float(intercept),
        r2         = r2,
        rms        = rms,
        fit_lo_idx = lo,
        fit_hi_idx = hi,
    )


# ── 1a-bis. fit_approach_invols ──────────────────────────────────────────────

@dataclass
class InvOLSFit:
    slope:       float   # nm deflection per nm piezo (≈1.0 if calibration is good)
    intercept:   float   # nm
    r2:          float   # coefficient of determination
    rms:         float   # residual RMS (nm)
    fit_lo_idx:  int     # inclusive — start of fit window in approach arrays
    fit_hi_idx:  int     # exclusive — end of fit window in approach arrays


def fit_approach_invols(
    low_appr:   np.ndarray,
    piezo_appr: np.ndarray,
    offset_pts: int,
    window_pts: int,
) -> InvOLSFit:
    """
    Fit a line through the deep-contact region of the approach, anchored by
    distance from the turnaround.

    The turnaround is the LAST sample of the approach array — deepest
    compression.  We skip `offset_pts` samples before the turnaround (piezo
    ringing, possible sample deformation at peak load), then fit the next
    `window_pts` samples back toward shallower contact.

    Parameters
    ----------
    low_appr   : low-frequency approach deflection (nm) — decomposed signal
    piezo_appr : approach piezo position (nm)
    offset_pts : points skipped between turnaround and the end of the fit window
    window_pts : number of points in the fit

    Returns
    -------
    InvOLSFit with slope, intercept, R², RMS, and the [lo, hi) index range.
    On failure (degenerate window), returns NaN stats with lo == hi.
    """
    low_appr = _as_finite_1d(low_appr, "low_appr", min_size=2)
    piezo_appr = _as_finite_1d(piezo_appr, "piezo_appr", min_size=2)
    n = int(len(low_appr))
    if n != int(len(piezo_appr)):
        raise ValueError("low_appr and piezo_appr must have equal lengths")
    if (isinstance(offset_pts, (bool, np.bool_)) or not np.isfinite(offset_pts)
            or int(offset_pts) != offset_pts):
        raise ValueError(f"offset_pts must be an integer, got {offset_pts}")
    if (isinstance(window_pts, (bool, np.bool_)) or not np.isfinite(window_pts)
            or int(window_pts) != window_pts):
        raise ValueError(f"window_pts must be an integer, got {window_pts}")
    offset_pts = int(offset_pts)
    window_pts = int(window_pts)
    if offset_pts < 0:
        raise ValueError(f"offset_pts must be non-negative, got {offset_pts}")
    if window_pts < 2:
        raise ValueError(f"window_pts must be at least 2, got {window_pts}")

    hi = n - offset_pts
    lo = hi - window_pts
    if lo < 0 or hi <= lo or hi > n:
        boundary = min(n, max(0, hi))
        return InvOLSFit(float('nan'), float('nan'), float('nan'), float('nan'),
                         boundary, boundary)

    x = np.asarray(piezo_appr[lo:hi], dtype=float)
    y = np.asarray(low_appr[lo:hi],   dtype=float)

    slope, intercept, r2, rms = _linear_fit(x, y)

    return InvOLSFit(
        slope      = float(slope),
        intercept  = float(intercept),
        r2         = r2,
        rms        = rms,
        fit_lo_idx = lo,
        fit_hi_idx = hi,
    )


# ── 1b. bessel_decompose ─────────────────────────────────────────────────────

def bessel_decompose(
    signal: np.ndarray,
    sample_rate_hz: float,
    cutoff_hz: float,
    order: int = _BESSEL_ORDER,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose a 1-D deflection signal (nm) into low- and high-frequency channels.

    Parameters
    ----------
    signal         : 1-D numpy array (deflection in nm)
    sample_rate_hz : acquisition sample rate in Hz  (must be > 0)
    cutoff_hz      : -3 dB low-pass cutoff in Hz    (must be > 0 and < Nyquist)
    order          : Bessel filter order (default 4)

    Returns
    -------
    (low, high) where low + high == signal to floating-point precision.
    low  = structural/thermodynamic channel (frequencies below cutoff_hz)
    high = thermal/mechanical channel       (exact residual; signal − low)
    """
    signal = _as_finite_1d(signal, "signal")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}")
    if not np.isfinite(cutoff_hz) or cutoff_hz <= 0:
        raise ValueError(f"cutoff_hz must be positive, got {cutoff_hz}")
    if (isinstance(order, (bool, np.bool_)) or not np.isfinite(order)
            or int(order) != order or order <= 0):
        raise ValueError(f"order must be a positive integer, got {order}")
    if cutoff_hz >= sample_rate_hz / 2:
        raise ValueError(
            f"cutoff_hz ({cutoff_hz}) must be below Nyquist ({sample_rate_hz / 2:.1f} Hz)"
        )

    sos = bessel(
        int(order),
        cutoff_hz,
        btype  = 'low',
        fs     = sample_rate_hz,
        output = 'sos',
        norm   = 'phase',
    )
    try:
        low = sosfiltfilt(sos, signal)
    except ValueError as exc:
        raise ValueError(
            f"signal cannot be zero-phase order-{int(order)} filtered "
            f"at its current length ({signal.size} samples)"
        ) from exc
    high = signal - low   # residual; low + high reconstructs to floating-point precision
    return low, high


# ── 1b. DecomposedCurve dataclass ─────────────────────────────────────────────

@dataclass
class DecomposedCurve:
    low_appr:       np.ndarray
    high_appr:      np.ndarray
    low_retr:       np.ndarray
    high_retr:      np.ndarray
    sample_rate_hz: float


# ── 1c. decompose_curve ───────────────────────────────────────────────────────

def decompose_curve(
    fc,
    cutoff_hz: float = _DEFAULT_CUTOFF_HZ,
    order: int = _BESSEL_ORDER,
) -> DecomposedCurve:
    """
    Spectrally decompose both approach and retract deflection signals of a ForceCurve.

    Parameters
    ----------
    fc        : ForceCurve from curve_loader.py
    cutoff_hz : low-pass cutoff in Hz (defaults to _DEFAULT_CUTOFF_HZ)
    order     : Bessel filter order

    Returns
    -------
    DecomposedCurve with low/high channels for approach and retract.
    """
    low_appr, high_appr = bessel_decompose(
        fc.defl_appr, fc.sample_rate_hz, cutoff_hz, order
    )
    low_retr, high_retr = bessel_decompose(
        fc.defl_retr, fc.sample_rate_hz, cutoff_hz, order
    )
    return DecomposedCurve(
        low_appr       = low_appr,
        high_appr      = high_appr,
        low_retr       = low_retr,
        high_retr      = high_retr,
        sample_rate_hz = fc.sample_rate_hz,
    )


# ── 1d. _moving_variance ─────────────────────────────────────────────────────

def _moving_variance(signal: np.ndarray, window_pts: int) -> np.ndarray:
    """Moving variance via E[x²] − E[x]² using uniform_filter1d."""
    signal = _as_finite_1d(signal, "signal")
    if window_pts < 1:
        raise ValueError(f"window_pts must be positive, got {window_pts}")
    mean_x  = uniform_filter1d(signal,      window_pts, mode='nearest')
    mean_x2 = uniform_filter1d(signal ** 2, window_pts, mode='nearest')
    return mean_x2 - mean_x ** 2


# ── 1e. _ms_to_pts ────────────────────────────────────────────────────────────

def _ms_to_pts(ms: float, sample_rate_hz: float, min_pts: int = 5) -> int:
    """Convert a duration in ms to an odd number of samples (>= min_pts)."""
    if not np.isfinite(ms) or ms <= 0:
        raise ValueError(f"ms must be positive and finite, got {ms}")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be positive and finite, got {sample_rate_hz}")
    if min_pts < 1:
        raise ValueError(f"min_pts must be positive, got {min_pts}")
    pts = int(round(ms * 1e-3 * sample_rate_hz))
    pts = max(pts, min_pts)
    if pts % 2 == 0:
        pts += 1
    return pts


# ── 1f. find_begin_in_contact ─────────────────────────────────────────────────

def find_begin_in_contact(
    dc: DecomposedCurve,
    var_window_ms: float = _DEFAULT_VAR_WINDOW_MS,
    threshold:     float = _DEFAULT_VAR_THRESHOLD,
    trim_pts:      int   = _TRIM_PTS,
) -> tuple[int, float]:
    """
    Return (index, threshold) for the contact onset on approach.

    index     : index into the original dc.low_appr / dc.high_appr arrays where
                the tip first touches the surface.  Returns 0 on fallback.
    threshold : the variance threshold (nm²) passed in directly — the first
                point where the moving variance exceeds this value is the
                contact onset.  Useful for plotting.

    Algorithm
    ---------
    The approach array is reversed so index 0 = in-contact end (high variance
    is suppressed by surface contact).  The variance profile steps UP when the
    cantilever leaves the surface.  Scanning left-to-right (from in-contact end)
    we find the first point where variance exceeds the threshold.
    The index is then mapped back to the original (unreversed) array.
    """
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError(f"threshold must be non-negative and finite, got {threshold}")
    if trim_pts < 0:
        raise ValueError(f"trim_pts must be non-negative, got {trim_pts}")
    high_appr = _as_finite_1d(dc.high_appr, "dc.high_appr")
    high_appr_rev = high_appr[::-1]   # index 0 = in-contact end
    n = len(high_appr_rev)

    window_pts = _ms_to_pts(var_window_ms, dc.sample_rate_hz)
    var = _moving_variance(high_appr_rev, window_pts)

    # Skip the turnaround region before searching.
    search_start = trim_pts

    contact_end_rev = 0   # fallback: report contact at the very end of approach
    for i in range(search_start, n):
        if var[i] > threshold:
            contact_end_rev = i
            break

    return (n - 1) - contact_end_rev, float(threshold)


# ── 1g. find_end_in_contact ───────────────────────────────────────────────────

def find_end_in_contact(
    dc: DecomposedCurve,
    var_window_ms: float = _DEFAULT_VAR_WINDOW_MS,
    threshold:     float = _DEFAULT_VAR_THRESHOLD,
    trim_pts:      int   = _TRIM_PTS,
) -> tuple[int, float]:
    """
    Return (index, threshold) for snap-off on retract.

    index     : index into dc.low_retr / dc.high_retr where contact ends.
                Returns len(dc.high_retr) - 1 on fallback.
    threshold : the variance threshold (nm²) passed in directly — the first
                point where the moving variance exceeds this value is snap-off.
                Useful for plotting.

    Algorithm
    ---------
    The retract array index 0 is already in-contact; variance is low at the
    start and steps UP at snap-off.  Scanning left-to-right we find the first
    point where variance exceeds the threshold.
    """
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError(f"threshold must be non-negative and finite, got {threshold}")
    if trim_pts < 0:
        raise ValueError(f"trim_pts must be non-negative, got {trim_pts}")
    high_retr = _as_finite_1d(dc.high_retr, "dc.high_retr")
    n = len(high_retr)

    window_pts = _ms_to_pts(var_window_ms, dc.sample_rate_hz)
    var = _moving_variance(high_retr, window_pts)

    # Skip the turnaround region (first trim_pts of retract = piezo reversal artifact).
    scan_start = trim_pts

    for i in range(scan_start, n):
        if var[i] > threshold:
            return i, float(threshold)

    return n - 1, float(threshold)   # fallback
