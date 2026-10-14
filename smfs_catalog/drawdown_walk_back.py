"""
drawdown_walk_back.py — one algorithm for locating a rupture on its loading
ramp: maximum drawdown, then a walk back from the bottom of the fall.

It answers with the last raw sample that is still at ramp height before the
force falls:

    1. Search region: from `fit_widths` d1-peak widths left of the left-hand
       d1 crossing (never before the segment start) to the right-hand crossing.
    2. Fit region: the same left end, up to one sample before the left-hand
       crossing.
    3. In the search region find the largest fall: the top (the highest
       sample) and the bottom (the lowest sample after it).
    4. In the fit region fit a least-squares line; the noise is 1.4826 x the
       median of |force - line|.
    5. Stop level = top force - noise_multiple x noise.
    6. From the bottom, step back one sample at a time.  The first sample at
       or above the stop level is the answer.  The walk never passes the top.

Maximum drawdown is the standard quantity of that name (Magdon-Ismail et al.,
2004) and the noise is the usual robust standard deviation.  The walk back,
and the combination, are defined here.

`force` is the RAW force, as the instrument saved it.  The three marks —
segment start, left-hand crossing, right-hand crossing — come from the d1
search (Segment.left_idx, Rupture.rise_idx, Rupture.fall_idx).

A segment shorter than `fit_widths` widths is run on the samples it has.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .regression import linear_fit

# What an answer says about itself.  A status is these joined by "; ", or
# FOUND when none applies.
#   TOP_ON_EDGE   the highest sample is the first one searched, so the force
#                 may be higher still just outside: the top is not a real top.
#   AT_TOP        the answer is the top itself: no later sample was at ramp
#                 height.  Sound when the ramp rises right up to the fall.
#   DID_NOT_MOVE  the bottom itself is at ramp height: the fall is smaller
#                 than the noise margin.
#   FIT_CUT_SHORT the segment is shorter than the fit region asked for.
#   NO_NOISE      nothing to measure the noise on, so no answer.
FOUND         = "found"
TOP_ON_EDGE   = "top on window edge"
AT_TOP        = "stopped at the top"
DID_NOT_MOVE  = "walk did not move"
FIT_CUT_SHORT = "fit region cut short"
NO_NOISE      = "no result: too few samples to measure the noise on"


@dataclass(frozen=True)
class WalkBack:
    """
    One answer and everything used to reach it.

    idx is an ABSOLUTE index into the force array, or None when no noise
    could be measured.  top_idx/bottom_idx bracket the largest fall; noise_pN
    and stop_level_pN are the two values the walk was decided on; fit_samples
    is how many ramp samples the noise was measured on.
    """
    idx:           Optional[int]
    status:        str
    top_idx:       int
    bottom_idx:    int
    noise_pN:      Optional[float]
    stop_level_pN: Optional[float]
    fit_samples:   int


def largest_fall(force: np.ndarray, lo: int, hi: int) -> tuple[int, int]:
    """
    Maximum drawdown of force[lo..hi]: the pair (top, bottom), top <= bottom,
    with the largest force[top] - force[bottom].  Absolute indices.  An exact
    tie returns the earliest.
    """
    seg = np.asarray(force[lo:hi + 1], dtype=float)
    depth = np.maximum.accumulate(seg) - seg
    bottom = int(np.argmax(depth))
    top = int(np.argmax(seg[:bottom + 1]))
    return lo + top, lo + bottom


def drawdown_walk_back(
    force:          np.ndarray,
    segment_start:  int,
    left_crossing:  int,
    right_crossing: int,
    fit_widths:     int,
    noise_multiple: float,
) -> WalkBack:
    """This algorithm's answer for one loading ramp — see the module docstring."""
    n = len(force)
    left_crossing = max(0, min(n - 1, int(left_crossing)))
    right_crossing = max(left_crossing, min(n - 1, int(right_crossing)))
    width = right_crossing - left_crossing + 1
    wanted = int(fit_widths) * width
    first = max(0, int(segment_start), left_crossing - wanted)
    first = min(first, left_crossing)

    top, bottom = largest_fall(force, first, right_crossing)
    fit_samples = left_crossing - first
    t = np.arange(first, left_crossing, dtype=float)
    y = np.asarray(force[first:left_crossing], dtype=float)
    line = linear_fit(t, y)
    if line is None:
        return WalkBack(None, NO_NOISE, top, bottom, None, None, fit_samples)
    resid = y - line.predict(t)
    noise = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))

    level = float(force[top]) - float(noise_multiple) * noise

    stop = bottom
    while stop > top and force[stop] < level:
        stop -= 1

    status = []
    if top == first:
        status.append(TOP_ON_EDGE)
    if stop == top:
        status.append(AT_TOP)
    if stop == bottom:
        status.append(DID_NOT_MOVE)
    if fit_samples < wanted:
        status.append(FIT_CUT_SHORT)
    return WalkBack(stop, "; ".join(status) or FOUND, top, bottom,
                    noise, level, fit_samples)
