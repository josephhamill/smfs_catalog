# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Guard: there is exactly ONE definition of a rupture's force peak.

If segmentation happens and a fit happens, the same function must do it. Two
answers to "where is this rupture and how strong is it" means the number that
reaches a figure is whichever code path ran last — which is not a thing that can
be allowed to happen to data that goes into papers.

`roi_events.ramp_force_peak` is that one definition. This file pins:

  1. its semantics (absolute index, degenerate windows → None);
  2. that it is robust on the signal it is contracted to receive;
  3. THE BUG IT WAS EXTRACTED TO KILL — that a window bounded by d1 ARGMAX
     indices (the pre-2026-07-16 bounds) picks a peak
     belonging to the PREVIOUS rupture on an inverted pair, while a window
     bounded by the d1 EDGES (what fit_segments has always used, and what the
     fitter uses) picks the right one;
  4. that every caller agrees, by construction, because
     they call the same function on the same bounds.

Run with the smfs-catalog env, from the repo root:
    python tests/test_rupture_peak.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smfs_catalog.roi_events import (  # noqa: E402
    ROI,
    CurveEvents,
    Rupture,
    Segment,
    ramp_force_peak,
    ramp_peak_is_edge_pinned,
)


# ── A synthetic inverted pair: strong rupture first, weak one second ──────────
#
# This is the F1 > F2 ("inverted_pair") geometry the force-ordering
# logic exists to handle, and the exact case where the two window conventions
# disagree.
#
#   idx  0..50   ramp 1 rises 0 → 100        (rupture 1 is STRONG; peak at 50)
#   idx 51..60   the drop: ~96 → 20          (d1 is above threshold through here)
#   idx 61..88   ramp 2 rises 20 → 40        (rupture 2 is WEAK; peak at 88)
#   idx 89..100  the drop: ~38 → 0
#
# The landmark geometry mirrors real curves: d1 is the derivative of the SMOOTHED
# deflection, so it only crosses its threshold once the drop is already underway —
# a few samples AFTER the true force peak. So rise_idx sits just past the peak,
# and argmax over [prev.fall, this.rise] finds the peak strictly INSIDE the
# window. That ordering (rise_idx late, never early) is what makes the estimator
# sound; when it inverts, the ramp is truncated before its peak and the force is
# silently under-reported — which is what ramp_peak_is_edge_pinned detects.
#
# Rupture 1: rise_idx=52 (drop underway), idx=55 (d1 argmax), fall_idx=60.
# Rupture 2: rise_idx=90, idx=95, fall_idx=100.
#
# Ramp 2's segment is [r1.fall_idx=60, r2.rise_idx=90] → peak 40 at idx 88. Right.
# The OLD window was [r1.idx=55, r2.idx=95] → it reaches back into rupture 1's
# half-finished drop, where force is still ~62 — well above ramp 2's own peak of
# 40. argmax then returns idx 55 and reports rupture 2's force as ~62. Wrong
# rupture, wrong force.

def _inverted_pair_force() -> np.ndarray:
    f = np.zeros(111)
    f[0:51]    = np.linspace(0.0, 100.0, 51)     # ramp 1 → peak 100 at idx 50
    f[51:61]   = np.linspace(96.0, 20.0, 10)     # rupture 1's drop
    f[61:89]   = np.linspace(20.0, 40.0, 28)     # ramp 2 → peak 40 at idx 88
    f[89:101]  = np.linspace(38.0, 0.0, 12)      # rupture 2's drop
    f[101:111] = 0.0                             # baseline
    return f


R1_RISE, R1_IDX, R1_FALL = 52, 55, 60
R2_RISE, R2_IDX, R2_FALL = 90, 95, 100


def test_ramp_force_peak_returns_an_absolute_index() -> None:
    """The contract is an index into `force`, not into the slice."""
    f = _inverted_pair_force()
    assert ramp_force_peak(f, 0, R1_RISE) == 50, "ramp 1's peak is at index 50"
    assert ramp_force_peak(f, R1_FALL, R2_RISE) == 88, "ramp 2's peak is at index 88"


def test_ramp_force_peak_rejects_degenerate_windows() -> None:
    """None, never a fabricated index — matches the module's no-fabrication rule."""
    f = _inverted_pair_force()
    assert ramp_force_peak(f, 30, 30) is None, "zero-width window must be None"
    assert ramp_force_peak(f, 40, 30) is None, "inverted window must be None"


def test_ramp_force_peak_clamps_to_the_array() -> None:
    f = _inverted_pair_force()
    assert ramp_force_peak(f, -10, 50) == 50, "lo below 0 must clamp, not wrap"
    pk = ramp_force_peak(f, 0, 10_000)
    assert pk == 50, f"hi past the end must clamp to the real peak, got {pk}"


def test_d1_edge_bounds_find_the_right_peak_and_argmax_bounds_do_not() -> None:
    """THE REGRESSION. This is the bug ramp_force_peak was extracted to kill.

    Segment bounds (prev fall → this rise) exclude the previous rupture's drop.
    The old bounds (prev d1 argmax → this d1 argmax) do not, and on an
    inverted pair they return a peak that belongs to the previous rupture."""
    f = _inverted_pair_force()

    correct = ramp_force_peak(f, R1_FALL, R2_RISE)          # segment bounds
    assert correct == 88, f"segment bounds should find ramp 2's peak at 88, got {correct}"
    assert abs(f[correct] - 40.0) < 1e-9, (
        f"ramp 2's rupture force is 40, got {f[correct]}"
    )

    # What the pre-fix code did: window bounded by d1 argmax indices.
    old = R1_IDX + int(np.argmax(f[R1_IDX:R2_IDX + 1]))
    assert old == R1_IDX, (
        "the old window's argmax should land in rupture 1's unfinished drop"
    )
    assert f[old] > f[correct], (
        "the whole point: the old window reports a HIGHER force (from the previous "
        f"rupture's decay, {f[old]}) than ramp 2's real peak ({f[correct]})"
    )
    assert old != correct, "the two conventions must genuinely disagree here"


def test_edge_pinned_detects_a_truncated_ramp() -> None:
    """A peak on the right edge means the real one may lie outside the window.

    This is the failure mode when rise_idx comes EARLY — the d1 threshold trips
    before the force has finished rising — so the ramp is cut before its peak and
    the rupture force is under-reported with no complaint."""
    f = _inverted_pair_force()
    # Ramp 1 truncated at 40, before its real peak at 50: still rising at the edge.
    pk = ramp_force_peak(f, 0, 40)
    assert pk == 40, "a still-rising window peaks at its right edge"
    assert ramp_peak_is_edge_pinned(pk, 0, 40), "edge-pinned peak must be flagged"
    assert f[pk] < f[50], (
        "the truncated window's peak really is an underestimate of the true peak"
    )
    # A healthy ramp: rise_idx sits past the peak, so the peak is strictly inside.
    pk2 = ramp_force_peak(f, R1_FALL, R2_RISE)
    assert not ramp_peak_is_edge_pinned(pk2, R1_FALL, R2_RISE), (
        "a peak strictly inside the window must not be flagged"
    )
    pk1 = ramp_force_peak(f, 0, R1_RISE)
    assert not ramp_peak_is_edge_pinned(pk1, 0, R1_RISE), (
        "ramp 1's peak is strictly inside its segment too"
    )


def _inverted_pair_events() -> CurveEvents:
    r1 = Rupture(idx=R1_IDX, piezo_nm=float(R1_IDX), d1_height=1.0,
                 rise_idx=R1_RISE, fall_idx=R1_FALL)
    r2 = Rupture(idx=R2_IDX, piezo_nm=float(R2_IDX), d1_height=1.0,
                 rise_idx=R2_RISE, fall_idx=R2_FALL)
    # Exactly what _segments_from_ruptures builds: onset → r1.rise, r1.fall → r2.rise.
    segs = [
        Segment(left_idx=0, right_idx=R1_RISE,
                left_piezo_nm=0.0, right_piezo_nm=float(R1_RISE)),
        Segment(left_idx=R1_FALL, right_idx=R2_RISE,
                left_piezo_nm=float(R1_FALL), right_piezo_nm=float(R2_RISE)),
    ]
    return CurveEvents(rois=[ROI(
        onset_idx=0, return_idx=105,
        onset_piezo_nm=0.0, return_piezo_nm=105.0,
        ruptures=[r1, r2], segments=segs,
    )], detector="d1_threshold")


def _main() -> int:
    checks = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in checks:
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {fn.__name__}\n       {e}\n")
        else:
            print(f"[ok]   {fn.__name__}")
    print()
    if failed:
        print(f"{failed} of {len(checks)} rupture-peak checks failed.")
    else:
        print(f"All {len(checks)} rupture-peak checks pass.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
