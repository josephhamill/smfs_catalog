# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/roi_events.py
#
# Multi-event ROI model: events within events.
#
# The existing pipeline (roi_detection.find_rupture / find_onset) locates ONE
# outermost onset→rupture excursion per curve.  This module generalises that to
# the full hierarchy a tethered-bond pull actually produces:
#
#     Curve ─┬─ ROI 0 ─┬─ Rupture r1 ┐
#            │         ├─ Rupture r2 ┼─ Segments (onset→r1, r1→r2, …)
#            │         └─ …          ┘   each segment = one valid WLC domain
#            ├─ ROI 1 …
#            └─ …
#
# Division of labour (unchanged from the discussion):
#   • d1        → detect ruptures (the sharp jumps).  Two interchangeable
#                 detectors are provided so they can be A/B'd on real curves.
#   • mean_dev  → draw the baseline fences that bound each ROI (a LEVEL signal;
#                 answers "are we back at baseline?", which d1 cannot).
#
# This module is PURE: no Qt, no DB, no file I/O.  It produces GEOMETRY only —
# rupture/segment indices and piezo landmarks.  Per-segment forces and WLC fits
# are left as None here and filled by a later force/fit pass (see fit_segments,
# stubbed at the bottom) so this stays trivially testable.
#
# Orientation follows roi_detection: all arrays are forward (index 0 = in-contact
# end, last index = far baseline).  Within one ROI a pull loads at LOWER index
# (onset, near surface), ruptures at HIGHER index, and returns to baseline just
# beyond.  So "walk left from a rupture to its onset" = walk toward index 0.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import find_peaks

from .models import fit_model, wlc
from .roi_detection import find_onset


# ── Leaf: a single rupture ────────────────────────────────────────────────────

@dataclass
class Rupture:
    """
    One rupture (a d1 peak) inside an ROI.

    idx        : index into the forward d1/piezo/mean_dev arrays.
    piezo_nm   : piezo value at idx.
    d1_height  : signed d1 value at the peak (the detector's response).
    prominence : topographic prominence of the peak (NaN when it came from the
                 bare-threshold detector, which does not compute one).
    force_pN   : rupture force — None until the force pass fills it.
    """
    idx:        int
    piezo_nm:   float
    d1_height:  float
    prominence: float = float("nan")
    force_pN:   Optional[float] = None
    # Index of the force PEAK (the physical rupture point, a few samples before
    # the d1-drop index above).  Filled by fit_segments; the F-x marker sits here.
    force_idx:  Optional[int] = None
    # Extension (nm) at force_idx, in the SAME shared coordinate fit_segments
    # fits WLC against — (piezo_retr - snapoff) - defl_corr.  NOT piezo_nm above
    # (that's raw piezo displacement, a different number).  None until
    # fit_segments runs.  This is the (x, y) = (extension_nm, force_pN) point a
    # rupture actually IS, and the basis for isoforce_dX_pairs below.
    extension_nm: Optional[float] = None
    # d1-peak EDGES: the run of samples where d1 stays above the detection
    # threshold.  rise_idx = first sample above (≈ the force peak, where force
    # STARTS dropping); fall_idx = last sample above (bottom of the drop).  These
    # define the loading-ramp windows: a ramp runs from the previous rupture's
    # fall_idx to this rupture's rise_idx.  Both default to idx for detectors that
    # don't compute a run.
    rise_idx:   Optional[int] = None
    fall_idx:   Optional[int] = None


# ── The domain of one WLC fit ─────────────────────────────────────────────────

@dataclass
class Segment:
    """
    The stretch of curve BETWEEN two consecutive landmarks within an ROI:
    onset→r1, r1→r2, …  Each segment is the only region over which a single WLC
    fit is physically valid, so each carries its own (l_p, l_c).

    A double-chain→single-chain transition shows up as adjacent segments fitting
    to different WLC parameters — orthogonal to the force-ordering label on the
    parent ROI.

    Fit fields are None until fit_segments() runs.
    """
    left_idx:       int
    right_idx:      int
    left_piezo_nm:  float
    right_piezo_nm: float
    l_p_nm:   Optional[float] = None
    l_c_nm:   Optional[float] = None
    l_p_err:  Optional[float] = None
    l_c_err:  Optional[float] = None
    n_pts:    int = 0
    # Actual fitted window (reload/onset bottom → force peak) — a sub-range of
    # [left_idx, right_idx].  Used to draw the WLC fit over exactly the points fit.
    fit_lo_idx: Optional[int] = None
    fit_hi_idx: Optional[int] = None
    # Extension (nm) on THIS segment's own rising ramp where force first climbs
    # back through the PREVIOUS rupture's force (the lab's "isoforce" distance —
    # e.g. bond ruptures at 100 pN/1500 nm, tether segment recrosses 100 pN at
    # 2000 nm here).  Fit-independent — a direct read of force/extension, not
    # derived from l_p/l_c.  None for a segment with no previous rupture (the
    # first segment in its ROI) or where the previous rupture's force is never
    # (cleanly) reached.  Must be stored, not recomputed on load like
    # dX_pairs/dF_pairs below — it needs the segment's own force/extension
    # trace, which event_map does not otherwise persist.
    isoforce_x_nm: Optional[float] = None
    # Integrated autocorrelation time of THIS segment's fit residual, in samples
    # l_p_err/l_c_err above already have sqrt(tau) applied; this is
    # stored so the size of an error bar can be explained rather than just
    # asserted.  1.0 means "no correction claimed" (too few points, or a
    # residual with no measurable structure) — see integrated_autocorr_time.
    tau: Optional[float] = None
    # Largest extension in the fitted window, nm.  Stored — not derived from
    # fit_hi_idx — because recovering it later would mean rebuilding this
    # segment's extension trace, which the event_map document does not keep
    # (same reason isoforce_x_nm is stored).  Set even when the FIT fails, so a
    # failed fit can still say how far it got. Feeds z_max below.
    x_max_nm: Optional[float] = None
    # True when the force peak sat on the fit window's right edge — the real
    # peak is outside the window, so the rupture force is an underestimate and
    # the fit stopped early. See ramp_peak_is_edge_pinned, which is the
    # one definition; this field is only where its answer is kept.  Common
    # rather than rare on real cohorts — measure the rate on your own.
    edge_pinned: Optional[bool] = None
    # Compact, user-facing outcome of the fit attempt.  Detailed numerical
    # diagnostics remain in their dedicated fields; this distinguishes a
    # verified missing fit from an unreported/failed calculation.
    fit_status: str = "not_attempted"
    fit_detail: Optional[str] = None

    @property
    def width_pts(self) -> int:
        return abs(self.right_idx - self.left_idx)

    @property
    def z_max(self) -> Optional[float]:
        """
        How close the fit window got to the Marko-Siggia pole: x_max / l_c,
        dimensionless, DERIVED (never stored — it is a ratio of two stored
        numbers, and storing it would be a third thing to keep consistent).

        This is the fit's conditioning number in all but name. Well
        below the pole the WLC curve is nearly straight and only the PRODUCT
        l_p*l_c is determined; everything that separates the two parameters
        comes from the final approach, where the fit turns up.  Analytically
        l_p ~ (1-z)^-2, so small differences in z_max move l_p a long way — on
        real cohorts a gap of a few hundredths in median z_max has tracked a
        difference of more than half again in median l_p, on the same analyte
        with closely-agreeing rupture forces.  Compare z_max before attributing
        an l_p difference to the molecule.

        None when either input is missing or l_c is non-positive.
        """
        if self.x_max_nm is None or self.l_c_nm is None:
            return None
        if not self.l_c_nm > 0.0:
            return None
        return float(self.x_max_nm) / float(self.l_c_nm)


# ── One baseline→excursion→return interval ────────────────────────────────────

@dataclass
class ROI:
    """
    One baseline-bounded excursion: the "outer" unit.  A tethered-bond event is
    an ROI whose ruptures list has length ≥ 2.

    onset_*  : where the excursion departs baseline (lower index / surface side).
    return_* : where it comes back to baseline (higher index / baseline side).
    ruptures : ascending by index (surface → baseline order).
    segments : onset→r1, r1→r2, …, r(n-1)→rn.  Always len(ruptures) segments.
    """
    onset_idx:       int
    return_idx:      int
    onset_piezo_nm:  float
    return_piezo_nm: float
    ruptures:        list[Rupture] = field(default_factory=list)
    segments:        list[Segment] = field(default_factory=list)

    @property
    def n_ruptures(self) -> int:
        return len(self.ruptures)

    @property
    def dX_pairs(self) -> list[float]:
        """Piezo separations between consecutive ruptures (pure geometry) — on
        piezo_nm, the raw stage displacement. See dX_ext_pairs for the same
        idea on extension_nm (the WLC-fit coordinate) instead."""
        return [
            self.ruptures[i + 1].piezo_nm - self.ruptures[i].piezo_nm
            for i in range(len(self.ruptures) - 1)
        ]

    @property
    def dF_pairs(self) -> list[Optional[float]]:
        """
        Force deltas between consecutive ruptures.  None entries where either
        force is not yet filled — never fabricates a delta.
        """
        out: list[Optional[float]] = []
        for i in range(len(self.ruptures) - 1):
            a, b = self.ruptures[i].force_pN, self.ruptures[i + 1].force_pN
            out.append(None if a is None or b is None else b - a)
        return out

    @property
    def dX_ext_pairs(self) -> list[Optional[float]]:
        """
        Plain extension gap between consecutive ruptures' own points — X1 - X2,
        i.e. ruptures[i].extension_nm - ruptures[i+1].extension_nm — on the
        WLC-fit extension_nm coordinate (NOT piezo_nm; see dX_pairs for that
        raw-piezo version). No crossing search, no force-matching: just the two
        already-known (extension_nm, force_pN) points, same shape as dF_pairs
        (a plain subtraction of two already-known scalars). None entries where
        either extension is not yet filled — never fabricates a delta.

        Unlike dF_pairs, this is NOT expected to be a mixed-sign, roughly-
        symmetric-about-zero quantity: extension increases through a pull
        essentially regardless of which rupture is stronger, so this comes out
        one sign almost always (negative, under the X1-X2 convention above).
        That is not a defect to fix — it is simply reporting that "later
        ruptures are further along" is a geometric near-certainty, unlike force
        ordering, which isn't. Deliberately distinct from isoforce_dX_pairs
        below: this makes no claim about equal-force reloading, and unlike
        that quantity, is defined regardless of which rupture is stronger.
        """
        out: list[Optional[float]] = []
        for i in range(len(self.ruptures) - 1):
            a, b = self.ruptures[i].extension_nm, self.ruptures[i + 1].extension_nm
            out.append(None if a is None or b is None else a - b)
        return out

    @property
    def isoforce_dX_pairs(self) -> list[Optional[float]]:
        """
        The lab's "isoforce" distance for ruptures[i] -> ruptures[i+1]: the gap
        between ruptures[i]'s own extension and the extension where segment
        i+1's rising force first climbs back through ruptures[i]'s force — NOT
        the raw ruptures[i+1].piezo_nm - ruptures[i].piezo_nm gap dX_pairs
        gives, and NOT the plain ruptures[i].extension_nm -
        ruptures[i+1].extension_nm gap dX_ext_pairs gives either (both mix two
        different forces into one number; this is the one FORCE-MATCHED
        comparison of the three).

        Deliberately one-directional, unlike dX_ext_pairs/dF_pairs: this
        answers "how much further did the survivor have to stretch to reload
        back up to the force that just broke?" — a question that only makes
        sense measured on the segment that comes AFTER ruptures[i], moving
        forward from it. When ruptures[i+1]'s own force never reaches
        ruptures[i]'s (segment i+1 peaks below it), the honest answer is that
        the reload never got there — None, not a value found by looking
        elsewhere. Searching the preceding segment would answer a causally
        backward question rather than measure reloading after the rupture.
        None where segments[i + 1].isoforce_x_nm wasn't found by fit_segments.
        """
        out: list[Optional[float]] = []
        for i in range(len(self.ruptures) - 1):
            rup, seg_next = self.ruptures[i], self.segments[i + 1]
            if rup.extension_nm is None or seg_next.isoforce_x_nm is None:
                out.append(None)
            else:
                out.append(seg_next.isoforce_x_nm - rup.extension_nm)
        return out

    @property
    def ordering(self) -> str:
        """
        Classify the ROI from its rupture forces (goal C).

            single         — one rupture
            expected_pair  — two ruptures, F1 < F2 (bond then stronger tether)
            inverted_pair  — two ruptures, F1 > F2
            multi          — three or more ruptures
            unknown        — needed forces not yet filled
        """
        n = len(self.ruptures)
        if n == 0:
            return "unknown"
        if n == 1:
            return "single"
        if n >= 3:
            return "multi"
        f1, f2 = self.ruptures[0].force_pN, self.ruptures[1].force_pN
        if f1 is None or f2 is None:
            return "unknown"
        return "expected_pair" if f1 < f2 else "inverted_pair"


# ── Whole-curve container ─────────────────────────────────────────────────────

@dataclass
class CurveEvents:
    """
    Every ROI found on one curve, plus the detector that produced them.

    rois are ordered surface → baseline (ascending onset index).  The
    OUTERMOST ROI's terminal rupture is what today's scalar pipeline reports;
    `primary` exposes it so the existing suite can keep reading one force/file
    while this richer structure lives alongside it.
    """
    rois:     list[ROI] = field(default_factory=list)
    detector: str = ""          # 'd1_threshold' | 'find_peaks'

    @property
    def n_rois(self) -> int:
        return len(self.rois)

    @property
    def primary(self) -> Optional[Rupture]:
        """Terminal (most baseline-ward) rupture of the outermost ROI, or None."""
        for roi in reversed(self.rois):
            if roi.ruptures:
                return roi.ruptures[-1]
        return None


# ── Detectors on d1 (interchangeable — A/B these on real curves) ──────────────

def detect_ruptures_threshold(
    d1:        np.ndarray,
    threshold: float,
    lo:        int,
    hi:        int,
) -> list[Rupture]:
    """
    Bare signed-threshold detector — the generalisation of find_rupture.

    Every maximal contiguous run of d1 > threshold within [lo, hi) collapses to
    ONE rupture at the run's argmax (so a rupture spanning several samples is not
    counted repeatedly).  Returned ascending by index.  prominence is left NaN.
    """
    ruptures: list[Rupture] = []
    lo = max(0, lo)
    hi = min(len(d1), hi)
    i = lo
    while i < hi:
        if d1[i] > threshold:
            j = i
            while j < hi and d1[j] > threshold:
                j += 1
            run = d1[i:j]
            k = i + int(np.argmax(run))
            # rise = i (peak's left edge = force peak), fall = j-1 (drop bottom).
            ruptures.append(Rupture(idx=k, piezo_nm=float("nan"),
                                    d1_height=float(d1[k]),
                                    rise_idx=i, fall_idx=j - 1))
            i = j
        else:
            i += 1
    return ruptures


def detect_ruptures_findpeaks(
    d1:          np.ndarray,
    height:      float,
    prominence:  float,
    distance_pts: int,
    lo:          int,
    hi:          int,
) -> list[Rupture]:
    """
    scipy.signal.find_peaks detector — prominence + min-distance rejects d1 noise
    and collapses tightly-clustered doubles.

    `height` reuses the existing d1 threshold as a floor; `prominence` is the
    real discriminator; `distance_pts` is the minimum peak separation in samples.
    Returned ascending by index, each carrying its measured prominence.
    """
    lo = max(0, lo)
    hi = min(len(d1), hi)
    if hi - lo < 3:
        return []
    seg = d1[lo:hi]
    idxs, props = find_peaks(
        seg,
        height=height,
        prominence=prominence,
        distance=max(1, int(distance_pts)),
    )
    proms = props.get("prominences", np.full(len(idxs), np.nan))
    out: list[Rupture] = []
    for p, pr in zip(idxs, proms):
        g = lo + int(p)
        # d1-peak edges: walk out while still above the height floor.
        l = g
        while l - 1 >= lo and d1[l - 1] > height:
            l -= 1
        r = g
        while r + 1 < hi and d1[r + 1] > height:
            r += 1
        out.append(Rupture(idx=g, piezo_nm=float("nan"),
                           d1_height=float(seg[p]), prominence=float(pr),
                           rise_idx=l, fall_idx=r))
    return out


# ── Outer loop: baseline fences from mean_dev ─────────────────────────────────

def segment_baseline_excursions(
    mean_dev:      np.ndarray,
    onset_thr:     float,
    lo:            int,
    hi:            int,
    min_width_pts: int = 1,
) -> list[tuple[int, int]]:
    """
    Cut [lo, hi) into baseline-bounded excursion intervals using mean_dev as a
    LEVEL signal.  "In excursion" ≡ mean_dev < onset_thr (the pipeline's signed
    convention: deflection is pulled toward the surface while loaded, so an
    excursion is BELOW threshold, matching find_onset's default −0.2 nm).

    Returns [(onset_idx, return_idx), …] ascending, where onset_idx is the low
    (surface-side) boundary and return_idx the high (baseline-side) boundary of
    each contiguous below-threshold run wider than min_width_pts.

    NOTE: single fixed threshold to mirror today's find_onset exactly.  Add
    two-level hysteresis here later if a noisy mean_dev shatters one excursion.
    """
    intervals: list[tuple[int, int]] = []
    lo = max(0, lo)
    hi = min(len(mean_dev), hi)
    i = lo
    while i < hi:
        if mean_dev[i] < onset_thr:
            j = i
            while j < hi and mean_dev[j] < onset_thr:
                j += 1
            if (j - i) >= min_width_pts:
                intervals.append((i, j - 1))
            i = j
        else:
            i += 1
    return intervals


# ── Orchestrator: geometry only (no forces, no fits) ──────────────────────────

@dataclass(frozen=True)
class OuterEventBoundary:
    """Validated coarse event boundary before any inner segmentation.

    The outer-threshold terminal rupture opens the candidate. Walking left on
    the level signal must then find its onset/baseline return; candidates with
    no such return are absent from this list rather than becoming events.
    """
    onset_idx:       int
    terminal_idx:    int
    terminal_run_lo: int
    terminal_run_hi: int


def find_outer_events(
    d1: np.ndarray,
    mean_dev: np.ndarray,
    piezo: np.ndarray,
    *,
    lo: int,
    hi: int,
    onset_thr: float,
    outer_threshold: float,
) -> list[OuterEventBoundary]:
    """Find coarse onset→terminal-rupture events, scanning baseline→surface.

    This is the single implementation of the outer event search used for both
    the curve verdict and the later finer segmentation. It makes no decision
    about how many segments an event contains.
    """
    outer = float(outer_threshold)
    lo = max(0, int(lo))
    hi = min(len(d1), int(hi))
    found: list[OuterEventBoundary] = []

    i = hi - 1
    while i >= lo:
        if d1[i] <= outer:
            i -= 1
            continue

        run_hi = i
        j = i
        while j >= lo and d1[j] > outer:
            j -= 1
        run_lo = j + 1
        terminal_idx = run_lo + int(np.argmax(d1[run_lo:run_hi + 1]))

        onset_idx = find_onset(
            mean_dev, piezo, terminal_idx, lo, onset_thr,
        ).onset_idx
        if onset_idx < 0:
            i = run_lo - 1
            continue

        found.append(OuterEventBoundary(
            onset_idx=onset_idx,
            terminal_idx=terminal_idx,
            terminal_run_lo=run_lo,
            terminal_run_hi=run_hi,
        ))
        i = onset_idx - 1

    found.reverse()
    return found


def build_curve_events(
    d1:              np.ndarray,
    mean_dev:        np.ndarray,
    piezo:           np.ndarray,
    *,
    lo:              int,
    hi:              int,
    onset_thr:       float,
    detector:        str = "d1_threshold",
    outer_threshold: float = 0.2,
    inner_threshold: float | None = None,
    prominence:      float = 0.1,
    distance_pts:    int   = 25,
    outer_events:    list[OuterEventBoundary] | None = None,
) -> CurveEvents:
    """
    Assemble the full ROI/rupture/segment geometry for one curve.

    [lo, hi) is the searchable band (post-snapoff mask → anchor), computed by the
    caller exactly as find_rupture does today.  `detector` selects which d1
    detector runs INSIDE each ROI ('d1_threshold' or 'find_peaks').

    Two-tier thresholds as a RIGHT→LEFT state machine — outer and inner never
    compete because they act in different states:

      HUNTING (outer is king): scan leftward ignoring everything until the first
        d1 crossing above outer_threshold.  That crossing is a junction's
        TERMINAL rupture.  Inner crossings before it (further baseline-ward) are
        meaningless and skipped.

      IN-JUNCTION (inner is king): the junction spans from that terminal LEFT to
        its onset — the mean_dev return-to-baseline (≥ onset_thr).  Within it the
        chosen detector runs at inner_threshold, so sub-ruptures are collected
        EVEN IF they exceed outer_threshold (a strong inner rupture does not
        start a new junction).  Then hunting resumes left of the onset.

    inner_threshold defaults to outer_threshold (single-tier) when None.
    Forces and WLC fits are left None — this is pure geometry.
    """
    outer = float(outer_threshold)
    inner = float(inner_threshold) if inner_threshold is not None else outer
    events = CurveEvents(detector=detector)
    lo = max(0, lo)
    hi = min(len(d1), hi)

    boundaries = outer_events if outer_events is not None else find_outer_events(
        d1, mean_dev, piezo, lo=lo, hi=hi, onset_thr=onset_thr,
        outer_threshold=outer,
    )
    for boundary in boundaries:
        onset_idx = boundary.onset_idx

        # Outer-ROI RIGHT edge = the terminal's OWN d1 excursion end.  Walk right
        # from the outer run only while d1 stays above INNER, then stop at the gap.
        # A feature further baseline-ward that never crosses OUTER (a weak post-
        # event bump) is separated by a d1<inner gap, so it is NOT enclosed and
        # cannot become an event — only an OUTER crossing opens/bounds an ROI.
        # (mean_dev's baseline return can lag far past the terminal when the tip
        # stays partially loaded, which is why it must NOT set this edge.)
        return_idx = boundary.terminal_run_hi
        while return_idx + 1 < hi and d1[return_idx + 1] > inner:
            return_idx += 1

        if detector == "find_peaks":
            ruptures = detect_ruptures_findpeaks(
                d1, height=inner, prominence=prominence,
                distance_pts=distance_pts, lo=onset_idx, hi=return_idx + 1,
            )
        else:
            ruptures = detect_ruptures_threshold(
                d1, threshold=inner, lo=onset_idx, hi=return_idx + 1,
            )
        if ruptures:
            for r in ruptures:
                r.piezo_nm = float(piezo[r.idx])
            events.rois.append(ROI(
                onset_idx=onset_idx, return_idx=return_idx,
                onset_piezo_nm=float(piezo[onset_idx]),
                return_piezo_nm=float(piezo[return_idx]),
                ruptures=ruptures,
                segments=_segments_from_ruptures(onset_idx, ruptures, piezo),
            ))

    return events


def _segments_from_ruptures(
    onset_idx: int,
    ruptures:  list[Rupture],
    piezo:     np.ndarray,
) -> list[Segment]:
    """
    One loading-ramp segment per rupture, bounded by d1-peak EDGES (not force
    magnitudes): ramp i runs from the previous rupture's fall_idx (the bottom of
    its drop) — or the ROI onset for the first ramp — up to this rupture's
    rise_idx (its d1-peak left edge = the force peak).  So the previous rupture's
    snap-back decay is excluded by construction, and no reload-trough hunt is
    needed.
    """
    segs: list[Segment] = []
    prev_right = onset_idx
    for r in ruptures:
        left = prev_right
        end  = r.rise_idx if r.rise_idx is not None else r.idx
        if end <= left:                      # degenerate — keep a valid window
            end = max(left + 1, r.idx)
        segs.append(Segment(
            left_idx=left, right_idx=end,
            left_piezo_nm=float(piezo[left]), right_piezo_nm=float(piezo[end]),
            n_pts=abs(end - left) + 1,
        ))
        prev_right = r.fall_idx if r.fall_idx is not None else r.idx
    return segs


# ── THE definition of a rupture's force peak ─────────────────────────────────

def ramp_force_peak(force: np.ndarray, lo: int, hi: int) -> Optional[int]:
    """
    THE single definition of where a loading ramp's force peak — the rupture
    point — sits.  Returns an ABSOLUTE index into `force`, or None if the window
    is degenerate.

    **Every consumer must call this.**  The segment fitter and
    anything downstream needing "the force at this rupture" answer the question
    here or not at all.  Two implementations means two rupture forces for the same
    rupture, and the one that reaches a figure is whichever happened to run last.

    Both arguments carry a requirement, and each has already been got wrong once:

      • `force` must be the SMOOTHED low-frequency force (k·(low_retr−offset)/invols).
        On the raw retract — a large sample-to-sample sawtooth — argmax locks onto
        a noise tooth rather than the peak.

      • [lo, hi] must be a d1-EDGE loading ramp: the previous rupture's fall_idx
        (bottom of its drop) → this rupture's rise_idx (where force starts
        dropping), i.e. exactly a Segment's [left_idx, right_idx] as built by
        _segments_from_ruptures.  A window bounded by d1 ARGMAX indices instead
        of the d1-peak EDGES leaks in the previous rupture's snap-back decay,
        whose tail can exceed this ramp's own peak whenever the previous rupture
        was the stronger one — argmax then reports a peak belonging to the
        previous event entirely.  That is the inverted_pair case, and it is the
        bug this function was extracted to kill.

    Caveat worth knowing, not currently handled: a peak landing on `hi` means the
    true maximum may lie OUTSIDE the window (rise_idx came early and truncated the
    ramp), which silently under-reports the force.  See `ramp_peak_is_edge_pinned`.
    """
    lo = max(0, int(lo))
    hi = min(len(force) - 1, int(hi))
    if hi <= lo:
        return None
    return lo + int(np.argmax(force[lo:hi + 1]))


def ramp_peak_is_edge_pinned(peak_idx: int, lo: int, hi: int) -> bool:
    """
    True when a ramp's force peak sits on the window's right edge — the classic
    signature that the real peak is outside the window and the reported rupture
    force is an underestimate.

    Diagnostic only: nothing acts on it yet, because acting would change which
    curves get fits and therefore force a recompute.  It is here so the condition
    has a name and one definition.
    """
    return hi > lo and int(peak_idx) >= int(hi)


def fit_segments(
    curve,
    events:       CurveEvents,
    offset_retr:  float,
    invols_slope: float,
    snapoff_piezo_nm: float,
    *,
    low_retr:  Optional[np.ndarray] = None,
    guess_l_p: float = 2.0,
    guess_l_c: float = 120.0,
) -> None:
    """
    Fill each Segment's (l_p, l_c, l_p_err, l_c_err) and each Rupture's force_pN
    IN PLACE.

    Signal: fit on the LOW-PASS force envelope (`low_retr`, the decomposed
    low-frequency retract the detector's d¹ already uses), NOT
    the raw retract.  The raw retract is a large sample-to-sample sawtooth; on it
    both argmax and a diff-based rupture finder lock onto noise teeth rather than
    the true peak.  Falls back to curve.defl_retr only if low_retr is not supplied.

    Physics per segment (onset→r1, r1→r2, …):
      • Global coordinates are built once, exactly as extract_segment does:
            defl_corr = (low_retr − offset) / invols
            force     = k · defl_corr                     (peak = argmax)
            extension = (piezo_retr − snapoff) − defl_corr
        The extension is valid for EVERY chain because all are surface-anchored,
        so the same x zero serves bond, tether and everything between.
      • The fit starts at the segment's left edge (the ROI onset) and ENDS at the
        force peak (argmax) — that peak IS the rupture force of the segment.

    Segments too short to fit (or whose optimiser fails) are left with None fits
    and their rupture keeps force_pN=None — never a fabricated value.  Mutates
    `events`; returns None.
    """
    if not invols_slope or not np.isfinite(invols_slope):
        for roi in events.rois:
            for seg in roi.segments:
                seg.fit_status = "no_fit"
                seg.fit_detail = "invalid calibration"
        return
    inv = invols_slope
    defl      = curve.defl_retr if low_retr is None else low_retr
    defl_corr = (defl - offset_retr) / inv
    force     = curve.spring_constant * defl_corr
    extension = (curve.piezo_retr - snapoff_piezo_nm) - defl_corr

    for roi in events.rois:
        # segments[i] ends on ruptures[i] (see _segments_from_ruptures).  Tracks
        # the previous rupture IN THIS ROI so each segment can look up "the
        # force I need to climb back through" for its isoforce crossing; reset
        # to None on any skip so a gap in the chain is never silently bridged.
        prev_rup: Optional[Rupture] = None
        for seg, rup in zip(roi.segments, roi.ruptures):
            a, b = seg.left_idx, seg.right_idx
            if b - a < 5:
                seg.fit_status = "no_fit"
                seg.fit_detail = "insufficient segment points"
                prev_rup = None
                continue

            F_slice = force[a:b + 1]
            x_slice = extension[a:b + 1]

            # The segment IS the loading ramp — its bounds came from d1 crossings
            # (previous rupture's fall edge → this rupture's rise edge), so the
            # previous rupture's snap-back decay is already excluded and NO
            # magnitude comparison between sub-regions is done.  Fit from the ramp
            # start (index 0) to its peak; that peak — the top of this single ramp —
            # is the rupture force.
            #
            # ramp_force_peak is THE definition of that peak; every caller uses
            # the same function on the same bounds, so the two cannot diverge.
            peak_abs = ramp_force_peak(force, a, b)
            if peak_abs is None:
                seg.fit_status = "no_fit"
                seg.fit_detail = "no force peak"
                prev_rup = None
                continue
            # Diagnostic, recorded for every segment that gets this far.
            # Nothing acts on it — it is here so a user can SEE which fits
            # stopped at the window edge and choose to exclude them, rather than
            # the app quietly deciding for them.
            seg.edge_pinned = ramp_peak_is_edge_pinned(peak_abs, a, b)

            peak_rel = peak_abs - a
            if peak_rel < 4:
                seg.fit_status = "no_fit"
                seg.fit_detail = "insufficient loading ramp"
                prev_rup = None
                continue

            rup.force_pN     = float(F_slice[peak_rel])
            rup.force_idx    = a + peak_rel
            rup.extension_nm = float(x_slice[peak_rel])

            # Isoforce crossing (lab convention, not a fit output): where THIS
            # segment's rising ramp first climbs back through the PREVIOUS
            # rupture's force.  Fit-independent by design — computed from the
            # loading ramp directly, same region fed to the WLC fit below, so a
            # bad/failed WLC fit never takes this down with it.
            if prev_rup is not None and prev_rup.force_pN is not None:
                seg.isoforce_x_nm = _isoforce_crossing_x(
                    F_slice[:peak_rel + 1], x_slice[:peak_rel + 1], prev_rup.force_pN,
                )

            x_fit = x_slice[:peak_rel + 1]
            F_fit = F_slice[:peak_rel + 1]
            mask  = x_fit > 0
            if int(mask.sum()) < 5:
                seg.fit_status = "no_fit"
                seg.fit_detail = "insufficient positive extension"
                prev_rup = rup
                continue
            x_fit, F_fit = x_fit[mask], F_fit[mask]

            # Recorded BEFORE the fit is attempted, so a segment whose optimiser
            # fails still reports how far the pull got.  z_max needs l_c
            # too and is therefore None for such a segment — but x_max is a
            # property of the data, not of the fit, and shouldn't vanish with it.
            seg.x_max_nm = float(np.max(x_fit))

            fit = _fit_wlc_window(x_fit, F_fit, guess_l_p, guess_l_c)
            if fit is not None:
                (seg.l_p_nm, seg.l_c_nm,
                 seg.l_p_err, seg.l_c_err, seg.tau) = fit
                seg.n_pts = int(mask.sum())
                seg.fit_lo_idx = a
                seg.fit_hi_idx = a + peak_rel
                seg.fit_status = "review" if seg.edge_pinned else "fit_available"
                seg.fit_detail = "peak at segment boundary" if seg.edge_pinned else None
            else:
                seg.fit_status = "no_fit"
                seg.fit_detail = "optimizer failed"

            prev_rup = rup


def _isoforce_crossing_x(
    F_slice: np.ndarray, x_slice: np.ndarray, F_target: float,
) -> Optional[float]:
    """
    Extension (nm) where F_slice, walked forward, first reaches F_target —
    linearly interpolated between the bracketing samples for sub-sample
    precision.  None if F_target is already met at the first sample (no rise to
    find — an ill-defined crossing) or never reached at all.  Pure geometry, no
    fit involved.
    """
    above = F_slice >= F_target
    if above.size == 0 or above[0] or not above.any():
        return None
    j = int(np.argmax(above))   # first True index
    f0, f1 = float(F_slice[j - 1]), float(F_slice[j])
    if f1 == f0:
        return float(x_slice[j])
    t = (F_target - f0) / (f1 - f0)
    return float(x_slice[j - 1] + t * (x_slice[j] - x_slice[j - 1]))


# Below this many residual points the autocorrelation estimator has nothing to
# work with (the sum below would be dominated by its own sampling noise), so
# integrated_autocorr_time reports 1.0 — "no correction claimed" — rather than a
# number invented from a handful of samples.
_TAU_MIN_PTS = 20


def integrated_autocorr_time(resid: np.ndarray) -> float:
    """
    Integrated autocorrelation time tau = 1 + 2*sum_k rho_k of a fit residual,
    in SAMPLES.  This is the factor by which the residual series over-counts its
    own information: N samples with correlation time tau carry the information
    of N/tau independent ones, so a standard error computed as if all N were
    independent is understated by sqrt(tau).

    fit_segments fits `low_retr`, the app's
    own Bessel low-pass (signal_processing.bessel_decompose), sampled at the
    instrument's rate.  Neighbouring samples of a low-passed signal are not
    independent observations; they are the same information read several times.
    curve_fit's covariance assumes independence, so it divides the residual
    scatter by sqrt(N) when it has only sqrt(N/tau) to spend.

    THE ESTIMATOR is Geyer's initial positive sequence: sum rho_k from lag 1
    until the first non-positive rho.  Past that lag the estimate is dominated
    by its own noise, and summing further adds variance, not information.

    THREE GUARDS, each with a reason:
      * tau >= 1 always.  tau < 1 would SHRINK an error bar — claim the fit knows
        more than the samples contain.  Whatever the estimator says, this
        correction only ever widens.
      * tau <= n/2.  Beyond that, rho_k is computed from too few overlapping
        pairs to mean anything; a larger "answer" is the estimator running out
        of data, not a longer correlation.
      * n < _TAU_MIN_PTS, a non-finite residual, or a residual with no variance
        at all -> 1.0.  No correction is claimed where none can be measured.

    LIMITATION, ON RECORD.  tau is estimated from the residual, and the residual
    contains both correlated noise AND model error — the WLC curve not quite
    describing this ramp.  Systematic mismatch wandering across the fit window
    is, to this estimator, indistinguishable from long-correlated noise.  So
    sqrt(tau) is a FLOOR on the true uncertainty, not the whole of it: it does
    not cover the endpoint systematic (see ramp_peak_is_edge_pinned) and it
    absorbs model error only incidentally.

    HOW MUCH OF tau IS THE FILTER — ASK, DO NOT ASSUME.  Pushing white noise
    through this app's own decomposition, with no fit and no model involved,
    gives the rule

        tau_filter ~= sample_rate / cutoff_hz

    (measured over four rate/cutoff pairs; test_fit_uncertainty.py checks the
    rule, not a frozen value).  Because `cutoff_hz` is fixed per parameter set,
    tau_filter scales with the SAMPLE RATE THE EXPERIMENTALIST CHOSE, and so
    differs cohort by cohort — a 50 kHz cohort carries three times the
    filter-induced redundancy of a 16.7 kHz one for the same science.  Whatever
    is left over after tau_filter is model error: the WLC curve not quite
    describing the ramp, which drifts across the fit window and is
    indistinguishable from long-correlated noise to any estimator working on a
    residual.

    So the split between "their acquisition" and "our model" is NOT a constant
    and must be measured per cohort before it is quoted.  Compute tau_filter
    from that cohort's own sample rate and compare it with the tau actually
    stored on its fits.  sqrt(tau) remains a FLOOR either way: it corrects
    correlated noise, absorbs model error only incidentally, and does not cover
    the endpoint systematic (see ramp_peak_is_edge_pinned).
    """
    r = np.asarray(resid, dtype=float)
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n < _TAU_MIN_PTS:
        return 1.0
    r = r - r.mean()
    denom = float(r @ r)
    if not np.isfinite(denom) or denom <= 0.0:
        return 1.0

    # FFT autocorrelation, not np.correlate: the fit windows are ~1,700 points
    # and this runs once per segment per curve over batches of thousands, where
    # the direct O(n^2) form costs tens of seconds for the same answer.
    nfft = 1 << int(2 * n - 1).bit_length()
    spec = np.fft.rfft(r, nfft)
    ac   = np.fft.irfft(spec * np.conjugate(spec), nfft)[:n]
    rho  = ac / denom

    tau     = 1.0
    max_lag = max(1, n // 2)
    for k in range(1, max_lag):
        rk = float(rho[k])
        if not np.isfinite(rk) or rk <= 0.0:
            break
        tau += 2.0 * rk
    return float(min(max(tau, 1.0), n / 2.0))


def _fit_wlc_window(
    x: np.ndarray, F: np.ndarray, guess_l_p: float, guess_l_c: float,
) -> Optional[tuple[float, float, float, float, float]]:
    """
    Fit Marko-Siggia WLC to one (x, F) ramp.  Returns (l_p, l_c, l_p_err,
    l_c_err, tau) or None on failure.  l_c is floored at 1.001x the largest
    observed extension so the WLC singularity stays outside the data — that
    floor is a mathematical necessity (the model has a pole at x = l_c), not a
    policy choice. l_c has no ceiling: an unphysically large
    fit is a real answer the data/model produced, not a computation failure, and
    is already flagged by its own large l_c_err rather than by a hard cap.

    The error bars are correlation-corrected. curve_fit's
    sqrt(diag(pcov)) is the correct standard error only for INDEPENDENT
    residuals; these residuals are a low-passed series and are not (see
    integrated_autocorr_time).  So the returned l_p_err/l_c_err are that value
    multiplied by sqrt(tau), and tau is returned alongside them so a reader can
    see WHY an error bar is the size it is rather than having to take it on
    trust.  The size of the change is whatever sqrt(tau) happens to be for that
    fit — it is a per-fit correction, not a constant, and on real cohorts it has
    ranged from a few-fold upward.

    This changes ONLY the reported uncertainty.  popt is untouched, no curve
    changes status, and nothing is rejected or flagged.
    """
    x_max    = float(np.max(x))
    lc_guess = max(guess_l_c, x_max * 1.1)
    bounds   = ([0.05, x_max * 1.001], [500.0, np.inf])
    try:
        popt, pcov = fit_model(
            wlc, x, F, p0=[guess_l_p, lc_guess], bounds=bounds
        )
    except Exception:
        return None
    perr = np.sqrt(np.diag(pcov))
    tau  = integrated_autocorr_time(F - wlc(x, *popt))
    scale = float(np.sqrt(tau))
    return (float(popt[0]), float(popt[1]),
            float(perr[0]) * scale, float(perr[1]) * scale, float(tau))


_PAYLOAD_VERSION = 4   # v4: segments carry tau (residual autocorrelation time),
                       # x_max_nm and edge_pinned. The l_p_err/
                       # l_c_err in a v4 document are sqrt(tau)-corrected and are
                       # NOT comparable with a v3 document's — which is the whole
                       # reason this bumped rather than adding fields quietly.
                       # v3: ruptures carry extension_nm; segments carry
                       # isoforce_x_nm (both fit-independent).
                       # v2: ruptures carry d1-peak rise_idx/fall_idx; segments
                       # are d1-edge loading ramps (prev fall → this rise)


def events_to_payload(events: CurveEvents) -> dict:
    """
    Serialise CurveEvents → a JSON-able dict for the event_map side table.

    dX_pairs/dF_pairs/ordering are NOT stored — recomputed by the dataclass
    properties on load from stored rupture points, so there is one source of
    truth.  isoforce_x_nm IS stored (on the segment) because it cannot be
    recomputed later without the segment's own force/extension trace, which
    this document does not otherwise keep — see Segment.isoforce_x_nm.  `v`
    guards the schema so a future field change reads old rows as a miss
    (mirrors the 2DH grid-key builders' "v" convention).
    """
    return {
        "v": _PAYLOAD_VERSION,
        "detector": events.detector,
        "rois": [
            {
                "onset_idx": roi.onset_idx, "return_idx": roi.return_idx,
                "onset_piezo_nm": roi.onset_piezo_nm,
                "return_piezo_nm": roi.return_piezo_nm,
                "ruptures": [
                    {"idx": r.idx, "piezo_nm": r.piezo_nm,
                     "d1_height": r.d1_height, "prominence": r.prominence,
                     "force_pN": r.force_pN, "force_idx": r.force_idx,
                     "rise_idx": r.rise_idx, "fall_idx": r.fall_idx,
                     "extension_nm": r.extension_nm}
                    for r in roi.ruptures
                ],
                "segments": [
                    {"left_idx": s.left_idx, "right_idx": s.right_idx,
                     "left_piezo_nm": s.left_piezo_nm,
                     "right_piezo_nm": s.right_piezo_nm,
                     "l_p_nm": s.l_p_nm, "l_c_nm": s.l_c_nm,
                     "l_p_err": s.l_p_err, "l_c_err": s.l_c_err,
                     "n_pts": s.n_pts,
                     "fit_lo_idx": s.fit_lo_idx, "fit_hi_idx": s.fit_hi_idx,
                     "isoforce_x_nm": s.isoforce_x_nm,
                     # v4.  z_max is NOT here — it is a property
                     # derived from x_max_nm/l_c_nm on read, so there is one
                     # definition of it and it cannot go stale against them.
                     "tau": s.tau, "x_max_nm": s.x_max_nm,
                     "edge_pinned": s.edge_pinned,
                     "fit_status": s.fit_status,
                     "fit_detail": s.fit_detail}
                    for s in roi.segments
                ],
            }
            for roi in events.rois
        ],
    }


def payload_to_events(payload: dict) -> Optional[CurveEvents]:
    """
    Rebuild CurveEvents from a stored document.  Returns None if the payload is
    from an incompatible schema version (caller then recomputes).
    """
    if not payload or payload.get("v") != _PAYLOAD_VERSION:
        return None
    rois: list[ROI] = []
    for rd in payload.get("rois", []):
        ruptures = [
            Rupture(idx=r["idx"], piezo_nm=r["piezo_nm"],
                    d1_height=r["d1_height"],
                    prominence=r.get("prominence", float("nan")),
                    force_pN=r.get("force_pN"), force_idx=r.get("force_idx"),
                    rise_idx=r.get("rise_idx"), fall_idx=r.get("fall_idx"),
                    extension_nm=r.get("extension_nm"))
            for r in rd.get("ruptures", [])
        ]
        segments = [
            Segment(left_idx=s["left_idx"], right_idx=s["right_idx"],
                    left_piezo_nm=s["left_piezo_nm"],
                    right_piezo_nm=s["right_piezo_nm"],
                    l_p_nm=s.get("l_p_nm"), l_c_nm=s.get("l_c_nm"),
                    l_p_err=s.get("l_p_err"), l_c_err=s.get("l_c_err"),
                    n_pts=s.get("n_pts", 0),
                    fit_lo_idx=s.get("fit_lo_idx"), fit_hi_idx=s.get("fit_hi_idx"),
                    isoforce_x_nm=s.get("isoforce_x_nm"),
                    tau=s.get("tau"), x_max_nm=s.get("x_max_nm"),
                    edge_pinned=s.get("edge_pinned"),
                    fit_status=s.get(
                        "fit_status",
                        "fit_available" if s.get("l_p_nm") is not None else "not_attempted",
                    ),
                    fit_detail=s.get("fit_detail"))
            for s in rd.get("segments", [])
        ]
        rois.append(ROI(
            onset_idx=rd["onset_idx"], return_idx=rd["return_idx"],
            onset_piezo_nm=rd["onset_piezo_nm"],
            return_piezo_nm=rd["return_piezo_nm"],
            ruptures=ruptures, segments=segments,
        ))
    return CurveEvents(rois=rois, detector=payload.get("detector", ""))
