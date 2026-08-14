# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/histogram_binning.py
#
# Pure, Qt-free histogram binning for the variable window.
#
# The problem: a distribution that is NARROW relative to its full range
# collapses into 1–2 featureless bins, because a few far outliers stretch the
# np.histogram edges (full min/max) so the bulk lands in almost no bins.
# Tighter thresholds make it worse.
#
# The recipe (the conventional robust-histogram practice):
#   1. Robust axis range  — edges span the 1st–99th percentile, so outliers
#      stop stretching the range and the bulk resolves into real bins.
#   2. Freedman–Diaconis width within that range — bin width 2·IQR/n^(1/3),
#      the textbook outlier-resistant rule, clamped to
#      [MIN_BINS, MAX_AUTO_BINS].
#   3. Tails fall OUT of range — values outside the robust range are simply not
#      binned (np.histogram drops them), so the bars are clean and the first/
#      last bins mean what they say (no inflation to cut off later).  Nothing is
#      lost analytically: the scatter still shows every point, and the count of
#      out-of-range values is reported so the histogram stays honest.
#
# Everything here is a deterministic function of (values + the constants below),
# with NO plot/widget state.  That is what makes report-time retrieval easy:
# a report recomputes identical edges/counts from the cached DB values by
# calling these same functions — it never digs a histogram out of a live plot
# (consistent with "export from the DB, never PyQtGraph state" and the
# "derived, never stored" rule).  The binning constants are pinned here so any
# rebuild — including a figure regenerated months later — reproduces the exact
# same axes.  If they ever become per-variable user settings, they would be
# persisted alongside the thresholds, same pattern as the bounds.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Pinned binning policy (see module docstring).  Change deliberately — these
# define the published figure.
ROBUST_LO_PCT = 1.0     # lower percentile for the robust axis range
ROBUST_HI_PCT = 99.0    # upper percentile for the robust axis range
MIN_BINS      = 10

# Two ceilings answer two different questions.
#
# MAX_AUTO_BINS is how many bins the AUTOMATIC rule may choose for itself.  It
# is a taste limit on a heuristic: past this the Freedman-Diaconis width is
# resolving noise, and nobody asked for it.
#
# MAX_USER_BINS is how many a PERSON may ask for, and it is deliberately huge.
# Bin count is the only resolution knob whenever the range is chosen
# automatically, so capping it low caps the only control the user has: "if it
# decides way too wide and I need resolution, I can only get this with more
# bins — my only knob."  Binning finely, exporting, and zooming into the range
# that matters afterwards is a legitimate way to work.
#
MAX_AUTO_BINS = 60
MAX_USER_BINS = 20_000


@dataclass(frozen=True)
class HistogramBins:
    """Reproducible bin geometry for one variable's distribution.

    `edges` are the bin boundaries over the robust range (len = n_bins + 1).
    `count(subset)` histograms any subset of the values into these edges; values
    outside the range are dropped (not piled into end bars), so pass/fail splits
    share one clean geometry.  `n_below`/`n_above` report how many values fell
    out of range, so the histogram stays honest about what it isn't showing.
    """
    edges:    np.ndarray   # bin edges over the robust range
    range_lo: float        # robust lower edge (p1)
    range_hi: float        # robust upper edge (p99)
    n_below:  int          # values below range_lo (shown in scatter, not binned)
    n_above:  int          # values above range_hi (shown in scatter, not binned)

    @property
    def n_bins(self) -> int:
        return max(int(self.edges.size) - 1, 0)

    @property
    def n_out_of_range(self) -> int:
        return self.n_below + self.n_above

    def count(self, values: np.ndarray) -> np.ndarray:
        """Bin `values` into these edges; out-of-range values are dropped."""
        return counts_in_range(np.asarray(values, dtype=float), self.edges)


def robust_bins(values: np.ndarray) -> HistogramBins | None:
    """Compute reproducible bin geometry for `values` (finite values only).

    Returns None when there is nothing to bin.  Degenerate inputs (all-equal,
    zero IQR, or a tiny sample) fall back to a single narrow band around the
    value so the caller always gets drawable, non-empty edges.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None

    lo = float(np.percentile(v, ROBUST_LO_PCT))
    hi = float(np.percentile(v, ROBUST_HI_PCT))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        # All-equal (or degenerate) data: one symmetric band around the value
        # so the bar is visible rather than zero-width.
        c = float(v[0]) if hi <= lo else 0.5 * (lo + hi)
        pad = abs(c) * 1e-3 if c != 0.0 else 1.0
        lo, hi = c - pad, c + pad
        edges = np.linspace(lo, hi, 2)
        return HistogramBins(edges, lo, hi,
                             int(np.count_nonzero(v < lo)),
                             int(np.count_nonzero(v > hi)))

    n_below = int(np.count_nonzero(v < lo))
    n_above = int(np.count_nonzero(v > hi))

    # Freedman–Diaconis width over the robust range, clamped to a sane count.
    iqr = float(np.subtract(*np.percentile(v, [75.0, 25.0])))
    width = 2.0 * iqr / (v.size ** (1.0 / 3.0)) if iqr > 0 else 0.0
    if width > 0:
        n_bins = int(round((hi - lo) / width))
    else:
        n_bins = int(round(np.sqrt(v.size)))
    n_bins = int(np.clip(n_bins, MIN_BINS, MAX_AUTO_BINS))

    edges = np.linspace(lo, hi, n_bins + 1)
    return HistogramBins(edges, lo, hi, n_below, n_above)


def user_bins(values: np.ndarray, n_bins: int,
              lo: float | None = None,
              hi: float | None = None) -> HistogramBins | None:
    """Bin geometry the USER chose: an explicit count, over an explicit range.

    `lo`/`hi` default to the data's true min/max, which is what a plain
    `np.histogram(values, bins=n)` does — so the default answer is unchanged and
    only an explicit range narrows anything.

    The returned `n_below`/`n_above` are the whole point.  Narrowing the range
    of a plot is a view decision; narrowing the range of a histogram that a
    fit is then computed over is a SELECTION decision, because the fit sees bin
    heights and nothing else.  The count of what fell outside has to travel with
    the geometry so the window and the export manifest can both state it — the
    same rule that every selection stage reports what it was given.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None

    n_bins = int(np.clip(int(n_bins), 1, MAX_USER_BINS))
    lo = float(v.min()) if lo is None else float(lo)
    hi = float(v.max()) if hi is None else float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        # Degenerate request (inverted or zero-width): fall back to the data's
        # own range rather than inventing edges the user did not ask for.
        lo, hi = float(v.min()), float(v.max())
        if hi <= lo:
            pad = abs(lo) * 1e-3 if lo != 0.0 else 1.0
            lo, hi = lo - pad, hi + pad

    edges = np.linspace(lo, hi, n_bins + 1)
    return HistogramBins(edges, lo, hi,
                         int(np.count_nonzero(v < lo)),
                         int(np.count_nonzero(v > hi)))


def full_range_bins(values: np.ndarray) -> HistogramBins | None:
    """Reproducible bin geometry that drops nothing: same resolution (bin
    width) as robust_bins(), but the edges are extended outward at that same
    width until they cover the true min/max — never just the 1st-99th
    percentile core. For export/report use, where "sum(counts) == number of
    finite values" must hold so a reader can check no data went missing,
    unlike the on-screen robust_bins() convention, which deliberately drops
    tail values to keep the core resolved (see module docstring above).

    Core bins share the same edges robust_bins() would draw (same lo/hi
    anchor), so the two conventions agree everywhere except the extended tail
    bins. For a genuinely all-equal input, robust_bins()'s single visible band
    already covers the full range. If the robust percentiles collapse despite
    rarer values elsewhere, there is no meaningful robust width to preserve;
    one bin over the true min/max is used instead. This avoids both dropping
    those values and allocating an unbounded number of arbitrary-width bins.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None

    based = robust_bins(v)
    if based is None:
        return based

    true_lo = float(v.min())
    true_hi = float(v.max())
    if based.n_bins <= 1:
        if based.edges[0] <= true_lo and based.edges[-1] >= true_hi:
            return based
        return user_bins(v, 1)

    width = float(based.edges[1] - based.edges[0])

    lo = based.range_lo
    n_extra_lo = max(0, int(np.ceil((lo - true_lo) / width - 1e-9)))
    lo -= n_extra_lo * width

    hi = based.range_hi
    n_extra_hi = max(0, int(np.ceil((true_hi - hi) / width - 1e-9)))
    hi += n_extra_hi * width

    n_bins = int(round((hi - lo) / width))
    edges = lo + width * np.arange(n_bins + 1)
    # Floating-point safety: guarantee the true extremes fall inside the
    # outermost bins even after repeated += width accumulation.
    if edges[0] > true_lo:
        edges[0] = true_lo
    if edges[-1] < true_hi:
        edges[-1] = true_hi

    return HistogramBins(edges, edges[0], edges[-1], 0, 0)


def counts_in_range(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Histogram `values` into `edges`; values outside the range are dropped.

    np.histogram already excludes anything outside [edges[0], edges[-1]], so the
    bars stay clean — the first/last bins are real bins, not inflated tail piles.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n_bins = max(int(edges.size) - 1, 0)
    if v.size == 0 or n_bins == 0:
        return np.zeros(n_bins, dtype=int)
    counts, _ = np.histogram(v, bins=edges)
    return counts
