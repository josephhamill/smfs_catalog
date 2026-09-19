# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""What one variable looks like as a criterion: its shape, its cut, its cost.

The numbers behind the criteria overview — one Card per variable, holding the
distribution to draw, the bounds in force, how many curves survive the cut,
and how many cannot answer the question at all.

Qt-free and plot-library-free on purpose, the way ledger and criteria_gate
are.  A card is arithmetic over stored values; drawing it is a separate job,
and the drawing is the half that would have to be rewritten for a web front
end.  Keeping the arithmetic out of the window is what makes that a rewrite
of the view rather than of the feature.

THE COST LINE IS THE POINT.  `n_missing` is how many curves have no finite
value for this variable, and it is shown on the card because that is the
moment the user is deciding whether to impose the criterion.  Gating on a
quantity half the cohort cannot produce is a decision, not a technicality —
the reload and rupture-separation distances exist only where an ROI has two
or more ruptures — and it should be visible before the bound is set, not
discovered afterwards as a mysteriously harsh gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import criteria_gate as _gate
from . import db as _db
from . import histogram_binning as _binning
from . import quantities as _quant


@dataclass(frozen=True)
class Card:
    """One variable's distribution over a cohort, and the cut through it."""

    key:      str
    label:    str
    values:   np.ndarray          # finite values only, one per curve that has one
    bins:     "_binning.HistogramBins | None"
    lower:    "float | None"      # the bound in force, not a proposal
    upper:    "float | None"
    n_curves: int                 # the whole cohort, including those with no value

    @property
    def in_force(self) -> bool:
        """Whether this variable currently gates the hit."""
        return self.lower is not None or self.upper is not None

    @property
    def n_missing(self) -> int:
        """Curves with no finite value — the ones this criterion cannot ask."""
        return self.n_curves - int(self.values.size)

    @property
    def is_integer(self) -> bool:
        """Whether this quantity only takes whole numbers."""
        return _quant.get(self.key).integer

    def within(self, lower=None, upper=None, *, proposed: bool = False) -> int:
        """How many curves this cut would keep.

        `proposed` distinguishes a bound being dragged from the bound in
        force, so a card can answer "how many if I let go here" without
        anything being written.
        """
        lo = lower if proposed else (self.lower if lower is None else lower)
        hi = upper if proposed else (self.upper if upper is None else upper)
        keep = np.ones(self.values.shape, dtype=bool)
        if lo is not None:
            keep &= self.values >= lo
        if hi is not None:
            keep &= self.values <= hi
        return int(np.count_nonzero(keep))

    def span(self) -> "tuple[float, float] | None":
        """The drawable range: the histogram's, widened to hold the bounds.

        A bound set outside the robust range would otherwise sit off the edge
        of its own card, which is precisely when a user most needs to see
        where it is.
        """
        if self.bins is None:
            return None
        lo, hi = float(self.bins.range_lo), float(self.bins.range_hi)
        for b in (self.lower, self.upper):
            if b is not None:
                lo, hi = min(lo, float(b)), max(hi, float(b))
        if hi <= lo:
            hi = lo + (abs(lo) * 1e-3 or 1.0)
        return lo, hi

    def cost_text(self) -> str:
        """The line under the card, stated as what the cut does and costs."""
        if self.n_curves == 0:
            return "no curves"
        if not self.in_force:
            base = (f"not gating  ·  {self.values.size:,} of "
                    f"{self.n_curves:,} have a value")
        else:
            base = f"{self.within():,} of {self.n_curves:,} within"
        if self.n_missing:
            base += f"  ·  {self.n_missing:,} have no value"
        return base


def cohort(
    variables: list[tuple[str, str]],
    event_paths: list[str],
    experimentalist: "str | None" = None,
    db_path: str = _db.DEFAULT_DB_PATH,
) -> list[Card]:
    """Build a Card per variable, in ONE pass over the cohort's values.

    One `variables.values` call for every key rather than one per card: the
    query is already the expensive part, and a card-at-a-time loop would run
    it a dozen times to answer a dozen questions about the same curves.

    Cards come back with the ones in force first, because "what is cutting my
    data" is the question this window is opened to answer.  Within a section
    they keep the caller's order, which is the queue's column order.
    """
    from . import variables as _vars

    keys = [k for k, _lbl in variables]
    if not keys:
        return []
    bounds = {
        row["analysis_type"]: (row["lower_bound"], row["upper_bound"])
        for row in _db.get_thresholds(experimentalist, db_path)
    }
    raw = _vars.values(event_paths, keys, db_path) if event_paths else {}
    normalized = [_db.normalize_path(p) for p in event_paths]

    cards: list[Card] = []
    for key, label in variables:
        column = np.array(
            [raw.get(p, {}).get(key) for p in normalized],
            dtype=object)
        finite = np.array(
            [float(v) for v in column
             if isinstance(v, (int, float)) and np.isfinite(float(v))],
            dtype=float)
        lo, hi = bounds.get(key, (None, None))
        cards.append(Card(
            key=key, label=label, values=finite,
            bins=_binning.robust_bins(finite) if finite.size else None,
            lower=lo, upper=hi, n_curves=len(normalized)))
    cards.sort(key=lambda c: not c.in_force)
    return cards


def joint(cards: list[Card], event_paths: list[str],
          db_path: str = _db.DEFAULT_DB_PATH) -> str:
    """The whole gate's verdict on this cohort, for the window's footer.

    Asked of criteria_gate rather than recomputed from the cards: a curve is
    a hit only if EVERY criterion passes, and a per-card count cannot answer
    a conjunction.  Two ways of counting hits would eventually disagree, and
    the gate's answer is the one the rest of the app acts on.
    """
    cls = _gate.classify(event_paths, db_path)
    total = len(event_paths)
    if not cls.gate:
        return f"No criteria set — every event is a hit ({total:,} events)."
    counts = cls.counts()
    return (f"{counts.get(_gate.HIT, 0):,} of {total:,} events are hits  ·  "
            f"{counts.get(_gate.NON_HIT, 0):,} non-hits")
