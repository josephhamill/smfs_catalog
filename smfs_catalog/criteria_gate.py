# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""Apply the active criteria to a cohort of event curves.

A BOUND IS THE CRITERION.  A thresholds row with either bound set gates the
hit; clearing both bounds removes it.  There is no second switch.

Two switches is what this subsystem had, and the cost was that neither one
was visible from where the verdict is read.  A user set a bound in the
variable window, watched the pass/fail split render in front of them, closed
the window, and nothing had happened — because participation lived in a
separate per-experimentalist profile key that nothing on screen mentioned.
Faceted search and flow-cytometry gating both answer this the same way: a
gate that exists, applies.

The verdict is never stored.  Membership is derived here, on demand, from the
bounds as they are right now — so a bound that moves cannot leave a stale
`hit` column behind it in the catalog.

Qt-free and DB-agnostic beyond db's own accessors, like ledger: the windows
adapt this, it never reaches for them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import db as _db
from . import ledger as _ledger
from . import quantities as _quant


# The two verdicts a curve can carry.  Named because the strings reach the
# queue's Hit column, the population control and every export manifest, and a
# typo in any of those is a silent wrong answer rather than a crash.
HIT = "hit"
NON_HIT = "non_hit"

# (variable key, the curve's value, why it failed, the bound it failed against)
# `kind` is one of "missing" / "below" / "above".  "missing" is kept distinct
# from the two violations because a curve with no value for a variable has not
# failed that test — it could not be asked it.  The Hit column's tooltip is
# the consumer that tells the two apart.
FailReason = tuple[str, "float | None", str, "float | None"]


@dataclass(frozen=True)
class Criterion:
    """One variable's bounds, as they gate the hit."""

    key:   str
    lower: "float | None"
    upper: "float | None"
    label: str = ""

    @property
    def name(self) -> str:
        """What to call this criterion on screen."""
        return self.label or self.key

    def text(self) -> str:
        """The criterion as a sentence — `WLC R² ≥ 0.95`.

        Digits and units come from quantities so this line, the bound shown
        beside the editing control and the spin box itself cannot disagree.
        """
        f = lambda v: _quant.format_value(self.key, v, with_unit=True)
        if self.lower is not None and self.upper is not None:
            return f"{f(self.lower)} ≤ {self.name} ≤ {f(self.upper)}"
        if self.lower is not None:
            return f"{self.name} ≥ {f(self.lower)}"
        return f"{self.name} ≤ {f(self.upper)}"

    def verdict(self, value: "float | None") -> "FailReason | None":
        """Why this value fails, or None when it passes.

        A value that is absent or not finite yields the "missing" kind: the
        question could not be asked of this curve.
        """
        if value is None or not math.isfinite(value):
            return (self.key, None, "missing", None)
        if self.lower is not None and value < self.lower:
            return (self.key, value, "below", self.lower)
        if self.upper is not None and value > self.upper:
            return (self.key, value, "above", self.upper)
        return None

    def manifest(self) -> dict:
        """This criterion, for an export manifest."""
        return {
            "key":   self.key,
            "label": self.name,
            "lower": self.lower,
            "upper": self.upper,
            "unit":  _quant.unit_of(self.key),
        }


@dataclass(frozen=True)
class Gate:
    """Every criterion in force, and whose bounds they are."""

    owner:    str
    criteria: tuple[Criterion, ...] = ()

    def __bool__(self) -> bool:
        """False when nothing constrains — then every event is a hit."""
        return bool(self.criteria)

    def names(self) -> str:
        """Which variables are cutting, for a status line.

        Names only: a real gate runs to a dozen criteria, and spelling out
        every bound turns one line into a paragraph.  text() is the version
        with the numbers, for a tooltip or a manifest.
        """
        if not self.criteria:
            return "none — every event is a hit"
        return f"{len(self.criteria)}: " + " · ".join(
            c.name for c in self.criteria)

    def text(self) -> str:
        """Everything in force, with its bounds."""
        if not self.criteria:
            return "none — every event is a hit"
        return "\n".join(c.text() for c in self.criteria)

    def manifest(self) -> dict:
        """The gate, for an export manifest.

        A figure's population is not reconstructable from a population name
        alone; whoever reads the manifest months later needs the bounds that
        defined it.
        """
        return {
            "criteria_owner": self.owner,
            "criteria":       [c.manifest() for c in self.criteria],
            "criteria_rule":  (
                "a curve is a hit when every criterion is within bounds; a "
                "non-hit when any is violated or has no finite value"
            ),
        }


@dataclass(frozen=True)
class Classification:
    """Which population each curve is in, and why any of them is not a hit."""

    gate:       Gate
    membership: dict[str, str]
    reasons:    dict[str, list[FailReason]]

    def of(self, path: str) -> str:
        """The population this one curve belongs to."""
        return self.membership.get(path, HIT)

    def population(self, which: str) -> list[str]:
        """Every path in `which`, in the order they were classified."""
        return [p for p, m in self.membership.items()
                if _ledger.in_population(which, m)]

    def why(self, path: str) -> list[FailReason]:
        """Every criterion this curve failed. Empty for a hit."""
        return self.reasons.get(path, [])

    def counts(self) -> dict[str, int]:
        """How many curves are in each population."""
        out = {HIT: 0, NON_HIT: 0}
        for m in self.membership.values():
            out[m] = out.get(m, 0) + 1
        return out


def active_owner(db_path: str = _db.DEFAULT_DB_PATH) -> str:
    """Return the profile owner whose criteria currently govern the queue."""
    return _db.active_param_owner(db_path)


def gate(
    experimentalist: str | None = None,
    db_path: str = _db.DEFAULT_DB_PATH,
) -> Gate:
    """The criteria in force for an experimentalist.

    db.get_thresholds already merges the shared default owner's rows under
    this owner's own, so an inherited bound gates exactly as a personal one
    does.  A row with neither bound set is not a criterion — that is how a
    criterion is removed.
    """
    owner = experimentalist or active_owner(db_path)
    criteria = tuple(
        Criterion(row["analysis_type"], row["lower_bound"], row["upper_bound"],
                  row["label"] or "")
        for row in _db.get_thresholds(owner, db_path)
        if row["lower_bound"] is not None or row["upper_bound"] is not None
    )
    return Gate(owner, tuple(sorted(criteria, key=lambda c: c.key)))


def _values(
    event_paths: list[str],
    keys: list[str],
    db_path: str,
) -> dict[str, dict[str, "float | None"]]:
    """Load the selected variables for the supplied paths."""
    from . import variables as _vars

    return _vars.values(event_paths, keys, db_path)


def classify(
    event_paths: list[str],
    db_path: str = _db.DEFAULT_DB_PATH,
) -> Classification:
    """Sort a cohort into populations under the queue's active criteria.

    With no criteria in force every event is a hit — the gate cannot claim a
    curve failed a test nobody set.  The dashboard says so in as many words,
    because "everything is a hit" and "everything passed" look identical on
    screen and mean very different things.
    """
    g = gate(None, db_path)
    if not event_paths:
        return Classification(g, {}, {})
    if not g:
        return Classification(g, {p: HIT for p in event_paths}, {})

    keys = [c.key for c in g.criteria]
    values = _values(event_paths, keys, db_path)
    membership: dict[str, str] = {}
    reasons: dict[str, list[FailReason]] = {}
    for path in event_paths:
        path_values = values.get(_db.normalize_path(path), {})
        failed = [r for r in (c.verdict(path_values.get(c.key))
                              for c in g.criteria) if r is not None]
        membership[path] = NON_HIT if failed else HIT
        if failed:
            reasons[path] = failed
    return Classification(g, membership, reasons)
