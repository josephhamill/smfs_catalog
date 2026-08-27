# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/ledger.py
#
# The drop ledger.  Pure logic, no UI, no DB.
#
# THE RULE: a stage that narrows a set reports what it was GIVEN, not only
# what it KEPT.  The event-summary and 2DH stages take cohorts and shrink
# them — no stored segmentation, no l_p/l_c, file won't load, landmarks
# missing.  Without a record of those exclusions, a window saying "306
# events" gives nobody a way to establish that the 307th was legitimately
# absent rather than lost, and every population count in a paper figure is
# an unverifiable claim.
#
# Two things are owed, and they are ONE mechanism, not two:
#
#   the tally    asked N, kept M, dropped K — on screen and in the manifest
#   the journey  follow ONE named curve through every stage and see where
#                it fell out
#
# If each stage records, per file, WHY it excluded that file, the tally is a
# group-by over those records and the journey is a filter on them. A Ledger
# therefore stores one record per reported exclusion. A path can have several
# records when it is refused at several stages or for several reasons.
#
# Same shape as criteria_gate.explain(), which returns per-file which criterion
# failed and against what bound.  Qt-free for the same reason criteria_gate is:
# the windows adapt it, it never reaches for them.
#
# NOT a cache.  A Ledger describes one pass of one stage and is rebuilt with
# whatever rebuilt the view.  Nothing persists it, so nothing has to
# invalidate it.

from __future__ import annotations

from dataclasses import dataclass


# ── The reason vocabulary ────────────────────────────────────────────────────
# Closed, like db.EVENT_VERDICTS: a reason string not in here is a typo, and
# tests/test_drop_ledger.py fails on one.  The value is the human sentence
# shown on screen and written into manifests — stated as what is MISSING, so
# it reads as an explanation rather than an accusation.
#
# These must be distinguishable from each other.
DROP_REASONS: dict[str, str] = {
    "not_in_population": "not in the selected population (hit/non-hit)",
    "no_stored_segments": "no stored segmentation under the current parameter set",
    "no_segment_chosen":  "the chosen segment doesn't exist on this curve",
    "no_fit":             "the chosen segment's WLC fit produced no l_p/l_c",
    "no_force":           "no rupture force for the chosen segment",
    "no_length":          "no contour length for the chosen segment",
    "not_in_catalog":     "file is not in the catalog",
    "unreadable":         "the curve file could not be read",
    "no_landmarks":       "landmarks missing — no ROI to build from",
    "no_histogram":       "the per-curve histogram could not be built",
    "not_finite":         "the value is missing or not a finite number",
    "cancelled":          "the build was cancelled before reaching this curve",
}


@dataclass(frozen=True)
class Drop:
    """One stage's refusal of one file.

    `detail` is free text for the specifics that make a drop actionable —
    "l_c present, l_p None", a bound that was missed.  Empty is fine; the
    reason alone already says more than a bare `continue` ever did.
    """
    path:   str
    stage:  str
    reason: str
    detail: str = ""

    @property
    def label(self) -> str:
        """Human sentence for this drop — reason text plus any specifics."""
        base = DROP_REASONS.get(self.reason, self.reason)
        return f"{base} ({self.detail})" if self.detail else base


class Ledger:
    """Records, per file, why a stage excluded it.

    Usage is deliberately blunt — the point is that a `continue` costs one
    extra line, so there is no excuse for a silent one:

        led = Ledger("2DH build", paths)
        for p in paths:
            fit = stored_fit(p)
            if fit is None:
                led.drop(p, "no_stored_segments")
                continue
            ...
        led.summary()      -> "asked 307 · kept 306 · dropped 1"

    A path may be dropped more than once (different stages) — `journey()`
    depends on that, and `kept()` is "asked minus everything ever dropped",
    so recording a second reason can never resurrect a curve.
    """

    def __init__(self, name: str, asked) -> None:
        self.name = name
        # A path identifies one curve throughout the ledger. Preserve the
        # caller's order, discard empty entries, and collapse duplicate paths
        # so list counts and the set-based drop accounting use the same unit.
        self._asked: list[str] = list(dict.fromkeys(p for p in asked if p))
        self._asked_set: set[str] = set(self._asked)
        self._drops: list[Drop] = []
        self._dropped: set[str] = set()

    # ── Recording ────────────────────────────────────────────────────────────

    def drop(self, path: str, reason: str, detail: str = "", stage: str = "") -> None:
        """Record that this stage excluded `path`, and why.

        Unknown reasons are recorded rather than raised — losing the count at
        runtime because a reason string was misspelled would be worse than
        the typo. The test is what enforces the vocabulary.
        """
        if not path:
            return
        self._drops.append(Drop(path, stage or self.name, reason, detail))
        self._dropped.add(path)

    def drop_all(self, paths, reason: str, detail: str = "", stage: str = "") -> None:
        for p in paths:
            self.drop(p, reason, detail, stage)

    def absorb(self, other: "Ledger") -> None:
        """Fold an upstream stage's drops into this one.

        This makes a journey work across stage boundaries: a downstream
        ledger can still explain why a curve never reached it. Only the drop
        records travel; `asked`, the tally, and manifest counts remain scoped
        to this stage's own input.
        """
        self._drops.extend(other._drops)
        self._dropped |= other._dropped & self._asked_set

    # ── Reading ──────────────────────────────────────────────────────────────

    @property
    def n_asked(self) -> int:
        return len(self._asked)

    @property
    def n_dropped(self) -> int:
        return len(self._dropped & self._asked_set)

    @property
    def n_kept(self) -> int:
        return self.n_asked - self.n_dropped

    def kept(self) -> list[str]:
        """Asked minus dropped, in the caller's original order."""
        return [p for p in self._asked if p not in self._dropped]

    def drops(self) -> list[Drop]:
        return list(self._drops)

    def by_reason(self) -> dict[str, int]:
        """{reason: how many files it accounts for}, largest first.

        Each dropped file is counted ONCE, under the FIRST reason recorded
        for it, so the breakdown sums to exactly n_dropped. That
        reconciliation is the point of the tally — a reader confronted with
        "dropped 12" and reasons adding to 17 has been handed a second
        puzzle, not an explanation. A file that failed several checks still
        shows all of them in journey(), which is where that detail belongs.
        """
        first: dict[str, str] = {}
        for d in self._drops:
            if d.path in self._asked_set and d.path not in first:
                first[d.path] = d.reason
        counts: dict[str, int] = {}
        for reason in first.values():
            counts[reason] = counts.get(reason, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def journey(self, path: str) -> list[Drop]:
        """Every stage that refused this one curve, in the order recorded.

        The owner's framing: "we'd like to see how each trace travels through
        the filtering." An empty list means
        the curve survived every stage this ledger knows about.
        """
        return [d for d in self._drops if d.path == path]

    # ── Reporting ────────────────────────────────────────────────────────────

    def summary(self, kept_word: str = "kept") -> str:
        """The standing line: `asked 307 · plotted 306 · dropped 1`.

        `kept_word` is the window's own verb — plotted, fitted, built — so
        the line reads as what that window did, while the shape stays
        identical everywhere and remains greppable.
        """
        line = f"asked {self.n_asked:,} · {kept_word} {self.n_kept:,}"
        if self.n_dropped:
            line += f" · dropped {self.n_dropped:,}"
        return line

    def breakdown_lines(self) -> list[str]:
        """One `12 × no stored segmentation…` line per reason, largest first."""
        return [
            f"{n:,} × {DROP_REASONS.get(reason, reason)}"
            for reason, n in self.by_reason().items()
        ]

    def report(self) -> str:
        """Multi-line tally + breakdown, for a tooltip or a dialog."""
        out = [f"{self.name}: {self.summary()}"]
        out.extend(f"  {line}" for line in self.breakdown_lines())
        return "\n".join(out)

    # ── Export ───────────────────────────────────────────────────────────────

    def rows(self) -> list[dict]:
        """One serializable row per recorded drop.

        Callers may pass these rows to ExportGroup.dict_table when a full
        per-curve audit table is wanted. Full paths, never basenames — the
        same filename can recur across directories.
        """
        return [
            {"path": d.path, "stage": d.stage, "reason": d.reason,
             "reason_text": DROP_REASONS.get(d.reason, d.reason), "detail": d.detail}
            for d in self._drops
        ]

    def manifest(self) -> dict:
        """The tally, for a window's export_provenance().

        Counts describe this ledger's own `asked` cohort. Drops absorbed from
        an upstream ledger remain available through journey() and rows(), but
        do not alter this stage's tally. A manifest is read months later by
        someone without the app, so these counts make its result population
        independently reconcilable.
        """
        return {
            "stage":           self.name,
            "n_asked":         self.n_asked,
            "n_kept":          self.n_kept,
            "n_dropped":       self.n_dropped,
            "dropped_by_reason": self.by_reason(),
        }
