# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/clustering.py
#
# The session's current cluster assignment. Qt-free, DB-free.
#
# DELIBERATELY EPHEMERAL — this is the whole design, not a shortcut.
#
# A k-means label is not a property of a curve.  It is a property of the SET:
# the centroids are determined by every member together, so removing a third
# of the cohort does not leave the survivors "valid but incomplete" — the
# clusters would have landed somewhere else entirely.  There is no honest
# partial answer to fall back on.
#
# Storing one would therefore mean keeping it true against cohort membership,
# the 2DH's align mode / segment / grid / selected area, the analysis
# parameters underneath, k, the seed, n_pcs and the code version — MORE inputs
# than the gate verdict had, and that was deleted (files.hit) for exactly this
# reason.  The queue is cleared at every launch, so a stored label would
# describe a cohort that is not even loaded.
#
# What makes ephemeral safe rather than lossy is that re-running costs seconds
# — unlike an analysis result, which costs hours — and THE EXPORT IS THE
# DURABLE RECORD: pca_window's _scores.csv already carries path, PC1..PCn and
# cluster, with k and the seed in its manifest.  That is the artefact that
# leaves the app and reaches a figure.  Nothing worth keeping is lost.
#
# Consequence: there is nothing to invalidate, so there are no staleness rules
# here.  A cohort can still move under a live clustering (edit a criterion,
# repopulate the queue), and the answer to that is the COVERAGE LINE every
# consumer draws plus the user's own Clear button — never an automatic rule
# deciding on their behalf. The view informs but does not gate.

from __future__ import annotations

import datetime
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Clustering:
    """One k-means run, and enough about it to say what it was.

    `labels` is keyed by normalized path so a consumer can look up whatever
    cohort it happens to be showing, which is never guaranteed to be the one
    that was clustered — the 2DH this ran on holds one population (hits OR
    non-hits), while Explore Events shows both.
    """
    labels:          dict[str, int]
    k:               int
    seed:            int
    n_pcs:           int
    sklearn_version: str
    source:          dict = field(default_factory=dict)
    created_at:      str = ""

    @property
    def n_labelled(self) -> int:
        return len(self.labels)

    def label_for(self, path: str) -> int | None:
        return self.labels.get(path)

    def describe(self) -> str:
        """The one line every consumer puts on screen beside the colours.

        It names the SOURCE MATRIX, not just k, because a reader seeing
        coloured dots on a force-vs-length scatter will otherwise assume the
        clustering happened in that space. It did not; this is only a
        projection of the clustering onto scalar variables.
        """
        bits = [f"k-means k={self.k}"]
        window = self.source.get("window")
        if window:
            bits.append(f"{window} 2DH")
        for key, prefix in (("align_mode", ""), ("segment", "segment ")):
            val = self.source.get(key)
            if val:
                bits.append(f"{prefix}{val}")
        if self.source.get("selection_window"):
            bits.append("selected feature space")
        if self.source.get("population"):
            bits.append(self.source["population"])
        if self.created_at:
            bits.append(self.created_at)
        return " · ".join(bits)


def first_pc_order(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Original cluster IDs ordered by their members' mean PC1 score."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return np.array([], dtype=int)
    pc1 = np.asarray(scores)[:, 0]
    centres = {c: float(pc1[labels == c].mean()) for c in np.unique(labels)}
    return np.asarray(sorted(centres, key=lambda c: centres[c]), dtype=int)


def order_by_first_pc(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Relabel so cluster 0 has the left-most PC1 centroid, 1 the next, ...

    k-means numbers its clusters arbitrarily, so the same data re-run with a
    different k — or with one curve added — hands the same physical group a
    different integer, and every colour in every window jumps for no reason.
    Ordering by a real coordinate makes the numbering mean something and keeps
    it stable across runs. Fit-component ordering uses the same remedy.
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return labels
    order = first_pc_order(labels, scores)
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[int(c)] for c in labels], dtype=int)


# ── The session's one clustering ─────────────────────────────────────────────
# A module-level singleton rather than something owned by a window: the run is
# produced in the PCA window and consumed in three others that know nothing
# about it, and none of them outlives or contains the rest.

_current: Clustering | None = None
_subscribers: list = []


def current() -> Clustering | None:
    return _current


def set_current(clustering: Clustering) -> None:
    """Publish a new run, replacing whatever was there.

    Replacing rather than accumulating reflects that only the current
    exploratory clustering is active; durable results belong in exports.
    """
    global _current
    _current = clustering
    _notify()


def clear() -> None:
    """The user's Clear button.

    Needed even though this all dies at exit, and for a different moment:
    change the criteria or repopulate the queue mid-session and the colouring
    is stale immediately.  Deliberately NOT automatic on a cohort change —
    that would be an invalidation rule, which is what this design exists
    without; the coverage line reports the mismatch and the user decides.
    """
    global _current
    _current = None
    _notify()


def subscribe(callback) -> None:
    """Register a plain callable, invoked with no arguments on any change.

    Plain callables rather than a Qt signal so this module stays importable
    headless and testable without a QApplication — the same reason
    scanner.scan_directory takes a progress callback instead of a QObject.
    """
    if callback not in _subscribers:
        _subscribers.append(callback)


def unsubscribe(callback) -> None:
    """Windows MUST call this on close: a subscriber list holding a deleted
    QWidget raises on the next publish, and the traceback points at whichever
    window happened to be first."""
    if callback in _subscribers:
        _subscribers.remove(callback)


def _notify() -> None:
    for cb in list(_subscribers):
        try:
            cb()
        except Exception:            # noqa: BLE001
            # One window failing to repaint must not stop the others being
            # told, and must not take down the k-means run that published.
            pass


def now_stamp() -> str:
    return datetime.datetime.now().strftime("%H:%M")


def coverage_text(paths: list[str]) -> str:
    """'1,700 of 1,725 curves labelled' for the cohort actually on screen.

    Every consumer shows this, because the cohort a window displays is rarely
    the one that was clustered: the 2DH holds one population, Explore Events
    shows both, and either can move under a live clustering.
    """
    c = _current
    if c is None:
        return ""
    n = sum(1 for p in paths if p in c.labels)
    return f"{n:,} of {len(paths):,} curves labelled"


def labels_for_rows(rows: list[dict], path_key: str = "path") -> list[dict]:
    """Copy rows and join the live cluster label by path when one exists.

    An uncovered curve receives None, never cluster 0. With no live clustering
    the original rows are returned and no cluster column is implied.
    """
    c = _current
    if c is None:
        return rows
    return [
        {**row, "cluster": c.label_for(str(row.get(path_key) or ""))}
        for row in rows
    ]


def provenance(paths: list[str], shown: bool) -> dict:
    """The clustering's facts for an export manifest, from any window.

    One implementation because three windows export cluster columns and a
    manifest key spelled differently in each is a file nobody can join on.

    `shown` records whether the colouring was actually ON SCREEN, separately
    from whether a clustering existed — a figure and its CSV must agree about
    what was drawn, the same reason mean_curve_window records which of
    +-1sigma / +-SE was displayed.

    Always emits the keys, with nulls when there is no clustering: an absent
    key is indistinguishable from an older export that never had one.
    """
    c = _current
    if c is None:
        return {"clustering": None, "cluster_colouring_shown": False}
    cohort = {str(p) for p in paths}
    labelled = set(c.labels)
    return {
        "clustering": {
            "k":               c.k,
            "seed":            c.seed,
            "n_pcs":           c.n_pcs,
            "sklearn_version": c.sklearn_version,
            "created_at":      c.created_at,
            "source_matrix":   "2DH ensemble",
            "source":          dict(c.source),
            "n_labelled_total":  c.n_labelled,
            "n_labelled_in_cohort": sum(1 for p in paths if p in c.labels),
            "n_in_cohort":     len(paths),
            "cohort_matches_clustering": cohort == labelled,
            # Ephemeral by design (see the module docstring): this run does not
            # survive the session, so THIS EXPORT is the durable record of it.
            "ephemeral":       True,
        },
        "cluster_colouring_shown": bool(shown),
    }
