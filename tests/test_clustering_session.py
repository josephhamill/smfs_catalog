# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Guards for the session's cluster assignment.

The load-bearing test is (a): the clustering must NOT be persisted anywhere.
Every other property here follows from that, and the pull to "just store it so
it survives a restart" is exactly what this design refuses — a k-means label is
a property of the SET, so a stored one describes a partition that no longer
exists the moment the cohort moves, and the queue is cleared at every launch.

Second is (b), the PC1 ordering: without it the same physical group gets a
different integer on every re-run and every colour in every window jumps.
"""

from __future__ import annotations

import numpy as np
import pytest

from smfs_catalog import clustering as _cl


@pytest.fixture(autouse=True)
def _clean_registry():
    _cl.clear()
    yield
    _cl.clear()


def _run(labels=None, paths=None, **kw) -> _cl.Clustering:
    labels = labels if labels is not None else [0, 1, 0, 1]
    paths = paths or [f"/tank/t/{i}.ibw" for i in range(len(labels))]
    kw.setdefault("k", len(set(labels)))
    kw.setdefault("seed", 42)
    kw.setdefault("n_pcs", 3)
    kw.setdefault("sklearn_version", "1.5.0")
    return _cl.Clustering(labels=dict(zip(paths, labels)), **kw)


# ── (a) THE ONE THAT MATTERS: nothing is persisted ───────────────────────────

def test_the_module_touches_no_database_and_no_file():
    """A cluster label is a property of the SET, not of a curve: the centroids
    are fixed by every member together, so a stored label describes a
    partition that stops existing the moment the cohort moves. Keeping one
    honest would mean tracking cohort membership, the 2DH settings, the
    analysis params, k, the seed, n_pcs and the code version — more inputs
    than the gate verdict had, and that was deleted for exactly this reason.
    """
    import inspect
    src = inspect.getsource(_cl)
    for forbidden in ("import sqlite3", "from . import db", "open(", "json.dump"):
        assert forbidden not in src, (
            f"clustering.py must not persist anything; found {forbidden!r}")


def test_a_fresh_session_has_no_clustering():
    """What the user sees the next day: nothing. Re-running costs seconds,
    and the export is the durable record."""
    assert _cl.current() is None
    assert _cl.coverage_text(["/tank/t/0.ibw"]) == ""


def test_clear_removes_it():
    _cl.set_current(_run())
    assert _cl.current() is not None
    _cl.clear()
    assert _cl.current() is None


def test_a_new_run_replaces_rather_than_accumulates():
    """Only the latest exploratory k-means run remains active."""
    _cl.set_current(_run(k=2))
    _cl.set_current(_run(labels=[0, 1, 2, 0], k=3))
    assert _cl.current().k == 3


# ── (b) label ordering is stable and means something ─────────────────────────

def test_clusters_are_renumbered_left_to_right_by_pc1():
    labels = np.array([2, 2, 0, 0, 1, 1])
    scores = np.array([[9.0], [8.0], [1.0], [2.0], [5.0], [4.0]])
    out = _cl.order_by_first_pc(labels, scores)
    assert out.tolist() == [2, 2, 0, 0, 1, 1]


def test_pc1_order_can_permute_other_cluster_indexed_results():
    """Exported centres must use the same numbering as labels and matrices."""
    labels = np.array([2, 2, 0, 0, 1, 1])
    scores = np.array([[1.0], [2.0], [5.0], [4.0], [9.0], [8.0]])
    centers_in_sklearn_order = np.array([[10.0], [20.0], [30.0]])

    order = _cl.first_pc_order(labels, scores)

    assert order.tolist() == [2, 0, 1]
    assert centers_in_sklearn_order[order].ravel().tolist() == [30.0, 10.0, 20.0]


def test_reordering_is_a_permutation_never_a_merge_or_split():
    """It must move labels between integers, never change which curves share
    one — that would silently alter the clustering itself."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, 200)
    scores = rng.normal(size=(200, 3))
    out = _cl.order_by_first_pc(labels, scores)
    assert len(set(map(int, out))) == len(set(map(int, labels)))
    for c in set(map(int, labels)):
        members = set(np.where(labels == c)[0])
        new = {int(v) for v in out[list(members)]}
        assert len(new) == 1, "a cluster's members must all get the same label"


def test_reordering_is_idempotent():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 4, 100)
    scores = rng.normal(size=(100, 2))
    once = _cl.order_by_first_pc(labels, scores)
    assert _cl.order_by_first_pc(once, scores).tolist() == once.tolist()


def test_reordering_survives_an_empty_run():
    assert _cl.order_by_first_pc(np.array([]), np.empty((0, 2))).size == 0


# ── (c) coverage is per-cohort, because the cohorts differ ───────────────────

def test_coverage_is_computed_for_the_cohort_on_screen():
    """The 2DH that produced the clustering holds ONE population; Explore
    Events shows both. A window must report its own coverage, not the run's
    total, or a half-coloured scatter reads as fully described."""
    _cl.set_current(_run(paths=["/a", "/b"], labels=[0, 1]))
    assert _cl.coverage_text(["/a", "/b", "/c", "/d"]) == "2 of 4 curves labelled"
    assert _cl.coverage_text(["/a"]) == "1 of 1 curves labelled"


def test_an_unlabelled_curve_reports_none_not_a_cluster():
    """It must be distinguishable from cluster 0 — otherwise every curve the
    clustering never saw joins the first cluster silently."""
    c = _run(paths=["/a"], labels=[0])
    assert c.label_for("/a") == 0
    assert c.label_for("/nope") is None


def test_live_labels_join_to_export_rows_without_inventing_cluster_zero():
    rows = [{"path": "/a", "value": 1}, {"path": "/outside", "value": 2}]
    assert _cl.labels_for_rows(rows) is rows
    _cl.set_current(_run(paths=["/a"], labels=[2]))
    joined = _cl.labels_for_rows(rows)
    assert joined[0]["cluster"] == 2
    assert joined[1]["cluster"] is None
    assert "cluster" not in rows[0], "joining must not mutate source rows"


# ── (d) the description names the source matrix ──────────────────────────────

def test_the_description_says_it_clustered_the_2dh_not_the_scatter():
    """A scalar scatter must state that clustering happened in 2DH space."""
    c = _run(source={"window": "physical", "align_mode": "rupture",
                     "segment": "last", "population": "hit"},
             created_at="14:32")
    text = c.describe()
    assert "k-means k=2" in text and "physical 2DH" in text
    assert "rupture" in text and "14:32" in text

    prov = _cl.provenance([], False)["clustering"]
    assert prov is None
    _cl.set_current(c)
    prov = _cl.provenance(["/tank/t/0.ibw"], True)
    assert prov["clustering"]["source_matrix"] == "2DH ensemble"
    assert prov["clustering"]["ephemeral"] is True
    assert prov["cluster_colouring_shown"] is True


def test_the_manifest_records_whether_the_colouring_was_actually_shown():
    """Separately from whether a clustering existed — a figure and its CSV
    must agree about what was drawn."""
    _cl.set_current(_run())
    assert _cl.provenance([], False)["cluster_colouring_shown"] is False
    assert _cl.provenance([], True)["cluster_colouring_shown"] is True


def test_provenance_says_whether_the_export_cohort_matches_the_clustered_cohort():
    _cl.set_current(_run(paths=["/a", "/b"], labels=[0, 1]))
    exact = _cl.provenance(["/b", "/a"], False)["clustering"]
    changed = _cl.provenance(["/a", "/c"], False)["clustering"]
    assert exact["cohort_matches_clustering"] is True
    assert changed["cohort_matches_clustering"] is False


def test_the_manifest_keys_are_present_even_with_no_clustering():
    """An absent key is indistinguishable from an older export that never had
    one."""
    m = _cl.provenance(["/a"], False)
    assert m["clustering"] is None and m["cluster_colouring_shown"] is False


# ── (e) subscribers ──────────────────────────────────────────────────────────

def test_subscribers_are_told_on_publish_and_on_clear():
    seen = []
    cb = lambda: seen.append(1)          # noqa: E731
    _cl.subscribe(cb)
    try:
        _cl.set_current(_run())
        _cl.clear()
        assert len(seen) == 2
    finally:
        _cl.unsubscribe(cb)


def test_one_failing_subscriber_does_not_stop_the_others():
    """A window failing to repaint must not take down the k-means run that
    published, nor stop the other windows being told."""
    seen = []

    def boom():
        raise RuntimeError("wrapped C/C++ object has been deleted")

    ok = lambda: seen.append(1)          # noqa: E731
    _cl.subscribe(boom)
    _cl.subscribe(ok)
    try:
        _cl.set_current(_run())
        assert seen == [1]
    finally:
        _cl.unsubscribe(boom)
        _cl.unsubscribe(ok)


def test_unsubscribe_is_required_and_works():
    seen = []
    cb = lambda: seen.append(1)          # noqa: E731
    _cl.subscribe(cb)
    _cl.unsubscribe(cb)
    _cl.set_current(_run())
    assert seen == []


def test_subscribing_twice_notifies_once():
    seen = []
    cb = lambda: seen.append(1)          # noqa: E731
    _cl.subscribe(cb)
    _cl.subscribe(cb)
    try:
        _cl.set_current(_run())
        assert seen == [1]
    finally:
        _cl.unsubscribe(cb)
