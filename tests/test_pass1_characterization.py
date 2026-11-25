# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Pass-1 characterization: the curve functions give exactly the recorded output.

The snapshot (tests/snapshots/pass1_characterization.json) was generated on the
code before the c3-c7 refactors by tests/pass1_snapshot.py.  The rule is exact
equality, NaN equal to NaN, floats compared as repr (17 digits): see that
module's docstring for what is recorded and what is left out.
"""
import json

import pytest

import pass1_snapshot as snap
from smfs_catalog import db


@pytest.fixture(scope="module")
def collected(tmp_path_factory):
    """Everything recorded now: one catalog, all 15 cases, the summaries."""
    path = str(tmp_path_factory.mktemp("pass1") / "catalog.db")
    db.initialise(path)   # the same throwaway catalog the `catalog` fixture builds
    return snap.to_json_data(snap.collect_all(path))


@pytest.fixture(scope="module")
def recorded():
    return json.loads(snap.SNAPSHOT_PATH.read_text(encoding="utf-8"))


def _assert_same(now, then):
    where = snap.first_difference(now, then)
    assert where is None, f"output changed (now != recorded) at {where}"


def test_the_snapshot_covers_every_case(collected, recorded):
    assert sorted(recorded["cases"]) == sorted(snap._case_labels())
    assert sorted(collected["cases"]) == sorted(recorded["cases"])


@pytest.mark.parametrize("label", snap._case_labels())
def test_case_output_is_unchanged(label, collected, recorded):
    _assert_same(collected["cases"][label], recorded["cases"][label])


def test_segment_summary_is_unchanged(collected, recorded):
    _assert_same(collected["summary"], recorded["summary"])


def test_cache_hit_returns_what_the_fresh_call_returned(collected):
    """Second (cached) call == first, for every case and both cached calls."""
    for label, out in collected["cases"].items():
        assert snap.first_difference(
            out["compute_cache_second"], out["compute_cache_first"]) is None, label
        assert snap.first_difference(
            out["compute_forced_on_conn"], out["compute_cache_first"]) is None, label


def test_calibrations_read_back_from_analyse_curve_rows_give_the_fresh_result(collected):
    """The real batch flow (offset, invOLS, snap-off all from cache) == no cache."""
    for label, out in collected["cases"].items():
        assert snap.first_difference(
            out["compute_after_analyse"], out["compute_no_cache"]) is None, label


@pytest.mark.parametrize("a, b", [
    (True, 1), (1, True), (False, 0), (None, 0), (0, None), ("1", 1), (None, "None"),
])
def test_the_comparison_tells_apart_values_that_are_equal_but_of_another_kind(a, b):
    assert snap.first_difference(a, b) is not None


def test_the_comparison_accepts_identical_leaves_and_nan_strings():
    same = {"dict": [["k", [True, 1, "x", None, {"float": "nan"}]]]}
    assert snap.first_difference(same, json.loads(json.dumps(same))) is None
