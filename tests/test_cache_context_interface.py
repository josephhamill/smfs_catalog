# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Pass-1 step c3: the CacheContext parameter object and the new
compute_curve_events(_coords) signatures, plus size limits on the functions c3
creates or changes.

CacheContext lives in smfs_catalog.curve_analysis (next to Stage1Search and
pipeline_params_from); roi_pipeline imports it, and curve_analysis only imports
roi_pipeline lazily inside a function, so there is no import cycle.
"""
import ast
import dataclasses
import inspect
from pathlib import Path

import pytest

from smfs_catalog import roi_pipeline
from smfs_catalog.curve_analysis import CacheContext
from size_limits import violations

OLD_CACHE_KEYWORDS = ("db_path", "code_ver", "file_id", "param_set", "conn")


def test_cache_context_is_a_frozen_dataclass_with_the_planned_fields():
    assert dataclasses.is_dataclass(CacheContext)
    assert [f.name for f in dataclasses.fields(CacheContext)] == [
        "db_path", "code_ver", "file_id", "conn", "param_set"]
    ctx = CacheContext("x.db", "v1", 7, None)
    assert ctx.param_set is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.file_id = 8


def test_cache_context_keeps_the_given_values():
    marker = object()
    ctx = CacheContext("x.db", "v1", 7, marker, param_set="ps")
    assert (ctx.db_path, ctx.code_ver, ctx.file_id, ctx.conn, ctx.param_set) == (
        "x.db", "v1", 7, marker, "ps")


@pytest.mark.parametrize("name", ["compute_curve_events", "compute_curve_events_coords"])
def test_compute_functions_take_cache_stage1_force_keyword_only(name):
    params = inspect.signature(getattr(roi_pipeline, name)).parameters
    assert list(params) == ["curve", "ep", "cache", "stage1", "force"]
    for kw in ("cache", "stage1", "force"):
        assert params[kw].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["cache"].default is None
    assert params["stage1"].default is None
    assert params["force"].default is False
    assert len(params) <= 5


@pytest.mark.parametrize("name", ["compute_curve_events", "compute_curve_events_coords"])
@pytest.mark.parametrize("old", OLD_CACHE_KEYWORDS)
def test_old_cache_keywords_are_gone(name, old):
    assert old not in inspect.signature(getattr(roi_pipeline, name)).parameters


def test_old_cache_keywords_are_rejected_at_call_time():
    with pytest.raises(TypeError):
        roi_pipeline.compute_curve_events(
            object(), object(), db_path="x.db", code_ver="v", file_id=1)


# ── source-size guard (CLAUDE.md limits), scoped to what c3 touches ──────────

C3_FUNCTIONS = {"compute_curve_events", "compute_curve_events_coords"}
# Every top-level function roi_pipeline.py had before c3.  Any other top-level
# function is a helper c3 introduced and is held to the same limits.
PRE_C3_FUNCTIONS = {
    "assemble_rows", "compute_curve_events", "compute_curve_events_coords",
    "coverage", "event_map_params_json", "event_params_from",
    "populate_event_map", "roi_count_histogram", "segment_summary_bulk",
}


def _functions() -> dict:
    tree = ast.parse(Path(roi_pipeline.__file__).read_text(encoding="utf-8"))
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _new_helpers() -> dict:
    return {n: f for n, f in _functions().items() if n not in PRE_C3_FUNCTIONS}


def test_compute_curve_events_coords_is_split_into_named_stages():
    assert len(_new_helpers()) >= 4


@pytest.mark.parametrize("name", sorted(C3_FUNCTIONS))
def test_c3_function_is_within_the_size_limits(name):
    assert violations(name, _functions()[name]) == []


def test_every_new_helper_is_within_the_size_limits():
    found = [v for n, f in _new_helpers().items() for v in violations(n, f)]
    assert found == []
