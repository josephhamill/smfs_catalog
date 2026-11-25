# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Pass-1 step c4: analyse_curve takes (curve, p, *, cache=None) with a CacheContext
instead of four loose cache keywords, and is split into named stages.  Size
limits apply to analyse_curve and every top-level helper c4 introduces in
curve_analysis.py.
"""
import ast
import inspect
from pathlib import Path

import pytest

from smfs_catalog import curve_analysis
from smfs_catalog.curve_analysis import analyse_curve
from size_limits import violations

OLD_CACHE_KEYWORDS = ("db_path", "code_ver", "file_id", "conn", "param_set")


def test_analyse_curve_takes_curve_p_and_a_keyword_only_cache():
    params = inspect.signature(analyse_curve).parameters
    assert list(params) == ["curve", "p", "cache"]
    assert params["cache"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["cache"].default is None
    assert len(params) <= 5


@pytest.mark.parametrize("old", OLD_CACHE_KEYWORDS)
def test_old_cache_keywords_are_gone(old):
    assert old not in inspect.signature(analyse_curve).parameters


def test_old_cache_keywords_are_rejected_at_call_time():
    with pytest.raises(TypeError):
        analyse_curve(object(), object(), db_path="x.db", code_ver="v", file_id=1)


# ── source-size guard (CLAUDE.md limits), scoped to what c4 touches ──────────

# Every top-level function curve_analysis.py had before c4.  Any other
# top-level function is a helper c4 introduced and is held to the same limits.
PRE_C4_FUNCTIONS = {
    "pipeline_params_from", "analyse_curve", "current_signature",
    "_lane_no_pipeline_yet", "_lane_force_extension", "analyse_file",
    "_persist_multi_event_roi",
}


def _functions() -> dict:
    tree = ast.parse(Path(curve_analysis.__file__).read_text(encoding="utf-8"))
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _new_helpers() -> dict:
    return {n: f for n, f in _functions().items() if n not in PRE_C4_FUNCTIONS}


def test_analyse_curve_is_split_into_named_stages():
    assert len(_new_helpers()) >= 4


def test_analyse_curve_is_within_the_size_limits():
    assert violations("analyse_curve", _functions()["analyse_curve"]) == []


def test_every_new_helper_is_within_the_size_limits():
    found = [v for n, f in _new_helpers().items() for v in violations(n, f)]
    assert found == []
