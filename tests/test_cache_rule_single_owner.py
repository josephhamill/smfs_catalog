# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
"When can a CacheContext be used?" has one owner: curve_analysis.py, next to
CacheContext.

The rule (db_path, code_ver and file_id are all set) is written once, as
curve_analysis.can_use_cache, and the cache reads and writes built on it are
shared too: curve_analysis.cache_get and cache_put_multi.  roi_pipeline.py uses
them and holds no copy of the rule and no direct scalar-cache db calls, so a
change to the rule is made in one place and the two modules cannot disagree
about what gets cached.

Checked on the source (AST), so it does not depend on the shared predicate's
name: it may be a CacheContext method or property, or a module function.
"""
import ast
from pathlib import Path

from smfs_catalog import curve_analysis, roi_pipeline

_IDENTITY_FIELDS = {"db_path", "code_ver", "file_id"}
# The db calls that read or write the scalar analysis-result cache.
_SCALAR_CACHE_DB_CALLS = {"get_analysis_result", "write_analysis_results_multi"}


def _tree(module):
    return ast.parse(Path(module.__file__).read_text(encoding="utf-8"))


def _functions(tree):
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _checks_cache_identity(fn) -> bool:
    """True if fn's body tests `<x>.db_path / code_ver / file_id is not None`."""
    for node in ast.walk(fn):
        if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Attribute)
                and node.left.attr in _IDENTITY_FIELDS
                and any(isinstance(op, (ast.IsNot, ast.Is)) for op in node.ops)):
            return True
    return False


def _owners_in_curve_analysis():
    return [f for f in _functions(_tree(curve_analysis)) if _checks_cache_identity(f)]


def test_roi_pipeline_does_not_restate_the_cache_usability_rule():
    found = [f.name for f in _functions(_tree(roi_pipeline)) if _checks_cache_identity(f)]
    assert found == []


def test_curve_analysis_states_the_rule_exactly_once():
    assert len(_owners_in_curve_analysis()) == 1


def test_roi_pipeline_uses_the_rule_owned_by_curve_analysis():
    owners = {f.name for f in _owners_in_curve_analysis()}
    used = set()
    for node in ast.walk(_tree(roi_pipeline)):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
    assert owners & used, f"roi_pipeline never uses {sorted(owners)} from curve_analysis"


def test_roi_pipeline_does_not_access_the_scalar_cache_itself():
    """Scalar-cache reads and writes go through curve_analysis' helpers."""
    direct = sorted({n.func.attr for n in ast.walk(_tree(roi_pipeline))
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and n.func.attr in _SCALAR_CACHE_DB_CALLS})
    assert direct == []
