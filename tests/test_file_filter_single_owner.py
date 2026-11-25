# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Two source guards for the FileFilter change.

1. The dashboard scope -> filter mapping is written once, in
   `scope.scope_to_query`.  dashboard_window and date_picker_dialog call it;
   neither copies the mapping out by hand (`scope.get("users") or None`, ...).
2. The changed db functions and `scope_to_query` stay inside the CLAUDE.md
   function-size limits (50 lines, 10 decision points, nesting 4, 5 arguments),
   counted by the shared tests/size_limits.py.
"""
import ast
from pathlib import Path

import pytest

from size_limits import violations
from smfs_catalog import scope
from source_inspect import function_node

_PKG = Path(scope.__file__).parent
_FACET_KEYS = {"users", "analytes", "solvents", "techniques", "substrates",
               "sample_preps", "afm_units", "curve_types"}


def _tree(path):
    return ast.parse(Path(path).read_text(encoding="utf-8"))


def _functions(tree):
    return [n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _facet_key_reads(fn):
    """Distinct facet keys read as ``x.get("key")`` or ``x["key"]`` in `fn`."""
    keys = set()
    for n in ast.walk(fn):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "get" and n.args
                and isinstance(n.args[0], ast.Constant)):
            keys.add(n.args[0].value)
        elif (isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)):
            keys.add(n.slice.value)
    return keys & _FACET_KEYS


def _names_used(tree):
    return ({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)})


# ---- 1. the mapping is written once ------------------------------------------

@pytest.mark.parametrize("module", ["dashboard_window.py", "date_picker_dialog.py"])
def test_window_uses_scope_to_query_and_keeps_no_mapping_of_its_own(module):
    tree = _tree(_PKG / module)
    assert "scope_to_query" in _names_used(tree), (
        f"{module} should turn its scope into a filter with scope_to_query")
    for fn in _functions(tree):
        assert len(_facet_key_reads(fn)) < 3, (
            f"{module}:{fn.name} hand-copies the scope -> facet mapping; "
            "use scope.scope_to_query")


def test_no_module_but_scope_maps_scope_keys_to_filters():
    offenders = []
    for path in sorted(_PKG.glob("*.py")):
        if path.name == "scope.py":
            continue
        for fn in _functions(_tree(path)):
            if len(_facet_key_reads(fn)) >= 3:
                offenders.append(f"{path.name}:{fn.name}")
    assert not offenders, f"scope -> filter mapping copied outside scope.py: {offenders}"


# ---- 2. size guard ------------------------------------------------------------

_CHANGED = {
    "db.py": ("_file_filter_clauses", "list_files", "get_facet_options",
              "get_distinct_dates", "get_distinct_values",
              "classification_report_rows"),
    "scope.py": ("scope_to_query",),
}


def _changed_functions():
    out = []
    for module, names in _CHANGED.items():
        for name in names:
            out.append((f"{module}:{name}", function_node(_PKG / module, name)))
    return out


def test_the_changed_functions_exist():
    assert len(_changed_functions()) == 7


@pytest.mark.parametrize("label, fn", _changed_functions(), ids=[l for l, _ in _changed_functions()])
def test_changed_function_is_within_the_size_limits(label, fn):
    assert not violations(label, fn)
