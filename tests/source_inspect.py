# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Source inspection shared by the source-guard tests: find one function's AST
node, and list the names a module imports.  Both read the file, never import
it, so a guard can check a module that must not (or cannot) be imported.
"""
import ast
from pathlib import Path


def function_node(path, name, cls=None):
    """The `def name` node at module level, or inside class `cls` when given.

    Raises AssertionError when it is not there, so a renamed function fails
    its guard loudly instead of being skipped.
    """
    scope = ast.parse(Path(path).read_text(encoding="utf-8")).body
    if cls is not None:
        scope = next((n.body for n in scope
                      if isinstance(n, ast.ClassDef) and n.name == cls), [])
    for node in scope:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    where = f"{cls}." if cls else ""
    raise AssertionError(f"{Path(path).name}: {where}{name} not found")


def imported_names(path) -> list:
    """Every module an `import` / `from ... import` names, dotted.

    `import a.b` gives "a.b"; `from a import b` gives "a" and "a.b".
    """
    names = []
    for node in ast.walk(ast.parse(Path(path).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names.append(mod)
            names += [f"{mod}.{a.name}" for a in node.names]
    return names
