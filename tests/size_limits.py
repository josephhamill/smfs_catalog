# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
The CLAUDE.md function-size rule, counted one way, for every source-size guard.

A function is within the limits when it has at most 50 lines, 10 decision
points (each if / for / while / conditional expression / except / comprehension /
match case, plus one per extra operand of `and` / `or`), nesting at most 4 deep
(if / for / while / try / with) and at most 5 arguments (`*args` and `**kwargs`
count one each).  `violations(name, fn)` returns one message per limit broken.
"""
import ast

_DECISION_NODES = (ast.If, ast.For, ast.While, ast.IfExp, ast.ExceptHandler,
                   ast.comprehension, ast.match_case)
_NEST_NODES = (ast.If, ast.For, ast.While, ast.Try, ast.With)


def decision_points(fn) -> int:
    n = 0
    for node in ast.walk(fn):
        if isinstance(node, _DECISION_NODES):
            n += 1
        elif isinstance(node, ast.BoolOp):
            n += len(node.values) - 1
    return n


def depth(node, d=0) -> int:
    deepest = d
    for child in ast.iter_child_nodes(node):
        deepest = max(deepest, depth(child, d + 1 if isinstance(child, _NEST_NODES) else d))
    return deepest


def arg_count(fn) -> int:
    a = fn.args
    return len(a.posonlyargs) + len(a.args) + len(a.kwonlyargs) + bool(a.vararg) + bool(a.kwarg)


def violations(name, fn) -> list:
    found = []
    if fn.end_lineno - fn.lineno + 1 > 50:
        found.append(f"{name}: more than 50 lines")
    if decision_points(fn) > 10:
        found.append(f"{name}: more than 10 decision points")
    if depth(fn) > 4:
        found.append(f"{name}: nesting deeper than 4")
    if arg_count(fn) > 5:
        found.append(f"{name}: more than 5 arguments")
    return found
