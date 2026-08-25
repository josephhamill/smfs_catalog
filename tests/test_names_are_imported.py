# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Guard test: every module imports what it names.

WHY THIS EXISTS.  Three windows called `_quant.configure_spinbox`
while only seven of the ten modules using it had written
`from . import quantities as _quant`.  Nothing caught it:

  * the modules IMPORT fine — a NameError inside a method only fires when that
    method runs, so `import smfs_catalog.pca_window` succeeds;
  * no test constructs these windows (that needs a QApplication, a populated DB
    and real curve files — see test_numeric_ui.py's header for why the guards
    here are source-level);
  * so the first thing to run the code was the user, clicking Run PCA, then
    Grid settings on the physical 2DH, then the FFT notch controls.

Three separate crashes in one session, from one missing line, each invisible
until a human opened that exact widget.  This is the cheapest possible check
for a whole class of bug that unit tests structurally cannot reach in a GUI
app: a name that is READ somewhere in a file and BOUND nowhere in it.

Deliberately conservative.  A name bound anywhere in the file — any function's
local, any loop variable, any import inside any `if` — counts as bound
everywhere in it.  That is not how Python scoping works, and it means this test
will not catch a genuine scope error.  It is not trying to: it is trying to
never cry wolf, because a guard test with an allowlist bolted onto it means
nothing (same reasoning as test_style_is_single_source.py's six-digit-hex-only
rule).  What survives that filter is a name the file never binds at all, which
is a missing import essentially every time.
"""

from __future__ import annotations

import ast
import builtins
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "smfs_catalog"

BUILTINS = set(dir(builtins))

# Names Python itself provides to a module or a method body.  __class__ is the
# implicit cell that a no-argument super() reads.
IMPLICIT = {
    "__name__", "__file__", "__doc__", "__spec__", "__package__",
    "__loader__", "__builtins__", "__debug__", "__class__", "__path__",
}


def _modules():
    return sorted(p for p in PKG.glob("*.py"))


def _bound_names(tree: ast.AST) -> set[str]:
    """Every name the file binds, anywhere, by any means."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                # `import a.b.c` binds `a`; `import a.b as ab` binds `ab`.
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.Global):
            bound.update(node.names)
        elif isinstance(node, ast.Nonlocal):
            bound.update(node.names)
        elif isinstance(node, ast.MatchAs) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.MatchStar) and node.name:
            bound.add(node.name)
    return bound


def _star_imported(tree: ast.AST) -> bool:
    return any(alias.name == "*"
               for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
               for alias in node.names)


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_every_name_read_is_bound_somewhere_in_its_own_file(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))

    # `from x import *` makes the bound set unknowable without importing x.
    if _star_imported(tree):
        pytest.skip("star import — bound names not statically knowable")

    bound = _bound_names(tree) | BUILTINS | IMPLICIT
    read = {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}

    missing = sorted(read - bound)
    assert not missing, (
        f"{path.name} reads {missing} but never binds them — a missing import.\n"
        "This fires at runtime only when the enclosing function runs, which in "
        "this app means when a user opens that window."
    )


def test_the_guard_would_have_caught_the_bug_it_was_written_for():
    """A guard nobody validates is just a second thing to be wrong.

    Re-create the exact defect — a module-alias call with no import —
    and assert the checker flags it, so a future refactor of _bound_names that
    quietly stops detecting anything fails here rather than passing silently
    across the whole package.
    """
    broken = ast.parse(
        "from . import style\n"
        "class W:\n"
        "    def build(self):\n"
        "        _quant.configure_spinbox(self._spin)\n"
    )
    bound = _bound_names(broken) | BUILTINS | IMPLICIT
    read = {n.id for n in ast.walk(broken)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    assert "_quant" in read - bound

    fixed = ast.parse(
        "from . import style\n"
        "from . import quantities as _quant\n"
        "class W:\n"
        "    def build(self):\n"
        "        _quant.configure_spinbox(self._spin)\n"
    )
    bound = _bound_names(fixed) | BUILTINS | IMPLICIT
    read = {n.id for n in ast.walk(fixed)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    assert not (read - bound)
