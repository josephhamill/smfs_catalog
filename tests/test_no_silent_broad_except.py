# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Guard test: no broad `except` swallows an error silently.

WHY THIS EXISTS.  Around twenty handlers of the form

    except Exception:
        return {}

were scattered through the windows.  Each turned a real failure into something
that looked like a normal empty state:

  * a failed freshness check showed "nothing stale";
  * a failed curve load showed "No curve";
  * a failed multi-event computation said "failed" with no error text;
  * a corrupt profile looked like "no experimentalist".

None of it reached the user or crash.log, so the first symptom was a wrong
screen and the cause was unrecoverable.  Unit tests cannot reach this in a GUI
app (see test_names_are_imported.py for why), so this guard is source-level.

THE RULE.  A handler is flagged when it catches Exception or BaseException
and does none of:

  1. re-raise: contains a `raise` anywhere in its body (bare `raise`, or
     raising a different error, which is still visible);
  2. use the caught exception: it binds `as exc` AND `exc` is read in the
     body (passed to report_handled / log_handled, formatted, returned...).
     `except Exception as exc: pass` binds but never uses it, so it is flagged;
  3. carry a reason on its `except` line:
         except Exception:  # noqa: BLE001  shutdown, nothing to report to
     The code BLE001 must be followed by a non-empty reason.  A bare
     `# noqa: BLE001` is flagged: the reason is the point.

WHAT COUNTS AS "BROAD".  `except Exception`, `except BaseException`, a bare
`except:`, and tuples containing either (`except (OSError, Exception)` is as
broad as `except Exception`).  Narrow handlers (`except ValueError`) are never
flagged: they are the fix, not the problem.  Only plain names are recognised
(`Exception`, not `builtins.Exception`); nobody writes the latter here.

Scope: every module in smfs_catalog/ and scripts/.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BROAD = {"Exception", "BaseException"}

_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)

# The marker comment: hash, the word noqa, a colon, the code BLE001, then
# whitespace and at least one non-space reason character.
NOQA_WITH_REASON = re.compile(r"#\s*noqa:\s*BLE001\s+\S")


def _modules():
    return sorted(list((ROOT / "smfs_catalog").glob("*.py"))
                  + list((ROOT / "scripts").glob("*.py")))


def _handler_names(handler: ast.ExceptHandler) -> list[ast.expr]:
    """The expressions an `except` clause lists; [] for a bare `except:`."""
    t = handler.type
    if t is None:
        return []
    return t.elts if isinstance(t, ast.Tuple) else [t]


def _caught_ids(handler: ast.ExceptHandler) -> set[str]:
    return {n.id for n in _handler_names(handler) if isinstance(n, ast.Name)}


def _is_broad(handler: ast.ExceptHandler) -> bool:
    return handler.type is None or bool(_caught_ids(handler) & BROAD)


def _raised_name(raise_node: ast.Raise) -> str | None:
    """Class name a raise statement raises; None for a bare `raise`."""
    exc = raise_node.exc
    if isinstance(exc, ast.Call):
        exc = exc.func
    return exc.id if isinstance(exc, ast.Name) else None


def _handler_catches(handler: ast.ExceptHandler, raised: str | None) -> bool:
    return (_is_broad(handler) or raised is None
            or raised in _caught_ids(handler))


def _raise_is_caught(raise_node: ast.Raise, handlers: list[ast.ExceptHandler]) -> bool:
    """Would one of an inner try's own handlers catch this raise?"""
    raised = _raised_name(raise_node)
    return any(_handler_catches(h, raised) for h in handlers)


def _swallowed(stmt: ast.stmt, handlers: list[ast.ExceptHandler]) -> bool:
    """A raise statement that one of the inner try's handlers catches."""
    return isinstance(stmt, ast.Raise) and _raise_is_caught(stmt, handlers)


def _live_raises(nodes) -> bool:
    """A `raise` that can escape: skips nested scopes (def, lambda, class) and
    raises that an inner try/except in the same body swallows."""
    for node in nodes:
        if isinstance(node, _SCOPES):
            continue
        if isinstance(node, ast.Raise):
            return True
        if isinstance(node, ast.Try):
            body = [n for n in node.body if not _swallowed(n, node.handlers)]
            rest = node.handlers + node.orelse + node.finalbody
            if _live_raises(body) or _live_raises(rest):
                return True
        elif _live_raises(ast.iter_child_nodes(node)):
            return True
    return False


def _reraises(handler: ast.ExceptHandler) -> bool:
    return _live_raises(handler.body)


def _uses_exception(handler: ast.ExceptHandler) -> bool:
    if not handler.name:
        return False
    return any(isinstance(n, ast.Name) and n.id == handler.name
               and isinstance(n.ctx, ast.Load)
               for stmt in handler.body for n in ast.walk(stmt))


def _is_visible(handler: ast.ExceptHandler, lines: list[str]) -> bool:
    return (_reraises(handler) or _uses_exception(handler)
            or bool(NOQA_WITH_REASON.search(lines[handler.lineno - 1])))


class _Finder(ast.NodeVisitor):
    def __init__(self, lines: list[str]):
        self.lines = lines
        self.scope: list[str] = []
        self.flagged: list[tuple[int, str]] = []

    def _enter(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _enter

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        if _is_broad(node) and not _is_visible(node, self.lines):
            self.flagged.append((node.lineno, ".".join(self.scope) or "<module>"))
        self.generic_visit(node)


def silent_broad_handlers(source: str) -> list[tuple[int, str]]:
    """(line, enclosing function) of every silent broad handler in `source`."""
    finder = _Finder(source.splitlines())
    finder.visit(ast.parse(source))
    return finder.flagged


@pytest.mark.parametrize("path", _modules(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_no_silent_broad_except_in_module(path: Path):
    flagged = silent_broad_handlers(path.read_text(encoding="utf-8"))
    listing = "\n".join(f"  {path.name}:{ln} in {fn}" for ln, fn in flagged)
    assert not flagged, (
        f"silent broad except handler(s):\n{listing}\n"
        "Surface it (qt_utils.report_handled), log it (crashlog.log_handled), "
        "narrow it to the errors you expect, or mark the except line "
        "`# noqa: BLE001  <reason>`."
    )


# --------------------------------------------------------------------------
# Self-tests: the checker itself, on small source strings.
# --------------------------------------------------------------------------

def _wrap(handler_src: str) -> str:
    body = "\n".join("    " + ln for ln in handler_src.splitlines())
    return f"def f():\n    try:\n        g()\n{body}\n"


def test_the_guard_would_have_caught_the_bug_it_was_written_for():
    """The original defect: swallow everything, return an empty value."""
    src = "class W:\n    def freshness(self):\n        try:\n            g()\n" \
          "        except Exception:\n            return {}\n"
    assert silent_broad_handlers(src) == [(5, "W.freshness")]


@pytest.mark.parametrize("handler", [
    "except Exception:\n    return {}",
    "except BaseException:\n    pass",
    "except:\n    pass",
    "except (OSError, Exception):\n    pass",
    "except Exception as exc:\n    pass",                 # binds, never uses
    "except Exception:  # noqa: BLE001\n    pass",        # no reason
    "except Exception:  # noqa: BLE001   \n    pass",     # blank reason
    "except Exception:  # noqa: E722  shutdown\n    pass",  # wrong code
    "except Exception:\n    # noqa: BLE001  reason on the wrong line\n    pass",
])
def test_flags_silent_handlers(handler):
    assert len(silent_broad_handlers(_wrap(handler))) == 1


@pytest.mark.parametrize("handler", [
    "except Exception as exc:\n    report_handled(self._lbl.setText, 'Freshness', exc)\n    return {}",
    "except Exception as exc:\n    log_handled('x', exc)",
    "except Exception:\n    raise",
    "except Exception as exc:\n    raise RuntimeError('x') from exc",
    "except BaseException:  # noqa: BLE001  shutdown\n    pass",
    "except Exception:  # noqa: BLE001  parked: D8 awaiting decision\n    pass",
    "except:  # noqa: BLE001  last resort\n    pass",
    "except ValueError:\n    pass",                        # narrow, not broad
    "except (OSError, ValueError):\n    return None",      # narrow tuple
])
def test_passes_visible_or_justified_handlers(handler):
    assert silent_broad_handlers(_wrap(handler)) == []


def test_reports_each_handler_with_its_line_and_function():
    src = (
        "def a():\n"
        "    try:\n        g()\n"
        "    except Exception:\n        pass\n"
        "def b():\n"
        "    try:\n        g()\n"
        "    except Exception as e:\n        print(e)\n"
        "    try:\n        g()\n"
        "    except BaseException:\n        pass\n"
    )
    assert silent_broad_handlers(src) == [(4, "a"), (13, "b")]


@pytest.mark.parametrize("handler", [
    # raise only inside a nested scope: never runs when the handler runs
    "except Exception:\n    def later():\n        raise RuntimeError('x')",
    "except Exception:\n    cb = lambda: (_ for _ in ()).throw(ValueError())",
    "except Exception:\n    class E:\n        def m(self):\n            raise ValueError",
    # raise swallowed by an inner try/except in the same body
    "except Exception:\n    try:\n        raise ValueError('x')\n    except ValueError:\n        pass",
    "except Exception:\n    try:\n        raise ValueError\n    except (OSError, ValueError):\n        pass",
])
def test_raise_in_nested_scope_or_swallowed_does_not_make_handler_visible(handler):
    assert len(silent_broad_handlers(_wrap(handler))) == 1


@pytest.mark.parametrize("handler", [
    # an inner try that does not catch the raise still lets it escape
    "except Exception:\n    try:\n        raise ValueError\n    except OSError:\n        pass",
    # a raise inside the inner handler itself escapes
    "except Exception:\n    try:\n        g()\n    except OSError:\n        raise",
    # a raise nested in an if still counts
    "except Exception:\n    if g():\n        raise",
])
def test_live_raise_inside_nested_statements_still_counts(handler):
    assert silent_broad_handlers(_wrap(handler)) == []
