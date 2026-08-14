# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
One idiom for the procedural guard tests (#84, 2026-08-07).

WHAT WAS WRONG.  This suite grew two ways of writing a test.  Most files use
ordinary pytest functions.  Ten grew up as standalone scripts instead — a
module-level run of `check(label, cond)` calls appending to a `failures` list,
ending in `sys.exit(1)`.  Both styles were run by the same `pytest tests/`
command, and the second one behaved badly under it in three ways:

  1. `sys.exit(1)` during collection is not a test failure, it is an
     INTERNALERROR.  **pytest stops the entire session on the spot**, so one
     broken guard hid the results of every file after it.  That is the exact
     inverse of what a regression suite is for.
  2. They contributed ZERO named tests, so "672 passed" silently did not
     count ten files — including the guards for the export convention, profile
     isolation, criteria isolation and curve qualification, which are the ones
     CLAUDE.md leans on hardest.
  3. A whole-file pass/fail told you nothing about WHICH check broke without
     scrolling the captured output.

WHAT THIS DOES.  `check` becomes a CheckRunner instance with the same call
signature, so the bodies of those files are untouched — they still read as a
narrative, which is their virtue and why they were not rewritten into fifty
fixtures.  It records each result instead of exiting.  The file then ends
with one line:

    test_check = checkstyle.pytest_cases(check)

which turns every recorded check into its own named, counted pytest case.  A
failure is now an ordinary failure: the run continues, the report names the
check, and the other nine files still report.

KNOWN AND DELIBERATE: the checks still execute at IMPORT time, i.e. during
collection, not inside the test functions.  Moving them would mean indenting
each file's whole body into a function, and these files interleave
module-level fixtures with the checks that read them — a mechanical reindent
would break that silently, and these are the guards protecting everything
else.  The cost is that `--collect-only` does real work and `-k` does not
skip it.  The cost of the alternative is worse.  If a file is ever rewritten
into real fixtures, it should stop importing this module rather than keep a
foot in both camps.
"""
from __future__ import annotations

import pytest


class CheckRunner:
    """
    Drop-in replacement for the `check(label, cond)` these files defined.

    Accepts the optional third `detail` argument test_export_convention.py
    used, so one runner serves all ten without their bodies changing.
    """

    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def __call__(self, label: str, cond, detail: str = "") -> None:
        ok = bool(cond)
        print(("PASS  " if ok else "FAIL  ") + label)
        self.results.append((label, ok, detail))

    @property
    def failures(self) -> list[str]:
        """Kept because a few files print or branch on this themselves."""
        return [lbl for lbl, ok, _ in self.results if not ok]


def pytest_cases(runner: CheckRunner):
    """
    Build a parametrized test function from everything `runner` recorded.

    Assign the result to a module-level name beginning with `test_`, as the
    LAST statement in the file — the parameters are read at decoration time,
    so every check must already have run.
    """
    params = [
        pytest.param(ok, detail, id=label)
        for label, ok, detail in runner.results
    ]

    @pytest.mark.parametrize("ok,detail", params)
    def test_check(ok, detail):
        assert ok, detail or "check failed"

    if not params:
        # A file that recorded nothing is a file whose body silently did not
        # run — exactly the invisibility this module exists to remove, so it
        # must fail rather than report a tidy zero.
        def test_check():                                   # noqa: F811
            raise AssertionError(
                "no checks were recorded — the module body did not run")

    return test_check
