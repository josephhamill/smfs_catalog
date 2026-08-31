# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""Glob patterns for selecting catalogued files by path.

The syntax is POSIX glob as :mod:`fnmatch` reads it -- ``*``, ``?``, ``[abc]``
and ``[!abc]`` -- the same vocabulary as the shell, so it is documented
everywhere and nothing here extends it.  Several patterns may be given at
once, separated by commas, and a file is in scope if any of them matches.
Matching runs against the whole stored path rather than the filename alone,
which leaves plain text working as a substring.
"""

from __future__ import annotations

import fnmatch
import functools
import re


@functools.lru_cache(maxsize=256)
def _compiled(pattern: str) -> tuple[re.Pattern, ...]:
    # fnmatch.translate anchors its pattern, so each is bracketed with stars to
    # keep the match unanchored.  Case folding is asked for here rather than
    # taken from fnmatch.fnmatch, whose folding follows the platform.
    return tuple(
        re.compile(fnmatch.translate(f"*{part}*"), re.IGNORECASE)
        for part in (p.strip() for p in pattern.split(",")) if part
    )


def path_matches(path: str | None, pattern: str | None) -> bool:
    """Whether any of ``pattern``'s comma-separated globs occurs in ``path``.

    Applied to the whole stored path, so a pattern reaches any ancestor folder
    as readily as the filename.  An empty pattern constrains nothing.
    """
    if not pattern:
        return True
    globs = _compiled(pattern)
    if not globs:
        return True
    if not path:
        return False
    return any(glob.match(path) for glob in globs)
