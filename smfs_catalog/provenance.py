# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""Application identity used in diagnostics and export provenance.

This module answers which application build produced an artifact or cached
calculation, both from a source checkout and from a frozen release.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import subprocess
import sys

from . import __version__


def app_version() -> str:
    """Return the packaged application version."""
    return __version__


def _frozen_build_commit() -> str | None:
    """Return the commit baked into a frozen build, if it has one."""
    try:
        from ._build_stamp import BUILD_COMMIT
    except (ImportError, AttributeError):
        return None
    return BUILD_COMMIT or None


def _checkout_root() -> Path | None:
    """Return this package's repository root, or ``None`` when installed."""
    package_dir = Path(__file__).resolve().parent
    root = package_dir.parent
    if package_dir != root / "smfs_catalog" or not (root / ".git").exists():
        return None
    return root


@lru_cache(maxsize=1)
def code_version() -> str | None:
    """Return the identity of the running application build.

    A source checkout reports its current commit, suffixed with ``-dirty``
    when tracked or untracked files differ from that commit. A frozen build
    reports its baked-in commit. ``None`` means the identity cannot be
    established safely.

    The result is cached for the process lifetime. Development tools that need
    to observe a mid-session commit may call ``code_version.cache_clear()``.
    """
    if getattr(sys, "frozen", False):
        return _frozen_build_commit()

    root = _checkout_root()
    if root is None:
        return None

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None

    if not commit:
        return None
    return f"{commit}-dirty" if status else commit


def cache_version() -> str | None:
    """Return a build identity only when persistent cache reuse is safe."""
    identity = code_version()
    if identity is None or identity.endswith("-dirty"):
        return None
    return identity
