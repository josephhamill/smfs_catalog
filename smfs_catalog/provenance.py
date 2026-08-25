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
import hashlib
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


def _package_source_digest(root: Path) -> str | None:
    """Digest the package source that differs from HEAD.

    Returns "" when the package source matches HEAD exactly (an edit
    elsewhere in the tree — a test, a doc — cannot change a result, because
    the application never imports it), a short hex digest when it differs,
    and None when the tree cannot be read.
    """
    pkg = "smfs_catalog"
    try:
        diff = subprocess.run(
            ["git", "diff", "HEAD", "--", pkg],
            cwd=root, check=True, capture_output=True,
        ).stdout
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "--", pkg],
            cwd=root, check=True, capture_output=True, text=True,
        ).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None

    if not diff and not untracked:
        return ""

    digest = hashlib.sha256(diff)
    for rel in sorted(untracked):
        try:
            content = (root / rel).read_bytes()
        except OSError:
            return None
        digest.update(rel.encode())
        digest.update(content)
    return digest.hexdigest()[:12]


@lru_cache(maxsize=1)
def cache_version() -> str | None:
    """Return the identity under which this build's results may be cached.

    A dirty checkout gets a real identity rather than being refused one: the
    commit plus a digest of the package source that differs from it. That is
    unique to the tree state, which is what cache reuse actually requires —
    two different sets of edits can never agree on an identity, so a result
    is only ever served back to the code that produced it.

    Pinned for the process lifetime, like code_version(). The identity must
    describe the code THIS process loaded, not the files now on disk; editing
    a module mid-session does not change what is already imported and running.
    """
    identity = code_version()
    if identity is None:
        return None
    if not identity.endswith("-dirty"):
        return identity

    root = _checkout_root()
    if root is None:
        return None
    digest = _package_source_digest(root)
    if digest is None:
        return None

    commit = identity[: -len("-dirty")]
    return commit if digest == "" else f"{commit}+wt.{digest}"
