# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Guard test: the declared version never lags the released one (2026-08-08).

WHAT WENT WRONG, and it is worth stating plainly because the existing guard
looked like it covered this.

  2026-05-22   v1.0.0 tagged
  2026-07-08   v1.1.0 tagged
  2026-07-31   pyproject.toml and __init__.py created -- BOTH SET TO 1.0.0

The version declarations were not merely allowed to drift.  They were written
three weeks after v1.1.0 was released and set to the version before it, so
they were wrong on the day they were created and stayed wrong for every
release, export and figure afterwards.

That is not cosmetic.  smfs_catalog.__version__ is written into EVERY export
manifest, and the manifest exists precisely so a colleague opening a CSV
months later -- with no access to the app or to this repository -- can tell
which code produced the numbers.  Every export written between 2026-07-31 and
2026-08-08 names a version from May.

WHY test_export_convention.py DID NOT CATCH IT.  It checks that pyproject.toml
and __init__.py agree WITH EACH OTHER.  They did: both said 1.0.0.  A
consistency check between two copies of the same wrong answer passes happily.
The missing check is against something OUTSIDE the pair -- the git tags, which
are what "released" actually means here.

THE RULE: the declared version is >= the newest v* tag, and EQUAL to it when
HEAD is that tag.  Deliberately >= rather than == in general, because between
releases the declared version legitimately runs ahead: it names the release
being prepared, so that when the tag is finally cut, the tag, __version__ in
every manifest, and the frozen build's own stamp all name the same code.  A
strict == would fail every commit after a release and be disabled within a
week.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT              # the package sits at the repo root; kept as two names
                           # because one asks "where is the code" and the other
                           # "where is git", and they were separate directories
                           # until the smfs_catalog_v1/ level was removed.
sys.path.insert(0, str(_ROOT))

from smfs_catalog import __version__ as PKG_VERSION      # noqa: E402

_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def _parse(text: str) -> tuple[int, int, int]:
    parts = text.split(".")
    assert len(parts) == 3, f"not a three-part version: {text!r}"
    return tuple(int(p) for p in parts)          # type: ignore[return-value]


def _git(*args: str) -> str | None:
    """Run a git command in the repo; None when git or the repo is unavailable."""
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(_REPO), capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _release_tags() -> list[tuple[tuple[int, int, int], str]]:
    listing = _git("tag", "--list", "v*")
    if listing is None:
        return []
    found = []
    for line in listing.splitlines():
        m = _TAG_RE.match(line.strip())
        if m:
            found.append((tuple(int(g) for g in m.groups()), line.strip()))
    return sorted(found)


@pytest.fixture(scope="module")
def declared() -> tuple[int, int, int]:
    return _parse(PKG_VERSION)


def test_pyproject_and_package_agree(declared):
    """
    The check that already existed, kept here so this file states the whole
    rule in one place.  It is necessary and it is not sufficient -- on its own
    it passed for three months while both copies were wrong.
    """
    data = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert _parse(data["project"]["version"]) == declared


def test_declared_version_is_not_behind_the_newest_release_tag(declared):
    tags = _release_tags()
    if not tags:
        pytest.skip("no v* tags visible (shallow clone or no git) — nothing to compare")
    newest, name = tags[-1]
    assert declared >= newest, (
        f"the code declares {PKG_VERSION} but {name} has already been released. "
        f"__version__ is stamped into every export manifest, so exports would "
        f"claim a version older than the one people already have. Set "
        f"pyproject.toml and smfs_catalog/__init__.py to at least "
        f"{'.'.join(map(str, newest))}.")


def test_a_tagged_commit_declares_exactly_its_own_tag(declared):
    """
    On a release commit the two must be identical — that is the moment the
    tag, the manifests and the frozen build's stamp are supposed to name one
    code state.  Off a tag this is silent, because running ahead is correct.
    """
    exact = _git("tag", "--points-at", "HEAD")
    if not exact:
        pytest.skip("HEAD is not a tagged release")
    here = [t for t in (_TAG_RE.match(l.strip()) for l in exact.splitlines()) if t]
    if not here:
        pytest.skip("HEAD carries no v* release tag")
    tagged = max(tuple(int(g) for g in m.groups()) for m in here)
    assert declared == tagged, (
        f"HEAD is tagged v{'.'.join(map(str, tagged))} but the code declares "
        f"{PKG_VERSION}. Build this commit and every manifest it writes names "
        f"the wrong release.")


def test_version_is_reachable_without_git_or_package_metadata():
    """
    __version__ must work from a checkout, an install AND a PyInstaller
    bundle, which is why it is a literal rather than read from pyproject.toml
    or from importlib.metadata at runtime.  A frozen build ships neither.
    """
    src = (_ROOT / "smfs_catalog" / "__init__.py").read_text(encoding="utf-8")
    assert re.search(r'^__version__\s*=\s*"[\d.]+"', src, re.M), (
        "__version__ must stay a plain literal — a frozen build has no "
        "pyproject.toml and no package metadata to read it from")
