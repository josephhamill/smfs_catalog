# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Temporary directories for test modules that build their catalogs at import
time.

Those modules run outside any fixture, so pytest's `tmp_path_factory` — which
cleans up after itself and keeps the last few runs — is not reachable from
them.  A bare `tempfile.mkdtemp()` there leaves its directory behind on every
run, for every run, forever.

`mkdtemp` here is that same call with the removal attached, via
`tempfile.TemporaryDirectory`, whose finalizer runs at interpreter exit.
Cleanup is best-effort: a directory still held open by a live sqlite
connection is left alone rather than failing the run.
"""
from __future__ import annotations

import tempfile

# The handles are kept only so they outlive the call and are finalized at exit
# rather than when the caller drops the path.
_HANDLES = []


def mkdtemp(prefix: str | None = None) -> str:
    """A temporary directory that removes itself when the session ends."""
    handle = tempfile.TemporaryDirectory(prefix=prefix, ignore_cleanup_errors=True)
    _HANDLES.append(handle)
    return handle.name
