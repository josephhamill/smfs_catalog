# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Regression test: the packaged app behaves like the app (#111).

WHY THIS EXISTS.  A PyInstaller bundle runs the same source under three
conditions a checkout never has — no .git directory, no guarantee of a `git`
binary, and data files unpacked to sys._MEIPASS rather than sitting next to
the script.  Each of those silently changed a behaviour, and none of them is
visible from a build log: the build succeeds, the app launches, and the
damage only shows up on a real cohort on somebody else's machine.

The build carries `provenance.code_version()`, recording the exact application
used for provenance and cache validity without depending on Git being installed
on the recipient's machine.

The contract under test:
(a) frozen + a build stamp  -> its build identity.
(b) frozen + no build stamp -> None, never a shared constant.  Two different
    builds must not be able to agree on a version they didn't earn; None is
    the honest "I cannot tell you what made this" and merely costs speed.
(c) frozen NEVER falls through to git.  A machine running the bundle may well
    have git and may well be sitting in some unrelated repository, in which
    case `git rev-parse HEAD` succeeds and answers about THAT repo — a stamp
    with no relationship to the code executing.  Worse than None.
(d) a checkout still uses git and is completely unaffected, so installing
    this invalidates nothing already stored.
(e) resource_path honours sys._MEIPASS, so bundled assets are found.
(f) the xpra development helper does not run in a frozen build.  On Lemaitre
    it puts a headless session's GUI on display :100; on a colleague's
    machine the same code shells out to a program they do not have, from a
    double-clicked icon that shows no terminal to explain itself.
(g) nothing imports the old name.  code_version() was _git_hash() until this
    change; a stale caller would still resolve if the old name lingered
    anywhere, and would then be a second answer to "what made this result".
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from smfs_catalog import provenance as _provenance  # noqa: E402
from smfs_catalog import db as _db  # noqa: E402

_PKG = _ROOT / "smfs_catalog"


def _in_subprocess(body: str) -> str:
    """
    Run `body` in a fresh interpreter and return its stdout.

    A subprocess rather than monkeypatching, because code_version() is
    lru_cached and reads sys.frozen — a test that set the flag in-process
    would leak that state into whatever ran next, and this suite shares one
    interpreter with the rest of the app's imports.
    """
    src = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(_ROOT)!r})
        {textwrap.indent(textwrap.dedent(body), " " * 8).lstrip()}
    """)
    out = subprocess.run(
        [sys.executable, "-c", src], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


@pytest.fixture()
def stamp_file():
    """Write _build_stamp.py the way the release build does, then remove it."""
    p = _PKG / "_build_stamp.py"
    assert not p.exists(), "a real _build_stamp.py is present; not overwriting it"
    p.write_text(
        'BUILD_COMMIT = "cafebabe0000"\n',
        encoding="utf-8",
    )
    try:
        yield p
    finally:
        p.unlink(missing_ok=True)


# ── (a)/(b) what a frozen build stamps ───────────────────────────────────────

def test_frozen_with_a_stamp_reports_it(stamp_file):
    got = _in_subprocess("""
        sys.frozen = True
        from smfs_catalog.provenance import code_version
        print(code_version())
    """)
    assert got == "cafebabe0000"


def test_frozen_without_a_stamp_is_none_not_a_shared_constant():
    got = _in_subprocess("""
        sys.frozen = True
        from smfs_catalog.provenance import code_version
        print(repr(code_version()))
    """)
    assert got == "None"


def test_unknown_build_identity_disables_scientific_cache(tmp_path):
    db_path = str(tmp_path / "catalog.db")
    _db.initialise(db_path)
    _db.write_analysis_result(1, "event", 1.0, "{}", None, db_path)
    assert _db.get_analysis_result(1, "event", "{}", None, db_path) is None
    conn = _db.get_connection(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM analysis_results").fetchone()[0] == 0
    finally:
        conn.close()


def _cache_version_of(monkeypatch, code_version, digest):
    """cache_version() for a stated build identity and package-source digest."""
    monkeypatch.setattr(_provenance, "code_version", lambda: code_version)
    monkeypatch.setattr(_provenance, "_checkout_root", lambda: _ROOT)
    monkeypatch.setattr(_provenance, "_package_source_digest", lambda root: digest)
    _provenance.cache_version.cache_clear()
    try:
        return _provenance.cache_version()
    finally:
        _provenance.cache_version.cache_clear()


def test_clean_build_is_its_persistent_cache_identity(monkeypatch):
    assert _cache_version_of(monkeypatch, "abc123", None) == "abc123"


def test_dirty_checkout_is_identified_by_what_it_changed(monkeypatch):
    """A dirty tree earns an identity; it is never the bare commit."""
    got = _cache_version_of(monkeypatch, "abc123-dirty", "deadbeef1234")
    assert got is not None, "a dirty checkout must still be cacheable"
    assert got != "abc123", "edits must not be served under the commit they left"
    assert "deadbeef1234" in got


def test_different_dirty_trees_never_share_an_identity(monkeypatch):
    """
    The property that matters.  'abc123-dirty' was refused as an identity
    because two unrelated sets of edits produce that same string, and a
    result cached under one would be served as fresh to the other.  A digest
    of the changed package source cannot collide that way.
    """
    one = _cache_version_of(monkeypatch, "abc123-dirty", "1111aaaa2222")
    two = _cache_version_of(monkeypatch, "abc123-dirty", "3333bbbb4444")
    assert one != two


def test_same_dirty_tree_is_stable_across_processes(monkeypatch):
    """Cache reuse is worthless if the identity moves while the tree does not."""
    one = _cache_version_of(monkeypatch, "abc123-dirty", "1111aaaa2222")
    two = _cache_version_of(monkeypatch, "abc123-dirty", "1111aaaa2222")
    assert one == two


def test_an_edit_outside_the_package_keeps_the_commits_cache(monkeypatch):
    """
    Editing a test or a doc leaves the package source identical to HEAD.  The
    application never imports those, so they cannot change a result, and
    invalidating a whole cohort's cache over one would be a lie about what ran.
    """
    assert _cache_version_of(monkeypatch, "abc123-dirty", "") == "abc123"


def test_an_unreadable_tree_is_still_refused(monkeypatch):
    """No digest means no honest identity, and the cache stays off."""
    assert _cache_version_of(monkeypatch, "abc123-dirty", None) is None


def test_editing_the_package_moves_the_cache_identity():
    """
    End to end, against git rather than a stubbed digest: adding a file to
    the package must change the identity results are cached under, whatever
    the tree looked like when the suite started.
    """
    probe = _ROOT / "smfs_catalog" / "_dirty_probe.py"
    assert not probe.exists(), "leftover probe from an earlier run"

    def identity():
        _provenance.code_version.cache_clear()
        _provenance.cache_version.cache_clear()
        return _provenance.cache_version()

    try:
        before = identity()
        probe.write_text("# transient probe\n")
        after = identity()
    finally:
        probe.unlink(missing_ok=True)
        _provenance.code_version.cache_clear()
        _provenance.cache_version.cache_clear()

    assert after is not None, "a dirty checkout must still be cacheable"
    assert after != before, "an edited package was served the old identity"
    assert "+wt." in after
    assert identity() == before, "removing the edit did not restore the identity"


# ── (c) the dangerous fall-through ───────────────────────────────────────────

def test_frozen_never_answers_with_a_surrounding_repos_commit():
    """
    The process below runs inside this repository — a checkout with a real
    HEAD — while claiming to be frozen.  If code_version() fell through to
    git it would return that HEAD, which is the exact wrong answer a bundle
    launched from a developer's folder would give.
    """
    got = _in_subprocess("""
        import os
        os.chdir(%r)                     # a directory with a live .git
        sys.frozen = True
        from smfs_catalog.provenance import code_version
        print(repr(code_version()))
    """ % str(_ROOT))
    assert got == "None", (
        "a frozen build read the commit of whatever repository it happened to "
        "be standing in")


# ── (d) a checkout is untouched ──────────────────────────────────────────────

def test_a_checkout_still_stamps_the_real_commit():
    cv = _provenance.code_version()
    assert cv is not None, "running from a checkout, so git should answer"
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(_ROOT), text=True).strip()
    assert cv.startswith(head), f"{cv!r} does not start with HEAD {head!r}"


def test_a_checkout_identity_does_not_depend_on_process_cwd(tmp_path):
    got = _in_subprocess(f"""
        import os
        os.chdir({str(tmp_path)!r})
        from smfs_catalog.provenance import code_version
        print(code_version())
    """)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(_ROOT), text=True).strip()
    assert got.startswith(head), (
        "checkout identity followed the process working directory instead of "
        "the imported source tree"
    )


def test_a_checkout_ignores_a_stray_build_stamp(stamp_file):
    """
    Installing the stamp mechanism must not change a dev run — otherwise
    shipping it would invalidate every cached analysis in the live catalog.
    """
    got = _in_subprocess("""
        from smfs_catalog.provenance import code_version
        print(code_version())
    """)
    assert got != "cafebabe0000"
    assert len(got) >= 40, f"expected a git hash, got {got!r}"


# ── (e)/(f) the entry point under PyInstaller ────────────────────────────────

def test_resource_path_follows_meipass():
    got = _in_subprocess("""
        sys.frozen = True
        sys._MEIPASS = "/tmp/somewhere-else"
        import run_dashboard as rd
        print(rd.resource_path("smfs_catalog", "assets", "icons", "icon.png"))
    """)
    assert got == str(Path("/tmp/somewhere-else/smfs_catalog/assets/icons/icon.png"))


def test_resource_path_in_a_checkout_finds_the_real_icon():
    assert (_ROOT / "smfs_catalog" / "assets" / "icons" / "icon.png").exists()
    got = _in_subprocess("""
        import run_dashboard as rd
        p = rd.resource_path("smfs_catalog", "assets", "icons", "icon.png")
        print(p.exists())
    """)
    assert got == "True"


def test_frozen_build_does_not_run_the_xpra_helper():
    """
    _ensure_display() must return immediately when frozen.  Driven with
    DISPLAY unset on linux — the one combination that otherwise shells out to
    `xpra`, or sys.exit()s telling the user to install it.
    """
    got = _in_subprocess("""
        import os
        os.environ.pop("DISPLAY", None)
        import run_dashboard as rd
        sys.platform = "linux"
        sys.frozen = True
        sys._MEIPASS = "/tmp/nope"
        rd._ensure_display()
        print("returned")
    """)
    assert got == "returned"


# ── (g) one name for the stamp ───────────────────────────────────────────────

def test_no_module_still_calls_the_old_name():
    offenders = []
    for path in sorted(_PKG.glob("*.py")) + [_ROOT / "run_dashboard.py"]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "_git_hash":
                offenders.append(f"{path.name}:{node.lineno}")
            elif isinstance(node, ast.Attribute) and node.attr == "_git_hash":
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "_git_hash was renamed code_version because in a frozen build it is "
        "not a git hash; these still name it: " + ", ".join(offenders))
