# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Regression test: the dashboard reclaims the xpra session it started.

WHY THIS EXISTS.  `_ensure_display()` started `xpra start :100` for headless
runs and never stopped it.  The server and its Xvfb then outlived the
application indefinitely — measured at ~300 MB resident plus ~1 GB paged to
swap, held for twenty hours by a session with no application windows left in
it.  Nothing about that is visible from the app: it exits cleanly, and the
cost lands on the machine as general slowness hours later.

Detaching the viewer does not reclaim it and is not supposed to; surviving a
client disconnect is what a persistent display is for.  Only an explicit
`xpra stop` ends the session, and nothing issued one.

The contract under test:
(a) a session we started is stopped when the app exits.
(b) a session we FOUND already live is reused and left running — it belongs to
    whoever started it, and stopping it would take their windows with it.
(c) the stop still happens when the app dies of an unhandled exception, which
    is the case that would otherwise leak the most often.
(d) a frozen build never arms any of this; it has a real desktop and no xpra.
(e) the teardown is idempotent, because an atexit handler that ran twice would
    issue a stop against a display someone else may have since claimed.
(f) the session lifetime is NOT expressed as an xpra flag.  `--exit-with-client`
    reads as the obvious fix and is a trap: it stops the server whenever the
    viewer detaches, which is exactly what an operator does when they
    disconnect and go home partway through an unattended run.

HOW.  A fake `xpra` on PATH records its argv, so these assert on what the
launcher DID rather than on live sessions — the real thing takes seconds to
start, needs a display number nothing else has claimed, and would leave debris
on the developer's machine when a test failed.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


@pytest.fixture()
def fake_xpra(tmp_path):
    """
    Put a recording `xpra` first on PATH and hand back its logfile.

    Set XPRA_PRETEND_LIVE in the child's environment to make `xpra list`
    report an existing session, which is the (b) branch.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "xpra.log"
    script = bindir / "xpra"
    script.write_text(
        '#!/bin/sh\n'
        'echo "$@" >> "$XPRA_LOG"\n'
        'if [ "$1" = "list" ] && [ -n "$XPRA_PRETEND_LIVE" ]; then\n'
        '    echo "LIVE session at :100"\n'
        'fi\n'
        'exit 0\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    return bindir, log


def _launch(body: str, bindir: Path, log: Path, pretend_live: bool = False):
    """
    Run `body` in a fresh interpreter with the fake xpra on PATH.

    A subprocess because the thing under test is an atexit handler: it only
    runs when an interpreter shuts down, and this suite's interpreter has to
    survive to report the result.  DISPLAY is stripped because a set DISPLAY
    is the one condition under which _ensure_display() does nothing.
    """
    env = dict(os.environ)
    env["PATH"] = f"{bindir}{os.pathsep}{env['PATH']}"
    env["XPRA_LOG"] = str(log)
    env.pop("DISPLAY", None)
    env["QT_QPA_PLATFORM"] = "offscreen"
    if pretend_live:
        env["XPRA_PRETEND_LIVE"] = "1"

    src = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(_ROOT)!r})
        # The socket wait is a real filesystem poll against /tmp/.X11-unix.
        # Satisfy it in memory rather than dropping a stray socket-shaped file
        # into a directory the host's real X servers use.
        _real_exists = os.path.exists
        os.path.exists = lambda p: (
            True if str(p).startswith("/tmp/.X11-unix/X") else _real_exists(p))
        {textwrap.indent(textwrap.dedent(body), " " * 8).lstrip()}
    """)
    return subprocess.run(
        [sys.executable, "-c", src], capture_output=True, text=True,
        timeout=180, env=env)


def _calls(log: Path) -> list[str]:
    if not log.exists():
        return []
    return [ln.strip() for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]


# ── (a) we clean up after ourselves ──────────────────────────────────────────

def test_a_session_we_started_is_stopped_when_the_app_exits(fake_xpra):
    bindir, log = fake_xpra
    out = _launch("""
        import run_dashboard as rd
        rd._ensure_display()
        print("display:", os.environ.get("DISPLAY"))
    """, bindir, log)
    assert out.returncode == 0, out.stderr
    calls = _calls(log)
    assert any(c.startswith("start :100") for c in calls), calls
    assert any(c.startswith("stop :100") for c in calls), (
        "the launcher started an xpra session and never stopped it.  "
        f"xpra was called with: {calls}")
    assert "display: :100" in out.stdout


# ── (b) somebody else's session is not ours to end ───────────────────────────

def test_a_session_we_found_already_live_is_left_running(fake_xpra):
    bindir, log = fake_xpra
    out = _launch("""
        import run_dashboard as rd
        rd._ensure_display()
    """, bindir, log, pretend_live=True)
    assert out.returncode == 0, out.stderr
    calls = _calls(log)
    assert not any(c.startswith("start") for c in calls), (
        f"reused a live session but started another anyway: {calls}")
    assert not any(c.startswith("stop") for c in calls), (
        "stopped a session this process did not start; a persistent display "
        f"may have had somebody else's windows in it.  Calls: {calls}")


# ── (c) the case that leaks most often ───────────────────────────────────────

def test_the_session_is_stopped_even_when_the_app_dies_badly(fake_xpra):
    bindir, log = fake_xpra
    out = _launch("""
        import run_dashboard as rd
        rd._ensure_display()
        raise RuntimeError("dashboard fell over")
    """, bindir, log)
    assert out.returncode != 0, "expected the child to die of its exception"
    assert "dashboard fell over" in out.stderr
    assert any(c.startswith("stop :100") for c in _calls(log)), (
        "a crash left the xpra session behind; a crashing run is exactly when "
        "nobody is watching to clean it up by hand")


# ── (d) a frozen build has a real desktop ────────────────────────────────────

def test_a_frozen_build_never_arms_the_teardown(fake_xpra):
    bindir, log = fake_xpra
    out = _launch("""
        import run_dashboard as rd
        sys.platform = "linux"
        sys.frozen = True
        sys._MEIPASS = "/tmp/nope"
        rd._ensure_display()
        print("armed:", rd._xpra_started_by_us)
    """, bindir, log)
    assert out.returncode == 0, out.stderr
    assert "armed: False" in out.stdout
    assert _calls(log) == [], (
        f"a frozen build shelled out to xpra, which colleagues do not have: {_calls(log)}")


# ── (e) an atexit handler gets exactly one go ────────────────────────────────

def test_the_teardown_is_idempotent(fake_xpra):
    bindir, log = fake_xpra
    out = _launch("""
        import run_dashboard as rd
        rd._ensure_display()
        rd._stop_xpra_session()
        rd._stop_xpra_session()
    """, bindir, log)
    assert out.returncode == 0, out.stderr
    stops = [c for c in _calls(log) if c.startswith("stop")]
    assert len(stops) == 1, (
        "the teardown ran more than once; a second stop could land on a "
        f"display another process has since claimed.  Stops: {stops}")


# ── (f) the flag that looks right and is not ─────────────────────────────────

def test_the_session_does_not_die_when_a_client_detaches(fake_xpra):
    """
    A guard on the argv, not a live detach.  `--exit-with-client=yes` is the
    tempting one-line version of this fix, and it would stop the server the
    moment the operator disconnects — taking down an unattended run that is
    still working.  The lifetime belongs to the application process.
    """
    bindir, log = fake_xpra
    out = _launch("""
        import run_dashboard as rd
        rd._ensure_display()
    """, bindir, log)
    assert out.returncode == 0, out.stderr
    start = [c for c in _calls(log) if c.startswith("start")]
    assert start, "no start recorded"
    assert "exit-with-client" not in start[0], (
        "the session was tied to the viewer; detaching would now kill a run "
        f"that is still going.  start argv: {start[0]}")
