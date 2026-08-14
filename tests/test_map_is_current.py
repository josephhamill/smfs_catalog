# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Guard: the structural claims in CLAUDE.md must match the code.

CLAUDE.md is the map Claude (and anyone new) uses to understand this app. A stale
map is worse than no map — this whole file exists because `models.py:4` has been
confidently stating the wrong force-sign convention for months, and because on
2026-07-16 an analysis concluded "the worker doesn't fit WLC" (it does) from a grep
that could not have found it.

Every claim in CLAUDE.md marked 🔍 is checked here.

IMPORTANT — HOW TO READ A FAILURE:
    A failure does NOT necessarily mean you broke something. It means the CODE and
    CLAUDE.md now disagree. Usually that is because you FIXED something the map
    still describes as broken.

    Either way the fix is the same: UPDATE CLAUDE.md as part of your change.

    Do not delete a check to make it pass. If a check no longer makes sense,
    delete it *and* the CLAUDE.md section it guards, together.

Run with the smfs-catalog env, from the repo root:
    python tests/test_map_is_current.py
or:
    pytest tests/test_map_is_current.py
"""
import ast
import re
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent / "smfs_catalog"
CLAUDE_MD = Path(__file__).resolve().parent.parent / "CLAUDE.md"

_SRC: dict[str, str] = {
    p.name: p.read_text(encoding="utf-8", errors="replace") for p in PKG.glob("*.py")
}


def _imports_of(module: str) -> list[str]:
    """Files that import `module`, by any import form used in this codebase
    (including the function-local imports that make grep unreliable here).

    AST-based, not regex: a regex version of this check (used until
    2026-07-29) missed the multi-name relative form `from . import a, b,
    module` — which is exactly how `app.py` imports `analysis_runner` — so
    it reported the dead chain as still-dead while it was actually live.
    Found while investigating #67. Walk every Import/ImportFrom node instead
    of pattern-matching source text.
    """
    out = []
    for name, src in _SRC.items():
        if name == f"{module}.py":
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name in (module, f"smfs_catalog.{module}")
                       for alias in node.names):
                    found = True
            elif isinstance(node, ast.ImportFrom):
                if node.module in (module, f"smfs_catalog.{module}"):
                    found = True
                elif node.level >= 1 and node.module in (None, "", module):
                    if any(alias.name == module for alias in node.names) or node.module == module:
                        found = True
            if found:
                break
        if found:
            out.append(name)
    return sorted(out)


# ── The checks ────────────────────────────────────────────────────────────────

def test_claude_md_still_points_at_the_running_order() -> None:
    """CLAUDE.md §0 sends every session to the running order before any work is
    proposed. That pointer is the entry point to the whole map — without it, each
    session invents its own order and the user's priorities get silently
    overwritten. CLAUDE.md loads automatically; the order does not.

    The order is a QUERY, not a document (2026-08-03, closing #87): the milestone
    description carries the one sequencing rule labels cannot express, and the
    issue list carries the rest. This used to assert `gh issue view 87`; when the
    order stopped living in an issue the assertion went stale and started failing
    for the wrong reason. What must hold is that BOTH halves of the query are in
    §0 — either alone is a half-answer."""
    if not CLAUDE_MD.exists():
        # The public snapshot of this repo ships the code and the tests but not
        # the working notes, so there is no map to check there. Inside this
        # repo the file is always present and the guard below always runs.
        pytest.skip("CLAUDE.md is not part of this checkout (public snapshot)")
    text = CLAUDE_MD.read_text(encoding="utf-8", errors="replace")
    for needle, what in (
        ("milestones/1 --jq .description", "the milestone description (the sequencing rule)"),
        ('gh issue list --milestone "v1 release"', "the open-issue query (the work list)"),
    ):
        assert needle in text, (
            f"CLAUDE.md §0 no longer runs {what}. The running order is a query over "
            "the milestone, not a document; if the query changed, update §0 and this "
            "check together. If it was deleted, sessions will now each invent their "
            "own order — which is the problem this replaced #87 to solve."
        )
    assert "there must not be one again" in text, (
        "CLAUDE.md §0 lost the prohibition on re-creating a running-order document. "
        "That sentence is the load-bearing half: #87 grew to 305 lines and 51 KB of "
        "session narrative before it was closed, and the urge to rewrite it is "
        "documented as recurring."
    )


def test_no_dynamic_imports() -> None:
    """CLAUDE.md §5 asserts the dead-code map is airtight *because* nothing
    resolves imports at runtime. If that stops being true, static analysis can no
    longer prove a module is unreachable, and the whole §5 map becomes a guess."""
    pat = re.compile(r"importlib|__import__|globals\(\)\[|getattr\(sys\.modules")
    hits = [n for n, src in _SRC.items() if pat.search(src)]
    assert not hits, (
        f"Dynamic imports appeared in {hits}. CLAUDE.md §5 claims the dead-code map "
        "is provable by static analysis — that claim is now unsafe. Re-verify §5 by "
        "hand and say how it was verified."
    )


def test_worker_reaches_the_wlc_fitter() -> None:
    """CLAUDE.md §2 ★ — THE claim that was got wrong on 2026-07-16.

    The worker fits WLC through a chain no grep for 'wlc' can find. If any link
    breaks, the map lies about the app's single most misunderstood behaviour."""
    chain = [
        ("analysis_worker.py", "analyse_and_classify"),
        ("curve_analysis.py", "_persist_multi_event_roi"),
        ("curve_analysis.py", "compute_curve_events"),
        ("roi_pipeline.py", "fit_segments"),
        ("roi_events.py", "def fit_segments"),
    ]
    for fname, needle in chain:
        assert needle in _SRC[fname], (
            f"CLAUDE.md §2 documents the worker's WLC chain as "
            f"analyse_and_classify → _persist_multi_event_roi → compute_curve_events "
            f"→ fit_segments. The link '{needle}' is no longer in {fname}, so that "
            "diagram is now wrong. Re-trace the chain and redraw §2 — this is the "
            "exact fact a grep cannot recover, so the map is the only defence."
        )


def test_fitter_a_stays_deleted() -> None:
    """CLAUDE.md §3: #86 closed 2026-07-22 by deleting wlc_fit.py outright. This
    replaces test_there_are_still_two_wlc_fitters, which fired (as designed) the
    day that deletion happened."""
    assert not (PKG / "wlc_fit.py").exists(), (
        "wlc_fit.py is back — did a merge/revert restore Fitter A? If this is "
        "deliberate (e.g. keeping it for comparison), CLAUDE.md §3 and §6 need "
        "rewriting to describe two fitters again."
    )
    assert "def process_event" not in _SRC.get("event_processor.py", ""), (
        "event_processor.process_event (Fitter A's per-event wrapper) is back. "
        "CLAUDE.md §3 says this was deleted with wlc_fit.py — update it if that's "
        "no longer true."
    )


def test_the_one_good_cache_key_still_derives_from_the_dataclass() -> None:
    """CLAUDE.md §4: event_map_params_json is the one key that has never drifted,
    because asdict() makes drift impossible. It is the pattern the other keys are
    told to copy — so if it stops being that pattern, the advice is wrong."""
    src = _SRC["roi_pipeline.py"]
    m = re.search(r"def event_map_params_json.*?(?=\ndef |\Z)", src, re.S)
    assert m, (
        "event_map_params_json is gone from roi_pipeline.py. CLAUDE.md §4 holds it up "
        "as the one cache key that cannot drift and tells the others to copy it. "
        "Update §4 with whatever the new exemplar is."
    )
    assert "asdict(" in m.group(0), (
        "event_map_params_json no longer builds its key from asdict(). It is now a "
        "hand-maintained field list — i.e. it can now drift like the keys in #80. "
        "This is very likely a regression; see CLAUDE.md §4."
    )


# test_models_py_still_documents_the_force_sign_backwards was removed when #79 was
# fixed on 2026-07-16 — it had done its job (it fired, and CLAUDE.md §4 was updated
# in the same change). The convention itself is now guarded behaviourally by
# tests/test_force_sign.py, which asserts what the sign IS rather than watching for
# a comment that lies about it.


# ── Runner (matches the repo's existing standalone-script convention) ─────────

def _main() -> int:
    checks = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in checks:
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {fn.__name__}\n       {e}\n")
        else:
            print(f"[ok]   {fn.__name__}")
    print()
    if failed:
        print(f"{failed} of {len(checks)} map checks failed.")
        print("CLAUDE.md and the code disagree. Update CLAUDE.md — do not delete checks.")
    else:
        print(f"All {len(checks)} map checks pass. CLAUDE.md matches the code.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
