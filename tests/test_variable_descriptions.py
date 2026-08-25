# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Guard test: a variable explains itself everywhere it is offered.

WHAT WENT WRONG.  Twenty-one good descriptions existed, in
dashboard_window._QUEUE_COL_TOOLTIPS — a private dict, reachable only by
hovering a queue-table column header.  Every OTHER place the app names a
variable showed the bare label: the any-vs-any scatter's X/Y dropdowns, the
per-variable drill-down window, and the criteria dialog, where ticking a box
you have not understood silently changes which curves count as hits for that
experimentalist's entire cohort.  Ten more variables — the instrument
metadata off the wave note — had no description anywhere at all, including
trigger_point_nn, which is a FORCE rather than a distance and had already
been documented backwards for months.

So this is the palette/spin-box/axis-label fault once more:
one decision, one copy, and only one window can see it.

THE RULE NOW: variables.DESCRIPTIONS is the ONE register, reached through
variables.describe().  Consumers ask; nobody keeps a copy.

The contract under test:
(a) every variable available() offers has a description — adding a variable
    without one is the regression, and it is silent, because a missing
    tooltip looks identical to a tooltip you happened not to hover.
(b) descriptions are non-trivial.  A one-word entry satisfies (a) while
    telling a reader nothing, which is the shape a future "fix the failing
    test" commit would take.
(c) describe() returns "" for an unknown key, never a placeholder — a caller
    can then decide not to set a tooltip at all, and an empty tooltip is not
    the same as one that says nothing useful.
(d) NO module outside variables.py keeps its own map of variable-key ->
    prose.  This is the check that actually stops the fork coming back;
    (a)-(c) would all still pass with a second copy sitting in the dashboard.
(e) the four fixed queue columns are covered too.  They are not variables —
    they are the row's identity and its three verdicts — so they live in
    dashboard_window._FIXED_COL_TOOLTIPS, and they carry three closed
    vocabularies ('stale params', 'unavailable' vs 'unusable', hit/non-hit)
    that a reader has no way to guess.
(f) the decomposition window's eight controls carry hover text.  It is the
    densest parameter window in the app, its labels are abbreviations
    ('Thr. appr', 'invOLS off'), and until this change all five of its
    setToolTip calls were on STATUS LABELS rather than on any control — so a
    grep for "does this window have tooltips" said yes and the answer was no.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from smfs_catalog import db as _db                      # noqa: E402
from smfs_catalog import variables as _vars             # noqa: E402

_PKG = _ROOT / "smfs_catalog"

# Long enough to be a sentence rather than a restated label.  The shortest
# real entry ("Deflection baseline offset subtracted before converting to
# force.") is 68 characters.
_MIN_DESCRIPTION = 40
_MAX_DESCRIPTION = 220


@pytest.fixture(scope="module")
def offered(tmp_path_factory):
    """Every variable the app would offer on an axis, against an empty DB."""
    db_path = str(tmp_path_factory.mktemp("desc") / "t.db")
    _db.initialise(db_path)
    return _vars.available(db_path)


# ── (a)/(b) coverage and substance ───────────────────────────────────────────

def test_every_offered_variable_has_a_description(offered):
    missing = [v.key for v in offered if not v.description]
    assert not missing, (
        "offered on an axis with nothing saying what they are: "
        + ", ".join(missing))


def test_descriptions_say_something(offered):
    thin = [(v.key, v.description) for v in offered
            if len(v.description) < _MIN_DESCRIPTION]
    assert not thin, (
        "these restate the label rather than explaining the measurement: "
        + ", ".join(f"{k} ({d!r})" for k, d in thin))


def test_ui_descriptions_fit_in_a_tooltip(offered):
    long = [(v.key, len(v.description)) for v in offered
            if len(v.description) > _MAX_DESCRIPTION]
    assert not long, (
        "UI descriptions have become reference documentation again: "
        + ", ".join(f"{k} ({n} chars)" for k, n in long))


def test_description_matches_the_register(offered):
    """Variable.description must be the register, not a second field."""
    for v in offered:
        assert v.description == _vars.describe(v.key)


# ── (c) the missing case is empty, not a placeholder ─────────────────────────

def test_unknown_key_describes_as_empty_string():
    assert _vars.describe("no_such_variable_key") == ""
    assert _vars.describe("") == ""


# ── (d) one register, no second copy ─────────────────────────────────────────

def _string_valued_key_maps(path: Path):
    """
    Module-level dicts mapping a known variable key to a long string.

    Deliberately keyed on OVERLAP with the real variable set rather than on a
    name like '*_TOOLTIPS': the dict that caused this was called
    _QUEUE_COL_TOOLTIPS, but the next one will be called something else.
    """
    found = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            name, val = getattr(node.target, "id", ""), node.value
        elif isinstance(node, ast.Assign):
            name, val = getattr(node.targets[0], "id", ""), node.value
        else:
            continue
        if not isinstance(val, ast.Dict):
            continue
        try:
            d = ast.literal_eval(val)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        prose = {k: v for k, v in d.items()
                 if isinstance(k, str) and isinstance(v, str)
                 and len(v) >= _MIN_DESCRIPTION}
        if len(set(prose) & set(_vars.DESCRIPTIONS)) >= 2:
            found.append((name, sorted(set(prose) & set(_vars.DESCRIPTIONS))))
    return found


@pytest.mark.parametrize(
    "path", sorted(p for p in _PKG.glob("*.py") if p.name != "variables.py"),
    ids=lambda p: p.name)
def test_no_second_copy_of_the_descriptions(path):
    dupes = _string_valued_key_maps(path)
    assert not dupes, (
        f"{path.name} keeps its own prose for variable keys; ask "
        f"variables.describe() instead so there is one place to edit: "
        + "; ".join(f"{n} -> {ks}" for n, ks in dupes))


# ── (e) the four fixed queue columns ─────────────────────────────────────────

def test_fixed_queue_columns_are_explained():
    from smfs_catalog import dashboard_window as dw
    fixed = [k for k, _ in dw._QUEUE_COLUMNS_FIXED]
    missing = [k for k in fixed if not dw._FIXED_COL_TOOLTIPS.get(k)]
    assert not missing, f"fixed queue columns with no hover text: {missing}"


@pytest.mark.parametrize("key,must_mention", [
    # Each of these is a closed vocabulary the column displays and nothing
    # else defines.  If a word is retired, this fails and says so.
    ("status", ["up to date", "stale params", "not analysed", "visited"]),
    ("event",  ["non_event", "unavailable", "unusable"]),
    ("hit",    ["non-hit"]),
])
def test_verdict_columns_spell_out_their_vocabulary(key, must_mention):
    from smfs_catalog import dashboard_window as dw
    tip = dw._FIXED_COL_TOOLTIPS[key]
    missing = [w for w in must_mention if w not in tip]
    assert not missing, (
        f"the {key!r} column can display {missing} and its hover text never "
        f"mentions them")


def test_hit_column_states_the_empty_criteria_rule():
    """
    'No criteria checked' means EVERY event is a hit.  That is the basis of
    the hand-built cohort workflow and is the single most surprising thing
    about the gate, so the column that shows the verdict has to say it.
    """
    from smfs_catalog import dashboard_window as dw
    tip = dw._FIXED_COL_TOOLTIPS["hit"].lower()
    assert "no criteria checked" in tip and "every event is a hit" in tip


def test_segment_selector_explains_single_segment_fallback():
    """Penultimate deliberately preserves one-segment curves; stale hover text
    used to claim they became blank, implying that the dropdown filtered."""
    src = (_PKG / "dashboard_window.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    tips = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setToolTip"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "_segment_select_combo"
                and node.args):
            tips.append(ast.literal_eval(node.args[0]))
    assert len(tips) == 1
    tip = tips[0].lower()
    assert "falls back to that same segment" in tip
    assert "does not become blank" in tip
    assert "roi segments ≥ 2" in tip


# ── (f) the decomposition window's controls ──────────────────────────────────

_DECOMP_CONTROLS = [
    "_cutoff_slider", "_trim_spinbox", "_var_win_spinbox",
    "_thresh_appr_spinbox", "_thresh_retr_spinbox", "_anchor_spinbox",
    "_invols_offset_spinbox", "_invols_window_spinbox",
]


@pytest.mark.parametrize("attr", _DECOMP_CONTROLS)
def test_decomposition_controls_have_hover_text(attr):
    """
    Source-level, like the other GUI guards here: constructing this window
    needs a QApplication, a populated DB and a real curve file.  What is
    checked is that setToolTip is called on the CONTROL — the bug was five
    setToolTip calls in this file that were all on status labels.
    """
    src = (_PKG / "decomposition_window.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    tipped = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setToolTip"
                and isinstance(node.func.value, ast.Attribute)):
            tipped.add(node.func.value.attr)
    assert attr in tipped, (
        f"decomposition_window.{attr} has no tooltip; its on-screen label is "
        f"an abbreviation and nothing else explains it")
