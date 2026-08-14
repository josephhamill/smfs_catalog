# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Guard test: qt_utils.set_si_label is the ONLY place an axis' unit and its SI
prefixing are decided.

WHY THIS EXISTS.  In the 2026-08-01 user test the raw-curve window's x-axis read
"Piezo (knm)" — kilonanometres — because setLabel(units="nm") tells pyqtgraph
that "nm" is a base unit and it prefixes what is already prefixed.  #95 had
already fixed exactly this bug, per axis, with enableAutoSIPrefix(False).  That
fix was applied to six windows and not to the other seven, because nothing made
the second line mandatory: fourteen axes carried it and twenty-six did not.

That is the same shape as the palette fork (test_style_is_single_source) and the
spin-box fork (test_numeric_ui): a rule that lives in whoever remembers it.  So
both halves of the decision now happen in one call, and this test keeps them
there.

Source-level, for the same reason those two are: building every window needs a
QApplication, a populated database and real curve files, and a guard that is
expensive to run is a guard that gets skipped.  The one thing that CANNOT be
checked from source — that pyqtgraph actually renders 1800 nm as "1.8 um" — is
checked against pyqtgraph's own siScale at the bottom, not against prose.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

import pytest

from smfs_catalog import quantities as q

PKG = Path(__file__).resolve().parents[1] / "smfs_catalog"

# qt_utils.py IS the display layer.  set_si_label decides units + prefixing;
# _DateAxis disables prefixing because Unix timestamps are not a physical
# quantity and pyqtgraph's rescaling corrupts them beyond recovery.
EXEMPT = {"qt_utils.py"}

UNITS_KWARG  = re.compile(r"\.setLabel\s*\([^)]*\bunits\s*=")
AUTO_SI      = re.compile(r"\.enableAutoSIPrefix\s*\(")
AXIS_SCALE   = re.compile(r"getAxis\s*\([^)]*\)\s*\.setScale\s*\(")


def _modules():
    return sorted(p for p in PKG.glob("*.py") if p.name not in EXEMPT)


def _code_lines(path: Path):
    """Every line of real code, comments and strings stripped.

    This module's own docstrings quote the banned patterns verbatim ('setLabel
    (units="nm")'), and so do the explanatory comments left at each converted
    call site.  A raw grep would flag the explanation of the bug as the bug.
    """
    src = path.read_text(encoding="utf-8")
    drop = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            for ln in range(tok.start[0], tok.end[0] + 1):
                drop.add(ln)
    return [(i, ln) for i, ln in enumerate(src.splitlines(), 1) if i not in drop]


# ── (a) The reach of the helper ───────────────────────────────────────────────

def test_no_module_sets_axis_units_itself():
    """setLabel(units=...) outside qt_utils is the "knm" bug in the making.

    Passing a unit means deciding whether it is a base SI unit, which is the
    half that kept being forgotten.  set_si_label makes it one decision.
    """
    bad = [f"{p.name}:{i}" for p in _modules() for i, ln in _code_lines(p)
           if UNITS_KWARG.search(ln)]
    assert not bad, (
        "setLabel(units=...) must go through qt_utils.set_si_label, which "
        "settles the SI prefix at the same time:\n  " + "\n  ".join(bad)
    )


def test_no_module_toggles_si_prefixing_itself():
    """The per-axis patch #95 introduced, and which this pass replaced.

    An enableAutoSIPrefix call anywhere else means an axis whose prefixing was
    decided away from its unit — the two drifting apart is the whole defect.
    """
    bad = [f"{p.name}:{i}" for p in _modules() for i, ln in _code_lines(p)
           if AUTO_SI.search(ln)]
    assert not bad, (
        "enableAutoSIPrefix belongs in qt_utils.set_si_label, beside the unit "
        "it depends on:\n  " + "\n  ".join(bad)
    )


def test_no_module_scales_an_axis_itself():
    """setScale is how the display layer converts; it is not a caller's tool.

    An axis scaled outside set_si_label would silently disagree with its own
    label — the same failure as the two above, with no visible unit to catch it.
    """
    bad = [f"{p.name}:{i}" for p in _modules() for i, ln in _code_lines(p)
           if AXIS_SCALE.search(ln)]
    assert not bad, "axis setScale belongs in set_si_label:\n  " + "\n  ".join(bad)


# ── (b) Units named at call sites are units the register knows ────────────────

def _declared_units() -> set[str]:
    """Every unit string quantities.py declares, however it is spelled."""
    named = {v for k, v in vars(q).items()
             if k.isupper() and isinstance(v, str) and not k.startswith("_")}
    return named | {qty.unit for qty in q.QUANTITIES.values()} \
                 | {qty.shown_unit for qty in q.QUANTITIES.values()}


def _calls_to(tree, *names):
    """Every ast.Call whose callee is one of `names` (dotted or bare)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        called = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        dotted = None
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            dotted = f"{f.value.id}.{f.attr}"
        if called in names or dotted in names:
            yield node


def _unit_arg(call):
    """The `unit` argument of a set_si_label call, positional or keyword."""
    for kw in call.keywords:
        if kw.arg == "unit":
            return kw.value
    return call.args[3] if len(call.args) > 3 else None


def test_call_sites_name_units_the_register_knows():
    """A unit passed as a bare literal is a unit nobody declared.

    set_si_label(plot, "left", "Force", "pN") would work, and would also be the
    first step back to thirteen windows each spelling their own units.  The
    register is what makes "is this SI?" answerable in one place.

    Read with ast, not regex: the third argument is free prose ("Piezo", "PC")
    and only the fourth is a unit, which no amount of pattern-matching on a
    line of source can tell apart.
    """
    declared = _declared_units()
    bad = []
    for p in _modules():
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for call in _calls_to(tree, "set_si_label"):
            arg = _unit_arg(call)
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value and arg.value not in declared:
                    bad.append(f"{p.name}:{arg.lineno}: {arg.value!r}")
    assert not bad, (
        "unit literals at the call site — declare them in quantities.py "
        "instead:\n  " + "\n  ".join(bad)
    )


def test_every_si_unit_is_a_declared_unit():
    """SI_UNITS keys must be units the app actually uses, not aspirational."""
    declared = _declared_units()
    unknown = sorted(set(q.SI_UNITS) - declared)
    assert not unknown, f"SI_UNITS names units nothing declares: {unknown}"


def test_non_si_units_are_absent_on_purpose():
    """The units that must never be prefixed, asserted rather than assumed.

    A prefixed Angstrom, a prefixed ratio and a prefixed decibel are all
    meaningless; their absence from SI_UNITS is a decision, so it is stated
    here rather than left to whoever next edits the table.
    """
    for unit in (q.NM_PER_NM, q.PTS, q.DB, q.RATIO, q.COUNT):
        assert q.si_for(unit) is None, (
            f"{unit!r} must not be SI-prefixed — see the table in quantities.py"
        )


# ── (c) Typeset text never reaches a plain-text surface ───────────────────────

def test_typeset_constants_never_label_a_line_or_text_item():
    """An InfiniteLine label is rendered by setPlainText, not setHtml.

    The normalized 2DH's singularity line showed the literal characters
    '<i>x&#771;</i> = 1 (<i>l</i><sub>c</sub>)' on screen for months, on a plot
    this app exists to produce for publication.  style.py's typeset constants
    are correct for axis labels and titles and wrong here; the plain twins
    beside them are for this.
    """
    from smfs_catalog import style

    typeset = {"L_P", "L_C", "F_STAR", "X_STAR", "X_TILDE",
               "DELTA_X", "DELTA_F", "FORCE", "EXTENSION"}
    bad = []
    for p in _modules():
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for call in _calls_to(tree, "pg.InfiniteLine", "pg.TextItem"):
            for node in ast.walk(call):
                if (isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Name)
                        and node.value.id == "style"
                        and node.attr in typeset):
                    bad.append(f"{p.name}:{node.lineno}: style.{node.attr}")
    assert not bad, (
        "typeset (HTML) constants on a plain-text item — use the *_PLAIN "
        "twins:\n  " + "\n  ".join(sorted(set(bad)))
    )


def test_plain_twins_carry_no_markup():
    """The plain spellings must actually be plain, or they fix nothing."""
    from smfs_catalog import style

    for name in dir(style):
        if not name.endswith("_PLAIN"):
            continue
        value = getattr(style, name)
        assert "<" not in value and "&" not in value, (
            f"style.{name} = {value!r} still contains markup"
        )


# ── (d) The claim itself, recomputed from pyqtgraph ───────────────────────────

@pytest.mark.parametrize("unit, value, expect", [
    # The reported bug: 1800 nm rendered as "-1.8 knm".
    (q.NM,  1800.0,  (1.8,   "µm")),
    (q.NM,    44.0,  (44.0,  "nm")),
    (q.NM,     0.26, (260.0, "pm")),   # l_p, the app's smallest real length
    (q.PN,   166.0,  (166.0, "pN")),   # median rupture force
    (q.PN,  4200.0,  (4.2,   "nN")),
    (q.HZ, 25000.0,  (25.0,  "kHz")),
])
def test_declared_factors_render_correctly(unit, value, expect):
    """What the user sees, computed by pyqtgraph's own siScale.

    This is the claim that was false before — not "the table says nm is 1e-9"
    but "a curve 1800 nm long is labelled in something a reader recognises".
    Asserting the table against itself would pass on the broken code too.
    """
    import pyqtgraph.functions as fn

    si = q.si_for(unit)
    assert si is not None, f"{unit} should be SI-prefixable"

    # Exactly what set_si_label arranges: setScale(factor) feeds
    # AxisItem.updateAutoSIPrefix, which calls siScale on range * scale.
    scale, prefix = fn.siScale(value * si.factor, power=si.power)
    shown = value * si.factor * scale
    want_value, want_unit = expect
    assert shown == pytest.approx(want_value, rel=1e-9)
    assert f"{prefix}{si.base}" == want_unit


# ── (e) The rendering, on a real axis ─────────────────────────────────────────
# The tests above check the arrangement and the arithmetic.  These two build an
# actual pyqtgraph axis and read what it would draw, because the one defect this
# pass introduced (and fixed) was invisible to both: a stale prefix left behind
# by pyqtgraph's own enableAutoSIPrefix.

def _axis_text(unit, lo, hi, *, si=True, relabel_after_range=False):
    """(label as plain text, first few tick strings) for a real axis."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import pyqtgraph as pg
    from PyQt6.QtWidgets import QApplication

    from smfs_catalog.qt_utils import set_si_label

    app = QApplication.instance() or QApplication([])
    pw = pg.PlotWidget()
    if not relabel_after_range:
        set_si_label(pw, "bottom", "X", unit, si=si)
    pw.setXRange(lo, hi, padding=0)
    if relabel_after_range:
        set_si_label(pw, "bottom", "X", unit, si=si)
    app.processEvents()

    ax = pw.getAxis("bottom")
    label = re.sub("<[^>]+>", "", ax.labelString()).strip()
    major = ax.tickValues(lo, hi, 500)[0][1]
    ticks = ax.tickStrings(sorted(major), ax.autoSIPrefixScale * ax.scale,
                           abs(hi - lo) / 5)
    return label, ticks


def test_a_long_pull_is_labelled_in_micrometres():
    """The reported bug, end to end: -1800 nm must not read "-1.8 knm"."""
    label, ticks = _axis_text(q.NM, -1800, -1100)
    assert label == "X (µm)", label
    assert any(t.startswith("-1.") for t in ticks), ticks


def test_a_pinned_axis_relabelled_over_a_live_range_keeps_its_unit():
    """The 2DH re-apply path, which is where a prefix could come back.

    base_2dh_window re-applies its axis labels every time Grid settings are
    applied — by then the plot spans a real ±2000 nm.  pyqtgraph's
    updateAutoSIPrefix() never consults the autoSIPrefix flag and
    enableAutoSIPrefix() calls it on the way past, so disabling prefixing at
    that moment SETS one.  set_si_label clears it; without that line this axis
    reads "X (knm)" with its ticks divided by a thousand, which is the original
    bug arriving through the fix for it.
    """
    label, ticks = _axis_text(q.NM, -2000, 2000, si=False,
                              relabel_after_range=True)
    assert label == "X (nm)", label
    assert "-2000" in ticks, ticks


def test_a_non_si_unit_is_never_prefixed():
    """A ratio has no prefix: "knm/nm" is not a unit, at any range."""
    label, _ = _axis_text(q.NM_PER_NM, 0, 4000, si=False)
    assert label == "X (nm/nm)", label


def test_a_pinned_squared_unit_stays_in_its_own_unit():
    """nm² IS SI-prefixable, so the variance axis has to be pinned, not lucky.

    Left free it would relabel itself pm² over the real 0.0016-0.037 nm² range
    (1 nm² = 1e6 pm²), while the two threshold spin boxes beside it — which set
    the very lines drawn on that axis — can carry no prefix at all.  That is the
    same one-number-two-scales trap the Ångström display used to hide.
    """
    assert q.si_for(q.NM2) is not None, "nm² is a prefixable SI unit"
    label, _ = _axis_text(q.NM2, 0.0, 0.04, si=False)
    assert label == "X (nm²)", label


def test_a_prefixed_unit_would_have_been_caught():
    """The null test: the old code's own arrangement produces the bug.

    Without this, the test above only proves the new numbers are self-
    consistent.  Feeding pyqtgraph the already-prefixed unit is what produced
    "knm", and it still does — which is why the unit register, not a habit, is
    what keeps it away.
    """
    import pyqtgraph.functions as fn

    scale, prefix = fn.siScale(1800.0)      # units="nm", scale left at 1.0
    assert f"{prefix}nm" == "knm"
    assert 1800.0 * scale == pytest.approx(1.8)
