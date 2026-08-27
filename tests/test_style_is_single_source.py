# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Guard test: style.py is the ONLY module that names a colour.

WHY THIS EXISTS.  style.py's own header lists the fourteen independent palettes
it was written to replace.  The app re-forked anyway — not in the plot
layer, which held, but in the Qt chrome around it,
which the original consolidation never covered: five different "muted grey"s
across thirteen labels, SIX copies of one selection blue, a second status
palette competing with style.py's, and a calendar highlight hand-copied from the
dashboard's row tint with the comment "same green as 'event'" standing in for a
shared constant.

That is the same lesson as tests/test_palette.py: a claim about the code that
nobody re-derives goes stale, so it has to be
machine-checked rather than hand-maintained.  test_palette.py already validates
the palette's *contents* (separation, contrast, CVD).  This one validates its
*reach* — that no other module quietly starts naming colours again.

If a new colour is genuinely needed, add it to style.py and use it from there.
This test failing is not a reason to add an allowlist entry.
"""

from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "smfs_catalog"

# style.py is the single source for palette values, including the published
# ColorBrewer control points used by the PCA loading map.
EXEMPT = {"style.py"}

# SIX (or eight) hex digits only, deliberately -- NOT the three-digit CSS
# shorthand.  '#555' is indistinguishable from an HTML numeric entity such as
# '&#771;', the combining tilde in the normalized 2DH's axis label, which
# appears inside a real string rather than a comment, so tokenising doesn't
# separate them.
# Every colour in style.py is written full-length, so requiring six digits costs
# nothing and keeps this test from crying wolf -- and a shorthand smuggled into
# a stylesheet is still caught by CSS_LITERAL below.
HEX = re.compile(r"#[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?\b")

# A CSS colour property with a LITERAL value.  'color: {style.INK_MUTED}' is the
# correct pattern and must pass; only 'color: #555' / 'color: red' is a colour
# chosen at the call site.
# The character class excludes whitespace as well as '{', so \s* cannot
# backtrack into matching the separating space and let an interpolation through.
CSS_LITERAL = re.compile(r"\b(?:background-)?color\s*:\s*([^\s;{'\"])", re.I)
QCOLOR_NUMERIC = re.compile(r"QColor\(\s*\d")
PG_CONFIG = re.compile(r"setConfigOption\(\s*[\"'](?:background|foreground)[\"']")
BARE_PG_COLOUR = re.compile(
    r"(?:mkPen|mkBrush|TextItem)\([^)]*?[\"'][rgbkwcmy][\"']"
)
POINT_SIZE_LITERAL = re.compile(r"\.setPointSize\(\s*\d")
QFONT_LITERAL = re.compile(r"\bQFont\(\s*[\"']")
CSS_TYPE_LITERAL = re.compile(
    r"\b(?:font-size|font-family|font-weight)\s*:\s*(?!\{)", re.I
)


def _modules():
    return sorted(p for p in PKG.glob("*.py") if p.name not in EXEMPT)


def _string_tokens(path: Path):
    """Every STRING token in a module, with its line number.

    Tokenising rather than grepping the raw text matters here: an HTML numeric
    entity is three or four digits behind a '#', so a regex over raw source
    would flag every one, and a test that cries wolf gets an allowlist bolted on
    until it means nothing.
    """
    src = path.read_text(encoding="utf-8")
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.STRING:
            out.append((tok.start[0], tok.string))
    return out


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_no_module_names_a_colour(path: Path):
    """No hex literal, and no CSS colour property, outside style.py."""
    bad = []
    for lineno, text in _string_tokens(path):
        # Docstrings narrate history and may quote a retired hex; only code-
        # carrying strings (stylesheets, HTML, colour args) are the risk.
        if HEX.search(text) and not text.lstrip().startswith(('"""', "'''")):
            bad.append(f"{path.name}:{lineno}  hex literal in {text[:70]!r}")
        if CSS_LITERAL.search(text):
            bad.append(f"{path.name}:{lineno}  literal CSS colour in {text[:70]!r}")
    assert not bad, (
        "colour named outside style.py — add it to style.py and import it:\n  "
        + "\n  ".join(bad)
    )


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_no_module_builds_a_colour_from_numbers(path: Path):
    """QColor(220, 245, 220) is a palette entry with no name and no home."""
    src = path.read_text(encoding="utf-8")
    hits = [
        f"{path.name}:{i}  {line.strip()[:70]!r}"
        for i, line in enumerate(src.splitlines(), 1)
        if QCOLOR_NUMERIC.search(line)
    ]
    assert not hits, (
        "QColor built from raw channel values — name it in style.py:\n  "
        + "\n  ".join(hits)
    )


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_plot_defaults_come_from_style(path: Path):
    """The background/foreground pair was written out longhand in 13 windows,
    while style.SURFACE was read by nothing.  style.apply_plot_defaults()."""
    src = path.read_text(encoding="utf-8")
    hits = [
        f"{path.name}:{i}  {line.strip()[:70]!r}"
        for i, line in enumerate(src.splitlines(), 1)
        if PG_CONFIG.search(line)
    ]
    assert not hits, (
        "call style.apply_plot_defaults() instead:\n  " + "\n  ".join(hits)
    )


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_no_bare_pyqtgraph_colour_letters(path: Path):
    """pg.mkPen('k') / TextItem(color='k') — a colour chosen at the call site."""
    src = path.read_text(encoding="utf-8")
    hits = [
        f"{path.name}:{i}  {line.strip()[:70]!r}"
        for i, line in enumerate(src.splitlines(), 1)
        if BARE_PG_COLOUR.search(line)
    ]
    assert not hits, (
        "use a style.* constant instead of a pyqtgraph colour letter:\n  "
        + "\n  ".join(hits)
    )


def test_style_never_imports_qtwidgets():
    """THE boundary that keeps style.py a style guide instead of a junk drawer.

    style.py answers "what does this look like?" with data and pure functions.
    Appearance needs QColor and pen styles; it never needs a widget.  Anything
    that reacts to a user — a spin box that must not display a value it doesn't
    hold, a validator, a dialog — needs QtWidgets, so this assertion bounces it
    into widgets.py automatically rather than leaving "is this style?" as a
    judgement call nobody makes the same way twice.
    """
    src = (PKG / "style.py").read_text(encoding="utf-8")
    offenders = [
        line.strip()
        for line in src.splitlines()
        if re.match(r"\s*(from|import)\b", line) and "QtWidgets" in line
    ]
    assert not offenders, (
        "style.py imported QtWidgets — that is behaviour, not appearance; it "
        "belongs in widgets.py:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_no_module_invents_typography(path: Path):
    """Font roles belong beside the palette, not as per-window point sizes.

    This is the typography twin of ``test_no_module_names_a_colour``.  A
    literal face, size or CSS font declaration outside style.py creates a
    second visual system even when every colour is centralized.
    """
    src = path.read_text(encoding="utf-8")
    bad = []
    for lineno, line in enumerate(src.splitlines(), 1):
        if POINT_SIZE_LITERAL.search(line):
            bad.append(f"{path.name}:{lineno}  literal point size")
        if QFONT_LITERAL.search(line):
            bad.append(f"{path.name}:{lineno}  literal font family")
    for lineno, text in _string_tokens(path):
        if CSS_TYPE_LITERAL.search(text):
            bad.append(f"{path.name}:{lineno}  literal CSS typography")
    assert not bad, (
        "typography named outside style.py — add a semantic role there:\n  "
        + "\n  ".join(bad)
    )


def test_row_tints_cover_the_event_vocabulary():
    """The queue tint follows the files.event verdict (the
    vocabulary), so a new verdict value cannot ship without a colour — which is
    how 'unavailable' came to need one in the first place."""
    from smfs_catalog import style

    from smfs_catalog import db

    for verdict in ("running", "default"):
        assert verdict in style.ROW_TINT, verdict
        assert style.ROW_TINT[verdict].startswith("#")

    # Read the vocabulary from db, not from a list retyped here — otherwise
    # this test only checks that two hand-maintained lists agree, which is the
    # fork it exists to prevent. Adding a verdict without a colour fails here.
    for verdict in db.EVENT_VERDICTS:
        assert verdict in style.ROW_TINT, (
            f"set_event accepts {verdict!r} but ROW_TINT has no colour for it"
        )
        assert style.ROW_TINT[verdict].startswith("#")

    assert style.ROW_TINT["unusable"] != style.ROW_TINT["unavailable"], (
        "'unusable' (never retried) must not look like 'unavailable' (retried "
        "every pass) — that is the distinction the colour exists to carry"
    )

    assert style.row_tint("event", "running") == style.ROW_TINT["running"], (
        "'running' must outrank the stored verdict — a row being worked on now "
        "is the more urgent fact"
    )
    assert style.row_tint(None, None) == style.ROW_TINT["default"]
    assert style.row_tint("not_a_verdict", None) == style.ROW_TINT["default"]


def test_status_text_is_dark_enough_to_read():
    """style.STATUS_* are marks; TEXT_* are the same meanings as words.

    STATUS_GOOD (#0ca30c) and STATUS_WARNING (#fab219) are tuned to be seen as a
    dot or a line, and are far too light to read as 10px text on white — which
    is precisely why the dashboard, the two fit windows and the 2DH grid dialog
    each invented their own darker version instead of using them.  Now there is
    one set; this asserts it stays legible so nobody 'unifies' TEXT_* back into
    STATUS_* and silently makes the status line unreadable.
    """
    from smfs_catalog import style

    def relative_luminance(hex_colour: str) -> float:
        rgb = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
               for c in rgb]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    for name in ("TEXT_GOOD", "TEXT_WARNING", "TEXT_BAD"):
        colour = getattr(style, name)
        contrast = 1.05 / (relative_luminance(colour) + 0.05)
        assert contrast >= 4.5, (
            f"style.{name} = {colour} is {contrast:.2f}:1 on white; body text "
            f"needs 4.5:1"
        )


def test_one_selection_highlight():
    """Six copies of '#2d6cdf' became one constant; both helpers derive from
    it, so a table and a list can never disagree about what 'selected' looks
    like."""
    from smfs_catalog import style

    assert style.SELECTION_BG in style.LIST_QSS
    assert style.SELECTION_BG in style.TABLE_QSS
    assert "QListWidget::item:selected" in style.LIST_QSS
    assert "QTableView::item:selected" in style.TABLE_QSS
    assert "QTableWidget::item:selected" in style.TABLE_QSS


def test_cluster_colours_come_from_the_palette():
    """A cluster is the same colour in every window that draws it.

    pca_window used pg.intColor(label, hues=k) for its two score scatters
    while event_summary, scatter_window, variable_window and
    categorical_window all used style.series_labeled(label).  Since PCA is
    where k-means DEFINES the labels, the definitive view disagreed with
    every window that consumed them — cluster 1 was one colour in the PCA
    tab and another everywhere else.

    pg.intColor walks pyqtgraph's HSV wheel, so it also cycles into new hues
    without limit, which is exactly what SERIES_LABELED exists to stop
    (colours never cycle into a new hue).
    """
    import pathlib
    import re

    pkg = pathlib.Path(__file__).resolve().parent.parent / "smfs_catalog"
    offenders = []
    for path in sorted(pkg.glob("*.py")):
        if path.name == "style.py":
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"\bpg\.intColor\s*\(", line):
                offenders.append(f"{path.name}:{n}  {line.strip()}")

    assert not offenders, (
        "pg.intColor bypasses style.py and cycles hues; use "
        "style.series_labeled(i) so every window agrees on a cluster's "
        "colour:\n  " + "\n  ".join(offenders)
    )
