# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# tests/test_numeric_ui.py
#
# Guards quantities.py and the arrow-step rule.
#
# Written the way test_palette.py is: RECOMPUTE the claim from the values
# rather than trusting the prose next to them.  The audit that produced this
# module found a comment in style.py claiming a colour separation that was
# measurably false, and 's whole point is that a hand-maintained
# claim rots.  So these tests derive the step from the decimals, the display
# unit from the factor, and the formatting from real measured magnitudes.
#
# Source-level tests (rather than constructing windows) for the same reason the
# style guard is source-level: building every window needs a QApplication, a
# populated DB and real curve files, and a guard that is expensive to run is a
# guard that gets skipped.

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

import pytest

from smfs_catalog import quantities as q

PKG = Path(__file__).resolve().parents[1] / "smfs_catalog"

# quantities.py is where the rule is implemented, so it is the one module
# allowed to call setSingleStep/setDecimals directly.
EXEMPT = {"quantities.py"}

SET_STEP     = re.compile(r"\.setSingleStep\s*\(")
SET_DECIMALS = re.compile(r"\.setDecimals\s*\(")
SET_TRACKING = re.compile(r"\.setKeyboardTracking\s*\(")


def _modules():
    return sorted(p for p in PKG.glob("*.py") if p.name not in EXEMPT)


def _code_lines(path: Path):
    """Every line of real code, comments and strings stripped.

    Same reason test_style_is_single_source tokenises: this codebase's comments
    quote the very patterns being banned ("it was setDecimals(6) with the
    default step"), and a raw grep would flag the explanation of the bug as the
    bug.
    """
    src = path.read_text(encoding="utf-8")
    drop = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            for ln in range(tok.start[0], tok.end[0] + 1):
                drop.add(ln)
    return [(i, ln) for i, ln in enumerate(src.splitlines(), 1) if i not in drop]


# ── The rule itself ───────────────────────────────────────────────────────────

def test_step_is_one_unit_of_the_last_digit_shown():
    """Rule 1, recomputed from decimals for every registered quantity.

    A box showing 0.150 must step to 0.151, never to 0.200.  Before this rule
    22 of 33 numeric inputs stepped by 5x to 1,000,000x their own display
    resolution, so the digits on screen could not be reached with the arrows.
    """
    for key, quantity in q.QUANTITIES.items():
        expected = 1.0 if quantity.integer else 10.0 ** -quantity.decimals
        assert quantity.step == pytest.approx(expected), (
            f"{key}: step {quantity.step} is not one unit of its last "
            f"displayed digit ({expected})"
        )


def test_step_never_grows_with_repeated_presses():
    """No adaptive/accelerated stepping anywhere.

    The owner's call: the arrows are for precision, not travel —
    7, 8, 9, 10, 11, never 7, 8, 9, 10, 20, 30.  Qt offers
    AdaptiveDecimalStepType for the opposite behaviour; nothing may opt in.
    """
    for path in _modules():
        for lineno, line in _code_lines(path):
            assert "AdaptiveDecimalStepType" not in line, (
                f"{path.name}:{lineno} opts into growing steps")
            assert "setStepType" not in line, (
                f"{path.name}:{lineno} sets a step type")


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_no_module_sets_its_own_step_or_decimals(path: Path):
    """configure_spinbox is the only place these are chosen.

    This is the structural fix for "one threshold spinner has one decimal place
    limit, while another has another": they were hand-typed at ten separate
    construction sites, so no two agreed and nothing could notice.
    """
    bad = [f"{path.name}:{n}" for n, line in _code_lines(path)
           if SET_STEP.search(line) or SET_DECIMALS.search(line)]
    assert not bad, (
        "set step/decimals directly instead of quantities.configure_spinbox: "
        + ", ".join(bad))


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_no_module_re_enables_keyboard_tracking(path: Path):
    """Keyboard tracking, kept shut off.

    With tracking on (Qt's default) a spin box re-commits on every keystroke,
    so typing 0.19 into a box holding 0.1 commits 0.1 and leaves 0.190000 on
    display.  configure_spinbox turns it off for every box; nothing may turn it
    back on.
    """
    bad = [f"{path.name}:{n}" for n, line in _code_lines(path)
           if SET_TRACKING.search(line)]
    assert not bad, "re-enables keyboard tracking: " + ", ".join(bad)


# ── Units, and the store-vs-display split ─────────────────────────────────────

def test_no_quantity_splits_its_unit_silently():
    """Rule 2, in its current state: nothing converts, and nothing needs to.

    Storing the decomposition thresholds in nm² and showing them in Å², with
    the factor living only as `* 100` inside decomposition_window, is what makes
    an export manifest record 0.00792 for a threshold the user set to 0.79,
    under a key whose name carries no unit at all.  They are stored and shown in
    nm², so no split exists.

    The mechanism stays and is exercised below; this asserts that no quantity
    is quietly using it, so "stored unit == shown unit" holds everywhere and a
    reader of a manifest needs no conversion table.
    """
    for key, quantity in q.QUANTITIES.items():
        assert quantity.shown_unit == quantity.unit, (
            f"{key} shows {quantity.shown_unit!r} but stores {quantity.unit!r} — "
            f"legitimate, but it must be a deliberate, documented choice "
            f"(quantities.py rule 2), not an accident"
        )
        assert quantity.display_factor == 1.0


def test_the_display_split_mechanism_still_works():
    """Kept honest even with no users: it is rule 2's enforcement point.

    An unexercised mechanism is one nobody can trust when the next quantity
    needs it, and the alternative — a bare factor in one window — is the bug
    that produced the manifest above.
    """
    split = q.Quantity(q.NM2, 4, display_unit="Å²", display_factor=100.0)
    assert split.shown_unit == "Å²" and split.unit == q.NM2
    assert split.to_display(0.0079) == pytest.approx(0.79)
    assert split.to_stored(0.79) == pytest.approx(0.0079)


def test_every_quantity_that_converts_declares_both_units():
    """A display_factor without a display_unit is a silent conversion — the
    exact shape of the bug above, in a new place."""
    for key, quantity in q.QUANTITIES.items():
        if quantity.display_factor != 1.0:
            assert quantity.display_unit, (
                f"{key} converts by {quantity.display_factor} but names no "
                f"display unit")


def test_manifest_declares_a_unit_for_every_parameter():
    """Every param_set key an export records has a stated unit.

    Not every unit is non-empty — several of these quantities are genuinely
    dimensionless — but every key must be PRESENT, so a reader months later can
    tell "dimensionless" from "nobody wrote it down".
    """
    from smfs_catalog import db as _db
    units = q.units_for(_db.PARAM_KEYS)
    missing = sorted(k for k in _db.PARAM_KEYS if k not in units)
    assert not missing, f"no unit declared for: {missing}"
    assert set(units) == set(_db.PARAM_KEYS)


def test_slope_units_name_both_measured_axes():
    """Dimensionless arithmetic does not justify an anonymous display unit."""
    for key in ("flatness_slope", "invols_slope",
                "roi_threshold_nm_per_nm", "roi_inner_threshold_nm_per_nm",
                "roi_prominence"):
        assert q.unit_of(key) == q.NM_PER_NM


# ── Formatting, against real measured magnitudes ──────────────────────────────

# Medians and extremes taken from a real catalog.  Hard-coded so this test
# states the magnitudes it is protecting rather than needing a database to
# run.
REAL_MAGNITUDES = {
    "seg_l_p_nm":     [0.05, 0.251029, 7.78787, 500.0],
    "seg_l_c_nm":     [15.03, 103.6485, 411.0, 2000.0],
    "seg_l_p_err":    [0.0001627, 0.00572633, 1.5547],
    "seg_force_pN":   [-718.8, 166.234, 352.7, 2452.0],
    "flatness_slope": [1.69e-05, 0.0008249, 1.447],
    "invols_slope":   [-1.679, -1.00812, -0.5132],
    "seg_n_segments": [1.0, 2.0, 6.0],
}


def test_no_real_measurement_is_ever_displayed_as_zero():
    """A nonzero measurement must never render as 0.00 / 0.0000.

    This is the concrete defect the ROI Explorer's blanket .2f caused: 47% of
    l_p_err values read exactly "0.00".  Showing a real number as zero is worse
    than showing it imprecisely — it reads as an absent or null result.
    """
    for key, values in REAL_MAGNITUDES.items():
        for v in values:
            if v == 0.0:
                continue
            text = q.format_value(key, v)
            assert float(text) != 0.0, (
                f"{key}={v!r} displays as {text!r}, i.e. zero")


def test_missing_values_are_blank_never_zero_or_nan():
    for key in ("seg_l_c_nm", "seg_n_segments", "flatness_slope"):
        assert q.format_value(key, None) == ""
        assert q.format_value(key, float("nan")) == ""
        assert q.format_value(key, float("inf")) == ""
        assert q.format_value(key, float("-inf")) == ""


def test_counts_display_as_counts():
    """seg_n_segments is a count of 1-6.  Offering it with six decimal places
    is how a stored gate bound ends up reading `0.701937 <= x <= 3.704446` for
    a number of segments."""
    assert q.format_value("seg_n_segments", 2.0) == "2"
    assert q.get("seg_n_segments").integer
    assert q.get("seg_n_segments").step == 1.0


def test_format_is_stable_across_windows():
    """The same value formats identically wherever it is shown — the whole
    point of a register.  Force read 166, 166.2 and 166.20 in three windows
    before this."""
    for key, values in REAL_MAGNITUDES.items():
        for v in values:
            assert q.format_value(key, v) == q.format_value(key, v)
            with_unit = q.format_value(key, v, with_unit=True)
            assert with_unit.startswith(q.format_value(key, v))


# ── Quantising the write path ─────────────────────────────────────────────────

def test_quantize_makes_screen_and_database_agree():
    """After quantize, what the box displays IS what the database holds.

    The decomposition drag stored the raw mouse position and displayed a
    rounded copy, so the two disagreed and the next nudge wrote the displayed
    one back — silently moving a real analysis parameter by 1.5%.
    """
    key = "detection_threshold_appr"
    quantity = q.get(key)
    raw = 0.001623588188512572          # a real stored value, from the live DB
    stored = q.quantize(key, raw)
    # What the box shows, parsed back, is exactly what is on disk.
    assert quantity.to_stored(float(q.format_value(key, stored))) == pytest.approx(stored)
    # And quantising twice changes nothing.
    assert q.quantize(key, stored) == pytest.approx(stored)


def test_quantize_leaves_no_float_noise():
    """0.1624 / 100 is 0.0016239999999999998 in binary floating point, and this
    value goes into params_json and every export manifest."""
    stored = q.quantize("detection_threshold_appr", 0.001623588188512572)
    assert repr(stored) == "0.001624", repr(stored)


def test_format_value_honours_declared_display_conversion():
    """Formatting is part of the conversion boundary, not a stored-unit leak."""
    original = q.QUANTITIES.get("split_test")
    q.QUANTITIES["split_test"] = q.Quantity(
        q.NM2, 2, display_unit="display-unit", display_factor=100.0)
    try:
        assert q.format_value("split_test", 0.0079) == "0.79"
        assert q.format_value("split_test", 0.0079, with_unit=True) == \
            "0.79 display-unit"
    finally:
        if original is None:
            del q.QUANTITIES["split_test"]
        else:
            q.QUANTITIES["split_test"] = original


def test_units_for_returns_only_requested_registered_keys():
    units = q.units_for(("spectral_cutoff_hz", "seg_force_pN", "unknown"))
    assert units == {"spectral_cutoff_hz": q.HZ, "seg_force_pN": q.PN}


def test_decimals_widen_rather_than_round_a_stored_bound():
    """A stored bound finer than its quantity's natural precision must be
    shown in full, not rounded to fit.

    The live DB holds `seg_n_segments <= 3.704446`, a p95 seed accepted through
    a six-decimal box.  Rounding that to <= 4 on display would change
    which curves are hits — a four-segment curve fails one and passes the
    other.  Widening is not a cosmetic nicety here; it is what stops a display
    change from silently re-gating a cohort.
    """
    assert q.get("seg_n_segments").decimals == 0
    assert q.decimals_for("seg_n_segments", 3.704446) == 6
    assert q.decimals_for("seg_n_segments", 4.0) == 0     # nothing to preserve
    assert q.decimals_for("seg_force_pN", 166.2) == 1


def test_audit_reports_rather_than_changes():
    """audit_stored_precision names the mismatches and touches nothing."""
    stored = {"seg_n_segments": 3.704446, "seg_force_pN": 166.2}
    before = dict(stored)
    found = q.audit_stored_precision(stored)
    assert stored == before, "audit must not modify what it is given"
    assert [k for k, _v, _d in found] == ["seg_n_segments"]


# ── The register itself ───────────────────────────────────────────────────────

def test_every_seg_summary_key_is_registered():
    """The queue's own key list and this register cannot drift apart."""
    from smfs_catalog.roi_pipeline import SEG_SUMMARY_KEYS
    missing = [k for k in SEG_SUMMARY_KEYS if k not in q.QUANTITIES]
    assert not missing, f"unregistered queue columns: {missing}"


def test_unknown_keys_fall_back_instead_of_raising():
    """A key nobody has registered is a display question, not an error — a
    newly added result column must not take a window down."""
    assert q.get("something_nobody_registered") is q.GENERIC
    assert q.format_value("something_nobody_registered", 1.23456789) == "1.2346"
    assert q.unit_of("something_nobody_registered") == ""


def test_quantities_holds_no_values_and_does_no_io():
    """This module describes quantities; it must never read or write one.

    Keeping it free of db/Qt imports is what lets it be imported from anywhere
    — including headless tests and the export layer, which db.py cannot import
    back (see export_utils' own note on that cycle).
    """
    src = (PKG / "quantities.py").read_text(encoding="utf-8")
    for banned in ("import db", "from .db", "import sqlite3", "QApplication"):
        assert banned not in src, f"quantities.py must not {banned!r}"


# ── Drags write what the box shows ────────────────────────────────────────────

def test_every_drag_handler_quantises_before_it_stores():
    """A dragged line hands back a raw mouse position; a spin box shows digits.

    Found after the nm²/Å² version of this had already been fixed:
    display_roi's three threshold handlers and decomposition's anchor handler
    still wrote the raw float straight to the database while the spin box beside
    each of them displayed it rounded to its own decimals.  The two then
    disagreed, and the next arrow press wrote the DISPLAYED number back over a
    real analysis parameter — the same trap, reached through a drag instead of
    a keystroke.

    It is also why every profile in the live database holds values like
    `roi_threshold_nm_per_nm = 1.8963198820547147` — nobody typed sixteen
    digits, the mouse produced them.  Quantising at the write path is what stops
    new ones appearing; the existing ones are left alone deliberately, because
    changing a stored parameter changes `event_map_params_json` and would
    invalidate every cached analysis result in the database.

    The check is that the handler quantises AT ALL — one exempt handler is how
    the four above stayed broken after the fifth was fixed.
    """
    handler = re.compile(r"def (_on_\w*line_moved)\b")
    bad = []
    for p in _modules():
        src = p.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not handler.match(f"def {node.name}"):
                continue
            body = ast.dump(node)
            if "quantize" not in body:
                bad.append(f"{p.name}:{node.lineno}: {node.name}")
    assert not bad, (
        "drag handler stores a raw mouse position — call quantities.quantize "
        "before the value reaches the spin box or the database:\n  "
        + "\n  ".join(bad)
    )
