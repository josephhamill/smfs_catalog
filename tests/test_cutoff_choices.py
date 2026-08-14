# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Guard test: the spectral-cutoff slider offers only cutoffs that can be USED, and
never displays a number the database does not hold (#141, 2026-08-06).

WHAT WAS WRONG.  The list read

    [100, 200, 500, 1000, 2000, 5000, 10000, 125000, 25000]

in which 125000 is a typo for 12500 AND is out of order, so the slider ran
100 → … → 10000 → 125000 → 25000: dragging right lowered the cutoff.  Worse,
signal_processing.bessel_decompose RAISES at or above Nyquist, and four of those
nine positions are at or above Nyquist for real data — 8,333 Hz for the 16.7 kHz
cohorts that are most of the catalog, 25,000 Hz for the fastest ever recorded.
They were not choices, they were crashes with a tick mark.

THE DISTINCTION THIS TEST PINS, because it is the whole design:

  * Nyquist is a NECESSITY.  The filter is mathematically undefined above it,
    the same class as the WLC fit's l_c > x_max floor, so it is ENFORCED.
  * "500 Hz over-smooths" is POLICY, and CLAUDE.md §4 (#97/#94) says the app
    informs and the user decides.  So the low end stays selectable, and the cost
    of a choice is stated (tau ~ f_s/f_c) rather than the choice being removed.

A test that only checked "the list is sorted" would pass on a list that gates
policy, which is the change this codebase keeps deciding against.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smfs_catalog import decomposition_window as dw     # noqa: E402
from smfs_catalog import signal_processing as sp        # noqa: E402

VALUES = dw._CUTOFF_VALUES

# The sample rates actually present in the live catalog, measured 2026-08-06.
# Nyquist for the commonest is 8,333 Hz, which is what the old list overran.
LIVE_SAMPLE_RATES = (16666.67, 20000.0, 50000.0)


# ── (a) The list itself ──────────────────────────────────────────────────────

def test_values_are_sorted_ascending_and_unique():
    """Slider position IS the index, so an unsorted list makes the control run
    backwards.  This is the 125000-before-25000 bug, caught by shape."""
    assert VALUES == sorted(VALUES), f"cutoff values out of order: {VALUES}"
    assert len(VALUES) == len(set(VALUES)), "duplicate cutoff values"
    assert all(v > 0 for v in VALUES)


def test_every_value_is_usable_on_every_live_sample_rate():
    """The real requirement: a position you can select is a filter that runs.

    Checked against bessel_decompose itself rather than against a copy of its
    rule, so the two cannot drift apart.
    """
    import numpy as np
    for rate in LIVE_SAMPLE_RATES:
        for hz in VALUES:
            assert hz < rate / 2.0, (
                f"{hz} Hz is at or above Nyquist for a {rate:,.0f} Hz cohort — "
                f"selecting it raises rather than filtering")
            sp.bessel_decompose(np.zeros(256), rate, float(hz))


def test_the_studied_values_are_all_reachable():
    """The 2026-08-02 tuning study measured these five.  A value with evidence
    behind it that the slider cannot reach is a measurement nobody can act on."""
    for hz in (500, 1000, 1500, 2000, 3000):
        assert hz in VALUES, f"{hz} Hz was measured but cannot be selected"


def test_the_over_smoothing_end_is_still_offered():
    """NOT a gate.  500 Hz has a residual/noise ratio of 0.37 — it discards
    signal that could have been fitted — but that is a judgement for the person
    looking at the curve, and it is only visible if it can be selected.  Same
    standing decision as the WLC fit's absent l_c ceiling (CLAUDE.md §4)."""
    assert min(VALUES) <= 500


# ── (b) Nyquist is enforced per curve, not per list ──────────────────────────

from PyQt6.QtCore import Qt                             # noqa: E402
from PyQt6.QtWidgets import QApplication, QSlider       # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv)


def _slider() -> QSlider:
    """A REAL QSlider, deliberately.

    A stub cannot express the thing that actually went wrong: QSlider clamps its
    value when the maximum drops below it AND emits valueChanged while doing so,
    which lands in _on_cutoff_slider and writes to the database. A hand-rolled
    stand-in would have passed the "clipping never moves the stored value" check
    while the real widget silently rewrote the cohort's cutoff on a queue scrub.
    """
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(0, len(VALUES) - 1)
    return s


class _Label:
    def __init__(self):
        self.text = ""
        self.visible = False

    def setText(self, t): self.text = t
    def setToolTip(self, t): pass
    def setVisible(self, v): self.visible = v


class _Curve:
    def __init__(self, rate): self.sample_rate_hz = rate


class _Probe:
    """Only the state _refresh_cutoff_limits/_refresh_tau_hint read.  Building a
    real DecompositionWindow needs a QApplication, a database and a curve file;
    a guard that expensive is a guard that gets skipped."""

    _refresh_cutoff_limits = dw.DecompositionWindow._refresh_cutoff_limits
    _refresh_tau_hint      = dw.DecompositionWindow._refresh_tau_hint
    _cutoff_index          = staticmethod(dw.DecompositionWindow._cutoff_index)
    _cutoff_text           = dw.DecompositionWindow._cutoff_text

    def __init__(self, rate, cutoff_hz=2000.0):
        self._current_curve = _Curve(rate) if rate is not None else None
        self._cutoff_hz = cutoff_hz
        self._cutoff_slider = _slider()
        self._cutoff_slider.setValue(self._cutoff_index(cutoff_hz))
        self._cutoff_limit_label = _Label()
        self._tau_label = _Label()
        # Everything _on_cutoff_slider would touch if it ever fired. It must not:
        # a slider clamped by Nyquist is the widget reporting a limit, not the
        # user choosing a value.
        self.writes = []
        self._cutoff_slider.valueChanged.connect(self.writes.append)


@pytest.mark.parametrize("rate", LIVE_SAMPLE_RATES)
def test_live_cohorts_keep_the_whole_slider(rate):
    """The guard must not cost anything on real data: every offered value is
    below Nyquist for every cohort in the catalog, so nothing is held back and
    the warning stays silent."""
    p = _Probe(rate)
    p._refresh_cutoff_limits()
    assert p._cutoff_slider.isEnabled()
    assert p._cutoff_slider.maximum() == len(VALUES) - 1
    assert not p._cutoff_limit_label.visible


def test_a_slow_curve_has_its_reach_clipped_at_nyquist():
    p = _Probe(4000.0)          # Nyquist 2000 → 2000 itself is NOT allowed
    p._refresh_cutoff_limits()
    assert p._cutoff_slider.isEnabled()
    assert VALUES[p._cutoff_slider.maximum()] < 2000.0
    assert p._cutoff_limit_label.visible, "clipped silently"


def test_a_curve_too_slow_for_any_cutoff_disables_the_slider():
    """Real in this catalog: a handful of files carry a header sample rate of
    1 Hz.  Every cutoff is undefined for them, so say so rather than letting the
    next drag raise."""
    p = _Probe(1.0)
    p._refresh_cutoff_limits()
    assert not p._cutoff_slider.isEnabled()
    assert p._cutoff_limit_label.visible


def test_no_curve_offers_the_whole_list():
    """An unknown sample rate is not a reason to invent a limit."""
    p = _Probe(None)
    p._refresh_cutoff_limits()
    assert p._cutoff_slider.isEnabled()
    assert p._cutoff_slider.maximum() == len(VALUES) - 1
    assert not p._cutoff_limit_label.visible


def test_clipping_the_slider_emits_nothing_and_moves_no_stored_value():
    """THE one that needs a real QSlider.

    setMaximum below the current value clamps it and fires valueChanged, which in
    the live window is _on_cutoff_slider — db.set_param, a profile save and
    analysis_params_changed. Unblocked, scrubbing from a 50 kHz curve onto a
    4 kHz one would rewrite the whole cohort's cutoff with no user action.
    """
    p = _Probe(50000.0, cutoff_hz=5000.0)
    p._refresh_cutoff_limits()
    assert p._cutoff_slider.value() == VALUES.index(5000)
    p.writes.clear()

    p._current_curve = _Curve(4000.0)      # Nyquist 2000 — 5000 is now unreachable
    p._refresh_cutoff_limits()

    assert p.writes == [], (
        f"the slider emitted {p.writes} while being clipped — in the real window "
        f"that is a write to the database nobody asked for")
    assert p._cutoff_hz == 5000.0, "the value in force moved"
    assert p._cutoff_slider.value() == p._cutoff_slider.maximum()


def test_widening_the_reach_again_also_stays_silent():
    """The way back: scrub off the slow curve and the reach returns, still
    without anyone having chosen anything."""
    p = _Probe(4000.0, cutoff_hz=5000.0)
    p._refresh_cutoff_limits()
    p.writes.clear()
    p._current_curve = _Curve(50000.0)
    p._refresh_cutoff_limits()
    assert p.writes == []
    assert p._cutoff_hz == 5000.0
    assert p._cutoff_slider.maximum() == len(VALUES) - 1
    # And the handle is back under the value in force, not left where the clamp
    # dropped it.
    assert p._cutoff_slider.value() == VALUES.index(5000)


# ── (c) What is shown is what is stored ─────────────────────────────────────

def test_an_off_list_stored_value_keeps_its_exact_number():
    """_cutoff_index chooses a slider POSITION and nothing else.

    It replaced _snap_cutoff, which returned the nearest listed value and was
    assigned straight to self._cutoff_hz — so a profile holding 2500 Hz was
    filtered, plotted and labelled at 2000 while the database and the batch
    worker used 2500, with nothing on screen saying they disagreed.  That is the
    displayed-vs-stored defect class this codebase treats as a bug, not a
    rounding convenience.
    """
    p = _Probe(16666.67, cutoff_hz=2500.0)
    assert p._cutoff_hz == 2500.0
    assert VALUES[p._cutoff_index(2500.0)] in (2000, 3000)   # position only
    assert "2,500" in p._cutoff_text()
    assert p._cutoff_text().endswith("*"), "off-list value not marked as such"


def test_a_listed_value_is_shown_unmarked():
    assert not _Probe(16666.67, cutoff_hz=2000.0)._cutoff_text().endswith("*")


def test_cutoff_index_round_trips_every_listed_value():
    for i, hz in enumerate(VALUES):
        assert dw.DecompositionWindow._cutoff_index(float(hz)) == i


# ── (d) The advice half ─────────────────────────────────────────────────────

def test_tau_hint_states_the_filtering_floor():
    """tau ~ f_s/f_c (CLAUDE.md §3) is what a cutoff costs the error bars, and it
    is the number the user is actually choosing between."""
    p = _Probe(16666.67, cutoff_hz=2000.0)
    p._refresh_tau_hint()
    assert p._tau_label.visible
    assert "8.3" in p._tau_label.text, p._tau_label.text     # 16666.67 / 2000
    assert "2.9" in p._tau_label.text, p._tau_label.text     # sqrt(8.3)


def test_tau_hint_is_silent_without_a_rate():
    p = _Probe(None)
    p._refresh_tau_hint()
    assert not p._tau_label.visible
