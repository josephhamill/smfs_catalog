# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Guard: a wave-note key is read from the line that STARTS with it, and the
pulling speed is resolved rather than assumed.

Asylum notes carry many keys ending in a shorter key's name — AmpInvOLS beside
InvOLS, DisplaySpringConstant beside SpringConstant, FMapXYVelocity and
RetractVelocity beside Velocity, OldForceDist beside ForceDist.  An unanchored
search matches the tail of any of them, so a file that simply does not carry a
field comes back holding a lookalike's value.  That is the dangerous direction:
a blank is visible and recoverable, a force map's stage speed reported as a
pulling velocity is neither.

The resolver exists for the mirror-image reason.  Three velocity keys and two
flags describe the ramp, and only one combination of flags has ever been
observed.  Under any other, the speed is left absent rather than guessed.

Run standalone:

    python tests/test_note_keys.py
"""

from __future__ import annotations

from smfs_catalog.curve_loader import _note_float, _retract_velocity, _spring_constant


def _close(a, b):
    """Scaling by 1e9 leaves the last bit adrift; the key is which line was read."""
    return a is not None and abs(a - b) < 1e-9 * max(1.0, abs(b))


def _note(*lines: str) -> bytes:
    """A wave note in the instrument's own layout: \\r-separated, \\r-terminated."""
    return ("\r".join(lines) + "\r").encode("latin-1")


# The collisions that exist in real notes, with the decoy placed FIRST so a
# match by position alone would take the wrong one.
DECOYS = _note(
    "FMapXYVelocity: 2e-05",
    "ApproachVelocity: 1.9841e-06",
    "RetractVelocity: 1.9841e-06",
    "UseVelocity: 0",
    "VelocitySynch: 1",
    "Velocity: 4.9984e-07",
    "OldForceDist: 1e-06",
    "ForceDist: 5e-07",
    "AmpInvOLS: 1e-07",
    "InvOLS: 2.394e-07",
    "DisplaySpringConstant: 1.05e-11",
    "SpringConstant: 0.0105",
)


def test_a_key_is_read_from_its_own_line():
    """Each field takes its own key's value, not a longer key's tail."""
    assert _close(_note_float(DECOYS, b"Velocity", 1e9), 499.84)
    assert _close(_note_float(DECOYS, b"ForceDist", 1e9), 500.0)
    assert _close(_note_float(DECOYS, b"InvOLS", 1e9), 239.4)
    assert _close(_spring_constant(DECOYS), 10.5)


def test_an_absent_key_reads_as_absent_not_as_a_lookalike():
    """A note carrying only the longer keys yields None.

    This is the live failure: images have no `Velocity` line at all, so an
    unanchored search took FMapXYVelocity and stored a stage speed of 20000
    nm/s as though it were a pulling velocity."""
    only_decoys = _note("FMapXYVelocity: 2e-05", "OldForceDist: 1e-06",
                        "AmpInvOLS: 1e-07")
    assert _note_float(only_decoys, b"Velocity", 1e9) is None
    assert _note_float(only_decoys, b"ForceDist", 1e9) is None
    assert _note_float(only_decoys, b"InvOLS", 1e9) is None


def test_a_key_on_the_very_first_line_is_found():
    """The anchor accepts the start of the note, not only a preceding \\r."""
    assert _close(_note_float(_note("Velocity: 4e-07"), b"Velocity", 1e9), 400.0)


def test_a_unit_suffix_does_not_hide_the_number():
    """Some notes write a unit after the value (e.g. StartHeadTemp)."""
    assert _note_float(_note("StartHeadTemp: 24.5 °C"), b"StartHeadTemp") == 24.5


def test_the_observed_configuration_resolves_to_velocity():
    """UseVelocity 0 with VelocitySynch 1: one speed drives both halves."""
    assert _close(_retract_velocity(DECOYS), 499.84)


def test_a_note_without_the_flags_still_resolves():
    """Their absence is not a contradiction — older notes simply omit them."""
    assert _close(_retract_velocity(_note("Velocity: 4e-07")), 400.0)


def test_an_unseen_flag_combination_yields_no_speed():
    """The toggle being flicked must not silently change what the number means.

    Nothing here has observed UseVelocity 1, so which of the three candidate
    speeds then applies is not established.  Absent is recoverable; a wrong
    pulling speed propagates into every rupture force compared against it."""
    flicked = _note("Velocity: 4.9984e-07", "RetractVelocity: 1.9841e-06",
                    "UseVelocity: 1", "VelocitySynch: 0")
    assert _retract_velocity(flicked) is None


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
        print(f"{failed} of {len(checks)} note-key checks failed.")
    else:
        print(f"All {len(checks)} note-key checks pass.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
