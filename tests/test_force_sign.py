# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Regression guard: the force sign convention (#79, fixed 2026-07-16).

For months `models.py:4` and `:22` stated the convention backwards — they claimed
wlc() returns negative force under tension. It returns strictly positive. The wrong
comment sat in the two most authoritative places in the polymer-models module and
taught the wrong convention to every reader, repeatedly.

The comments are now correct. This file exists so that the convention is pinned by a
test of BEHAVIOUR rather than by a test watching the prose — a comment can drift back
to being wrong without anything failing, but these assertions cannot.

The convention, in full — there are two spaces:
    raw deflection    (curve.defl_retr)   NEGATIVE under tension
    transformed force (k * defl_corr)     POSITIVE under tension
because invols_slope is itself negative (~-0.9 to -1.4), so dividing by it flips the
sign. Consequence: peak searches in transformed-force space use argmax, not argmin.

Run with the smfs-catalog env, from the repo root:
    python tests/test_force_sign.py
or:
    pytest tests/test_force_sign.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smfs_catalog.models import wlc  # noqa: E402


def test_wlc_returns_positive_force_over_its_whole_domain() -> None:
    """wlc() is strictly non-negative for every physical (x, l_p, l_c).

    This is the claim models.py:22 got backwards. The bracket
    1/(4(1-z)^2) - 0.25 + z has its minimum 0 at z=0 and rises from there, and
    kT/l_p > 0, so the product can never be negative."""
    for l_p in (0.4, 1.0, 5.0, 50.0):
        for l_c in (50.0, 500.0, 2000.0):
            x = np.linspace(0.0, l_c, 2000)
            F = wlc(x, l_p=l_p, l_c=l_c)
            assert np.all(F >= 0.0), (
                f"wlc() returned negative force for l_p={l_p}, l_c={l_c} "
                f"(min={F.min()}). The convention is POSITIVE under tension — see #79 "
                f"and CLAUDE.md §4. If this genuinely changed, the argmax rupture-peak "
                f"search and every downstream sign assumption change with it."
            )


def test_wlc_is_zero_at_zero_extension_and_rises_monotonically() -> None:
    """Pins the shape, which is what makes the sign meaningful.

    If wlc() were ever flipped to negative, this catches it even where the values
    happen to clip at a bound."""
    x = np.linspace(0.0, 1999.0, 4000)
    F = wlc(x, l_p=0.4, l_c=2000.0)
    assert F[0] == 0.0, f"wlc() should be exactly 0 at zero extension, got {F[0]}"
    assert np.all(np.diff(F) >= 0.0), "wlc() should rise monotonically with extension"
    assert F[-1] > F[0], "wlc() should increase from zero extension to near-full"


def test_transformed_force_is_positive_under_tension() -> None:
    """The arithmetic that makes raw-negative become transformed-positive.

    This is the step everyone (including Claude) gets backwards: defl_retr is
    negative AND invols_slope is negative, so the quotient is positive. Reproduced
    here with the real signs rather than trusting a comment."""
    defl_retr = np.array([-0.05, -0.10, -0.20])   # negative under tension
    offset_retr = 0.0
    invols_slope = -1.1                            # consistently negative
    spring_constant = 40.0                         # pN/nm

    defl_corr = (defl_retr - offset_retr) / invols_slope
    force = spring_constant * defl_corr

    assert np.all(defl_corr > 0.0), (
        f"defl_corr should be positive (negative deflection / negative slope), "
        f"got {defl_corr}"
    )
    assert np.all(force > 0.0), f"transformed force should be positive, got {force}"
    assert int(np.argmax(force)) == len(force) - 1, (
        "the largest-magnitude tension should be found by argmax in this space — "
        "this is why the rupture peak search uses argmax, not argmin"
    )


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
        print(f"{failed} of {len(checks)} force-sign checks failed.")
    else:
        print(f"All {len(checks)} force-sign checks pass.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
