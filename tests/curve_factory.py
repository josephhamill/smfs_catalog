# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Whole synthetic ForceCurves for tests that need the real analysis path.

`make_curve(variant)` returns a `curve_loader.ForceCurve` (approach AND retract,
spring constant, sample rate) that `analyse_curve`, `compute_curve_events_coords`
and `fit_segments` all treat as a real curve: contact is found, events are
detected under default parameters, and segments are fitted under all three
rupture algorithms.  Deterministic: noise comes from a fresh
`np.random.default_rng(0)` per call, never a global RNG.

Variants (see VARIANTS): one_rupture, two_rupture, flat, noisy_two_rupture,
two_outer_events (two ruptures with the lever fully relaxed between them, so
the detector sees two separate outer events instead of one).

Conventions, matching the loader (see tests/test_force_sign.py):
  * piezo is positive away from the surface; the retract piezo increases;
  * defl = INVOLS * (bending in nm), INVOLS negative, so raw deflection is
    negative under tension and positive when pressed into the surface;
  * under tension the cantilever bends F/k and the tip sits at
    extension = (z - z_contact) - F/k, i.e. z = extension + F/k.

Geometry (nm of piezo): contact at z = 0, approach from +300 to -150, retract
from -150 to +700.  Chains are Marko-Siggia WLC; each is loaded to a fixed
fraction of its contour length and then lets go.  A chain starts bearing load
well past the 100 nm post-snap-off mask, so the detector's search band sees it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smfs_catalog.curve_loader import ForceCurve
from smfs_catalog.models import wlc

SAMPLE_RATE_HZ = 10_000.0
SPRING_CONSTANT = 8.0           # pN/nm (a soft lever: a rupture is a large jump in nm)
INVOLS = -1.05                  # nm deflection per nm piezo in contact (negative)
STEP_NM = 0.05                  # piezo step per sample (500 nm/s at 10 kHz)
APPROACH_FROM_NM = 300.0
TURNAROUND_NM = -150.0
RETRACT_TO_NM = 700.0
L_P_NM = 0.4                    # persistence length of every chain
BREAK_FRACTION = 0.88           # fraction of contour length where a chain lets go
CONTACT_SIGMA_NM = 0.1          # deflection noise while on the surface


@dataclass(frozen=True)
class Variant:
    """Contour lengths (nm) of the chains loaded in series, and free-lever noise."""
    contours_nm: tuple[float, ...]
    free_sigma_nm: float
    gap_nm: float = 0.0   # relaxed stretch (nm of piezo) between one break and the next chain
    start_nm: float = 0.0  # piezo position where the first chain starts to load


VARIANTS = {
    "one_rupture": Variant((350.0,), 0.5),
    "two_rupture": Variant((350.0, 450.0), 0.5),
    "flat": Variant((), 0.5),
    "noisy_two_rupture": Variant((350.0, 450.0), 1.2),
    "two_outer_events": Variant((160.0, 170.0), 0.4, gap_nm=60.0, start_nm=110.0),
}


def _chain_table(l_c: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(extension, z, force) of one chain on a dense grid; z = x + F/k."""
    x = np.linspace(0.0, 0.99 * l_c, 20_000)
    force = wlc(x, L_P_NM, l_c)
    return x, x + force / SPRING_CONSTANT, force


def _bending_nm(z: np.ndarray, spec: "Variant") -> np.ndarray:
    """Cantilever bending F/k (nm) at each retract position, for chains in series.

    With gap_nm > 0 the lever relaxes (no bending) for gap_nm after each break,
    and the next chain is then loaded from zero extension at that z.  With 0
    the next chain takes over where the last broke, as before.  The first chain
    loads from start_nm (0: from the surface).
    """
    bending = np.zeros_like(z)
    origin = start = spec.start_nm
    for l_c in spec.contours_nm:
        x_tab, z_tab, force = _chain_table(l_c)
        stop = origin + float(np.interp(BREAK_FRACTION * l_c, x_tab, z_tab))
        on = (z >= start) & (z < stop)
        bending[on] = np.interp(z[on] - origin, z_tab, force) / SPRING_CONSTANT
        start = stop
        if spec.gap_nm > 0.0:
            origin = start = stop + spec.gap_nm
    return bending


def _noise(rng: np.random.Generator, z: np.ndarray, free_sigma: float) -> np.ndarray:
    """White deflection noise: quiet on the surface (z < 0), the free level off it."""
    sigma = np.where(z < 0.0, CONTACT_SIGMA_NM, free_sigma)
    return rng.standard_normal(z.size) * sigma


def _approach(rng: np.random.Generator, free_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    z = np.arange(APPROACH_FROM_NM, TURNAROUND_NM, -STEP_NM)
    defl = np.where(z < 0.0, INVOLS * z, 0.0)
    return z, defl + _noise(rng, z, free_sigma)


def _retract(rng: np.random.Generator, variant: Variant) -> tuple[np.ndarray, np.ndarray]:
    z = np.arange(TURNAROUND_NM, RETRACT_TO_NM, STEP_NM)
    defl = np.where(z < 0.0, INVOLS * z, INVOLS * _bending_nm(z, variant))
    return z, defl + _noise(rng, z, variant.free_sigma_nm)


def make_curve(variant: str) -> ForceCurve:
    """The named synthetic curve (a key of VARIANTS), identical on every call."""
    spec = VARIANTS[variant]
    rng = np.random.default_rng(0)
    piezo_appr, defl_appr = _approach(rng, spec.free_sigma_nm)
    piezo_retr, defl_retr = _retract(rng, spec)
    return ForceCurve(
        path=f"synthetic_{variant}.ibw",
        piezo_appr=piezo_appr, defl_appr=defl_appr,
        piezo_retr=piezo_retr, defl_retr=defl_retr,
        spring_constant=SPRING_CONSTANT,
        sample_rate_hz=SAMPLE_RATE_HZ,
        velocity_nm_s=STEP_NM * SAMPLE_RATE_HZ,
        inv_ols_nm_v=None,
    )
