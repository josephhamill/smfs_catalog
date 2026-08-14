# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/models.py
#
# Polymer chain force-extension models and a generic least-squares fitter.
#
# Sign convention — there are two spaces, and this module works in the second:
#   raw deflection   (curve.defl_retr)  is NEGATIVE under tension
#   transformed force (k * defl_corr)   is POSITIVE under tension
# because invols_slope is itself negative, so dividing by it flips the sign.
# The models here take transformed force, and wlc() returns POSITIVE force
# under tension. Peak searches in this space therefore use argmax, not argmin.
#
# Add new models here; the fitter stays the same. If a future model returns a
# different sign, say so in its own docstring — do not restate it up here.

import numpy as np
from scipy.optimize import curve_fit as _curve_fit

_TEMPERATURE = 293.15    # K
_k_B         = 1.38065e-2  # pN nm K⁻¹


def wlc(x: np.ndarray, l_p: float, l_c: float) -> np.ndarray:
    """
    Marko-Siggia worm-like chain (1995).

    x   : extension (nm)
    l_p : persistence length (nm)
    l_c : contour length (nm)

    Returns force in pN, strictly positive: for z in [0, 0.9999] the bracket
    1/(4(1-z)^2) - 0.25 + z is never negative, and kT/l_p > 0.
    z is clipped below 1 to prevent the singularity at full extension.
    """
    kT = _k_B * _TEMPERATURE
    z  = np.clip(x / l_c, 0.0, 0.9999)
    return (kT / l_p) * (1.0 / (4.0 * (1.0 - z) ** 2) - 0.25 + z)


def normalize_wlc(
    x: np.ndarray, F: np.ndarray, l_p: float, l_c: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform (x, F) → dimensionless WLC coordinates.
      x_norm = x / l_c        (fractional extension; singularity at 1)
      F_norm = F * l_p / kT   (dimensionless force)
    All WLC curves collapse onto the universal master curve in these units.
    """
    kT = _k_B * _TEMPERATURE
    return x / l_c, F * l_p / kT


def fit_model(
    model_fn,
    x:      np.ndarray,
    F:      np.ndarray,
    p0:     list[float],
    bounds: tuple = (-np.inf, np.inf),
    maxfev: int   = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Least-squares fit of model_fn(x, *params) to F.
    Returns (popt, pcov) from scipy.optimize.curve_fit.
    Raises RuntimeError if the optimiser fails to converge.
    """
    return _curve_fit(model_fn, x, F, p0=p0, bounds=bounds, maxfev=maxfev)
