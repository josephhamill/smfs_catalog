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

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import brentq as _brentq
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


# ── Further chain models, one chosen per profile ─────────────────────────────

# Bouchiat et al. (1999) polynomial correction to Marko-Siggia, i = 2..7.
_BOUCHIAT_A = (-0.5164228, -2.737418, 16.07497, -38.87607, 39.49944, -14.17718)


def wlc_bouchiat(x: np.ndarray, l_p: float, l_c: float) -> np.ndarray:
    """Bouchiat et al. (1999) WLC: Marko-Siggia plus a 7th-order polynomial in
    z = x/l_c, accurate to ~0.1% across extension. Force in pN, positive."""
    z = np.clip(np.asarray(x, dtype=float) / l_c, 0.0, 0.9999)
    poly = sum(a * z ** i for i, a in enumerate(_BOUCHIAT_A, start=2))
    return (_k_B * _TEMPERATURE / l_p) * (
        1.0 / (4.0 * (1.0 - z) ** 2) - 0.25 + z + poly)


def _langevin(u: np.ndarray) -> np.ndarray:
    """coth(u) - 1/u, with its series where the difference cancels."""
    u = np.asarray(u, dtype=float)
    small = u < 1e-4
    safe = np.where(small, 1.0, u)
    return np.where(small, u / 3.0, 1.0 / np.tanh(safe) - 1.0 / safe)


def ewlc_extension(F: np.ndarray, l_p: float, l_c: float, k0: float) -> np.ndarray:
    """Odijk (1995) extensible WLC, x(F) in nm; valid at high force."""
    kT = _k_B * _TEMPERATURE
    return l_c * (1.0 - 0.5 * np.sqrt(kT / (F * l_p)) + F / k0)


def fjc_extension(F: np.ndarray, b: float, l_c: float) -> np.ndarray:
    """Freely jointed chain, x(F) in nm; b is the Kuhn length."""
    return l_c * _langevin(F * b / (_k_B * _TEMPERATURE))


def efjc_extension(F: np.ndarray, b: float, l_c: float, k_s: float) -> np.ndarray:
    """Extensible FJC (Smith et al. 1996), x(F) in nm; k_s is the segment
    stretch modulus in pN."""
    return fjc_extension(F, b, l_c) * (1.0 + F / k_s)


# The force grid F(x) is read from, pN, log-spaced. A point below the model's
# lowest extension reads as the floor; one beyond its highest as the ceiling.
# Interpolating ln F across 1024 points is accurate to ~1e-4 relative.
_F_GRID = np.logspace(-3.0, 5.0, 1024)
_LN_F_GRID = np.log(_F_GRID)


def _force_from_extension(extension: Callable, x: np.ndarray, *params) -> np.ndarray:
    """F(x) for a model defined as x(F), read off x(F) evaluated once on
    _F_GRID. Every x(F) here increases monotonically with F, which is what
    makes the grid a valid interpolation table."""
    x_grid = extension(_F_GRID, *params)
    return np.exp(np.interp(np.asarray(x, dtype=float), x_grid, _LN_F_GRID))


def ewlc(x: np.ndarray, l_p: float, l_c: float, k0: float) -> np.ndarray:
    return _force_from_extension(ewlc_extension, x, l_p, l_c, k0)


def fjc(x: np.ndarray, b: float, l_c: float) -> np.ndarray:
    return _force_from_extension(fjc_extension, x, b, l_c)


def efjc(x: np.ndarray, b: float, l_c: float, k_s: float) -> np.ndarray:
    return _force_from_extension(efjc_extension, x, b, l_c, k_s)


@dataclass(frozen=True)
class ChainModel:
    """
    One force-extension model: its parameters in `force`'s argument order,
    each one's fit bounds, and whether l_c must exceed every observed
    extension (true only of inextensible chains, whose force diverges at l_c).
    `stretch_k` names the extensibility modulus, if the model has one.
    `extension` is x(F), for a model defined that way.
    """
    key:       str
    name:      str
    force:     Callable
    params:    tuple[str, ...]
    bounds:    tuple[tuple[float, float], ...]
    lc_floor:  bool
    stretch_k: str | None = None
    extension: Callable | None = None

    def p0(self, l_p: float, l_c: float) -> list[float]:
        """A start point from the Marko-Siggia fit of the same window."""
        guess = {"l_p": l_p, "b": 2.0 * l_p, "l_c": l_c,
                 "k0": 1e3, "k_s": 1e3}
        return [guess[p] for p in self.params]


_INF = np.inf
# A profile's chain_model_idx is a position in this registry. Append only:
# reordering would change what every stored profile means.
CHAIN_MODELS: dict[str, ChainModel] = {m.key: m for m in (
    ChainModel("wlc",  "WLC (Marko-Siggia)", wlc, ("l_p", "l_c"),
               ((0.05, 500.0), (0.0, _INF)), lc_floor=True),
    ChainModel("bwlc", "WLC (Bouchiat)", wlc_bouchiat, ("l_p", "l_c"),
               ((0.05, 500.0), (0.0, _INF)), lc_floor=True),
    ChainModel("ewlc", "Extensible WLC", ewlc, ("l_p", "l_c", "k0"),
               ((0.05, 500.0), (1e-3, _INF), (1.0, 1e7)), lc_floor=False,
               stretch_k="k0", extension=ewlc_extension),
    ChainModel("fjc",  "FJC", fjc, ("b", "l_c"),
               ((0.01, 1000.0), (0.0, _INF)), lc_floor=True,
               extension=fjc_extension),
    ChainModel("efjc", "Extensible FJC", efjc, ("b", "l_c", "k_s"),
               ((0.01, 1000.0), (1e-3, _INF), (1.0, 1e7)), lc_floor=False,
               stretch_k="k_s", extension=efjc_extension),
)}
CHAIN_MODEL_BY_IDX: tuple[str, ...] = tuple(CHAIN_MODELS)


@dataclass(frozen=True)
class ChainFit:
    """One fitted chain model: its CHAIN_MODELS key and its parameter values
    in that model's order. Everything that draws or inverts a stored fit
    asks this, so each segment is read with its own model."""
    model:  str
    params: tuple[float, ...]

    @property
    def chain(self) -> ChainModel:
        return CHAIN_MODELS[self.model]

    @property
    def l_c(self) -> float:
        return self.params[self.chain.params.index("l_c")]

    def force(self, x: np.ndarray) -> np.ndarray:
        return self.chain.force(np.asarray(x, dtype=float), *self.params)

    def x_at_force(self, F_target: float) -> float | None:
        """Extension (nm) where this fit reaches F_target (pN), or None if it
        never does. Read off x(F) where the model has one; otherwise Brent's
        method on the monotone F(x) in (0, l_c)."""
        if self.chain.extension is not None:
            with np.errstate(invalid="ignore"):
                x = float(self.chain.extension(float(F_target), *self.params))
            return x if np.isfinite(x) else None
        x_lo, x_hi = 1e-6 * self.l_c, 0.9999 * self.l_c
        try:
            if float(self.force(x_hi)) < F_target:
                return None   # above the pole: unreachable
            return float(_brentq(lambda x: float(self.force(x)) - F_target, x_lo, x_hi))
        except Exception:
            return None
