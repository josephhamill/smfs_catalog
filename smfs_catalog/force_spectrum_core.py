# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/force_spectrum_core.py
#
# Dynamic force spectroscopy: rupture force against ln(loading rate), fitted
# with the Dudko-Hummer-Szabo mean-force expression
#
#   <F> = (ΔG/(ν x_b)) · [1 − ((kT/ΔG) · ln(kT·k_off·e^(ΔG/kT+γ) / (x_b·r)))^ν]
#
# Each rupture is one point, so least squares estimates the MEAN force, which
# is why Euler's γ appears. ν = 1 is Bell-Evans: ΔG cancels and the curve is a
# straight line in ln r. ν = 2/3 (cusp) and 1/2 (linear-cubic) curve, and the
# curvature is what measures ΔG.
#
# Units: F pN, r pN/s, kT pN·nm, x_b nm, k_off s⁻¹ (reported as log10),
# ΔG in kT. Qt-free and DB-free.

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit

from . import regression
from .dist_fit_core import at_bound_flags
from .models import _k_B

EULER_GAMMA = float(np.euler_gamma)

# Temperature sources in order of preference: the thermal-calibration
# temperature is in K, the head temperature in °C.
TEMPERATURE_KEYS = ("ThermalTemperature", "StartHeadTemp")
_CELSIUS_TO_K = 273.15

# (label, ν). ν = 1 is fitted as a line; the others by nonlinear least squares.
MODELS: tuple[tuple[str, float], ...] = (
    ("Bell-Evans",          1.0),
    ("DHS cusp (ν=2/3)",    2.0 / 3.0),
    ("DHS linear-cubic (ν=1/2)", 0.5),
)

PARAM_NAMES = ("x_b", "log10 k_off", "ΔG")
PARAM_UNITS = ("nm", "log10 s⁻¹", "kT")

# Box for the nonlinear fits, in PARAM_NAMES order.
_LOWS  = (1e-3, -12.0, 1.0)
_HIGHS = (10.0,   6.0, 200.0)
_DG_STARTS = (10.0, 30.0, 80.0)


def temperature_K(meta: list[dict]) -> tuple[float, int]:
    """(median temperature in K, curves that had one) over per-curve metadata.

    Each curve uses ThermalTemperature when stored, else StartHeadTemp. NaN
    when no curve has either."""
    temps = []
    for m in meta:
        t = m.get("ThermalTemperature")
        if t is None and m.get("StartHeadTemp") is not None:
            t = float(m["StartHeadTemp"]) + _CELSIUS_TO_K
        if t is not None and np.isfinite(float(t)):
            temps.append(float(t))
    if not temps:
        return float("nan"), 0
    return float(np.median(temps)), len(temps)


def kT_of(temperature_K: float) -> float:
    """kT in pN·nm."""
    return _k_B * temperature_K


def mean_force(ln_r, x_b, log10_koff, dG, nu, kT):
    """DHS mean rupture force (pN) at ln(loading rate). ν = 1 is Bell-Evans,
    where dG drops out."""
    ln_r = np.asarray(ln_r, dtype=float)
    koff = 10.0 ** log10_koff
    if nu == 1.0:
        return (kT / x_b) * (ln_r + np.log(x_b / (koff * kT)) - EULER_GAMMA)
    # Clip to the model's domain: below 0 the barrier is gone (F = F_c),
    # above 1 the force would be negative (F = 0).
    base = np.clip(_barrier_fraction(ln_r, x_b, log10_koff, dG, kT), 0.0, 1.0)
    return (dG * kT / (nu * x_b)) * (1.0 - base ** nu)


def _barrier_fraction(ln_r, x_b, log10_koff, dG, kT):
    """The DHS bracket (kT/ΔG)·ln(…); the model is defined where it lies in (0, 1)."""
    L = np.log(kT * 10.0 ** log10_koff / x_b) + dG + EULER_GAMMA - np.asarray(ln_r, dtype=float)
    return L / dG


@dataclass
class SpectrumFit:
    """One model fitted to (ln r, F)."""
    label:    str
    nu:       float
    kT:       float
    params:   dict                 # name -> (value, ±1σ); ΔG absent for Bell-Evans
    at_bound: dict = field(default_factory=dict)   # name -> 'lo' | 'hi'
    stats:    dict = field(default_factory=dict)

    def predict(self, ln_r) -> np.ndarray:
        x_b = self.params["x_b"][0]
        lk  = self.params["log10 k_off"][0]
        dG  = self.params.get("ΔG", (1.0, 0.0))[0]
        return mean_force(ln_r, x_b, lk, dG, self.nu, self.kT)

    def in_domain(self, ln_r) -> np.ndarray:
        """Where the model gives a force of its own rather than a clamped
        0 or F_c. Bell-Evans is defined everywhere."""
        ln_r = np.asarray(ln_r, dtype=float)
        if self.nu == 1.0:
            return np.ones(ln_r.shape, dtype=bool)
        frac = _barrier_fraction(ln_r, self.params["x_b"][0],
                                 self.params["log10 k_off"][0],
                                 self.params["ΔG"][0], self.kT)
        return (frac > 0.0) & (frac < 1.0)


def _stats(f: np.ndarray, f_fit: np.ndarray, n_params: int) -> dict:
    """Information criteria on the same per-sample basis as
    dist_fit_core.fit_stats: Gaussian residual log-likelihood, k counting the
    residual variance as a parameter."""
    n = int(f.size)
    rss = float(np.sum((f - f_fit) ** 2))
    ss_tot = float(np.sum((f - f.mean()) ** 2))
    k = n_params + 1
    ll = -0.5 * n * (np.log(2.0 * np.pi * max(rss, 1e-300) / n) + 1.0)
    aic = 2 * k - 2 * ll
    aicc = aic + 2 * k * (k + 1) / max(n - k - 1, 1)
    bic = k * np.log(max(n, 1)) - 2 * ll
    return {
        "R²":             1.0 - rss / max(ss_tot, 1e-300),
        "AIC":            aic,
        "AICc":           aicc,
        "BIC":            bic,
        "RSS":            rss,
        "log-likelihood": ll,
        "n (for IC)":     n,
        "k (for IC)":     k,
    }


def fit_bell_evans(ln_r: np.ndarray, f: np.ndarray, kT: float) -> SpectrumFit | None:
    lf = regression.linear_fit(ln_r, f)
    if lf is None or lf.slope <= 0.0:
        return None
    s, b = lf.slope, lf.intercept
    cov = np.array(lf.slope_cov)
    x_b = kT / s
    x_b_err = kT * lf.slope_se / s ** 2
    ln_koff = -np.log(s) - b / s - EULER_GAMMA
    grad = np.array([-1.0 / s + b / s ** 2, -1.0 / s])
    ln_koff_err = float(np.sqrt(max(grad @ cov @ grad, 0.0)))
    fit = SpectrumFit(
        label="Bell-Evans", nu=1.0, kT=kT,
        params={"x_b": (x_b, x_b_err),
                "log10 k_off": (ln_koff / np.log(10.0), ln_koff_err / np.log(10.0))},
    )
    fit.stats = _stats(f, fit.predict(ln_r), 2)
    return fit


def fit_dhs(ln_r: np.ndarray, f: np.ndarray, kT: float, nu: float, label: str,
            start: SpectrumFit) -> SpectrumFit | None:
    """Nonlinear DHS fit for one ν, started from the Bell-Evans solution at
    several barrier heights; the lowest-RSS result is kept."""
    x_b0 = np.clip(start.params["x_b"][0], _LOWS[0] * 10, _HIGHS[0] / 10)
    lk0  = np.clip(start.params["log10 k_off"][0], _LOWS[1] + 1, _HIGHS[1] - 1)

    def model(x, x_b, lk, dG):
        return mean_force(x, x_b, lk, dG, nu, kT)

    best = None
    for dG0 in _DG_STARTS:
        try:
            popt, pcov = curve_fit(model, ln_r, f, p0=[x_b0, lk0, dG0],
                                   bounds=(_LOWS, _HIGHS), maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        rss = float(np.sum((f - model(ln_r, *popt)) ** 2))
        if best is None or rss < best[0]:
            best = (rss, popt, pcov)
    if best is None:
        return None
    _rss, popt, pcov = best
    perr = np.sqrt(np.clip(np.diag(pcov), 0.0, np.inf))
    flags = at_bound_flags(popt, _LOWS, _HIGHS)
    fit = SpectrumFit(
        label=label, nu=nu, kT=kT,
        params={name: (float(v), float(e)) for name, v, e in zip(PARAM_NAMES, popt, perr)},
        at_bound={name: fl for name, fl in zip(PARAM_NAMES, flags) if fl},
    )
    fit.stats = _stats(f, fit.predict(ln_r), 3)
    return fit


def fit_all(rate: np.ndarray, force: np.ndarray, kT: float) -> tuple[list[SpectrumFit], int]:
    """Every model in MODELS over the paired (rate, force) values.

    Returns (fits that converged, pairs dropped because r ≤ 0 or either value
    is missing). ln r does not exist for r ≤ 0; nothing else is excluded."""
    rate = np.asarray(rate, dtype=float)
    force = np.asarray(force, dtype=float)
    ok = np.isfinite(rate) & np.isfinite(force) & (rate > 0.0)
    n_dropped = int(rate.size - np.count_nonzero(ok))
    ln_r, f = np.log(rate[ok]), force[ok]
    if f.size < 5 or not np.isfinite(kT):
        return [], n_dropped
    bell = fit_bell_evans(ln_r, f, kT)
    if bell is None:
        return [], n_dropped
    fits = [bell]
    for label, nu in MODELS[1:]:
        d = fit_dhs(ln_r, f, kT, nu, label, bell)
        if d is not None:
            fits.append(d)
    return fits, n_dropped
