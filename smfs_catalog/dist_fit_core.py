# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/dist_fit_core.py
#
# Pure math for histogram distribution fitting.
# Used by the distribution-fit window and by other fit/uncertainty consumers.
#
# Distributions : Gaussian, LogNormal, Gamma, Weibull
# Multi-peak    : arbitrary sums of the above
# GOF           : R², Reduced χ², AIC, AICc, BIC

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy import optimize, special
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d


# ── Palette ───────────────────────────────────────────────────────────────────
# ── Distribution PDFs ─────────────────────────────────────────────────────────

def _pos(x: np.ndarray) -> np.ndarray:
    """Replace non-positive values with NaN so log-based PDFs stay finite."""
    return np.where(x > 0, x, np.nan)


def pdf_gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def pdf_lognormal(x, amp, mu_log, sigma_log):
    xp = _pos(x)
    return amp * np.where(
        x > 0,
        np.exp(-0.5 * ((np.log(xp) - mu_log) / sigma_log) ** 2)
        / (xp * sigma_log * np.sqrt(2 * np.pi)),
        0.0,
    )


def pdf_gamma(x, amp, k, theta):
    xp = _pos(x)
    return amp * np.where(
        x > 0,
        xp ** (k - 1) * np.exp(-xp / theta) / (theta ** k * special.gamma(k)),
        0.0,
    )


def pdf_weibull(x, amp, k, scale):
    return amp * np.where(
        x > 0,
        (k / scale) * (x / scale) ** (k - 1) * np.exp(-(x / scale) ** k),
        0.0,
    )


# ── Model registry ────────────────────────────────────────────────────────────

class ModelSpec:
    def __init__(self, name, param_names, pdf_fn, bounds_fn, guess_fn):
        self.name        = name
        self.param_names = param_names
        self.pdf_fn      = pdf_fn
        self.bounds_fn   = bounds_fn
        self.guess_fn    = guess_fn

    @property
    def n_params(self):
        return len(self.param_names)


def _s(data: np.ndarray) -> dict:
    d   = data[np.isfinite(data)]
    pos = d[d > 0]
    return dict(
        mn=d.min(), mx=d.max(),
        mean=d.mean(), std=max(d.std(), 1e-12),
        rng=max(d.max() - d.min(), 1e-12),
        log_mean=float(np.log(pos).mean())           if len(pos) > 0 else 0.0,
        log_std=max(float(np.log(pos).std()), 0.15)  if len(pos) > 1 else 0.3,
        pos_mean=float(pos.mean())                    if len(pos) > 0 else 1.0,
    )


MODELS: dict[str, ModelSpec] = {}


def _reg(name, params, pdf_fn, bounds_fn, guess_fn):
    MODELS[name] = ModelSpec(name, params, pdf_fn, bounds_fn, guess_fn)


_reg("Gaussian", ["amp", "μ", "σ"], pdf_gaussian,
     lambda d: ([1e-9, _s(d)["mn"] - _s(d)["rng"], 1e-9],
                [np.inf, _s(d)["mx"] + _s(d)["rng"], _s(d)["rng"] * 3]),
     lambda d, mu: [1.0,
                    mu if mu is not None else _s(d)["mean"],
                    _s(d)["std"] * 0.5])

_reg("LogNormal", ["amp", "μ_log", "σ_log"], pdf_lognormal,
     lambda d: ([1e-9,
                 np.log(max((d[d > 0]).min(), 1e-10)) if (d > 0).any() else -10,
                 0.01],
                [np.inf, np.log(max(d.max(), 1e-10)) + 3, 5.0]),
     lambda d, mu: [1.0,
                    np.log(max(mu, 1e-10)) if mu is not None else _s(d)["log_mean"],
                    _s(d)["log_std"]])

_reg("Gamma", ["amp", "k", "θ"], pdf_gamma,
     lambda d: ([1e-9, 0.01, 1e-10], [np.inf, np.inf, np.inf]),
     lambda d, mu: (lambda s: [1.0,
                                max((s["mean"] / s["std"]) ** 2, 0.1),
                                max(s["std"] ** 2 / max(s["mean"], 1e-10), 1e-10)])(_s(d)))

_reg("Weibull", ["amp", "k", "scale"], pdf_weibull,
     lambda d: ([1e-9, 0.01, 1e-10], [np.inf, 20.0, np.inf]),
     lambda d, mu: [1.0, 1.5, max(_s(d)["pos_mean"], 1e-10)])

MODEL_NAMES = list(MODELS.keys())


# ── Composite model helpers ───────────────────────────────────────────────────

def make_composite(components: list[ModelSpec]):
    def composite(x, *params):
        y = np.zeros(len(x), dtype=float)
        i = 0
        for comp in components:
            n = comp.n_params
            y += comp.pdf_fn(x, *params[i:i + n])
            i += n
        return y
    return composite


# ── Parameter constraints ─────────────────────────────────────────────────────
#
# A constraint is PRIOR INFORMATION THE USER SUPPLIED, not something the data
# said, and every number downstream of one has to keep saying so.  The bound
# itself is nothing new — every model has always had a `bounds_fn` — so this is
# an override of an existing box, not a new mechanism.
#
# The override REPLACES the automatic limit rather than intersecting with it: a
# user who knows a scale from an independent calibration may legitimately want a
# range the automatic heuristic would not have allowed, and silently keeping the
# tighter of the two would apply a bound they did not ask for.
#
# Ranges only.  `least_squares` requires lb < ub strictly, so a pin would have
# to be either a degenerately narrow interval or the removal of the parameter
# from the fit — and a removed parameter has to leave `k` in fit_stats as well,
# or the information criteria are counting a freedom that no longer exists.

# A fitted value this close to its limit is sitting ON it: trf converges to
# within solver tolerance of an active bound rather than exactly onto it.
AT_BOUND_RTOL = 1e-6
AT_BOUND_ATOL = 1e-12


def composite_bounds(components: list[ModelSpec], data: np.ndarray,
                     constraints: Optional[list] = None):
    """Flat (lows, highs) for the composite, with the user's limits applied.

    `constraints[i]`, when given, is a per-parameter list of (lo, hi) for
    component i, either element None to leave that limit automatic.  Anything
    the user did not set keeps exactly the value `bounds_fn` produced, so the
    unconstrained answer is unchanged until somebody sets a limit.

    Raises ValueError when a supplied limit leaves an empty interval, naming
    the parameter.  Quietly widening it back would fit a model the user did not
    ask for and report it under their constraint.
    """
    lows, highs = [], []
    for i, comp in enumerate(components):
        lo, hi = comp.bounds_fn(data)
        lo, hi = list(lo), list(hi)
        want = (constraints[i] if constraints and i < len(constraints) else None)
        for j in range(comp.n_params):
            if not want or j >= len(want) or want[j] is None:
                continue
            w_lo, w_hi = want[j]
            if w_lo is not None:
                lo[j] = float(w_lo)
            if w_hi is not None:
                hi[j] = float(w_hi)
            if not lo[j] < hi[j]:
                raise ValueError(
                    f"Peak {i + 1} {comp.param_names[j]}: "
                    f"minimum {lo[j]:.6g} must be below maximum {hi[j]:.6g}."
                )
        lows.extend(lo)
        highs.extend(hi)
    return lows, highs


def flatten_constraints(components: list[ModelSpec],
                        constraints: Optional[list]) -> list:
    """Per-parameter (lo, hi) of what the USER set, None where they set nothing.

    Flat and in popt order, so it lines up with every other per-parameter array
    and survives the same permutation.  Distinguishing a user's limit from the
    automatic one is the whole point: only the former makes an interval
    conditional, and only the former is reported as an imposition.
    """
    flat = []
    for i, comp in enumerate(components):
        want = (constraints[i] if constraints and i < len(constraints) else None)
        for j in range(comp.n_params):
            flat.append(want[j] if want and j < len(want) else None)
    return flat


def at_bound_flags(popt, lows, highs) -> list:
    """Per parameter: 'lo', 'hi', or None — which limit the fit is sitting on.

    A parameter at its limit was not measured.  Its value is the limit, and so
    is the near edge of any interval around it, because every bootstrap refit
    was handed the same box.
    """
    out = []
    for p, lo, hi in zip(np.asarray(popt, dtype=float), lows, highs):
        tol_lo = AT_BOUND_ATOL + AT_BOUND_RTOL * abs(float(lo))
        tol_hi = AT_BOUND_ATOL + AT_BOUND_RTOL * abs(float(hi))
        if np.isfinite(lo) and abs(p - float(lo)) <= tol_lo:
            out.append("lo")
        elif np.isfinite(hi) and abs(p - float(hi)) <= tol_hi:
            out.append("hi")
        else:
            out.append(None)
    return out


def composite_guess(
    components:  list[ModelSpec],
    data:        np.ndarray,
    bin_centers: np.ndarray,
    density:     np.ndarray,
    starts:      Optional[list] = None,
) -> list[float]:
    """
    Build initial parameter guesses anchored to peaks detected in the histogram.
    Falls back to evenly-spaced positions when fewer peaks are detected than needed.

    `starts[i]`, when not None, REPLACES the position this function would have
    chosen for component i. Everything else about that component — and
    every component the user did not touch — is unchanged, so the automatic
    answer is exactly what it always was until somebody sets a value.

    It is a starting point and nothing more: no bound is narrowed and no
    parameter is fixed, so the fit can and usually does move away from it.
    That is deliberate: user-supplied positions are initialization hints, not
    fixed parameters. The window owns that control; this function owns how the
    hint changes the numerical starting vector.
    """
    n      = len(components)
    s      = _s(data)
    smooth = uniform_filter1d(density, size=max(2, len(density) // 8))

    min_dist  = max(2, len(bin_centers) // (2 * n + 1))
    peaks_idx, props = find_peaks(
        smooth, prominence=smooth.max() * 0.07, distance=min_dist,
    )

    if len(peaks_idx) >= n:
        order        = np.argsort(props["prominences"])[::-1]
        peaks_idx    = np.sort(peaks_idx[order[:n]])
        mu_centers   = bin_centers[peaks_idx].astype(float)
        peak_heights = density[peaks_idx].astype(float)
    else:
        mu_centers   = np.linspace(s["mn"] + s["rng"] * 0.15,
                                   s["mx"] - s["rng"] * 0.15, n)
        peak_heights = np.full(n, density.max() / max(n, 1))
        for i, pi in enumerate(sorted(peaks_idx)[:n]):
            mu_centers[i]   = bin_centers[pi]
            peak_heights[i] = density[pi]
        order = np.argsort(mu_centers)
        mu_centers, peak_heights = mu_centers[order], peak_heights[order]

    # The user's own positions replace whatever the finder chose, one component
    # at a time.  Applied BEFORE the widths are derived below, so a hand-placed
    # component gets a width consistent with where it actually sits rather than
    # one computed from a position it no longer has.  Its height is read off
    # the histogram at that x, which is a better amplitude seed than the
    # detected peak's for a peak the detector did not find.
    if starts:
        for i, want in enumerate(starts[:n]):
            if want is None or not np.isfinite(want):
                continue
            mu_centers[i] = float(want)
            j = int(np.argmin(np.abs(bin_centers - float(want))))
            peak_heights[i] = float(density[j])

    if n > 1:
        # Distance to the NEAREST OTHER component, per component.  Identical to
        # the previous left/right-spacing rule whenever the centres are sorted,
        # and still correct when they are not — a user-set start may place
        # component 2 to the left of component 1, and each component must keep
        # its own identity rather than be re-sorted into someone else's slot.
        gaps = np.abs(mu_centers[:, None] - mu_centers[None, :])
        np.fill_diagonal(gaps, np.inf)
        sigmas = gaps.min(axis=1) / 3.0
    else:
        sigmas = np.array([s["std"] * 0.6])
    bw_min = (bin_centers[-1] - bin_centers[0]) / max(len(bin_centers), 1)
    sigmas = np.maximum(sigmas, bw_min * 2)

    guess = []
    for i, comp in enumerate(components):
        mu  = float(mu_centers[i])
        sig = float(sigmas[i])
        h   = float(peak_heights[i])
        if comp.name == "Gaussian":
            amp = max(h * sig * np.sqrt(2 * np.pi), 1e-6)
            guess += [amp, mu, max(sig, 1e-6)]
        elif comp.name == "LogNormal":
            mu_log    = np.log(max(mu, 1e-10))
            sigma_log = float(np.clip(sig / max(mu, 1e-10), 0.05, 2.0))
            x_mode    = np.exp(mu_log - sigma_log ** 2)
            amp       = max(h * x_mode * sigma_log * np.sqrt(2 * np.pi), 1e-6)
            guess += [amp, mu_log, sigma_log]
        elif comp.name == "Gamma":
            theta = max(sig ** 2 / max(mu, 1e-10), 1e-10)
            k_val = max(mu / max(theta, 1e-10), 0.01)
            guess += [max(h, 1e-6), k_val, theta]
        elif comp.name == "Weibull":
            guess += [max(h, 1e-6), 1.5, max(mu, 1e-6)]
    return guess


# ── Post-fit component sorting ────────────────────────────────────────────────

def _peak_centre(comp: ModelSpec, params: np.ndarray) -> float:
    """Representative x-position for a fitted component (for left-to-right ordering)."""
    if comp.name == "Gaussian":  return float(params[1])
    if comp.name == "LogNormal": return float(np.exp(params[1]))
    if comp.name == "Gamma":     return float(params[1] * params[2])
    if comp.name == "Weibull":   return float(params[2])
    return 0.0


# Sampling parameters for total_fit_ci below. Named constants, not bare
# defaults, because every export that carries a CI band must record how it was
# generated — a Monte Carlo band is only reproducible alongside its draw count
# and seed, the same reason pca_window records the k-means seed. A manifest
# that hardcoded "400" separately from the signature could silently drift.
CI_N_DRAWS = 400
CI_PCT     = 95.0
CI_SEED    = 0


COV_CI_METHOD = ("Monte Carlo over the full parameter covariance; "
                 "pointwise percentile envelope")


def ci_manifest_fields(pcov, band_drawn: bool, *,
                       method: str = COV_CI_METHOD,
                       n_draws: int = CI_N_DRAWS,
                       pct: float = CI_PCT,
                       seed: int = CI_SEED,
                       extra: Optional[dict] = None) -> dict:
    """Manifest fields describing a drawn confidence band, for an export.

    Lives next to the two estimators so the description can never drift from
    the thing described — every window that draws one of these bands calls
    this rather than restating the method.

    The keyword arguments exist because the two estimators are NOT the same
    and a manifest that said they were would be lying about one of them:
    mean_curve_window samples the fit's covariance (the defaults here), while
    dist_fit_window bootstraps the raw values because its covariance
    is computed from 20 binned heights and knows nothing about how many
    curves are behind each one.  `method` is the field a reader checks first,
    so it must name what actually ran.

    Carries the FULL covariance matrix whenever there is one, not just its
    diagonal — the components are correlated and the diagonal alone
    overstates the spread (see total_fit_ci).  For a bootstrapped fit the
    matrix is still recorded because it is a property of the fit. It is not
    the source of the reported interval; `ci_method` identifies that source.
    """
    cov = np.asarray(pcov, dtype=float) if pcov is not None else None
    out = {
        "ci_band_drawn": bool(band_drawn),
        "ci_method":     method,
        "ci_pct":        pct,
        "ci_n_draws":    n_draws,
        "ci_seed":       seed,
        "param_covariance": (
            cov.tolist() if cov is not None and np.isfinite(cov).all() else None),
    }
    if extra:
        out.update(extra)
    return out


def total_fit_ci(fn, x, popt, pcov, n_draws: int = CI_N_DRAWS,
                 pct: float = CI_PCT, seed: int = CI_SEED):
    """Pointwise confidence band for the TOTAL fitted curve.

    The parameter covariance from curve_fit has always been available here —
    `perr`/`ci_lo`/`ci_hi` were derived from it, written into the results table
    and into the export — but nothing ever drew it, so on screen a fit looked
    exactly as certain as it looked, which is the one thing a fit is not.

    Draws parameter vectors from the fit's own multivariate normal (the FULL
    covariance, not just its diagonal — components of a mixture are strongly
    correlated, and using only the diagonal would overstate the band), evaluates
    the model for each, and returns the pointwise percentile envelope.

    Returns (lo, hi), or None if the covariance is unusable — curve_fit returns
    inf covariance for an unconstrained parameter, and a band is meaningless
    then.  None is the honest answer there, not a fabricated interval.
    """
    pcov = np.asarray(pcov, dtype=float)
    if pcov.size == 0 or not np.isfinite(pcov).all():
        return None
    rng = np.random.default_rng(seed)
    try:
        draws = rng.multivariate_normal(np.asarray(popt, dtype=float), pcov,
                                        size=n_draws)
    except (np.linalg.LinAlgError, ValueError):
        return None
    with np.errstate(all="ignore"):
        ys = np.array([fn(x, *p) for p in draws], dtype=float)
    ok = np.isfinite(ys).all(axis=1)
    if ok.sum() < 20:
        return None
    ys = ys[ok]
    half = (100.0 - pct) / 2.0
    return (np.percentile(ys, half, axis=0), np.percentile(ys, 100.0 - half, axis=0))


def centre_permutation(components: list[ModelSpec], popt) -> list[int]:
    """Indices of `components` in ascending fitted-centre order.

    The ONE place that ordering is derived, so anything travelling alongside
    the fitted parameters — a user's starting position, a colour, a label —
    can be permuted the same way instead of a caller re-deriving it and
    drifting.
    """
    popt = np.asarray(popt, dtype=float)
    centres, idx = [], 0
    for comp in components:
        centres.append(_peak_centre(comp, popt[idx:idx + comp.n_params]))
        idx += comp.n_params
    return sorted(range(len(components)), key=lambda i: centres[i])


def sort_components_by_centre(components, popt, perr, ci_lo, ci_hi, colors, labels):
    """
    Re-order fitted components by ascending centre position so Peak 1 is always
    the leftmost peak regardless of which order the user added components.
    """
    slices, idx = [], 0
    for comp in components:
        n = comp.n_params
        slices.append(dict(
            comp=comp,
            popt=popt[idx:idx + n],
            perr=perr[idx:idx + n],
            ci_lo=ci_lo[idx:idx + n],
            ci_hi=ci_hi[idx:idx + n],
        ))
        idx += n
    slices.sort(key=lambda s: _peak_centre(s["comp"], s["popt"]))
    s_comp  = [s["comp"]  for s in slices]
    s_popt  = np.concatenate([s["popt"]  for s in slices])
    s_perr  = np.concatenate([s["perr"]  for s in slices])
    s_cilo  = np.concatenate([s["ci_lo"] for s in slices])
    s_cihi  = np.concatenate([s["ci_hi"] for s in slices])
    perm     = centre_permutation([s["comp"] for s in slices],
                                  np.concatenate([s["popt"] for s in slices]))
    s_colors = [colors[i] for i in perm]
    s_labels = [f"Peak {r + 1} ({s_comp[r].name})" for r in range(len(s_comp))]
    return s_comp, s_popt, s_perr, s_cilo, s_cihi, s_colors, s_labels


# ── Bootstrap confidence intervals ──────────────────────────────────────────
#
# The covariance is not used for the reported confidence interval.
#
# The fit is least squares on `n_bins` histogram HEIGHTS (20 by default).  A few
# hundred measured values are compressed into 20 numbers, and nothing in that
# objective records how many curves are behind each bar — so curve_fit's
# covariance cannot know the sample size, and its ±1.96σ interval is a symmetric
# Gaussian on bounded parameters, potentially producing impossible negative
# widths or mixing fractions outside [0, 1].
#
# Sample size does not rescue a flat, badly posed objective; its covariance can
# remain meaningless even for a large cohort.
#
# The bootstrap answers the question the covariance cannot: resample the RAW
# values with replacement, re-bin, refit, and read the spread of the refits.
# It cannot return a negative width or a fraction outside [0, 1], because every
# refit is subject to the same bounds and every fraction is computed from a real
# refit. Where the data says little, the bootstrap interval widens accordingly.
#
# This does not alter the fit itself. `popt` is untouched, no curve is
# rejected, and nothing is gated; the interval is diagnostic.
# Fitting the raw values by maximum likelihood instead of binned least squares
# is a separate estimator change and remains outside this routine.

# Below this many converged refits a percentile is describing the resampler,
# not the data.  Same threshold total_fit_ci uses for its own draws.
_MIN_BOOTSTRAP_OK = 20

BOOTSTRAP_CI_METHOD = ("nonparametric bootstrap over the raw values "
                       "(resample → re-bin on the original edges → refit); "
                       "percentile interval")


@dataclass(frozen=True)
class BootstrapCI:
    """The spread of `n_ok` refits — see bootstrap_fit_ci."""
    lo:        np.ndarray            # per parameter, in popt order
    hi:        np.ndarray
    sd:        np.ndarray
    frac_lo:   np.ndarray            # per component mixing fraction
    frac_hi:   np.ndarray
    band:      Optional[tuple]       # pointwise (lo, hi) over x_fine, or None
    n_draws:   int                   # asked for
    n_ok:      int                   # converged
    n_failed:  int                   # did not converge — REPORTED, never hidden
    pct:       float
    seed:      int
    cancelled: bool

    def manifest_fields(self, pcov) -> dict:
        """This interval's provenance, for an export manifest."""
        return ci_manifest_fields(
            pcov, self.band is not None,
            method=BOOTSTRAP_CI_METHOD,
            n_draws=self.n_draws, pct=self.pct, seed=self.seed,
            extra={"ci_n_draws_ok": self.n_ok,
                   "ci_n_draws_failed": self.n_failed,
                   "ci_cancelled": self.cancelled},
        )


def order_params_by_centre(components: list[ModelSpec],
                           params: np.ndarray) -> np.ndarray:
    """Reorder one parameter vector so its components run left-to-right.

    Every bootstrap draw must be ordered by the SAME rule as the fit it
    describes, or the percentiles mix parameters from different peaks: draw A's
    "Peak 2" is the right-hand hump and draw B's is the left-hand one, and
    averaging them describes neither.  This is the standard label-switching
    trap for mixture bootstraps, and with overlapping components it is not
    rare — it is what happens whenever a resample nudges two similar peaks past
    each other.

    Same ordering key (_peak_centre) as sort_components_by_centre, which does
    the fuller job of reordering the component list and its labels too.
    """
    slices, i = [], 0
    for comp in components:
        n = comp.n_params
        slices.append((_peak_centre(comp, params[i:i + n]), params[i:i + n]))
        i += n
    slices.sort(key=lambda t: t[0])
    return np.concatenate([s[1] for s in slices]) if slices else np.asarray(params)


def _mixing_fractions(components: list[ModelSpec], params: np.ndarray) -> np.ndarray:
    """Each component's amplitude as a fraction of the total.

    Amplitude is always parameter 0 of a component (see the model registry).
    Computed per draw rather than by dividing an interval, which is how a
    proportion acquired a negative lower bound: dividing [−0.14, 0.86] by a
    total is still an interval that starts below zero.
    """
    idx, i = [], 0
    for comp in components:
        idx.append(i)
        i += comp.n_params
    amps = np.array([params[j] for j in idx], dtype=float)
    total = amps.sum()
    if not np.isfinite(total) or abs(total) < 1e-12:
        return np.full(len(idx), np.nan)
    return amps / total


def bootstrap_fit_ci(
    components: list[ModelSpec],
    data:       np.ndarray,
    edges:      np.ndarray,
    normalize:  bool,
    popt:       np.ndarray,
    bounds:     tuple,
    x_fine:     Optional[np.ndarray] = None,
    n_draws:    int   = CI_N_DRAWS,
    pct:        float = CI_PCT,
    seed:       int   = CI_SEED,
    progress:   Optional[Callable[[int, int], bool]] = None,
) -> Optional[BootstrapCI]:
    """Percentile confidence intervals for a binned mixture fit.

    `components`, `popt` and `bounds` must already be in the order the caller
    reports (left-to-right); every draw is ordered the same way.

    Three things are deliberately held FIXED across draws, because they are
    the caller's decisions and not properties of the sample:
      * the bin EDGES — re-deriving them per draw would let the histogram
        geometry move underneath the fit, mixing binning variability into an
        interval that is supposed to describe sampling variability;
      * the BOUNDS — they define the model being fitted;
      * the starting point (`popt`) — a bootstrap describes the spread of THIS
        fit, so each refit starts from it rather than re-running the peak
        finder and possibly landing in a different basin.

    `progress(i, n_draws) -> cancelled` is an ordinary callable, never a Qt
    object: this module has to stay usable from a test and a script.  The GUI
    adapts its own progress dialog on its side.

    Returns None when there is nothing honest to report (fewer than
    _MIN_BOOTSTRAP_OK refits converged, or the caller cancelled before that
    many).  None means "no interval", NEVER a silent fall back to the
    covariance: substituting a different method behind the same label is worse
    than reporting nothing.
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 2 or len(components) == 0:
        return None
    popt = np.asarray(popt, dtype=float)
    edges = np.asarray(edges, dtype=float)
    bw = float(edges[1] - edges[0])
    centres = (edges[:-1] + edges[1:]) / 2
    fn = make_composite(components)
    rng = np.random.default_rng(seed)

    draws, n_failed, cancelled = [], 0, False
    for i in range(n_draws):
        if progress is not None and progress(i, n_draws):
            cancelled = True
            break
        sample = rng.choice(data, size=len(data), replace=True)
        counts, _ = np.histogram(sample, bins=edges)
        if normalize:
            total = counts.sum() * bw
            heights = counts / max(total, 1e-10)
        else:
            heights = counts.astype(float)
        try:
            p, _ = optimize.curve_fit(
                fn, centres, heights, p0=popt, bounds=bounds,
                maxfev=20000, method="trf",
            )
        except Exception:
            # A draw that will not converge is a fact about this fit's
            # stability, so it is counted and reported rather than dropped.
            n_failed += 1
            continue
        draws.append(order_params_by_centre(components, p))

    if len(draws) < _MIN_BOOTSTRAP_OK:
        return None

    arr = np.array(draws, dtype=float)
    half = (100.0 - pct) / 2.0
    lo = np.percentile(arr, half, axis=0)
    hi = np.percentile(arr, 100.0 - half, axis=0)

    fracs = np.array([_mixing_fractions(components, p) for p in arr])
    with np.errstate(all="ignore"):
        frac_lo = np.nanpercentile(fracs, half, axis=0)
        frac_hi = np.nanpercentile(fracs, 100.0 - half, axis=0)

    band = None
    if x_fine is not None and len(x_fine):
        with np.errstate(all="ignore"):
            ys = np.array([fn(np.asarray(x_fine, dtype=float), *p) for p in arr])
        ok = np.isfinite(ys).all(axis=1)
        if ok.sum() >= _MIN_BOOTSTRAP_OK:
            ys = ys[ok]
            band = (np.percentile(ys, half, axis=0),
                    np.percentile(ys, 100.0 - half, axis=0))

    return BootstrapCI(
        lo=lo, hi=hi, sd=arr.std(axis=0),
        frac_lo=frac_lo, frac_hi=frac_hi, band=band,
        n_draws=n_draws, n_ok=len(draws), n_failed=n_failed,
        pct=pct, seed=seed, cancelled=cancelled,
    )


# ── Goodness-of-fit statistics ────────────────────────────────────────────────
#
# Information criteria use a true per-sample log-likelihood.
#
# Parameters are estimated by least squares on histogram bins, while the
# information criterion evaluates the per-sample likelihood at that estimate.
# It must not be replaced with a Gaussian-SSE surrogate over bin heights,
# because that is not comparable with the GMM's per-sample likelihood.
# Estimating by
# maximum likelihood on the raw values (which would also make the ESTIMATOR
# match the GMM's) is a separate estimator change and is deliberately not here.
#
# THE PARAMETER COUNT DELIBERATELY DROPS ONE, and this is what makes the two
# windows comparable rather than merely similar.  Every component here carries
# an `amp` on top of its shape parameters, but the total amplitude is fixed by
# the data — a density integrates to 1 — so it carries no distributional
# information.  Counting it would overcount by exactly one against sklearn's
# convention (means + covariances + k-1 weights).  Checked both ways:
#   1 Gaussian : ours 3 params - 1 = 2;  sklearn 1 + 1 + 0 = 2.
#   2 Gaussians: ours 6 params - 1 = 5;  sklearn 2 + 2 + 1 = 5.

# Stamped into every gof dict so a stored fit says which basis produced its
# AICc, and a comparison table can refuse to rank the two against each other.
# The absence of the key identifies the legacy bin-count basis because the
# number itself does not identify the method that produced it.
IC_BASIS        = "per_sample_loglik"
IC_BASIS_LEGACY = "binned_sse"


def composite_density(components: list[ModelSpec], popt) -> "Callable | None":
    """The fitted composite, renormalised to integrate to 1.

    The fit's own amplitudes are what it integrates to (every registered pdf is
    a normalised density scaled by `amp`, always its first parameter), so the
    normaliser is just their sum — no numerical integration, which would be
    fragile on the heavy-tailed components.

    None when that sum is not a usable positive number: a density that cannot
    be normalised has no likelihood, and returning one anyway would put a
    fabricated AICc in a comparison table.

    THE RENORMALISATION IS NOT COSMETIC.  Least squares on bin heights does not
    constrain the amplitudes to sum to 1, and on a badly mis-specified model it
    lands nowhere near: a single Gaussian LS-fitted to an obviously bimodal
    histogram takes the taller hump alone and comes back with sum(amp) = 0.60,
    abandoning the other 40 % of the data.  An unnormalised curve has no
    likelihood at all, so the score is of the SHAPE the fit found, with its
    total mass restored to 1 — the charitable reading, and the only defined
    one.
    """
    fn    = make_composite(components)
    total = 0.0
    i     = 0
    for comp in components:
        total += float(popt[i])
        i     += comp.n_params
    if not np.isfinite(total) or total <= 0.0:
        return None
    return lambda x: np.asarray(fn(np.asarray(x, dtype=float), *popt)) / total


def per_sample_loglik(values: np.ndarray, density) -> tuple[float, int]:
    """(sum of log f(x_i), how many points the model gave no density at).

    Points at zero density are floored rather than dropped.  Dropping them
    would quietly shrink n for whichever model fails worst, which is the
    opposite of what a model-selection statistic should do; the count is
    returned so the window can say so instead of the number pretending.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), 0
    d = np.asarray(density(v), dtype=float)
    bad = ~np.isfinite(d) | (d <= 0.0)
    d = np.where(bad, 1e-300, d)
    return float(np.sum(np.log(d))), int(np.count_nonzero(bad))


def fit_stats(y_obs: np.ndarray, y_fit: np.ndarray, n_params: int,
              values: np.ndarray, components: list[ModelSpec], popt) -> dict:
    """Goodness of fit for one composite fit.

    R2/reduced chi-2/RSS describe the fit TO THE HISTOGRAM — that is what least
    squares actually minimised, and they stay binned quantities.  The
    information criteria describe the fit to the VALUES; see the note above.
    """
    n_bins = len(y_obs)
    ss_res = float(np.sum((y_obs - y_fit) ** 2))
    ss_tot = float(np.sum((y_obs - y_obs.mean()) ** 2))
    r2     = 1.0 - ss_res / max(ss_tot, 1e-300)
    dof    = max(n_bins - n_params, 1)

    n_values = int(np.count_nonzero(np.isfinite(np.asarray(values, dtype=float))))
    density  = composite_density(components, popt)
    k        = max(n_params - 1, 1)

    if density is None or n_values == 0:
        ll = aic = aicc = bic = float("nan")
        n_zero = 0
    else:
        ll, n_zero = per_sample_loglik(values, density)
        aic  = 2 * k - 2 * ll
        aicc = aic + 2 * k * (k + 1) / max(n_values - k - 1, 1)
        bic  = k * np.log(max(n_values, 1)) - 2 * ll
    return {
        "R²":            r2,
        "Reduced χ²":    ss_res / dof,
        "AIC":           aic,
        "AICc":          aicc,
        "BIC":           bic,
        "DOF":           dof,
        "RSS":           ss_res,
        "log-likelihood": ll,
        "n (for IC)":    n_values,
        "k (for IC)":    k,
        "ic_basis":      IC_BASIS,
        "n_zero_density": n_zero,
    }
