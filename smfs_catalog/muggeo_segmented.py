"""
muggeo_segmented.py — Muggeo's segmented regression as one algorithm for
locating a rupture on its loading ramp.

Three straight lines joined at ψ₁ < ψ₂ are fitted to the saved force around
the fall, with x the sample index: the ramp, the fall, and what follows it.
The rupture point is the last sample at or before ψ₁.

The fit follows the procedure of Muggeo's R package segmented 2.2-2
(segmented.lm, seg.lm.fit, seg.lm.fit.boot) step by step, at his default
controls apart from the tolerance and the restarts, which the profile sets.
The numerical routines under it are numpy's and scipy's: least squares, the
bounded Brent line search, quantiles and the random draws.

Muggeo, V. M. R. (2003). Estimating regression models with unknown
break-points. Statistics in Medicine 22:3055–3071. doi:10.1002/sim.1545
Muggeo, V. M. R. (2008). segmented: an R package to fit regression models
with broken-line relationships. R News 8/1:20–25.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar

# The tolerances a profile can choose (muggeo_tolerance_idx indexes these),
# and restarts off or on (muggeo_restarts_idx indexes these).
TOLERANCES = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
RESTARTS = (False, True)

# segmented's controls, held at Muggeo's defaults (seg.control's names).
N_BOOT     = 10    # n.boot: at most this many restarts
BREAK_BOOT = 5     # break.boot: restarts stop once the best RSS is unchanged this often
IT_MAX     = 30    # it.max: passes per fit
H          = 1.25  # h: step factor on each pass's proposed move
FC         = 0.95  # fc: factor that moves a join away from a segment too thin
MIN_NJ     = 2     # min.nj: fewest samples a segment may hold
ALPHA_MIN  = 0.05  # alpha = max(0.05, 1/T): each join stays between these quantiles of x
# seg.lm.fit.boot's own constants: after this many equal results the start
# jumps to a quantile of x (n.boot.rev), first this one (alpha), then 1 − it.
EQUAL_RESULTS = 3
JUMP_QUANTILE = 0.1

# What an answer says about itself.  A warning or a reason follows ": ".
FITTED = "fitted"
WARNED = "fitted with segmented's warning"
NO_FIT = "no fit"


@dataclass(frozen=True)
class MuggeoFit:
    idx:      Optional[int]   # floor(ψ₁), absolute; None on no fit
    status:   str
    psi:      Optional[tuple[float, float]]
    rss:      Optional[float]
    passes:   int             # passes of the fit kept
    restarts: int             # restarts that produced a fit
    first:    int             # the window, inclusive
    last:     int


class _Stop(Exception):
    """segmented stops with an error: no fit."""


@dataclass
class _Fit:
    psi:      np.ndarray
    ss:       float
    passes:   int
    beta:     Optional[np.ndarray]   # slope changes at the joins; None for a best-of-history answer
    warnings: list[str] = field(default_factory=list)


def fit_window(segment_start: int, left_crossing: int, right_crossing: int,
               window_end: int, multiplier: int) -> tuple[int, int]:
    """The samples the fit sees: `multiplier` d¹-peak widths each side of the
    fall, never before the segment start nor past `window_end` (the next
    rupture's left crossing, or the curve's last sample)."""
    w = right_crossing - left_crossing + 1
    return (max(segment_start, left_crossing - multiplier * w),
            min(window_end, right_crossing + multiplier * w))


def muggeo_segmented(force: np.ndarray, segment_start: int, left_crossing: int,
                     right_crossing: int, window_end: int, multiplier: int,
                     tolerance: float, restarts: bool) -> MuggeoFit:
    """Fit the window around one rupture, joins started at the two d¹
    crossings, and return the rupture point floor(ψ₁)."""
    first, last = fit_window(segment_start, left_crossing, right_crossing,
                             window_end, multiplier)
    x = np.arange(first, last + 1, dtype=float)
    y = np.asarray(force[first:last + 1], dtype=float)
    try:
        fit, n_restarts = _segmented(y, x, (float(left_crossing), float(right_crossing)),
                                     tolerance, restarts)
    except _Stop as stop:
        return MuggeoFit(None, f"{NO_FIT}: {stop}", None, None, 0, 0, first, last)
    status = f"{WARNED}: {'; '.join(fit.warnings)}" if fit.warnings else FITTED
    psi = (float(fit.psi[0]), float(fit.psi[1]))
    return MuggeoFit(int(math.floor(psi[0])), status, psi, fit.ss, fit.passes,
                     n_restarts, first, last)


# ── segmented.lm ──────────────────────────────────────────────────────────────

def _segmented(y, x, psi0, tol, restarts) -> tuple[_Fit, int]:
    psi = np.sort(np.asarray(psi0, dtype=float))
    for p in psi:
        if np.all(x <= p) or np.all(x >= p):
            raise _Stop("starting psi out of the admissible range")
    a = max(ALPHA_MIN, 1.0 / len(y))
    alpha = (a, 1.0 - a)
    if restarts:
        fit, n_restarts = _fit_boot(y, x, psi, tol, alpha)
    else:
        fit, n_restarts = _fit(y, x, psi, tol, IT_MAX, alpha), 0
    if fit is None:
        raise _Stop("estimation failed")
    # The final linear model with the joins fixed: a coefficient it cannot
    # estimate stops segmented.
    if fit.beta is None or not np.all(np.isfinite(fit.beta)):
        raise _Stop("at least one coef is NA")
    U = _u(x, fit.psi)
    vxb = -(x[:, None] > fit.psi[None, :]).astype(float) * fit.beta[None, :]
    model = np.column_stack([np.ones_like(x), x, U, vxb])
    if np.linalg.matrix_rank(model) < model.shape[1]:
        raise _Stop("at least one coef is NA: breakpoint(s) at the boundary?")
    return fit, n_restarts


# ── seg.lm.fit ────────────────────────────────────────────────────────────────

def _u(x, psi):
    """Uₖ = (x − ψₖ)₊, one column per join."""
    d = x[:, None] - np.asarray(psi)[None, :]
    return d * (d > 0)


def _ls(X, y):
    """Least squares of y on X: (coefficients, residual sum of squares), or
    None when X does not have full column rank."""
    try:
        coef, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < X.shape[1]:
        return None
    r = y - X @ coef
    return coef, float(r @ r)


def _limits(x, alpha):
    return np.quantile(x, alpha)


def _far_enough(x, psi, fc):
    """seg.lm.fit's far.psi: which joins leave at least 2 samples in the
    segment before them, and the factor that moves the others."""
    groups = (x[:, None] > psi[None, :]).sum(axis=1) + 1
    nj = np.bincount(groups)[1:]
    if len(nj) != len(psi) + 1:
        groups = (x[:, None] >= psi[None, :]).sum(axis=1) + 1
        nj = np.bincount(groups)[1:]
    ok = (nj >= 2)[:-1]
    factor = np.where(np.diff(nj) > 0, 1.0 / fc, fc)
    return np.resize(ok, len(psi)), np.resize(factor, len(psi))


def _fit(y, x, psi, tol, it_max, alpha, all_sol=False):
    """One fit from the starts `psi`.  With `all_sol`, a fit that cannot go
    on returns the best of the passes made so far instead of stopping."""
    lo, hi = _limits(x, alpha)
    psi = np.clip(np.asarray(psi, dtype=float), lo, hi)
    if not np.all((psi >= lo) & (psi <= hi)):
        raise _Stop("starting psi out of the range")
    far_ok, _ = _far_enough(x, psi, FC)
    if not np.all(far_ok):
        raise _Stop("psi starting values too close each other or at the boundaries")
    X0 = np.column_stack([np.ones_like(x), x])
    k = len(psi)

    if it_max == 0:
        # The joins fixed where they are: one linear fit.
        X = np.column_stack([X0, _u(x, psi)])
        coef, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ coef
        beta = coef[2:2 + k] if rank == X.shape[1] else np.full(k, np.nan)
        return _Fit(psi, float(r @ r), 0, beta)

    U = _u(x, psi)
    V = -(x[:, None] > psi[None, :]).astype(float)
    start = _ls(np.column_stack([X0, U]), y)
    if start is None:
        raise _Stop("the starting fit failed")
    L0 = start[1]
    history = [(L0, psi.copy())]
    line_tols = np.linspace(0.001, np.finfo(float).eps ** 0.25, it_max)
    passes, epsilon = 0, 10.0
    hit_max, moved_last = False, False
    while abs(epsilon) > tol:
        passes += 1
        step = _ls(np.column_stack([X0, U, V]), y)
        if step is None:
            if all_sol:
                return _best_of(history)
            raise _Stop("breakpoint estimate too close or at the boundary "
                        "causing NA estimates")
        beta, gamma = step[0][2:2 + k], step[0][2 + k:2 + 2 * k]
        psi_old = psi
        psi = np.sort(np.clip(psi_old + H * gamma / beta, lo, hi))
        L_prev = L0

        def rss_at(h, psi=psi, psi_old=psi_old):
            fit = _ls(np.column_stack([X0, _u(x, psi * h + psi_old * (1 - h))]), y)
            return L_prev + 10 if fit is None else fit[1]

        found = minimize_scalar(rss_at, bounds=(0.0, 1.0), method="bounded",
                                options={"xatol": line_tols[passes - 1]})
        h, L1 = float(found.x), float(found.fun)
        psi = np.clip(psi * h + psi_old * (1 - h), lo, hi)
        U = _u(x, psi)
        V = -(x[:, None] > psi[None, :]).astype(float)
        epsilon = (L0 - L1) / (abs(L0) + 0.1)
        L0 = L1
        history.append((L0, psi.copy()))
        far_ok, factor = _far_enough(x, psi, FC)
        in_ok = (psi >= lo) & (psi <= hi)
        moved_last = False
        if not np.all(in_ok & far_ok):
            # Only the joins move; U and V stay as this pass left them.
            psi = psi * np.where(far_ok, 1.0, factor)
            moved_last = passes == it_max
        if passes >= it_max:
            hit_max = True
            break

    warnings = []
    if moved_last:
        warnings.append("Some psi (" + ",".join(str(i + 1) for i in np.flatnonzero(~far_ok))
                        + ") changed after the last iter.")
    if hit_max:
        warnings.append(f"max number of iterations ({passes}) attained")
    psi = np.sort(psi)
    final = _ls(np.column_stack([X0, _u(x, psi)]), y)
    if final is None:
        raise _Stop("the final fit failed")
    return _Fit(psi, final[1], passes, final[0][2:2 + k], warnings)


def _best_of(history) -> _Fit:
    """seg.lm.fit.boot's extract.psi: the lowest RSS among the passes."""
    ss, psi = min(history, key=lambda item: item[0])
    return _Fit(psi, ss, 0, None)


# ── seg.lm.fit.boot ───────────────────────────────────────────────────────────

def _seed(y) -> int:
    """segmented's seed from the data: the first five non-zero digits of
    mean(y), written to 15 significant digits."""
    digits = [c for c in f"{float(np.mean(y)):.15g}".replace(".", "") if c != "0"][:5]
    try:
        seed = int(float("".join(digits))) if digits else 1
    except ValueError:
        raise _Stop("no seed from the data")
    return seed % 2 ** 32


def _fit_boot(y, x, psi, tol, alpha) -> tuple[Optional[_Fit], int]:
    rng = np.random.default_rng(_seed(y))
    lo, hi = _limits(x, alpha)
    n = len(y)
    try:
        best = _fit(y, x, psi, tol, IT_MAX, alpha)
    except _Stop:
        # segmented's retry from its K = 10 quantile starts cannot run with
        # two joins, so it goes straight to keeping the best pass.
        best = _fit(y, x, psi, tol, IT_MAX, alpha, all_sol=True)
    start = best.psi
    kept_ss: list[float] = []
    n_restarts = 0
    jump = JUMP_QUANTILE
    for _ in range(N_BOOT):
        recent = np.diff(kept_ss)[::-1]
        if (len(recent) >= EQUAL_RESULTS - 1
                and np.all(np.round(recent[:EQUAL_RESULTS - 1], 6) == 0)):
            q = np.array([np.mean(p >= x) for p in start])
            q = np.where(np.abs(q - 0.5) < 0.1, jump, q)
            jump = 1 - jump
            start = np.array([np.quantile(x, 1 - qi) for qi in q])
        idx = rng.integers(0, n, size=n)
        try:
            on_resample = _fit(y[idx], x[idx], start, tol, IT_MAX, alpha).psi
        except _Stop:
            on_resample = np.sort(rng.uniform(lo, hi, size=len(start)))
        try:
            fit = _fit(y, x, on_resample, tol, IT_MAX, alpha, all_sol=True)
        except _Stop:
            start = rng.uniform(lo, hi, size=len(start))
            try:
                fit = _fit(y, x, start, tol, 0, alpha)
            except _Stop:
                fit = None
        if fit is not None:
            n_restarts += 1
            if fit.ss <= best.ss:
                best = fit
            start = best.psi
            kept_ss.append(best.ss)
        start = np.sort(start)
        if (len(kept_ss) > BREAK_BOOT
                and np.all(np.round(np.diff(kept_ss)[::-1][:BREAK_BOOT - 1], 6) == 0)):
            break

    if best.beta is None:
        # The best answer came from a fit that could not finish: segmented
        # moves each join to leave enough samples on both sides and refits
        # with the joins fixed.
        best = _outdistanced(y, x, np.sort(start), lo, hi, alpha)
        if best is None:
            return None, n_restarts
        best.warnings.append("Breakpoint estimate(s) outdistanced to allow "
                             "finite estimates and st.errs")
    return best, n_restarts


def _outdistanced(y, x, psi, lo, hi, alpha) -> Optional[_Fit]:
    min_n = MIN_NJ - 1
    xs = np.unique(x)
    psi = psi.copy()
    for j in range(len(psi)):
        breaks = np.concatenate([[lo - 1e8], psi, [hi + 1e8]])
        groups = np.searchsorted(breaks, xs, side="left")   # (b₀, b₁] → 1, …
        nj = np.bincount(groups[(groups >= 1) & (groups < len(breaks))])[1:]
        if len(nj) < j + 2:
            raise _Stop("missing value where TRUE/FALSE needed")
        upper = (xs[groups == j + 1].max() if nj[j] > 0 else 0.0) - 1e6 * (nj[j] <= min_n)
        lower = ((xs[groups == j + 2].min() if nj[j + 1] > 0 else 0.0)
                 + 1e6 * ((nj[j + 1] if nj[j + 1] > 0 else nj[j]) <= min_n))
        if abs(upper - psi[j]) < abs(lower - psi[j]):
            psi[j] = upper - 1e-4
        else:
            psi[j] = lower + 1e-4
    try:
        return _fit(y, x, psi, 0.0, 0, alpha)
    except _Stop:
        return None
