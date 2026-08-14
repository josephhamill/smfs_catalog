# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/gmm_fit_core.py
#
# Pure math for 2D Gaussian Mixture Model fitting.
# Wraps sklearn.mixture.GaussianMixture with AICc computation,
# ellipse geometry, and parameter-count formulae.

from __future__ import annotations

import numpy as np


# ── Palette ───────────────────────────────────────────────────────────────────

# Component identities use the application's labelled-series palette.
from . import style

COMPONENT_COLORS = list(style.SERIES_LABELED)


# ── Covariance type registry ──────────────────────────────────────────────────

# Display label → sklearn covariance_type string
COV_TYPES: dict[str, str] = {
    "Full":      "full",
    "Tied":      "tied",
    "Diagonal":  "diag",
    "Spherical": "spherical",
}
COV_TYPE_LABELS = list(COV_TYPES.keys())


# ── Parameter counting ────────────────────────────────────────────────────────

def n_params_gmm(k: int, cov_label: str, d: int = 2) -> int:
    """Free parameters for a K-component GMM in d dimensions."""
    cov = COV_TYPES.get(cov_label, cov_label)
    weights = k - 1
    means   = k * d
    if cov == "full":
        covar = k * d * (d + 1) // 2      # K × 3 for d=2
    elif cov == "tied":
        covar = d * (d + 1) // 2          # 3 for d=2
    elif cov == "diag":
        covar = k * d                      # K × 2
    else:                                  # spherical
        covar = k
    return weights + means + covar


def aicc_from_aic(aic: float, n_params: int, n: int) -> float:
    """Return AICc, or NaN when the finite-sample correction is undefined."""
    denom = n - n_params - 1
    if denom <= 0 or not np.isfinite(aic):
        return float("nan")
    return float(aic + 2.0 * n_params * (n_params + 1) / denom)


def component_order(gm) -> list[int]:
    """Internal component indices in the stable order shown to the user."""
    return [int(k) for k in np.argsort(gm.weights_)[::-1]]


def component_display_ids(gm) -> np.ndarray:
    """Map sklearn's arbitrary component indices to displayed C1..CK IDs."""
    order = component_order(gm)
    display_ids = np.empty(len(order), dtype=int)
    for rank, internal_index in enumerate(order, start=1):
        display_ids[internal_index] = rank
    return display_ids


def json_safe_statistics(values: dict) -> dict:
    """Convert non-finite statistics to JSON null instead of NaN/Infinity."""
    return {
        key: (float(value) if np.isfinite(value) else None)
        for key, value in values.items()
    }


# ── Covariance extraction (handles all sklearn shapes) ────────────────────────

def get_component_cov(gm, k: int) -> np.ndarray:
    """Return the (2, 2) covariance matrix for component k."""
    ct = gm.covariance_type
    if ct == "full":
        return np.array(gm.covariances_[k])
    elif ct == "tied":
        return np.array(gm.covariances_)
    elif ct == "diag":
        return np.diag(gm.covariances_[k])
    else:  # spherical
        return np.eye(2) * float(gm.covariances_[k])


# ── Ellipse geometry ──────────────────────────────────────────────────────────

def ellipse_curve(
    mean: np.ndarray,
    cov:  np.ndarray,
    scale: float = 1.0,
    n_pts: int   = 120,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (x, y) arrays tracing a covariance ellipse at Mahalanobis radius
    ``scale``. In two dimensions the radii 1 and 2 contain approximately 39%
    and 86% of a Gaussian component, not the one-dimensional 68% and 95%.
    """
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)   # guard tiny negatives from float rounding
    theta = np.linspace(0.0, 2.0 * np.pi, n_pts)
    unit  = np.stack([np.cos(theta), np.sin(theta)])
    pts   = mean[:, None] + vecs @ (np.sqrt(vals)[:, None] * scale * unit)
    return pts[0], pts[1]


# ── Per-component summary statistics ─────────────────────────────────────────

def component_stats(gm, k: int) -> dict:
    """
    weight, mu_x, mu_y, sigma_x, sigma_y, rho for component k.
    rho is the Pearson correlation; meaningful only for full/tied.
    """
    cov = get_component_cov(gm, k)
    mu  = gm.means_[k]
    w   = gm.weights_[k]
    s_x = float(np.sqrt(max(cov[0, 0], 0.0)))
    s_y = float(np.sqrt(max(cov[1, 1], 0.0)))
    denom = s_x * s_y
    rho   = float(cov[0, 1] / denom) if denom > 1e-12 else 0.0
    return {
        "weight":  float(w),
        "mu_x":    float(mu[0]),
        "mu_y":    float(mu[1]),
        "sigma_x": s_x,
        "sigma_y": s_y,
        "rho":     rho,
    }


