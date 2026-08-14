from __future__ import annotations

import numpy as np

from smfs_catalog.gmm_fit_core import (
    aicc_from_aic,
    component_display_ids,
    component_order,
    json_safe_statistics,
    n_params_gmm,
)


class _MixtureStub:
    weights_ = np.array([0.2, 0.7, 0.1])


def test_aicc_is_unavailable_when_sample_cannot_support_correction():
    p = n_params_gmm(2, "Full")
    assert np.isnan(aicc_from_aic(100.0, p, p + 1))
    assert np.isnan(aicc_from_aic(100.0, p, p))
    assert np.isfinite(aicc_from_aic(100.0, p, p + 2))


def test_display_component_ids_follow_weight_order():
    gm = _MixtureStub()
    assert component_order(gm) == [1, 0, 2]
    assert component_display_ids(gm).tolist() == [2, 1, 3]


def test_non_finite_statistics_become_json_null_values():
    assert json_safe_statistics({"AIC": 12.5, "AICc": np.nan}) == {
        "AIC": 12.5, "AICc": None,
    }
