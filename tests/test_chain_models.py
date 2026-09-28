# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""The chain models beyond Marko-Siggia recover what they were given."""

from __future__ import annotations

import numpy as np
import pytest

from smfs_catalog import models as _m
from smfs_catalog.roi_events import _fit_chain_model

# (model key, true params). Forces run to ~500 pN so a stretch modulus is
# constrained, as it is on real high-force ramps.
TRUE = {
    "bwlc": (0.4, 100.0),
    "ewlc": (0.4, 100.0, 2000.0),
    "fjc":  (0.8, 100.0),
    "efjc": (0.8, 100.0, 5000.0),
}


def _synthetic(key: str, rng) -> tuple[np.ndarray, np.ndarray]:
    m = _m.CHAIN_MODELS[key]
    p = TRUE[key]
    if key == "bwlc":
        x = np.linspace(5.0, 0.97 * p[1], 300)
    else:
        extension = {"ewlc": _m.ewlc_extension, "fjc": _m.fjc_extension,
                     "efjc": _m.efjc_extension}[key]
        x = extension(np.linspace(20.0, 500.0, 300), *p)
    F = m.force(x, *p)
    return x, F + rng.normal(0.0, 2.0, F.size)


@pytest.mark.parametrize("key", list(TRUE))
def test_each_model_recovers_its_own_parameters(key):
    x, F = _synthetic(key, np.random.default_rng(0))
    fit = _fit_chain_model(_m.CHAIN_MODELS[key], x, F, 0.6, 1.3 * float(np.max(x)))
    assert fit is not None
    for name, want in zip(_m.CHAIN_MODELS[key].params, TRUE[key]):
        assert fit[name] == pytest.approx(want, rel=0.05), name
        assert fit[name + "_err"] > 0


@pytest.mark.parametrize("key", ["ewlc", "fjc", "efjc"])
def test_force_from_extension_inverts_the_model(key):
    extension = {"ewlc": _m.ewlc_extension, "fjc": _m.fjc_extension,
                 "efjc": _m.efjc_extension}[key]
    F = np.array([1.0, 10.0, 100.0, 1000.0])
    x = extension(F, *TRUE[key])
    assert _m.CHAIN_MODELS[key].force(x, *TRUE[key]) == pytest.approx(F, rel=1e-3)
