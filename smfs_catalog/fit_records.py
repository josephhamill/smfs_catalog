# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""The row objects the fit savers in db.py take."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistFitRecord:
    """One committed distribution fit, as stored in distribution_fits."""

    variable: str
    units: str
    n_values: int
    n_peaks: int
    model_label: str
    params_json: str
    gof_json: str
    fit_config_json: str
    experimentalist: str = ""


@dataclass(frozen=True)
class GmmFitRecord:
    """One committed 2D GMM fit, as stored in gmm_fits."""

    x_variable: str
    y_variable: str
    n_values: int
    k_components: int
    cov_type: str
    means_json: str
    covs_json: str
    weights_json: str
    gof_json: str
    fit_config_json: str = "{}"
    experimentalist: str = ""
