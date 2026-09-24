# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""Force-spectrum fits recover the landscape they were generated from, and
the information criteria pick the shape the data were drawn with."""

import numpy as np

from smfs_catalog import db
from smfs_catalog import force_spectrum_core as FS

KT = FS.kT_of(298.0)


def _spectrum(x_b, lk, dG, nu, n=600, noise=3.0, seed=0):
    rng = np.random.default_rng(seed)
    ln_r = rng.uniform(np.log(1e2), np.log(1e6), n)
    f = FS.mean_force(ln_r, x_b, lk, dG, nu, KT) + rng.normal(0.0, noise, n)
    return np.exp(ln_r), f


def _by_label(fits):
    return {fit.label: fit for fit in fits}


def test_bell_evans_recovers_its_own_line():
    rate, force = _spectrum(0.4, -2.0, 1.0, 1.0)
    fits, dropped = FS.fit_all(rate, force, KT)
    bell = _by_label(fits)["Bell-Evans"]
    assert dropped == 0
    x_b, x_err = bell.params["x_b"]
    lk, lk_err = bell.params["log10 k_off"]
    assert abs(x_b - 0.4) < 3 * x_err
    assert abs(lk - (-2.0)) < 3 * lk_err


def test_cusp_data_are_fitted_and_preferred_as_cusp():
    # Chosen so every rate sits inside the model's domain (0 < F < F_c).
    rate, force = _spectrum(0.5, -4.0, 25.0, 2.0 / 3.0)
    fits = _by_label(FS.fit_all(rate, force, KT)[0])
    cusp = fits["DHS cusp (ν=2/3)"]
    assert not cusp.at_bound
    for name, true in zip(FS.PARAM_NAMES, (0.5, -4.0, 25.0)):
        v, e = cusp.params[name]
        assert abs(v - true) < 3 * e, name
    best = min(fits.values(), key=lambda fit: fit.stats["AICc"])
    assert best.nu != 1.0


def test_only_nonpositive_or_missing_rates_are_dropped():
    rate, force = _spectrum(0.4, -2.0, 1.0, 1.0, n=50)
    rate[:3] = [0.0, -5.0, np.nan]
    _fits, dropped = FS.fit_all(rate, force, KT)
    assert dropped == 3


def test_temperature_prefers_thermal_then_head():
    meta = [{"ThermalTemperature": 300.0},
            {"StartHeadTemp": 25.0},
            {}]
    t, n = FS.temperature_K(meta)
    assert n == 2
    assert np.isclose(t, np.median([300.0, 298.15]))


def test_metadata_bulk_reads_by_path(tmp_path):
    path = str(tmp_path / "meta.sqlite")
    db.initialise(path)
    for name, temp in (("a", 301.0), ("b", 302.0)):
        db.upsert_file(
            {"path": f"/data/{name}.ibw", "filename": f"{name}.ibw",
             "first_seen": "2026-01-01", "last_seen": "2026-01-01"},
            db_path=path,
        )
        db.write_file_metadata(db.get_file_id(f"/data/{name}.ibw", path), {"ThermalTemperature": temp, "ImagingMode": "Contact"},
                               db_path=path)
    out = db.get_file_metadata_bulk(["/data/a.ibw", "/data/b.ibw"],
                                    ["ThermalTemperature"], path)
    assert {p: m["ThermalTemperature"] for p, m in out.items()} == {
        db.normalize_path("/data/a.ibw"): 301.0,
        db.normalize_path("/data/b.ibw"): 302.0,
    }
