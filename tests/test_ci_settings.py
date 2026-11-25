# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
CiSettings: the confidence-interval sampling settings as one object (PR d, d5).

Three kinds of test:

1. Interface: `CiSettings(method, n_draws, pct, seed)` is a frozen dataclass
   with today's defaults, and the three CI functions take it as `ci` instead
   of separate method / n_draws / pct / seed arguments.
2. Behaviour pins: with a fixed seed, `ci_manifest_fields`, `total_fit_ci` and
   `bootstrap_fit_ci` give exactly today's numbers.  The expected values are
   literals recorded from the pre-d5 code, not recomputed.  Calls go through
   the three adapter functions below, the only place that knows how to spell
   "these settings" in the current signatures; d5 changes only those.
3. Size guard: the functions stay inside the CLAUDE.md limits
   (`bootstrap_fit_ci` is exempt from the argument limit; plan D5 lists it).
"""
import dataclasses
import inspect

import numpy as np
import pytest

from size_limits import violations
from source_inspect import function_node
from smfs_catalog import dist_fit_core as core
from smfs_catalog.dist_fit_core import (
    BOOTSTRAP_CI_METHOD,
    CI_N_DRAWS,
    CI_PCT,
    CI_SEED,
    COV_CI_METHOD,
    MODELS,
    bootstrap_fit_ci,
    ci_manifest_fields,
    composite_bounds,
    total_fit_ci,
)


# ---- adapters: the only code that spells the settings out ---------------------

def _manifest(pcov, band_drawn, **settings):
    """ci_manifest_fields with settings given by name (method/n_draws/pct/seed)."""
    return ci_manifest_fields(pcov, band_drawn, core.CiSettings(**settings))


def _total(fn, x, popt, pcov, **settings):
    """total_fit_ci with settings given by name (n_draws/pct/seed)."""
    return total_fit_ci(fn, x, popt, pcov, core.CiSettings(**settings))


def _bootstrap(fixture, with_band=True, **settings):
    """bootstrap_fit_ci on the `_boot_fixture()` tuple (normalised data), with
    settings given by name (n_draws/pct/seed).  with_band=False passes no
    x_fine, so no band is drawn."""
    comps, data, edges, popt, bounds, x_fine = fixture
    return bootstrap_fit_ci(comps, data, edges, True, popt, bounds,
                            x_fine if with_band else None,
                            core.CiSettings(**settings))


# ---- 1. interface -------------------------------------------------------------

def _settings_class():
    assert hasattr(core, "CiSettings"), "dist_fit_core.CiSettings is missing"
    return core.CiSettings


def test_ci_settings_has_exactly_the_four_fields_in_order():
    cls = _settings_class()
    assert dataclasses.is_dataclass(cls)
    assert [f.name for f in dataclasses.fields(cls)] == [
        "method", "n_draws", "pct", "seed"]


def test_ci_settings_defaults_are_todays_defaults():
    ci = _settings_class()()
    assert ci.method == COV_CI_METHOD
    assert (ci.n_draws, ci.pct, ci.seed) == (CI_N_DRAWS, CI_PCT, CI_SEED)
    assert (CI_N_DRAWS, CI_PCT, CI_SEED) == (400, 95.0, 0)


def test_ci_settings_takes_each_field_positionally_and_by_name():
    cls = _settings_class()
    a = cls("m", 10, 80.0, 5)
    b = cls(method="m", n_draws=10, pct=80.0, seed=5)
    assert a == b
    assert (a.method, a.n_draws, a.pct, a.seed) == ("m", 10, 80.0, 5)


def test_ci_settings_is_frozen():
    ci = _settings_class()()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ci.seed = 9


def _params(fn):
    return list(inspect.signature(fn).parameters)


def test_total_fit_ci_signature():
    assert _params(total_fit_ci) == ["fn", "x", "popt", "pcov", "ci"]


def test_ci_manifest_fields_signature():
    params = _params(ci_manifest_fields)
    assert params[:3] == ["pcov", "band_drawn", "ci"]
    assert not {"method", "n_draws", "pct", "seed"} & set(params)


def test_bootstrap_fit_ci_signature():
    assert _params(bootstrap_fit_ci) == [
        "components", "data", "edges", "normalize", "popt", "bounds",
        "x_fine", "ci", "progress"]


# ---- 2. behaviour pins --------------------------------------------------------

_COV = np.array([[0.04, 0.03], [0.03, 0.09]])
_X = np.linspace(0, 2, 5)


def _decay(x, a, b):
    return a * np.exp(-b * x)


def test_manifest_fields_default_settings():
    assert _manifest(_COV, True) == {
        "ci_band_drawn": True,
        "ci_method": COV_CI_METHOD,
        "ci_pct": 95.0,
        "ci_n_draws": 400,
        "ci_seed": 0,
        "param_covariance": [[0.04, 0.03], [0.03, 0.09]],
    }


def test_manifest_fields_custom_settings_are_recorded():
    out = _manifest(_COV, False, method="none", n_draws=12, pct=80.0, seed=4)
    assert out["ci_band_drawn"] is False
    assert (out["ci_method"], out["ci_n_draws"], out["ci_pct"], out["ci_seed"]) \
        == ("none", 12, 80.0, 4)


def test_manifest_fields_unusable_or_missing_covariance_is_none():
    bad = np.array([[np.inf, 0.0], [0.0, 1.0]])
    assert _manifest(bad, False)["param_covariance"] is None
    assert _manifest(None, False)["param_covariance"] is None


_TOTAL_DEFAULT = (
    [0.581989765912643, 0.5111004434338793, 0.3621673669305171,
     0.22211994848273217, 0.1320137598069166],
    [1.359239973536334, 1.0762220483339842, 1.02862449457868,
     1.0621684358722259, 1.0817663901271335],
)
_TOTAL_CUSTOM = (
    [0.6936159088854229, 0.5813229967698782, 0.4077741811687705,
     0.27598521509771956, 0.17364214724753624],
    [1.2550914812769836, 0.9431135539011202, 0.9103280176568606,
     0.7867567369057691, 0.8474527830112312],
)


def test_total_fit_ci_default_settings_match_recorded_band():
    lo, hi = _total(_decay, _X, [1.0, 0.5], _COV)
    assert np.array_equal(lo, _TOTAL_DEFAULT[0])
    assert np.array_equal(hi, _TOTAL_DEFAULT[1])


def test_total_fit_ci_custom_settings_match_recorded_band():
    lo, hi = _total(_decay, _X, [1.0, 0.5], _COV, n_draws=100, pct=90.0, seed=7)
    assert np.array_equal(lo, _TOTAL_CUSTOM[0])
    assert np.array_equal(hi, _TOTAL_CUSTOM[1])


def test_total_fit_ci_seed_selects_the_draws():
    a = _total(_decay, _X, [1.0, 0.5], _COV, seed=1)
    b = _total(_decay, _X, [1.0, 0.5], _COV, seed=1)
    c = _total(_decay, _X, [1.0, 0.5], _COV, seed=2)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[0], c[0])


def test_total_fit_ci_unusable_covariance_is_none():
    bad = np.array([[np.inf, 0.0], [0.0, 1.0]])
    assert _total(_decay, _X, [1.0, 0.0], bad) is None


def test_total_fit_ci_too_few_draws_is_none():
    assert _total(_decay, _X, [1.0, 0.5], _COV, n_draws=10) is None


def _boot_fixture():
    rng = np.random.default_rng(0)
    data = np.concatenate([rng.normal(100, 25, 90), rng.normal(150, 45, 40)])
    edges = np.histogram_bin_edges(data, bins=20)
    comps = [MODELS["Gaussian"]] * 2
    popt = np.array([0.8079459459962334, 103.02187056174049, 28.34040788862904,
                     0.24369393573964887, 170.82673738148702, 44.81746981415101])
    x_fine = np.linspace(edges[0], edges[-1], 5)
    return comps, data, edges, popt, composite_bounds(comps, data), x_fine


def test_bootstrap_matches_recorded_interval_for_a_fixed_seed():
    b = _bootstrap(_boot_fixture(), n_draws=60, pct=90.0, seed=3)
    assert np.array_equal(b.lo, [
        0.12351606005041663, 80.72617750524319, 2.317962704538476,
        0.058945645738301494, 111.27096602445954, 2.3960931437153477])
    assert np.array_equal(b.hi, [
        0.992795885469874, 111.04065917715843, 35.59771838533772,
        1.830085726108159, 242.1023226959464, 80.29502322264769])
    assert np.array_equal(b.sd, [
        0.35211578900554497, 32.17355597606711, 25.537345549427346,
        11.321745190470667, 62.78768363465992, 54.18884017655214])
    assert np.array_equal(b.frac_lo, [0.10509595774032031, 0.05769001801157718])
    assert np.array_equal(b.frac_hi, [0.9423099819884229, 0.8949040422596797])
    assert np.array_equal(b.band[0], [
        0.0005186461965121584, 0.008466720815060417, 0.006143924548315972,
        0.0005967951715807862, 1.7422533798216334e-06])
    assert np.array_equal(b.band[1], [
        0.0022346539963647672, 0.019074896989136374, 0.010102308446357497,
        0.003281638178619723, 0.0028096212667288846])
    assert (b.n_draws, b.n_ok, b.n_failed, b.pct, b.seed, b.cancelled) \
        == (60, 60, 0, 90.0, 3, False)


def test_bootstrap_seed_selects_the_resamples():
    fixture = _boot_fixture()
    a = _bootstrap(fixture, with_band=False, n_draws=40, seed=1)
    b = _bootstrap(fixture, with_band=False, n_draws=40, seed=1)
    c = _bootstrap(fixture, with_band=False, n_draws=40, seed=2)
    assert np.array_equal(a.lo, b.lo) and np.array_equal(a.hi, b.hi)
    assert not np.array_equal(a.lo, c.lo)
    assert a.band is None


def test_bootstrap_manifest_carries_the_run_settings():
    b = _bootstrap(_boot_fixture(), n_draws=60, pct=90.0, seed=3)
    assert b.manifest_fields(None) == {
        "ci_band_drawn": True,
        "ci_method": BOOTSTRAP_CI_METHOD,
        "ci_pct": 90.0,
        "ci_n_draws": 60,
        "ci_seed": 3,
        "param_covariance": None,
        "ci_n_draws_ok": 60,
        "ci_n_draws_failed": 0,
        "ci_cancelled": False,
    }


# The manifest is written to disk without sort_keys, so key order is behaviour.
_BASE_KEYS = ["ci_band_drawn", "ci_method", "ci_pct", "ci_n_draws", "ci_seed",
              "param_covariance"]
_BOOT_KEYS = _BASE_KEYS + ["ci_n_draws_ok", "ci_n_draws_failed", "ci_cancelled"]


def test_manifest_key_order_for_the_covariance_band():
    assert list(_manifest(_COV, True)) == _BASE_KEYS


def test_manifest_key_order_for_no_interval_computed():
    out = _manifest(_COV, False, method="none — no interval computed")
    assert list(out) == _BASE_KEYS


def test_bootstrap_manifest_key_order():
    b = _bootstrap(_boot_fixture(), n_draws=60, pct=90.0, seed=3)
    assert list(b.manifest_fields(None)) == _BOOT_KEYS


def test_bootstrap_failed_draw_handler_is_kept():
    src = inspect.getsource(bootstrap_fit_ci)
    assert "noqa: BLE001" in src and "n_failed += 1" in src


# ---- 3. size guard ------------------------------------------------------------

@pytest.mark.parametrize("name", ["ci_manifest_fields", "total_fit_ci"])
def test_ci_functions_meet_the_size_limits(name):
    assert violations(name, function_node(core.__file__, name)) == []


def test_bootstrap_fit_ci_meets_size_limits_apart_from_arguments():
    found = [v for v in violations("bootstrap_fit_ci",
                                   function_node(core.__file__, "bootstrap_fit_ci"))
             if "arguments" not in v]
    assert found == []
