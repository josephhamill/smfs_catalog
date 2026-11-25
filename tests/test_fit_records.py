# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Fit savers take one row object each (quality pass 1, step d4).

`DistFitRecord` and `GmmFitRecord` (smfs_catalog/fit_records.py, no Qt, no db
import) are frozen dataclasses holding exactly today's row parameters of
`save_distribution_fit` / `save_gmm_fit`, minus `db_path`:

  DistFitRecord: variable, units, n_values, n_peaks, model_label, params_json,
                 gof_json, fit_config_json, experimentalist=""
  GmmFitRecord:  x_variable, y_variable, n_values, k_components, cov_type,
                 means_json, covs_json, weights_json, gof_json,
                 fit_config_json="{}", experimentalist=""

The savers become `save_distribution_fit(record, db_path)` and
`save_gmm_fit(record, db_path)`.  The pins write rows through the adapters
below (the only place that knows the call shape), read every column back and
check the saved-fit history loaders return the same rows.
"""
import dataclasses
import inspect
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from smfs_catalog import db

sys.path.insert(0, str(Path(__file__).parent))
from size_limits import violations  # noqa: E402
from source_inspect import function_node, imported_names  # noqa: E402

DIST_FIELDS = ("variable", "units", "n_values", "n_peaks", "model_label",
               "params_json", "gof_json", "fit_config_json", "experimentalist")
GMM_FIELDS = ("x_variable", "y_variable", "n_values", "k_components",
              "cov_type", "means_json", "covs_json", "weights_json",
              "gof_json", "fit_config_json", "experimentalist")

DIST = dict(
    variable="rupture_force", units="pN", n_values=120, n_peaks=2,
    model_label="2-peak Gaussian",
    params_json=json.dumps([{"name": "mu1", "value": 41.5}]),
    gof_json=json.dumps({"r2": 0.97}),
    fit_config_json=json.dumps({"n_bins": 30, "constrained": False}),
)
GMM = dict(
    x_variable="rupture_force", y_variable="loading_rate", n_values=80,
    k_components=2, cov_type="full",
    means_json=json.dumps([[1.0, 2.0], [3.0, 4.0]]),
    covs_json=json.dumps([[[1.0, 0.0], [0.0, 1.0]]] * 2),
    weights_json=json.dumps([0.4, 0.6]),
    gof_json=json.dumps({"bic": 12.5}),
)


# -- adapters: the only code that knows how the savers are called -----------

def _save_dist(catalog, **kw):
    db.save_distribution_fit(_records().DistFitRecord(**{**DIST, **kw}), catalog)


def _save_gmm(catalog, **kw):
    db.save_gmm_fit(_records().GmmFitRecord(**{**GMM, **kw}), catalog)


def _rows(catalog, table):
    conn = sqlite3.connect(catalog)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(f"SELECT * FROM {table}")]
    finally:
        conn.close()


def _records():
    from smfs_catalog import fit_records
    return fit_records


# -- interface (fail until the record objects exist) -------------------------

@pytest.mark.parametrize("cls_name, fields", [
    ("DistFitRecord", DIST_FIELDS), ("GmmFitRecord", GMM_FIELDS)])
def test_record_has_exactly_todays_fields(cls_name, fields):
    cls = getattr(_records(), cls_name)
    assert tuple(f.name for f in dataclasses.fields(cls)) == fields


@pytest.mark.parametrize("cls_name, kwargs", [
    ("DistFitRecord", DIST), ("GmmFitRecord", GMM)])
def test_record_is_frozen(cls_name, kwargs):
    rec = getattr(_records(), cls_name)(**kwargs)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.n_values = 1


def test_record_defaults_match_todays_signature():
    r = _records()
    d = r.DistFitRecord(**DIST)
    g = r.GmmFitRecord(**GMM)
    assert d.experimentalist == "" and g.experimentalist == ""
    assert g.fit_config_json == "{}"
    required = {f.name for f in dataclasses.fields(r.DistFitRecord)
                if f.default is dataclasses.MISSING}
    assert required == set(DIST)          # units and fit_config_json stay required


def test_fit_records_module_imports_no_qt_and_no_db():
    segments = {seg.lower() for n in imported_names(_records().__file__)
                for seg in n.split(".")}
    banned = {s for s in segments
              if s == "db" or s.startswith(("pyqt", "pyside", "qt"))}
    assert not banned, f"fit_records must not import {sorted(banned)}"


def test_save_distribution_fit_takes_record_and_db_path():
    assert list(inspect.signature(db.save_distribution_fit).parameters) == [
        "record", "db_path"]


def test_save_gmm_fit_takes_record_and_db_path():
    assert list(inspect.signature(db.save_gmm_fit).parameters) == [
        "record", "db_path"]


# -- behaviour pins (pass today and after) -----------------------------------

def test_distribution_fit_row_has_every_column(catalog):
    _save_dist(catalog)
    (row,) = _rows(catalog, "distribution_fits")
    created = row.pop("created_at")
    assert len(created) == 19 and created[10] == "T"
    assert row == {**DIST, "id": 1, "experimentalist": ""}


def test_distribution_fit_keeps_experimentalist(catalog):
    _save_dist(catalog, experimentalist="alice")
    (row,) = _rows(catalog, "distribution_fits")
    assert row["experimentalist"] == "alice"


def test_gmm_fit_row_has_every_column_and_default_config(catalog):
    _save_gmm(catalog)
    (row,) = _rows(catalog, "gmm_fits")
    created = row.pop("created_at")
    assert len(created) == 19 and created[10] == "T"
    assert row == {**GMM, "id": 1, "fit_config_json": "{}",
                   "experimentalist": ""}


def test_gmm_fit_keeps_config_and_experimentalist(catalog):
    _save_gmm(catalog, fit_config_json='{"n_init": 5}', experimentalist="bob")
    (row,) = _rows(catalog, "gmm_fits")
    assert row["fit_config_json"] == '{"n_init": 5}'
    assert row["experimentalist"] == "bob"


def test_saved_distribution_fits_load_back_unchanged(catalog):
    _save_dist(catalog)
    _save_dist(catalog, variable="other")
    got = db.get_distribution_fits("rupture_force", catalog)
    assert [dict(r) for r in got] == _rows(catalog, "distribution_fits")[:1]
    assert json.loads(got[0]["params_json"]) == [{"name": "mu1", "value": 41.5}]


def test_saved_gmm_fits_load_back_unchanged(catalog):
    _save_gmm(catalog)
    _save_gmm(catalog, y_variable="other")
    got = db.get_gmm_fits("rupture_force", "loading_rate", catalog)
    assert [dict(r) for r in got] == _rows(catalog, "gmm_fits")[:1]
    assert json.loads(got[0]["weights_json"]) == [0.4, 0.6]


# -- size guard ---------------------------------------------------------------

@pytest.mark.parametrize("name", ["save_distribution_fit", "save_gmm_fit"])
def test_saver_within_size_limits(name):
    assert violations(name, function_node(db.__file__, name)) == []


def test_dist_fit_model_pane_save_fit_is_within_size_limits():
    path = Path(db.__file__).parent / "dist_fit_window.py"
    fn = function_node(path, "_save_fit", "_ModelPane")
    assert violations("_ModelPane._save_fit", fn) == []
