"""Scientific values stay exact until the quantities-owned display boundary."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from smfs_catalog.analysis_params import AnalysisParams
from smfs_catalog.curve_analysis import pipeline_params_from
from smfs_catalog.event_processor import _phys_grid_params, _wlc_grid_params
from smfs_catalog.roi_pipeline import event_map_params_json, event_params_from


def test_stage_cache_keys_do_not_merge_distinct_parameter_values():
    base = AnalysisParams()
    changed = replace(base, roi_threshold_nm_per_nm=base.roi_threshold_nm_per_nm + 1e-8)
    p0 = pipeline_params_from(base)
    p1 = pipeline_params_from(changed)
    assert p0.params_roi != p1.params_roi
    assert p0.all_params != p1.all_params


def test_event_map_key_keeps_full_snapshot_precision():
    base = AnalysisParams()
    changed = replace(base, roi_prominence=base.roi_prominence + 1e-8)
    assert event_map_params_json(event_params_from(base)) != \
        event_map_params_json(event_params_from(changed))


def test_physical_grid_key_does_not_round_f_star():
    assert _phys_grid_params(50.001) != _phys_grid_params(50.002)


def test_normalized_grid_key_covers_ranges_and_segment():
    base = _wlc_grid_params()
    assert base != _wlc_grid_params(x_range=(-0.1001, 1.2))
    assert base != _wlc_grid_params(f_range=(-5.0, 45.001))
    assert base != _wlc_grid_params(align_segment="first")


def test_loader_and_scanner_do_not_round_sample_rate_before_storage():
    root = Path(__file__).resolve().parent.parent / "smfs_catalog"
    for name in ("curve_loader.py", "scanner.py"):
        source = (root / name).read_text(encoding="utf-8")
        assert "round(1.0 / sfa" not in source, name


def test_matrix_default_round_trips_float64(tmp_path):
    from smfs_catalog import db as _db
    from smfs_catalog import export_utils as export

    db_path = str(tmp_path / "catalog.db")
    _db.initialise(db_path)
    export.set_export_dir_override(str(tmp_path), db_path)
    value = np.nextafter(np.float64(1.0), np.float64(2.0))
    with export.export_group(
        db_path, "precision", ["_matrix.csv"], kind="precision_test",
    ) as group:
        group.matrix("_matrix.csv", [[value]])
    loaded = np.loadtxt(group.path("_matrix.csv"), delimiter=",")
    assert np.float64(loaded) == value


def test_retired_free_form_measurement_formatter_is_absent():
    root = Path(__file__).resolve().parent.parent / "smfs_catalog"
    for path in root.glob("*.py"):
        assert "def _fmt(" not in path.read_text(encoding="utf-8"), path.name
    assert "_fmt(" not in (root / "rawcurve_window.py").read_text(encoding="utf-8")
