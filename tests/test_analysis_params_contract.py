"""Architecture guards for the one-snapshot analysis-parameter contract."""

from dataclasses import fields
from pathlib import Path

from smfs_catalog.analysis_params import (
    ANALYSIS_PARAM_DEFAULTS,
    ANALYSIS_PARAM_KEYS,
    AnalysisParams,
)
from smfs_catalog import db as _db


def test_one_declaration_drives_keys_defaults_and_serialization():
    declared = {item.name for item in fields(AnalysisParams)}
    params = AnalysisParams()
    assert declared == set(ANALYSIS_PARAM_KEYS)
    assert declared == set(ANALYSIS_PARAM_DEFAULTS)
    assert declared == set(params.as_dict())
    assert AnalysisParams.from_mapping({}).as_dict() == params.as_dict()


def test_legacy_outer_threshold_still_seeds_missing_inner_threshold():
    params = AnalysisParams.from_mapping({"roi_threshold_nm_per_nm": 3.5})
    assert params.roi_threshold_nm_per_nm == 3.5
    assert params.roi_inner_threshold_nm_per_nm == 3.5


def test_inner_roi_threshold_cannot_exceed_outer_threshold():
    params = AnalysisParams.from_mapping({
        "roi_threshold_nm_per_nm": 3.5,
        "roi_inner_threshold_nm_per_nm": 8.0,
    })
    assert params.roi_inner_threshold_nm_per_nm == 3.5


def test_new_profile_materializes_declared_safe_roi_defaults(tmp_path):
    db_path = str(tmp_path / "catalog.db")
    _db.initialise(db_path)

    params = _db.load_analysis_params(db_path)

    assert params.roi_threshold_nm_per_nm == AnalysisParams().roi_threshold_nm_per_nm
    assert (
        params.roi_inner_threshold_nm_per_nm
        == AnalysisParams().roi_inner_threshold_nm_per_nm
    )


def test_snapshot_is_immutable_and_revision_changes_with_any_field():
    params = AnalysisParams()
    for item in fields(params):
        old = getattr(params, item.name)
        changed = params.with_update(item.name, old + 1)
        assert changed.revision != params.revision, item.name
        assert getattr(params, item.name) == old


def test_retired_parameter_access_paths_cannot_return():
    root = Path(__file__).resolve().parent.parent / "smfs_catalog"
    retired = ("load_param_set", "get_param(", "set_param(", "read_event_params", "_pipeline_params")
    offenders = []
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in retired:
            if token in text:
                offenders.append(f"{path.name}: {token}")
    assert not offenders, "retired parameter path returned: " + ", ".join(offenders)


def test_numerical_core_does_not_load_parameters_from_storage():
    root = Path(__file__).resolve().parent.parent / "smfs_catalog"
    numerical_core = ("signal_processing.py", "roi_detection.py", "roi_events.py")
    for name in numerical_core:
        text = (root / name).read_text(encoding="utf-8")
        assert "load_analysis_params" not in text, name
        assert "experimentalist_profiles" not in text, name
