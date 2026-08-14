import os
import csv

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtWidgets import QApplication, QMessageBox

from smfs_catalog import db, export_utils
from smfs_catalog.variable_window import VariableStatsWindow


_app = QApplication.instance() or QApplication([])


def test_empty_scope_still_loads_effective_thresholds(tmp_path):
    db_path = str(tmp_path / "catalog.db")
    db.initialise(db_path)
    db.set_threshold("metric", 1.25, 4.75, "Metric", "owner", db_path)

    win = VariableStatsWindow(
        "metric", "Metric", [], db_path, experimentalist="owner"
    )
    try:
        assert win._lo == 1.25
        assert win._hi == 4.75
        assert win._chk_lo.isChecked()
        assert win._chk_hi.isChecked()
        assert win._spin_lo.value() == 1.25
        assert win._spin_hi.value() == 4.75
    finally:
        win.close()


def test_apply_rejects_reversed_bounds_without_writing(tmp_path, monkeypatch):
    db_path = str(tmp_path / "catalog.db")
    db.initialise(db_path)
    db.set_threshold("metric", 1.0, 5.0, "Metric", "owner", db_path)
    warnings = []
    monkeypatch.setattr(
        QMessageBox, "warning",
        lambda *args: warnings.append(args[2]),
    )

    win = VariableStatsWindow(
        "metric", "Metric", [], db_path, experimentalist="owner"
    )
    try:
        win._spin_lo.setValue(9.0)
        win._spin_hi.setValue(2.0)
        win._apply_thresholds()
        row = db.get_threshold("metric", "owner", db_path)
        assert (row["lower_bound"], row["upper_bound"]) == (1.0, 5.0)
        assert warnings == [
            "The lower threshold must not be greater than the upper threshold."
        ]
    finally:
        win.close()


def test_missing_values_remain_in_scope_but_not_in_finite_or_plot_arrays(
    tmp_path, monkeypatch
):
    db_path = str(tmp_path / "catalog.db")
    db.initialise(db_path)
    db.set_threshold("metric", 1.0, 3.0, "Metric", "owner", db_path)
    paths = [db.normalize_path(str(tmp_path / name)) for name in ("a", "b", "c")]

    monkeypatch.setattr(
        "smfs_catalog.variable_window._vars.values",
        lambda supplied, keys, path: {
            paths[0]: {"metric": 2.0},
            paths[1]: {"metric": None},
            paths[2]: {"metric": np.inf},
        },
    )
    monkeypatch.setattr(
        "smfs_catalog.variable_window._db.get_measured_datetimes",
        lambda supplied, path: {p: "2026-01-01 12:00:00" for p in paths},
    )
    monkeypatch.setattr(
        "smfs_catalog.variable_window._db.get_derived_results_bulk_latest",
        lambda supplied, keys, path: {p: {} for p in paths},
    )

    win = VariableStatsWindow(
        "metric", "Metric", paths, db_path, experimentalist="owner"
    )
    try:
        assert win._raw_paths == paths
        assert win._raw_vals.size == 3
        assert win._finite_v.tolist() == [2.0]
        assert win._plot_paths == [paths[0]]
        assert win._n_missing_value == 2
        assert "2 missing/non-finite" in win._info.text()

        export_utils.set_export_dir_override(str(tmp_path), db_path)
        monkeypatch.setattr(QMessageBox, "information", lambda *args: None)
        win._on_export()
        exported = next(tmp_path.glob("variable_metric*_timeseries.csv"))
        with exported.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 3
        assert [row["passes_bounds"] for row in rows] == ["True", "False", "False"]
        assert [row["value"] for row in rows] == ["2.0", "", ""]
    finally:
        win.close()
