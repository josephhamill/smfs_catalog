from __future__ import annotations

import csv
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtWidgets import QApplication

from smfs_catalog import db, export_utils
from smfs_catalog.gmm_fit_window import _ModelPane, _ScatterPane


_APP = QApplication.instance() or QApplication([])


def _reject_nonstandard_json(token: str):
    raise ValueError(f"non-standard JSON constant: {token}")


def test_gmm_export_is_complete_strict_json_and_uses_displayed_ids(
    tmp_path, monkeypatch,
):
    db_path = str(tmp_path / "catalog.db")
    db.initialise(db_path)
    export_utils.set_export_dir_override(str(tmp_path), db_path)

    rng = np.random.default_rng(12)
    xy = np.vstack([
        rng.normal([80.0, 60.0], [3.0, 4.0], size=(15, 2)),
        rng.normal([160.0, 130.0], [5.0, 6.0], size=(10, 2)),
    ])
    paths = [f"C:/data/curve_{i}.ibw" for i in range(len(xy))]
    scatter = _ScatterPane(xy, "length", "force")
    pane = _ModelPane(scatter, xy, "length", "force", db_path, paths=paths)
    pane._k_spin.setValue(2)
    pane._fit()
    monkeypatch.setattr(
        "smfs_catalog.gmm_fit_window.QMessageBox.information",
        lambda *args, **kwargs: None,
    )
    pane._export_fit()

    manifests = list(tmp_path.glob("fit_gmm_length_force_*_manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(
        manifests[0].read_text(encoding="utf-8"),
        parse_constant=_reject_nonstandard_json,
    )
    assert manifest["export"] == "gmm_fit"
    assert manifest["n_files"] == len(xy)
    assert len(manifest["data_files"]) == 2

    base = manifests[0].name.removesuffix("_manifest.json")
    with open(tmp_path / f"{base}_components.csv", newline="", encoding="utf-8") as f:
        components = list(csv.DictReader(f))
    with open(tmp_path / f"{base}_points.csv", newline="", encoding="utf-8") as f:
        points = list(csv.DictReader(f))

    assert [int(row["component"]) for row in components] == [1, 2]
    assert {int(row["component"]) for row in points} <= {1, 2}
    assert [row["path"] for row in points] == paths
    assert all(0.0 <= float(row["responsibility"]) <= 1.0 for row in points)
