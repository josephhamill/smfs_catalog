import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from smfs_catalog import criteria_gate as gate
from smfs_catalog import db
from smfs_catalog.criteria_dialog import CriteriaDialog


_app = QApplication.instance() or QApplication([])


def _mixed_queue(db_path: str, tmp_path: Path) -> list[str]:
    paths = [db.normalize_path(str(tmp_path / name)) for name in ("a.ibw", "b.ibw")]
    conn = db.get_connection(db_path)
    with conn:
        ids = []
        for path, owner in zip(paths, ("A", "B")):
            cursor = conn.execute(
                "INSERT INTO files(path, filename, first_seen, "
                "last_seen, experimentalist, event) VALUES (?, ?, ?, ?, ?, ?)",
                (path, Path(path).name, "now", "now", owner, "event"),
            )
            ids.append(cursor.lastrowid)
    conn.close()
    db.enqueue_files(ids, db_path)
    return paths


def test_mixed_queue_dialog_uses_gate_owner_and_unbounded_is_inactive(
    tmp_path, monkeypatch
):
    db_path = str(tmp_path / "criteria.db")
    db.initialise(db_path)
    paths = _mixed_queue(db_path, tmp_path)
    gate.set_criterion("metric", True, "A", db_path)

    dialog = CriteriaDialog([("metric", "Metric")], paths, db_path)
    try:
        assert dialog._experimentalist == gate.active_owner(db_path) == "A"
        assert dialog._context_label.text() == "Criteria owner: A"
        assert dialog._rows[0][1].isChecked()
        assert dialog._count_lbl.text().startswith("No active criteria")
        assert "without a bound it does not constrain" in dialog._rows[0][1].toolTip()

        captured = {}

        class Signal:
            def connect(self, callback):
                pass

        class FakeVariableWindow:
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)
                self.thresholds_changed = Signal()
                self.view_file_requested = Signal()

            def show(self):
                pass

            def raise_(self):
                pass

            def activateWindow(self):
                pass

        from smfs_catalog import variable_window

        monkeypatch.setattr(variable_window, "VariableStatsWindow", FakeVariableWindow)
        dialog._edit_bounds("metric", "Metric")
        assert captured["experimentalist"] == "A"
    finally:
        dialog.close()
