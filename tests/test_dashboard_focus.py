"""Dashboard activation must remain a presentation-only event."""

from pathlib import Path


def test_dashboard_focus_does_not_recompute_queue_freshness():
    source = (
        Path(__file__).parents[1] / "smfs_catalog" / "dashboard_window.py"
    ).read_text(encoding="utf-8")

    assert "def changeEvent(" not in source
    assert "QEvent.Type.ActivationChange" not in source
