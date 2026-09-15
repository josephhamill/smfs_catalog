"""The Isoforce empty message says whether the other population has curves."""

import pytest

from smfs_catalog import event_summary_window as esw
from smfs_catalog.event_summary_window import EventSummaryWindow


def _empty_message(monkeypatch, pop, qualifying):
    win = EventSummaryWindow.__new__(EventSummaryWindow)
    win._active_population = pop
    monkeypatch.setattr(win, "_isoforce_paths",
                        lambda which: qualifying.get(which, []), raising=False)
    shown = []
    monkeypatch.setattr(esw.QMessageBox, "information",
                        lambda _parent, _title, text: shown.append(text))
    win._on_view_isoforce()
    assert len(shown) == 1
    return shown[0]


def test_empty_everywhere(monkeypatch):
    msg = _empty_message(monkeypatch, "hit", {})
    assert "current Hits population" in msg
    assert "Switch" not in msg


@pytest.mark.parametrize("pop, other, other_name", [
    ("hit", "non_hit", "Non-Hits"),
    ("non_hit", "hit", "Hits"),
])
def test_candidates_only_in_other_population(monkeypatch, pop, other, other_name):
    msg = _empty_message(monkeypatch, pop, {other: ["a", "b", "c"]})
    assert f"3 curve(s) in {other_name} do" in msg
    assert f"Switch to {other_name}" in msg
