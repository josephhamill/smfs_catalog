"""Explore Events explains each unplottable curve with its stored fit outcome."""

import json
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from smfs_catalog import db as _db
from smfs_catalog import roi_pipeline as _rp
from smfs_catalog.event_summary_window import (
    EventSummaryWindow, drop_breakdown_lines, fit_outcome_text,
)
from smfs_catalog.ledger import Ledger
from smfs_catalog.provenance import cache_version
from smfs_catalog.roi_events import (
    ROI, CurveEvents, Rupture, Segment, events_to_payload,
)


def _outcome(status, detail=None, *, force=True, length=True, n_segments=1):
    return {"n_segments": n_segments, "fit_status": status, "fit_detail": detail,
            "has_force": force, "has_length": length}


def _window(outcomes):
    win = EventSummaryWindow.__new__(EventSummaryWindow)
    n = len(outcomes)
    win._results = [{"path": f"/data/c{i}.ibw"} for i in range(n)]
    win._force_arr = np.full(n, np.nan)
    win._length_arr = np.full(n, np.nan)
    win._seg_outcome = list(outcomes)
    win._segment_select = "ultimate"
    return win


def _drops(win):
    return {d.path: (d.reason, d.detail) for d in win._plottability_ledger().drops()}


def test_each_drop_names_its_stored_reason():
    win = _window([
        _outcome("fit_available"),                                   # plotted
        None,                                                        # never loaded
        _outcome(None, n_segments=None),                             # no analysis
        _outcome(None, n_segments=1),                                # no such segment
        _outcome("no_fit", "no force peak", force=False, length=False),
        _outcome("no_fit", "optimizer failed", force=True, length=False),
    ])
    win._force_arr[0], win._length_arr[0] = 50.0, 30.0

    drops = _drops(win)

    assert "/data/c0.ibw" not in drops
    assert drops["/data/c1.ibw"] == ("no_fit", "segment: ultimate")
    assert drops["/data/c2.ibw"][0] == "no_stored_segments"
    assert drops["/data/c3.ibw"] == ("no_segment_chosen", "segment: ultimate")
    assert drops["/data/c4.ibw"] == ("no_fit", "no force peak; segment: ultimate")
    assert drops["/data/c5.ibw"] == ("no_length", "optimizer failed; segment: ultimate")


def test_population_ledger_uses_the_same_reasons(monkeypatch):
    win = _window([_outcome("no_fit", "insufficient segment points",
                            force=False, length=False)])
    monkeypatch.setattr(win, "_live_hit_mask", lambda: np.array([True]), raising=False)

    (drop,) = win.population_ledger("hit").drops()

    assert (drop.reason, drop.detail) == (
        "no_fit", "insufficient segment points; segment: ultimate")


def test_breakdown_splits_a_reason_by_stored_outcome():
    led = Ledger("t", ["a", "b", "c"])
    led.drop("a", "no_fit", "no force peak; segment: ultimate")
    led.drop("b", "no_fit", "no force peak; segment: ultimate")
    led.drop("c", "no_fit", "insufficient loading ramp; segment: ultimate")

    lines = drop_breakdown_lines(led)

    assert len(lines) == 2
    assert lines[0].startswith("2 × ") and "no force peak" in lines[0]
    assert lines[1].startswith("1 × ") and "insufficient loading ramp" in lines[1]


@pytest.mark.parametrize("status, detail, text", [
    ("fit_available", None, "fit"),
    ("review", "peak at segment boundary", "review: peak at segment boundary"),
    ("no_fit", "optimizer failed", "optimizer failed"),
    ("not_attempted", None, "not attempted"),
])
def test_outcome_text(status, detail, text):
    assert fit_outcome_text({"fit_status": status, "fit_detail": detail}) == text


def test_segment_summary_reports_the_selected_segments_outcome(tmp_path):
    db = str(tmp_path / "t.sqlite")
    _db.initialise(db)
    path = _db.normalize_path("/data/curve.ibw")
    conn = _db.get_connection(db)
    with conn:
        conn.execute(
            "INSERT INTO files (path, filename, first_seen, last_seen, event)"
            " VALUES (?, 'curve.ibw', datetime('now'), datetime('now'), 'event')",
            (path,))
    fid = conn.execute("SELECT id FROM files WHERE path=?", (path,)).fetchone()[0]
    conn.close()

    def seg(left, right, **kw):
        return Segment(left_idx=left, right_idx=right,
                       left_piezo_nm=float(left), right_piezo_nm=float(right), **kw)

    events = CurveEvents(detector="test", rois=[ROI(
        onset_idx=0, return_idx=300, onset_piezo_nm=0.0, return_piezo_nm=300.0,
        ruptures=[Rupture(idx=i, piezo_nm=float(i), d1_height=1.0, prominence=1.0)
                  for i in (100, 200)],
        segments=[seg(0, 100, fit_status="review", fit_detail="peak at segment boundary"),
                  seg(100, 200, fit_status="no_fit", fit_detail="optimizer failed")],
    )])
    _db.write_event_map(fid, json.dumps(events_to_payload(events)),
                        json.dumps({"tag": "t"}), cache_version() or "test", db)

    ult = _rp.segment_summary_bulk([path], "ultimate", db)[path]
    pen = _rp.segment_summary_bulk([path], "penultimate", db)[path]

    assert (ult["fit_status"], ult["fit_detail"]) == ("no_fit", "optimizer failed")
    assert (pen["fit_status"], pen["fit_detail"]) == ("review", "peak at segment boundary")
    assert "fit_status" not in _rp.SEG_SUMMARY_FIELD.values()
