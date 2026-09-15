"""ROI/segment export rows carry each segment's stored fit outcome."""

import json

from smfs_catalog.event_summary_window import EventSummaryWindow
from smfs_catalog.roi_assembly import project_curve_events
from smfs_catalog.roi_events import (
    ROI, CurveEvents, Rupture, Segment, events_to_payload, payload_to_events,
)


def _segment(left, right, **kw) -> Segment:
    return Segment(left_idx=left, right_idx=right,
                   left_piezo_nm=float(left), right_piezo_nm=float(right), **kw)


def test_rows_carry_the_stored_fit_outcome():
    outcomes = [
        ("fit_available", None),
        ("review", "peak at segment boundary"),
        ("no_fit", "insufficient segment points"),
    ]
    events = CurveEvents(detector="test", rois=[ROI(
        onset_idx=0, return_idx=400, onset_piezo_nm=0.0, return_piezo_nm=400.0,
        ruptures=[Rupture(idx=i, piezo_nm=float(i), d1_height=1.0, prominence=1.0)
                  for i in (100, 200, 300)],
        segments=[_segment(l, r, fit_status=s, fit_detail=d)
                  for (l, r), (s, d) in zip([(0, 100), (100, 200), (200, 300)], outcomes)],
    )])
    stored = payload_to_events(json.loads(json.dumps(events_to_payload(events))))

    rows = project_curve_events(stored, mode="all", path="curve")

    assert [(r["fit_status"], r["fit_detail"]) for r in rows] == outcomes


def test_export_columns_include_the_fit_outcome():
    keys = [k for _h, k in EventSummaryWindow._ROI_SEGMENT_COLUMNS]
    assert "fit_status" in keys
    assert "fit_detail" in keys
