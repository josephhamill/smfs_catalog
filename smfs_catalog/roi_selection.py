"""Presentation policy for selecting and overriding reported ROI segments.

This module deliberately contains no event detection, fitting, or event-map
persistence. It translates the stored Ultimate/Penultimate policy and manual
Primary/Secondary picks into segment indices for reporting consumers. Manual
picks are accepted only while their event geometry still matches.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from . import db as _db


def read_segment_select(db_path: str) -> str:
    """Return the global dashboard reporting policy."""
    return (
        "ultimate"
        if _db.get_setting("summary_segment_select", 1.0, db_path) >= 0.5
        else "penultimate"
    )


def write_segment_select(value: str, db_path: str) -> None:
    """Persist the global dashboard reporting policy."""
    if value not in {"ultimate", "penultimate"}:
        raise ValueError(f"unknown segment selection policy: {value!r}")
    _db.set_setting(
        "summary_segment_select", 1.0 if value == "ultimate" else 0.0, db_path,
    )


@dataclass(frozen=True)
class ReportedSegmentChoice:
    """The one segment supplying dashboard/scatter/histogram scalar values."""

    segment_idx: int
    source: str  # "ultimate" | "penultimate" | "manual_primary"


def resolve_reported_segment(
    n_segments: int,
    policy: str,
    primary_idx: int | None = None,
) -> ReportedSegmentChoice | None:
    """Resolve the reported scalar segment; unrelated to 2DH alignment."""
    if policy not in {"ultimate", "penultimate"}:
        raise ValueError(f"unknown segment selection policy: {policy!r}")
    if n_segments <= 0:
        return None
    if primary_idx is not None and 0 <= primary_idx < n_segments:
        return ReportedSegmentChoice(primary_idx, "manual_primary")
    if policy == "penultimate" and n_segments >= 2:
        return ReportedSegmentChoice(n_segments - 2, "penultimate")
    return ReportedSegmentChoice(n_segments - 1, "ultimate")


@dataclass(frozen=True)
class SegmentOverrideResolution:
    primary_idx: int | None
    secondary_idx: int | None
    status: str  # "none" | "current" | "needs_review"


def event_geometry_identity(events) -> str:
    """Return the stable identity of boundaries a manual pick refers to."""
    geometry = [
        {
            "onset": roi.onset_idx,
            "return": roi.return_idx,
            "ruptures": [
                [rup.idx, rup.rise_idx, rup.fall_idx] for rup in roi.ruptures
            ],
            "segments": [
                [seg.left_idx, seg.right_idx] for seg in roi.segments
            ],
        }
        for roi in events.rois
    ]
    encoded = json.dumps(geometry, separators=(",", ":"), sort_keys=True).encode()
    return "geometry:v1:" + hashlib.sha256(encoded).hexdigest()


def resolve_segment_override_state(
    override: dict,
    current_params_json: str | None,
    n_segments: int,
    current_geometry: str | None = None,
) -> SegmentOverrideResolution:
    """Resolve a stored pick while retaining an explicit stale/review state."""
    primary_raw = override.get("primary_segment_idx")
    secondary_raw = override.get("secondary_segment_idx")
    if primary_raw is None and secondary_raw is None:
        return SegmentOverrideResolution(None, None, "none")

    tag = override.get("params_json")
    if isinstance(tag, str) and tag.startswith("geometry:v1:"):
        matches = current_geometry is not None and tag == current_geometry
    else:
        # Legacy picks predate geometry identities and were tied to the entire
        # event-map parameter document. Continue honoring them until replaced.
        matches = (
            tag is not None
            and current_params_json is not None
            and tag == current_params_json
        )
    if not matches:
        return SegmentOverrideResolution(None, None, "needs_review")

    def _valid(idx):
        return idx if isinstance(idx, int) and 0 <= idx < n_segments else None

    primary = _valid(primary_raw)
    secondary = _valid(secondary_raw)
    status = "current" if primary is not None or secondary is not None else "needs_review"
    return SegmentOverrideResolution(primary, secondary, status)


def resolve_segment_override(
    override: dict,
    current_params_json: str | None,
    n_segments: int,
    current_geometry: str | None = None,
) -> tuple[int | None, int | None]:
    """Return the currently usable primary and secondary segment indices."""
    state = resolve_segment_override_state(
        override, current_params_json, n_segments, current_geometry,
    )
    return state.primary_idx, state.secondary_idx


def resolve_isoforce_pair(
    n_segments: int,
    primary_idx: int | None,
    secondary_idx: int | None,
) -> tuple[int, int] | None:
    """Return the rupture/segment pair used for isoforce measurements.

    A complete manual pair takes precedence and is ordered by position, not by
    which role was clicked first.  With no complete pair, retain the established
    last-two-ruptures default.  Isoforce geometry exists only for adjacent
    segments because each segment's crossing is defined relative to the
    immediately preceding rupture.
    """
    if primary_idx is not None and secondary_idx is not None:
        lo, hi = sorted((primary_idx, secondary_idx))
        return (lo, hi) if hi == lo + 1 else None
    if n_segments >= 2:
        return n_segments - 2, n_segments - 1
    return None
