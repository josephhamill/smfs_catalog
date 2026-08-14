"""Canonical, immutable analysis-parameter snapshot.

Declare an analysis parameter here exactly once.  Storage materialisation,
defaults, enumeration, serialization and revision identity are derived from
this dataclass; numerical code receives an instance and never reads the DB.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class AnalysisParams(Mapping[str, float]):
    """One validated parameter snapshot; the inner ROI tier never exceeds outer."""
    spectral_cutoff_hz: float = 2000.0
    turnaround_trim_pts: int = 100
    var_window_ms: float = 5.0
    detection_threshold_appr: float = 0.05
    detection_threshold_retr: float = 0.05
    baseline_anchor_nm: float = 150.0
    invols_offset_pts: int = 50
    invols_window_pts: int = 200
    roi_window_pts: int = 31
    roi_threshold_nm_per_nm: float = 10.0
    roi_inner_threshold_nm_per_nm: float = 9.0
    roi_post_snapoff_mask_nm: float = 100.0
    roi_onset_threshold_nm: float = -0.2
    roi_detector_mode_idx: int = 0
    roi_prominence: float = 0.1
    roi_min_distance_pts: int = 25

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "AnalysisParams":
        """Validate/coerce a complete or partial stored profile."""
        defaults = cls()
        converted: dict[str, float | int] = {}
        int_names = {
            "turnaround_trim_pts", "invols_offset_pts", "invols_window_pts",
            "roi_window_pts", "roi_detector_mode_idx", "roi_min_distance_pts",
        }
        for item in fields(cls):
            raw = values.get(item.name, getattr(defaults, item.name))
            converted[item.name] = int(float(raw)) if item.name in int_names else float(raw)
        # Legacy profiles predate the inner threshold and stored only the
        # outer threshold.  Preserve that single-tier meaning, but do not let
        # the migration rule override this class's declared default for an
        # empty or otherwise unrelated partial mapping.
        if (
            "roi_threshold_nm_per_nm" in values
            and "roi_inner_threshold_nm_per_nm" not in values
        ):
            converted["roi_inner_threshold_nm_per_nm"] = converted["roi_threshold_nm_per_nm"]
        converted["roi_inner_threshold_nm_per_nm"] = min(
            converted["roi_inner_threshold_nm_per_nm"],
            converted["roi_threshold_nm_per_nm"],
        )
        return cls(**converted)

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def with_update(self, key: str, value: float) -> "AnalysisParams":
        if key not in ANALYSIS_PARAM_KEYS:
            raise KeyError(f"unknown analysis parameter: {key}")
        return AnalysisParams.from_mapping({**self.as_dict(), key: value})

    @property
    def revision(self) -> str:
        """Canonical identity used to detect an edit during a calculation."""
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))

    # Mapping compatibility keeps profile/export consumers simple while the
    # only object they can receive remains typed and immutable.
    def __getitem__(self, key: str) -> float | int:
        if key not in ANALYSIS_PARAM_KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(ANALYSIS_PARAM_KEYS)

    def __len__(self) -> int:
        return len(ANALYSIS_PARAM_KEYS)


ANALYSIS_PARAM_KEYS = frozenset(item.name for item in fields(AnalysisParams))
ANALYSIS_PARAM_DEFAULTS = AnalysisParams().as_dict()
