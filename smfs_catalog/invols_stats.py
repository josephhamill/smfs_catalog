"""Pure invOLS arithmetic: no Qt, no database."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

DIVISOR_SIGN: float = -1.0

TRANSFORM_KEY = "transform_invols_nm_v"

AVERAGES: tuple[str, ...] = ("arithmetic", "harmonic")


@dataclass(frozen=True)
class Summary:
    n: int
    mean: float | None
    harmonic_mean: float | None
    sd: float | None
    se: float | None


@dataclass(frozen=True)
class DisplayRange:
    lo: float
    hi: float
    side: tuple[int, ...]


def pick(n: int, one: str, many: str) -> str:
    return one if n == 1 else many


def plural(n: int, one: str, many: str) -> str:
    return f"{n} {pick(n, one, many)}"


def is_finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def is_drawn(value: float | None, time_s: float | None) -> bool:
    """A trace is drawn, and counts toward a mean, only with a finite own value
    and a finite acquisition time."""
    return is_finite(value) and is_finite(time_s)


def usable_instrument(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0


def own_transform_nm_v(instrument_nm_v: float | None, slope: float | None) -> float | None:
    if slope is None or not usable_instrument(instrument_nm_v):
        return None
    if not math.isfinite(slope) or slope * DIVISOR_SIGN <= 0:
        return None
    value = instrument_nm_v / abs(slope)
    return value if math.isfinite(value) else None


def summarise(values: Iterable[float | None]) -> Summary:
    kept = [float(v) for v in values if v is not None and math.isfinite(v)]
    n = len(kept)
    if n == 0:
        return Summary(0, None, None, None, None)
    mean = math.fsum(kept) / n
    harmonic = None if min(kept) <= 0 else n / math.fsum(1.0 / v for v in kept)
    if n == 1:
        return Summary(1, mean, harmonic, None, None)
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in kept) / (n - 1))
    return Summary(n, mean, harmonic, sd, sd / math.sqrt(n))


def applied_value(summary: Summary, average: str) -> float | None:
    if average == "arithmetic":
        return summary.mean
    if average == "harmonic":
        return summary.harmonic_mean
    raise ValueError(f"unknown average: {average!r}")


def ensemble_divisor(instrument_nm_v: float, transform_nm_v: float) -> float:
    return DIVISOR_SIGN * instrument_nm_v / transform_nm_v


def resolve_divisor(own_slope: float | None, ensemble: float | None) -> float | None:
    if ensemble is not None:
        return ensemble
    if own_slope is not None and math.isfinite(own_slope) and own_slope != 0:
        return own_slope
    return None


def cache_key(all_params: str, ensemble: float | None) -> str:
    if ensemble is None:
        return all_params
    if not math.isfinite(ensemble):
        raise ValueError("ensemble divisor must be finite")
    return f"{all_params}|invols:{json.dumps(float(ensemble))}"


def display_range(values: Sequence[float]) -> DisplayRange:
    arr = np.asarray(values, dtype=float)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    low_fence = q1 - 1.5 * iqr
    high_fence = q3 + 1.5 * iqr
    inside = (arr >= low_fence) & (arr <= high_fence)
    bulk = arr[inside]
    lo, hi = float(bulk.min()), float(bulk.max())
    span = hi - lo
    if span == 0:
        pad = 0.01 * abs(lo) if lo != 0 else 1.0
    else:
        pad = 0.05 * span
    side = np.where(arr < low_fence, -1, np.where(arr > high_fence, 1, 0))
    return DisplayRange(lo - pad, hi + pad, tuple(int(s) for s in side))


def points_in_box(t: Sequence[float], y: Sequence[float],
                  box: tuple[float, float, float, float]) -> np.ndarray:
    t0, t1, y0, y1 = box
    t_arr = np.asarray(t, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    return ((t_arr >= min(t0, t1)) & (t_arr <= max(t0, t1))
            & (y_arr >= min(y0, y1)) & (y_arr <= max(y0, y1)))


def in_time_span(t: Sequence[float], y: Sequence[float],
                 box: tuple[float, float, float, float]) -> np.ndarray:
    t0, t1, _, _ = box
    return points_in_box(t, y, (t0, t1, -math.inf, math.inf))


def box_cut_count(t: Sequence[float], y: Sequence[float],
                  box: tuple[float, float, float, float]) -> int:
    in_span = in_time_span(t, y, box)
    return int(np.count_nonzero(in_span & ~points_in_box(t, y, box)))
