# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Records what the pass-1 curve functions return today, as plain JSON, so the
c3-c7 refactors can be shown to change nothing.

`collect_all(db_path)` runs, for each of the 5 synthetic curves x the 3 rupture
algorithms: `analyse_curve`, `compute_curve_events_coords` (no cache, from
stage1, cache miss, cache hit, forced), `fit_segments` (and three edge cases),
then `segment_summary_bulk` over the populated catalog for every `select`.
`flatten` turns each output into plain data; `first_difference` compares two
of them.  tests/test_pass1_characterization.py compares against
tests/snapshots/pass1_characterization.json.

Re-generate the snapshot (only ever on code known to be right):

    python -I tests/pass1_snapshot.py --update

Comparison rule: EXACT.  A float is stored as repr(float) (all 17 significant
digits), so `87.31234567890123` vs `87.31234567890124` differs; `nan` equals
`nan` (both are the string "nan").  A numpy array of up to INLINE_ARRAY_MAX
elements is stored value by value.  A longer one (the curves have ~17,000
samples) is stored as dtype, shape, a SHA-256 of its own dtype's raw bytes (NaN
written as one canonical NaN, so NaN equals NaN) plus head/tail/min/max for
reading a failure: equal hash means equal to the last bit.

Left out, and why (everything else is recorded):
  * file ids and the paths of the catalog rows: the summary dict is re-keyed by
    "<variant>/<algorithm>" because the stored path is the machine's drive-letter
    form of the temp folder;
  * database timestamps (computed_at) and the temp folder: never read into an
    output;
  * object identity (the plan says the contact-cache miss changes
    Stage1Search.dc identity; `is` is not recorded, only the contents).

Call adapters.  `_analyse`, `_compute` and `_fit` are the only places this
file spells out how the three functions are called, so a signature change
(c3-c5) is a one-spot edit here.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from typing import NamedTuple
from unittest import mock

import numpy as np

SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "pass1_characterization.json"
ALGORITHMS = ("low_pass_max", "drawdown_walk_back", "muggeo_segmented")
VARIANTS = ("one_rupture", "two_rupture", "flat", "noisy_two_rupture",
            "two_outer_events")
ORIGINAL_CASES = 12     # the first four variants x 3 algorithms; their catalog rows keep their numbers
SELECTS = ("ultimate", "penultimate")
BAD_SELECT = "bogus"
MISSING_PATH = "/data/session/not_in_catalog.ibw"
CODE_VER = "test"
INLINE_ARRAY_MAX = 32


# ── flattening ───────────────────────────────────────────────────────────────

def _flat_float(x) -> dict:
    return {"float": repr(float(x))}


def _digest(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    if a.dtype.kind == "f":
        a = np.where(np.isnan(a), np.array(np.nan, dtype=a.dtype), a)
    return hashlib.sha256(a.tobytes()).hexdigest()


def _flat_array(a: np.ndarray) -> dict:
    body = {"dtype": a.dtype.str, "shape": list(a.shape)}
    if a.size <= INLINE_ARRAY_MAX:
        body["values"] = flatten(a.tolist())
        return {"ndarray": body}
    flat = a.ravel()
    body["sha256"] = _digest(a)
    body["head"] = flatten(flat[:3].tolist())
    body["tail"] = flatten(flat[-3:].tolist())
    body["min"] = flatten(np.nanmin(flat).item())
    body["max"] = flatten(np.nanmax(flat).item())
    return {"ndarray": body}


def _flat_fields(name: str, values: dict) -> dict:
    return {"class": name, "fields": {k: flatten(v) for k, v in values.items()}}


def _flat_dataclass(o) -> dict:
    return _flat_fields(type(o).__name__,
                        {f.name: getattr(o, f.name) for f in dataclasses.fields(o)})


def _flat_namedtuple(o) -> dict:
    return _flat_fields(type(o).__name__, o._asdict())


def _flat_dict(o: dict) -> dict:
    """A list of [key, value] pairs, so insertion order is part of the record."""
    return {"dict": [[flatten(k), flatten(v)] for k, v in o.items()]}


_HANDLERS = (
    (lambda o: o is None, lambda o: None),
    (lambda o: isinstance(o, (bool, np.bool_)), bool),
    (lambda o: isinstance(o, (int, np.integer)), int),
    (lambda o: isinstance(o, (float, np.floating)), _flat_float),
    (lambda o: isinstance(o, str), str),
    (lambda o: isinstance(o, np.ndarray), _flat_array),
    (lambda o: dataclasses.is_dataclass(o) and not isinstance(o, type), _flat_dataclass),
    (lambda o: isinstance(o, tuple) and hasattr(o, "_fields"), _flat_namedtuple),
    (lambda o: isinstance(o, tuple), lambda o: {"tuple": [flatten(v) for v in o]}),
    (lambda o: isinstance(o, list), lambda o: [flatten(v) for v in o]),
    (lambda o: isinstance(o, dict), _flat_dict),
)


def flatten(obj):
    """Plain JSON data for obj.  An unknown type raises: nothing is skipped."""
    for matches, handler in _HANDLERS:
        if matches(obj):
            return handler(obj)
    raise TypeError(f"cannot flatten {type(obj).__name__}")


def _dict_difference(a: dict, b: dict, path: str) -> str | None:
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            return f"{path}/{key}: present on one side only"
        found = first_difference(a[key], b[key], f"{path}/{key}")
        if found:
            return found
    return None


def _list_difference(a: list, b: list, path: str) -> str | None:
    for i, (x, y) in enumerate(zip(a, b)):
        found = first_difference(x, y, f"{path}[{i}]")
        if found:
            return found
    return None


def first_difference(a, b, path: str = "") -> str | None:
    """Where two flattened values first differ (None when equal)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return _dict_difference(a, b, path)
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return _list_difference(a, b, path)
    if type(a) is type(b) and a == b:   # same value AND same kind (True is not 1)
        return None
    return (f"{path}: {type(a).__name__} {str(a)[:80]} != "
            f"{type(b).__name__} {str(b)[:80]}")


def _captured(call, *errors):
    """flatten(call()); an expected error becomes a recorded {"raised": ...}."""
    try:
        return flatten(call())
    except errors as exc:
        return {"raised": type(exc).__name__, "message": str(exc)}


# ── call adapters (the only place the signatures are spelled out) ────────────

class Cache(NamedTuple):
    db_path: str
    file_id: int
    param_set: object = None
    conn: object = None


def _analyse(curve, pipeline, cache: Cache | None = None):
    from smfs_catalog.curve_analysis import CacheContext, analyse_curve
    kw = {} if cache is None else {"cache": CacheContext(
        cache.db_path, CODE_VER, cache.file_id, None)}
    return analyse_curve(curve, pipeline, **kw)


def _compute(curve, ep, *, cache: Cache | None = None,
             stage1=None, force: bool = False):
    from smfs_catalog.roi_pipeline import compute_curve_events_coords
    from smfs_catalog.curve_analysis import CacheContext
    kw = {"stage1": stage1, "force": force}
    if cache is not None:
        kw["cache"] = CacheContext(
            cache.db_path, CODE_VER, cache.file_id, cache.conn,
            param_set=cache.param_set)
    return compute_curve_events_coords(curve, ep, **kw)


def _fit(curve, events, res, ep, scenario: tuple) -> None:
    """scenario = (rupture algorithm, invOLS slope to calibrate with)."""
    from smfs_catalog.roi_events import FitSettings, SegmentCalibration, fit_segments
    algorithm, invols = scenario
    fit_segments(
        curve, events,
        SegmentCalibration(res.offset, invols, res.snap_piezo, low_retr=res.dc.low_retr),
        FitSettings(
            chain_model=ep.chain_model, rupture_algorithm=algorithm,
            walk_back_fit_widths=ep.walk_back_fit_widths,
            walk_back_noise_multiple=ep.walk_back_noise_multiple,
            muggeo_fit_window_multiplier=ep.muggeo_fit_window_multiplier,
            muggeo_tolerance=ep.muggeo_tolerance,
            muggeo_restarts=ep.muggeo_restarts,
        ),
    )


# ── one case: (variant, algorithm) ───────────────────────────────────────────

def _params(algorithm: str):
    from smfs_catalog.analysis_params import AnalysisParams
    from smfs_catalog.rupture_algorithms import RUPTURE_ALGORITHM_BY_IDX
    return AnalysisParams(
        rupture_algorithm_idx=RUPTURE_ALGORITHM_BY_IDX.index(algorithm))


def _analyse_outputs(curve, pipeline, cache: Cache):
    """analyse_curve with no cache, then on a cold and a warm cache."""
    result, stage1 = _analyse(curve, pipeline)
    outputs = {
        "analyse_curve": flatten((result, stage1)),
        "analyse_curve_cache_first": flatten(_analyse(curve, pipeline, cache)),
        "analyse_curve_cache_second": flatten(_analyse(curve, pipeline, cache)),
    }
    return outputs, stage1


def _compute_after_analyse(curve, ep, cache: Cache):
    """compute_curve_events_coords on the row analyse_curve just filled.

    The real batch flow: an event-map miss, with offset, invOLS and snap-off all
    read from analyse_curve's cached rows.  The event map it writes is dropped
    again so the summary still sees this row as having none.
    """
    from smfs_catalog import db
    flat = flatten(_compute(curve, ep, cache=cache))
    db.delete_event_map(cache.file_id, cache.db_path)
    return flat


def _compute_outputs(curve, ep, stage1, cache: Cache) -> dict:
    """compute_curve_events_coords: fresh, from stage1, cache miss, hit, forced."""
    outputs = {
        "compute_no_cache": flatten(_compute(curve, ep)),
        "compute_from_stage1": flatten(_compute(curve, ep, stage1=stage1)),
        "compute_cache_first": flatten(_compute(curve, ep, cache=cache)),
        "compute_cache_second": flatten(_compute(curve, ep, cache=cache)),
    }
    from smfs_catalog import db
    conn = db.get_connection(cache.db_path)
    try:
        with conn:
            forced = _compute(curve, ep, cache=cache._replace(conn=conn),
                              force=True)
        outputs["compute_forced_on_conn"] = flatten(forced)
    finally:
        conn.close()
    return outputs


def _unfitted(curve, ep):
    """The events as built, before fit_segments (its first call is a no-op here)."""
    from smfs_catalog import roi_events
    with mock.patch.object(roi_events, "fit_segments", lambda *a, **k: None):
        return _compute(curve, ep)


def _fit_scenarios(ep) -> dict:
    scenarios = {"fit_segments": (ep.rupture_algorithm, None)}
    if ep.rupture_algorithm == "low_pass_max":   # edge cases once per variant
        scenarios["fit_invalid_calibration"] = (ep.rupture_algorithm, float("nan"))
        scenarios["fit_unknown_algorithm"] = ("no_such_algorithm", None)
        scenarios["fit_invalid_calibration_unknown_algorithm"] = (
            "no_such_algorithm", float("nan"))
    return scenarios


def _fit_outputs(curve, ep) -> dict:
    """The events after fit_segments mutated them in place."""
    base = _unfitted(curve, ep)
    outputs = {"unfitted_events": flatten(base.events)}
    for name, (algorithm, invols) in _fit_scenarios(ep).items():
        events = copy.deepcopy(base.events)
        scenario = (algorithm, base.invols if invols is None else invols)

        def call():
            return _fit(curve, events, base, ep, scenario)
        outputs[name] = {"returned": _captured(call, KeyError),
                         "events": flatten(events)}
    return outputs


def case_outputs(variant: str, algorithm: str, db_path: str,
                 file_ids: tuple[int, int]) -> dict:
    """Every recorded output for one (variant, algorithm), keyed by call name."""
    from curve_factory import make_curve
    from smfs_catalog.curve_analysis import pipeline_params_from
    from smfs_catalog.roi_pipeline import event_params_from
    curve, ap = make_curve(variant), _params(algorithm)
    pipeline, ep = pipeline_params_from(ap), event_params_from(ap)
    analyse_cache = Cache(db_path, file_ids[0])
    compute_cache = Cache(db_path, file_ids[1], param_set=ap)
    outputs, stage1 = _analyse_outputs(curve, pipeline, analyse_cache)
    outputs["compute_after_analyse"] = _compute_after_analyse(
        curve, ep, analyse_cache._replace(param_set=ap))
    outputs.update(_compute_outputs(curve, ep, stage1, compute_cache))
    outputs.update(_fit_outputs(curve, ep))
    return outputs


# ── the catalog and the summary ──────────────────────────────────────────────

def _case_labels() -> list[str]:
    return [f"{v}/{a}" for v in VARIANTS for a in ALGORITHMS]


def _summary_outputs(db_path: str, paths: list[str], labels: dict) -> dict:
    from smfs_catalog.roi_pipeline import segment_summary_bulk
    outputs = {}
    for select in SELECTS + (BAD_SELECT,):
        def call(select=select):
            out = segment_summary_bulk(paths, select, db_path)
            return {labels[k]: v for k, v in out.items()}
        outputs[select] = _captured(call, ValueError)
    return outputs


def _row_layout(n_cases: int) -> tuple[list[int], list[int]]:
    """Catalog row of each case's compute call and of its analyse_curve cache call.

    Rows 0-11 compute and 12-23 analyse for the original cases; any later case
    goes after them, so the original rows keep their numbers and their summary.
    """
    old, extra = min(n_cases, ORIGINAL_CASES), max(0, n_cases - ORIGINAL_CASES)
    compute = list(range(old)) + list(range(2 * old, 2 * old + extra))
    analyse = (list(range(old, 2 * old))
               + list(range(2 * old + extra, 2 * old + 2 * extra)))
    return compute, analyse


def collect_all(db_path: str) -> dict:
    """Everything recorded, on an initialised (empty) catalog at db_path.

    Each case has its own compute row and its own analyse_curve row (see
    _row_layout), so neither call finds the other's cached values, except where
    a call is meant to (compute_after_analyse).
    """
    from conftest import add_files
    from smfs_catalog import db
    cases = [(v, a) for v in VARIANTS for a in ALGORITHMS]
    stored = add_files(db_path, 2 * len(cases))
    fids = [db.get_file_id(p, db_path) for p in stored]
    names = _case_labels()
    compute_rows, analyse_rows = _row_layout(len(cases))
    out = {"cases": {}}
    for i, (variant, algorithm) in enumerate(cases):
        file_ids = (fids[analyse_rows[i]], fids[compute_rows[i]])
        out["cases"][names[i]] = case_outputs(
            variant, algorithm, db_path, file_ids)
    row_names = {r: names[i] for i in range(len(cases))
                 for r in (compute_rows[i], analyse_rows[i])}
    labels = {p: f"row{i:02d}:{row_names[i]}" for i, p in enumerate(stored)}
    labels[db.normalize_path(MISSING_PATH)] = "missing"
    out["summary"] = _summary_outputs(db_path, list(labels), labels)
    return out


def to_json_data(collected: dict):
    """What json.dump + json.load would give (so lists, not tuples)."""
    return json.loads(json.dumps(collected))


def _update() -> None:
    import tempfile
    from smfs_catalog import db
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "catalog.db")
        db.initialise(path)
        data = collect_all(path)
    SNAPSHOT_PATH.parent.mkdir(exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {SNAPSHOT_PATH} ({SNAPSHOT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    sys.path[:0] = [str(root), str(root / "tests")]
    if sys.argv[1:] != ["--update"]:
        sys.exit("usage: python -I tests/pass1_snapshot.py --update")
    _update()
