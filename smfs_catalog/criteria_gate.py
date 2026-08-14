# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""Apply the active criteria profile to a cohort of event curves."""

from __future__ import annotations

import math

from . import db as _db


_PREFIX = "criteria_use:"
FailReason = tuple[str, "float | None", str, "float | None"]


def get_criteria(
    experimentalist: str | None = None,
    db_path: str = _db.DEFAULT_DB_PATH,
) -> set[str]:
    """Return the variable keys enabled in an experimentalist's profile."""
    key = experimentalist or _db.DEFAULT_EXPERIMENTALIST
    profile = _db.get_experimentalist_profile(key, db_path)
    if not profile and key != _db.DEFAULT_EXPERIMENTALIST:
        profile = _db.get_experimentalist_profile(
            _db.DEFAULT_EXPERIMENTALIST, db_path)
    return {
        key[len(_PREFIX):]
        for key, enabled in (profile or {}).items()
        if key.startswith(_PREFIX) and enabled
    }


def set_criterion(
    key: str,
    on: bool,
    experimentalist: str | None = None,
    db_path: str = _db.DEFAULT_DB_PATH,
) -> None:
    """Enable or disable one variable in an experimentalist's profile."""
    owner = experimentalist or _db.DEFAULT_EXPERIMENTALIST
    _db.merge_experimentalist_profile(
        owner, {_PREFIX + key: 1.0 if on else 0.0}, db_path)


def active_owner(db_path: str = _db.DEFAULT_DB_PATH) -> str:
    """Return the profile owner whose criteria currently govern the queue."""
    return _db.active_param_owner(db_path)


def _bounds(
    experimentalist: str | None,
    db_path: str,
) -> dict[str, tuple[float | None, float | None]]:
    """Return the effective lower and upper bounds by variable key."""
    return {
        row["analysis_type"]: (row["lower_bound"], row["upper_bound"])
        for row in _db.get_thresholds(experimentalist, db_path)
    }


def _active_gate(
    owner: str,
    db_path: str,
) -> tuple[list[str], dict[str, tuple[float | None, float | None]]]:
    """Return enabled variables that have at least one defined bound."""
    bounds = _bounds(owner, db_path)
    checked = sorted(
        key
        for key in get_criteria(owner, db_path)
        if bounds.get(key, (None, None)) != (None, None)
    )
    return checked, bounds


def get_active_criteria(
    experimentalist: str | None = None,
    db_path: str = _db.DEFAULT_DB_PATH,
) -> set[str]:
    """Return enabled criteria that have at least one effective bound."""
    owner = experimentalist or _db.DEFAULT_EXPERIMENTALIST
    checked, _ = _active_gate(owner, db_path)
    return set(checked)


def _values(
    event_paths: list[str],
    checked: list[str],
    db_path: str,
) -> dict[str, dict[str, "float | None"]]:
    """Load the selected variables for the supplied paths."""
    from . import variables as _vars

    return _vars.values(event_paths, checked, db_path)


def _failures(
    paths: list[str],
    checked: list[str],
    bounds: dict[str, tuple[float | None, float | None]],
    db_path: str,
) -> dict[str, list[FailReason]]:
    """Return every failed criterion for each failing path."""
    values = _values(paths, checked, db_path)
    failures: dict[str, list[FailReason]] = {}
    for path in paths:
        path_values = values.get(_db.normalize_path(path), {})
        reasons: list[FailReason] = []
        for key in checked:
            value = path_values.get(key)
            if value is None or not math.isfinite(value):
                reasons.append((key, None, "missing", None))
                continue
            lower, upper = bounds[key]
            if lower is not None and value < lower:
                reasons.append((key, value, "below", lower))
            elif upper is not None and value > upper:
                reasons.append((key, value, "above", upper))
        if reasons:
            failures[path] = reasons
    return failures


def has_criteria_checked(
    paths: list[str],
    db_path: str = _db.DEFAULT_DB_PATH,
) -> dict[str, bool]:
    """Report whether the cohort's active profile has any bounded criterion."""
    owner = active_owner(db_path)
    checked, _ = _active_gate(owner, db_path)
    active = bool(checked)
    return {path: active for path in paths}


def evaluate(
    event_paths: list[str],
    db_path: str = _db.DEFAULT_DB_PATH,
) -> tuple[list[str], list[str]]:
    """Split paths into hits and non-hits using the first queued file's profile."""
    if not event_paths:
        return [], []
    owner = active_owner(db_path)
    checked, bounds = _active_gate(owner, db_path)
    if not checked:
        return list(event_paths), []
    failures = _failures(event_paths, checked, bounds, db_path)
    return (
        [path for path in event_paths if path not in failures],
        [path for path in event_paths if path in failures],
    )


def explain(
    event_paths: list[str],
    db_path: str = _db.DEFAULT_DB_PATH,
) -> dict[str, list[FailReason]]:
    """Explain every non-hit using the first queued file's active profile."""
    if not event_paths:
        return {}
    owner = active_owner(db_path)
    checked, bounds = _active_gate(owner, db_path)
    if not checked:
        return {}
    return _failures(event_paths, checked, bounds, db_path)
