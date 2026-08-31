# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""The dashboard's file-cohort definition and database-query translation."""

from __future__ import annotations


def new_scope() -> dict:
    """Return an unconstrained scope: every file eligible for normal work."""
    return {
        "users": [],
        "analytes": [],
        "solvents": [],
        "techniques": [],
        "substrates": [],
        "sample_preps": [],
        "afm_units": [],
        "curve_types": [],
        "date_from": None,
        "date_to": None,
        "search": None,
        # These invert the normal eligibility filter so set-aside records can
        # be selected explicitly for inspection or catalog removal.
        "only_unusable": False,
        "only_duplicates": False,
    }


def scope_to_query(scope: dict) -> dict:
    """Translate dashboard scope state into ``db.list_files`` arguments."""
    query = {}
    for key in ("users", "analytes", "solvents", "techniques", "substrates",
                "sample_preps", "afm_units", "curve_types"):
        if scope.get(key):
            query[key] = list(scope[key])
    for key in ("date_from", "date_to", "search"):
        if scope.get(key):
            query[key] = scope[key]

    # A normal scope is work eligible for analysis. Set-aside files remain in
    # the catalog and each flag can invert its corresponding eligibility rule.
    query["parse_ok"] = True
    query["usable"] = not bool(scope.get("only_unusable"))
    query["unique"] = not bool(scope.get("only_duplicates"))
    return query
