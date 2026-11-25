# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
The classification report's manifest records the scope as JSON, written with
json.dump(indent=2) and no sort_keys, so the order of the scope dict's keys is
the order of the written text.  The dashboard writes
`scope_to_query(scope).set_fields()`; its key order must stay what it has
always been: the eight facets, then date_from, date_to, search, then the
eligibility flags parse_ok, usable, unique.
"""
import json

from smfs_catalog.scope import new_scope, scope_to_query

_OLD_ORDER = ["users", "analytes", "solvents", "techniques", "substrates",
              "sample_preps", "afm_units", "curve_types", "date_from",
              "date_to", "search", "parse_ok", "usable", "unique"]


def _full_scope():
    scope = new_scope()
    scope.update({
        "users": ["Alice"], "analytes": ["titin"], "solvents": ["PBS"],
        "techniques": ["SMFS"], "substrates": ["mica"],
        "sample_preps": ["drop-cast"], "afm_units": ["AFM-1"],
        "curve_types": ["force_extension"], "date_from": "2026-01-01",
        "date_to": "2026-01-31", "search": "Image0001",
    })
    return scope


def test_scope_note_keys_keep_their_written_order():
    note = scope_to_query(_full_scope()).set_fields()
    assert list(note.keys()) == _OLD_ORDER


def test_scope_note_text_is_what_it_was():
    note = scope_to_query(_full_scope()).set_fields()
    assert json.dumps(note) == (
        '{"users": ["Alice"], "analytes": ["titin"], "solvents": ["PBS"], '
        '"techniques": ["SMFS"], "substrates": ["mica"], '
        '"sample_preps": ["drop-cast"], "afm_units": ["AFM-1"], '
        '"curve_types": ["force_extension"], "date_from": "2026-01-01", '
        '"date_to": "2026-01-31", "search": "Image0001", '
        '"parse_ok": true, "usable": true, "unique": true}')


def test_a_partial_scope_keeps_the_same_relative_order():
    scope = {**new_scope(), "search": "x", "users": ["A"], "only_unusable": True}
    note = scope_to_query(scope).set_fields()
    assert list(note.keys()) == ["users", "search", "parse_ok", "usable", "unique"]
    assert note["usable"] is False
