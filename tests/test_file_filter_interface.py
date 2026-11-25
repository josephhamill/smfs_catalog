# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
The file filter is one `FileFilter` object, not twenty keywords.

`FileFilter` (smfs_catalog/file_filter.py) is a frozen dataclass; every field
defaults to None, meaning "no constraint".  Its fields are today's filter
keywords of `db.list_files`:

  directory, directories       under one folder / under each of several folders
  curve_type, curve_types      one curve type / any of several
  parse_ok                     True = only parsed files, False = only unparsed
  quality                      the stored quality label
  search                       comma-separated globs on the whole path
  users, analytes, solvents, techniques, substrates, sample_preps, afm_units
                               any of the listed values of that column
  date_from, date_to           measured_date range, inclusive
  k_min, k_max                 spring constant range (pN/nm), inclusive
  usable                       True = no unusable_reason, False = has one
  unique                       True = canonical copies only, False = redundant copies

The db functions take it as `flt`; the dashboard scope becomes one with
`scope_to_query`.  The module imports neither Qt nor db.
"""
import dataclasses
import inspect
from pathlib import Path

import pytest

from smfs_catalog import db, file_filter, scope
from smfs_catalog.file_filter import FileFilter
from source_inspect import imported_names

FIELDS = (
    "directory", "directories", "curve_type", "curve_types", "parse_ok",
    "quality", "search", "users", "analytes", "solvents", "techniques",
    "substrates", "sample_preps", "afm_units", "date_from", "date_to",
    "k_min", "k_max", "usable", "unique",
)


def _params(fn):
    return list(inspect.signature(fn).parameters)


# ---- FileFilter -------------------------------------------------------------

def test_file_filter_is_a_dataclass_with_exactly_todays_twenty_keywords():
    assert dataclasses.is_dataclass(FileFilter)
    assert {f.name for f in dataclasses.fields(FileFilter)} == set(FIELDS)
    assert len(dataclasses.fields(FileFilter)) == 20


def test_every_field_defaults_to_none():
    flt = FileFilter()
    assert all(getattr(flt, name) is None for name in FIELDS)


def test_file_filter_is_frozen():
    flt = FileFilter(users=["Alice"])
    with pytest.raises(dataclasses.FrozenInstanceError):
        flt.users = ["Bob"]


def test_file_filter_builds_from_keywords_and_compares_by_value():
    assert FileFilter(directory="/d", unique=True) == FileFilter(directory="/d", unique=True)
    assert FileFilter(directory="/d") != FileFilter(directory="/e")
    assert dataclasses.replace(FileFilter(users=["A"]), parse_ok=True).parse_ok is True


def test_file_filter_module_imports_no_qt_and_no_db():
    names = imported_names(file_filter.__file__)
    segments = {seg.lower() for n in names for seg in n.split(".")}
    banned = {s for s in segments
              if s == "db" or s.startswith(("pyqt", "pyside", "qt"))}
    assert not banned, f"file_filter must not import {sorted(banned)}"


# ---- the five db functions' new signatures ----------------------------------

def test_file_filter_clauses_takes_a_filter_and_a_keyword_only_exclude_facet():
    assert _params(db._file_filter_clauses) == ["flt", "exclude_facet"]
    p = inspect.signature(db._file_filter_clauses).parameters["exclude_facet"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is None


def test_list_files_is_db_path_then_flt():
    assert _params(db.list_files) == ["db_path", "flt"]
    sig = inspect.signature(db.list_files).parameters
    assert sig["db_path"].default == db.DEFAULT_DB_PATH
    assert sig["flt"].default is None


def test_get_facet_options_is_db_path_then_flt():
    assert _params(db.get_facet_options) == ["db_path", "flt"]


def test_get_distinct_dates_is_db_path_then_flt():
    assert _params(db.get_distinct_dates) == ["db_path", "flt"]


def test_get_distinct_values_lost_its_five_filter_parameters():
    assert _params(db.get_distinct_values) == ["column", "table", "db_path"]
    sig = inspect.signature(db.get_distinct_values).parameters
    assert sig["table"].default == "files"
    assert sig["db_path"].default == db.DEFAULT_DB_PATH


def test_scope_to_query_returns_a_file_filter():
    out = scope.scope_to_query(scope.new_scope())
    assert isinstance(out, FileFilter)


# ---- behaviour through the new signatures -----------------------------------

@pytest.fixture
def three(catalog):
    """Three rows: a (parsed, usable), b (duplicate of a), c (unparsed)."""
    conn = db.get_connection(catalog)
    try:
        with conn:
            for name, date, parse_ok, sha in (
                    ("a", "2026-01-01", 1, "same"), ("b", "2026-01-02", 1, "same"),
                    ("c", "2026-01-03", 0, "other")):
                db.upsert_file(
                    {"path": f"/data/{name}.ibw", "filename": f"{name}.ibw",
                     "first_seen": "2026-01-01", "last_seen": "2026-01-01",
                     "technique": "SMFS", "measured_date": date,
                     "parse_ok": parse_ok, "content_sha256": sha},
                    catalog, conn=conn)
                conn.execute("UPDATE files SET experimentalist='Alice' WHERE filename=?",
                             (f"{name}.ibw",))
    finally:
        conn.close()
    return catalog


def _names(rows):
    return sorted(r["filename"] for r in rows)


def test_list_files_takes_the_filter_as_its_second_argument(three):
    assert _names(db.list_files(three)) == ["a.ibw", "b.ibw", "c.ibw"]
    assert _names(db.list_files(three, None)) == ["a.ibw", "b.ibw", "c.ibw"]
    assert _names(db.list_files(three, FileFilter(unique=True))) == ["a.ibw", "c.ibw"]
    assert _names(db.list_files(three, FileFilter(users=["Alice"], parse_ok=True,
                                                  unique=True))) == ["a.ibw"]


def test_list_files_flt_alone_scopes_a_directory(three):
    d = str(Path(db.normalize_path("/data/a.ibw")).parent)
    rows = db.list_files(db_path=three, flt=FileFilter(directory=d))
    assert _names(rows) == ["a.ibw", "b.ibw", "c.ibw"]
    other = str(Path(d) / "elsewhere")
    assert db.list_files(db_path=three, flt=FileFilter(directory=other)) == []


def test_facet_options_force_parsed_usable_unique_whatever_the_filter_says(three):
    # b is a duplicate and c is unparsed, so only a is counted, even when the
    # filter asks for the opposite: those three rules are not the caller's.
    wide = FileFilter(parse_ok=False, usable=False, unique=False)
    assert db.get_facet_options(three, wide)["users"] == [("Alice", 1)]
    assert db.get_facet_options(three, FileFilter())["users"] == [("Alice", 1)]


def test_facet_options_cascade_through_the_filter(three):
    facets = db.get_facet_options(three, FileFilter(date_from="2026-02-01"))
    assert facets["users"] == []                       # a is before the range
    facets = db.get_facet_options(three, FileFilter(users=["Alice"]))
    assert facets["users"] == [("Alice", 1)]           # a facet ignores its own selection


def test_distinct_dates_keep_only_parsed_files_whatever_the_filter_says(three):
    expected = [("2026-01-01", 1), ("2026-01-02", 1)]  # c is unparsed
    assert db.get_distinct_dates(three, FileFilter()) == expected
    assert db.get_distinct_dates(three, FileFilter(parse_ok=False)) == expected
    assert db.get_distinct_dates(three, FileFilter(users=["Bob"])) == []


def test_clauses_for_an_empty_filter_are_empty_and_exclude_facet_is_honoured():
    assert db._file_filter_clauses(FileFilter()) == ([], [])
    kept, _ = db._file_filter_clauses(FileFilter(users=["A"]))
    dropped, _ = db._file_filter_clauses(FileFilter(users=["A"]), exclude_facet="users")
    assert kept and not dropped


def test_scope_to_query_matches_what_the_scope_selects(three):
    s = {**scope.new_scope(), "users": ["Alice"]}
    rows = db.list_files(three, scope.scope_to_query(s))
    assert _names(rows) == ["a.ibw"]               # not the duplicate, not the unparsed
