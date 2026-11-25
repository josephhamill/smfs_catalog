# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
What the catalog's file filters select, pinned on a real temporary catalog.

These tests do not care how a filter is passed (today twenty keywords, planned
one `FileFilter`), only which rows come back.  Everything goes through the
small adapters just below, which are the only code that changes when the
signatures do.

The catalog (file letter: owner, analyte, solvent, technique, substrate, prep,
instrument, type, date, k in pN/nm, flags):

  a  /data/s1/a.ibw      Alice titin    PBS   SMFS mica drop-cast AFM-1 force_extension 2026-01-01 0.05 good
  b  /data/s1/b.ibw      Alice titin    PBS   SMFS mica drop-cast AFM-1 force_extension 2026-01-02 0.06 good  (same bytes as a)
  c  /data/s10/c.ibw     Dana  DNA      water UV   gold etching    AFM-2 indentation     2026-02-01 0.10 bad
  d  /data/s2/d.ibw      Bob   collagen PBS   SMFS glass blocking   AFM-1 indentation     2026-02-01 0.02      (unusable)
  e  /data/s2/e.ibw      Carol actin    water UV   mica etching    AFM-2 force_extension 2026-03-01 0.20      (parse_ok = 0)
  f  /data/s2/sub/f.ibw  Alice actin    PBS   UV   gold drop-cast  AFM-2 force_extension 2026-03-05 0.50
"""
import pytest

from smfs_catalog import db
from smfs_catalog.file_filter import FileFilter

# path, owner, analyte, solvent, technique, substrate, prep, instrument, type,
# date, k, quality, parse_ok, unusable_reason, sha
_ROWS = {
    "a": ("/data/s1/a.ibw", "Alice", "titin", "PBS", "SMFS", "mica", "drop-cast",
          "AFM-1", "force_extension", "2026-01-01", 0.05, "good", 1, None, "same"),
    "b": ("/data/s1/b.ibw", "Alice", "titin", "PBS", "SMFS", "mica", "drop-cast",
          "AFM-1", "force_extension", "2026-01-02", 0.06, "good", 1, None, "same"),
    "c": ("/data/s10/c.ibw", "Dana", "DNA", "water", "UV", "gold", "etching",
          "AFM-2", "indentation", "2026-02-01", 0.10, "bad", 1, None, "other"),
    "d": ("/data/s2/d.ibw", "Bob", "collagen", "PBS", "SMFS", "glass", "blocking",
          "AFM-1", "indentation", "2026-02-01", 0.02, None, 1, "nonfinite", "bad"),
    "e": ("/data/s2/e.ibw", "Carol", "actin", "water", "UV", "mica", "etching",
          "AFM-2", "force_extension", "2026-03-01", 0.20, None, 0, None, "unparsed"),
    "f": ("/data/s2/sub/f.ibw", "Alice", "actin", "PBS", "UV", "gold", "drop-cast",
          "AFM-2", "force_extension", "2026-03-05", 0.50, None, 1, None, None),
}


@pytest.fixture
def filled(catalog):
    conn = db.get_connection(catalog)
    try:
        with conn:
            for (path, user, analyte, solvent, technique, substrate, prep, unit,
                 ctype, date, k, quality, parse_ok, reason, sha) in _ROWS.values():
                db.upsert_file(
                    {"path": path, "filename": path.rsplit("/", 1)[-1],
                     "first_seen": "2026-01-01", "last_seen": "2026-01-01",
                     "analyte": analyte, "technique": technique,
                     "substrate": substrate, "sample_prep": prep,
                     "curve_type": ctype, "measured_date": date,
                     "spring_constant_pn_nm": k, "quality": quality,
                     "parse_ok": parse_ok, "unusable_reason": reason,
                     "content_sha256": sha},
                    catalog, conn=conn)
                # owner, solvent and instrument are not upsert_file fields
                conn.execute(
                    "UPDATE files SET experimentalist=?, solvent=?, afm_unit=?"
                    " WHERE path=?",
                    (user, solvent, unit, db.normalize_path(path)))
    finally:
        conn.close()
    return catalog


# ---- adapters: the only code that changes with the signatures ---------------

def _list(path, **flt):
    return db.list_files(path, FileFilter(**flt))


def _facets(path, **flt):
    return db.get_facet_options(path, FileFilter(**flt))


def _dates(path, **flt):
    return db.get_distinct_dates(path, FileFilter(**flt))


def _distinct(path, column, table="files"):
    return db.get_distinct_values(column, table=table, db_path=path)


def _clauses(exclude_facet, **flt):
    return db._file_filter_clauses(FileFilter(**flt), exclude_facet=exclude_facet)


# ---- helpers (independent of the signatures) --------------------------------

def _letters(paths):
    return "".join(sorted(p.replace("\\", "/").rsplit("/", 1)[-1][0] for p in paths))


def _hits(path, **flt):
    return _letters(r["path"] for r in _list(path, **flt))


def _clause_hits(path, exclude_facet=None, **flt):
    """Letters of the rows `_file_filter_clauses` lets through."""
    clauses, params = _clauses(exclude_facet, **flt)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    conn = db.get_connection(path)
    try:
        rows = conn.execute(f"SELECT f.path FROM files f{where}", params).fetchall()
    finally:
        conn.close()
    return _letters(r["path"] for r in rows)


# ---- list_files: every field at least once ----------------------------------

@pytest.mark.parametrize("flt, expected", [
    ({}, "abcdef"),
    ({"directory": "/data/s1"}, "ab"),                 # /data/s10 is not under /data/s1
    ({"directory": "/data/s2"}, "def"),                # a subfolder is under it
    ({"directories": ["/data/s2/sub"]}, "f"),
    # today every directory is its own predicate, all of them required
    ({"directory": "/data/s2", "directories": ["/data/s2/sub"]}, "f"),
    ({"curve_type": "indentation"}, "cd"),
    ({"curve_types": ["indentation"]}, "cd"),
    ({"curve_types": ["indentation", "force_extension"]}, "abcdef"),
    ({"parse_ok": True}, "abcdf"),
    ({"parse_ok": False}, "e"),
    ({"quality": "good"}, "ab"),
    ({"search": "s10,f.ibw"}, "cf"),
    ({"users": ["Alice"]}, "abf"),
    ({"users": ["Dana", "Bob"]}, "cd"),
    ({"analytes": ["actin"]}, "ef"),
    ({"solvents": ["water"]}, "ce"),
    ({"techniques": ["UV"]}, "cef"),
    ({"substrates": ["gold"]}, "cf"),
    ({"sample_preps": ["etching"]}, "ce"),
    ({"afm_units": ["AFM-1"]}, "abd"),
    ({"date_from": "2026-02-01"}, "cdef"),
    ({"date_to": "2026-01-31"}, "ab"),
    ({"date_from": "2026-01-02", "date_to": "2026-02-01"}, "bcd"),
    ({"k_min": 0.10}, "cef"),
    ({"k_max": 0.05}, "ad"),
    ({"k_min": 0.05, "k_max": 0.10}, "abc"),
    ({"usable": True}, "abcef"),
    ({"usable": False}, "d"),
    ({"unique": True}, "acdef"),
    ({"unique": False}, "b"),
])
def test_list_files_selects_the_rows_it_always_did(filled, flt, expected):
    assert _hits(filled, **flt) == expected


@pytest.mark.parametrize("flt, expected", [
    ({"users": ["Alice"], "unique": True, "parse_ok": True}, "af"),
    ({"techniques": ["UV"], "date_from": "2026-03-01", "k_max": 0.3}, "e"),
    ({"users": ["Alice"], "analytes": ["actin"], "substrates": ["gold"]}, "f"),
    ({"users": ["Alice"], "analytes": ["DNA"]}, ""),
    ({"directory": "/data/s2", "usable": True, "parse_ok": True}, "f"),
    ({"afm_units": ["AFM-2"], "solvents": ["PBS"], "curve_types": ["force_extension"]}, "f"),
    ({"quality": "good", "unique": False}, "b"),
])
def test_filters_combine_with_and(filled, flt, expected):
    assert _hits(filled, **flt) == expected


def test_list_files_returns_whole_rows_with_the_derived_columns(filled):
    rows = {r["filename"]: r for r in _list(filled)}
    assert set(rows) == {"a.ibw", "b.ibw", "c.ibw", "d.ibw", "e.ibw", "f.ibw"}
    # b holds the same bytes as a, so it names a as its original; a does not
    assert rows["b.ibw"]["duplicate_of"] == db.normalize_path("/data/s1/a.ibw")
    assert rows["a.ibw"]["duplicate_of"] is None
    assert rows["a.ibw"]["experimentalist"] == "Alice"


# ---- _file_filter_clauses and exclude_facet ---------------------------------

@pytest.mark.parametrize("exclude, flt, expected", [
    (None, {"users": ["Alice"]}, "abf"),
    ("users", {"users": ["Alice"]}, "abcdef"),
    ("users", {"users": ["Alice"], "analytes": ["actin"]}, "ef"),
    ("analytes", {"users": ["Alice"], "analytes": ["actin"]}, "abf"),
    ("curve_types", {"curve_types": ["indentation"], "users": ["Dana"]}, "c"),
    ("techniques", {"techniques": ["UV"], "afm_units": ["AFM-1"]}, "abd"),
    ("afm_units", {"techniques": ["UV"], "afm_units": ["AFM-1"]}, "cef"),
    ("substrates", {"substrates": ["gold"], "parse_ok": True}, "abcdf"),
    ("sample_preps", {"sample_preps": ["etching"], "usable": True}, "abcef"),
    ("solvents", {"solvents": ["water"], "unique": True}, "acdef"),
])
def test_exclude_facet_drops_only_that_facets_own_selection(filled, exclude, flt, expected):
    assert _clause_hits(filled, exclude_facet=exclude, **flt) == expected


def test_the_clauses_with_no_filter_select_everything(filled):
    assert _clause_hits(filled) == "abcdef"


# ---- get_facet_options: eligibility is fixed, other facets cascade ----------

def test_facets_count_only_parsed_usable_unique_files(filled):
    facets = _facets(filled)
    # b (duplicate), d (unusable) and e (unparsed) are never counted
    assert facets["users"] == [("Alice", 2), ("Dana", 1)]
    assert facets["analytes"] == [("DNA", 1), ("actin", 1), ("titin", 1)]
    assert facets["techniques"] == [("SMFS", 1), ("UV", 2)]
    assert facets["curve_types"] == [("force_extension", 2), ("indentation", 1)]
    assert facets["afm_units"] == [("AFM-1", 1), ("AFM-2", 2)]


def test_facets_cascade_by_the_other_dimensions_not_their_own(filled):
    facets = _facets(filled, users=["Alice"], analytes=["actin"])
    assert facets["users"] == [("Alice", 1)]        # own selection ignored; actin -> f
    assert facets["analytes"] == [("actin", 1), ("titin", 1)]   # Alice -> a, f
    assert facets["solvents"] == [("PBS", 1)]


def test_facets_honour_date_range_and_search(filled):
    assert _facets(filled, date_from="2026-02-01")["users"] == [("Alice", 1), ("Dana", 1)]
    assert _facets(filled, date_to="2026-01-31")["users"] == [("Alice", 1)]
    assert _facets(filled, search="s10")["users"] == [("Dana", 1)]


# ---- get_distinct_dates: parsed files only, no other fixed rule -------------

def test_distinct_dates_keep_parsed_files_but_not_the_usable_or_unique_rules(filled):
    # e (parse_ok = 0, 2026-03-01) is out; duplicate b and unusable d are in
    assert _dates(filled) == [
        ("2026-01-01", 1), ("2026-01-02", 1), ("2026-02-01", 2), ("2026-03-05", 1)]


def test_distinct_dates_narrow_by_the_other_dimensions(filled):
    assert _dates(filled, users=["Alice"]) == [
        ("2026-01-01", 1), ("2026-01-02", 1), ("2026-03-05", 1)]
    assert _dates(filled, techniques=["UV"]) == [("2026-02-01", 1), ("2026-03-05", 1)]
    assert _dates(filled, search="s10") == [("2026-02-01", 1)]


# ---- get_distinct_values: no filters ----------------------------------------

def test_distinct_values_are_sorted_and_unfiltered(filled):
    assert _distinct(filled, "experimentalist") == ["Alice", "Bob", "Carol", "Dana"]
    assert _distinct(filled, "technique") == ["SMFS", "UV"]
    assert _distinct(filled, "curve_type") == ["force_extension", "indentation"]


def test_distinct_values_ignore_unknown_columns_and_tables(filled):
    assert _distinct(filled, "path") == []
    assert _distinct(filled, "technique", table="other") == []


# ---- classification_report_rows: one row per file the filter selects --------

def _report(path, **flt):
    return db.classification_report_rows(path, FileFilter(**flt) if flt else None)


def test_classification_report_has_one_row_per_selected_file(filled):
    header, rows = _report(filled, users=["Alice"], unique=True, parse_ok=True)
    assert header[:3] == ["path", "filename", "curve_type"]
    assert all(len(r) == len(header) for r in rows)
    assert _letters(r[0] for r in rows) == "af"
    by_name = {r[1]: r for r in rows}
    assert by_name["a.ibw"][header.index("experimentalist")] == "Alice"
    assert by_name["f.ibw"][header.index("measured_date")] == "2026-03-05"


def test_classification_report_with_no_filter_covers_every_file(filled):
    header, rows = db.classification_report_rows(filled)
    assert _letters(r[0] for r in rows) == "abcdef"
    assert len(header) == len(rows[0])


def test_classification_report_of_an_empty_selection_is_header_only(filled):
    header, rows = _report(filled, users=["Nobody"])
    assert rows == []
    assert header[:2] == ["path", "filename"]
