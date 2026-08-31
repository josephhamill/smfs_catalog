# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""The dashboard facets and final file cohort must describe the same scope."""

from smfs_catalog import db
from smfs_catalog.scope import new_scope, scope_to_query


def test_scope_translation_preserves_every_dashboard_filter():
    scope = new_scope()
    scope.update({
        "users": ["Alice", "Dana"],
        "analytes": ["titin"],
        "solvents": ["PBS"],
        "afm_units": ["AFM-1"],
        "curve_types": ["force_extension"],
        "date_from": "2026-01-01",
        "date_to": "2026-01-31",
        "search": "Image0001",
    })

    assert scope_to_query(scope) == {
        "users": ["Alice", "Dana"],
        "analytes": ["titin"],
        "solvents": ["PBS"],
        "afm_units": ["AFM-1"],
        "curve_types": ["force_extension"],
        "date_from": "2026-01-01",
        "date_to": "2026-01-31",
        "search": "Image0001",
        "parse_ok": True,
        "usable": True,
        "unique": True,
    }


def test_set_aside_flags_invert_only_their_own_eligibility_rule():
    assert scope_to_query({**new_scope(), "only_unusable": True}) == {
        "parse_ok": True, "usable": False, "unique": True,
    }
    assert scope_to_query({**new_scope(), "only_duplicates": True}) == {
        "parse_ok": True, "usable": True, "unique": False,
    }


def _catalog(tmp_path):
    path = str(tmp_path / "scope.sqlite")
    db.initialise(path)
    rows = [
        # path, user, analyte, solvent, technique, substrate, prep,
        # instrument, type, date, parse, reason, sha
        ("/data/a.ibw", "Alice", "titin", "PBS", "SMFS", "mica", "drop-cast",
         "AFM-1", "force_extension", "2026-01-01", 1, None, "same"),
        ("/copy/a.ibw", "Alice", "titin", "PBS", "SMFS", "mica", "drop-cast",
         "AFM-1", "force_extension", "2026-01-01", 1, None, "same"),
        ("/data/b.ibw", "Dana", "DNA", "water", "UV", "gold", "etching",
         "AFM-2", "indentation", "2026-02-01", 1, None, "other"),
        ("/data/unusable.ibw", "Bob", "collagen", "PBS", "SMFS", "glass", "blocking",
         "AFM-1", "indentation", "2026-02-01", 1, "nonfinite", "bad"),
        ("/data/unparsed.ibw", "Carol", "actin", "water", "UV", "mica", "etching",
         "AFM-2", "force_extension", "2026-03-01", 0, None, "unparsed"),
    ]
    conn = db.get_connection(path)
    with conn:
        conn.executemany(
            """
            INSERT INTO files (
                path, filename, experimentalist, analyte, solvent, technique,
                substrate, sample_prep, afm_unit,
                curve_type, measured_date, parse_ok, unusable_reason,
                content_sha256, first_seen, last_seen
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '2026-01-01', '2026-01-01')
            """,
            [(row[0], row[0].rsplit("/", 1)[-1], *row[1:]) for row in rows],
        )
    conn.close()
    return path


def test_scope_query_combines_filters_and_enforces_normal_eligibility(tmp_path):
    path = _catalog(tmp_path)
    scope = new_scope()
    scope["users"] = ["Alice", "Dana"]
    scope["curve_types"] = ["indentation"]

    rows = db.list_files(db_path=path, **scope_to_query(scope))

    assert [row["path"] for row in rows] == ["/data/b.ibw"]


def test_facets_count_the_same_eligible_cohort_as_list_files(tmp_path):
    path = _catalog(tmp_path)

    facets = db.get_facet_options(path)
    rows = db.list_files(db_path=path, **scope_to_query(new_scope()))

    assert [row["path"] for row in rows] == ["/data/a.ibw", "/data/b.ibw"]
    assert facets["users"] == [("Alice", 1), ("Dana", 1)]
    assert facets["analytes"] == [("DNA", 1), ("titin", 1)]


def test_technique_substrate_and_prep_narrow_the_cohort(tmp_path):
    path = _catalog(tmp_path)

    for key, value, expected in (
        ("techniques",   "UV",        ["/data/b.ibw"]),
        ("substrates",   "mica",      ["/data/a.ibw"]),
        ("sample_preps", "drop-cast", ["/data/a.ibw"]),
    ):
        scope = {**new_scope(), key: [value]}
        rows = db.list_files(db_path=path, **scope_to_query(scope))
        assert [row["path"] for row in rows] == expected, key


def test_the_new_dimensions_are_offered_as_cascading_facets(tmp_path):
    path = _catalog(tmp_path)

    facets = db.get_facet_options(path)
    assert facets["techniques"] == [("SMFS", 1), ("UV", 1)]
    assert facets["substrates"] == [("gold", 1), ("mica", 1)]
    assert facets["sample_preps"] == [("drop-cast", 1), ("etching", 1)]

    # Narrowed by another dimension, the way every other facet is.
    narrowed = db.get_facet_options(path, techniques=["UV"])
    assert narrowed["substrates"] == [("gold", 1)]


def test_a_glob_in_the_search_reaches_the_database(tmp_path):
    path = _catalog(tmp_path)
    scope = {**new_scope(), "search": "[ab].ibw"}

    rows = db.list_files(db_path=path, **scope_to_query(scope))
    assert [row["path"] for row in rows] == ["/data/a.ibw", "/data/b.ibw"]

    # A pattern narrows the cohort and cannot widen it past eligibility:
    # /copy/a.ibw is the redundant copy of /data/a.ibw.
    scope["search"] = "/copy/*"
    assert db.list_files(db_path=path, **scope_to_query(scope)) == []

    scope["search"] = "b.ibw,nothing_at_all"
    rows = db.list_files(db_path=path, **scope_to_query(scope))
    assert [row["path"] for row in rows] == ["/data/b.ibw"]


def test_each_facet_is_cascaded_by_the_other_dimensions_not_itself(tmp_path):
    path = _catalog(tmp_path)

    facets = db.get_facet_options(
        path,
        users=["Alice"],
        analytes=["DNA"],
    )

    # The user facet ignores Alice but is narrowed by DNA; the analyte facet
    # ignores DNA but is narrowed by Alice. This keeps alternatives visible.
    assert facets["users"] == [("Dana", 1)]
    assert facets["analytes"] == [("titin", 1)]
