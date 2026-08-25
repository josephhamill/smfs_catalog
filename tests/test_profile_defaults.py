# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Guard test: ONE profile, ONE seed, ONE inheritance rule (#141).

A profile holds two kinds of setting — the analysis parameters and the 2DH view
settings — and until this change only the first kind was seeded into the
DEFAULT_EXPERIMENTALIST row.  The second kind had no Default value to inherit, so
a never-tuned experimentalist got the lab's working numbers for their analysis
and raw code constants for their 2DH. That is exactly what load_analysis_params's
docstring says must not happen.

The failure was invisible because each half worked: read a parameter, get a
sensible number; read a grid setting, get a sensible number.  Only the SOURCE
differed, and nothing on screen names the source.  So the checks below are about
where a value comes from, not whether it looks reasonable.

What is deliberately NOT tested: any way to edit the Default row.  There isn't
one and there should not be — it is a seed, not a control surface.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smfs_catalog import db as _db                      # noqa: E402
from smfs_catalog import event_processor as _ep         # noqa: E402


def _fresh_db() -> str:
    path = os.path.join(tempfile.mkdtemp(prefix="smfs_defaults_"), "t.db")
    _db.initialise(path)
    return path


def _default_row(db_path: str) -> dict:
    return _db.get_experimentalist_profile(_db.DEFAULT_EXPERIMENTALIST, db_path) or {}


# ── (a) A fresh database's Default row holds EVERY default ───────────────────

def test_fresh_db_seeds_every_default_into_the_default_row():
    """initialise() is the only thing that has to run for the lab defaults to
    exist.  Nobody opens a window, nobody presses a button."""
    db_path = _fresh_db()
    row = _default_row(db_path)
    missing = sorted(k for k in _db.profile_defaults() if k not in row)
    assert not missing, f"Default row is missing seeded keys: {missing}"
    for key, want in _db.profile_defaults().items():
        got = row[key]
        if isinstance(want, str):
            assert got == want, f"{key}: seeded {got!r}, expected {want!r}"
        else:
            assert abs(float(got) - float(want)) < 1e-9, f"{key}: {got} != {want}"


def test_the_seed_covers_both_halves_of_a_profile():
    """The regression itself: view settings must be in the seed, not only the
    analysis parameters.  Before #141 profile_defaults() was PARAM_DEFAULTS."""
    seeded = _db.profile_defaults()
    assert set(_db.PARAM_DEFAULTS) <= set(seeded)
    assert set(_db.view_defaults()) <= set(seeded)
    assert _db.view_defaults(), "view defaults must not be empty"
    # And the two halves must not overlap — a key seeded twice has two owners.
    assert not (set(_db.PARAM_DEFAULTS) & set(_db.view_defaults()))


# ── (b) Every grid key a window declares is seeded ───────────────────────────

def test_every_declared_grid_key_has_a_seeded_default():
    """THE drift guard.

    _profile_spec() is where each 2DH window declares what it persists.  Adding a
    key there without adding it to view_defaults() recreates the exact bug this
    change fixed — a setting with no lab default, silently falling through to a
    code constant for everyone who has not tuned it.  Neither _profile_spec is
    an instance method in practice (both return a literal list), so they can be
    called off the class without building a window.
    """
    from smfs_catalog.physical_2dh_window import Physical2DHWindow
    from smfs_catalog.normalized_2dh_window import Normalized2DHWindow

    seeded = set(_db.view_defaults())
    for cls in (Physical2DHWindow, Normalized2DHWindow):
        declared = {key for _attr, key, _cast, _dflt in cls._profile_spec(None)}
        missing = sorted(declared - seeded)
        assert not missing, (
            f"{cls.__name__} persists {missing} but db.view_defaults() does not "
            f"seed them, so a never-tuned experimentalist inherits nothing for "
            f"them. Add them to view_defaults()."
        )


def test_declared_defaults_match_the_seeded_ones():
    """The constant a window falls back to and the value seeded into Default must
    be the same number.  Two answers to 'what is the default' is how the two 2DH
    windows came to disagree about the align segment in the first place."""
    from smfs_catalog.physical_2dh_window import Physical2DHWindow
    from smfs_catalog.normalized_2dh_window import Normalized2DHWindow

    seeded = _db.view_defaults()
    for cls in (Physical2DHWindow, Normalized2DHWindow):
        for _attr, key, caster, dflt in cls._profile_spec(None):
            assert caster(seeded[key]) == caster(dflt), (
                f"{cls.__name__} falls back to {dflt!r} for {key} but Default is "
                f"seeded with {seeded[key]!r}"
            )


def test_both_2dh_windows_agree_on_the_align_segment_default():
    """They did not, until 2026-08-06: physical said "first", normalized said
    "last".  One decision, two copies — open both on an untuned profile and they
    built from different segments with nothing on screen saying so."""
    from smfs_catalog import physical_2dh_window as phys
    from smfs_catalog import normalized_2dh_window as norm

    assert phys._DEFAULT_ALIGN_SEG == norm._DEFAULT_ALIGN_SEG == _ep.ALIGN_SEG_DEFAULT
    # Ultimate — the tether/final rupture, matching the queue's own default
    # (roi_pipeline.read_segment_select returns "ultimate" for an unset toggle).
    assert _ep.ALIGN_SEG_DEFAULT == "last"


def test_physical_2dh_defaults_to_a_fit_free_alignment():
    """rupture, not onset (#141), and the property that makes it safe: it needs
    no WLC fit, so a curve whose fit failed is still registered."""
    assert _ep.PHYS_ALIGN_DEFAULT == "rupture"
    assert _ep.PHYS_ALIGN_DEFAULT in _ep.PHYS_ALIGN_ANCHORS
    from smfs_catalog.physical_2dh_window import Physical2DHWindow
    assert _ep.PHYS_ALIGN_DEFAULT not in Physical2DHWindow._FIT_DEPENDENT_ALIGN_MODES


# ── (c) View settings are NOT analysis parameters ────────────────────────────

def test_view_settings_are_not_param_keys():
    """PARAM_KEYS means "the settings that define an analysis result" — it is
    what get_param_set stamps onto exported results and what test_numeric_ui
    demands a declared unit for.  A grid range changes what you LOOK at, not what
    was computed; putting it in PARAM_KEYS would write a view setting into the
    provenance of every fit."""
    assert not (set(_db.view_defaults()) & _db.PARAM_KEYS)
    assert set(_db.PARAM_DEFAULTS) == set(_db.PARAM_KEYS)


def test_analysis_snapshot_returns_only_analysis_parameters():
    """Seeding view settings into the same row must not leak them into the
    parameter set the pipeline reads — that set feeds event_map_params_json."""
    db_path = _fresh_db()
    got = set(_db.load_analysis_params(db_path))
    assert got == set(_db.PARAM_KEYS)


# ── (d) Inheritance: owner -> Default -> constant ────────────────────────────

def test_a_never_tuned_experimentalist_materializes_the_default_row_once():
    db_path = _fresh_db()
    _db.merge_experimentalist_profile(
        _db.DEFAULT_EXPERIMENTALIST, {"spectral_cutoff_hz": 1234.0}, db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO files (path, filename, experimentalist, parse_ok, "
        "                   first_seen, last_seen) "
        "VALUES ('/x/a.ibw', 'a.ibw', 'Newcomer', 1, "
        "        '2026-08-06T00:00:00', '2026-08-06T00:00:00')")
    fid = conn.execute("SELECT id FROM files").fetchone()[0]
    conn.execute("INSERT INTO analysis_queue (file_id, status, enqueued_at) "
                 "VALUES (?, 'pending', '2026-08-06T00:00:00')", (fid,))
    conn.commit()
    conn.close()

    assert _db.active_param_owner(db_path) == "Newcomer"
    assert _db.get_experimentalist_profile("Newcomer", db_path) in (None, {})
    assert abs(_db.load_analysis_params(db_path).spectral_cutoff_hz - 1234.0) < 1e-9

    stored = _db.get_experimentalist_profile("Newcomer", db_path) or {}
    assert set(_db.PARAM_KEYS) <= set(stored), (
        "loading the effective set must durably fill every missing analysis key")

    # Default is a seed, not a live parent. Once Newcomer's value has been
    # brought into their profile, changing Default cannot change their science.
    _db.merge_experimentalist_profile(
        _db.DEFAULT_EXPERIMENTALIST, {"spectral_cutoff_hz": 4321.0}, db_path)
    assert abs(_db.load_analysis_params(db_path).spectral_cutoff_hz - 1234.0) < 1e-9


def test_first_param_edit_completes_the_owner_profile_atomically():
    """The user need not load a tuning window before making the first edit."""
    db_path = _fresh_db()
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO files (path, filename, experimentalist, parse_ok, "
        "                   first_seen, last_seen) "
        "VALUES ('/x/b.ibw', 'b.ibw', 'Newcomer', 1, "
        "        '2026-08-06T00:00:00', '2026-08-06T00:00:00')")
    fid = conn.execute("SELECT id FROM files").fetchone()[0]
    conn.execute("INSERT INTO analysis_queue (file_id, status, enqueued_at) "
                 "VALUES (?, 'pending', '2026-08-06T00:00:00')", (fid,))
    conn.commit()
    conn.close()

    _db.update_analysis_param("spectral_cutoff_hz", 1777.0, db_path)
    stored = _db.get_experimentalist_profile("Newcomer", db_path) or {}
    assert set(_db.PARAM_KEYS) <= set(stored)
    assert stored["spectral_cutoff_hz"] == 1777.0


def test_single_param_read_also_materializes_the_complete_set():
    db_path = _fresh_db()
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO files (path, filename, experimentalist, parse_ok, "
        "                   first_seen, last_seen) "
        "VALUES ('/x/c.ibw', 'c.ibw', 'Newcomer', 1, "
        "        '2026-08-06T00:00:00', '2026-08-06T00:00:00')")
    fid = conn.execute("SELECT id FROM files").fetchone()[0]
    conn.execute("INSERT INTO analysis_queue (file_id, status, enqueued_at) "
                 "VALUES (?, 'pending', '2026-08-06T00:00:00')", (fid,))
    conn.commit()
    conn.close()

    assert _db.load_analysis_params(db_path).spectral_cutoff_hz > 0
    stored = _db.get_experimentalist_profile("Newcomer", db_path) or {}
    assert set(_db.PARAM_KEYS) <= set(stored)


def test_grid_params_fall_back_to_default_not_to_the_code_constant():
    """The bug this change fixes, checked through the real read path.

    _load_grid_params is exercised directly rather than by building a window: a
    2DH needs a QApplication, a populated catalog and real curve files, and a
    guard that expensive is a guard that gets skipped.
    """
    from smfs_catalog.base_2dh_window import _TwoDHWindowBase
    from smfs_catalog.physical_2dh_window import Physical2DHWindow

    db_path = _fresh_db()
    # The lab tunes Default away from the code constant.
    _db.merge_experimentalist_profile(
        _db.DEFAULT_EXPERIMENTALIST, {"phys_f_star": 137.0}, db_path)

    class _Probe:
        _db_path = db_path
        _profile_key = "Newcomer"          # no row of their own
        _profile_spec = Physical2DHWindow._profile_spec
        _load_grid_params = _TwoDHWindowBase._load_grid_params

    probe = _Probe()
    probe._load_grid_params()
    assert probe._f_star == 137.0, (
        "a never-tuned experimentalist read the code constant instead of the "
        "lab's Default value — the #141 regression")

    # Their OWN value still wins over Default's.
    _db.merge_experimentalist_profile("Newcomer", {"phys_f_star": 42.0}, db_path)
    probe._load_grid_params()
    assert probe._f_star == 42.0

    # And Default reading its own row must not consult itself twice and stall on
    # a missing key — it falls straight to the declared constant.
    probe._profile_key = _db.DEFAULT_EXPERIMENTALIST
    probe._load_grid_params()
    assert probe._f_star == 137.0


# ── (e) The seed never overwrites a value someone chose ──────────────────────

def test_reinitialise_backfills_missing_keys_but_never_clobbers():
    """json_patch(new, existing) — existing wins.  This is what makes changing a
    code default safe on a live database: it reaches a fresh DB and any key added
    since, and touches nothing anyone has set."""
    db_path = _fresh_db()
    _db.merge_experimentalist_profile(
        _db.DEFAULT_EXPERIMENTALIST, {"spectral_cutoff_hz": 999.0}, db_path)
    # Drop a seeded key to stand in for "added to the code since this DB was made".
    conn = sqlite3.connect(db_path)
    row = json.loads(conn.execute(
        "SELECT params_json FROM experimentalist_profiles WHERE experimentalist = ?",
        (_db.DEFAULT_EXPERIMENTALIST,)).fetchone()[0])
    row.pop("phys_align_mode")
    conn.execute(
        "UPDATE experimentalist_profiles SET params_json = ? WHERE experimentalist = ?",
        (json.dumps(row), _db.DEFAULT_EXPERIMENTALIST))
    conn.commit()
    conn.close()

    _db.initialise(db_path)

    after = _default_row(db_path)
    assert after["spectral_cutoff_hz"] == 999.0, "initialise() overwrote a chosen value"
    assert after["phys_align_mode"] == _ep.PHYS_ALIGN_DEFAULT, "missing key not backfilled"
