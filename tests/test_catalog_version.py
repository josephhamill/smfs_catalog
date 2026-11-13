# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
The catalog's version.

A catalog newer than the app is refused and left byte for byte as it was; an
older one is upgraded only after its backup is written; and the layout cannot
change without the version number changing with it.

The test on the catalog v1.3.0 built needs
~/.local/share/smfs_catalog/test_v1.3.0.db, made by
`python scripts/build_catalog.py v1.3.0`, and skips without it.

Run with the smfs-catalog-dev env, from the repo root:
    python -m pytest tests/test_catalog_version.py -q
"""
import ast
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from smfs_catalog import curve_loader, db, scanner          # noqa: E402
import run_dashboard as rd                                   # noqa: E402

BUILT_V130 = os.path.expanduser("~/.local/share/smfs_catalog/test_v1.3.0.db")

# The layout each version stands for, as a fingerprint of every table and its
# columns.  When the layout changes, the layout test fails: raise
# db.CATALOG_VERSION, add the step that carries the version before it, and
# record the new fingerprint here.
LAYOUT = {1: "1b441cda936a3478ae5b30846ae353daef1a2ee7853a6a7165c0e376d6951de4"}


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _layout(path):
    """Every table and its columns."""
    con = sqlite3.connect(path)
    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        return {t: sorted(r[1] for r in con.execute(f"PRAGMA table_info({t})"))
                for t in tables}
    finally:
        con.close()


def _fingerprint(layout):
    return hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()


def _version(path):
    con = sqlite3.connect(path)
    try:
        return con.execute("PRAGMA user_version").fetchone()[0]
    finally:
        con.close()


def _set_version(path, n):
    con = sqlite3.connect(path)
    con.execute(f"PRAGMA user_version = {n}")
    con.close()


def _new(tmp_path, name="catalog.db"):
    path = str(tmp_path / name)
    db.initialise(path)
    return path


def _unnumbered_v130_layout(tmp_path):
    """A catalog in v1.3.0's layout with no version number.

    It has watched_directories and files.directory_id, lacks substrate and
    sample_prep, and holds one curve v1.3.0 typed force_extension.
    """
    path = _new(tmp_path, "older.db")
    con = sqlite3.connect(path)
    with con:
        con.execute("ALTER TABLE files DROP COLUMN substrate")
        con.execute("ALTER TABLE files DROP COLUMN sample_prep")
        con.execute("CREATE TABLE watched_directories (id INTEGER PRIMARY KEY, path TEXT)")
        con.execute("ALTER TABLE files ADD COLUMN directory_id INTEGER")
        con.execute("CREATE INDEX idx_files_directory_id ON files(directory_id)")
        con.execute("INSERT INTO files (path, filename, curve_type, first_seen, last_seen) "
                    "VALUES ('/d/a.ibw', 'a.ibw', 'force_extension', 'x', 'x')")
    con.close()
    _set_version(path, 0)
    return path


# ── The number ────────────────────────────────────────────────────────────────

def test_a_new_catalog_carries_this_version(tmp_path):
    assert _version(_new(tmp_path)) == db.CATALOG_VERSION


def test_no_file_and_a_file_with_no_tables_are_no_catalog_yet(tmp_path):
    assert db.catalog_version(str(tmp_path / "absent.db")) is None
    empty = tmp_path / "empty.db"
    sqlite3.connect(str(empty)).close()
    assert db.catalog_version(str(empty)) is None


def test_a_catalog_made_before_the_number_reads_0(tmp_path):
    assert db.catalog_version(_unnumbered_v130_layout(tmp_path)) == 0


def test_a_file_that_is_not_a_database_raises(tmp_path):
    junk = tmp_path / "junk.db"
    junk.write_bytes(b"not a database " * 100)
    with pytest.raises(sqlite3.DatabaseError):
        db.catalog_version(str(junk))


def _other_database(tmp_path):
    """Another program's SQLite database, carrying a version number of its own."""
    path = tmp_path / "other.db"
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE t (x)")
    con.execute("PRAGMA user_version = 1")
    con.close()
    return str(path)


def test_a_database_with_no_files_table_is_not_a_catalog(tmp_path):
    with pytest.raises(db.NotACatalog):
        db.catalog_version(_other_database(tmp_path))


def test_a_negative_version_is_not_a_catalog(tmp_path):
    path = _new(tmp_path)
    _set_version(path, -1)
    with pytest.raises(db.NotACatalog):
        db.catalog_version(path)


def test_every_version_has_its_step():
    assert sorted(db._UPGRADES) == list(range(1, db.CATALOG_VERSION + 1))


def test_the_layout_cannot_change_without_the_version(tmp_path):
    layout = _layout(_new(tmp_path))
    assert LAYOUT.get(db.CATALOG_VERSION) == _fingerprint(layout), (
        "the catalog layout is not the one recorded for version %d.  Raise "
        "db.CATALOG_VERSION, add the step that carries the version before it, "
        "and record this fingerprint: %s\nlayout:\n%s"
        % (db.CATALOG_VERSION, _fingerprint(layout),
           json.dumps(layout, indent=1, sort_keys=True)))


# ── Newer, current, older ─────────────────────────────────────────────────────

def test_a_newer_catalog_is_refused_and_left_byte_for_byte(tmp_path):
    path = _new(tmp_path)
    _set_version(path, db.CATALOG_VERSION + 1)
    before = _digest(path)
    with pytest.raises(db.CatalogTooNew) as exc:
        db.initialise(path)
    assert _digest(path) == before
    assert path in str(exc.value)
    assert "version %d" % (db.CATALOG_VERSION + 1) in str(exc.value)


def test_a_current_catalog_opens_with_its_layout_unchanged(tmp_path):
    path = _new(tmp_path)
    before = _layout(path)
    db.initialise(path)
    assert _layout(path) == before
    assert _version(path) == db.CATALOG_VERSION


def test_an_unnumbered_catalog_gets_every_step(tmp_path):
    fresh = _layout(_new(tmp_path, "fresh.db"))
    path = _unnumbered_v130_layout(tmp_path)
    db.initialise(path)
    assert _layout(path) == fresh
    assert _version(path) == db.CATALOG_VERSION


def test_an_interrupted_step_leaves_the_catalog_as_it_was(tmp_path, monkeypatch):
    """A step and its number are one transaction: a failure at its very end
    undoes the schema changes it already made."""
    path = _unnumbered_v130_layout(tmp_path)
    before = _layout(path)
    real = db._UPGRADES[1]

    def fails_at_the_end(conn):
        real(conn)
        raise RuntimeError("interrupted")

    monkeypatch.setitem(db._UPGRADES, 1, fails_at_the_end)
    with pytest.raises(RuntimeError):
        db.initialise(path)
    assert _layout(path) == before
    assert _version(path) == 0


# ── The backup ────────────────────────────────────────────────────────────────

def test_the_backup_is_the_catalog_and_replaces_the_one_before(tmp_path):
    path = _unnumbered_v130_layout(tmp_path)
    backup = db.backup_catalog(path)
    assert backup == db.backup_path(path)
    assert _layout(backup) == _layout(path)
    original, copy = sqlite3.connect(path), sqlite3.connect(backup)
    try:
        assert copy.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert copy.execute("PRAGMA user_version").fetchone()[0] == 0
        assert (copy.execute("SELECT * FROM files").fetchall()
                == original.execute("SELECT * FROM files").fetchall())
    finally:
        original.close()
        copy.close()

    db.backup_catalog(path)
    kept = sorted(p.name for p in tmp_path.iterdir()
                  if p.name.startswith("older.db.backup")
                  and not p.name.endswith(("-wal", "-shm")))
    assert kept == ["older.db.backup"], "one backup, and no partial one left"


def test_the_backup_is_one_self_contained_file(tmp_path):
    """No write-ahead log of its own, and none left from the backup it replaces."""
    path = _unnumbered_v130_layout(tmp_path)
    backup = db.backup_path(path)
    Path(backup + "-wal").write_bytes(b"a log left beside an older backup")
    db.backup_catalog(path)
    assert not os.path.exists(backup + "-wal")
    con = sqlite3.connect(backup)
    try:
        assert con.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    finally:
        con.close()


def test_the_backup_reports_its_progress_step_by_step(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "BACKUP_PAGES_PER_STEP", 1)
    path = _unnumbered_v130_layout(tmp_path)
    seen = []
    db.backup_catalog(path, progress=lambda done, total: seen.append((done, total)))
    assert len(seen) > 1
    assert [done for done, _ in seen] == sorted(done for done, _ in seen)
    assert seen[-1][0] == seen[-1][1]


def test_a_cancelled_backup_keeps_the_last_backup_and_leaves_no_partial(tmp_path,
                                                                       monkeypatch):
    monkeypatch.setattr(db, "BACKUP_PAGES_PER_STEP", 1)
    path = _unnumbered_v130_layout(tmp_path)
    backup = db.backup_catalog(path)
    last_backup, catalog = _digest(backup), _digest(path)
    with pytest.raises(db.BackupCancelled):
        db.backup_catalog(path, progress=lambda done, total: done < 2)
    assert _digest(backup) == last_backup
    assert _digest(path) == catalog
    assert not [p for p in tmp_path.iterdir() if ".partial" in p.name]


# ── At start-up: run_dashboard.open_catalog ───────────────────────────────────

class _Dialogs:
    """Stands in for the two dialogs, answering the upgrade question with `answer`."""

    def __init__(self, answer):
        self.answer = answer
        self.asked = []
        self.refused = []

    def confirm(self, text):
        self.asked.append(text)
        return self.answer

    def refuse(self, text):
        self.refused.append(text)


def test_quit_leaves_an_older_catalog_unchanged_and_writes_no_backup(tmp_path):
    path = _unnumbered_v130_layout(tmp_path)
    before = _digest(path)
    dialogs = _Dialogs(answer=False)
    assert rd.open_catalog(path, dialogs.confirm, dialogs.refuse) is False
    assert len(dialogs.asked) == 1
    assert _digest(path) == before
    assert not os.path.exists(db.backup_path(path))


def test_upgrade_writes_the_backup_then_upgrades(tmp_path):
    path = _unnumbered_v130_layout(tmp_path)
    old_layout = _layout(path)
    dialogs = _Dialogs(answer=True)
    assert rd.open_catalog(path, dialogs.confirm, dialogs.refuse) is True

    backup = db.backup_path(path)
    text = dialogs.asked[0]
    assert path in text and backup in text
    assert "1 curve carries a curve type from the older version" in text
    assert _layout(backup) == old_layout
    assert _version(backup) == 0
    assert _version(path) == db.CATALOG_VERSION


def test_cancelling_the_backup_leaves_the_catalog_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "BACKUP_PAGES_PER_STEP", 1)
    path = _unnumbered_v130_layout(tmp_path)
    before = _digest(path)
    dialogs = _Dialogs(answer=True)
    assert rd.open_catalog(path, dialogs.confirm, dialogs.refuse,
                           progress=lambda done, total: False) is False
    assert _digest(path) == before
    assert _version(path) == 0
    assert not os.path.exists(db.backup_path(path))


def test_the_progress_window_follows_the_copy_and_cancels(tmp_path):
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
    bar = rd._BackupProgress(str(tmp_path / "smfs_catalog.db"))
    assert bar(1, 4) is True
    assert bar.box.windowTitle() == "Backing up catalog"
    assert bar.box.labelText() == "Copying smfs_catalog.db to smfs_catalog.db.backup"
    assert bar.box.value() == 25
    bar.box.cancel()
    assert bar(2, 4) is False
    bar.close()


def test_a_stray_enter_or_escape_presses_quit():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
    box, upgrade = rd._upgrade_box("upgrade?")
    assert box.defaultButton().text() == "Quit"
    assert box.escapeButton().text() == "Quit"
    assert upgrade.text() == "Upgrade"


def test_a_newer_catalog_is_refused_at_start_up_without_asking(tmp_path):
    path = _new(tmp_path)
    _set_version(path, db.CATALOG_VERSION + 1)
    before = _digest(path)
    dialogs = _Dialogs(answer=True)
    assert rd.open_catalog(path, dialogs.confirm, dialogs.refuse) is False
    assert dialogs.asked == []
    assert len(dialogs.refused) == 1
    assert _digest(path) == before


def test_no_room_for_the_backup_refuses_unchanged(tmp_path, monkeypatch):
    path = _unnumbered_v130_layout(tmp_path)
    before = _digest(path)
    monkeypatch.setattr(rd.shutil, "disk_usage", lambda _p: types.SimpleNamespace(free=0))
    dialogs = _Dialogs(answer=True)
    assert rd.open_catalog(path, dialogs.confirm, dialogs.refuse) is False
    assert dialogs.asked == []
    assert "free" in dialogs.refused[0]
    assert _digest(path) == before


def test_a_file_that_is_not_a_catalog_is_refused_at_start_up(tmp_path):
    junk = tmp_path / "junk.db"
    junk.write_bytes(b"not a database " * 100)
    before = _digest(junk)
    dialogs = _Dialogs(answer=True)
    assert rd.open_catalog(str(junk), dialogs.confirm, dialogs.refuse) is False
    assert len(dialogs.refused) == 1
    assert _digest(junk) == before


def test_another_programs_database_is_refused_at_start_up(tmp_path):
    other = _other_database(tmp_path)
    before = _digest(other)
    dialogs = _Dialogs(answer=True)
    assert rd.open_catalog(other, dialogs.confirm, dialogs.refuse) is False
    assert dialogs.asked == []
    assert len(dialogs.refused) == 1
    assert _digest(other) == before


def test_a_catalog_upgraded_while_the_dialog_was_open_keeps_its_backup(tmp_path):
    path = _unnumbered_v130_layout(tmp_path)
    db.backup_catalog(path)                 # the other window's backup of the original
    original = _digest(db.backup_path(path))

    def another_window_upgrades_first(_text):
        db.initialise(path)
        return True

    assert rd.open_catalog(path, another_window_upgrades_first, lambda _t: None) is True
    assert _digest(db.backup_path(path)) == original
    assert _version(path) == db.CATALOG_VERSION


def test_a_new_or_current_catalog_opens_without_a_dialog(tmp_path):
    path = str(tmp_path / "new.db")
    dialogs = _Dialogs(answer=False)
    assert rd.open_catalog(path, dialogs.confirm, dialogs.refuse) is True
    assert rd.open_catalog(path, dialogs.confirm, dialogs.refuse) is True
    assert dialogs.asked == []
    assert dialogs.refused == []


def test_a_frozen_build_takes_the_same_path(tmp_path):
    path = _new(tmp_path)
    _set_version(path, db.CATALOG_VERSION + 1)
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(ROOT)!r})
        sys.frozen = True
        sys._MEIPASS = "/tmp/nope"
        import run_dashboard as rd
        refused = []
        ok = rd.open_catalog({path!r}, lambda text: True, refused.append)
        print(ok, len(refused))
    """)
    out = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "False 1"


# ── Curve types an older version wrote ────────────────────────────────────────

def test_curve_types_lists_every_type_the_classifier_returns():
    tree = ast.parse(Path(curve_loader.__file__).read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_modality")
    returned = {c.value for node in ast.walk(fn) if isinstance(node, ast.Return)
                for c in ast.walk(node.value)
                if isinstance(c, ast.Constant) and isinstance(c.value, str)}
    assert returned == set(curve_loader.CURVE_TYPES)


def test_fill_in_visits_a_curve_typed_by_an_older_version(tmp_path):
    path = _new(tmp_path)
    con = sqlite3.connect(path)
    with con:
        con.executemany(
            "INSERT INTO files (path, filename, curve_type, content_sha256, "
            "first_seen, last_seen) VALUES (?, ?, ?, 'x', 'x', 'x')",
            [(str(tmp_path / "old.ibw"), "old.ibw", "force_extension"),
             (str(tmp_path / "today.ibw"), "today.ibw", "continuous_stretch")])
    con.close()
    assert scanner.count_outdated_curve_types(path) == 1
    assert scanner.count_needing_recheck(path) == 1
    # Neither file exists, so the visit reads nothing and writes nothing.
    summary = scanner.requalify_catalog(path, only_missing=True)
    assert summary["seen"] == 1
    assert summary["unreadable"] == 1


# ── The catalog v1.3.0 built ──────────────────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(BUILT_V130),
                    reason="needs test_v1.3.0.db from scripts/build_catalog.py")
def test_the_v130_built_catalog_upgrades_with_its_bounds_kept(tmp_path):
    path = str(tmp_path / "test_v1.3.0.db")
    shutil.copy(BUILT_V130, path)
    fresh = _layout(_new(tmp_path, "fresh.db"))
    bounds_sql = ("SELECT COUNT(*) FROM thresholds "
                  "WHERE lower_bound IS NOT NULL OR upper_bound IS NOT NULL")
    con = sqlite3.connect(path)
    bounds = con.execute(bounds_sql).fetchone()[0]
    n_files = con.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    con.close()
    assert db.catalog_version(path) == 0
    assert bounds > 0
    # v1.3.0 typed every curve it could analyse force_extension.
    assert scanner.count_outdated_curve_types(path) == n_files

    dialogs = _Dialogs(answer=True)
    assert rd.open_catalog(path, dialogs.confirm, dialogs.refuse) is True

    assert _version(path) == db.CATALOG_VERSION
    assert _layout(path) == fresh
    con = sqlite3.connect(path)
    assert con.execute(bounds_sql).fetchone()[0] == bounds
    con.close()
    assert scanner.count_needing_recheck(path) == n_files
    assert len(db.list_files(db_path=path)) == n_files
