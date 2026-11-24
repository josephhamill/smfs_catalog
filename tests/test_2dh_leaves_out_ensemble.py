# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
The 2D histograms leave out ensemble traces, and say so.

A trace with an ensemble InvOLS in force would be drawn from two conversions:
x and F from the trace's own slope, the chain fit from the event map made under
the ensemble divisor.  Until a shared-conversion change lands (Joseph,
2026-11-24: "leave them out and be transparent that you are"), both 2D
histograms leave every such trace out.

What these tests rely on:
  * "in force" is db.ensemble_divisors(file_ids): an assigned file whose
    instrument value is unusable has no divisor and stays in;
  * the drop goes through the window's drop ledger under the reason
    "invols_ensemble" (ledger.DROP_REASONS), so Ledger.by_reason() counts it;
  * the window's stats line says "<n> left out: ensemble InvOLS" and its
    tooltip carries the ledger report, "<n> x <reason text>".

The stored event maps carry no fit, so each window's fit test is lifted and
each grid comes from a patched _compute_from_curve (a distinct matrix per
file, so "unchanged" is checkable).  The upstream Explore Events window is a
stub that only answers population_ledger.  Every catalog is a temporary file
passed explicitly.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt6.QtWidgets import QApplication                     # noqa: E402
_app = QApplication.instance() or QApplication([])           # before any smfs_catalog import

import invols_synth as synth                                 # noqa: E402
from smfs_catalog import db, ledger, provenance              # noqa: E402
from smfs_catalog import invols_store as store               # noqa: E402
from smfs_catalog.normalized_2dh_window import Normalized2DHWindow  # noqa: E402
from smfs_catalog.physical_2dh_window import Physical2DHWindow      # noqa: E402

INSTRUMENT = 199.14
GROUP_T = 210.0
REASON = "invols_ensemble"
WINDOWS = [Physical2DHWindow, Normalized2DHWindow]

# Files 1 and 3 keep their own fit; 2, 4 and 5 are assigned to a group.
OWN_NUMBERS = (1, 3)
ENSEMBLE_NUMBERS = (2, 4, 5)


@pytest.fixture(autouse=True)
def _pinned(monkeypatch):
    monkeypatch.setattr(provenance, "cache_version", lambda: synth.CODE)
    monkeypatch.setattr(store, "cache_version", lambda: synth.CODE, raising=False)
    for cls in WINDOWS:
        monkeypatch.setattr(cls, "_fit_drop_reason", lambda self, fit: None)
        monkeypatch.setattr(cls, "_compute_from_curve", _grid_of_path)


def _grid_of_path(self, path, pre_fetched=None, conn=None):
    """A matrix that depends on which file it is, so a test can tell them apart."""
    number = int(Path(path).stem[-4:])
    return np.full((2, 2), number, dtype=np.uint32)


class _Upstream:
    """The only thing a 2DH asks of Explore Events."""

    def __init__(self, paths):
        self._paths = list(paths)

    def population_ledger(self, which, axes=None):
        return ledger.Ledger("Explore Events population", self._paths)


def _catalog(tmp_path, n=5, assigned=ENSEMBLE_NUMBERS):
    files = [{"instrument": INSTRUMENT, "slope": -0.95} for _ in range(n)]
    db_path, ids = synth.make_catalog(tmp_path, files)
    synth.stamp_fit_key(db_path, ids)
    for fid in ids:
        synth.put_event_map(db_path, fid, "P", synth.event_map_doc(1))
    if assigned:
        synth.add_group(db_path, [ids[i - 1] for i in assigned], GROUP_T)
    paths = [synth.path_of(i) for i in range(1, n + 1)]
    return db_path, ids, paths


def _window(cls, db_path, paths):
    win = cls([{"path": p} for p in paths], db_path, population="all")
    win.sync_from_event_summary(_Upstream(paths))
    return win


def _numbers(win):
    return sorted(int(Path(p).stem[-4:]) for p in win._event_histograms)


@pytest.mark.parametrize("cls", WINDOWS)
def test_ensemble_traces_are_not_in_the_data_and_own_fit_traces_are(tmp_path, cls):
    db_path, _, paths = _catalog(tmp_path)
    win = _window(cls, db_path, paths)
    assert _numbers(win) == list(OWN_NUMBERS)
    for p, H in win._event_histograms.items():     # the own-fit grids are untouched
        number = int(Path(p).stem[-4:])
        assert H.toarray().tolist() == [[number] * 2] * 2
    win.close()


@pytest.mark.parametrize("cls", WINDOWS)
def test_the_summed_picture_is_the_own_fit_traces_alone(tmp_path, cls):
    db_path, _, paths = _catalog(tmp_path)
    win = _window(cls, db_path, paths)
    # grids full of 1 and of 3 over two traces: (1 + 3) / 2 in every bin
    assert win._cumulative.tolist() == [[2.0, 2.0], [2.0, 2.0]]
    win.close()


@pytest.mark.parametrize("cls", WINDOWS)
def test_the_ledger_records_each_left_out_file_under_the_named_reason(tmp_path, cls):
    db_path, _, paths = _catalog(tmp_path)
    win = _window(cls, db_path, paths)
    assert REASON in ledger.DROP_REASONS
    drops = [d for d in win._ledger.drops() if d.reason == REASON]
    assert sorted(d.path for d in drops) == sorted(
        synth.path_of(i) for i in ENSEMBLE_NUMBERS)
    assert win._ledger.by_reason() == {REASON: 3}
    assert (win._ledger.n_asked, win._ledger.n_kept, win._ledger.n_dropped) == (5, 2, 3)
    win.close()


@pytest.mark.parametrize("cls", WINDOWS)
def test_the_reason_is_in_the_export_manifest(tmp_path, cls):
    db_path, _, paths = _catalog(tmp_path)
    win = _window(cls, db_path, paths)
    assert win._ledger.manifest()["dropped_by_reason"] == {REASON: 3}
    win.close()


def test_the_reason_text_says_what_is_happening():
    assert ledger.DROP_REASONS[REASON] == (
        "uses an ensemble InvOLS; the 2D histograms still convert each trace "
        "with its own fit")


@pytest.mark.parametrize("cls", WINDOWS)
def test_the_window_says_how_many_were_left_out_and_why(tmp_path, cls):
    db_path, _, paths = _catalog(tmp_path)
    win = _window(cls, db_path, paths)
    line = win._stats_label.text()
    assert line.startswith("2 events")
    assert "3 left out: ensemble InvOLS" in line
    assert ("3 × uses an ensemble InvOLS; the 2D histograms still convert each "
            "trace with its own fit") in win._stats_label.toolTip()
    win.close()


@pytest.mark.parametrize("cls", WINDOWS)
def test_with_no_ensemble_trace_the_window_does_not_mention_one(tmp_path, cls):
    db_path, _, paths = _catalog(tmp_path, assigned=())
    win = _window(cls, db_path, paths)
    assert _numbers(win) == [1, 2, 3, 4, 5]
    assert "ensemble" not in win._stats_label.text()
    assert "ensemble" not in win._stats_label.toolTip()
    win.close()


@pytest.mark.parametrize("cls", WINDOWS)
def test_add_event_leaves_an_ensemble_trace_out_and_takes_an_own_fit_one(tmp_path, cls):
    db_path, _, paths = _catalog(tmp_path)
    win = cls([{"path": p} for p in paths], db_path, population="all")
    win.add_event(synth.path_of(2))        # ensemble
    assert win._event_histograms == {}
    win.add_event(synth.path_of(1))        # own fit
    assert _numbers(win) == [1]
    win.add_event(synth.path_of(4))        # ensemble
    assert _numbers(win) == [1]
    win.close()


@pytest.mark.parametrize("cls", WINDOWS)
def test_an_assigned_file_with_an_unusable_instrument_value_stays_in(tmp_path, cls):
    db_path, ids, paths = _catalog(tmp_path)
    synth.set_instrument(db_path, ids[1], None)       # file 2: assigned, no value
    synth.set_instrument(db_path, ids[3], 0.0)        # file 4: assigned, zero
    win = _window(cls, db_path, paths)
    assert _numbers(win) == [1, 2, 3, 4]
    assert win._ledger.by_reason() == {REASON: 1}     # only file 5
    fresh = cls([{"path": p} for p in paths], db_path, population="all")
    fresh.add_event(synth.path_of(2))
    assert _numbers(fresh) == [2]
    win.close()
    fresh.close()


@pytest.mark.parametrize("cls", WINDOWS)
def test_a_file_reverted_to_its_own_fit_comes_back_on_the_next_sync(tmp_path, cls):
    db_path, ids, paths = _catalog(tmp_path)
    win = _window(cls, db_path, paths)
    assert _numbers(win) == [1, 3]
    store.apply_own([ids[1], ids[3]], db_path)        # files 2 and 4
    win.sync_from_event_summary(_Upstream(paths))
    assert _numbers(win) == [1, 2, 3, 4]
    assert win._ledger.by_reason() == {REASON: 1}     # file 5 is still ensemble
    store.apply_own([ids[4]], db_path)
    win.sync_from_event_summary(_Upstream(paths))
    assert _numbers(win) == [1, 2, 3, 4, 5]
    assert win._ledger.n_dropped == 0
    assert "ensemble" not in win._stats_label.text()
    win.close()


@pytest.mark.parametrize("cls", WINDOWS)
def test_a_newly_assigned_file_leaves_on_the_next_sync(tmp_path, cls):
    db_path, ids, paths = _catalog(tmp_path)
    win = _window(cls, db_path, paths)
    synth.add_group(db_path, [ids[0]], GROUP_T)       # file 1 joins a group
    win.sync_from_event_summary(_Upstream(paths))
    assert _numbers(win) == [3]
    assert win._ledger.by_reason() == {REASON: 4}
    win.close()


@pytest.mark.parametrize("cls", WINDOWS)
def test_a_sync_asks_for_the_divisors_once_not_once_per_file(tmp_path, monkeypatch, cls):
    db_path, ids, paths = _catalog(tmp_path)
    calls = []
    real = db.ensemble_divisors

    def counting(file_ids, *a, **k):
        calls.append(None if file_ids is None else list(file_ids))
        return real(file_ids, *a, **k)

    monkeypatch.setattr(db, "ensemble_divisors", counting)
    win = _window(cls, db_path, paths)
    assert len(calls) == 1
    assert sorted(calls[0]) == sorted(ids)
    win.close()
