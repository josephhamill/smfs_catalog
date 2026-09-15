"""The deflection histogram: one grid, measured at import, summable after."""

from __future__ import annotations

import os

import numpy as np
import pytest

import tmpdirs

from smfs_catalog import db as _db
from smfs_catalog import event_processor as _ep
from smfs_catalog.curve_loader import retract_deflection_nm


_NOW = "2026-01-01 00:00:00"


@pytest.fixture
def catalog():
    path = os.path.join(tmpdirs.mkdtemp(prefix="smfs_deflhist_"), "catalog.db")
    _db.initialise(path)
    return path


def _add_file(db_path: str, path: str) -> int:
    _db.upsert_file(
        {"path": path, "filename": os.path.basename(path), "parse_ok": 1,
         "first_seen": _NOW, "last_seen": _NOW},
        db_path=db_path)
    return _db.get_file_id(path, db_path)


def test_counts_survive_the_round_trip_and_only_on_their_own_grid(catalog):
    fid = _add_file(catalog, "/data/a.ibw")
    defl = np.concatenate([np.zeros(500), np.linspace(0.0, 40.0, 500)])
    counts, below, above = _ep.compute_deflection_histogram(defl)
    _db.write_deflection_histogram(
        fid, counts, below, above, _ep.defl_grid_params(), catalog)

    got, n_below, n_above = _db.get_deflection_histogram(
        fid, _ep.defl_grid_params(), catalog)
    assert np.array_equal(got, counts)
    assert got.dtype == np.uint32
    assert (n_below, n_above) == (0, 0)
    assert int(got.sum()) == defl.size

    # A different grid is a miss, not a silently mismatched answer.
    assert _db.get_deflection_histogram(
        fid, _ep.defl_grid_params(bins=64), catalog) is None


def test_a_population_is_the_sum_of_its_curves(catalog):
    """The property the whole design rests on: fixed edges make rows addable,
    so a cohort histogram never needs the curves re-read."""
    rng = np.random.default_rng(0)
    samples, paths = [], []
    for i in range(5):
        defl = rng.normal(loc=i, scale=2.0, size=400)
        fid = _add_file(catalog, f"/data/{i}.ibw")
        counts, below, above = _ep.compute_deflection_histogram(defl)
        _db.write_deflection_histogram(
            fid, counts, below, above, _ep.defl_grid_params(), catalog)
        samples.append(defl)
        paths.append(f"/data/{i}.ibw")

    total, below, above, n = _db.sum_deflection_histograms(
        paths, _ep.defl_grid_params(), _ep.DEFL_HIST_BINS, catalog)
    direct, d_below, d_above = _ep.compute_deflection_histogram(
        np.concatenate(samples))

    assert np.array_equal(total.astype(np.uint32), direct)
    assert (n, below, above) == (5, d_below, d_above)
    assert total.dtype == np.uint64


def test_samples_off_the_grid_are_counted_not_dropped(catalog):
    lo, hi = _ep.DEFL_HIST_RANGE
    defl = np.array([lo - 10.0, lo - 1.0, 0.0, 5.0, hi + 50.0])
    counts, below, above = _ep.compute_deflection_histogram(defl)
    assert (below, above) == (2, 1)
    assert int(counts.sum()) == 2

    fid = _add_file(catalog, "/data/wide.ibw")
    _db.write_deflection_histogram(
        fid, counts, below, above, _ep.defl_grid_params(), catalog)
    _, n_below, n_above = _db.get_deflection_histogram(
        fid, _ep.defl_grid_params(), catalog)
    assert (n_below, n_above) == (2, 1)


def test_non_finite_samples_are_in_no_bin_and_are_not_out_of_range():
    defl = np.array([0.0, 1.0, np.nan, np.inf, -np.inf])
    counts, below, above = _ep.compute_deflection_histogram(defl)
    assert int(counts.sum()) == 2
    assert (below, above) == (0, 0)


def test_the_scanner_and_the_loader_mean_the_same_thing_by_retract_deflection():
    """The scanner bins from the raw wave and the viewer plots from a loaded
    ForceCurve; both go through retract_deflection_nm, so they cannot diverge."""
    wdata = np.zeros((10, 2), dtype=float)
    wdata[:, 0] = np.arange(10)                 # Raw (piezo)
    wdata[:, 1] = np.arange(10) * 1e-9          # Defl, metres
    labels = [[], [b"", b"Raw", b"Defl"]]

    got = retract_deflection_nm(wdata, labels, idx_turn=4)
    # Samples 5..9, in nm, referenced to sample 0.
    assert np.allclose(got, [5.0, 6.0, 7.0, 8.0, 9.0])


_LABELS = [[], [b"", b"Raw", b"Defl", b"ZSnsr", b"Time"], [], []]
_NOTE = b"\rSpringConstant: 0.05\r"


def _stretch_wave(n: int = 2000) -> np.ndarray:
    """A minimal wave that qualifies as a usable continuous stretch."""
    half = n // 2
    piezo = np.concatenate([np.linspace(0.0, 1e-6, half),
                            np.linspace(1e-6, 0.0, n - half)])
    w = np.zeros((n, 4), dtype=float)
    w[:, 0] = piezo
    w[:, 2] = piezo
    w[:, 1] = -1e-8 * np.sin(np.linspace(0, np.pi, n)) - 1e-9
    w[:, 3] = np.linspace(0, 1, n)
    return w


def _serve_wave(monkeypatch, wdata) -> None:
    from smfs_catalog import scanner
    monkeypatch.setattr(scanner, "load_ibw", lambda _buf: {
        "wave": {"note": _NOTE, "wData": wdata, "labels": _LABELS,
                 "wave_header": {"sfA": [1e-4]}}})


def test_only_a_stretch_wave_is_binned_at_import(monkeypatch, tmp_path):
    """An image or a force-clamp trace has no retract half to describe, and a
    truncated one has nothing in it; none of them get a row."""
    from smfs_catalog import scanner

    stretch = _stretch_wave()
    truncated = stretch.copy()
    truncated[stretch.shape[0] // 2 - 1:, 1] = 0.0

    def parse(wdata):
        f = tmp_path / "w.ibw"
        f.write_bytes(b"x")
        _serve_wave(monkeypatch, wdata)
        return scanner._parse_ibw(str(f))

    assert "_defl_histogram" in parse(stretch)
    assert "_defl_histogram" not in parse(truncated)
    assert "_defl_histogram" not in parse(np.zeros((64, 64), dtype=float))


def test_a_handed_in_connection_is_left_for_its_owner_to_commit(catalog):
    """The scanner commits in batches; a histogram write must not commit the
    batch out from under it."""
    import sqlite3

    fid = _add_file(catalog, "/data/batch.ibw")
    counts, below, above = _ep.compute_deflection_histogram(np.zeros(10))
    conn = sqlite3.connect(catalog)
    try:
        _db.write_deflection_histogram(
            fid, counts, below, above, _ep.defl_grid_params(), catalog,
            conn=conn)
        conn.rollback()
    finally:
        conn.close()
    assert _db.get_deflection_histogram(
        fid, _ep.defl_grid_params(), catalog) is None


def test_a_stored_histogram_is_used_and_a_missing_one_is_binned_and_stored(
        catalog):
    fid = _add_file(catalog, "/data/once.ibw")
    first = np.zeros(50) + 2.0

    counts, _b, _a, stored_now = _db.get_or_store_deflection_histogram(
        fid, first, catalog)
    assert stored_now
    assert int(counts.sum()) == 50

    # Different samples, same file: the stored row answers, nothing is rebinned.
    counts, _b, _a, stored_now = _db.get_or_store_deflection_histogram(
        fid, np.zeros(7), catalog)
    assert not stored_now
    assert int(counts.sum()) == 50


def test_recheck_bins_the_curves_it_reads(monkeypatch, tmp_path, catalog):
    from smfs_catalog import scanner

    f = tmp_path / "old.ibw"
    f.write_bytes(b"x")
    fid = _add_file(catalog, str(f))
    assert _db.get_deflection_histogram(
        fid, _ep.defl_grid_params(), catalog) is None

    _serve_wave(monkeypatch, _stretch_wave())
    scanner.requalify_catalog(catalog)

    row = _db.get_deflection_histogram(fid, _ep.defl_grid_params(), catalog)
    assert row is not None
    assert int(row[0].sum()) > 0


def test_the_panel_takes_its_grid_from_the_one_place_that_defines_it():
    """The stored rows and the drawn bars must share edges, or the population
    histogram is summed on one grid and plotted against another."""
    import inspect
    from smfs_catalog import class_lineplot_window as w

    src = inspect.getsource(w)
    assert "_ep.defl_bin_edges()" in src
    assert str(_ep.DEFL_HIST_BINS) not in src, (
        "the window retypes the bin count instead of importing it")


def test_a_row_of_the_wrong_length_is_skipped_not_broadcast(catalog):
    """numpy would spread a length-1 array across all 1400 bins and return a
    population histogram nobody measured. The shortfall is reported instead."""
    import sqlite3
    import zlib

    good = _add_file(catalog, "/data/good.ibw")
    counts, below, above = _ep.compute_deflection_histogram(
        np.zeros(100) + 3.0)
    _db.write_deflection_histogram(
        good, counts, below, above, _ep.defl_grid_params(), catalog)

    bad = _add_file(catalog, "/data/bad.ibw")
    conn = sqlite3.connect(catalog)
    with conn:
        conn.execute(
            "INSERT INTO deflection_histograms "
            "(file_id, counts, n_bins, n_below, n_above, params_json, "
            " computed_at) VALUES (?,?,?,?,?,?,?)",
            (bad, zlib.compress(np.array([7], dtype=np.uint32).tobytes()),
             _ep.DEFL_HIST_BINS, 0, 0, _ep.defl_grid_params(), _NOW))
    conn.close()

    total, _below, _above, n = _db.sum_deflection_histograms(
        ["/data/good.ibw", "/data/bad.ibw"], _ep.defl_grid_params(),
        _ep.DEFL_HIST_BINS, catalog)

    assert n == 1, "the malformed row was counted as a curve"
    assert np.array_equal(total.astype(np.uint32), counts)
