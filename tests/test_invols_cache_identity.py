# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Per-file cache identity (handover check 3): a file's stored results are keyed
by the global parameter text plus, only when an ensemble invOLS applies to it,
the divisor actually used.  Own-fit files keep today's key byte for byte.

Fixture: two files with instrument 199.14 nm/V; file 1 is assigned to a group
whose value is 210 nm/V, file 2 keeps its own fit.  The ensemble divisor of
file 1 is -199.14 / 210 (hand calculation: -0.94828571...); a second group of
220 nm/V gives -199.14 / 220.  Expected keys are written out from the plan's
rule, never read back from the code under test.  Every call passes an explicit
temporary catalog, and the code version is pinned so no git checkout is needed.
"""
import ast
import json
import math
import os
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fakeworker                                            # noqa: E402
import invols_synth as synth                                 # noqa: E402
from PyQt6.QtWidgets import QApplication                     # noqa: E402
from smfs_catalog import curve_analysis, db, roi_events, roi_pipeline  # noqa: E402
from smfs_catalog import rawcurve_window as raw              # noqa: E402
from smfs_catalog import signal_processing, roi_detection    # noqa: E402
from smfs_catalog.analysis_params import AnalysisParams      # noqa: E402

CODE = synth.CODE
INSTRUMENT = 199.14
GROUP_A = 210.0
GROUP_B = 220.0
DIV_A = -199.14 / 210
DIV_B = -199.14 / 220
PACKAGE = Path(__file__).resolve().parents[1] / "smfs_catalog"


# ── helpers and fixtures ─────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def pinned_version(monkeypatch):
    """The code version every module stamps and checks, fixed so these tests
    need no git checkout (invols_synth writes its rows under the same text)."""
    for module in (curve_analysis, roi_pipeline, raw):
        monkeypatch.setattr(module, "cache_version", lambda: CODE)


@pytest.fixture()
def catalog(tmp_path):
    """File 1 (assigned to the 210 nm/V group) and file 2 (own fit)."""
    db_path, ids = synth.make_catalog(tmp_path, [
        {"instrument": INSTRUMENT, "curve_type": "continuous_stretch"},
        {"instrument": INSTRUMENT, "curve_type": "continuous_stretch"},
    ])
    group = synth.add_group(db_path, [ids[0]], GROUP_A)
    return SimpleNamespace(db=db_path, ens=ids[0], own=ids[1], group=group)


def _all_params(db_path):
    """Today's one global parameter text, as the pipeline stores verdicts."""
    param_set = db.load_analysis_params(db_path)
    return curve_analysis.pipeline_params_from(param_set).all_params


def _event_key(db_path, divisor=None):
    """The plan's rule for a file's verdict key."""
    base = _all_params(db_path)
    return base if divisor is None else f"{base}|invols:{json.dumps(divisor)}"


def _event_params(db_path):
    param_set = db.load_analysis_params(db_path)
    return roi_pipeline.event_params_from(param_set)


def _map_key(db_path, divisor=None):
    """The plan's rule for an event_map key: today's document, plus one key
    "invols_divisor" when an ensemble applies."""
    document = asdict(_event_params(db_path))
    if divisor is not None:
        document["invols_divisor"] = divisor
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


def _event_row(db_path, file_id):
    """(params_json, value) of the file's stored verdict, or None."""
    rows = synth.fetch(
        db_path, "SELECT params_json, value FROM analysis_results "
        "WHERE file_id = ? AND analysis_type = 'event'", (file_id,))
    return rows[0] if rows else None


def _map_key_stored(db_path, file_id):
    rows = synth.fetch(db_path, "SELECT params_json FROM event_map "
                       "WHERE file_id = ?", (file_id,))
    return rows[0][0] if rows else None


# ── event_map_params_json ────────────────────────────────────────────────────

def test_plain_event_map_key_is_todays_text_byte_for_byte():
    ep = roi_pipeline.event_params_from(AnalysisParams())
    todays = json.dumps(asdict(ep), sort_keys=True, separators=(",", ":"))
    assert roi_pipeline.event_map_params_json(ep) == todays
    assert "invols_divisor" not in todays


def test_an_explicit_none_divisor_gives_the_same_plain_text():
    ep = roi_pipeline.event_params_from(AnalysisParams())
    todays = json.dumps(asdict(ep), sort_keys=True, separators=(",", ":"))
    assert roi_pipeline.event_map_params_json(ep, None) == todays
    assert roi_pipeline.event_map_params_json(
        ep, ensemble_divisor=None) == todays


def test_a_divisor_adds_exactly_one_key_to_the_document():
    ep = roi_pipeline.event_params_from(AnalysisParams())
    text = roi_pipeline.event_map_params_json(ep, -0.95)
    expected = json.loads(json.dumps({**asdict(ep), "invols_divisor": -0.95}))
    assert json.loads(text) == expected
    assert json.loads(text)["invols_divisor"] == -0.95


@pytest.mark.parametrize("divisor", [-0.95, DIV_A, -1.0 / 3.0, -0.9482857142857143])
def test_the_divisor_survives_the_text_exactly(divisor):
    ep = roi_pipeline.event_params_from(AnalysisParams())
    text = roi_pipeline.event_map_params_json(ep, divisor)
    assert json.loads(text)["invols_divisor"] == divisor


def test_the_same_divisor_twice_gives_the_same_text():
    ep = roi_pipeline.event_params_from(AnalysisParams())
    first = roi_pipeline.event_map_params_json(ep, DIV_A)
    second = roi_pipeline.event_map_params_json(ep, DIV_A)
    assert first == second


def test_different_divisors_give_different_texts_down_to_the_last_bit():
    ep = roi_pipeline.event_params_from(AnalysisParams())
    plain = roi_pipeline.event_map_params_json(ep)
    a = roi_pipeline.event_map_params_json(ep, DIV_A)
    b = roi_pipeline.event_map_params_json(ep, DIV_B)
    neighbour = roi_pipeline.event_map_params_json(
        ep, math.nextafter(DIV_A, 0.0))
    assert len({plain, a, b, neighbour}) == 4


# ── the source scan ──────────────────────────────────────────────────────────

_NAME = "event_map_params_json"


def _passes_second_argument(call):
    positional = call.args
    if any(isinstance(arg, ast.Starred) for arg in positional):
        return False                      # cannot be shown to carry two
    if len(positional) >= 2:
        return True
    return any(kw.arg == "ensemble_divisor" for kw in call.keywords)


def _bound_names(tree):
    """Every name `event_map_params_json` can be reached by in this module:
    its own name, plus any `import ... as alias`."""
    names = {_NAME}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(a.asname for a in node.names
                         if a.name == _NAME and a.asname)
    return names


def _is_target(func, names):
    if isinstance(func, ast.Name):
        return func.id in names
    return isinstance(func, ast.Attribute) and func.attr in names


def _scan_calls(tree, names):
    """(calls inspected, the id of each called function node, problems) for the
    calls of the function in `tree`."""
    calls, call_funcs, problems = 0, set(), []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_target(node.func, names):
            calls += 1
            call_funcs.add(id(node.func))
            if not _passes_second_argument(node):
                problems.append(f"line {node.lineno}: call without a "
                                f"second argument")
    return calls, call_funcs, problems


def _is_reference(node, names):
    by_name = (isinstance(node, ast.Name) and node.id in names
               and isinstance(node.ctx, ast.Load))
    return by_name or (isinstance(node, ast.Attribute) and node.attr in names)


def _uncalled_uses(tree, names, call_funcs):
    """The uses of the function that are not the function of a call."""
    return [f"line {node.lineno}: used without being called"
            for node in ast.walk(tree)
            if _is_reference(node, names) and id(node) not in call_funcs]


def scan_source(source):
    """(calls inspected, problems) for one module's text.  A problem is a call
    that omits the second argument, or a use of the function that is not a call
    (a callable handed on cannot be checked)."""
    tree = ast.parse(source)
    names = _bound_names(tree)
    calls, call_funcs, problems = _scan_calls(tree, names)
    return calls, problems + _uncalled_uses(tree, names, call_funcs)


def test_the_scan_flags_a_call_with_no_second_argument():
    calls, problems = scan_source("x = event_map_params_json(ep)\n")
    assert calls == 1 and len(problems) == 1


def test_the_scan_flags_a_method_style_call_with_one_argument():
    calls, problems = scan_source("x = roi_pipeline.event_map_params_json(ep)\n")
    assert calls == 1 and len(problems) == 1


def test_the_scan_flags_a_call_through_an_import_alias():
    source = ("from .roi_pipeline import event_map_params_json as emp\n"
              "x = emp(ep)\n")
    # `emp` is reached by the alias; the scan must still see it.
    calls, problems = scan_source(source)
    assert calls == 1 and len(problems) == 1


def test_the_scan_flags_unpacked_arguments_it_cannot_check():
    assert scan_source("x = event_map_params_json(*args)\n")[1]
    assert scan_source("x = event_map_params_json(ep, **kw)\n")[1]


def test_the_scan_flags_the_function_handed_on_without_a_call():
    source = "g = functools.partial(event_map_params_json, ep)\n"
    assert scan_source(source)[1]


@pytest.mark.parametrize("call", [
    "event_map_params_json(ep, divisor)",
    "event_map_params_json(ep, None)",
    "event_map_params_json(ep, ensemble_divisor=divisor)",
    "roi_pipeline.event_map_params_json(ep, divisors.get(fid))",
])
def test_the_scan_accepts_a_call_that_passes_the_second_argument(call):
    calls, problems = scan_source(f"x = {call}\n")
    assert calls == 1 and problems == []


def test_the_definition_is_not_a_call_and_is_not_flagged():
    source = ("def event_map_params_json(ep, ensemble_divisor=None):\n"
              "    return ''\n")
    assert scan_source(source) == (0, [])


def test_every_call_in_the_package_passes_the_second_argument():
    inspected_in, problems = {}, []
    for path in sorted(PACKAGE.glob("*.py")):
        calls, found = scan_source(path.read_text(encoding="utf-8"))
        inspected_in[path.name] = calls
        problems += [f"{path.name} {item}" for item in found]
    assert inspected_in["roi_pipeline.py"] >= 1
    assert inspected_in["rawcurve_window.py"] >= 1
    assert problems == []


# ── analyse_file ─────────────────────────────────────────────────────────────

@pytest.fixture()
def lane(monkeypatch):
    """Stand-ins for loading and analysing a curve, so the lane's own cache
    logic runs against the real catalog.  `hooks` run inside an analyse_curve
    call; `persist_hooks` run inside the stored-event-map step, which is
    reached (and recorded in `persists`) only when `event` is set."""
    state = SimpleNamespace(calls=[], hooks=[], persists=[], persist_hooks=[],
                            event=False)

    def fake_analyse_curve(curve, p, *, file_id, **kwargs):
        state.calls.append(file_id)
        if state.hooks:
            state.hooks.pop(0)()
        verdict = "event" if state.event else "non_event"
        return SimpleNamespace(event=state.event, verdict=verdict), None

    def fake_persist(file_id, curve, db_path, code_ver, **kwargs):
        state.persists.append(file_id)
        if state.persist_hooks:
            state.persist_hooks.pop(0)()

    monkeypatch.setattr(curve_analysis, "_persist_multi_event_roi",
                        fake_persist)
    monkeypatch.setattr(curve_analysis, "load_force_curve",
                        lambda path: SimpleNamespace(defl_retr=np.zeros(4)))
    monkeypatch.setattr(curve_analysis, "analyse_curve", fake_analyse_curve)
    monkeypatch.setattr(curve_analysis._db, "get_or_store_deflection_histogram",
                        lambda *a, **k: None)
    return state


def _analyse(catalog, file_id):
    number = 1 if file_id == catalog.ens else 2
    return curve_analysis.analyse_file(file_id, synth.path_of(number), catalog.db)


def test_an_own_fit_file_with_a_plain_key_row_is_served_from_cache(catalog, lane):
    synth.put_result(catalog.db, catalog.own, "event", 1.0,
                     _event_key(catalog.db))
    assert _analyse(catalog, catalog.own) == ("event", True)
    assert lane.calls == []


def test_an_ensemble_file_with_only_a_plain_key_row_is_analysed_again(catalog, lane):
    synth.put_result(catalog.db, catalog.ens, "event", 1.0,
                     _event_key(catalog.db))
    verdict, was_cached = _analyse(catalog, catalog.ens)
    assert was_cached is False
    assert lane.calls == [catalog.ens]
    assert _event_row(catalog.db, catalog.ens)[0] == _event_key(catalog.db, DIV_A)


def test_a_second_pass_over_an_ensemble_file_is_cached(catalog, lane):
    _analyse(catalog, catalog.ens)
    assert _event_row(catalog.db, catalog.ens)[0] == _event_key(catalog.db, DIV_A)
    _verdict, was_cached = _analyse(catalog, catalog.ens)
    assert was_cached is True
    assert lane.calls == [catalog.ens]


def test_a_new_group_makes_the_file_uncached_again(catalog, lane):
    _analyse(catalog, catalog.ens)
    synth.add_group(catalog.db, [catalog.ens], GROUP_B)
    _verdict, was_cached = _analyse(catalog, catalog.ens)
    assert was_cached is False
    assert lane.calls == [catalog.ens, catalog.ens]
    assert _event_row(catalog.db, catalog.ens)[0] == _event_key(catalog.db, DIV_B)


def test_reapplying_the_same_value_leaves_the_file_cached(catalog, lane):
    _analyse(catalog, catalog.ens)
    synth.add_group(catalog.db, [catalog.ens], GROUP_A)       # a new group, same value
    _verdict, was_cached = _analyse(catalog, catalog.ens)
    assert was_cached is True
    assert lane.calls == [catalog.ens]


def test_going_back_to_the_own_fit_returns_to_the_plain_key(catalog, lane):
    _analyse(catalog, catalog.ens)
    synth.unassign(catalog.db, [catalog.ens])
    _verdict, was_cached = _analyse(catalog, catalog.ens)
    assert was_cached is False
    assert _event_row(catalog.db, catalog.ens)[0] == _event_key(catalog.db)


def test_a_divisor_that_changes_during_a_pass_reruns_the_file(catalog, lane):
    lane.hooks.append(
        lambda: synth.add_group(catalog.db, [catalog.ens], GROUP_B))
    verdict, was_cached = _analyse(catalog, catalog.ens)
    assert lane.calls == [catalog.ens, catalog.ens]
    assert was_cached is False
    assert verdict == "non_event"
    assert _event_row(catalog.db, catalog.ens)[0] == _event_key(catalog.db, DIV_B)


def test_a_divisor_that_changes_while_the_event_map_is_stored_reruns_the_file(
        catalog, lane):
    lane.event = True
    lane.persist_hooks.append(
        lambda: synth.add_group(catalog.db, [catalog.ens], GROUP_B))
    verdict, was_cached = _analyse(catalog, catalog.ens)
    assert lane.calls == [catalog.ens, catalog.ens]
    assert lane.persists == [catalog.ens, catalog.ens]
    assert (verdict, was_cached) == ("event", False)
    assert _event_row(catalog.db, catalog.ens) == (
        _event_key(catalog.db, DIV_B), 1.0)


def test_a_divisor_that_changes_during_analysis_stores_no_event_map_for_the_old_one(
        catalog, lane):
    lane.event = True
    lane.hooks.append(
        lambda: synth.add_group(catalog.db, [catalog.ens], GROUP_B))
    verdict, was_cached = _analyse(catalog, catalog.ens)
    assert lane.calls == [catalog.ens, catalog.ens]
    assert lane.persists == [catalog.ens]          # only the re-run stores one
    assert (verdict, was_cached) == ("event", False)
    assert _event_row(catalog.db, catalog.ens) == (
        _event_key(catalog.db, DIV_B), 1.0)


def test_a_divisor_that_holds_during_a_pass_does_not_rerun_the_file(catalog, lane):
    _analyse(catalog, catalog.ens)
    assert lane.calls == [catalog.ens]


@pytest.mark.parametrize("instrument", [None, 0.0, -5.0, float("inf")])
def test_an_assigned_file_without_a_usable_instrument_value_uses_its_plain_key(
        catalog, lane, instrument):
    synth.set_instrument(catalog.db, catalog.ens, instrument)
    assert db.file_params_key_for(catalog.ens, "P", catalog.db) == "P"
    synth.put_result(catalog.db, catalog.ens, "event", 1.0,
                     _event_key(catalog.db))
    assert _analyse(catalog, catalog.ens) == ("event", True)
    assert lane.calls == []
    # and what a pass writes goes under the plain key too
    con = sqlite3.connect(catalog.db)
    with con:
        con.execute("DELETE FROM analysis_results WHERE file_id = ? AND "
                    "analysis_type = 'event'", (catalog.ens,))
    con.close()
    _verdict, was_cached = _analyse(catalog, catalog.ens)
    assert was_cached is False
    assert _event_row(catalog.db, catalog.ens)[0] == _event_key(catalog.db)


@pytest.mark.parametrize("instrument", [None, 0.0, -5.0])
def test_an_assigned_file_without_a_usable_instrument_reads_its_plain_event_map(
        catalog, instrument):
    synth.set_instrument(catalog.db, catalog.ens, instrument)
    synth.put_event_map(catalog.db, catalog.ens, _map_key(catalog.db),
                        synth.event_map_doc(2))
    rows = roi_pipeline.assemble_rows(
        [synth.path_of(1)], catalog.db, mode="all")
    assert {row["file_id"] for row in rows} == {catalog.ens}
    assert roi_pipeline.coverage([synth.path_of(1)], catalog.db) == (1, 1)


# ── the batch readers ────────────────────────────────────────────────────────

# (key kind of file 1's document, key kind of file 2's document,
#  the files whose document matches their own key, histogram of those)
# File 1 is the ensemble file with a 2-rupture document; file 2 is the own-fit
# file with a 1-rupture document.  A "suffixed" document sits under the
# ensemble key; a "plain" one under today's key.
READER_CASES = [
    ("suffixed", "plain", {1, 2}, {1: 1, 2: 1}),
    ("plain", "suffixed", set(), {}),
    ("plain", "plain", {2}, {1: 1}),
    ("suffixed", "suffixed", {1}, {2: 1}),
]
READER_IDS = ["ens-suffixed/own-plain", "ens-plain/own-suffixed",
              "both-plain", "both-suffixed"]
PATHS = [synth.path_of(1), synth.path_of(2)]


@pytest.fixture()
def seeded(catalog):
    """Put a document for each file under the key kind named."""
    def seed(ens_kind, own_kind):
        for file_id, kind, n in ((catalog.ens, ens_kind, 2),
                                 (catalog.own, own_kind, 1)):
            key = _map_key(catalog.db, DIV_A if kind == "suffixed" else None)
            synth.put_event_map(catalog.db, file_id, key,
                                synth.event_map_doc(n))
        return catalog
    return seed


@pytest.mark.parametrize("ens_kind, own_kind, found, _hist", READER_CASES,
                         ids=READER_IDS)
def test_assemble_rows_finds_only_documents_under_each_files_own_key(
        seeded, ens_kind, own_kind, found, _hist):
    catalog = seeded(ens_kind, own_kind)
    rows = roi_pipeline.assemble_rows(PATHS, catalog.db, mode="all")
    assert {row["file_id"] for row in rows} == found
    assert len(rows) == (2 if 1 in found else 0) + (1 if 2 in found else 0)


@pytest.mark.parametrize("ens_kind, own_kind, _found, hist", READER_CASES,
                         ids=READER_IDS)
def test_roi_count_histogram_counts_only_documents_under_each_files_own_key(
        seeded, ens_kind, own_kind, _found, hist):
    catalog = seeded(ens_kind, own_kind)
    assert roi_pipeline.roi_count_histogram(PATHS, catalog.db) == hist


@pytest.mark.parametrize("ens_kind, own_kind, found, _hist", READER_CASES,
                         ids=READER_IDS)
def test_coverage_counts_only_documents_under_each_files_own_key(
        seeded, ens_kind, own_kind, found, _hist):
    catalog = seeded(ens_kind, own_kind)
    assert roi_pipeline.coverage(PATHS, catalog.db) == (len(found), 2)


@pytest.mark.parametrize("ens_kind, own_kind, found, _hist", READER_CASES,
                         ids=READER_IDS)
def test_populate_event_map_skips_a_file_only_when_its_own_key_has_a_document(
        seeded, monkeypatch, ens_kind, own_kind, found, _hist):
    catalog = seeded(ens_kind, own_kind)
    computed = []
    monkeypatch.setattr(roi_pipeline, "load_force_curve", lambda path: object())
    monkeypatch.setattr(
        roi_pipeline, "compute_curve_events",
        lambda curve, ep, **kw: computed.append(kw["file_id"]))
    result = roi_pipeline.populate_event_map(PATHS, catalog.db)
    assert result["skipped"] == len(found)
    assert set(computed) == {1, 2} - found
    assert result["ok"] == 2 - len(found)


# ── compute_curve_events_coords ──────────────────────────────────────────────

@pytest.fixture()
def numerics(monkeypatch):
    """Stand-ins for the numerical stages, so the function's own key handling
    runs against the real catalog.  `builds` counts event builds."""
    arr = np.arange(20, dtype=float)
    curve = SimpleNamespace(piezo_retr=arr, piezo_appr=arr, defl_retr=arr,
                            spring_constant=1.0, inv_ols_nm_v=None)
    dc = SimpleNamespace(low_retr=arr, low_appr=arr)
    sigs = SimpleNamespace(piezo=arr, d1=arr, low=arr)
    builds = []

    def build(*_args, **_kwargs):
        builds.append(1)
        return roi_events.CurveEvents(rois=[], detector="")

    monkeypatch.setattr(signal_processing, "fit_retract_baseline",
                        lambda *_: SimpleNamespace(
                            offset=0.0, slope=0.0, intercept=0.0, rms=0.0,
                            fit_lo_idx=0, fit_hi_idx=1))
    monkeypatch.setattr(signal_processing, "decompose_curve",
                        lambda *_a, **_k: dc)
    monkeypatch.setattr(signal_processing, "find_end_in_contact",
                        lambda *_a, **_k: (2, None))
    monkeypatch.setattr(signal_processing, "fit_approach_invols",
                        lambda *_a, **_k: SimpleNamespace(
                            slope=-1.0, intercept=0.0, rms=0.0,
                            fit_lo_idx=0, fit_hi_idx=10))
    monkeypatch.setattr(roi_detection, "compute_detection_signals",
                        lambda **_k: sigs)
    monkeypatch.setattr(roi_detection, "rupture_search_bounds",
                        lambda *_a: (3, 15))
    monkeypatch.setattr(roi_events, "build_curve_events", build)
    monkeypatch.setattr(roi_events, "fit_segments", lambda *_a, **_k: None)
    return SimpleNamespace(curve=curve, builds=builds)


def _compute(catalog, numerics, file_id):
    roi_pipeline.compute_curve_events_coords(
        numerics.curve, _event_params(catalog.db), db_path=catalog.db,
        code_ver=CODE, file_id=file_id,
        param_set=db.load_analysis_params(catalog.db))


def _histogram_row(db_path, file_id):
    return synth.fetch(
        db_path, "SELECT histogram, x_bins, f_bins, params_json, computed_at "
        "FROM event_histograms WHERE file_id = ?", (file_id,))


def test_an_ensemble_file_gets_its_document_under_the_suffixed_key(
        catalog, numerics):
    _compute(catalog, numerics, catalog.ens)
    assert _map_key_stored(catalog.db, catalog.ens) == _map_key(catalog.db, DIV_A)


def test_an_own_fit_file_gets_its_document_under_the_plain_key(
        catalog, numerics):
    _compute(catalog, numerics, catalog.own)
    assert _map_key_stored(catalog.db, catalog.own) == _map_key(catalog.db)


def test_a_document_under_the_files_own_key_is_reused(catalog, numerics):
    synth.put_event_map(catalog.db, catalog.ens, _map_key(catalog.db, DIV_A),
                        synth.event_map_doc(2))
    _compute(catalog, numerics, catalog.ens)
    assert numerics.builds == []
    assert _map_key_stored(catalog.db, catalog.ens) == _map_key(catalog.db, DIV_A)


def test_the_histogram_row_is_unchanged_when_the_divisor_changed(
        catalog, numerics):
    synth.add_other_rows(catalog.db, catalog.ens)
    synth.put_event_map(catalog.db, catalog.ens, _map_key(catalog.db, DIV_A),
                        synth.event_map_doc(2))
    synth.add_group(catalog.db, [catalog.ens], GROUP_B)
    before = _histogram_row(catalog.db, catalog.ens)
    assert len(before) == 1
    _compute(catalog, numerics, catalog.ens)
    assert numerics.builds == [1]
    assert _map_key_stored(catalog.db, catalog.ens) == _map_key(catalog.db, DIV_B)
    assert _histogram_row(catalog.db, catalog.ens) == before


def test_the_histogram_row_is_unchanged_when_the_divisor_stayed_the_same(
        catalog, numerics):
    synth.add_other_rows(catalog.db, catalog.ens)
    synth.put_event_map(catalog.db, catalog.ens, _map_key(catalog.db, DIV_A),
                        synth.event_map_doc(2))
    before = _histogram_row(catalog.db, catalog.ens)
    assert len(before) == 1
    _compute(catalog, numerics, catalog.ens)
    assert numerics.builds == []
    assert _map_key_stored(catalog.db, catalog.ens) == _map_key(catalog.db, DIV_A)
    assert _histogram_row(catalog.db, catalog.ens) == before


def test_the_histogram_row_is_unchanged_when_the_file_went_back_to_its_own_fit(
        catalog, numerics):
    synth.add_other_rows(catalog.db, catalog.ens)
    synth.put_event_map(catalog.db, catalog.ens, _map_key(catalog.db, DIV_A),
                        synth.event_map_doc(2))
    synth.unassign(catalog.db, [catalog.ens])
    before = _histogram_row(catalog.db, catalog.ens)
    assert len(before) == 1
    _compute(catalog, numerics, catalog.ens)
    assert numerics.builds == [1]
    assert _map_key_stored(catalog.db, catalog.ens) == _map_key(catalog.db)
    assert _histogram_row(catalog.db, catalog.ens) == before


# ── the curve view's overlay reader ──────────────────────────────────────────

def _raw_window(db_path):
    app = QApplication.instance() or QApplication([])
    win = raw.RawCurveWindow(fakeworker.FakeWorker(), db_path=db_path)
    win._test_app = app
    return win


def _store_overlays(catalog, file_id, verdict_divisor, map_divisor):
    synth.put_result(catalog.db, file_id, "event", 1.0,
                     _event_key(catalog.db, verdict_divisor))
    synth.put_event_map(catalog.db, file_id, _map_key(catalog.db, map_divisor),
                        synth.event_map_doc(1))


@pytest.mark.parametrize("verdict_divisor, map_divisor, drawn", [
    (None, None, False),        # both under plain keys: not this file's
    (DIV_A, DIV_A, True),       # both under its own key
    (DIV_A, None, False),       # the verdict is found, the event map is not
    (None, DIV_A, False),       # the event map is there, the verdict is not
], ids=["plain-rows", "suffixed-rows", "verdict-only", "map-only"])
def test_the_overlay_reader_looks_under_an_ensemble_files_own_keys(
        catalog, verdict_divisor, map_divisor, drawn):
    _store_overlays(catalog, catalog.ens, verdict_divisor, map_divisor)
    win = _raw_window(catalog.db)
    try:
        assert win._draw_persisted_overlays(catalog.ens) is drawn
    finally:
        win.close()


def test_the_overlay_reader_still_draws_an_own_fit_file_from_plain_keys(catalog):
    _store_overlays(catalog, catalog.own, None, None)
    win = _raw_window(catalog.db)
    try:
        assert win._draw_persisted_overlays(catalog.own) is True
    finally:
        win.close()


def test_the_overlay_reader_does_not_draw_an_own_fit_file_from_suffixed_keys(
        catalog):
    _store_overlays(catalog, catalog.own, DIV_A, DIV_A)
    win = _raw_window(catalog.db)
    try:
        assert win._draw_persisted_overlays(catalog.own) is False
    finally:
        win.close()
