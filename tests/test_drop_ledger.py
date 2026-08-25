# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Regression test: windows report what they were GIVEN, not only what they kept
(#117), and the queue says what it will actually cost (#96).

The contract under test:
(a) The reason vocabulary is closed.  Every reason any module records is one
    DROP_REASONS declares — a typo'd reason would otherwise vanish into a
    tally nobody can reconcile, which is the failure this issue is about.
(b) The tally reconciles.  by_reason() sums to exactly n_dropped, and
    n_asked = n_kept + n_dropped.  A breakdown that doesn't add up hands the
    reader a second puzzle instead of an explanation.
(c) The journey survives a stage boundary.  A curve dropped upstream can
    still be explained by a downstream window, via absorb() — the "follow one
    trace through the filtering" half of #117.
(d) Gate membership and plottability are separate questions.
    population_ledger() distinguishes "not a hit" from "no fit", because the
    remedies differ (retune the criteria vs. re-analyse the curve) and
    population_paths() used to fold both into one silent answer.
(e) A fit-free 2DH align mode does NOT require a WLC fit.  onset (the
    DEFAULT), snap-off and rupture all anchor on real data points; requiring
    l_p/l_c for them discarded curves that plot perfectly well.  #117 flagged
    this as unverified — it was real.
(f) Every exporting window's provenance carries the tally, so a manifest read
    months later without the app can account for its own row count.
(g) queue_freshness answers the same question curve_analysis's fast path
    asks, per row: stored under today's params AND today's code version.
    Anything else is work the ETA must price as work.
(h) The freshness verdict is DERIVED, never stored — no column anywhere
    holds it, so no parameter edit or checkout can leave a stale copy behind
    (the files.hit lesson).

Run with the smfs-catalog env, from the repo root:
    python tests/test_drop_ledger.py
"""
import ast
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smfs_catalog import db as _db
from smfs_catalog.provenance import cache_version
from smfs_catalog.ledger import DROP_REASONS, Ledger

# One shared idiom for these procedural guards — see checkstyle.py.
import checkstyle                                          # noqa: E402

check = checkstyle.CheckRunner()


PKG = Path(__file__).resolve().parent.parent / "smfs_catalog"

# ── (a) the reason vocabulary is closed ──────────────────────────────────────
# Every literal handed to .drop()/.drop_all() as its `reason` argument, read
# with ast rather than a regex so a reason NAMED IN A COMMENT (this file
# explains several) can't be mistaken for one being recorded.
used_reasons: set[str] = set()
for path in sorted(PKG.glob("*.py")):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not isinstance(fn, ast.Attribute) or fn.attr not in ("drop", "drop_all"):
            continue
        # drop(path, reason, ...) — reason is the second positional arg.
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) \
                and isinstance(node.args[1].value, str):
            used_reasons.add(node.args[1].value)
        for kw in node.keywords:
            if kw.arg == "reason" and isinstance(kw.value, ast.Constant):
                used_reasons.add(kw.value.value)

unknown = sorted(used_reasons - set(DROP_REASONS))
check(f"every recorded drop reason is declared (undeclared: {unknown})", not unknown)
check("the vocabulary is actually used", len(used_reasons) >= 5)
check("every declared reason has a human sentence",
      all(isinstance(v, str) and v for v in DROP_REASONS.values()))

# ── (b) the tally reconciles ─────────────────────────────────────────────────
led = Ledger("stage", ["a", "b", "c", "d", "e"])
led.drop("b", "no_fit", "l_p None")
led.drop("c", "no_stored_segments")
led.drop("c", "no_fit")          # same file, second reason
led.drop("d", "no_force")

check("n_asked = n_kept + n_dropped", led.n_asked == led.n_kept + led.n_dropped)
check("a file dropped twice counts once", led.n_dropped == 3)
check("by_reason sums to exactly n_dropped",
      sum(led.by_reason().values()) == led.n_dropped)
check("kept() is asked minus dropped, in order", led.kept() == ["a", "e"])
check("summary names the stage's own verb", "plotted 2" in led.summary("plotted"))
check("a file's full journey keeps BOTH its reasons", len(led.journey("c")) == 2)
check("a surviving file has an empty journey", led.journey("a") == [])
check("the manifest carries asked as well as kept",
      led.manifest()["n_asked"] == 5 and led.manifest()["n_kept"] == 2)

# A path is the ledger's identity for one curve. Repeated input entries must
# not make list-based asked/kept counts disagree with set-based drop counts.
duplicate = Ledger("stage", ["a", "a", "b", "", "b"])
check("duplicate and empty input paths do not create phantom curves",
      duplicate.n_asked == 2 and duplicate.kept() == ["a", "b"])
duplicate.drop("a", "no_fit")
check("duplicate input paths preserve the tally invariant after a drop",
      duplicate.n_asked == duplicate.n_kept + duplicate.n_dropped and
      duplicate.kept() == ["b"])

empty = Ledger("stage", [])
check("an empty ledger is not a division by zero",
      empty.n_asked == 0 and empty.n_kept == 0 and empty.summary() and
      empty.breakdown_lines() == [])

# Recording a second reason must never resurrect a curve.
led.drop("b", "no_length")
check("a second reason cannot un-drop a file", "b" not in led.kept())

# ── (c) the journey survives a stage boundary ────────────────────────────────
upstream = Ledger("Explore Events population", ["a", "b", "c", "d", "e"])
upstream.drop("e", "not_in_population", "non-hit")
downstream = Ledger("2DH build", upstream.kept())
downstream.drop("d", "no_histogram")
downstream.absorb(upstream)

check("a downstream stage can explain an upstream drop",
      [d.reason for d in downstream.journey("e")] == ["not_in_population"])
check("absorbing does not inflate the downstream stage's own asked count",
      downstream.n_asked == 4)
check("absorbing an upstream drop of a file this stage never saw doesn't "
      "count against this stage", downstream.n_dropped == 1)
check("the absorbed drop names the stage that made it",
      downstream.journey("e")[0].stage == "Explore Events population")

# ── (e) a fit-free align mode does not require a WLC fit ─────────────────────
from smfs_catalog.physical_2dh_window import Physical2DHWindow
from smfs_catalog.normalized_2dh_window import Normalized2DHWindow
from smfs_catalog.event_processor import PHYS_ALIGN_ANCHORS

fit_dependent = Physical2DHWindow._FIT_DEPENDENT_ALIGN_MODES
check("every fit-dependent mode is a real anchor",
      fit_dependent <= set(PHYS_ALIGN_ANCHORS))
check("onset — the DEFAULT — does not require a fit", "onset" not in fit_dependent)
check("rupture does not require a fit", "rupture" not in fit_dependent)
check("snap-off does not require a fit", "snapoff" not in fit_dependent)
check("F* does require a fit", "fstar" in fit_dependent)
check("l_c does require a fit", "lc" in fit_dependent)
# The base class's answer is True, and normalized inherits it unchanged —
# there is no x̃ without an l_c to divide by, so that curve genuinely cannot
# be placed.  Physical is the only window with an opinion of its own.
from smfs_catalog.base_2dh_window import _TwoDHWindowBase
check("the base class requires a fit by default",
      _TwoDHWindowBase._requires_wlc_fit(object()) is True)
check("the normalized 2DH does not override it — it divides x by l_c",
      "_requires_wlc_fit" not in vars(Normalized2DHWindow))
check("the physical 2DH is the one that overrides it",
      "_requires_wlc_fit" in vars(Physical2DHWindow))

# The build loop must ASK, not assume: a bare unconditional l_p/l_c guard in
# sync_from_event_summary is exactly what dropped those curves before.
sync_src = (PKG / "base_2dh_window.py").read_text()
sync_body = sync_src[sync_src.index("def sync_from_event_summary"):]
sync_body = sync_body[:sync_body.index("def _load_or_compute")]
check("the 2DH build loop consults _requires_wlc_fit",
      "_requires_wlc_fit()" in sync_body)
check("the 2DH build loop records a drop instead of a bare continue",
      sync_body.count("led.drop") >= 4)

# ── (f) provenance carries the tally ─────────────────────────────────────────
for mod, needle in (
    ("base_2dh_window.py",      '"drops"'),
    ("event_summary_window.py", '"population_drops"'),
):
    src = (PKG / mod).read_text()
    prov = src[src.index("def export_provenance"):]
    prov = prov[:prov.index("\n    def ", 1)]
    check(f"{mod}'s export_provenance carries the drop tally", needle in prov)

# ── (h) freshness is derived, never stored ───────────────────────────────────
schema_src = (PKG / "db.py").read_text()
for banned in ("freshness ", "is_stale", "params_fresh"):
    check(f"no stored column named like {banned.strip()!r}",
          f"ALTER TABLE analysis_queue ADD COLUMN {banned}" not in schema_src)
check("queue_freshness is a SELECT, never a write",
      "def queue_freshness" in schema_src and
      "UPDATE analysis_queue SET" not in
      schema_src[schema_src.index("def queue_freshness"):
                 schema_src.index("def queue_freshness") + 3000])


# ── (g) freshness matches the fast path, on a real DB ────────────────────────
tmp = tempfile.mkdtemp(prefix="ledger_")
DB = os.path.join(tmp, "test.sqlite")
_db.initialise(DB)

PATHS = [
    _db.normalize_path(f"/tank/testdata/led/Image{i:04d}.ibw")
    for i in (1, 2, 3)
]
conn = _db.get_connection(DB)
with conn:
    for p in PATHS:
        conn.execute(
            "INSERT INTO files (path, filename, first_seen, last_seen,"
            " experimentalist) VALUES (?, ?, datetime('now'), datetime('now'), 'led')",
            (p, os.path.basename(p)))
conn.close()

IDS = [_db.get_file_id(p, DB) for p in PATHS]
_db.enqueue_files(IDS, DB)

PARAMS, CODE = '{"a": 1}', cache_version() or "test-build"
# file 0: stored under today's signature.  file 1: stored under an older
# commit.  file 2: never analysed.
_db.write_analysis_result(IDS[0], "event", 1.0, PARAMS, CODE, DB)
_db.write_analysis_result(IDS[1], "event", 1.0, PARAMS, "OLDBUILD", DB)

fresh = _db.queue_freshness(PARAMS, CODE, DB)
check("a verdict under today's params and code reads fresh",
      fresh[IDS[0]] == "fresh")
check("a verdict from another commit reads stale — it WILL be recomputed",
      fresh[IDS[1]] == "stale")
check("a file with nothing stored reads new", fresh[IDS[2]] == "new")
check("every queued file gets an answer", set(fresh) == set(IDS))

# Changing the parameters alone must move a row to stale — this is the whole
# point of #96: a fully populated row that says "pending" tells you nothing
# about whether it costs anything.
fresh2 = _db.queue_freshness('{"a": 2}', CODE, DB)
check("changing a parameter makes a fresh row stale", fresh2[IDS[0]] == "stale")
check("changing a parameter cannot make an unanalysed row anything but new",
      fresh2[IDS[2]] == "new")

# The classes are exactly what the dashboard knows how to label.
from smfs_catalog import dashboard_window as _dash
check("every freshness class has a Status-column label",
      set(_db.QUEUE_FRESHNESS) == set(_dash._FRESHNESS_LABEL))
check("no freshness class is labelled 'pending' — the word that meant two "
      "things (#96)",
      "pending" not in {v.lower() for v in _dash._FRESHNESS_LABEL.values()})

dash_src = (PKG / "dashboard_window.py").read_text()

# The Status header's drill-down must report the same classes as the column.
# Feeding CategoricalStatsWindow the raw analysis_queue.status would put
# "pending" — the word that meant two things — straight back on screen, in the
# very window meant to explain the classes.
check("Status is not drilled down from the raw queue column",
      1 not in _dash.DashboardWindow._CATEGORICAL_COLS)
check("one function decides the Status class for cell and drill-down alike",
      "def _status_class" in dash_src and dash_src.count("_status_class(") >= 3)

# The ETA must not be reachable from "total minus done" any more: that formula
# subtracted every file finished this session, while the worker revisits them.
check("the ETA no longer prices the queue as total-minus-done",
      "self._queue_total - len(self._done_ids)" not in dash_src)
check("the ETA counts from the playhead in the direction of travel",
      "def _files_ahead" in dash_src and "_files_ahead()" in dash_src)
check("the ETA prices cached and uncached files separately",
      "def _mean_cost" in dash_src and "_mean_cost(True)" in dash_src
      and "_mean_cost(False)" in dash_src)

# ── The ETA arithmetic itself, exercised rather than grepped ─────────────────
# A whole DashboardWindow is far too heavy to build here, and the bug was
# never in the widgets — it was in which files get counted and at what price.
# So bind the real methods to a stand-in holding only the state they read: if
# the arithmetic regresses, this fails, and it fails on numbers rather than on
# the presence of a string.
from collections import deque

class _Stub:
    _files_ahead   = _dash.DashboardWindow._files_ahead
    _mean_cost     = _dash.DashboardWindow._mean_cost
    _eta_text      = _dash.DashboardWindow._eta_text
    # _fmt_eta is a staticmethod on the real class; taken off it we get the
    # bare function, so it has to be re-wrapped or `self` would be passed in
    # as the seconds argument.
    _fmt_eta       = staticmethod(_dash.DashboardWindow._fmt_eta)

stub = _Stub()
stub._queue_row_ids   = [10, 11, 12, 13, 14]
stub._queue_id_to_row = {fid: i for i, fid in enumerate(stub._queue_row_ids)}
stub._freshness       = {10: "fresh", 11: "fresh", 12: "stale",
                         13: "new", 14: "fresh"}
stub._current_direction  = +1
stub._current_playhead_id = 11          # row 1 of 5
# 2 s per real analysis, 0.01 s per cache hit.
stub._cost_samples = deque([(2.0, False), (2.0, False), (0.01, True), (0.01, True)])

check("forward counts only what is ahead of the playhead",
      stub._files_ahead() == [12, 13, 14])
stub._current_direction = -1
check("reverse counts what is behind it, nearest first",
      stub._files_ahead() == [10])
stub._current_direction = +1

check("a file already analysed this session is still counted — it is still "
      "visited", 14 in stub._files_ahead())

stub._current_playhead_id = 14
check("at the queue edge there is nothing left to estimate",
      stub._files_ahead() == [] and stub._eta_text() is None)

stub._current_playhead_id = None
check("with no playhead the whole queue lies ahead",
      stub._files_ahead() == stub._queue_row_ids)

stub._current_playhead_id = 11
eta = stub._eta_text()
# 2 files needing real work (stale + new) at 2.0 s, 1 cache hit at 0.01 s.
check(f"the ETA prices the work, not the visits ({eta})",
      eta is not None and "2 to analyse" in eta and "1 cached" in eta)
check("a mostly-cached queue is not priced at the analysis rate",
      "4.0s" in eta or "4s" in eta or "4.01s" in eta)

# The old formula's failure mode, stated as a number: blending the two costs
# into one mean would price these 3 files at 3 × 1.005 s ≈ 3.0 s, and a queue
# that is 90% cached at wildly more.  Prove the split is doing real work.
stub._freshness = {fid: "fresh" for fid in stub._queue_row_ids}
stub._freshness[12] = "stale"
eta_mostly_cached = stub._eta_text()
check(f"a queue that is mostly cache hits reports a short ETA "
      f"({eta_mostly_cached})",
      "2 cached" in eta_mostly_cached and "1 to analyse" in eta_mostly_cached)

# One class never timed → a floor, not a guess.
stub._cost_samples = deque([(2.0, False)])
stub._freshness = {10: "fresh", 11: "fresh", 12: "stale", 13: "fresh", 14: "fresh"}
check("an unpriced class makes the estimate a floor, not a fabrication",
      stub._eta_text().startswith("ETA ≥"))

stub._cost_samples = deque()
check("with nothing measured there is no ETA at all",
      stub._eta_text() is None)

# ── (d) gate membership and plottability are separate questions ──────────────
# On a real window, because this is exactly where the two used to be folded
# into one silent answer.
import numpy as np
from PyQt6.QtWidgets import QApplication
_app = QApplication.instance() or QApplication([])
from smfs_catalog.event_summary_window import EventSummaryWindow

ev = EventSummaryWindow([{"path": p} for p in PATHS], DB)
ev._segment_select = "ultimate"
# file 0 fully plottable; file 1 has force but no length; file 2 has neither.
ev._force_arr[:]  = [10.0, 20.0, np.nan]
ev._length_arr[:] = [50.0, np.nan, np.nan]
ev._rebuild()

pled = ev.population_ledger("hit")
reasons = {d.path: d.reason for d in pled.drops()}
check("a curve with no fit at all is reported as no_fit",
      reasons.get(PATHS[2]) == "no_fit")
check("a curve missing only the length says so, not 'no fit'",
      reasons.get(PATHS[1]) == "no_length")
check("population_paths is exactly the ledger's survivors",
      ev.population_paths("hit") == pled.kept())

# With no criteria checked the gate passes everything, so every drop here must
# be a plottability drop — never a membership one.  That is the split.
check("nothing is dropped for membership when the gate passes everything",
      "not_in_population" not in set(reasons.values()))

# The other population is the complement of the SAME gate answer, and its
# drops must name membership rather than silently vanishing.
nled = ev.population_ledger("non_hit")
check("the complement population reports membership drops explicitly",
      all(d.reason == "not_in_population" for d in nled.drops()))
check("the two populations ask the same question of the same cohort",
      nled.n_asked == pled.n_asked == len(PATHS))

led_plot = ev._plottability_ledger()
check("the plottability tally explains the title-vs-stats gap (#117's 307/306)",
      led_plot.n_asked == len(PATHS) and led_plot.n_dropped == 2)
check("the manifest carries the tally out of the app",
      ev.export_provenance()["population_drops"]["n_asked"] == len(PATHS))
check("the fit windows' caption names what it was drawn from",
      "of 3 events" in ev._provenance_caption(1))

# Hidden 2DH windows are removed before live Event Summary synchronization.
from PyQt6.QtWidgets import QWidget
hidden_2dh = QWidget()
ev._2dh_wins = [hidden_2dh]
ev._prepopulate()
check("a closed 2DH is removed from live synchronization",
      hidden_2dh not in ev._2dh_wins)

# Statistical fits are snapshots. Upstream changes label an open result; the
# next button press explicitly recalculates it.
fit_snapshot = QWidget()
fit_snapshot.setWindowTitle("Fit 2D GMM")
fit_snapshot._event_summary_revision = ev._data_revision
fit_snapshot.show()
_app.processEvents()
ev._fit_wins = {"test": fit_snapshot}
ev._prepopulate()
check("a no-op worker refresh does not stale a fit snapshot",
      not fit_snapshot.windowTitle().endswith("outdated snapshot"))
ev._data_revision += 1
ev._mark_fit_windows_stale()
check("an open fit is marked stale rather than recomputed",
      fit_snapshot.windowTitle().endswith("outdated snapshot"))
fit_snapshot.close()

# set_results is a real public replacement path: parallel arrays must always
# match the new cohort, including when it changes size.
ev.set_results([{"path": PATHS[0]}])
check("set_results resizes every per-curve array",
      len(ev._results) == len(ev._force_arr) == len(ev._length_arr) == 1)

print()


# Every check above becomes its own named pytest case.  Must be last:
# pytest_cases reads what the module body recorded.
test_check = checkstyle.pytest_cases(check)
