# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Regression test: per-user analysis-setting isolation.

This bug has recurred three times ("changing Anastasiia's settings changes
Alexandre's"), so keep this runnable and green.  Root cause each time: a code
path that froze the profile key at window construction (or never passed one),
so every user collapsed onto the shared default bucket while the windows
loaded from the global last-writer-wins settings table.

The contract under test — ONE parameter set is in force, and the file at
POSITION ONE OF THE ANALYSIS QUEUE decides whose it is (this
supersedes the older "profile follows the curve on screen" rule, which was a
SECOND way of deciding the same thing and let one computation be assembled
from two people's profiles — see db.active_param_owner):
(a) each user's profile row holds their own values,
(b) widgets switch when the QUEUE changes whose set is in force — not when the
    playhead crosses to a curve owned by somebody else,
(b2) adding a file to the BOTTOM of a populated queue changes nothing, because
    row one is unchanged,
(c) changing one user's knobs never changes another's profile,
(d) partial profiles (e.g. only 2DH grid keys) are backfilled without
    clobbering existing keys,
(e) the parameter set is never mirrored into the catalog-wide settings table,
(f) a NEVER-BEFORE-SEEN experimentalist (zero profile row, not even a
    partial one) seeds from the shared __default__ bucket, not from
    whatever the previously-displayed owner left sitting in the settings
    table / widgets — `if profile:` guards around a
    `get_experimentalist_profile()` call that returns None for a genuinely
    missing row skipped the load entirely, so the seed-write right after it
    saved the PREVIOUS owner's stale values under the new person's name.

Run with the smfs-catalog env:
    python tests/test_profile_isolation.py
"""
import json
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
app = QApplication.instance() or QApplication(sys.argv)

from smfs_catalog import db as _db
from smfs_catalog.display_roi import ROIWindow
from smfs_catalog.decomposition_window import DecompositionWindow

tmp = tempfile.mkdtemp(prefix="profile_iso_")
DB = os.path.join(tmp, "test.sqlite")
_db.initialise(DB)

ALEX_FILE   = _db.normalize_path("/tank/testdata/alexandre/Image0001.ibw")
ANA_FILE    = _db.normalize_path("/tank/testdata/anastasiia/Image0002.ibw")
CARLOS_FILE = _db.normalize_path("/tank/testdata/carlos/Image0003.ibw")
DANA_FILE   = _db.normalize_path("/tank/testdata/dana/Image0004.ibw")

conn = _db.get_connection(DB)
with conn:
    # experimentalist is file-level (#110), and a file's folder is read off
    # its own path — there is no directory registry to seed.
    for path, who in [
        (ALEX_FILE, "alexandre"), (ANA_FILE, "anastasiia"),
        (CARLOS_FILE, "carlos"), (DANA_FILE, "dana"),
    ]:
        conn.execute(
            "INSERT INTO files (path, filename, first_seen, last_seen, experimentalist)"
            " VALUES (?, ?, datetime('now'), datetime('now'), ?)",
            (path, os.path.basename(path), who))
conn.close()

# carlos mirrors a real historic DB state: a profile with ONLY 2DH grid keys
# (written by the ensemble windows), no analysis knobs at all.
_db.set_experimentalist_profile(
    "carlos", json.dumps({"phys_x_bins": 256, "phys_f_bins": 256}), DB)

# dana has NO row at all — the never-tuned-before case. __default__ gets a
# value distinct from every real user's, so a wrong fallback (or no fallback)
# is unambiguous in the assertions below.
_db.set_experimentalist_profile(
    _db.DEFAULT_EXPERIMENTALIST,
    json.dumps({"roi_threshold_nm_per_nm": 0.123, "roi_window_pts": 42}), DB)

def use(path):
    """Put `path` at position one of the queue — the ONE way the parameter
    set in force is chosen.  Windows re-read it via _sync_profile_owner."""
    _db.clear_analysis_queue(DB)
    _db.enqueue_files([_db.get_file_id(path, DB)], DB)


# One shared idiom for these procedural guards — see checkstyle.py for why
# `sys.exit(1)` at the bottom of a file was aborting the whole pytest run.
import checkstyle                                          # noqa: E402

check = checkstyle.CheckRunner()


# owner resolution
check("owner(alex file) == alexandre",
      _db.get_experimentalist_for_file(ALEX_FILE, DB) == "alexandre")
check("owner(ana file) == anastasiia",
      _db.get_experimentalist_for_file(ANA_FILE, DB) == "anastasiia")

# ── ROI window: knob isolation across owner changes ──────────────────────────
roi = ROIWindow(DB)                      # dashboard-style: NO experimentalist arg
use(ALEX_FILE)                           # his file is row one of the queue
roi._sync_profile_owner(None)
check("ROI owner is the queue's row-one owner (alexandre)",
      roi._experimentalist == "alexandre")

roi._preview_threshold(0.500)            # Alexandre tunes d1 threshold —
roi._commit_threshold()                  # preview+commit: matches a spinbox
                                          # valueChanged tick + editingFinished
p_alex = _db.get_experimentalist_profile("alexandre", DB)
check("Alexandre's profile stores his threshold 0.5",
      p_alex and abs(p_alex["roi_threshold_nm_per_nm"] - 0.5) < 1e-9)

use(ANA_FILE)                            # queue now says Anastasiia
roi._sync_profile_owner(None)
check("ROI owner follows the queue to anastasiia",
      roi._experimentalist == "anastasiia")
roi._preview_threshold(0.900); roi._commit_threshold()   # Anastasiia tunes HER threshold
roi._preview_window(101);      roi._commit_window()       # and her window size

p_alex = _db.get_experimentalist_profile("alexandre", DB)
p_ana  = _db.get_experimentalist_profile("anastasiia", DB)
check("changing hers does NOT change his (threshold still 0.5)",
      abs(p_alex["roi_threshold_nm_per_nm"] - 0.5) < 1e-9)
check("her profile has her threshold 0.9",
      abs(p_ana["roi_threshold_nm_per_nm"] - 0.9) < 1e-9)
check("her profile has her window 101",
      abs(p_ana["roi_window_pts"] - 101) < 1e-9)

use(ALEX_FILE); roi._sync_profile_owner(None)     # queue back to Alexandre
check("widgets restored to HIS threshold 0.5",
      abs(roi._spin_threshold.value() - 0.5) < 1e-9)
# The parameter set lives in the owner's profile and NOWHERE else. It used to
# be mirrored into the catalog-wide `settings` table on every profile load,
# which made that table mean "whoever was displayed last" and let one
# experimentalist's numbers be applied to everybody's curves. That mirror is
# gone, so `settings` must NOT carry this key at all.
check("parameter is NOT mirrored into the catalog-wide settings table",
      _db.get_setting("roi_threshold_nm_per_nm", -999.0, DB) == -999.0)
check("HIS value is still readable from his own profile",
      abs(_db.get_experimentalist_profile("alexandre", DB)["roi_threshold_nm_per_nm"] - 0.5) < 1e-9)

use(ANA_FILE); roi._sync_profile_owner(None)      # and back to hers
check("widgets switch back to HER threshold 0.9",
      abs(roi._spin_threshold.value() - 0.9) < 1e-9)
check("her window spinbox restored to 101", roi._spin_window.value() == 101)

# save-on-open clobber is gone: reopening a window must not rewrite profiles
before = json.dumps(_db.get_experimentalist_profile("alexandre", DB), sort_keys=True)
roi2 = ROIWindow(DB)
after = json.dumps(_db.get_experimentalist_profile("alexandre", DB), sort_keys=True)
check("opening a new ROI window does not rewrite Alexandre's profile", before == after)

# ── Decomposition window: same contract ──────────────────────────────────────
dec = DecompositionWindow(DB)
use(ALEX_FILE); dec._sync_profile_owner(None)
dec._on_cutoff_slider(2)                 # Alexandre picks a cutoff
alex_cutoff = dec._cutoff_hz
use(ANA_FILE); dec._sync_profile_owner(None)
dec._on_cutoff_slider(5)                 # Anastasiia picks a different one
p_alex = _db.get_experimentalist_profile("alexandre", DB)
check("decomp: his cutoff survives her change",
      abs(p_alex["spectral_cutoff_hz"] - alex_cutoff) < 1e-9)
use(ALEX_FILE); dec._sync_profile_owner(None)
check("decomp: slider restored to HIS cutoff when the queue says him",
      abs(dec._cutoff_hz - alex_cutoff) < 1e-9)
check("decomp: parameter is NOT mirrored into the settings table",
      _db.get_setting("spectral_cutoff_hz", -999.0, DB) == -999.0)

# ── Partial profile (live-DB Alexandre case): backfill without clobber ──────
use(CARLOS_FILE); roi._sync_profile_owner(None)
p_carlos = _db.get_experimentalist_profile("carlos", DB)
check("partial profile: 2DH keys preserved after ROI sync",
      p_carlos["phys_x_bins"] == 256 and p_carlos["phys_f_bins"] == 256)
check("partial profile: ROI knobs backfilled on first sighting",
      "roi_threshold_nm_per_nm" in p_carlos and "roi_window_pts" in p_carlos)
p_ana = _db.get_experimentalist_profile("anastasiia", DB)
check("backfilling carlos did not touch her profile",
      abs(p_ana["roi_threshold_nm_per_nm"] - 0.9) < 1e-9)

# ── Queue-level: ONE parameter set in force, decided by the QUEUE ────────────
# The set no longer follows the curve on screen, and the worker no longer
# swaps it per file. It is the queue owner's set, derived from the queue
# itself (db.active_param_owner) so it cannot go stale, and adding anyone new
# re-decides it.
_db.clear_analysis_queue(DB)
_db.enqueue_files([_db.get_file_id(ANA_FILE, DB)], DB)
check("queue owner is anastasiia", _db.active_param_owner(DB) == "anastasiia")
check("params in force are HERS (0.9)",
      abs(_db.load_analysis_params(DB).roi_threshold_nm_per_nm - 0.9) < 1e-9)

_db.enqueue_files([_db.get_file_id(ALEX_FILE, DB)], DB)   # add to the BOTTOM
check("adding a file to the bottom does NOT re-decide the owner",
      _db.active_param_owner(DB) == "anastasiia")
check("params in force are still HERS (0.9)",
      abs(_db.load_analysis_params(DB).roi_threshold_nm_per_nm - 0.9) < 1e-9)

_db.clear_analysis_queue(DB)
check("empty queue answers Default - never nothing",
      _db.active_param_owner(DB) == _db.DEFAULT_EXPERIMENTALIST)

# ── Never-before-seen owner: must seed from __default__, not the previous
#    owner's leftovers ─────────────────────────────────────────────────────
use(ALEX_FILE); roi._sync_profile_owner(None)     # a REAL profile first,
                                          # so dana's sync has something wrong
                                          # to inherit if the fallback is broken
check("setup: widgets show his 0.5 before dana's first sighting",
      abs(roi._spin_threshold.value() - 0.5) < 1e-9)

use(DANA_FILE); roi._sync_profile_owner(None)     # dana has zero profile rows
check("dana: widgets seeded from __default__ (0.123), not his leftover 0.5",
      abs(roi._spin_threshold.value() - 0.123) < 1e-9)
check("dana: parameter is NOT mirrored into the settings table",
      _db.get_setting("roi_threshold_nm_per_nm", -999.0, DB) == -999.0)

p_dana = _db.get_experimentalist_profile("dana", DB)
check("dana: her newly-created profile row was seeded from __default__",
      p_dana is not None and abs(p_dana["roi_threshold_nm_per_nm"] - 0.123) < 1e-9)

p_alex = _db.get_experimentalist_profile("alexandre", DB)
check("dana's seeding did not touch Alexandre's profile",
      abs(p_alex["roi_threshold_nm_per_nm"] - 0.5) < 1e-9)

# same at the pipeline level: a queue of dana's files must never serve
# alexandre's numbers, no matter who was analysed or displayed before her.
_db.clear_analysis_queue(DB)
_db.enqueue_files([_db.get_file_id(ALEX_FILE, DB)], DB)
_db.enqueue_files([_db.get_file_id(DANA_FILE, DB)], DB)
check("row one decides, not the newest addition (alexandre is row one)",
      _db.active_param_owner(DB) == "alexandre")
check("params in force are row one's (0.5)",
      abs(_db.load_analysis_params(DB).roi_threshold_nm_per_nm - 0.5) < 1e-9)
use(DANA_FILE)   # now SHE is row one
check("dana at row one puts her set in force, not alexandre's",
      abs(_db.load_analysis_params(DB).roi_threshold_nm_per_nm - 0.5) > 1e-9)

# ── Lost-update race: two independent partial writes to the SAME profile
#    must never clobber each other ("old undone by new").
#    display_roi.py / decomposition_window.py / base_2dh_window.py /
#    export_utils.py each used to save their own subset of a profile by
#    reading the WHOLE stored blob, editing it in Python, and writing the
#    WHOLE blob back — a classic TOCTOU: whichever of two racing writers
#    (a GUI window vs. the analysis worker's QThread, or two open windows)
#    landed second, based on a now-stale read, silently discarded the
#    other's just-saved keys. db.merge_experimentalist_profile replaces
#    that with one atomic SQL statement (json_patch inside INSERT ... ON
#    CONFLICT), so this reproduces the exact shape of the race — two
#    merges into the same row, neither one re-reading the other's write in
#    between — and asserts BOTH sets of keys survive.
_db.set_experimentalist_profile(
    "race_target", json.dumps({"roi_window_pts": 10.0}), DB)
before_read = _db.get_experimentalist_profile("race_target", DB)   # "writer 1" reads
_ = _db.get_experimentalist_profile("race_target", DB)              # "writer 2" reads the SAME stale state
_db.merge_experimentalist_profile(
    "race_target", {"spectral_cutoff_hz": 2000.0}, DB)              # writer 1 saves ITS key only
_db.merge_experimentalist_profile(
    "race_target", {"roi_threshold_nm_per_nm": 0.7}, DB)            # writer 2 saves ITS key only,
                                                                      # without ever seeing writer 1's write
p_race = _db.get_experimentalist_profile("race_target", DB)
check("race: original key survived both concurrent partial writes",
      p_race is not None and abs(p_race["roi_window_pts"] - 10.0) < 1e-9)
check("race: writer 1's key was not discarded by writer 2's write",
      p_race is not None and abs(p_race["spectral_cutoff_hz"] - 2000.0) < 1e-9)
check("race: writer 2's key was not discarded (i.e. was actually saved)",
      p_race is not None and abs(p_race["roi_threshold_nm_per_nm"] - 0.7) < 1e-9)


# Every check above becomes its own named pytest case.  Must be last:
# pytest_cases reads what the module body recorded.
test_check = checkstyle.pytest_cases(check)
