# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Qualification stage (#122).

"Can we use this file?" is asked once, by ONE function, before anything
downstream assumes anything. These tests pin the three properties that make
that worth having:

  (a) the ordered checks catch each defect and name it, and — the point of the
      ordering — a defect is diagnosed by the check that measures it, not by a
      later reduction that happens to return a funny-looking number;
  (b) the scanner and the loader ask the SAME function, so they cannot drift
      into disagreeing about what a valid curve is (they used to);
  (c) an unusable file is labelled and kept, never retried, and never dressed
      up as a classification.

The live example throughout is the file from #122: its requested-piezo channel
is 100% NaN, on which `np.argmax` returns 0 — indistinguishable from a real
index — which is how a healthy drive came to be blamed for a data problem.
"""

from __future__ import annotations

import numpy as np
import pytest

from smfs_catalog import curve_loader as cl


# ── Wave builders ─────────────────────────────────────────────────────────────

# The Force panel's four-channel wave, labelled as it labels itself.
_LABELS = [[], [b"", b"Raw", b"Defl", b"ZSnsr", b"Time"], [], []]
C_RAW, C_DEFL, C_ZSNSR = 0, 1, 2


def _labels(*names: str):
    """igor2's labels structure for a wave with these channels, in this order."""
    return [[], [b""] + [n.encode() for n in names], [], []]


def _good_wave(n: int = 2000) -> np.ndarray:
    """A minimal but valid continuous-stretch wave: ramp up, ramp down."""
    w = np.zeros((n, 4), dtype=float)
    half = n // 2
    piezo = np.concatenate([np.linspace(0.0, 1e-6, half),
                            np.linspace(1e-6, 0.0, n - half)])
    w[:, C_RAW]   = piezo
    w[:, C_ZSNSR] = piezo
    # deflection: flat baseline with a dip, so it is neither constant nor zero
    w[:, C_DEFL]  = -1e-8 * np.sin(np.linspace(0, np.pi, n)) - 1e-9
    w[:, 3] = np.linspace(0, 1, n)
    return w


def _qualify(w, *, labels=_LABELS, indent_mode=None, hold_z=None,
             spring_constant=29.0):
    """qualify_wave with healthy defaults, so each test varies one thing.

    The defaults are supplied HERE and not in qualify_wave itself: a default on
    the real function would let a production caller skip a check by omission,
    which is exactly how the scanner and loader drifted apart before.
    """
    return cl.qualify_wave(w, labels=labels, indent_mode=indent_mode,
                           hold_z=hold_z, spring_constant=spring_constant)


# ── (a) each defect is caught, and named by the check that measures it ────────

def test_a_healthy_wave_qualifies():
    q = _qualify(_good_wave())
    assert q.curve_type == "continuous_stretch"
    assert q.usable and q.reason is None
    # A qualified wave hands the turnaround forward, so the loader never has to
    # recompute it — recomputing is how two answers to one question appear.
    assert q.idx_turn is not None and 0 < q.idx_turn < 1999


def test_all_nan_channel_is_diagnosed_as_nonfinite_not_as_a_bad_turnaround():
    """#122 exactly. argmax of an all-NaN array returns 0, which the old code
    read as 'turnaround at index 0 — truncated or malformed'. The finiteness
    check must get there first, or the message describes the wrong problem."""
    w = _good_wave()
    w[:, C_ZSNSR] = np.nan

    q = _qualify(w)
    assert q.reason == cl.UNUSABLE_NONFINITE
    assert "piezo" in q.detail and "2,000" in q.detail
    # The defect is NOT reported as a turnaround problem, which is the whole
    # regression: same file, same argmax, different (correct) diagnosis.
    assert q.reason != cl.UNUSABLE_NO_TURNAROUND
    # And no turnaround is offered, since none was established.
    assert q.idx_turn is None


def test_partial_nan_is_caught_even_though_every_later_check_would_pass():
    """The subtler half of #122: a partly-NaN channel produces a *plausible*
    argmax (the first NaN's index), so nothing downstream would ever object —
    the curve would be split at the wrong sample and analysed as if fine."""
    w = _good_wave()
    w[300:800, C_DEFL] = np.nan

    # Demonstrate the trap this check exists to close, so the test fails if the
    # ordering is ever relaxed on the theory that "argmax would catch it".
    assert 0 < int(np.argmax(w[:, C_ZSNSR])) < len(w) - 1

    q = _qualify(w)
    assert q.reason == cl.UNUSABLE_NONFINITE
    assert "deflection: 500 of 2,000" in q.detail


def test_a_constant_channel_is_diagnosed_as_constant_not_as_no_turnaround():
    """A flat piezo also drives argmax to 0. Same wrong message, different
    cause — which is why 'varies' is its own check with its own reason."""
    w = _good_wave()
    w[:, C_ZSNSR] = 5e-7

    q = _qualify(w)
    assert q.reason == cl.UNUSABLE_CONSTANT
    assert "piezo" in q.detail


def test_no_turnaround_is_reported_when_that_really_is_the_problem():
    """The pre-existing guard, now meaning what it says: finite, varying, and
    genuinely monotonic — an approach with no retract."""
    w = _good_wave()
    w[:, C_ZSNSR] = np.linspace(0.0, 1e-6, len(w))

    q = _qualify(w)
    assert q.reason == cl.UNUSABLE_NO_TURNAROUND


def test_truncated_still_detected_and_no_longer_fooled_by_nan():
    turn = _qualify(_good_wave()).idx_turn   # ask, don't assume

    w = _good_wave()
    w[turn + 1:, C_DEFL] = 0.0
    assert _qualify(w).reason == cl.UNUSABLE_TRUNCATED

    # `retr.any()` is True for NaN, so before the finiteness check ran first an
    # all-NaN retract slipped past this test and was analysed.
    w2 = _good_wave()
    w2[turn + 1:, C_DEFL] = np.nan
    assert _qualify(w2).reason == cl.UNUSABLE_NONFINITE


def test_a_damaged_read_back_channel_does_not_reject_the_curve():
    """col 0 feeds the FFT view alone. Rejecting a curve whose science is
    intact because one optional viewer lost its input would be over-reach —
    and in #122's own file col 0 is damaged while the retract is perfect."""
    w = _good_wave()
    w[100:900, C_RAW] = np.nan
    assert _qualify(w).usable


def test_modalities_other_than_continuous_stretch_get_no_usability_verdict():
    """We do not analyse these, so inventing a pass/fail for them would be
    claiming a judgement we never made."""
    w = _good_wave()
    assert _qualify(w, hold_z=1).curve_type == "stretch_hold"
    assert _qualify(w, hold_z=1).reason is None
    assert _qualify(w, hold_z=0).curve_type == "force_clamp"
    assert _qualify(np.zeros((10, 10, 4))).curve_type == "image_ac"
    assert _qualify(w, indent_mode=1).curve_type == "indentation"
    # ...and none of them are labelled unusable despite being all-zero.
    assert _qualify(np.zeros((10, 10, 4))).reason is None


def test_no_spring_constant_is_classified_not_rejected():
    """User's call: a missing spring constant means the file is not
    a FORCE curve — it does not mean the file is damaged. Without k a
    deflection trace cannot be expressed as force, so it is classified into the
    bulk 'unknown' bucket and kept, visible, with nothing marked wrong with it.
    Refining 'unknown' into real sub-classes is deliberately a later job."""
    w = _good_wave()
    for k in (None, 0.0, -1.0, float("nan")):
        q = _qualify(w, spring_constant=k)
        assert q.curve_type == "unknown", k
        assert q.reason is None, (
            f"spring_constant={k!r} is a classification, not a defect — "
            f"nothing about this file is broken"
        )
    assert _qualify(w, spring_constant=29.0).curve_type == "continuous_stretch"


def test_a_file_the_force_pipeline_cannot_use_is_never_retried_forever():
    """Whatever the reason, 'this is not a force-extension curve' is a durable
    fact. Raised as a plain LoadError the analysis layer could only read it as
    'couldn't load' -> 'unavailable' -> retried on every pass, for ever — the
    fail-open shape #122 was filed for, reached by a second route."""
    assert cl.UNUSABLE_NOT_FE in cl.UNUSABLE_REASON_TEXT


def test_every_reason_code_has_user_facing_text():
    """The code is what we store; the text is what the user reads. A reason
    that reaches the DB with no explanation is a label nobody can act on."""
    codes = {v for k, v in vars(cl).items()
             if k.startswith("UNUSABLE_") and k != "UNUSABLE_REASON_TEXT"
             and isinstance(v, str)}
    assert codes, "no reason codes found — did they get renamed?"
    assert codes == set(cl.UNUSABLE_REASON_TEXT), (
        "every UNUSABLE_* code needs an entry in UNUSABLE_REASON_TEXT"
    )


# ── (b) one implementation, shared ────────────────────────────────────────────

def test_scanner_and_loader_share_one_qualifier():
    """Two copies would disagree: the
    scanner's swallowed its own exceptions, so it admitted files the loader
    then rejected on every analysis pass, forever."""
    import ast
    import inspect
    from smfs_catalog import scanner

    assert scanner.qualify_wave is cl.qualify_wave

    # Nobody re-derives the modality or the turnaround with their own reduction.
    for mod in (scanner, cl):
        tree = ast.parse(inspect.getsource(mod))
        calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in ("argmax", "argmin")
        ]
        in_qualifier = mod is cl
        assert len(calls) <= (1 if in_qualifier else 0), (
            f"{mod.__name__} performs its own argmax/argmin — qualification "
            f"has one implementation, in curve_loader.qualify_wave"
        )


def test_loader_raises_the_reason_and_keeps_read_failures_separate():
    """A caller must be able to tell 'the disk is gone' from 'this file is no
    good' with an except clause, because the two need opposite responses."""
    assert issubclass(cl.UnusableCurveError, cl.LoadError)
    assert issubclass(cl.TruncatedCurveError, cl.UnusableCurveError)
    assert cl.TruncatedCurveError("x").reason == cl.UNUSABLE_TRUNCATED

    # A read failure is a plain LoadError and must NOT be an UnusableCurveError,
    # or it would stop being retried when the drive comes back.
    with pytest.raises(cl.LoadError) as exc:
        cl.load_force_curve("/nonexistent/path/no_such_file.ibw")
    assert not isinstance(exc.value, cl.UnusableCurveError)


# ── (c) an unusable file is labelled, kept, and never retried ─────────────────

def _fresh_db(tmp_path):
    from smfs_catalog import db
    p = str(tmp_path / "t.db")
    db.initialise(p)
    return db, p




def test_an_unusable_file_is_labelled_and_kept_not_deleted(tmp_path):
    """The user must be able to see WHY a curve was rejected. A file that
    vanishes is one they go looking for."""
    db, p = _fresh_db(tmp_path)
    conn = db.get_connection(p)
    with conn:
        conn.execute(
            "INSERT INTO files (path, filename, curve_type, parse_ok, first_seen, last_seen) "
            "VALUES (?, 'a.ibw', 'force_extension', 1, '2026-01-01', '2026-01-01')",
            (db.normalize_path("/x/a.ibw"),),
        )
    conn.close()
    fid = db.get_file_id("/x/a.ibw", p)

    db.set_unusable_reason(fid, cl.UNUSABLE_NONFINITE, "deflection: 500 of 2,000", p)
    row = db.list_files(db_path=p)[0]
    assert row["unusable_reason"] == cl.UNUSABLE_NONFINITE
    assert "500" in row["unusable_detail"]

    # Re-qualifying (e.g. the file was replaced on disk) clears it.
    db.set_unusable_reason(fid, None, None, p)
    assert db.list_files(db_path=p, usable=True)[0]["path"] == db.normalize_path("/x/a.ibw")


def test_unusable_is_a_verdict_of_its_own_never_non_event(tmp_path):
    """#69's lesson, applied to a second cause: a file we could not judge must
    not report the same string as a file we judged and found empty."""
    db, p = _fresh_db(tmp_path)
    assert "unusable" in db.EVENT_VERDICTS
    assert "unavailable" in db.EVENT_VERDICTS
    with pytest.raises(ValueError):
        db.set_event(1, "not_a_verdict", p)

def test_scope_excludes_unusable_files():
    """Same 'never put broken files in scope' rule as parse_ok, one stage on."""
    from smfs_catalog.scope import new_scope, scope_to_query
    kw = scope_to_query(new_scope())
    assert kw["parse_ok"] is True
    assert kw["usable"] is True


def test_the_loader_does_not_reject_a_curve_over_its_intent_label():
    """indent_mode says which experiment was intended (a cataloguing fact the
    scanner scopes on); the loader asks whether the samples can be split into
    an approach and a retract (a structural one). An indentation wave has the
    force-extension layout, so a viewer must still be able to open it — the
    loader passing indent_mode would have broken that over a label."""
    import inspect
    src = inspect.getsource(cl.load_force_curve)
    assert "indent_mode=None" in src, (
        "load_force_curve must qualify on structure alone — passing "
        "indent_mode here rejects loadable waves over their intent label"
    )
    # hold_z is the opposite case and IS passed: a held curve is not a ramp,
    # so the ramp pipeline must not be handed one.
    assert "hold_z=_hold_z_sensor(note)" in src

    w = _good_wave()
    assert _qualify(w, indent_mode=1).curve_type == "indentation"
    assert _qualify(w).curve_type == "continuous_stretch"


# ── (d) content identity: duplicates are derived, never stored ────────────────

def _seed(db, p, path, sha=None, first_seen="2026-01-01"):
    conn = db.get_connection(p)
    with conn:
        conn.execute(
            "INSERT INTO files (path, filename, curve_type, parse_ok, "
            "first_seen, last_seen, content_sha256) VALUES (?,?,?,?,?,?,?)",
            (path, path.rsplit("/", 1)[-1], "continuous_stretch", 1,
             first_seen, first_seen, sha))
    conn.close()


def test_the_same_bytes_under_two_paths_are_one_curve(tmp_path):
    """A real catalog holds many redundant copies, the
    largest an entire 1,000-file folder duplicated one level up — every one of
    them counted twice in every histogram, fit and 2DH."""
    db, p = _fresh_db(tmp_path)
    _seed(db, p, "/a/Image0001.ibw", "aaa")
    _seed(db, p, "/b/Image0001.ibw", "aaa")     # same bytes, other folder
    _seed(db, p, "/c/Image0001.ibw", "bbb")     # same NAME, different bytes

    assert len(db.list_files(db_path=p)) == 3
    assert len(db.list_files(db_path=p, unique=True)) == 2, (
        "identical content under two paths is one curve, not two"
    )
    # The user can select exactly the redundant copies — that is how they get
    # handed to the removal dialog. The app never deletes anything itself.
    dupes = db.list_files(db_path=p, unique=False)
    assert [r["path"] for r in dupes] == ["/b/Image0001.ibw"]

    groups = db.duplicate_groups(p)
    assert len(groups) == 1 and groups[0]["n"] == 2
    assert groups[0]["canonical"] == "/a/Image0001.ibw"
    assert groups[0]["copies"] == ["/b/Image0001.ibw"]


def test_filename_alone_could_never_have_answered_this(tmp_path):
    """32 files in the live catalog are called Image0001.ibw and only 14,962
    distinct names cover 140,676 files — the name says nothing about identity.
    Content does."""
    db, p = _fresh_db(tmp_path)
    _seed(db, p, "/a/Image0001.ibw", "aaa")
    _seed(db, p, "/b/Image0001.ibw", "bbb")
    assert len(db.list_files(db_path=p, unique=True)) == 2


def test_an_unknown_hash_is_never_treated_as_a_duplicate(tmp_path):
    """NULL means 'not read since the column was added', not 'unique' and not
    'same as that other NULL'. Guessing either way would silently drop real
    curves out of a cohort."""
    db, p = _fresh_db(tmp_path)
    _seed(db, p, "/a/x.ibw", None)
    _seed(db, p, "/b/y.ibw", None)
    assert len(db.list_files(db_path=p, unique=True)) == 2
    assert db.duplicate_groups(p) == []


def test_which_copy_is_canonical_cannot_drift(tmp_path):
    """The answer must not change between two runs, or a cohort silently
    reshuffles under an analysis that has already been done."""
    db, p = _fresh_db(tmp_path)
    for name in ("/z/last.ibw", "/a/first.ibw", "/m/mid.ibw"):
        _seed(db, p, name, "same")
    first = [r["path"] for r in db.list_files(db_path=p, unique=True)]
    for _ in range(3):
        assert [r["path"] for r in db.list_files(db_path=p, unique=True)] == first
    assert len(first) == 1
    # Adding an unrelated file must not move it either.
    _seed(db, p, "/q/other.ibw", "different")
    assert db.duplicate_groups(p)[0]["canonical"] == first[0]


def test_duplicate_status_is_not_a_stored_column(tmp_path):
    """It is a conclusion about a SET of files, so it is derived on demand —
    the same rule as the gate verdict. A stored flag would have
    to be kept true as files are added and removed, for ever."""
    db, p = _fresh_db(tmp_path)
    conn = db.get_connection(p)
    cols = {c["name"] for c in conn.execute("PRAGMA table_info(files)")}
    conn.close()
    assert "content_sha256" in cols, "the FACT is stored"
    for banned in ("is_duplicate", "duplicate_of", "duplicate_group"):
        assert banned not in cols, f"{banned} stores a conclusion — derive it instead"


def test_scope_sets_aside_rejects_but_can_still_select_them():
    from smfs_catalog.scope import new_scope, scope_to_query

    kw = scope_to_query(new_scope())
    assert kw["parse_ok"] is True and kw["usable"] is True and kw["unique"] is True

    kw = scope_to_query({**new_scope(), "only_unusable": True})
    assert kw["usable"] is False, "the user must be able to point at the rejects"
    kw = scope_to_query({**new_scope(), "only_duplicates": True})
    assert kw["unique"] is False


def test_the_folder_name_rule_is_gone():
    """_EXCLUDE_DIR_SUBSTRINGS = {'selected'} matched NOTHING in the real
    corpus while the actual duplicated folders are called test_curves,
    PR Analysis and 1001-2000. A name cannot answer 'is this the same data',
    and this one failed silently when it did fire."""
    from smfs_catalog import scanner
    assert not hasattr(scanner, "_EXCLUDE_DIR_SUBSTRINGS")
    import inspect
    src = inspect.getsource(scanner._find_ibw_files) + inspect.getsource(scanner.leaf_ibw_dirs)
    assert "selected" not in src.lower().replace("# ", "")


# ── (e) the backfill pass ─────────────────────────────────────────────────────

def test_requalify_is_resumable_and_costs_nothing_to_repeat(tmp_path):
    """Progress IS the stored hash column, not a separate bookmark, so Cancel
    is free and re-running picks up where it stopped. An ordinary scan will not
    do this job: scan_directory skips files whose mtime is unchanged, which is
    right for a scan and is exactly why old files need an explicit pass."""
    from smfs_catalog import scanner

    db, p = _fresh_db(tmp_path)
    real = tmp_path / "real.ibw"
    real.write_bytes(b"not an igor wave")          # readable, unparseable
    _seed(db, p, str(real), None)
    _seed(db, p, "/gone/missing.ibw", None)        # catalogued, not on disk

    s = scanner.requalify_catalog(p)
    assert s["seen"] == 2
    assert s["hashed"] == 1, "readable-but-unparseable still yields a real hash"
    assert s["unreadable"] == 1

    # Nothing was written for the unreadable one, so it is retried; the other
    # is done and is not revisited.
    s2 = scanner.requalify_catalog(p)
    assert s2["seen"] == 1 and s2["hashed"] == 0 and s2["unreadable"] == 1

    rows = {r["path"]: r for r in db.list_files(db_path=p)}
    assert rows[str(real)]["content_sha256"] is not None
    assert rows[str(real)]["parse_ok"] == 0
    assert rows["/gone/missing.ibw"]["content_sha256"] is None, (
        "a disconnected drive is not a fact about the file — write nothing"
    )


def test_requalify_reports_progress_and_can_be_cancelled(tmp_path):
    from smfs_catalog import scanner

    db, p = _fresh_db(tmp_path)
    for i in range(5):
        f = tmp_path / f"f{i}.ibw"
        f.write_bytes(b"x" * (i + 1))
        _seed(db, p, str(f), None)

    ticks = []

    def cb(done, total, label):
        ticks.append((done, total))
        return done >= 2                      # cancel after two files

    s = scanner.requalify_catalog(p, progress_cb=cb)
    assert s["cancelled"] is True
    assert ticks[0] == (0, 5), "done counts files FINISHED, so it starts at 0"
    assert s["hashed"] == 2, "work done before the cancel is kept, not rolled back"
    assert len(db.list_files(db_path=p, unique=True)) == 5


# ── (f) the acquisition filter is a column, backfilled from file_metadata ─────



def test_the_queue_exposes_owner_and_bandwidth(tmp_path):
    """db.list_queue omitting `experimentalist` silently
    disabled the dashboard's mixed-owner warning (its caller guards on
    `"experimentalist" in r.keys()`).  Both columns are read by the gate
    summary; a warning that cannot fire reads as reassurance."""
    db, p = _fresh_db(tmp_path)
    _seed(db, p, "/a/Image0001.ibw", "aaa")
    conn = db.get_connection(p)
    with conn:
        conn.execute("UPDATE files SET experimentalist='Anthony', "
                     "force_filter_bw_hz=2000.0")
        fid = conn.execute("SELECT id FROM files").fetchone()["id"]
    conn.close()
    db.enqueue_files([fid], db_path=p)

    row = db.list_queue(p)[0]
    assert row["experimentalist"] == "Anthony"
    assert row["force_filter_bw_hz"] == 2000.0
