# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/curve_loader.py
#
# Loads .ibw force-extension curves into a clean ForceCurve dataclass.
# Does NOT import from pysmfs — uses igor2 directly.
# Follows pysmfs conventions: deflection stored in metres, piezo in metres.

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from igor2.binarywave import load as load_ibw

from .db import normalize_path  # single source of truth for files.path identity

# Silence igor2's own logging.  On a truncated / non-force-curve .ibw, igor2's
# reshape-failure path logs `logger.error('could not reshape data ...', shape,
# data_b)` where data_b is the wave's RAW byte buffer — Python's last-resort
# handler then dumps that entire binary blob to stderr.  igor2 re-raises the
# exception anyway (we catch it and return "non_event"), so the log record carries
# no information we act on; muting it only stops the per-bad-file binary spew.
# Scoped to the "igor2" logger so genuine errors elsewhere still surface.
logging.getLogger("igor2").setLevel(logging.CRITICAL)


class LoadError(Exception):
    """
    Raised when an .ibw file cannot be loaded as a ForceCurve.

    Plain LoadError means the file could not be READ — an I/O failure, a
    disconnected drive, a corrupt container.  That is a statement about the
    environment, is assumed transient, and the analysis layer retries it.
    A file whose CONTENTS disqualify it raises UnusableCurveError below, which
    is a durable fact about the file and is never retried.  Keeping those two
    apart is the whole point: they need opposite responses, and conflating them
    is what sent a user to check a healthy hard drive (#122).
    """
    pass


class UnusableCurveError(LoadError):
    """
    Raised when a file was read fine but its contents cannot be analysed.

    Carries `.reason` — one of the UNUSABLE_* codes below — so the analysis
    layer can record WHY on the file rather than collapsing every content
    defect into one string.  A subclass of LoadError so callers that only care
    about "couldn't load" still catch it with one except.
    """

    def __init__(self, message: str, reason: str):
        super().__init__(message)
        self.reason = reason


class TruncatedCurveError(UnusableCurveError):
    """
    Aborted acquisition: the retract half is entirely zero because the buffer
    was never written.  Kept as its own class because callers already catch it
    by name; it is now one reason among several rather than the only one.
    """

    def __init__(self, message: str):
        super().__init__(message, UNUSABLE_TRUNCATED)


# ── Qualification ─────────────────────────────────────────────────────────────
# "Can we use this file?" is a question about the file, asked ONCE, at import,
# before anything downstream is allowed to assume anything.  It is deliberately
# separate from "what does this curve show?" — that is analysis, and it runs
# only on files that got through here.
#
# Reason codes.  These are stored in files.unusable_reason and shown to the
# user, so they are stable identifiers, not prose.
UNUSABLE_NONFINITE     = "nonfinite_data"     # NaN/inf in a channel we depend on
UNUSABLE_CONSTANT      = "constant_channel"   # a channel never changes value
UNUSABLE_NO_TURNAROUND = "no_turnaround"      # no approach→retract reversal
UNUSABLE_TRUNCATED     = "truncated"          # retract half never written
UNUSABLE_NOT_FE        = "not_force_extension" # a real record, of something else

# What each code means in one line, for tooltips and the queue table.
UNUSABLE_REASON_TEXT = {
    UNUSABLE_NONFINITE:     "a required channel contains NaN/inf samples",
    UNUSABLE_CONSTANT:      "a required channel never changes value",
    UNUSABLE_NO_TURNAROUND: "no approach→retract turnaround in the piezo ramp",
    UNUSABLE_TRUNCATED:     "retract half is all-zero (aborted acquisition)",
    UNUSABLE_NOT_FE:        "not a force-extension curve — see its Type",
}

# Channel names, as the wave writes them in its own dimension labels.
#   Defl    deflection (m) — every force the app reports
#   ZSnsr   measured Z sensor (m); Raw is the commanded position (m)
#   Time    seconds; saved by some panels, absent from others
CH_DEFL  = "Defl"
CH_RAW   = "Raw"
CH_ZSNSR = "ZSnsr"
CH_TIME  = "Time"

# Fallback order for a wave carrying no labels, which here is only ever an
# image or a bare spectrum.  Position is NOT a substitute for a label: waves
# with the same channel count exist in different orders (Raw,Defl,ZSnsr from
# the Force panel against Defl,Raw,Lateral from the Force Clamp panel), so a
# fixed index reads a different quantity depending on which panel wrote it.
_POSITIONAL = (CH_RAW, CH_DEFL, CH_ZSNSR, CH_TIME)


def channel_map(labels, n_cols: int) -> dict[str, int]:
    """
    Channel name -> column index for one wave.

    `labels` is igor2's ``wave["labels"]``; its second entry holds the column
    names preceded by the dimension's own (empty) label.
    """
    names: list[str] = []
    if len(labels) > 1 and len(labels[1]) > 1:
        names = [b.decode("latin-1", "replace") if isinstance(b, bytes) else str(b)
                 for b in labels[1][1:]]
    if len(names) != n_cols:
        names = list(_POSITIONAL[:n_cols])
    return {name: i for i, name in enumerate(names) if name}


def piezo_column(channels: dict[str, int]) -> int | None:
    """
    The column carrying tip-sample position: the measured sensor when the wave
    saved one, else the commanded position.  None when the wave has neither.
    """
    for name in (CH_ZSNSR, CH_RAW):
        if name in channels:
            return channels[name]
    return None


@dataclass(frozen=True)
class Qualification:
    """
    The verdict of the qualification stage for one wave.

    curve_type : what the file IS — the acquisition modality.  Independent of
                 whether it is usable, so an aborted force-extension curve is
                 still labelled 'force_extension' and not turned into a
                 modality of its own.
    reason     : None when usable; otherwise an UNUSABLE_* code.
    detail     : human-readable specifics ("deflection: 13,685 of 41,962
                 samples are NaN/inf"), for the tooltip and the error message.
    idx_turn   : the approach/retract turnaround, when one was established.
                 None whenever the file did not get that far — a caller must
                 never fall back to computing it itself.
    """
    curve_type: str
    reason:     str | None = None
    detail:     str | None = None
    idx_turn:   int | None = None

    @property
    def usable(self) -> bool:
        return self.reason is None


def _is_truncated(deflection: np.ndarray, idx_turn: int) -> bool:
    """
    True when the retract half of the deflection trace is entirely zero.

    This is the failure mode behind "force-extension" files with no retract
    (e.g. Image0003.ibw: 16,627 retract samples all exactly 0.0, versus zero
    exact-zero samples in a healthy file) — the acquisition aborted and the
    second-half buffer was never filled.

    Only meaningful on FINITE data, which is why qualify_wave runs the
    finiteness check before this one: `retr.any()` is True for an all-NaN
    array, so on a NaN trace this returns False and waves a dead curve through.
    """
    retr = deflection[idx_turn + 1:]
    return retr.size > 0 and not retr.any()


def _spring_constant(note: bytes) -> float | None:
    """
    Cantilever spring constant from the wave note, in pN/nm, or None if the
    note carries no usable one.

    THE one parser for this field — the scanner reads it for its own column
    and qualification reads it to decide whether the file is a force curve at
    all, and those two must agree by construction, not by two regexes that
    look similar.  (N/m and pN/nm are the same number, hence the ×1000.)
    """
    m = re.search(rb"SpringConstant: ?([0-9]*\.?[0-9]+)\r", note)
    if m is None:
        return None
    try:
        return float(m.group(1)) * 1000.0
    except (ValueError, IndexError):
        return None


def _hold_z_sensor(note: bytes) -> int | None:
    """
    ``FCPHoldZSensor`` from the wave note: 1 held Z, 0 held force, None when
    the Force Clamp panel did not write this note at all.

    The panel stamping its own keys is what makes the distinction a statement
    by the file rather than an inference from its shape.
    """
    m = re.search(rb"FCPHoldZSensor: ?([0-1])\r", note)
    return int(m.group(1)) if m else None


def _modality(
    wdata: np.ndarray,
    channels: dict[str, int],
    indent_mode: int | None,
    hold_z: int | None,
    spring_constant: float | None,
) -> str:
    """
    Which experiment is this?  Every panel in the AFM software stamps its own
    keys in the wave note, so the file states its own modality and nothing here
    infers one from how many channels the operator chose to save.  This asks
    what the file IS, never whether it is any good.

    **A missing spring constant is a classification, not a defect.**  Without k
    a deflection trace cannot be expressed as force, so whatever the file
    records, it is not a force curve — but it is still a real recording of
    something and gets catalogued, labelled and left visible like any other.
    It lands in the existing bulk `unknown` bucket rather than a new class of
    its own; refining `unknown` into meaningful sub-classes is deliberately a
    later job, and nothing is lost meanwhile because the empty
    `spring_constant_pn_nm` column says exactly why the file is there.
    """
    if wdata.ndim == 3:
        if wdata.shape[2] == 4:
            return "image_ac"
        if wdata.shape[2] == 3:
            return "image_contact"
        return "unknown"
    if wdata.ndim != 2:
        return "unknown"

    usable_k = (spring_constant is not None
                and np.isfinite(spring_constant) and spring_constant > 0)
    if not usable_k:
        return "unknown"
    if CH_DEFL not in channels or piezo_column(channels) is None:
        return "unknown"

    if hold_z is not None:
        return "stretch_hold" if hold_z else "force_clamp"
    if indent_mode == 1:
        return "indentation"
    return "continuous_stretch"


def qualify_wave(
    wdata: np.ndarray,
    *,
    labels,
    indent_mode:     int | None,
    hold_z:          int | None,
    spring_constant: float | None,
) -> Qualification:
    """
    Decide whether one already-read wave can be analysed, and say why not.

    Every keyword argument is REQUIRED and has no default, deliberately. A
    default would let a caller skip a check by omission and never know — which
    is precisely how the scanner and the loader came to disagree in the first
    place. `spring_constant=None` means "the wave note has no usable one",
    which is a rejection, not "don't check"; `hold_z=None` means "no Force
    Clamp panel wrote this note", which is what makes a ramp a ramp.

    THE one implementation.  The scanner calls it at import (where wData is
    already in memory, so it is free) and the loader calls it on every load;
    they cannot drift into disagreeing about what a valid curve is, which they
    previously did — the scanner's copy swallowed exceptions and silently kept
    a file the loader would go on to reject on every pass.

    The checks run in a fixed order and each one may assume its predecessors
    passed.  That ordering is load-bearing, not stylistic:

      1. modality       — is this even a continuous-stretch acquisition?
      2. finite         — are the samples numbers at all?
      3. varies         — does anything actually change?
      4. turnaround     — is there an approach and a retract?
      5. retract real   — was the retract buffer written?

    Steps 4 and 5 are reductions (argmax, any) that CANNOT fail: they return an
    answer for every input, including inputs where the question is meaningless.
    Running them on unvalidated data means reading their output as a validity
    signal, which is inference, not measurement — argmax of an all-NaN array
    returns 0, and of a partly-NaN array returns the first NaN's index, both of
    which look exactly like a legitimate result.  Steps 2 and 3 exist so that
    when steps 4 and 5 do fire, they mean what they say.

    Nothing here judges the SCIENCE.  "There are no numbers in this channel"
    and "nothing in this channel ever changed" are statements about data being
    absent; an unusual-but-real curve is not this function's business (see
    CLAUDE.md §4 on informing rather than gating).
    """
    channels = channel_map(labels, wdata.shape[1] if wdata.ndim > 1 else 1)
    curve_type = _modality(wdata, channels, indent_mode, hold_z, spring_constant)
    if curve_type != "continuous_stretch":
        # Nothing else in the app analyses these, so there is no "usable"
        # judgement to make about them.  Saying otherwise would be inventing a
        # verdict for files we never touch.
        return Qualification(curve_type)

    n = wdata.shape[0]
    col_defl  = channels[CH_DEFL]
    col_piezo = piezo_column(channels)
    # A dead position READ-back costs the FFT view and nothing else, so it is
    # absent here: rejecting a curve whose science is intact would be a worse
    # answer than one missing viewer.
    required = ((col_defl, "deflection"), (col_piezo, "piezo"))

    # 2. Finite — every sample in the channels we depend on is a real number.
    for col, name in required:
        bad = ~np.isfinite(wdata[:, col])
        n_bad = int(bad.sum())
        if n_bad:
            return Qualification(
                curve_type, UNUSABLE_NONFINITE,
                f"{name}: {n_bad:,} of {n:,} samples are NaN/inf",
            )

    # 3. Varies — a channel that never changes recorded nothing.  ptp is safe
    #    here only because step 2 established there are no NaNs.
    for col, name in required:
        if float(np.ptp(wdata[:, col])) == 0.0:
            return Qualification(
                curve_type, UNUSABLE_CONSTANT,
                f"{name}: constant at {float(wdata[0, col]):.6g} for all {n:,} samples",
            )

    # 4. Turnaround — the piezo ramp reverses, with real data on both sides.
    idx_turn = int(np.argmax(wdata[:, col_piezo]))
    if idx_turn == 0 or idx_turn >= n - 1:
        return Qualification(
            curve_type, UNUSABLE_NO_TURNAROUND,
            f"piezo peaks at sample {idx_turn:,} of {n:,} "
            f"— no approach and retract either side of it",
        )

    # 5. Retract real — the buffer was actually written.
    if _is_truncated(wdata[:, col_defl], idx_turn):
        return Qualification(
            curve_type, UNUSABLE_TRUNCATED,
            f"all {n - idx_turn - 1:,} retract samples are exactly zero",
            idx_turn,
        )

    return Qualification(curve_type, None, None, idx_turn)


@dataclass
class ForceCurve:
    """
    A single force-extension curve loaded from an .ibw file.

    Spatial units : nm
    Force units   : pN  (deflection × spring_constant)
    Position units: µm

    piezo_appr / piezo_retr : tip-sample separation proxy, positive away from surface
    defl_appr / defl_retr   : cantilever deflection, baseline-subtracted at first point
    spring_constant         : pN/nm — multiply deflection by this to get force in pN
    """
    path:            str
    piezo_appr:      np.ndarray
    defl_appr:       np.ndarray
    piezo_retr:      np.ndarray
    defl_retr:       np.ndarray
    spring_constant: float           # pN/nm
    xpos:            float | None = None  # µm
    ypos:            float | None = None  # µm
    sample_rate_hz:  float = 0.0
    measured_date:   str | None = None
    velocity_nm_s:   float | None = None
    trigger_point_nn: float | None = None   # FORCE (nN), not a distance
    force_dist_nm:   float | None = None
    inv_ols_nm_v:    float | None = None
    # ACQUISITION low-pass bandwidth (Hz) set in the AFM software — this trace
    # arrived already band-limited to it.  Carried on the curve, not looked up
    # per display, so a window comparing it against spectral_cutoff_hz costs no
    # DB query during playback.  None = the wave note did not state one.
    force_filter_bw_hz: float | None = None

    # ── Raw (unsplit) arrays — used by FftWindow ─────────────────────────────
    # Full time-series before the approach/retract split.  None when the file
    # was loaded before this field was added (should not occur in practice).
    raw_defl:       np.ndarray | None = None   # nm, baseline-subtracted, full trace
    raw_piezo_read: np.ndarray | None = None   # nm, full read-piezo trace
    idx_turn:       int                = 0     # turnaround sample index

    # ── Derived quantities ────────────────────────────────────────────────────

    @property
    def force_appr(self) -> np.ndarray:
        """Approach force in pN."""
        return self.defl_appr * self.spring_constant

    @property
    def force_retr(self) -> np.ndarray:
        """Retraction force in pN."""
        return self.defl_retr * self.spring_constant

    @property
    def filename(self) -> str:
        return Path(self.path).name


# ── Loader ────────────────────────────────────────────────────────────────────

def load_force_curve(path: str) -> ForceCurve:
    """
    Load a force-extension .ibw file and return a ForceCurve.

    Raises:
      LoadError           — the file could not be read (I/O, corrupt container).
                            Assumed transient; the analysis layer retries it.
      UnusableCurveError  — the file read fine but does not qualify (see
                            qualify_wave).  Carries .reason; never retried.

    Everything below the qualification call may assume the data is finite, is
    not constant, and has a real turnaround, because qualify_wave established
    all three.  Do not add a defensive re-check here — add it there.
    """
    # ── Read file ─────────────────────────────────────────────────────────────
    try:
        wave   = load_ibw(path)
        wdata  = wave["wave"]["wData"]
        note   = wave["wave"]["note"]
        header = wave["wave"]["wave_header"]
    except Exception as exc:
        raise LoadError(f"Could not read {path}: {exc}") from exc

    # ── Qualify ───────────────────────────────────────────────────────────────
    # The scanner ran this same function at import and stored its verdict, so
    # in the normal case this re-confirms a decision already on record.  It is
    # repeated rather than trusted because a file can be overwritten on disk
    # after it was catalogued, and because the loader has other callers.
    #
    # indent_mode is deliberately passed as None here, and that is not an
    # oversight.  It answers "which experiment did the operator intend", which
    # is a cataloguing fact the scanner needs in order to scope; this function
    # is being asked "can these samples be split into an approach and a
    # retract", which is about structure.  An indentation wave has the
    # force-extension layout and loads fine, so refusing it here would break
    # viewers over a label rather than over the data.  Same function, different
    # question, the information each caller actually has.
    k = _spring_constant(note)
    q = qualify_wave(
        wdata, labels=wave["wave"]["labels"], indent_mode=None,
        hold_z=_hold_z_sensor(note), spring_constant=k,
    )
    if q.curve_type != "continuous_stretch":
        # Durable, so UnusableCurveError and not a plain LoadError: a file that
        # is a force-clamp trace, an image, or a wave with no spring constant
        # will still be all of those things next pass.  Raised as a plain
        # LoadError it read as "couldn't load" → 'unavailable' → retried
        # forever, the same fail-open shape as #122.  The file itself is not
        # rejected from the catalog — it keeps its own curve_type and stays
        # visible; this only says the force-curve pipeline cannot consume it.
        raise UnusableCurveError(
            f"{Path(path).name}: not a force-extension curve "
            f"(wData shape {wdata.shape}, SpringConstant {k!r} → {q.curve_type})",
            UNUSABLE_NOT_FE,
        )
    if not q.usable:
        msg = f"{Path(path).name}: {UNUSABLE_REASON_TEXT[q.reason]} — {q.detail}"
        raise (TruncatedCurveError(msg) if q.reason == UNUSABLE_TRUNCATED
               else UnusableCurveError(msg, q.reason))

    channels   = channel_map(wave["wave"]["labels"], wdata.shape[1])
    deflection = wdata[:, channels[CH_DEFL]]
    piezo      = wdata[:, piezo_column(channels)]
    idx_turn   = q.idx_turn

    # ── Scale and split ───────────────────────────────────────────────────────
    # Piezo : × −1e9  → nm, positive = tip moving away from surface
    # Defl  : × 1e9   → nm, baseline-subtract to first point (follows pysmfs)
    baseline = deflection[0]

    piezo_appr = piezo[:idx_turn]            * -1e9
    defl_appr  = (deflection[:idx_turn]      - baseline) * 1e9
    piezo_retr = piezo[idx_turn + 1:]        * -1e9
    defl_retr  = (deflection[idx_turn + 1:]  - baseline) * 1e9

    # ── Raw unsplit arrays for FFT inspection ─────────────────────────────────
    raw_defl_full       = (deflection - baseline) * 1e9        # nm, full trace
    # The commanded position.  Deliberately not a channel qualification
    # depends on: it feeds the FFT view alone, so when it is damaged the honest
    # cost is one viewer, not a rejected curve.  It may therefore hold NaNs
    # here where the qualified channels do not.
    raw_piezo_read_full = (
        wdata[:, channels.get(CH_RAW, piezo_column(channels))] * -1e9)  # nm, sign-matched

    # ── Wave note fields ──────────────────────────────────────────────────────
    # k was already read and validated by the qualification above — a wave
    # without one is not classified 'force_extension' and never reaches here.
    spring_constant = k

    xpos_m = re.search(rb"XLVDT: ?(-?[0-9]*\.?[0-9]*e?-?[0-9]*)\r", note)
    ypos_m = re.search(rb"YLVDT: ?(-?[0-9]*\.?[0-9]*e?-?[0-9]*)\r", note)
    xpos = float(xpos_m.group(1)) * 1e6 if xpos_m else None
    ypos = float(ypos_m.group(1)) * 1e6 if ypos_m else None

    try:
        sfa = header["sfA"][0]
        sample_rate_hz = float(1.0 / sfa) if sfa > 0 else 0.0
    except (KeyError, IndexError, ZeroDivisionError):
        sample_rate_hz = 0.0

    date_m = re.search(rb"\rDate:([^\r]+)\r", note)
    measured_date = (
        date_m.group(1).decode("latin-1").strip() if date_m else None
    )

    def _safe(pattern: bytes, scale: float = 1.0) -> float | None:
        m = re.search(pattern, note)
        if m is None:
            return None
        try:
            return float(m.group(1)) * scale
        except (ValueError, IndexError):
            return None

    velocity_nm_s    = _safe(rb"Velocity: ([0-9]*\.?[0-9]*e?[+-]?[0-9]*)\r",    1e9)
    # TriggerPoint is stored in Newtons (SI) in the Asylum Research wave note.
    # scale=1e9 converts N → nN.  The field is trigger_point_nn — a FORCE (nN),
    # not a distance.  Confirmed: trigger(nN) × (1/k) = max_deflection(nm).
    trigger_point_nn = _safe(rb"TriggerPoint: ([0-9]*\.?[0-9]*e?[+-]?[0-9]*)\r", 1e9)
    force_dist_nm    = _safe(rb"ForceDist: ([0-9]*\.?[0-9]*e?[+-]?[0-9]*)\r",    1e9)
    inv_ols_nm_v     = _safe(rb"InvOLS: ?([0-9]*\.?[0-9]*e?[+-]?[0-9]*)\r",      1e9)
    # Hz already in the wave note — no scaling.  Same key the scanner promotes
    # to files.force_filter_bw_hz; parsed here too so a loaded curve knows its
    # own bandwidth without a lookup.
    force_filter_bw_hz = _safe(rb"ForceFilterBW: ?([0-9]*\.?[0-9]*e?[+-]?[0-9]*)\r")

    return ForceCurve(
        path             = normalize_path(path),
        piezo_appr       = piezo_appr,
        defl_appr        = defl_appr,
        piezo_retr       = piezo_retr,
        defl_retr        = defl_retr,
        spring_constant  = spring_constant,
        xpos             = xpos,
        ypos             = ypos,
        sample_rate_hz   = sample_rate_hz,
        measured_date    = measured_date,
        velocity_nm_s    = velocity_nm_s,
        trigger_point_nn = trigger_point_nn,
        force_dist_nm    = force_dist_nm,
        inv_ols_nm_v     = inv_ols_nm_v,
        force_filter_bw_hz = force_filter_bw_hz,
        raw_defl         = raw_defl_full,
        raw_piezo_read   = raw_piezo_read_full,
        idx_turn         = idx_turn,
    )
