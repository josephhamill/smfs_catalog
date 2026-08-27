# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/variables.py
#
# Central registry and bulk value router for variables shown by the UI.
# Values come from segment summaries, analysis_results, or file metadata.
# quantities.py owns units and formatting; the producer stores own their keys.

from __future__ import annotations

import datetime
from dataclasses import dataclass

import numpy as np

from . import db as _db
from . import quantities as _quant
from .roi_pipeline import (
    SEG_SUMMARY_KEYS, SEG_SUMMARY_FIELD, read_segment_select, segment_summary_bulk,
)

# Segment values need their own route because they follow the live segment
# selection and per-curve overrides.
SOURCE_SEGMENT  = "segment"
SOURCE_ANALYSIS = "analysis_result"
SOURCE_FILE     = "file"


@dataclass(frozen=True)
class Variable:
    key:    str
    label:  str
    source: str

    @property
    def unit(self) -> str:
        """The unit a stored value is in — asked of quantities.py, never
        declared here.  Two registers naming the same unit is how they drift."""
        return _quant.unit_of(self.key)

    @property
    def description(self) -> str:
        """
        What this variable means — see DESCRIPTIONS below.

        A property rather than a field for the same reason `unit` is one: a
        field would have to be filled in at every construction site, and the
        one that forgot would produce a Variable that silently describes
        itself as nothing.  Asking the register cannot forget.
        """
        return describe(self.key)

    @property
    def is_time(self) -> bool:
        return self.key == TIME_KEY


# Acquisition time is an ordinary variable so drift is simply value vs time.
TIME_KEY = "measured_at_ts"

# File columns that plausibly explain a measurement and belong on an axis.
_FILE_COLUMNS: dict[str, str] = {
    TIME_KEY:                "Acquisition time",
    "spring_constant_pn_nm": "Spring constant (pN/nm)",
    "velocity_nm_s":         "Pulling velocity (nm/s)",
    "force_dist_nm":         "Force distance (nm)",
    "trigger_point_nn":      "Trigger point (nN)",
    "sample_rate_hz":        "Sample rate (Hz)",
    "force_filter_bw_hz":    "Acquisition filter BW (Hz)",
    "inv_ols_nm_v":          "InvOLS (nm/V)",
    "xpos_um":               "Stage X (µm)",
    "ypos_um":               "Stage Y (µm)",
}

# ── The piezo landmarks are measured FROM snap-off, not from the stage ───────
#
# Raw landmark positions are dominated by stage drift. User-facing landmarks
# are therefore distances from snap-off, the zero already used by WLC fits.
# These distances are derived on read and are never stored separately.
REFERENCE_KEY = "snapoff_piezo_nm"

# Derived key -> (minuend, subtrahend). Contact lies before snap-off, hence its
# reversed subtraction.
REFERENCED: dict[str, tuple[str, str]] = {
    "contact_dx_nm": (REFERENCE_KEY, "contact_piezo_nm"),
    "onset_dx_nm":   ("onset_piezo_nm",   REFERENCE_KEY),
    "rupture_dx_nm": ("rupture_piezo_nm", REFERENCE_KEY),
}

# Known display labels; unknown analysis keys fall back to their key.
_ANALYSIS_LABELS: dict[str, str] = {
    "snapoff_piezo_nm": "Snap-off, abs. piezo (nm)",
    "contact_dx_nm":    "Contact→snap-off (nm)",
    "onset_dx_nm":      "Onset from snap-off (nm)",
    "rupture_dx_nm":    "Rupture from snap-off (nm)",
    "offset_retr":      "Offset",
    "flatness_slope":   "Flatness",
    "baseline_rms":     "Baseline RMS (nm)",
    "invols_slope":     "InvOLS",
    "invols_rms":       "InvOLS RMS (nm)",
}

# THE label for each key — one name, wherever it is shown.  The dashboard's
# queue headers come from here too (see label() below); it used to keep a
# second hand-written list, and six of twenty-one keys had drifted apart on it.
#
# "ΔX" is deliberately absent: roi_events uses it for the PIEZO separation
# (ROI.dX_pairs, "the raw stage displacement"), so labelling an extension-axis
# quantity with it names the wrong coordinate.
_SEG_LABELS: dict[str, str] = {
    "seg_n_segments":  "ROI Segments",
    "seg_force_pN":    "Seg Force (pN)",
    "seg_x_rupture_nm":  "Seg rupture extension (nm)",
    "seg_x_junction_nm": "Seg junction extension (nm)",
    "seg_l_p_nm":      "Seg l_p (nm)",
    "seg_l_p_err":     "Seg l_p err (nm)",
    "seg_l_c_nm":      "Seg l_c (nm)",
    "seg_l_c_err":     "Seg l_c err (nm)",
    "seg_tau":         "Seg τ (samples)",
    "seg_z_max":       "Seg z_max",
    "seg_edge_pinned": "Seg edge-pinned",
    "seg_dF_pN":       "ΔF ult−pen (pN)",
    "seg_dX_iso_nm":   "Reload distance (nm)",
    "seg_dX_ext_nm":   "Rupture separation (nm)",
}


# ── What each variable MEANS ─────────────────────────────────────────────────
#
# One short definition per offered variable, shared by every UI consumer.
DESCRIPTIONS: dict[str, str] = {
    "snapoff_piezo_nm": "Absolute piezo position where the tip leaves the surface on retract. Plot it against acquisition time to inspect stage drift.",
    "contact_dx_nm": "Piezo travel from approach contact to retract snap-off. It tracks whether the tip engages the surface consistently.",
    "rupture_dx_nm": "Commanded piezo travel from snap-off to the outermost rupture, read at the d1 peak. Stage displacement, not extension — the cantilever's own bending is still in it. See Seg rupture extension.",
    "onset_dx_nm": "Commanded piezo travel from snap-off to the start of the outermost rupture's loading ramp. Stage displacement, not extension — see Seg junction extension for the corrected span.",
    "offset_retr": "Deflection baseline offset subtracted before converting deflection to force.",
    "flatness_slope": "Slope of the retract baseline, used as one indication that a curve contains a real event.",
    "baseline_rms": "Residual RMS of the far-retract baseline fit. Larger values indicate a noisier or drifting baseline.",
    "invols_slope": "Per-curve inverse optical-lever sensitivity fitted in the deep-contact approach region; ideally close to one.",
    "invols_rms": "Residual RMS of the per-curve InvOLS fit. Larger values indicate a noisier or more nonlinear contact region.",
    "seg_force_pN": "Rupture force terminating the currently selected Ultimate or Penultimate segment, or a manually picked Primary segment on curves that have one.",
    "seg_x_rupture_nm": "Extension at that same rupture, from snap-off, on the deflection-corrected axis the WLC fits use. The junction's end-to-end length when it broke, so it is the one comparable with Seg l_c.",
    "seg_x_junction_nm": "How far the junction had stretched when that rupture happened: the same extension measured from the junction's onset, not snap-off. A stretch, not a length — not comparable with Seg l_c.",
    "seg_l_p_nm": "WLC persistence length fitted to the currently selected Ultimate or Penultimate segment.",
    "seg_l_c_nm": "WLC contour length fitted to the currently selected Ultimate or Penultimate segment.",
    "seg_l_p_err": "Correlation-corrected fit uncertainty (±1σ) on the selected segment's persistence length; it is a lower bound on total uncertainty.",
    "seg_l_c_err": "Correlation-corrected fit uncertainty (±1σ) on the selected segment's contour length; it is a lower bound on total uncertainty.",
    "seg_tau": "Residual correlation time in samples. Larger values mean more neighbouring samples act like repeated observations rather than independent data.",
    "seg_z_max": "Maximum fitted extension divided by contour length. Higher values generally mean the WLC fit is better conditioned.",
    "seg_edge_pinned": "Whether the force peak lies on the fitted window's right edge (1 = edge, 0 = interior), where force may be underestimated.",
    "seg_dF_pN": "Ultimate rupture force minus penultimate rupture force. Blank when fewer than two ruptures are available.",
    "seg_dX_iso_nm": "Reload distance at the penultimate rupture force after that rupture — the force-matched twin of Rupture separation. Blank if the force is not reached again.",
    "seg_dX_ext_nm": "Extension gap between the last two rupture points, without force matching — the unmatched twin of Reload distance. Blank when fewer than two ruptures are available.",
    "seg_n_segments": "Number of ruptured segments in the right-most outer ROI; two or more means a distinct penultimate segment exists.",
    TIME_KEY: "Instrument acquisition time. Put it on the X axis to measure drift; older files may have date-only resolution.",
    "spring_constant_pn_nm": "Cantilever stiffness recorded by the instrument. It scales every force converted from deflection.",
    "velocity_nm_s": "Piezo retraction speed. Rupture force depends on loading rate, so compare cohorts at similar velocities.",
    "force_dist_nm": "Total commanded piezo travel for the ramp, not the molecule's extension.",
    "trigger_point_nn": "Force setpoint at which approach stopped and retract began; despite its name, this is not a distance.",
    "sample_rate_hz": "Samples recorded per second. Interpret it together with the acquisition and analysis filter bandwidths.",
    "force_filter_bw_hz": "Low-pass bandwidth applied by the AFM at acquisition, before this application's own filtering.",
    "inv_ols_nm_v": "Instrument-recorded deflection sensitivity in nanometres per volt, distinct from the per-curve InvOLS fit.",
    "xpos_um": "Stage X position on the sample, useful for detecting spatially localized surface effects.",
    "ypos_um": "Stage Y position on the sample, useful for detecting spatially localized surface effects.",
}


def describe(key: str) -> str:
    """
    What this variable means, for a reader who has not used the app.

    Empty string for a key with no description rather than a placeholder: a
    caller can then decide whether to set a tooltip at all, and an empty
    tooltip is not the same as a tooltip that says nothing useful.
    """
    return DESCRIPTIONS.get(key, "")


# ── Keys that are IN analysis_results but are not variables ──────────────────
#
# THE ONE ANSWER, and it lives here rather than in dashboard_window because
# this module is Qt-free and the dashboard can import it (not the reverse).
# It was hand-copied there as `_QUEUE_HIDE`, and writing a second copy for the
# scatter window immediately went wrong: the copy invented two key names that
# are not in the database at all and missed all the real ones.  Exactly §6.
#
# Exclusion is a presentation decision, not a deletion decision. Most entries
# below are live cache plumbing; the explicitly named legacy set remains
# readable only so old catalogs and exports can still be interpreted.
INTERNAL_RESULT_KEYS: frozenset[str] = frozenset({
    # The verdict cache: a float stored purely so a re-run can skip recompute.
    "event",
    # Raw sample indices, cached beside their piezo-nm counterparts.
    "onset_idx", "rupture_idx", "snapoff_idx",
    # The array-index bounds of the baseline / invOLS calibration fit windows.
    "baseline_fit_lo_idx", "baseline_fit_hi_idx",
    "invols_fit_lo_idx", "invols_fit_hi_idx",
})

RAW_LANDMARK_KEYS: frozenset[str] = frozenset({
    # Cached so curve_analysis can skip the ROI search on a
    # later pass (curve_analysis.py's `_get("rupture_piezo_nm", ...)`).  They
    # are offered as contact_dx_nm/onset_dx_nm/rupture_dx_nm instead — see
    # REFERENCED above.  Offering both would put a sum in the same dropdown as
    # its own two parts, and the sum is the one that means nothing on its own.
    "contact_piezo_nm", "onset_piezo_nm", "rupture_piezo_nm",
})

CALIBRATION_PLUMBING_KEYS: frozenset[str] = frozenset({
    # The intercepts of the baseline / invOLS calibration lines, which
    # decomposition_window uses to redraw those lines in absolute piezo space
    # (decomposition_window.py:1038).  An intercept is pinned at piezo = 0, so
    # on an axis it is the stage position again under a calibration's name:
    # invols_intercept measures r = +0.994 against snapoff_piezo_nm.
    "baseline_intercept", "invols_intercept",
})

# Readable for old catalogs, but superseded or retired.  Keeping their
# Quantity registrations is intentional backward compatibility; offering them
# beside the current measurements is not.
LEGACY_RESULT_KEYS: frozenset[str] = frozenset({
    "wlc_l_p_nm", "wlc_l_c_nm", "wlc_l_p_err", "wlc_l_c_err",
    "rupture_force_pn", "baseline_r2", "invols_r2",
})

EXCLUDED_VARIABLE_KEYS: frozenset[str] = frozenset().union(
    INTERNAL_RESULT_KEYS,
    RAW_LANDMARK_KEYS,
    CALIBRATION_PLUMBING_KEYS,
    LEGACY_RESULT_KEYS,
)

# Compatibility name for external callers. New code should use the name that
# says what the set actually does.
NOT_A_VARIABLE = EXCLUDED_VARIABLE_KEYS


def _label(key: str) -> str:
    return (_SEG_LABELS.get(key) or _ANALYSIS_LABELS.get(key)
            or _FILE_COLUMNS.get(key) or key)


def label(key: str) -> str:
    """The name this key is shown under — asked of the register, never spelled
    out a second time by a consumer.

    The dashboard used to keep its own (key, label) list for the queue headers.
    Six of twenty-one keys had drifted: seg_dX_ext_nm read "Ext ΔX (nm)" in the
    queue and "Rupture separation (nm)" in the scatter — the same number under
    two names on two screens, with nothing to say they were the same number.
    Two lists that merely agree today are the fork; asking one register is the
    only version that cannot come apart.
    """
    return _label(key)


def provenance_key(key: str) -> str:
    """The analysis_results key whose stored params_json describes `key`.

    A referenced landmark is computed from two stored numbers, so "which
    parameter set produced this value" has two candidate answers.  Take the
    first of the pair, which is the one whose own search the answer is about:
    the ROI params for onset/rupture, the contact-detection params for
    contact_dx_nm (whose other end is the contact point, from the same
    detection pass).  Identity for everything else.
    """
    return REFERENCED[key][0] if key in REFERENCED else key


def source_of(key: str) -> str:
    if key in SEG_SUMMARY_KEYS:
        return SOURCE_SEGMENT
    if key in _FILE_COLUMNS:
        return SOURCE_FILE
    return SOURCE_ANALYSIS


def available(paths: list[str], db_path: str = _db.DEFAULT_DB_PATH) -> list[Variable]:
    """Every variable that could be put on an axis for this cohort.

    Data-driven for analysis_results (whatever has actually been computed),
    fixed for the seg_* family and the file columns — those exist whether or
    not this particular cohort has values yet, and offering an axis that comes
    back empty is a better answer than silently not offering it: the empty
    plot says "nothing measured here", a missing dropdown entry says nothing
    at all.
    """
    out: list[Variable] = [
        Variable(k, _label(k), SOURCE_SEGMENT) for k in SEG_SUMMARY_KEYS
    ]
    try:
        present = set(_db.get_queue_analysis_types(db_path))
    except Exception:
        present = set()
    present |= set(_ANALYSIS_LABELS)
    for k in sorted(present):
        if (k in EXCLUDED_VARIABLE_KEYS or k in SEG_SUMMARY_KEYS or k in _FILE_COLUMNS):
            continue
        out.append(Variable(k, _label(k), SOURCE_ANALYSIS))
    out += [Variable(k, lbl, SOURCE_FILE) for k, lbl in _FILE_COLUMNS.items()]
    return out


def _timestamp(text: str | None) -> float:
    """Acquisition datetime -> Unix seconds; NaN when absent or unparseable.

    Accepts the full 'YYYY-MM-DD HH:MM:SS' and the day-only fallback, matching
    variable_window's own reader — files not yet re-scanned with time capture
    carry only the date.
    """
    if not text:
        return float("nan")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(text.strip(), fmt).timestamp()
        except (ValueError, OSError, OverflowError):
            continue
    return float("nan")


def values(
    paths: list[str], keys: list[str], db_path: str = _db.DEFAULT_DB_PATH,
) -> dict[str, dict[str, float | None]]:
    """{normalized_path: {key: value_or_None}} for any mix of the three sources.

    One query per source, never one per file.  A key with no value for a path
    is present and None, not absent — a caller must be able to tell "measured
    as nothing" from "I forgot to ask", and an absent key silently reads as
    the former.
    """
    resolved = [_db.normalize_path(p) for p in paths]
    out: dict[str, dict[str, float | None]] = {rp: {} for rp in resolved}

    seg_keys  = [k for k in keys if source_of(k) == SOURCE_SEGMENT]
    file_keys = [k for k in keys if source_of(k) == SOURCE_FILE]
    ar_keys   = [k for k in keys if source_of(k) == SOURCE_ANALYSIS]

    if ar_keys:
        # A referenced landmark needs both its own raw key and the zero it is
        # measured from, so the fetch list is widened rather than the caller
        # being asked to know that.  Either raw key may ALSO have been asked
        # for in its own right (snapoff_piezo_nm is a variable); the set union
        # handles that without fetching anything twice.
        fetch = set(ar_keys)
        for k in ar_keys:
            if k in REFERENCED:
                fetch.discard(k)
                fetch.update(REFERENCED[k])
        derived = _db.get_derived_results_bulk_latest(paths, sorted(fetch), db_path)
        for rp in resolved:
            d = derived.get(rp, {})
            raw = {k: (d[k][0] if d.get(k) is not None else None) for k in fetch}
            for k in ar_keys:
                if k in REFERENCED:
                    a, b = (raw.get(x) for x in REFERENCED[k])
                    # Missing either end means no distance — None, never a
                    # value computed against a zero we do not have.
                    out[rp][k] = None if a is None or b is None else float(a) - float(b)
                else:
                    out[rp][k] = raw.get(k)

    if seg_keys:
        seg = segment_summary_bulk(paths, read_segment_select(db_path), db_path)
        for rp in resolved:
            d = seg.get(rp, {})
            for k in seg_keys:
                v = d.get(SEG_SUMMARY_FIELD[k])
                # edge_pinned is a bool downstream of the dataclass and a 0/1
                # everywhere else, so that it can be bounded like any other
                # criterion.  An axis needs the number.
                out[rp][k] = float(v) if isinstance(v, bool) else v

    if file_keys:
        wanted = [k for k in file_keys if k != TIME_KEY]
        if TIME_KEY in file_keys:
            wanted += ["measured_at", "measured_date"]
        rows = _db.get_file_columns(paths, wanted, db_path)
        for rp in resolved:
            r = rows.get(rp, {})
            for k in file_keys:
                if k == TIME_KEY:
                    # measured_at (second resolution) where it exists, else the
                    # day-only fallback for files not yet re-scanned with time
                    # capture — the same precedence variable_window uses.
                    out[rp][k] = _timestamp(r.get("measured_at")
                                            or r.get("measured_date"))
                else:
                    v = r.get(k)
                    out[rp][k] = float(v) if v is not None else None
    return out


def columns(
    paths: list[str], keys: list[str], db_path: str = _db.DEFAULT_DB_PATH,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """(paths in order, {key: float array aligned with them}), missing = NaN.

    Aligned by construction, which is the point: a scatter that drops missing
    values per-array rather than per-pair slides one axis against the other
    and confidently correlates the wrong curves together.  Callers mask
    pairwise from these arrays (regression._finite_pairs does exactly that).
    """
    resolved = [_db.normalize_path(p) for p in paths]
    v = values(paths, keys, db_path)
    cols: dict[str, np.ndarray] = {}
    for k in keys:
        cols[k] = np.array(
            [(np.nan if v[rp].get(k) is None else float(v[rp][k])) for rp in resolved],
            dtype=float,
        )
    return resolved, cols
