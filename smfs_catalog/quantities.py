# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/quantities.py
#
# Units, display precision, and input steps for measured values and analysis
# parameters. Values remain stored in the units declared here; this module
# owns presentation metadata and conversions, not measurements or I/O.
#
# This is measurement metadata, not visual styling. DB and pipeline modules
# own values and key inventories; consumers use this registry for consistent
# text, controls, axes, and exports.
#
# Two rules define the boundary:
#
# * Calculations, database values, cache identities, and machine-readable
#   exports retain their available numeric precision. Rounding happens only
#   when a value crosses into user-facing text or a precision-limited control.
# * One arrow step changes the least significant displayed digit. Coarse input
#   is entered by typing or dragging.
# * Any stored/displayed unit conversion is declared on Quantity. Call sites do
#   not apply private scale factors.
#
# UI-only counters and fit statistics use INTEGER/GENERIC or purpose-specific
# formatting at their owning surface.

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

# ── Units ─────────────────────────────────────────────────────────────────────
# Plain ASCII/Unicode text, machine-readable, deliberately NOT typeset.  Same
# boundary style.mathify() draws: typeset on plots, plain everywhere else.
# An empty unit means genuinely dimensionless, which is not the same as unknown.

NM        = "nm"
UM        = "µm"
PN        = "pN"
NN        = "nN"
HZ        = "Hz"
S         = "s"
MS        = "ms"
DB        = "dB"
PTS       = "pts"      # samples / point indices — a count, not a length
COUNT     = ""         # a plain count (segments, ruptures)
NM2       = "nm²"
NM_PER_NM = "nm/nm"    # d¹, a slope — a ratio that names what it is a ratio OF
NM_PER_S  = "nm/s"
NM_PER_V  = "nm/V"
PN_PER_NM = "pN/nm"
RATIO     = ""         # dimensionless by construction (F·l_p/kT, x/l_c)


# ── SI: what a unit IS, in base units ─────────────────────────────────────────
# pyqtgraph prefixes base SI units. This table records the base representation
# and scale of each prefixable displayed unit. qt_utils.set_si_label() is the
# sole Qt-side consumer.

@dataclass(frozen=True)
class SiUnit:
    """The base-SI form of a unit this app displays.

        base_value = shown_value * factor ** power

    `power` exists for squared units: pyqtgraph's siScale takes it so that
    1e-18 m² prefixes to 1 nm² rather than to 1 am².  It is passed straight
    through as setLabel(unitPower=...), which pyqtgraph 0.14 supports.
    """
    base:   str
    factor: float
    power:  int = 1


# A unit ABSENT from this table does not get SI-prefixed, and that is a real
# answer rather than an omission:
#   nm/nm   a ratio of two identical units — "knm/nm" says nothing.
#   pts     a count of samples.
#   dB      already logarithmic; a prefixed decibel is not a quantity.
#   ""      dimensionless by construction (x/l_c, F·l_p/kT).
# Non-SI units belong in the same unprefixed category.
SI_UNITS: dict[str, SiUnit] = {
    NM:  SiUnit("m",  1e-9),
    UM:  SiUnit("m",  1e-6),
    PN:  SiUnit("N",  1e-12),
    NN:  SiUnit("N",  1e-9),
    HZ:  SiUnit("Hz", 1.0),      # already base SI; prefixing gives kHz, MHz
    S:   SiUnit("s",  1.0),
    MS:  SiUnit("s",  1e-3),
    NM2: SiUnit("m²", 1e-18, power=2),
}


def si_for(unit: str) -> "SiUnit | None":
    """The base-SI form of `unit`, or None if it must not be SI-prefixed."""
    return SI_UNITS.get(unit)


@dataclass(frozen=True)
class Quantity:
    """What one physical quantity is, and how precisely it is worth showing.

    `unit` is the unit the value is STORED in — the unit that reaches the
    database, the parameter set and the export manifest.  `display_unit` and
    `display_factor` describe a different unit used on SCREEN, if any:

        displayed = stored * display_factor

    `decimals` applies to the displayed unit, since that is what the user types
    and reads.
    """
    unit:           str
    decimals:       int
    display_unit:   str   = ""      # "" = shown in the stored unit
    display_factor: float = 1.0
    integer:        bool  = False

    @property
    def shown_unit(self) -> str:
        return self.display_unit or self.unit

    @property
    def step(self) -> float:
        """Rule 1: one arrow click moves the last digit shown, by one."""
        return 1.0 if self.integer else 10.0 ** -self.decimals

    @property
    def suffix(self) -> str:
        """Spin-box suffix, with the leading space Qt needs. Blank if unitless."""
        return f" {self.shown_unit}" if self.shown_unit else ""

    def to_display(self, stored: float) -> float:
        return stored * self.display_factor

    def to_stored(self, displayed: float) -> float:
        return displayed / self.display_factor


INTEGER = Quantity(COUNT, 0, integer=True)
GENERIC = Quantity("", 4)      # unknown key: 4 decimals, no unit claimed


# ── The register ──────────────────────────────────────────────────────────────
# Keyed by the name the rest of the app already uses: analysis_results types,
# roi_pipeline.SEG_SUMMARY_KEYS, and db.PARAM_KEYS.  One key, one answer,
# wherever it is displayed.

QUANTITIES: dict[str, Quantity] = {

    # ── File/instrument metadata ──────────────────────────────────────────────
    # These values come from the .ibw wave note rather than analysis_results,
    # but they are ordinary axis/export variables and need the same explicit
    # measurement metadata. Acquisition time is stored as Unix seconds and
    # formatted as a date/time by its UI consumers.
    "measured_at_ts":         Quantity(S, 0),
    "spring_constant_pn_nm": Quantity(PN_PER_NM, 3),
    "velocity_nm_s":         Quantity(NM_PER_S, 1),
    "force_dist_nm":         Quantity(NM, 1),
    "trigger_point_nn":      Quantity(NN, 3),
    "sample_rate_hz":        Quantity(HZ, 1),
    "force_filter_bw_hz":    Quantity(HZ, 1),
    "inv_ols_nm_v":          Quantity(NM_PER_V, 3),
    "xpos_um":               Quantity(UM, 2),
    "ypos_um":               Quantity(UM, 2),

    # ── Segment/ROI results (roi_pipeline.SEG_SUMMARY_KEYS) ──────────────────
    # Precision reflects the observed scale of each result.
    "seg_l_p_nm":      Quantity(NM, 3),    # median 0.251, 83% below 1 nm
    "seg_l_c_nm":      Quantity(NM, 1),    # median 104, p95 411
    "seg_l_p_err":     Quantity(NM, 4),    # median 0.0057 — 2 decimals reads 0.00
    "seg_l_c_err":     Quantity(NM, 3),    # median 0.184
    "seg_force_pN":    Quantity(PN, 1),    # median 166, p95 353
    # The reported rupture's position on the extension axis, under its two
    # zeros. Same scale as the other extension-axis distances below.
    "seg_x_rupture_nm":  Quantity(NM, 1),
    "seg_x_junction_nm": Quantity(NM, 1),
    "seg_dF_pN":       Quantity(PN, 1),
    "seg_dX_iso_nm":   Quantity(NM, 1),
    "seg_dX_ext_nm":   Quantity(NM, 1),
    "seg_n_segments":  Quantity(COUNT, 0, integer=True),   # 1..6, a count
    # Correlation time is measured in samples and must not be SI-prefixed.
    "seg_tau":         Quantity(PTS, 1),
    # x_max/l_c is dimensionless; three decimals preserve useful conditioning
    # differences near the WLC pole.
    "seg_z_max":       Quantity(RATIO, 3),
    # z_max's numerator, in nm — same scale as the other extension distances.
    "seg_x_max_nm":    Quantity(NM, 1),
    # A flag, carried as 0/1 so it can be bounded like any other criterion
    # ("seg_edge_pinned <= 0" = exclude edge-pinned fits).
    "seg_edge_pinned": Quantity(COUNT, 0, integer=True),

    # ── Per-curve landmarks and calibration (analysis_results) ───────────────
    "contact_piezo_nm": Quantity(NM, 1),   # ~ -900 .. 3600
    "snapoff_piezo_nm": Quantity(NM, 1),
    "onset_piezo_nm":   Quantity(NM, 1),
    "rupture_piezo_nm": Quantity(NM, 1),
    # Distances from snap-off are derived on read and are smaller than absolute
    # stage positions, so they retain one additional decimal place.
    "contact_dx_nm":    Quantity(NM, 2),
    "onset_dx_nm":      Quantity(NM, 2),
    "rupture_dx_nm":    Quantity(NM, 2),
    "offset_retr":      Quantity(NM, 4),   # deflection offset, median -0.083
    "flatness_slope":   Quantity(NM_PER_NM, 6),  # deflection / piezo
    "baseline_rms":     Quantity(NM, 4),   # 0.088 .. 3.1
    "baseline_r2":      Quantity(RATIO, 4),
    "invols_slope":     Quantity(NM_PER_NM, 4),  # -1.68 .. -0.51
    "invols_rms":       Quantity(NM, 4),   # 0.011 .. 0.29
    "invols_r2":        Quantity(RATIO, 4),
    "baseline_intercept": Quantity(NM, 4),
    "invols_intercept":   Quantity(NM, 4),
    "snapoff_idx":      INTEGER,
    "rupture_idx":      INTEGER,
    "onset_idx":        INTEGER,
    "baseline_fit_lo_idx": INTEGER,
    "baseline_fit_hi_idx": INTEGER,
    "invols_fit_lo_idx":   INTEGER,
    "invols_fit_hi_idx":   INTEGER,

    # Legacy result keys remain readable for existing catalogs.
    "wlc_l_p_nm":       Quantity(NM, 3),
    "wlc_l_c_nm":       Quantity(NM, 1),
    "wlc_l_p_err":      Quantity(NM, 4),
    "wlc_l_c_err":      Quantity(NM, 3),
    "rupture_force_pn": Quantity(PN, 1),

    # ── Analysis parameters (db.PARAM_KEYS) ──────────────────────────────────
    "spectral_cutoff_hz":            Quantity(HZ, 0, integer=True),
    "turnaround_trim_pts":           Quantity(PTS, 0, integer=True),
    "var_window_ms":                 Quantity(MS, 2),
    # Detection variance is stored and displayed in nm².
    "detection_threshold_appr":      Quantity(NM2, 6),
    "detection_threshold_retr":      Quantity(NM2, 6),
    "baseline_anchor_nm":            Quantity(NM, 0, integer=True),
    "invols_offset_pts":             Quantity(PTS, 0, integer=True),
    "invols_window_pts":             Quantity(PTS, 0, integer=True),
    "roi_window_pts":                Quantity(PTS, 0, integer=True),
    # Stored to full float precision by the draggable line; 4 decimals is what
    # the spin box can show without disagreeing with the drag.
    "roi_threshold_nm_per_nm":       Quantity(NM_PER_NM, 4),
    "roi_inner_threshold_nm_per_nm": Quantity(NM_PER_NM, 4),
    "roi_post_snapoff_mask_nm":      Quantity(NM, 1),
    "roi_onset_threshold_nm":        Quantity(NM, 3),
    "roi_detector_mode_idx":         INTEGER,
    "roi_prominence":                Quantity(NM_PER_NM, 3),
    "roi_min_distance_pts":          Quantity(PTS, 0, integer=True),

    # ── 2DH grid settings ────────────────────────────────────────────────────
    # Normalized axes x/l_c and F·l_p/kT are dimensionless.
    "wlc_x_min":  Quantity(RATIO, 2), "wlc_x_max":  Quantity(RATIO, 2),
    "wlc_f_min":  Quantity(RATIO, 1), "wlc_f_max":  Quantity(RATIO, 1),
    "wlc_x_bins": INTEGER,            "wlc_f_bins": INTEGER,
    "phys_x_min": Quantity(NM, 1),    "phys_x_max": Quantity(NM, 1),
    "phys_f_min": Quantity(PN, 1),    "phys_f_max": Quantity(PN, 1),
    "phys_x_bins": INTEGER,           "phys_f_bins": INTEGER,
    "phys_f_star": Quantity(PN, 1),

    # ── FFT / notch controls ─────────────────────────────────────────────────
    "notch_f0_hz":    Quantity(HZ, 1),
    "notch_bw_hz":    Quantity(HZ, 1),
    "notch_depth_db": Quantity(DB, 1),
}


# ── Lookup ────────────────────────────────────────────────────────────────────

def get(key: str) -> Quantity:
    """The quantity for `key`, or GENERIC if it isn't a known physical one.

    Never raises: an unknown key is a display question, not an error, and a
    missing entry must not take a window down.  It also means a newly added
    result key still renders sensibly before anyone registers it here.
    """
    return QUANTITIES.get(key, GENERIC)


def unit_of(key: str) -> str:
    """The unit a STORED value of `key` is in — what the manifest records."""
    return get(key).unit


def units_for(keys) -> dict[str, str]:
    """Return the stored unit for each requested registered key.

    A manifest is read months later without the app, so a bare number under
    a key whose name carries no unit (detection_threshold_appr, roi_prominence,
    phys_f_max) is unreadable. Unknown keys are omitted: an absent declaration
    remains distinguishable from the empty unit of a registered dimensionless
    quantity.
    """
    return {key: QUANTITIES[key].unit for key in keys if key in QUANTITIES}


# ── Formatting text ───────────────────────────────────────────────────────────

def format_value(key: str, value, *, with_unit: bool = False) -> str:
    """Render a stored value as text, at its quantity's declared precision.

    Missing and non-finite measurements render blank. A nonzero value below the
    fixed-decimal resolution uses four significant figures instead of appearing
    to be zero.
    """
    if value is None:
        return ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not isfinite(v):
        return ""
    q = get(key)
    shown = q.to_display(v)
    if q.integer:
        text = f"{shown:.0f}"
    else:
        rounded = round(shown, q.decimals)
        if rounded == 0.0 and shown != 0.0:
            text = f"{shown:.4g}"    # too small for the declared decimals
        else:
            text = f"{shown:.{q.decimals}f}"
    return f"{text} {q.shown_unit}" if (with_unit and q.shown_unit) else text


# ── Configuring a spin box ────────────────────────────────────────────────────

def decimals_for(key: str, *displayed_values: float) -> int:
    """Declared decimals, widened if needed to show `displayed_values` exactly.

    QDoubleSpinBox rounds its value to its configured decimals. Widening when a
    stored value is seeded prevents the control from silently changing that
    value. The result is capped at Qt's practical limit of 12 decimals.
    """
    want = get(key).decimals
    for v in displayed_values:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if not isfinite(f):
            continue
        d = want
        while d < 12 and round(f, d) != f:
            d += 1
        want = max(want, d)
    return want


def quantize(key: str, stored_value: float) -> float:
    """Round a gesture-produced stored value to its displayed precision.

    Use this on drag/write paths so the control and stored parameter agree. Do
    not use it when seeding a control from existing data; `decimals_for()`
    widens the control instead.
    """
    q = get(key)
    try:
        v = float(stored_value)
    except (TypeError, ValueError):
        return stored_value
    if not isfinite(v):
        return v
    if q.integer:
        return float(round(v))
    # Round again in the stored unit to remove binary noise introduced by the
    # inverse display conversion before serialization.
    return round(q.to_stored(round(q.to_display(v), q.decimals)), 12)


def configure_spinbox(spin, key: str | None = None, *,
                      decimals: int | None = None,
                      suffix: bool = True) -> None:
    """Apply the arrow-step rule (and the unit suffix) to one spin box.

    This is the single owner of numeric precision and step size. Keyboard
    tracking is disabled so partially typed values are not committed.

    `key` is a quantities key; omit it for a UI-only counter (bins, k), which
    gets step 1 and no suffix.  `decimals` overrides the declared precision —
    pass decimals_for(key, stored_value) when seeding a box from stored data.
    """
    from PyQt6.QtWidgets import QDoubleSpinBox

    q = get(key) if key else INTEGER
    spin.setKeyboardTracking(False)

    if isinstance(spin, QDoubleSpinBox):
        dec = q.decimals if decimals is None else decimals
        spin.setDecimals(dec)
        spin.setSingleStep(1.0 if q.integer else 10.0 ** -dec)
    else:
        spin.setSingleStep(1)

    if suffix and key and q.suffix:
        spin.setSuffix(q.suffix)


# ── Auditing what is already stored ───────────────────────────────────────────

def audit_stored_precision(stored: dict[str, float]) -> list[tuple[str, float, int]]:
    """Which stored values carry more digits than their quantity warrants.

    Returns (key, value, digits_needed) for each one.  Nothing is changed —
    this reports so the user can decide, because rounding a stored bound can
    change results: `seg_n_segments <= 3.704446` and `<= 4` genuinely differ for
    a curve with four segments.  Tidying these is a deliberate act, not a side
    effect of a display change.
    """
    out: list[tuple[str, float, int]] = []
    for key, value in stored.items():
        if value is None or key not in QUANTITIES:
            continue
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if not isfinite(f):
            continue
        q = QUANTITIES[key]
        shown = q.to_display(f)
        need = decimals_for(key, shown)
        if need > q.decimals:
            out.append((key, f, need))
    return out
