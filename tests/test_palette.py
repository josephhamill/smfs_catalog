# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Guard test for smfs_catalog/style.py's palettes.

WHY THIS EXISTS.  The palette this replaced carried a comment stating it had
been "re-checked for this use: worst-pair CVD ΔE 14.8 (protan/deutan/tritan)".
In fact its orange/green pair is **3.2** under protanopia — the claim looks
like it was read off deutan (15.6) or tritan (33.9) and never off the one that
failed.  Nothing caught it, because a colour claim in a comment is exactly the
kind of thing nobody re-derives.  Same failure shape as the force-sign
comment, and the same remedy: assert the property, not the prose.

So: these checks recompute the palette's colour-blindness separation, contrast
and lightness from the hex values themselves, every run.  Edit a hex in
style.py and this tells you immediately whether it still works, instead of a
colleague discovering two of your ROIs are the same colour to them.

The maths is the data-viz reference validator's, ported to Python (that
implementation is JavaScript, and this machine has no Node): OKLab ΔE ×100 with
Machado-Oliveira-Fernandes (2009) severity-1.0 CVD transforms.  The port is
checked against the reference's own published numbers in
test_validator_matches_reference_numbers, so a silent drift in the maths fails
too — a validator nobody validates is just a second thing to be wrong.

Pairlist is ALL PAIRS, not adjacent: every colour in these palettes is drawn
simultaneously on one plot (several ROIs on one curve, several fit components
on one histogram), so any two of them can end up side by side.
"""

from __future__ import annotations

import itertools
import math

import pytest

from smfs_catalog import style

# ── thresholds (data-viz reference validator) ────────────────────────────────
BAND_LIGHT = (0.43, 0.77)     # OKLCH L
CHROMA_FLOOR = 0.10           # OKLCH C
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0

MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868],
               [0.114503, 0.786281, 0.099216],
               [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968],
               [0.280085, 0.672501, 0.047413],
               [-0.011820, 0.042940, 0.968881]],
}


def _lin(h: str):
    h = h.strip().lstrip("#")
    out = []
    for i in (0, 2, 4):
        c = int(h[i:i + 2], 16) / 255
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return out


def _oklab(rgb):
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def _oklch(h):
    L, a, b = _oklab(_lin(h))
    return L, math.hypot(a, b)


def _simulate(h, kind):
    r, g, b = _lin(h)
    M = MACHADO[kind]
    return [min(1.0, max(0.0, M[i][0] * r + M[i][1] * g + M[i][2] * b))
            for i in range(3)]


def _delta_e(h1, h2, kind=None):
    a = _oklab(_simulate(h1, kind) if kind else _lin(h1))
    b = _oklab(_simulate(h2, kind) if kind else _lin(h2))
    return 100 * math.dist(a, b)


def _contrast(a, b):
    def lum(h):
        r, g, bb = _lin(h)
        return 0.2126 * r + 0.7152 * g + 0.0722 * bb
    hi, lo = sorted([lum(a), lum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _worst_cvd(palette):
    return min(min(_delta_e(a, b, "protan"), _delta_e(a, b, "deutan"))
               for a, b in itertools.combinations(palette, 2))


def _worst_normal(palette):
    return min(_delta_e(a, b) for a, b in itertools.combinations(palette, 2))


# ── (a) the port itself ──────────────────────────────────────────────────────

def test_validator_matches_reference_numbers():
    """The reference palette's documented figures, recomputed here.

    If this fails the maths above drifted, and every other assertion in this
    file became meaningless — so it is checked first and separately.
    """
    ref8 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
            "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    adj_cvd = min(min(_delta_e(ref8[i], ref8[i + 1], "protan"),
                      _delta_e(ref8[i], ref8[i + 1], "deutan"))
                  for i in range(len(ref8) - 1))
    adj_nrm = min(_delta_e(ref8[i], ref8[i + 1]) for i in range(len(ref8) - 1))
    assert round(adj_cvd, 1) == 9.1, adj_cvd
    assert round(adj_nrm, 1) == 19.6, adj_nrm

    first3 = ref8[:3]
    assert round(_worst_cvd(first3), 1) == 9.2
    assert round(_worst_normal(first3), 1) == 24.0


# ── (b) the palettes this app actually ships ─────────────────────────────────

@pytest.mark.parametrize("name,palette", [
    ("SERIES_LINE", style.SERIES_LINE),
    ("SERIES_LABELED", style.SERIES_LABELED),
])
def test_palette_separation(name, palette):
    """Any two slots must be tellable apart — by a colour-blind reader (CVD)
    and by a full-colour one (normal-vision floor).  All pairs, because these
    are drawn simultaneously."""
    cvd = _worst_cvd(palette)
    nrm = _worst_normal(palette)
    assert cvd >= CVD_FLOOR, f"{name}: worst CVD ΔE {cvd:.1f} < floor {CVD_FLOOR}"
    assert cvd >= CVD_TARGET, f"{name}: worst CVD ΔE {cvd:.1f} < target {CVD_TARGET}"
    assert nrm >= NORMAL_FLOOR, f"{name}: worst normal ΔE {nrm:.1f} < {NORMAL_FLOOR}"


@pytest.mark.parametrize("name,palette", [
    ("SERIES_LINE", style.SERIES_LINE),
    ("SERIES_LABELED", style.SERIES_LABELED),
])
def test_palette_band_and_chroma(name, palette):
    for hexv in palette:
        L, C = _oklch(hexv)
        assert BAND_LIGHT[0] <= L <= BAND_LIGHT[1], f"{name} {hexv}: L={L:.3f}"
        assert C >= CHROMA_FLOOR, f"{name} {hexv}: chroma {C:.3f} reads as grey"


def test_series_line_is_contrast_safe_as_a_bare_line():
    """SERIES_LINE's whole reason to exist: it is used where there may be no
    legend, so every slot must clear 3:1 against the white plot surface.
    (SERIES_LABELED deliberately does NOT have to — it always ships a legend.)"""
    for hexv in style.SERIES_LINE:
        c = _contrast(hexv, style.SURFACE)
        assert c >= CONTRAST_MIN, f"{hexv}: {c:.2f}:1 on {style.SURFACE}"


def test_landmarks_do_not_collide_in_the_panel_they_share():
    """The piezo-space panels draw two signal traces and three landmark colours
    together.  That specific co-drawn set must separate — note it deliberately
    excludes the ROI hues, which live only on the force-extension panel (the
    union of the two sets does NOT pass, which is exactly why they are scoped
    apart rather than merged into one big palette)."""
    co_drawn = [style.SIG_MEAN_DEV, style.SIG_D1,
                style.LM_ONSET, style.LM_RUPTURE, style.LM_RUPTURE_I]
    assert _worst_cvd(co_drawn) >= CVD_TARGET
    assert _worst_normal(co_drawn) >= NORMAL_FLOOR


def test_data_marks_are_never_a_hue():
    """THE structural fix for #65 ("red fit line over red data").

    It is not "don't pick red for the fit" — that is a rule someone has to
    remember.  It is that every mark carrying DATA is neutral (chroma below the
    point where a colour reads as a hue at all), so a coloured line on a plot is
    always a model and can never be the same thing as what it is drawn over.
    Fit colours are then free, and no future slot choice can recreate the
    collision.
    """
    for name in ("DATA", "INK_STRONG", "INK_FAINT", "INK_MUTED"):
        hexv = getattr(style, name)
        _, chroma = _oklch(hexv)
        assert chroma < CHROMA_FLOOR, f"{name}={hexv} has chroma {chroma:.3f}"
    for rgba_name in ("HIT_RGBA", "NON_HIT_RGBA"):
        r, g, b, _a = getattr(style, rgba_name)
        _, chroma = _oklch("#%02x%02x%02x" % (r, g, b))
        assert chroma < CHROMA_FLOOR, f"{rgba_name} is not neutral"


def test_status_colours_are_not_series_colours():
    """Status is reserved: a status colour must never double as "series N", or
    a fit component would be indistinguishable from a pass/fail cue.

    Note the bar this sets — identity, not distance.  Slot-2 orange measures
    only 10.8 ΔE from status-critical red, and the reference palette documents
    the same overlap for its own slots; the mitigation there and here is that a
    status colour never appears alone (it carries a label, and it never shares a
    panel with a series mark as "the other line").  Asserting a 15 ΔE gap here
    would be asserting something the design does not actually claim.
    """
    status = {style.STATUS_GOOD, style.STATUS_WARNING,
              style.STATUS_CRITICAL, style._COLOR_PASS, style._COLOR_FAIL}
    for palette in (style.SERIES_LINE, style.SERIES_LABELED):
        assert not (set(palette) & status), "a status colour is being used as a series"


def test_reference_colour_is_not_an_overlay_colour():
    """Anything drawn over a 2DH that is NOT a user trace (master WLC curve,
    Δx=0 / F* anchors, ROI rectangle) uses style.REFERENCE.  It must not be a
    slot in the overlay palette, or a reference line and the Nth overlaid curve
    would be the same colour."""
    assert style.REFERENCE not in style.SERIES_LABELED


def test_2dh_ramp_is_monochrome_and_clips_to_black():
    """The ramp must stay hue-free (so the overlay palette is unconstrained by
    it) and must stop short of black, so pure black unambiguously means
    over-exposed rather than just 'a lot'."""
    lut = style.intensity_lut()
    assert lut.shape == (256, 3)
    ramp = lut[:255]
    assert (ramp[:, 0] == ramp[:, 1]).all() or _oklch(
        "#%02x%02x%02x" % tuple(ramp[128])
    )[1] < CHROMA_FLOOR, "2DH ramp must be neutral"
    assert tuple(lut[255]) == (0, 0, 0), "over-clip row must be black"
    assert tuple(ramp[-1]) != (0, 0, 0), "ramp must stop short of the clip colour"
    # monotonically darkening, so 'darker' always means 'more'
    lums = [0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in ramp.astype(float)]
    assert all(a >= b - 1e-9 for a, b in zip(lums, lums[1:]))


def test_pca_loading_colormap_is_diverging_and_zero_centered():
    """Signed loadings use blue for negative values, a neutral zero, and red
    for positive values. Each caller gets its own mutable pyqtgraph object."""
    cmap = style.pca_loading_colormap()
    lut = cmap.getLookupTable(nPts=257, alpha=False)

    assert tuple(lut[0]) == (5, 48, 97)
    assert tuple(lut[128]) == (247, 247, 247)
    assert tuple(lut[-1]) == (103, 0, 31)
    assert style.pca_loading_colormap() is not cmap


# ── (c) the ROI ordering rule ────────────────────────────────────────────────

def test_rightmost_roi_is_always_the_same_colour():
    """The bug this replaced: hues were assigned by enumerate() order, so the
    right-most ROI — the tether/final rupture, the one Ultimate points at — was
    blue on a one-ROI curve and orange on a two-ROI curve.  Colour has to
    follow the entity, not its rank in a list whose length varies per curve."""
    for n_rois in range(1, 6):
        rightmost = n_rois - 1
        assert style.roi_hue(rightmost, n_rois) == style.SERIES_LINE[0]
    for n_rois in range(2, 6):
        assert style.roi_hue(n_rois - 2, n_rois) == style.SERIES_LINE[1]
    for n_rois in range(3, 6):
        assert style.roi_hue(n_rois - 3, n_rois) == style.SERIES_LINE[2]


def test_segment_shading_darkens_towards_the_terminal_segment():
    """Within one ROI the terminal segment must be the most prominent — it is
    the one carrying the rupture the queue reports."""
    n = 4
    vals = [style.roi_segment_qcolor(0, 1, i, n).valueF() for i in range(n)]
    assert all(a > b for a, b in zip(vals, vals[1:])), vals


def test_palettes_never_invent_a_hue_when_they_wrap():
    """Past the last slot the accessors repeat the fixed order — the caller is
    told to switch to a dashed line instead (secondary encoding).  Generating
    an extra hue is how the old tab10 rota ended up with an orange and a green
    0.7 ΔE apart under protanopia."""
    n = len(style.SERIES_LABELED)
    assert style.series_labeled(n) == style.SERIES_LABELED[0]
    assert not style.series_dashed(n - 1, style.SERIES_LABELED)
    assert style.series_dashed(n, style.SERIES_LABELED)


# ── (d) typesetting ──────────────────────────────────────────────────────────

def test_math_labels_are_html_not_source_spelling():
    assert style.L_P == "<i>l</i><sub>p</sub>"
    assert style.L_C == "<i>l</i><sub>c</sub>"
    assert "<sup>*</sup>" in style.F_STAR


def test_mathify_prefers_the_longest_token():
    """'Seg l_p err (nm)' must not become 'σ' applied to half the phrase."""
    assert style.mathify("Seg l_p err (nm)") == f"Seg &sigma;({style.L_P}) (nm)"
    assert style.mathify("Seg l_c (nm)") == f"Seg {style.L_C} (nm)"


def test_mathify_leaves_plain_labels_alone():
    """Export files and table headers must keep machine-readable plain text —
    mathify is only ever applied at the plot boundary, and must be a no-op for
    anything with no math in it."""
    for plain in ("Acquisition time", "Count", "Status", "ROI Segments"):
        assert style.mathify(plain) == plain
