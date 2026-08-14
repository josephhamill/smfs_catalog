# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/style.py — the application's visual language.
#
# This is the single source for visual decisions: colour, line and marker
# weight, opacity, plot chrome, Qt text roles, small QSS treatments, decorative
# motifs, and scientific typesetting.  A window should ask for a semantic role
# (data, model, guide, warning text, caption) rather than restating how that role
# happens to look.
#
# Quick map:
#
#   A–F  palettes and fixed colour meanings
#   G–J  plot marks, series identity, overlays, and the 2DH ramp
#   K    global pyqtgraph chrome
#   L    Qt chrome and typography
#   M    scientific labels, in HTML and plain-text forms
#
# ── The three rules ───────────────────────────────────────────────────────────
#
#   DATA   is neutral grey, thin, opaque.  It is the substrate; it is never
#          coloured, so a coloured thing on a plot is always a *conclusion*.
#   MODEL  (any fit, any computed curve) is a palette hue, BOLD (3.0 px) and
#          semi-transparent (alpha 180), so it reads as an overlay and the data
#          shows through it rather than being erased by it.
#   GUIDE  (thresholds, anchors, zero lines, ROI spans) is bold-dashed (2.0 px)
#          or a <=10% fill.  Guides for the eye are allowed to be assertive.
#
# The first rule makes a coloured fit over identically coloured data
# structurally impossible: data is never a series hue, and red is reserved for
# status rather than models.
#
# ── The palettes ──────────────────────────────────────────────────────────────
#
# Both are subsets of the validated data-viz reference palette, re-checked for
# THIS app's surface (pyqtgraph background is pure white "w", not the reference
# #fcfcfb) on the ALL-PAIRS pairlist, because everything here is drawn
# simultaneously on one plot rather than as adjacent stack segments.
# tests/test_palette.py re-runs those checks — if you edit a hex here, that test
# tells you whether it still passes instead of you finding out from a colleague
# who can't read the figure.
#
# Measured (OKLab dE x100, Machado-Oliveira-Fernandes severity 1.0):
#
#   SERIES_LINE  worst CVD 13.0, worst normal-vision 16.3, all >= 3:1 on white.
#   SERIES_LABELED  worst CVD 13.0, worst normal-vision 16.3.  Two slots sit
#                below 3:1 on white, which is legal ONLY because every consumer
#                of this palette ships a legend or a coloured dot in a table.
#   LANDMARK + the two signal hues, as co-drawn in the piezo-space panels:
#                worst CVD 13.0, worst normal-vision 16.3.
#
# Colours never cycle into a new hue. Past the last slot the same hues repeat
# with a dashed line style as the secondary encoding, preserving distinguishable
# series under colour-vision deficiencies.

# ── What belongs in this file, and what does not ──────────────────────────────
#
# This module answers ONE question: *what does this look like?*  It holds data
# (colours, weights, alphas, font roles, QSS fragments, static geometry) and
# pure functions over that data.  It holds no application state or interaction.
#
# The boundary is mechanical, so it needs no judgement call and no memory:
#
#     style.py may import QtCore, QtGui and pyqtgraph.
#     style.py may NEVER import QtWidgets.
#
# Appearance needs QColor, QFont, pen styles, and QSS strings; it never needs a
# widget.  Anything that reacts to a user belongs in widgets.py or its owning
# window.  Returning an appearance value stays here; applying it is the
# caller's job.
#
# tests/test_style_is_single_source.py enforces the import boundary and prevents
# other modules from naming colours or inventing typography.

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont

# ── Surface ───────────────────────────────────────────────────────────────────

SURFACE = "#ffffff"          # pyqtgraph background ("w") — what contrast is vs.


# ── A. SERIES_LINE — identity, drawn as a bare coloured line on white ─────────
# For things that may have no legend: ROI identity, per-segment WLC fits, the
# two signal traces in a decomposition panel.  Every slot clears 3:1 on white,
# which a naked line needs and a labelled/legended mark does not.

SERIES_LINE: list[str] = [
    "#2a78d6",   # 0  blue    4.42:1
    "#eb6834",   # 1  orange  3.20:1
    "#4a3aa7",   # 2  violet  8.56:1
]

# ── B. SERIES_LABELED — identity, but always beside a legend or a table dot ───
# Fit components, 2DH trace overlays, PCA clusters, categorical bands.  Buys two
# more slots than SERIES_LINE by spending the >=3:1 contrast floor, which the
# accompanying label pays back (the data-viz "relief rule").

SERIES_LABELED: list[str] = [
    "#2a78d6",   # 0  blue
    "#eda100",   # 1  yellow    2.17:1 — relief rule: needs its legend entry
    "#e87ba4",   # 2  magenta   2.69:1 — ditto
    "#008300",   # 3  green
    "#4a3aa7",   # 4  violet
]

# ── C. Neutrals ───────────────────────────────────────────────────────────────

INK          = "#0b0b0b"     # primary text / the total-fit curve
INK_STRONG   = "#52514e"     # prominent neutral: hits, draggable thresholds
INK_MUTED    = "#898781"     # axis labels, captions, instructions
INK_FAINT    = "#b0afa9"     # recessive neutral: non-hits
GRID         = "#e1e0d9"     # hairline dividers
DATA         = "#6e6e6b"     # THE data-trace grey (rule 1)

# Established semantic aliases used directly by window modules.
_COLOR_DIVIDER = GRID
_COLOR_MUTED   = "#555555"   # unchanged: this one is UI text, not a plot ink

# ── D. Status — RESERVED.  Never a series colour, never a fit. ────────────────
# Red exists only here and as the 2DH over-exposure marker.  That reservation is
# reserved from data-series and model palettes.

STATUS_GOOD     = "#0ca30c"
STATUS_WARNING  = "#fab219"
STATUS_CRITICAL = "#d03b3b"

_COLOR_PASS = "#006400"      # dark enough to read as text
_COLOR_FAIL = "#c00000"

# ── E. Landmarks — fixed meanings, piezo-space panels only ───────────────────
# Deliberately scoped: these never share a panel with SERIES_LINE, because the
# union of the two sets does NOT pass (orange vs green is 3.2 under protanopia).
# Force-extension panels use ROI hues for their markers instead, keeping the
# ROI and WLC windows visually consistent.

LM_CONTACT   = "#008300"     # contact onset          (green — unchanged)
LM_SNAPOFF   = "#eb6834"     # snap-off               (orange — unchanged)
LM_ONSET     = "#eda100"     # ROI onset              (amber)
LM_RUPTURE   = "#008300"     # terminal/outer rupture (green — unchanged)
LM_RUPTURE_I = "#e87ba4"     # inner sub-event rupture (magenta — unchanged)
LM_THRESHOLD = INK_STRONG    # draggable threshold line

# Signal traces in the piezo panels.  Two derived signals share a panel, so they
# take identity hues rather than the neutral of rule 1.  d1 was red, which
# collided with the rupture markers drawn ON it — a collision display_roi.py's
# own comment had already noticed and half-worked-around.
SIG_MEAN_DEV = SERIES_LINE[0]   # blue
SIG_D1       = SERIES_LINE[2]   # violet
SIG_APPROACH = SERIES_LINE[1]   # orange (was 'r' — red is reserved now)
SIG_RETRACT  = SERIES_LINE[0]   # blue
SIG_DEFL     = SERIES_LINE[0]   # blue
SIG_PIEZO    = SERIES_LINE[1]   # orange (was red)
SIG_FILTERED = SERIES_LINE[2]   # violet

# ── F. Hit / non-hit — tone, not hue ─────────────────────────────────────────
# Separating populations by TONE keeps every hue free for whatever is drawn over
# them, so a fit over the hits can use any model hue without colliding.
# Measured separation is 31.8 dE and 3.61:1 contrast against
# each other.

HIT_RGBA     = (0x52, 0x51, 0x4e, 110)
NON_HIT_RGBA = (0xb0, 0xaf, 0xa9, 110)
_COLOR_FAIL_FILL_RGBA = NON_HIT_RGBA
_COLOR_ROI_FILL_RGBA  = (180, 220, 255, 55)   # translucent selection region

# ── G. Mark specs ─────────────────────────────────────────────────────────────

W_DATA   = 1.4     # data traces — thin, they are the substrate
W_MODEL  = 3.0     # fits — bold, per the brief
W_GUIDE  = 2.0     # guides for the eye — bold-dashed
W_HAIR   = 1.0     # zero lines, dividers

A_MODEL  = 180     # fits: see the data through them
A_GUIDE  = 200
A_FILL   = 40      # ROI spans / masked bands

DOT_SIZE  = 5
DOT_ALPHA = 110    # … with more transparency, so density reads as tone
DOT_LABEL_SIZE_PX = 15

# The app's quiet signature: a two-event force-extension trace, echoing the
# application icon.  It is used only as line art in genuinely empty panels,
# never behind data.  Keeping its geometry here makes the decorative motif as
# centralized and reproducible as the palette.
SIGNATURE_X = np.array([0.08, 0.24, 0.34, 0.43, 0.435, 0.50,
                        0.61, 0.615, 0.70, 0.82, 0.92])
SIGNATURE_Y = np.array([0.34, 0.34, 0.39, 0.64, 0.29, 0.35,
                        0.73, 0.27, 0.34, 0.52, 0.68])
SIGNATURE_ALPHA = 65


def rgba(color, alpha: int = 255) -> tuple[int, int, int, int]:
    """(r, g, b, alpha) from a '#rrggbb' string, an (r, g, b[, a]) tuple, or a
    QColor — callers pass all three (roi_segment_qcolor returns a QColor, the
    per-window colour constants are tuples, the palettes are hex strings)."""
    if isinstance(color, QColor):
        return (color.red(), color.green(), color.blue(), alpha)
    if isinstance(color, str):
        c = QColor(color)
        return (c.red(), c.green(), c.blue(), alpha)
    r, g, b = color[:3]
    return (int(r), int(g), int(b), alpha)


def data_pen(color: str = DATA, width: float = W_DATA) -> pg.mkPen:
    """Rule 1: the substrate.  Thin, opaque, neutral unless it's one of several
    signals sharing a panel (then pass a SERIES_LINE hue)."""
    return pg.mkPen(color, width=width)


def model_pen(color, width: float = W_MODEL, alpha: int = A_MODEL) -> pg.mkPen:
    """Rule 2: a fit.  Bold and semi-transparent so it never erases the data
    underneath it."""
    return pg.mkPen(rgba(color, alpha), width=width)


def guide_pen(color, width: float = W_GUIDE, alpha: int = A_GUIDE,
              style=Qt.PenStyle.DashLine) -> pg.mkPen:
    """Rule 3: a guide for the eye.  Bold, dashed, unmistakably not data."""
    return pg.mkPen(rgba(color, alpha), width=width, style=style)


def hair_pen(color: str = GRID, style=Qt.PenStyle.DashLine) -> pg.mkPen:
    """Zero lines and dividers — recessive chrome, not content."""
    return pg.mkPen(color, width=W_HAIR, style=style)


def signature_pen() -> pg.mkPen:
    """Recessive pen for the force-curve line-art signature in empty space."""
    return pg.mkPen(rgba(SERIES_LINE[0], SIGNATURE_ALPHA), width=W_GUIDE)


def marker_brush(color, alpha: int = 235) -> pg.mkBrush:
    return pg.mkBrush(rgba(color, alpha))


MARKER_SIZE = 11
MARKER_PEN  = pg.mkPen(INK, width=1)      # dark ring — survives any background


def band_brush(color, alpha: int = 55) -> pg.mkBrush:
    """Fill for a confidence band / ROI span.  Light enough to read through."""
    return pg.mkBrush(rgba(color, alpha))


def scatter_brush(color, alpha: int = DOT_ALPHA) -> pg.mkBrush:
    return pg.mkBrush(rgba(color, alpha))


# ── H. Overlay casing — required, not decorative ──────────────────────────────
# A coloured trace drawn over the 2DH heatmap is unreadable wherever the ramp
# passes through mid-grey: measured against a #8a8a85 cell, EVERY candidate hue
# falls below 3:1 (the best, violet, manages 2.47).  Making the ramp monochrome
# does not fix this by itself; it just moves the collision to the middle of the
# ramp.  So an overlay is drawn twice — a white casing underneath, the colour on
# top — which is legible against any ramp value including the black clip.

CASING_COLOR = "#ffffff"
W_CASING     = 5.0
W_OVERLAY    = 2.5

# The reference/anchor colour for anything drawn over a 2DH that is NOT one of
# the user's overlaid traces: the ideal-WLC master curve, the x̃=1 singularity,
# the Δx=0 / F* registration lines, the ROI selection rectangle.  Orange
# deliberately, because orange is the one SERIES_LINE hue that is NOT in
# SERIES_LABELED — so a reference line can never be mistaken for an overlay
# trace no matter how many traces are ticked.  It replaces black (now the
# over-exposure colour) and blue (now the first overlay).
REFERENCE = SERIES_LINE[1]


def casing_pen() -> pg.mkPen:
    return pg.mkPen(CASING_COLOR, width=W_CASING)


def overlay_pen(color) -> pg.mkPen:
    return pg.mkPen(rgba(color, 255), width=W_OVERLAY)


def add_cased_curve(plot, x, y, color, name: str | None = None) -> list:
    """Draw one overlay trace as casing + colour.  Returns both items so the
    caller can remove them together."""
    under = plot.plot(x, y, pen=casing_pen())
    over  = plot.plot(x, y, pen=overlay_pen(color), name=name)
    return [under, over]


# ── I. Series accessors — fixed order, never a generated hue ──────────────────

def series_line(i: int) -> str:
    return SERIES_LINE[i % len(SERIES_LINE)]


def series_labeled(i: int) -> str:
    return SERIES_LABELED[i % len(SERIES_LABELED)]


def series_dashed(i: int, palette: list[str]) -> bool:
    """True once the palette has wrapped — the caller must then dash the line.
    This is the secondary encoding that replaces inventing a new hue."""
    return i >= len(palette)


def roi_hue(roi_idx: int, n_rois: int) -> str:
    """Hue for an OUTER ROI, ranked FROM THE RIGHT.

    The right-most ROI — the tether / final rupture, the one the Ultimate
    selection points at — is ALWAYS SERIES_LINE[0] (blue); the next one left is
    always orange, then violet.  Colour follows the entity, not its rank in a
    list whose length varies per curve: before this, `enumerate(rois)` meant the
    ROI you care about was blue on a one-ROI curve and orange on a two-ROI one,
    so the same physical thing changed colour depending on what else happened to
    be detected beside it.
    """
    return series_line(max(0, n_rois - 1 - roi_idx))


def roi_segment_qcolor(roi_idx: int, n_rois: int, seg_idx: int, n_segs: int,
                       alpha: int = 255) -> QColor:
    """QColor for INNER segment `seg_idx` of an outer ROI: the ROI's hue, shaded
    so the terminal segment is darkest/most prominent and earlier ones lighter.
    Same-ROI segments therefore read as one group at a glance."""
    base = QColor(roi_hue(roi_idx, n_rois))
    h, s, v, _ = base.getHsvF()
    if n_segs > 1:
        frac = seg_idx / (n_segs - 1)            # 0 = first … 1 = terminal
        v = 0.90 - 0.42 * frac                   # first lightest → terminal darkest
        s = s * (0.80 + 0.20 * frac)
    c = QColor.fromHsvF(h, min(max(s, 0.0), 1.0), min(max(v, 0.0), 1.0))
    c.setAlpha(alpha)
    return c


# ── J. Scientific colour maps ──────────────────────────────────
# Greyscale keeps the overlay palette unconstrained. The ramp stops at #3d3d3a
# rather than running to black, so that
# pure black is unambiguously "over-exposed / clipped" and not just "a lot".

RAMP_LO   = (255, 255, 255)
RAMP_HI   = (61, 61, 58)
CLIP_RGB  = (0, 0, 0)


def intensity_lut(clip_color: tuple[int, int, int] = CLIP_RGB) -> np.ndarray:
    """(256, 3) uint8 LUT for the 2D-histogram windows: 255 greyscale steps
    white → dark grey, plus a final BLACK row for over-clip values."""
    t = np.linspace(0.0, 1.0, 255)[:, None]
    ramp = (np.array(RAMP_LO) * (1 - t) + np.array(RAMP_HI) * t).astype(np.uint8)
    return np.vstack([ramp, np.array([clip_color], dtype=np.uint8)])


# PCA loadings are signed, so they need a diverging map centred on a neutral
# zero. These are the published ColorBrewer RdBu 11-class control points;
# reversing them maps negative loadings to blue and positive loadings to red.
# Keeping the values local avoids a matplotlib dependency solely for palette
# lookup and makes the rendering deterministic on fresh installations.
_RDBU = np.array([
    [103,   0,  31], [178,  24,  43], [214,  96,  77], [244, 165, 130],
    [253, 219, 199], [247, 247, 247], [209, 229, 240], [146, 197, 222],
    [ 67, 147, 195], [ 33, 102, 172], [  5,  48,  97],
], dtype=np.uint8)


def pca_loading_colormap() -> pg.ColorMap:
    """Diverging map for PCA loadings: blue negative, neutral zero, red positive."""
    colors = _RDBU[::-1].copy()
    return pg.ColorMap(
        pos=np.linspace(0.0, 1.0, len(colors)),
        color=colors,
    )


# ── K. Plot chrome ────────────────────────────────────────────────────────────
# These are process-global pyqtgraph options.  Calling this helper before plot
# construction keeps every window on the same surface and foreground.

def apply_plot_defaults() -> None:
    """pyqtgraph's global background/foreground.  Call once per window, before
    building any PlotWidget."""
    pg.setConfigOption("background", SURFACE)
    pg.setConfigOption("foreground", INK)


# ── L. UI chrome — what Qt draws, as opposed to what pyqtgraph draws ──────────
# Everything above this line is a plot.  Everything below it is the application
# around the plot: table tints, list selections, hint captions, status text,
# font roles, and the few repeated QSS treatments worth naming.

# ── L1. Qt text and chrome colours ────────────────────────────────────────────

# Selection highlight for tables and lists.
# Applied as QSS rather than a palette role because the per-cell tints below are
# painted through BackgroundRole, which otherwise draws OVER the selection and
# makes a Ctrl/Shift selection invisible — styling only the :selected state lets
# selection win for selected cells while unselected cells keep their tint.
SELECTION_BG   = "#2d6cdf"
SELECTION_TEXT = "#ffffff"

# UI text inks.  Small informational text on a widget, NOT a plot ink — these
# are read as words at 10-11px, so they need more contrast than INK_MUTED
# (#898781), which is tuned for axis furniture.
UI_TEXT  = "#222222"    # readouts, monospace file paths — primary small text
UI_MUTED = "#555555"    # hints, captions, instructions
UI_FAINT = "#777777"    # de-emphasised secondary (empty states, bounds hints)

# Status AS TEXT.  Section D's STATUS_* are marks — STATUS_GOOD/#0ca30c and
# STATUS_WARNING/#fab219 are far too light to read as words on white.  These are
# the darkened text equivalents, which is exactly why _COLOR_PASS/_COLOR_FAIL
# provide dark text equivalents. All three clear 4.5:1 on white, as asserted
# by the guard test.
TEXT_GOOD    = _COLOR_PASS      # #006400
TEXT_WARNING = "#bb5700"
TEXT_BAD     = _COLOR_FAIL      # #c00000

# Recessive fill behind an inset readout (the worker status pill).  Chrome, not
# data and not status — it carries no meaning, it just separates a line from the
# background.
CHROME_FILL = "#f0f0f0"

# ── L2. Typography ────────────────────────────────────────────────────────────
# QFont uses points because it follows display scaling.  QSS helpers accept
# pixels because Qt stylesheet syntax does; callers should still use a named
# semantic size rather than introducing a literal at the call site.
# Typography is part of the same visual system as colour.  These are semantic
# roles, not a bag of numbers: a caption should not become a different size
# merely because it lives in a PCA window rather than an FFT window.  Keep the
# native platform UI face for controls; reserve a modern monospace face for
# numerical readouts where alignment carries information.  Qt falls back to the
# platform monospace family if Cascadia Mono is not installed.
FONT_CAPTION_PT = 8
FONT_SMALL_PT = 9
FONT_TITLE_PT = 16
FONT_MONO_FAMILY = "Cascadia Mono"


def font(base: QFont | None = None, *, size_pt: int | None = None,
         bold: bool | None = None, mono: bool = False) -> QFont:
    """Return a styled copy of ``base``; never mutate a widget's font in place."""
    out = QFont(base) if base is not None else QFont()
    if size_pt is not None:
        out.setPointSize(size_pt)
    if bold is not None:
        out.setBold(bold)
    if mono:
        out.setFamilies([FONT_MONO_FAMILY, "monospace"])
        out.setStyleHint(QFont.StyleHint.Monospace)
    return out

# ── L3. State tints and repeated QSS treatments ───────────────────────────────
# Queue/database row tints, keyed by the files.event vocabulary
# so the tint follows the verdict rather than a private list in one window.
# 'unavailable' must read as visibly different from both 'running' and a
# real 'non_event', or a disconnected drive looks like a normal queue state.
# 'unusable' is warm-vs-cool from 'unavailable' on purpose: both are
# data problems rather than verdicts, but one is expected to fix itself and
# will be retried, and the other never will. Reading them as the same colour
# would hide exactly the distinction that matters when deciding what to do.
ROW_TINT: dict[str, str] = {
    "event":       "#dcf5dc",   # pale green
    "non_event":   "#e8e8e8",   # grey
    "running":     "#fff7c8",   # pale yellow
    "unavailable": "#ffd2be",   # pale orange     — transient, will be retried
    "unusable":    "#dde4ec",   # pale blue-grey  — settled, never retried
    "default":     SURFACE,
}

# Fit-table row states (dist_fit_window / gmm_fit_window held identical copies).
TABLE_TINT_SAVED  = "#e8f4e8"   # pale green — persisted to the DB
TABLE_TINT_BEST   = "#d4edda"   # green      — best AICc
TABLE_TINT_RECENT = "#fff3cd"   # yellow     — most recent


def row_tint(event: str | None, status: str | None = None) -> str:
    """Tint for one queue/database row.  'running' outranks the verdict."""
    if status == "running":
        return ROW_TINT["running"]
    return ROW_TINT.get(event or "", ROW_TINT["default"])


def selection_qss(*widgets: str) -> str:
    """QSS making the selection highlight win over per-cell BackgroundRole
    tints.  Pass the widget classes to style, e.g. "QTableView"."""
    sel = ", ".join(f"{w}::item:selected" for w in widgets)
    return f"{sel} {{ background-color: {SELECTION_BG}; color: {SELECTION_TEXT}; }}"


LIST_QSS  = selection_qss("QListWidget")
TABLE_QSS = selection_qss("QTableView", "QTableWidget")


def qss_text(color: str = UI_MUTED, size_px: int | None = None,
             bold: bool = False, mono: bool = False) -> str:
    """QSS for a small informational label.  Callers say what the text IS
    (muted hint, status) rather than picking a grey each time."""
    parts = [f"color: {color};"]
    if size_px is not None:
        parts.append(f"font-size: {size_px}px;")
    if bold:
        parts.append("font-weight: bold;")
    if mono:
        parts.append(f'font-family: "{FONT_MONO_FAMILY}", monospace;')
    return " ".join(parts)


QSS_EMPHASIS = "font-weight: bold;"
QSS_PRIMARY_ACTION = "font-weight: bold; font-size: 13px; padding: 5px;"
QSS_COLLAPSIBLE_HEADER = (
    "QToolButton { border: none; font-weight: bold; padding: 2px; }"
)


def qss_emphasis(color: str | None = None, selector: str | None = None) -> str:
    """Bold text, optionally colored and scoped to a QSS selector."""
    body = QSS_EMPHASIS + (f" color: {color};" if color else "")
    return f"{selector} {{ {body} }}" if selector else body


def qss_inset(*, fill: bool = False) -> str:
    """Compact padded readout; optionally set it off with a quiet fill."""
    parts = ["padding: 3px 8px;"]
    if fill:
        parts.extend((f"background: {CHROME_FILL};", "border-radius: 4px;"))
    return " ".join(parts)


def html_text(text: str, color: str = UI_MUTED, bold: bool = False) -> str:
    """Inline-coloured span for a rich-text QLabel (the dashboard status line).
    Same colours as qss_text — a status word must not change meaning depending
    on whether the label happens to be rich text."""
    weight = " font-weight:bold;" if bold else ""
    return f"<span style='color:{color};{weight}'>{text}</span>"


# ── M. Typesetting ────────────────────────────────────────────────────────────
# pyqtgraph renders axis labels and plot titles as HTML (AxisItem.labelString
# wraps them in a <span>; _updateLabel calls setHtml), so real subscripts and
# superscripts work — the app just never used them and shipped "l_p" as literal
# source-code spelling on publication figures.  ISO convention: the variable is
# italic, its descriptive subscript upright.
#
# QTableWidget headers do NOT render rich text, and Unicode can't stand in
# (there is a subscript p, U+209A, but no subscript c at all), so table headers
# keep the plain spelling.  The one place it matters is deliberate, not an
# oversight.

def var(name: str, sub: str = "", sup: str = "") -> str:
    """Typeset a variable: italic base, upright sub/superscript."""
    out = f"<i>{name}</i>"
    if sub:
        out += f"<sub>{sub}</sub>"
    if sup:
        out += f"<sup>{sup}</sup>"
    return out


L_P      = var("l", "p")             # persistence length
L_C      = var("l", "c")             # contour length
F_STAR   = var("F", sup="*")
X_STAR   = var("x", sup="*")
X_TILDE  = "<i>x&#771;</i>"          # x with combining tilde
DELTA_X  = f"&Delta;{var('x')}"
DELTA_F  = f"&Delta;{var('F')}"
FORCE    = var("F")
EXTENSION = var("x")


def pm(value: str, err: str) -> str:
    return f"{value} &plusmn; {err}"


# ── Plain twins, for the surfaces that do NOT render HTML ─────────────────────
# The typeset constants above are HTML, and the note at the top of this section
# is right that pyqtgraph renders axis labels and titles as HTML. A pg.InfiniteLine
# label reaches InfLineLabel.valueChanged -> TextItem.setText -> setPlainText,
# so the normalized 2DH's singularity line rendered, on screen, as the literal
# characters "<i>x&#771;</i> = 1 (<i>l</i><sub>c</sub>)".
#
# Calling setHtml() on the label afterwards does not hold: valueChanged() rewrites
# it as plain text every time the line moves, which physical's F* line does.
# Plain text is the only stable answer, so the plain spelling lives here beside
# the typeset one and cannot drift from it.
#
# Unicode gets us the tilde but not the subscript: there is a subscript p
# (U+209A) and no subscript c at all, so "l_c" is the honest spelling — the same
# one the queue table headers and every export file already use.
X_TILDE_PLAIN = "x̃"
L_P_PLAIN     = "l_p"
L_C_PLAIN     = "l_c"
F_STAR_PLAIN  = "F*"
X_STAR_PLAIN  = "x*"
DELTA_X_PLAIN = "Δx"
DELTA_F_PLAIN = "ΔF"


# Plain column/variable labels (the spellings the queue table and the export
# files use, which must stay plain text) → their typeset form, for the one
# place the same string is reused as a PLOT axis label.  Longest first so
# "l_p err" wins over "l_p".
_MATH_TOKENS = [
    ("l_p err", f"&sigma;({L_P})"),
    ("l_c err", f"&sigma;({L_C})"),
    ("l_p",     L_P),
    ("l_c",     L_C),
    ("ΔF",      DELTA_F),
    ("ΔX",      DELTA_X),
    ("Δx",      DELTA_X),
    ("F*",      F_STAR),
]


def mathify(label: str) -> str:
    """Typeset a plain variable label for use on a plot.

    Deliberately NOT applied to exported files, table headers, or provenance
    strings — those must stay machine-readable plain text, and QTableWidget
    headers cannot render rich text anyway (nor can Unicode help: there is a
    subscript p, but no subscript c at all).  Plots are the one surface that
    both renders HTML and is read by humans as a figure.
    """
    out = label
    for plain, html in _MATH_TOKENS:
        if plain in out:
            out = out.replace(plain, html)
    return out
