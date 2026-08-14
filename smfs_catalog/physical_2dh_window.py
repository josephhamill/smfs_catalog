# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/physical_2dh_window.py
#
# Physical2DHWindow — cohort 2D histogram (mean bin count per contributing
# trace) in force-registered physical coordinates:
#
#   Δx = x − anchor   (nm)   anchor chosen by the "Align at" menu
#   F  = k · δ            (pN)   unscaled
#
# Grid parameters (bins, axis ranges, F*, align mode/segment) are
# configurable per-user via the "Grid settings…" dialog and persisted in
# experimentalist_profiles.
#
# Shares its rebuild loop, selection-window/PCA machinery, grid persistence,
# and trace overlay with Normalized2DHWindow through _TwoDHWindowBase. The
# anchor-shift menu is physical-only because shifting normalized extension
# would destroy alignment of the WLC singularity at x/l_c = 1.

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QPushButton

from . import style
from . import quantities as _quant
from .base_2dh_window import _TwoDHWindowBase, _GridDialog
from .event_processor import (
    PHYS_X_BINS, PHYS_F_BINS, PHYS_X_RANGE, PHYS_F_RANGE,
    _phys_grid_params, compute_physical_histogram_at,
    PHYS_ALIGN_ANCHORS, PHYS_ALIGN_DEFAULT,
    ALIGN_SEG_DEFAULT, PHYS_F_STAR_DEFAULT,
)

# "Align at" maps one-to-one onto the event_processor.phys_anchor_* subroutines.
_ALIGN_MODE_CHOICES = [
    ("Onset (loading start)",     "onset"),
    ("F* (force)",                "fstar"),
    ("Snap-off (contact)",        "snapoff"),
    ("Contour length l_c",        "lc"),
    ("Rupture (segment's end)",   "rupture"),
]
# Defaults are shared with the normalized window through event_processor.
_DEFAULT_ALIGN_MODE = PHYS_ALIGN_DEFAULT
_DEFAULT_ALIGN_SEG  = ALIGN_SEG_DEFAULT
_DEFAULT_F_STAR     = PHYS_F_STAR_DEFAULT

# Single source of truth for "what Δx=0 means under this align mode" —
# consumed by BOTH the axis title (_axis_labels) and the vline's on-canvas
# annotation (_apply_align_visuals).
# Values are HTML (style.py § K) — pyqtgraph renders both the axis label and
# the InfiniteLine's on-canvas annotation as rich text.
_ALIGN_MODE_ANCHOR_DESC = {
    "onset":   "onset",
    "fstar":   f"{style.X_STAR}({style.F_STAR})",
    "snapoff": "contact",
    "lc":      style.L_C,
    "rupture": "rupture",
}


class _PhysicalGridDialog(_GridDialog):
    """_GridDialog plus the two fields only a shift-based, physical-units
    transform needs: F* (the alignment force) and the "Align at" anchor menu."""

    _TITLE = "Physical grid settings"

    def __init__(self, parent, x_bins, f_bins, x_min, x_max, f_min, f_max,
                 align_segment, f_star, align_mode, defaults):
        self._f_star_init    = f_star
        self._align_mode_init = align_mode
        super().__init__(parent, x_bins, f_bins, x_min, x_max, f_min, f_max,
                          align_segment, defaults)

    def _x_bins_label(self) -> str:
        return "X bins (Δx):"

    def _f_bins_label(self) -> str:
        return "F bins (F):"

    # Physical axes carry real units, and the spin boxes' own suffixes already
    # say " nm"/" pN" — so these stay plain, unlike the normalized window's.
    def _x_range_label(self) -> str:
        return "X range:"

    def _f_range_label(self) -> str:
        return "F range:"

    def _range_spec(self) -> tuple[tuple[float, float, str], ...]:
        # Real units on both axes, so one wide range and the phys_* keys —
        # no widget code here, that lives once in _GridDialog.
        return tuple((-5000.0, 5000.0, k) for k in
                     ("phys_x_min", "phys_x_max", "phys_f_min", "phys_f_max"))

    def _add_extra_rows(self) -> None:
        self._f_star = QDoubleSpinBox()
        self._f_star.setRange(1.0, 500.0)
        _quant.configure_spinbox(self._f_star, "phys_f_star")
        self._f_star.setValue(self._f_star_init)

        self._align = QComboBox()
        for lbl, key in _ALIGN_MODE_CHOICES:
            self._align.addItem(lbl, key)
        i = self._align.findData(self._align_mode_init)
        self._align.setCurrentIndex(i if i >= 0 else 0)

        self._form.addRow("F* (alignment force):", self._f_star)
        self._form.addRow("Align at:",             self._align)

    def _reset(self) -> None:
        super()._reset()
        d = self._defaults
        self._f_star.setValue(d["f_star"])
        self._align.setCurrentIndex(max(0, self._align.findData(d["align_mode"])))

    @property
    def values(self) -> dict:
        v = super().values
        v["f_star"] = self._f_star.value()
        v["align_mode"] = self._align.currentData()
        return v


class Physical2DHWindow(_TwoDHWindowBase):
    """Force-registered 2D histogram for one Event Summary population."""

    _physical = True

    def __init__(
        self,
        prepass_results: list[dict],
        db_path:         str,
        session_info:    dict | None = None,
        experimentalist:    str | None  = None,
        population:      str = "hit",
    ) -> None:
        self._mean_wins: list = []
        super().__init__(
            prepass_results, db_path, session_info, experimentalist,
            window_title="SMFS — physical 2DH",
            population=population,
        )

    # ── Grid params ───────────────────────────────────────────────────────────

    def _profile_spec(self):
        return [
            ("_x_bins", "phys_x_bins", int,   PHYS_X_BINS),
            ("_f_bins", "phys_f_bins", int,   PHYS_F_BINS),
            ("_x_min",  "phys_x_min",  float, PHYS_X_RANGE[0]),
            ("_x_max",  "phys_x_max",  float, PHYS_X_RANGE[1]),
            ("_f_min",  "phys_f_min",  float, PHYS_F_RANGE[0]),
            ("_f_max",  "phys_f_max",  float, PHYS_F_RANGE[1]),
            ("_f_star", "phys_f_star", float, _DEFAULT_F_STAR),
            ("_align_mode",    "phys_align_mode",    str, _DEFAULT_ALIGN_MODE),
            ("_align_segment", "phys_align_segment", str, _DEFAULT_ALIGN_SEG),
        ]

    @property
    def _grid_key(self) -> str:
        return _phys_grid_params(
            self._f_star,
            x_bins=self._x_bins, f_bins=self._f_bins,
            x_range=(self._x_min, self._x_max),
            f_range=(self._f_min, self._f_max),
            align_mode=self._align_mode, align_segment=self._align_segment,
        )

    def _axis_labels(self) -> tuple[tuple[str, str], tuple[str, str]]:
        desc = _ALIGN_MODE_ANCHOR_DESC.get(self._align_mode, self._align_mode)
        return ((f"{style.DELTA_X}  ({style.EXTENSION} − {desc})", _quant.NM),
                (style.FORCE, _quant.PN))

    def _make_grid_dialog(self) -> _PhysicalGridDialog:
        return _PhysicalGridDialog(
            self, self._x_bins, self._f_bins, self._x_min, self._x_max,
            self._f_min, self._f_max, self._align_segment,
            self._f_star, self._align_mode,
            defaults={
                "x_bins": PHYS_X_BINS, "f_bins": PHYS_F_BINS,
                "x_min": PHYS_X_RANGE[0], "x_max": PHYS_X_RANGE[1],
                "f_min": PHYS_F_RANGE[0], "f_max": PHYS_F_RANGE[1],
                "align_segment": _DEFAULT_ALIGN_SEG,
                "f_star": _DEFAULT_F_STAR, "align_mode": _DEFAULT_ALIGN_MODE,
            },
        )

    def _apply_extra_dialog_values(self, v: dict) -> None:
        self._f_star     = v["f_star"]
        self._align_mode = v["align_mode"]

    def _after_grid_settings_applied(self) -> None:
        self._hline.setValue(self._f_star)
        self._apply_align_visuals()

    def _provenance_extra(self) -> str:
        desc = _ALIGN_MODE_ANCHOR_DESC.get(self._align_mode, self._align_mode)
        if self._align_mode == "fstar":
            f_star = _quant.format_value('phys_f_star', self._f_star, with_unit=True)
            return f"align: F* = {f_star}"
        return f"align: {desc}"

    def _export_provenance_extra(self) -> dict:
        d = {"align_mode": self._align_mode}
        if self._align_mode == "fstar":
            d["f_star_pN"] = self._f_star
        return d

    # ── Registration reference lines ─────────────────────────────────────────

    def _after_plot_setup(self) -> None:
        self._vline = pg.InfiniteLine(
            pos=0, angle=90, movable=False,
            pen=pg.mkPen(style.rgba(style.REFERENCE, style.A_GUIDE),
                         width=style.W_GUIDE, style=Qt.PenStyle.DashLine),
            # Plain-text aliases are required by InfiniteLine labels.
            label=f"{style.X_STAR_PLAIN}({style.F_STAR_PLAIN})",
            labelOpts={"position": 0.92, "color": style.REFERENCE},
        )
        self._hline = pg.InfiniteLine(
            pos=self._f_star, angle=0, movable=False,
            pen=pg.mkPen(style.rgba(style.REFERENCE, style.A_GUIDE),
                         width=style.W_GUIDE, style=Qt.PenStyle.DashLine),
            label=style.F_STAR_PLAIN,
            labelOpts={"position": 0.05, "color": style.REFERENCE},
        )
        self._plot.addItem(self._vline)
        self._plot.addItem(self._hline)
        self._apply_align_visuals()

    def _apply_align_visuals(self) -> None:
        """Label the Δx=0 line for the current alignment mode, and show the F*
        force line only when actually registering on F*. Uses the same
        _ALIGN_MODE_ANCHOR_DESC as _axis_labels() so the vline and the axis
        title never disagree about what the current anchor is."""
        desc = _ALIGN_MODE_ANCHOR_DESC.get(self._align_mode, self._align_mode)
        label = desc if self._align_mode == "fstar" else f"{desc} (Δx=0)"
        try:
            self._vline.label.setFormat(label)
        except Exception:
            pass
        self._hline.setVisible(self._align_mode == "fstar")

    # ── Coordinate transform ─────────────────────────────────────────────────

    def _compute_anchor(self, x, F, lo, hi, l_p, l_c, right_idx):
        """The extension x (nm) that maps to Δx=0 for this curve, per the
        current align mode. l_p/l_c/right_idx are the CHOSEN segment's stored
        values, already resolved upstream by base._resolve_fit."""
        x_roi, F_roi = x[lo:hi + 1], F[lo:hi + 1]
        a_lp, a_lc, a_rupture_x = None, None, None
        if self._align_mode == "rupture":
            if right_idx is None or not (0 <= right_idx < len(x)):
                return None
            a_rupture_x = float(x[right_idx])
        elif self._align_mode in ("fstar", "lc"):
            if l_p is None or l_c is None:
                return None   # fstar/lc are undefined without a real fit
            a_lp, a_lc = l_p, l_c
        anchor_fn = PHYS_ALIGN_ANCHORS.get(self._align_mode, PHYS_ALIGN_ANCHORS[PHYS_ALIGN_DEFAULT])
        return anchor_fn(x_roi, F_roi, a_lp, a_lc, self._f_star, a_rupture_x)

    # Which anchors actually consume the WLC fit.  onset/snapoff/rupture are
    # all observed data points, so requiring l_p/l_c for them would discard
    # curves whose registration itself is fully defined.
    _FIT_DEPENDENT_ALIGN_MODES = frozenset({"fstar", "lc"})

    def _requires_wlc_fit(self) -> bool:
        return self._align_mode in self._FIT_DEPENDENT_ALIGN_MODES

    def _build_histogram(self, x, F, lo, hi, l_p, l_c, right_idx):
        anchor = self._compute_anchor(x, F, lo, hi, l_p, l_c, right_idx)
        if anchor is None:
            return None
        return compute_physical_histogram_at(
            x, F, anchor,
            x_bins=self._x_bins, f_bins=self._f_bins,
            x_range=(self._x_min, self._x_max),
            f_range=(self._f_min, self._f_max),
        )

    def _build_overlay_xF(self, x, F, lo, hi, l_p, l_c, right_idx):
        anchor = self._compute_anchor(x, F, lo, hi, l_p, l_c, right_idx)
        if anchor is None:
            return None
        return x - anchor, F

    # ── Gaussian ridge / WLC (physical only) ─────────────────────────────────

    def _build_extra_controls(self, ctrl_layout) -> None:
        self._mean_btn = QPushButton("Gaussian ridge…")
        self._mean_btn.setEnabled(False)
        self._mean_btn.setToolTip(
            "Per-column Gaussian ridge + WLC fit on this 2DH "
            "(the selected 2DH area if one is drawn, else the full Total)."
        )
        self._mean_btn.clicked.connect(self._open_mean_curve)
        ctrl_layout.addWidget(self._mean_btn)
        ctrl_layout.addSpacing(6)

    def _on_refresh_extra(self, n: int) -> None:
        self._mean_btn.setEnabled(n > 0)

    def _open_mean_curve(self) -> None:
        """Open the Gaussian-ridge/WLC view on the selected 2DH area, or on the
        full displayed 2DH when no area is selected."""
        from .mean_curve_window import MeanCurveWindow

        if self._cumulative is None:
            return

        counts  = self._cumulative
        x_range = (self._x_min, self._x_max)
        f_range = (self._f_min, self._f_max)
        n       = len(self._event_histograms)
        title   = f"Total 2DH  (n={n})"

        if self._selection is not None:
            state = self._selection.getState()
            rx, ry = float(state['pos'][0]), float(state['pos'][1])
            rw, rh = float(state['size'][0]), float(state['size'][1])
            x_lo = min(rx, rx + rw);  x_hi = max(rx, rx + rw)
            f_lo = min(ry, ry + rh);  f_hi = max(ry, ry + rh)
            dx = (self._x_max - self._x_min) / self._x_bins
            df = (self._f_max - self._f_min) / self._f_bins
            xi_lo = max(0, int((x_lo - self._x_min) / dx))
            xi_hi = min(self._x_bins, int(np.ceil((x_hi - self._x_min) / dx)))
            fi_lo = max(0, int((f_lo - self._f_min) / df))
            fi_hi = min(self._f_bins, int(np.ceil((f_hi - self._f_min) / df)))
            if xi_hi > xi_lo and fi_hi > fi_lo:
                counts  = self._cumulative[xi_lo:xi_hi, fi_lo:fi_hi]
                x_range = (self._x_min + xi_lo * dx, self._x_min + xi_hi * dx)
                f_range = (self._f_min + fi_lo * df, self._f_min + fi_hi * df)
                title   = f"Selection-window 2DH  (n={n})"

        display = np.sqrt(counts)
        x_label, f_label = self._axis_labels()
        win = MeanCurveWindow(
            title    = title,
            display  = display,
            counts   = counts,
            auto_max = max(float(display.max()), 1e-10),
            z_pct    = self._z_pct,
            lut      = self._lut,
            x_range  = x_range,
            f_range  = f_range,
            x_label  = x_label,
            f_label  = f_label,
            caption  = self._provenance_caption(),
            physical = True,
            paths      = list(self._event_histograms.keys()),
            overlay_fn = self._overlay_xF,
            db_path    = self._db_path,
            provenance = self.export_provenance(),
        )
        self._mean_wins.append(win)
        win.show()
