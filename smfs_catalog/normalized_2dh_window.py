# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/normalized_2dh_window.py
#
# Normalized2DHWindow — cohort 2D histogram (mean bin count per contributing
# trace) in normalized WLC coordinates (x̃ = x/l_c, F̃ = F·l_p/kT).
#
# Grid parameters (bins, axis ranges, fit segment) are configurable per-user
# via the "Grid settings…" dialog and persisted in experimentalist_profiles.
#
# Shares its rebuild loop, selection-window/PCA machinery, grid persistence,
# and trace overlay with physical_2dh_window.py through _TwoDHWindowBase.
# Segment selection is configurable like the physical window. The physical-only
# "align mode" menu does not generalize here: it shifts x, which would move
# the WLC singularity off x̃=1 for every curve differently and break the
# universal-collapse property the master curve below depicts. The fixed
# x_range/f_range clips the full retract to the displayed domain.

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt

from . import style
from .base_2dh_window import _TwoDHWindowBase, _GridDialog
from .event_processor import (
    WLC_X_BINS, WLC_F_BINS, WLC_X_RANGE, WLC_F_RANGE,
    _wlc_grid_params, compute_wlc_histogram, ALIGN_SEG_DEFAULT,
)

# Both 2DH windows use the same default segment.
_DEFAULT_ALIGN_SEG = ALIGN_SEG_DEFAULT


class Normalized2DHWindow(_TwoDHWindowBase):
    """Normalized 2D histogram for one Event Summary population."""

    def __init__(
        self,
        prepass_results: list[dict],
        db_path:         str,
        session_info:    dict | None = None,
        experimentalist:    str | None  = None,
        population:      str = "hit",
    ) -> None:
        super().__init__(
            prepass_results, db_path, session_info, experimentalist,
            window_title="SMFS — normalized 2DH",
            population=population,
        )

    # ── Grid params ───────────────────────────────────────────────────────────

    def _profile_spec(self):
        return [
            ("_x_bins", "wlc_x_bins", int,   WLC_X_BINS),
            ("_f_bins", "wlc_f_bins", int,   WLC_F_BINS),
            ("_x_min",  "wlc_x_min",  float, WLC_X_RANGE[0]),
            ("_x_max",  "wlc_x_max",  float, WLC_X_RANGE[1]),
            ("_f_min",  "wlc_f_min",  float, WLC_F_RANGE[0]),
            ("_f_max",  "wlc_f_max",  float, WLC_F_RANGE[1]),
            ("_align_segment", "wlc_align_segment", str, _DEFAULT_ALIGN_SEG),
        ]

    @property
    def _grid_key(self) -> str:
        return _wlc_grid_params(
            x_bins=self._x_bins, f_bins=self._f_bins,
            x_range=(self._x_min, self._x_max),
            f_range=(self._f_min, self._f_max),
            align_segment=self._align_segment,
        )

    def _axis_labels(self) -> tuple[tuple[str, str], tuple[str, str]]:
        # Both axes are dimensionless BY CONSTRUCTION (x/l_c and F·l_p/kT),
        # which is why the unit is "" rather than missing — see quantities.py.
        return ((f"{style.X_TILDE}  ({style.EXTENSION} / {style.L_C})", ""),
                (f"<i>F&#771;</i>  ({style.FORCE} · {style.L_P} / <i>kT</i>)", ""))

    def _make_grid_dialog(self) -> _GridDialog:
        return _GridDialog(
            self, self._x_bins, self._f_bins, self._x_min, self._x_max,
            self._f_min, self._f_max, self._align_segment,
            defaults={
                "x_bins": WLC_X_BINS, "f_bins": WLC_F_BINS,
                "x_min": WLC_X_RANGE[0], "x_max": WLC_X_RANGE[1],
                "f_min": WLC_F_RANGE[0], "f_max": WLC_F_RANGE[1],
                "align_segment": _DEFAULT_ALIGN_SEG,
            },
        )

    # ── Coordinate transform ─────────────────────────────────────────────────
    # No anchor/shift here, ever — see module docstring. lo/hi (the ROI span)
    # are unused now that the data window is the full retract; kept in the
    # shared signature so both windows implement the same hooks.

    def _build_histogram(self, x, F, lo, hi, l_p, l_c, right_idx):
        if l_p is None or l_c is None:
            return None
        return compute_wlc_histogram(
            x, F, l_p, l_c,
            x_bins=self._x_bins, f_bins=self._f_bins,
            x_range=(self._x_min, self._x_max),
            f_range=(self._f_min, self._f_max),
        )

    def _build_overlay_xF(self, x, F, lo, hi, l_p, l_c, right_idx):
        if l_p is None or l_c is None:
            return None
        from .models import normalize_wlc
        return normalize_wlc(x, F, l_p, l_c)

    # ── Master WLC curve overlay (universal collapse line) ───────────────────

    def _after_plot_setup(self) -> None:
        x_master = np.linspace(0.02, 0.96, 500)
        F_master  = 1.0 / (4.0 * (1.0 - x_master) ** 2) - 0.25 + x_master
        master_curve = self._plot.plot(
            x_master.tolist(), F_master.tolist(),
            pen=pg.mkPen(style.rgba(style.REFERENCE, 230),
                         width=style.W_MODEL, style=Qt.PenStyle.DashLine),
        )
        # A standalone LegendItem, NOT plotItem.legend (self._plot.addLegend())
        # — the shared TraceOverlayPanel (base_2dh_window.py) adds each
        # overlaid curve via addItem(..., name=filename), and pyqtgraph's
        # PlotItem.addItem auto-registers any named item into plotItem.legend
        # whenever one exists. A standalone LegendItem sidesteps that, so
        # this legend only ever shows the one reference curve, not a growing
        # list of overlay filenames every time a checkbox is ticked.
        legend = pg.LegendItem(offset=(-10, 10))
        legend.setParentItem(self._plot.getPlotItem().getViewBox())
        legend.addItem(master_curve, "ideal WLC (master curve)")

        # x̃=1 is the WLC singularity every curve is normalized to align on —
        # the whole point of this window. The master curve above is plotted
        # only up to x̃=0.96 (the model diverges at 1) with nothing marking
        # where/why it stops; this line makes that explicit.
        singularity = pg.InfiniteLine(
            pos=1.0, angle=90, movable=False,
            pen=pg.mkPen(style.rgba(style.REFERENCE, style.A_GUIDE),
                         width=style.W_GUIDE, style=Qt.PenStyle.DotLine),
            # InfiniteLine labels require plain-text symbol aliases.
            label=f"{style.X_TILDE_PLAIN} = 1 ({style.L_C_PLAIN})",
            labelOpts={"position": 0.92, "color": style.REFERENCE},
        )
        self._plot.addItem(singularity)
