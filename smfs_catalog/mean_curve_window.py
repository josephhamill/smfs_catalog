# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/mean_curve_window.py
#
# MeanCurveWindow — a single 2D-histogram view with a per-column Gaussian
# ridge overlay (force vs extension) and, on physical axes, a
# Marko-Siggia WLC fit over a draggable extension region. Optionally also a
# TraceOverlayPanel (checkbox list of the individual curves behind this
# histogram, ticked ones plotted on top) so a popped-out
# cluster/Total view carries the same per-curve overlay + filename access as
# the live 2DH windows, without owning any DB/alignment logic itself: the
# caller passes `paths` (this histogram's contributing files) and
# `overlay_fn` (how to fetch one curve's already-transformed (x, F) trace —
# in practice a bound _TwoDHWindowBase._overlay_xF, which already knows the
# alignment settings that produced this histogram).
#
# Shared by:
#   • Physical2DHWindow — the Total / selection-window 2DH
#   • PCAWindow         — per-cluster and Total cluster popouts
#
# NOTE on the WLC fit: a single-WLC fit of a *total-2DH* ridge is only
# well-posed when the population is near-monodisperse in contour length.
# Heterogeneous l_c (even a few %) makes the ridge non-WLC and the extracted
# l_p / l_c effective shape parameters, not molecular constants.

from __future__ import annotations

from typing import Callable

import numpy as np
import pyqtgraph as pg
from scipy.optimize import curve_fit

from . import style
from . import export_utils as _export
from . import models
# One implementation of "sample a fit's covariance for a pointwise band",
# shared rather than re-derived here.
from .dist_fit_core import ci_manifest_fields, total_fit_ci as _fit_ci_band
from .export_utils import slug as _slug
from .qt_utils import FixedDomainPlot, set_plot_title, set_si_label, fit_on_screen
from .quantities import format_value as _q   # ONE formatter: unit and
# meaningful digits come from quantities.py, so the same measurement
# cannot print as 166, 166.2 and 166.20 in three different windows.
from .trace_overlay_panel import TraceOverlayPanel
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


def _gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _bin_centres(bounds: tuple[float, float], n_bins: int) -> np.ndarray:
    """Centres of equal-width histogram bins within ``bounds``."""
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    lo, hi = bounds
    width = (hi - lo) / n_bins
    return lo + (np.arange(n_bins, dtype=float) + 0.5) * width


def _column_gaussian_means(
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a Gaussian to each row's force profile to trace its ridge.

    `counts` is a raw 2D histogram with shape (n_x, n_f): rows index the
    x / extension axis, columns the F / force axis. For each row the force
    distribution is fit with amp·exp(-½((c-μ)/σ)²), seeded from the
    intensity-weighted moments.

    Returns (centre_col, sigma_col), each length n_x in *column-index* units,
    NaN where the column is empty or the fit fails to converge. This is a
    descriptive ridge estimator, not an inferential population mean.
    """
    n_x, n_f = counts.shape
    col   = np.arange(n_f, dtype=float)
    mean  = np.full(n_x, np.nan)
    sigma = np.full(n_x, np.nan)
    if n_f < 3:  # Three Gaussian parameters cannot be identified below this.
        return mean, sigma
    for r in range(n_x):
        y   = counts[r]
        tot = y.sum()
        if tot <= 0:
            continue
        mu0  = np.average(col, weights=y)
        var0 = np.average((col - mu0) ** 2, weights=y)
        sd0  = np.sqrt(var0) if var0 > 0 else 1.0
        try:
            popt, _ = curve_fit(
                _gaussian, col, y,
                p0=[float(y.max()), mu0, max(sd0, 0.5)],
                bounds=([0.0, 0.0, 0.25],
                        [np.inf, float(n_f - 1), float(max(n_f, 1))]),
                maxfev=2000,
            )
        except (RuntimeError, ValueError):
            continue
        mean[r]  = popt[1]
        sigma[r] = abs(popt[2])
    return mean, sigma


class MeanCurveWindow(QMainWindow):
    """
    Single 2DH view (Total, selection window, or a cluster) with a per-column
    Gaussian ridge and descriptive ±1σ band. On physical axes a Marko-Siggia WLC
    model can be fit to the ridge over a draggable extension region.
    """

    def __init__(
        self,
        title:    str,
        display:  np.ndarray,
        counts:   np.ndarray,
        auto_max: float,
        z_pct:    int,
        lut:      np.ndarray,
        x_range:  tuple[float, float],
        f_range:  tuple[float, float],
        x_label:  str,
        f_label:  str,
        x_unit:   str = "",
        f_unit:   str = "",
        caption:    str = "",
        physical:   bool = False,
        paths:      "list[str] | None" = None,
        overlay_fn: "Callable[[str], tuple | None] | None" = None,
        db_path:    "str | None" = None,
        provenance: dict | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("SMFS — mean curve")
        self.setWindowFlag(Qt.WindowType.Window)
        fit_on_screen(self, 640 + (190 if paths else 0), 560)
        self._display  = display
        self._counts   = counts
        self._auto_max = auto_max
        self._z_pct    = z_pct
        self._lut      = lut
        self._x_range  = x_range
        self._f_range  = f_range
        self._physical = physical
        self._title    = title
        self._caption  = caption
        self._x_label  = x_label
        self._f_label  = f_label
        self._x_unit   = x_unit
        self._f_unit   = f_unit
        self._db_path  = db_path
        self._paths    = list(paths) if paths else []
        # The parent 2DH/PCA window's own export_provenance(), so a mean-curve
        # export says which population/grid/alignment produced the histogram
        # underneath it — this window computes a curve, it does not know where
        # the histogram came from.
        self._provenance = dict(provenance) if provenance else {}
        # Last WLC fit, kept for the export: it is a RESULT (l_p/l_c of the
        # ridge).
        self._wlc_fit: dict | None = None
        self._wlc_curve: tuple | None = None   # (x, y, band) as drawn
        self._wlc_pcov = None                  # full parameter covariance

        # Physical axis geometry. The per-column Gaussian ridge is computed
        # in _recompute_mean() so it can be refreshed when the corner mask moves.
        n_x, n_f = counts.shape
        x0, x1 = x_range
        f0, f1 = f_range
        self._f0        = f0
        self._f_per_col = (f1 - f0) / n_f
        self._x_phys = _bin_centres(x_range, n_x)
        self._f_col  = _bin_centres(f_range, n_f)
        self._f_mean = np.full(n_x, np.nan)
        self._f_sig  = np.full(n_x, np.nan)
        self._valid  = np.zeros(n_x, dtype=bool)

        self._mean_items: list = []
        self._wlc_items:  list = []
        self._region = None
        self._corner = None

        central = QWidget()
        self.setCentralWidget(central)
        lay = QVBoxLayout(central)
        lay.setSpacing(4)
        lay.setContentsMargins(8, 6, 8, 6)

        # ── Row 1: intensity clip ─────────────────────────────────────────────
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Clip:"))
        self._z_slider = QSlider(Qt.Orientation.Horizontal)
        self._z_slider.setRange(1, 200)
        self._z_slider.setValue(z_pct)
        self._z_slider.setFixedWidth(150)
        self._z_slider.setToolTip(
            "Intensity clip: drag left to highlight dense regions (clipped values shown in red)"
        )
        self._z_label = QLabel(f"{z_pct}%")
        self._z_label.setMinimumWidth(36)
        auto_btn = QPushButton("Auto")
        auto_btn.setFixedWidth(42)
        self._z_slider.valueChanged.connect(self._on_slider)
        auto_btn.clicked.connect(lambda: self._z_slider.setValue(100))
        ctrl.addWidget(self._z_slider)
        ctrl.addSpacing(4)
        ctrl.addWidget(self._z_label)
        ctrl.addWidget(auto_btn)
        ctrl.addStretch()
        lay.addLayout(ctrl)

        # ── Row 2: Gaussian ridge + WLC ───────────────────────────────────────
        row2 = QHBoxLayout()
        self._mean_chk = QCheckBox("Gaussian ridge")
        self._mean_chk.setChecked(True)
        self._mean_chk.toggled.connect(self._draw_mean)
        row2.addWidget(self._mean_chk)
        self._corner_chk = QCheckBox("Corner mask")
        self._corner_chk.setToolTip(
            "Drag the corner marker to the upper-left of the baseline; everything "
            "to its right and below is excluded from the per-column Gaussian fit"
        )
        self._corner_chk.toggled.connect(self._on_corner_toggle)
        row2.addWidget(self._corner_chk)
        spread_label = QLabel("band: ±1σ")
        spread_label.setToolTip(
            "Descriptive width of the Gaussian fitted to each force-bin profile; "
            "not a population confidence interval."
        )
        row2.addWidget(spread_label)

        if physical:
            row2.addSpacing(12)
            self._wlc_btn = QPushButton("Fit WLC")
            self._wlc_btn.setToolTip(
                "Fit a Marko-Siggia WLC to the Gaussian ridge over the shaded extension region"
            )
            self._wlc_btn.clicked.connect(self._fit_wlc)
            row2.addWidget(self._wlc_btn)
            self._wlc_label = QLabel("WLC: —")
            row2.addWidget(self._wlc_label)
        row2.addStretch()
        if db_path is not None:
            self._export_btn = QPushButton("Export…")
            self._export_btn.setToolTip(
                "Write this view to the export folder: the 2DH matrix, the "
                "per-column Gaussian ridge with its spread, the WLC fit if one has "
                "been made, and a manifest."
            )
            self._export_btn.clicked.connect(self._on_export)
            row2.addWidget(self._export_btn)
        lay.addLayout(row2)

        # ── Plot ──────────────────────────────────────────────────────────────
        self._pw = FixedDomainPlot(x_range, f_range)
        set_plot_title(self._pw, title, caption)
        # si=False: this view sits on the 2DH grid whose ranges are set in
        # plain nm/pN spin boxes, and its WLC label prints l_p/l_c in nm.
        set_si_label(self._pw, "bottom", x_label, x_unit, si=False)
        set_si_label(self._pw, "left",   f_label, f_unit, si=False)
        self._img = pg.ImageItem()
        self._pw.addItem(self._img)

        self._overlay_panel = None
        if paths and overlay_fn is not None:
            self._overlay_panel = TraceOverlayPanel(self._pw, overlay_fn)
            self._overlay_panel.set_paths(paths)
            splitter = QSplitter(Qt.Orientation.Horizontal)
            splitter.addWidget(self._pw)
            splitter.addWidget(self._overlay_panel)
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 0)
            lay.addWidget(splitter, stretch=1)
        else:
            lay.addWidget(self._pw, stretch=1)

        # Corner mask: everything to the right of and below the marker is excluded
        # from the per-column Gaussian fit. Place the marker at the upper-left of
        # the bottom-right baseline blob; the rising ridge to the left is kept.
        f_lo, f_hi = f_range
        x_lo, x_hi = x_range
        self._corner = pg.TargetItem(
            pos=(x_lo + 0.45 * (x_hi - x_lo), f_lo + 0.12 * (f_hi - f_lo)),
            size=12,
            movable=True,
            pen=pg.mkPen(style.REFERENCE, width=style.W_GUIDE),
            brush=style.band_brush(style.REFERENCE, alpha=60),
        )
        self._corner.setZValue(9)
        self._corner.setVisible(False)
        self._corner.sigPositionChanged.connect(self._on_corner_changed)
        self._pw.addItem(self._corner)

        # Draggable extension region for the WLC fit (physical axes only).
        if physical:
            x0, x1 = x_range
            span = x1 - x0
            self._region = pg.LinearRegionItem(
                values=(x0 + 0.3 * span, x0 + 0.7 * span),
                brush=pg.mkBrush(*style.rgba(style.INK_MUTED, 40)),
            )
            self._region.setZValue(10)
            self._region.sigRegionChanged.connect(self._invalidate_wlc)
            self._pw.addItem(self._region)

        self._draw()
        self._recompute_mean()

    def _draw(self) -> None:
        clip_level = max(self._auto_max * self._z_pct / 100.0, 1e-10)
        scaled = np.clip(self._display / clip_level * 254.0, 0, 254).astype(np.uint8)
        scaled[self._display > clip_level] = 255
        self._img.setImage(scaled, autoLevels=False, lut=self._lut, levels=(0, 255))
        self._img.setRect(QRectF(
            self._x_range[0], self._f_range[0],
            self._x_range[1] - self._x_range[0],
            self._f_range[1] - self._f_range[0],
        ))

    def _recompute_mean(self) -> None:
        """Per-column Gaussian fit, optionally excluding the corner-mask quadrant."""
        self._invalidate_wlc()
        counts = self._counts
        if self._corner_chk.isChecked() and self._corner is not None:
            p = self._corner.pos()
            x_c, f_c = p.x(), p.y()
            x_mask = self._x_phys >= x_c       # to the right
            f_mask = self._f_col  <= f_c       # and below
            counts = self._counts.copy()
            counts[np.ix_(x_mask, f_mask)] = 0.0
        mean_c, sig_c = _column_gaussian_means(counts)
        self._f_mean = self._f0 + (mean_c + 0.5) * self._f_per_col
        self._f_sig  = sig_c * self._f_per_col
        self._valid  = np.isfinite(self._f_mean)
        self._draw_mean()

    def _on_corner_toggle(self, on: bool) -> None:
        self._corner.setVisible(on)
        self._recompute_mean()

    def _on_corner_changed(self) -> None:
        if self._corner_chk.isChecked():
            self._recompute_mean()

    def _draw_mean(self) -> None:
        for it in self._mean_items:
            self._pw.removeItem(it)
        self._mean_items = []
        if not (self._mean_chk.isChecked() and self._valid.any()):
            return

        # ±1σ is the descriptive fitted spread in each x bin. The aggregate
        # stores mean counts/trace, so it cannot support a defensible σ/√N band;
        # population uncertainty would require resampling contributing traces.
        err = self._f_sig
        v   = self._valid & np.isfinite(err)
        xb  = self._x_phys[v]
        up  = pg.PlotDataItem(xb, (self._f_mean + err)[v])
        lo  = pg.PlotDataItem(xb, (self._f_mean - err)[v])
        band = pg.FillBetweenItem(up, lo,
                                  brush=style.band_brush(style.SERIES_LINE[0],
                                                         alpha=90))
        self._pw.addItem(band)

        # Cased, for the same reason the 2DH trace overlays are (style.py
        # § H): this curve is drawn over the histogram image, and
        # every hue drops below 3:1 somewhere in the middle of the ramp.  A
        # ridge that vanishes into the densest part of its own 2DH is
        # exactly where it matters most.
        casing = pg.PlotDataItem(self._x_phys, self._f_mean,
                                 connect="finite", pen=style.casing_pen())
        self._pw.addItem(casing)
        line = pg.PlotDataItem(
            self._x_phys, self._f_mean,
            connect="finite",
            pen=style.model_pen(style.SERIES_LINE[0], alpha=255),
        )
        self._pw.addItem(line)
        self._mean_items = [band, casing, line]

    def _fit_wlc(self) -> None:
        for it in self._wlc_items:
            self._pw.removeItem(it)
        self._wlc_items = []

        lo, hi = self._region.getRegion()
        m = self._valid & (self._x_phys >= lo) & (self._x_phys <= hi)
        z = self._x_phys[m]
        f = self._f_mean[m]
        if z.size < 4:
            self._wlc_fit = None
            self._wlc_curve = None
            self._wlc_pcov = None
            self._wlc_label.setText("WLC: need ≥4 points in region")
            return

        # Fit in the Δx coordinate with a free origin offset: the molecule's
        # force-onset sits at some Δx (= −x*(F*)), not necessarily Δx=0. A single
        # constant offset makes this equivalent to fitting in raw extension.
        def wlc_off(x, l_p, l_c, x_off):
            return models.wlc(x + x_off, l_p, l_c)

        span = max(z.max() - z.min(), 1e-3)
        try:
            # pcov was being discarded into `_`.  curve_fit computes it either
            # way, so the uncertainty on this l_p/l_c was being thrown away at
            # the moment it was produced — the same "computed, then dropped"
            # shape as the fit CIs elsewhere, one line earlier in the pipeline.
            popt, pcov = curve_fit(
                wlc_off, z, f,
                p0=[0.4, 1.5 * span, -z.min()],
                bounds=([1e-3, 0.5 * span, -np.inf], [1e3, np.inf, np.inf]),
                maxfev=10000,
            )
        except (RuntimeError, ValueError):
            self._wlc_fit = None
            self._wlc_curve = None
            self._wlc_pcov = None
            self._wlc_label.setText("WLC: fit failed")
            return

        l_p, l_c, x_off = (float(v) for v in popt)
        perr = np.sqrt(np.diag(np.asarray(pcov, dtype=float)))
        l_p_err, l_c_err = (float(perr[0]), float(perr[1])) if np.isfinite(
            perr[:2]).all() else (float("nan"), float("nan"))

        z_line = np.linspace(z.min(), z.max(), 400)
        y_line = wlc_off(z_line, l_p, l_c, x_off)

        self._wlc_items = []
        # 95% band from the fit's own covariance, drawn under the curve.  Same
        # method as dist_fit_core.total_fit_ci: sample the FULL covariance
        # rather than propagate the diagonal, because l_p, l_c and the offset
        # are strongly correlated in a WLC fit and the diagonal alone would
        # overstate the band.
        band = _fit_ci_band(wlc_off, z_line, popt, pcov)
        if band is not None:
            b_lo, b_hi = band
            item = pg.FillBetweenItem(
                pg.PlotDataItem(z_line, b_hi), pg.PlotDataItem(z_line, b_lo),
                brush=style.band_brush(style.SERIES_LINE[1], alpha=90),
            )
            self._pw.addItem(item)
            self._wlc_items.append(item)

        casing = pg.PlotDataItem(z_line, y_line, pen=style.casing_pen())
        self._pw.addItem(casing)
        curve = pg.PlotDataItem(
            z_line, y_line,
            pen=style.guide_pen(style.SERIES_LINE[1], width=style.W_MODEL),
        )
        self._pw.addItem(curve)
        self._wlc_items += [casing, curve]

        self._wlc_fit = {
            "l_p_nm": l_p, "l_p_err_nm": l_p_err,
            "l_c_nm": l_c, "l_c_err_nm": l_c_err,
            "x_offset_nm": x_off,
            "region_lo": float(lo), "region_hi": float(hi),
            "n_points_fitted": int(z.size),
        }
        # The drawn curve and its band, kept for the export. l_p_err/l_c_err
        # above are the covariance's diagonal and cannot regenerate this band
        # (see dist_fit_core.total_fit_ci) — the exported lo/hi columns and
        # the full matrix in the manifest are what make it reproducible.
        self._wlc_curve = (z_line, y_line, band)
        self._wlc_pcov  = pcov
        self._wlc_label.setText(
            f"WLC:  l_p = {_q('seg_l_p_nm', l_p)} ± {_q('seg_l_p_err', l_p_err, with_unit=True)}   "
            f"l_c = {_q('seg_l_c_nm', l_c)} ± {_q('seg_l_c_err', l_c_err, with_unit=True)}   "
            f"(offset {_q('seg_l_c_nm', x_off, with_unit=True)})"
        )

    def _invalidate_wlc(self) -> None:
        """Erase a WLC result as soon as its ridge or fit range changes."""
        for item in self._wlc_items:
            self._pw.removeItem(item)
        self._wlc_items = []
        self._wlc_fit = None
        self._wlc_curve = None
        self._wlc_pcov = None
        if hasattr(self, "_wlc_label"):
            self._wlc_label.setText("WLC: —")

    # ── Export ────────────────────────────────────────────────────────────────

    def export_provenance(self) -> dict:
        """This view's settings, for an export manifest — the parent window's
        provenance plus what this window itself decided."""
        d = dict(self._provenance)
        corner = self._corner.pos() if self._corner is not None else None
        fit_region = self._region.getRegion() if self._region is not None else None
        d.update({
            "window":       "mean_curve",
            "view_title":   self._title,
            "caption":      self._caption,
            "physical":     self._physical,
            "x_label":      self._x_label,
            "f_label":      self._f_label,
            "x_unit":       self._x_unit,
            "f_unit":       self._f_unit,
            "x_range":      list(self._x_range),
            "f_range":      list(self._f_range),
            "z_clip_pct":   self._z_pct,
            "ridge_estimator": "per-x-bin Gaussian centre",
            "band":         "±1σ fitted Gaussian spread",
            "corner_mask":  self._corner_chk.isChecked(),
            "corner_mask_position": (
                [float(corner.x()), float(corner.y())] if corner is not None else None
            ),
            "wlc_fit_region": (
                [float(fit_region[0]), float(fit_region[1])]
                if fit_region is not None else None
            ),
        })
        return d

    def _on_export(self) -> None:
        """Write this view's numbers out.

        A cluster popout is a figure in its own right. Export the histogram,
        Gaussian ridge, optional WLC fit, and the provenance needed to
        interpret them together."""
        parts = ["_matrix.csv", "_mean_curve.csv"]
        if self._wlc_fit:
            parts.append("_wlc_fit.csv")
        if self._wlc_curve is not None:
            parts.append("_wlc_curve.csv")
        with _export.export_group(
            self._db_path, f"mean_curve_{_slug(self._title)}", parts,
            kind="mean_curve",
        ) as g:
            g.contributing_files(self._paths)
            g.note_dict(self.export_provenance())
            g.note(n_columns=int(self._x_phys.size),
                   n_valid_columns=int(self._valid.sum()),
                   wlc_fit=self._wlc_fit)
            if self._wlc_fit:
                g.note_dict(ci_manifest_fields(
                    self._wlc_pcov,
                    self._wlc_curve is not None and self._wlc_curve[2] is not None))

            g.matrix("_matrix.csv", self._counts)
            g.table(
                "_mean_curve.csv",
                ["x", "f_mean", "f_sigma", "valid"],
                [(float(self._x_phys[i]), float(self._f_mean[i]),
                  float(self._f_sig[i]), bool(self._valid[i]))
                 for i in range(self._x_phys.size)],
            )
            if self._wlc_fit:
                g.table("_wlc_fit.csv", ["quantity", "value"],
                        list(self._wlc_fit.items()))

            # The fitted WLC as drawn, with its confidence band — so the
            # exported figure can show the same uncertainty the screen did
            # without re-running the Monte Carlo.
            if self._wlc_curve is not None:
                x_line, y_line, band = self._wlc_curve
                header = ["x", "f_wlc"] + (["ci_lo", "ci_hi"] if band else [])
                rows = []
                for i in range(len(x_line)):
                    row = [float(x_line[i]), float(y_line[i])]
                    if band:
                        row += [float(band[0][i]), float(band[1][i])]
                    rows.append(row)
                g.table("_wlc_curve.csv", header, rows)

        QMessageBox.information(self, "Export", g.message())

    def _on_slider(self, val: int) -> None:
        self._z_pct = val
        self._z_label.setText(f"{val}%")
        self._draw()
