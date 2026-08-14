# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/gmm_fit_window.py
#
# GmmFitWindow — pop-out 2D Gaussian Mixture Model fitting window.
#
# Opened from EventSummaryWindow ("Fit 2D…" button).
# Receives the active Event Summary population as
# (x, y) = (contour_length_nm, rupture_force_pN).
#
# Left pane  : scatter plot with radius-1/radius-2 covariance contours.
# Right pane : model builder (K + covariance type) + fit results table +
#              model comparison table (AICc, BIC) + DB save.
#
# Backend: sklearn.mixture.GaussianMixture with n_init restarts for robustness.

from __future__ import annotations

import json
import warnings

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from . import style
from . import db as _db
from . import quantities as _quant
from . import export_utils as _export
from .gmm_fit_core import (
    COMPONENT_COLORS,
    COV_TYPE_LABELS, COV_TYPES,
    aicc_from_aic, component_stats, ellipse_curve,
    component_display_ids, component_order, get_component_cov,
    json_safe_statistics, n_params_gmm,
)
from .qt_utils import set_plot_title, set_si_label, fit_on_screen
from .quantities import format_value as _q   # ONE formatter: unit and
# meaningful digits come from quantities.py, so the same measurement
# cannot print as 166, 166.2 and 166.20 in three different windows.

_N_INIT = 10   # EM restarts — trades runtime for robustness against local optima


def _sklearn_version() -> str:
    """Recorded in the export manifest: a mixture fit's exact component
    assignment can depend on the implementation, so reproducing one needs the
    version as well as the settings."""
    try:
        import sklearn
        return str(sklearn.__version__)
    except Exception:
        return "unknown"


# ── Scatter pane ──────────────────────────────────────────────────────────────

class _ScatterPane(QWidget):
    """
    Scatter plot of (contour_length, rupture_force) values.
    After a fit, overlays covariance contours at Mahalanobis radii 1 (dashed)
    and 2 (solid) for each component.
    """

    def __init__(
        self,
        xy:      np.ndarray,   # (N, 2): col 0 = length nm, col 1 = force pN
        x_label: str,
        y_label: str,
        x_unit:  str = "",
        y_unit:  str = "",
        caption: str = "",
        parent   = None,
    ):
        super().__init__(parent)
        self._xy            = xy
        self._ellipse_items: list = []
        self._mean_items:   list  = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        self._gw = pg.GraphicsLayoutWidget()
        lay.addWidget(self._gw)

        self._plot = self._gw.addPlot()
        # si=False: this window prints the fitted means and covariances in
        # plain nm/pN in the table beside the plot, so an axis free to
        # relabel itself µm would show one fit at two scales at once.
        set_si_label(self._plot, "bottom", x_label, x_unit, si=False)
        set_si_label(self._plot, "left",   y_label, y_unit, si=False)
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._legend = self._plot.addLegend(offset=(10, 10))
        # On-canvas (survives pyqtgraph's Export…, unlike the header QLabel
        # in GmmFitWindow) — what population/segment this fit's pass_xy came from.
        set_plot_title(self._plot, caption=caption)

        self._scatter = pg.ScatterPlotItem(
            x=xy[:, 0].tolist(),
            y=xy[:, 1].tolist(),
            size=style.DOT_SIZE,
            pen=pg.mkPen(None),
        # The data is the substrate and stays neutral beneath model colours.
            brush=style.scatter_brush(style.DATA),
        )
        self._plot.addItem(self._scatter)

    # ── Fit overlay ───────────────────────────────────────────────────────────

    def show_fit(self, gm: GaussianMixture) -> None:
        self.clear_fit()

        # Sort components largest weight first for consistent color assignment
        order = component_order(gm)

        for rank, k in enumerate(order):
            color = COMPONENT_COLORS[rank % len(COMPONENT_COLORS)]
            mean  = gm.means_[k]
            cov   = get_component_cov(gm, k)
            w_pct = gm.weights_[k] * 100.0
            label = f"C{rank + 1}  ({w_pct:.0f} %)"

            # Mahalanobis radius 1 (about 39% probability mass in 2D), dashed.
            ex1, ey1 = ellipse_curve(mean, cov, scale=1.0)
            item1 = self._plot.plot(
                ex1.tolist(), ey1.tolist(),
                pen=style.guide_pen(color, width=style.W_GUIDE),
            )
            self._ellipse_items.append(item1)

            # Mahalanobis radius 2 (about 86% probability mass in 2D), solid.
            ex2, ey2 = ellipse_curve(mean, cov, scale=2.0)
            item2 = self._plot.plot(
                ex2.tolist(), ey2.tolist(),
                pen=style.model_pen(color),
                name=label,
            )
            self._ellipse_items.append(item2)

            # Mean marker
            mean_item = pg.ScatterPlotItem(
                x=[float(mean[0])], y=[float(mean[1])],
                size=10, symbol="+",
                pen=pg.mkPen(color, width=2),
                brush=pg.mkBrush(None),
            )
            self._plot.addItem(mean_item)
            self._mean_items.append(mean_item)

    def clear_fit(self) -> None:
        for item in self._ellipse_items:
            self._plot.removeItem(item)
        self._ellipse_items.clear()
        for item in self._mean_items:
            self._plot.removeItem(item)
        self._mean_items.clear()
        self._legend.clear()


# ── Model pane ────────────────────────────────────────────────────────────────

class _ModelPane(QWidget):
    """K + covariance type selector, fit results, model comparison, and DB save."""

    def __init__(
        self,
        scatter_pane: _ScatterPane,
        xy:           np.ndarray,
        x_variable:   str,
        y_variable:   str,
        db_path:      str,
        parent        = None,
        paths:        list[str] | None = None,
        caption:      str = "",
    ):
        super().__init__(parent)
        self._scatter     = scatter_pane
        self._xy          = xy
        self._x_variable  = x_variable
        self._y_variable  = y_variable
        self._db_path     = db_path
        self._paths       = list(paths) if paths else []
        self._caption     = caption
        self._last_fit:   dict = {}
        self._last_gm:    GaussianMixture | None = None
        self._fit_history: list[dict] = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        # ── Model Builder ─────────────────────────────────────────────────────
        bg = QGroupBox("Model Builder")
        bl = QVBoxLayout(bg)

        k_row = QHBoxLayout()
        k_row.addWidget(QLabel("Components K:"))
        self._k_spin = QSpinBox()
        self._k_spin.setRange(1, 10)
        _quant.configure_spinbox(self._k_spin)
        self._k_spin.setValue(2)
        self._k_spin.setToolTip(
            "Number of Gaussian components.\n"
            "Start with K=1 and increase. Compare AICc/BIC to choose."
        )
        k_row.addWidget(self._k_spin)
        k_row.addStretch()
        bl.addLayout(k_row)

        cov_row = QHBoxLayout()
        cov_row.addWidget(QLabel("Covariance:"))
        self._cov_combo = QComboBox()
        self._cov_combo.addItems(COV_TYPE_LABELS)
        self._cov_combo.setToolTip(
            "Full   — each component has its own free covariance (tilted ellipses)\n"
            "Tied   — all components share one covariance (same shape, different positions)\n"
            "Diagonal — axes-aligned ellipses, no cross-correlation\n"
            "Spherical — circular blobs (one variance per component)"
        )
        cov_row.addWidget(self._cov_combo, 1)
        bl.addLayout(cov_row)

        fit_btn = QPushButton("▶  Fit!")
        fit_btn.setStyleSheet(style.QSS_PRIMARY_ACTION)
        fit_btn.clicked.connect(self._fit)
        bl.addWidget(fit_btn)
        lay.addWidget(bg)

        # ── Fit Results ───────────────────────────────────────────────────────
        rg = QGroupBox("Fit Results")
        rl = QVBoxLayout(rg)

        self._params_tbl = QTableWidget(0, 7)
        self._params_tbl.setHorizontalHeaderLabels(
            ["Component", "Weight", "μ_x  (nm)", "μ_y  (pN)",
             "σ_x  (nm)", "σ_y  (pN)", "ρ"]
        )
        self._params_tbl.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self._params_tbl.horizontalHeader().setStretchLastSection(True)
        self._params_tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._params_tbl.setAlternatingRowColors(True)
        self._params_tbl.setMinimumHeight(90)
        rl.addWidget(self._params_tbl)

        self._stats_box = QTextEdit()
        self._stats_box.setReadOnly(True)
        self._stats_box.setMaximumHeight(130)
        self._stats_box.setFont(style.font(
            self._stats_box.font(), size_pt=style.FONT_SMALL_PT, mono=True))
        rl.addWidget(self._stats_box)

        btn_row = QHBoxLayout()
        self._save_btn = QPushButton("Save fit to DB")
        self._save_btn.setEnabled(False)
        self._save_btn.setToolTip("Save current fit result to the catalog database.")
        self._save_btn.clicked.connect(self._save_fit)
        btn_row.addWidget(self._save_btn)
        # Save persists the model in the catalog; Export writes a portable
        # component table, assigned point cloud, and manifest.
        self._export_btn = QPushButton("Export…")
        self._export_btn.setEnabled(False)
        self._export_btn.setToolTip(
            "Write this fit to the export folder: per-component weight/mean/"
            "covariance, every point with its assigned component, and a "
            "manifest (model, goodness-of-fit, file list)."
        )
        self._export_btn.clicked.connect(self._export_fit)
        btn_row.addWidget(self._export_btn)
        self._save_status = QLabel("")
        self._save_status.setStyleSheet(style.qss_text(style.TEXT_GOOD, size_px=10))
        btn_row.addWidget(self._save_status)
        btn_row.addStretch()
        rl.addLayout(btn_row)
        lay.addWidget(rg)

        # ── Model Comparison ──────────────────────────────────────────────────
        cg = QGroupBox("Model Comparison")
        cl = QVBoxLayout(cg)

        guide = QLabel(
            "Exploratory Gaussian approximation. Compare only fits in this "
            "window: lower AICc / BIC = better.  "
            "As a rule of thumb, ΔAICc ≤ 2 indicates similar support, "
            "4–7 substantially less support, and > 10 little support.  "
            "AICc is unavailable when the sample is too small."
        )
        guide.setWordWrap(True)
        guide.setStyleSheet(style.qss_text(size_px=10))
        cl.addWidget(guide)

        self._cmp_tbl = QTableWidget(0, 7)
        self._cmp_tbl.setHorizontalHeaderLabels(
            ["K", "Cov", "n_params", "n", "AICc", "ΔAICc", "ΔBIC"]
        )
        self._cmp_tbl.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self._cmp_tbl.horizontalHeader().setStretchLastSection(True)
        self._cmp_tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._cmp_tbl.setAlternatingRowColors(True)
        self._cmp_tbl.setMinimumHeight(100)
        cl.addWidget(self._cmp_tbl)

        clr_btn = QPushButton("Clear session history")
        clr_btn.setToolTip(
            "Remove unsaved fits from this window's comparison table."
        )
        clr_btn.clicked.connect(self._clear_session_history)
        cl.addWidget(clr_btn)
        lay.addWidget(cg, 1)

    # ── Fitting ───────────────────────────────────────────────────────────────

    def _fit(self) -> None:
        k        = self._k_spin.value()
        cov_lbl  = self._cov_combo.currentText()
        cov_type = COV_TYPES[cov_lbl]
        n        = len(self._xy)

        if n < k:
            QMessageBox.warning(
                self, "Too few points",
                f"Need at least K={k} data points; only {n} available."
            )
            return

        if not np.all(np.isfinite(self._xy)):
            QMessageBox.warning(
                self, "Invalid data",
                "The scatter contains non-finite values and cannot be fitted."
            )
            return

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            gm = GaussianMixture(
                n_components      = k,
                covariance_type   = cov_type,
                n_init            = _N_INIT,
                random_state      = 0,
                max_iter          = 300,
            )
            try:
                gm.fit(self._xy)
            except (ValueError, np.linalg.LinAlgError) as exc:
                QMessageBox.warning(
                    self, "Fit failed",
                    "The mixture could not estimate stable covariances for "
                    "these points. Try fewer components or a simpler "
                    f"covariance type.\n\nDetails: {exc}"
                )
                return

        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        if not converged:
            reply = QMessageBox.warning(
                self, "Convergence warning",
                "The EM algorithm did not fully converge.\n"
                "Results may be unreliable.\n\n"
                "Try fewer components or a different covariance type.\n\n"
                "Use result anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._last_gm = gm

        n_params  = n_params_gmm(k, cov_lbl)
        aic       = float(gm.aic(self._xy))
        bic       = float(gm.bic(self._xy))
        aicc      = aicc_from_aic(aic, n_params, n)
        log_l     = float(gm.score(self._xy) * n)   # total log-likelihood

        gof = {"log_L": log_l, "AIC": aic, "AICc": aicc, "BIC": bic}
        model_label = f"K={k} ({cov_lbl})"

        self._scatter.show_fit(gm)
        self._update_params_table(gm, cov_lbl)
        self._update_stats_box(gof, k, n_params, n, cov_lbl, converged)
        self._record_fit(model_label, k, cov_lbl, n_params, n, gof, saved=False)
        self._save_status.setText("")

        self._last_fit = {
            "model_label":    model_label,
            "k_components":   k,
            "cov_type":       cov_type,
            "cov_label":      cov_lbl,
            "n_values":       n,
            "n_params":       n_params,
            "means":          gm.means_.tolist(),
            "weights":        gm.weights_.tolist(),
            "gof":            gof,
        }
        # Store covariances in a type-agnostic serialisable form (always full 2×2)
        self._last_fit["covs"] = [
            get_component_cov(gm, k_).tolist() for k_ in range(k)
        ]
        self._save_btn.setEnabled(True)
        self._export_btn.setEnabled(True)

    # ── Results display ───────────────────────────────────────────────────────

    def _update_params_table(self, gm: GaussianMixture, cov_lbl: str) -> None:
        k     = gm.n_components
        order = component_order(gm)
        has_rho = cov_lbl in ("Full", "Tied")

        self._params_tbl.setRowCount(k)
        for rank, ki in enumerate(order):
            s     = component_stats(gm, ki)
            color = COMPONENT_COLORS[rank % len(COMPONENT_COLORS)]
            rho_str = f"{s['rho']:.3f}" if has_rho else "—"

            cells = [
                f"C{rank + 1}",
                f"{s['weight'] * 100:.1f} %",
                f"{s['mu_x']:.2f}",
                f"{s['mu_y']:.2f}",
                f"{s['sigma_x']:.2f}",
                f"{s['sigma_y']:.2f}",
                rho_str,
            ]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if c == 0:
                    item.setForeground(QColor(color))
                self._params_tbl.setItem(rank, c, item)

    def _update_stats_box(
        self,
        gof:       dict,
        k:         int,
        n_params:  int,
        n:         int,
        cov_lbl:   str,
        converged: bool,
    ) -> None:
        lines = [
            f"K = {k}   cov = {cov_lbl}   n_params = {n_params}   n = {n}",
            "─" * 42,
            f"  {'log-likelihood':<18} {gof['log_L']:.3f}",
            f"  {'AIC':<18} {gof['AIC']:.2f}",
            (f"  {'AICc':<18} {gof['AICc']:.2f}"
             if np.isfinite(gof["AICc"])
             else f"  {'AICc':<18} unavailable (n <= n_params + 1)"),
            f"  {'BIC':<18} {gof['BIC']:.2f}",
        ]
        if not converged:
            lines += ["", "⚠  Convergence warning — results may be unreliable."]
        lines += [
            "",
            "Exploratory Gaussian approximation; inspect the scatter too.",
            "Guide: compare only this cohort; lower AICc / BIC = better",
            "  ΔAICc ≤ 2 similar support · 4–7 less · > 10 little support",
        ]
        self._stats_box.setPlainText("\n".join(lines))

    # ── Comparison table ──────────────────────────────────────────────────────

    def _record_fit(
        self,
        model_label: str,
        k:           int,
        cov_lbl:     str,
        n_params:    int,
        n:           int,
        gof:         dict,
        saved:       bool,
    ) -> None:
        self._fit_history.append({
            "model":    model_label,
            "k":        k,
            "cov":      cov_lbl,
            "n_params": n_params,
            "n":        n,
            "AICc":     gof["AICc"],
            "BIC":      gof["BIC"],
            "saved":    saved,
        })
        self._update_comparison_table()

    def _update_comparison_table(self) -> None:
        h = self._fit_history
        if not h:
            self._cmp_tbl.setRowCount(0)
            return
        finite_aicc = [x["AICc"] for x in h if np.isfinite(x["AICc"])]
        finite_bic = [x["BIC"] for x in h if np.isfinite(x["BIC"])]
        best_aicc = min(finite_aicc, default=float("nan"))
        best_bic = min(finite_bic, default=float("nan"))
        self._cmp_tbl.setRowCount(len(h))
        for r, entry in enumerate(h):
            d_aicc  = entry["AICc"] - best_aicc
            d_bic   = entry["BIC"]  - best_bic
            is_best = np.isfinite(d_aicc) and d_aicc < 1e-9
            is_last = r == len(h) - 1 and not entry["saved"]
            vals    = [
                str(entry["k"]),
                str(entry["cov"]),
                str(entry["n_params"]),
                str(entry["n"]),
                (f"{entry['AICc']:.1f}" if np.isfinite(entry["AICc"])
                 else "unavailable"),
                ("★ best" if is_best else f"+{d_aicc:.1f}"
                 if np.isfinite(d_aicc) else "—"),
                ("★ best" if np.isfinite(d_bic) and d_bic < 1e-9
                 else f"+{d_bic:.1f}" if np.isfinite(d_bic) else "—"),
            ]
            for c, text in enumerate(vals):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if entry["saved"]:
                    item.setBackground(QColor(style.TABLE_TINT_SAVED))
                elif is_best:
                    item.setBackground(QColor(style.TABLE_TINT_BEST))
                elif is_last:
                    item.setBackground(QColor(style.TABLE_TINT_RECENT))
                self._cmp_tbl.setItem(r, c, item)

    def _clear_session_history(self) -> None:
        self._fit_history = [e for e in self._fit_history if e["saved"]]
        self._update_comparison_table()

    # ── DB save ───────────────────────────────────────────────────────────────

    def export_provenance(self) -> dict:
        """This fit's settings, for an export manifest — same protocol method
        as the other exporting windows."""
        f = self._last_fit
        return {
            "window":       "gmm_fit",
            "x_variable":   self._x_variable,
            "y_variable":   self._y_variable,
            "model_label":  f.get("model_label"),
            "k_components": f.get("k_components"),
            "cov_type":     f.get("cov_label"),
            "n_values":     f.get("n_values"),
            "n_params":     f.get("n_params"),
            "n_init":       _N_INIT,
            "caption":      self._caption,
        }

    def _export_fit(self) -> None:
        """Write the current GMM out as data files: the fitted components
        (weight, mean, covariance per component), and the point cloud with
        each point's assigned component and maximum posterior responsibility.
        The manifest records the model settings, goodness-of-fit statistics,
        software version, provenance caption, and contributing files."""
        if not self._last_fit or self._last_gm is None:
            return
        f  = self._last_fit
        gm = self._last_gm
        with _export.export_group(
            self._db_path, "fit_gmm_length_force",
            ["_components.csv", "_points.csv"], kind="gmm_fit",
        ) as g:
            g.contributing_files(self._paths)
            g.note_dict(self.export_provenance())
            g.note(goodness_of_fit=json_safe_statistics(f["gof"]),
                   sklearn_version=_sklearn_version())

            rows = []
            order = component_order(gm)
            display_ids = component_display_ids(gm)
            for k in order:
                mean = f["means"][k]
                cov  = f["covs"][k]
                rows.append((
                    int(display_ids[k]), int(k), float(f["weights"][k]),
                    float(mean[0]), float(mean[1]),
                    float(cov[0][0]), float(cov[1][1]), float(cov[0][1]),
                ))
            g.table(
                "_components.csv",
                ["component", "sklearn_component", "weight", "mean_x", "mean_y",
                 "cov_xx", "cov_yy", "cov_xy"],
                rows,
            )

            # Point cloud with its cluster assignment — the 2-D analogue of
            # the PCA window's per-curve cluster labels, and the reason this
            # export is worth having: a GMM nobody can map back onto
            # individual curves cannot be acted on.
            internal_labels = gm.predict(self._xy)
            labels = display_ids[internal_labels]
            resp   = gm.predict_proba(self._xy).max(axis=1)
            have_paths = len(self._paths) == len(self._xy)
            header = (["path"] if have_paths else []) + [
                "x", "y", "component", "responsibility"]
            prows = []
            for i in range(len(self._xy)):
                row = [self._paths[i]] if have_paths else []
                row += [float(self._xy[i, 0]), float(self._xy[i, 1]),
                        int(labels[i]), float(resp[i])]
                prows.append(row)
            g.table("_points.csv", header, prows)

        QMessageBox.information(self, "Export fit", g.message())

    def _save_fit(self) -> None:
        if not self._last_fit:
            return
        f = self._last_fit
        try:
            _db.save_gmm_fit(
                x_variable      = self._x_variable,
                y_variable      = self._y_variable,
                n_values        = f["n_values"],
                k_components    = f["k_components"],
                cov_type        = f["cov_label"],
                means_json      = json.dumps(f["means"]),
                covs_json       = json.dumps(f["covs"]),
                weights_json    = json.dumps(f["weights"]),
                gof_json        = json.dumps(json_safe_statistics(f["gof"]),
                                             allow_nan=False),
                fit_config_json = json.dumps({"n_init": _N_INIT}),
                db_path         = self._db_path,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return

        for entry in reversed(self._fit_history):
            if not entry["saved"]:
                entry["saved"] = True
                break
        self._update_comparison_table()
        self._save_btn.setEnabled(False)
        self._save_status.setText("Saved ✔")


# ── Main window ───────────────────────────────────────────────────────────────

class GmmFitWindow(QMainWindow):
    """
    Pop-out 2D GMM fitting sandbox.

    Locked to the pass values supplied at construction.
    Re-opening raises the existing window (caller's responsibility).
    """

    _X_VARIABLE = "Contour length (WLC fit, l_c)"
    _Y_VARIABLE = "Rupture force (selected segment)"
    # Text and unit are separate: the unit is declared once in quantities.py,
    # and the plain text is what reaches the export manifest.
    _X_LABEL    = _X_VARIABLE
    _Y_LABEL    = _Y_VARIABLE
    _X_UNIT     = _quant.NM
    _Y_UNIT     = _quant.PN

    def __init__(
        self,
        pass_xy: np.ndarray,   # (N, 2): col 0 = length nm, col 1 = force pN
        db_path: str,
        caption: str = "",
        paths:   list[str] | None = None,
    ) -> None:
        super().__init__()
        n = len(pass_xy)
        # Curves behind these points, positionally aligned with pass_xy —
        # carried so an export can name the data it fitted. Same reasoning as
        # DistFitWindow._paths.
        self._paths = list(paths) if paths else []
        self.setWindowTitle(
            f"SMFS — 2D GMM fit — {self._X_VARIABLE} × {self._Y_VARIABLE}")
        self.setWindowFlag(Qt.WindowType.Window)
        fit_on_screen(self, 1200, 700)
        style.apply_plot_defaults()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(4)
        root.setContentsMargins(8, 6, 8, 6)

        # Header
        x = pass_xy[:, 0]
        y = pass_xy[:, 1]
        hdr = QLabel(
            f"{n} pass values   |   "
            f"length: {_q('seg_l_c_nm', x.min())}–{_q('seg_l_c_nm', x.max(), with_unit=True)}  "
            f"(mean {_q('seg_l_c_nm', x.mean())})   |   "
            f"force: {_q('seg_force_pN', y.min())}–{_q('seg_force_pN', y.max(), with_unit=True)}  "
            f"(mean {_q('seg_force_pN', y.mean())})"
        )
        hdr.setFont(style.font(hdr.font(), size_pt=style.FONT_SMALL_PT))
        root.addWidget(hdr)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        scatter_pane = _ScatterPane(pass_xy, self._X_LABEL, self._Y_LABEL,
                                    self._X_UNIT, self._Y_UNIT, caption=caption)
        model_pane   = _ModelPane(
            scatter_pane,
            pass_xy,
            self._X_VARIABLE,
            self._Y_VARIABLE,
            db_path,
            paths=self._paths,
            caption=caption,
        )

        splitter.addWidget(scatter_pane)
        splitter.addWidget(model_pane)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([800, 400])
        root.addWidget(splitter, stretch=1)
