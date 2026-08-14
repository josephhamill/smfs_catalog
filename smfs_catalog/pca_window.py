# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/pca_window.py
#
# PCAWindow — PCA + k-means analysis of per-curve 2D histogram profiles.
#
# Pipeline (runs at open time):
#   1. Stack and row-normalize event histograms → X (n_events × n_bins)
#   2. Drop always-zero bin features
#   3. Standardise: zero mean, unit variance per feature
#   4. PCA — top min(n_events-1, 20) components via randomised SVD
#
# K-means runs on button click in PC1–3 score space.
#
# Opened from "Run PCA" in either 2DH window.

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from . import style
from . import clustering as _clustering
from . import quantities as _quant
from . import export_utils as _export
from .mean_curve_window import MeanCurveWindow
from .qt_utils import FixedDomainPlot, set_plot_title, set_si_label, fit_on_screen
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

_N_COMPONENTS = 20   # max PCs to compute (scree up to this)
_PCA_SVD_SOLVER = "randomized"
_PCA_SEED = 0

# K-means is seeded so a re-run on the same data reproduces the same labels.
# These are recorded in the export manifest, not just applied here: "k-means
# on the first 3 PCs, seed 42" is part of what a cluster figure MEANS, and a
# number that exists only in this source file cannot be restated by anyone
# holding the exported file.
_KMEANS_SEED     = 42
_KMEANS_N_PCS    = 3     # k-means runs in PC1-_KMEANS_N_PCS space
_KMEANS_N_INIT   = "auto"


def _sklearn_version() -> str:
    """Recorded in the manifest: k-means' exact labels can depend on the
    implementation, so "same seed" is only reproducible alongside the version
    that honoured it."""
    try:
        import sklearn
        return str(sklearn.__version__)
    except Exception:
        return "unknown"


_N_LOADINGS   = 3    # loading panels shown (PC1, PC2, PC3)
class _ClickablePlot(pg.PlotWidget):
    """PlotWidget that fires a callback on double-click."""

    def __init__(self, *args, on_double_click=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_double_click = on_double_click

    def mouseDoubleClickEvent(self, event):
        if self._on_double_click:
            self._on_double_click()
        super().mouseDoubleClickEvent(event)


def _layout_driven(plot):
    """Let the PCA grid, not plot contents or titles, choose panel size."""
    plot.setMinimumSize(0, 0)
    plot.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
    return plot


def _relative_frequency_rows(counts: np.ndarray) -> np.ndarray:
    """Convert raw per-trace bin counts to equal-weight PCA profiles."""
    X = np.asarray(counts, dtype=np.float32)
    totals = X.sum(axis=1, keepdims=True)
    return np.divide(X, totals, out=np.zeros_like(X), where=totals > 0)


class PCAWindow(QMainWindow):
    """
    PCA + k-means for per-curve 2D histogram profiles.

    Tab 1 — PCA:
        Scree plot (individual + cumulative variance)
        PC1 / PC2 / PC3 loading heatmaps (RdBu_r, symmetric limits)
        PC1 vs PC2 score scatter (coloured by cluster after k-means)

    Tab 2 — K-means:
        Elbow plot (k = 1–10, run on demand)
        k spinbox + Run K-means button
        PC1 vs PC2 scatter coloured by cluster
        Per-cluster counts/trace histograms
    """

    def __init__(
        self,
        histograms:         dict[str, np.ndarray],
        x_bins:             int,
        f_bins:             int,
        x_range:            tuple[float, float],
        f_range:            tuple[float, float],
        x_label:            str = "x̃",
        f_label:            str = "F̃",
        x_unit:             str = "",
        f_unit:             str = "",
        caption:            str = "",
        provenance:         dict | None = None,
        db_path:            str | None = None,
        display_histograms: dict[str, np.ndarray] | None = None,
        display_x_range:    tuple[float, float] | None = None,
        display_f_range:    tuple[float, float] | None = None,
        physical:           bool = False,
        overlay_fn=None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("SMFS — PCA")
        self.setWindowFlag(Qt.WindowType.Window)
        fit_on_screen(self, 1100, 760)
        self._physical   = physical
        # Structured version of `caption` (population/segment/grid/align/z-clip
        # — see base_2dh_window.export_provenance()), for the export manifest.
        # `caption` itself stays the human-readable line shown on screen.
        self._provenance = provenance or {}
        self._db_path    = db_path
        # Bound _TwoDHWindowBase._overlay_xF from the 2DH window this PCA run
        # was opened from — already knows the alignment settings that
        # produced these histograms, so popouts can offer per-curve overlay +
        # filenames without owning any DB/alignment logic themselves.
        self._overlay_fn = overlay_fn

        style.apply_plot_defaults()

        self._x_bins  = x_bins
        self._f_bins  = f_bins
        self._x_range = x_range
        self._f_range = f_range
        self._x_label = x_label
        self._f_label = f_label
        self._x_unit  = x_unit
        self._f_unit  = f_unit
        self._caption = caption
        self._cluster_labels: np.ndarray | None = None
        self._cluster_hists:  list[np.ndarray]  = []
        self._cluster_centers: np.ndarray | None = None
        self._elbow: list[tuple[int, float]]    = []
        self._kmeans_k = self._kmeans_n_pcs = 0
        self._kmeans_inertia: float | None      = None

        self._z_pct:             int              = 100
        self._cluster_auto_max:  float            = 1.0
        self._cluster_displays:  list[np.ndarray] = []
        self._cluster_imgs:      list             = []
        self._popout_wins:       list             = []
        self._lut = self._build_lut()

        # ── Build data matrix ─────────────────────────────────────────────────
        self._paths = list(histograms.keys())
        n = len(self._paths)

        X_raw = np.stack(
            [histograms[p].ravel().astype(np.float32) for p in self._paths]
        )                                           # (n, x_bins * f_bins)

        # Display matrix for cluster visualisation — full 2DHs when a selection
        # window is active,
        # otherwise identical to the PCA data matrix.
        if display_histograms is not None:
            sample = next(iter(display_histograms.values()))
            self._display_x_bins  = sample.shape[0]
            self._display_f_bins  = sample.shape[1]
            self._X_display = np.stack(
                [display_histograms[p].ravel().astype(np.float32) for p in self._paths]
            )
            self._display_x_range = display_x_range if display_x_range is not None else x_range
            self._display_f_range = display_f_range if display_f_range is not None else f_range
        else:
            self._display_x_bins  = x_bins
            self._display_f_bins  = f_bins
            self._X_display       = X_raw
            self._display_x_range = x_range
            self._display_f_range = f_range

        # Give every trace equal total weight in PCA regardless of how many
        # samples its selected segment contains. Integer counts remain the
        # source representation used for cohort counts/trace displays.
        X_profiles = _relative_frequency_rows(X_raw)

        # Drop bins that are zero in every sample
        live         = X_raw.any(axis=0)
        X            = X_profiles[:, live]
        self._live   = live
        n_features   = int(live.sum())

        # ── Standardise (float32 throughout — avoids sklearn float64 promotion) ─
        mean = X.mean(axis=0)
        std  = X.std(axis=0)
        std[std == 0] = 1.0
        X_scaled = (X - mean) / std                 # float32

        # ── PCA ───────────────────────────────────────────────────────────────
        from sklearn.decomposition import PCA
        if n < 2:
            raise ValueError("PCA requires at least two event histograms.")
        if n_features == 0:
            raise ValueError("PCA feature space contains no non-zero bins.")
        n_comp = min(n - 1, n_features, _N_COMPONENTS)
        pca    = PCA(
            n_components=n_comp,
            svd_solver=_PCA_SVD_SOLVER,
            random_state=_PCA_SEED,
        )
        self._scores   = pca.fit_transform(X_scaled).astype(np.float32)
        self._loadings = pca.components_.astype(np.float32)
        self._var      = pca.explained_variance_ratio_.astype(np.float32)

        # ── UI ────────────────────────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(4)
        root.setContentsMargins(8, 6, 8, 6)

        # Header
        def _vpct(i: int) -> str:
            return f"{self._var[i]*100:.1f}%" if i < len(self._var) else "n/a"

        hdr_text = (
            f"{n} events    {n_features} features used  "
            f"(of {x_bins * f_bins} bins,  {x_bins * f_bins - n_features} dead)    "
            f"PC1 {_vpct(0)}   PC2 {_vpct(1)}   PC3 {_vpct(2)}"
        )
        hdr_row = QHBoxLayout()
        hdr = QLabel(hdr_text)
        font = hdr.font()
        font.setPointSize(style.FONT_SMALL_PT)
        hdr.setFont(font)
        hdr.setMinimumWidth(0)
        hdr.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        hdr_row.addWidget(hdr)
        hdr_row.addStretch()
        export_btn = QPushButton("Export…")
        export_btn.setToolTip(
            "Write this analysis to the export folder: per-curve PC scores and "
            "cluster labels, the scree data, the PC1-3 loading matrices, each "
            "cluster's counts/trace 2DH, the elbow data, and a manifest recording "
            "the parameters (k, seed, file list) needed to reproduce it."
        )
        export_btn.clicked.connect(self._on_export_pca)
        hdr_row.addWidget(export_btn)
        root.addLayout(hdr_row)

        # What produced the histograms PCA ran on (population, segment, grid,
        # align mode/F*, z-clip at "Run PCA" time) — this snapshot doesn't
        # change afterward the way the live 2DH windows' captions do, so a
        # second, visually distinct line rather than folding into hdr above.
        if self._caption:
            cap = QLabel(self._caption)
            cap_font = cap.font()
            cap_font.setPointSize(style.FONT_CAPTION_PT)
            cap.setFont(cap_font)
            cap.setStyleSheet(f"color: {style.INK_MUTED};")
            cap.setMinimumWidth(0)
            cap.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            root.addWidget(cap)

        tabs = QTabWidget()
        tabs.setMinimumSize(0, 0)
        tabs.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        root.addWidget(tabs, stretch=1)
        tabs.addTab(self._build_pca_tab(), "PCA")
        tabs.addTab(self._build_kmeans_tab(n), "K-means")

    # ── Tab 1: PCA ────────────────────────────────────────────────────────────

    def _build_pca_tab(self) -> QWidget:
        w   = QWidget()
        lay = QHBoxLayout(w)

        # LEFT: scree + the PC loading maps as a 2x2 grid.
        #
        # Two columns at ordinary desktop sizes; every plot remains free to
        # shrink with the window, with a double-click popout for close reading.
        grid = QGridLayout()
        grid.setSpacing(4)
        grid.addWidget(_layout_driven(self._build_scree_plot()), 0, 0)

        n_load = min(_N_LOADINGS, len(self._loadings))
        for i in range(n_load):
            grid.addWidget(_layout_driven(self._make_loading_plot(i)),
                           (i + 1) // 2, (i + 1) % 2)
        for c in (0, 1):
            grid.setColumnStretch(c, 1)
        for r in (0, 1):
            grid.setRowStretch(r, 1)

        grid_host = QWidget()
        grid_host.setLayout(grid)
        grid_host.setMinimumSize(0, 0)
        lay.addWidget(grid_host, stretch=2)

        # RIGHT: PC1 vs PC2 scatter, beside the grid rather than under it.
        n     = len(self._paths)
        sx    = self._scores[:, 0].tolist()
        sy    = (self._scores[:, 1].tolist()
                 if self._scores.shape[1] > 1 else [0.0] * n)
        self._pca_scatter_pw = _ClickablePlot(
            on_double_click=self._open_score_popout)
        self._pca_scatter_pw.setToolTip(
            "Double-click to open larger view, and to compare PC pairs")
        self._pca_scatter_pw.setLabel("bottom",
            f"PC1  ({self._var[0]*100:.1f}%)" if len(self._var) > 0 else "PC1")
        self._pca_scatter_pw.setLabel("left",
            f"PC2  ({self._var[1]*100:.1f}%)" if len(self._var) > 1 else "PC2")
        self._pca_scatter_pw.showGrid(x=True, y=True, alpha=0.15)
        self._pca_scatter = pg.ScatterPlotItem(
            x=sx, y=sy,
            size=6, pen=None,
            brush=style.scatter_brush(style.DATA),
        )
        self._pca_scatter_pw.addItem(self._pca_scatter)
        lay.addWidget(_layout_driven(self._pca_scatter_pw), stretch=1)

        return w

    # ── Tab 2: K-means ────────────────────────────────────────────────────────

    def _build_kmeans_tab(self, n: int) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)

        # Controls
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("k:"))
        self._k_spin = QSpinBox()
        self._k_spin.setRange(2, min(n, 10))
        _quant.configure_spinbox(self._k_spin)
        self._k_spin.setValue(min(3, n))
        ctrl.addWidget(self._k_spin)
        ctrl.addSpacing(8)
        btn_elbow = QPushButton("Elbow")
        btn_elbow.clicked.connect(self._run_elbow)
        ctrl.addWidget(btn_elbow)
        btn_km = QPushButton("Run K-means")
        btn_km.clicked.connect(self._run_kmeans)
        ctrl.addWidget(btn_km)
        btn_total = QPushButton("Total 2DH…")
        btn_total.setToolTip("Gaussian-ridge view of the counts/trace 2DH over all events")
        btn_total.clicked.connect(self._open_total_popout)
        ctrl.addWidget(btn_total)
        hint = QLabel("  clusters in up to PC1–3 space")
        hint.setStyleSheet(f"color: {style.INK_MUTED};")
        ctrl.addWidget(hint)
        ctrl.addStretch()
        ctrl.addWidget(QLabel("Clip:"))
        self._z_slider_km = QSlider(Qt.Orientation.Horizontal)
        self._z_slider_km.setRange(1, 200)
        self._z_slider_km.setValue(100)
        self._z_slider_km.setFixedWidth(110)
        self._z_slider_km.setToolTip(
            "Intensity clip: drag left to highlight dense regions (clipped values shown in red)"
        )
        self._z_label_km = QLabel("100%")
        self._z_label_km.setMinimumWidth(36)
        self._z_auto_btn_km = QPushButton("Auto")
        self._z_auto_btn_km.setFixedWidth(42)
        self._z_slider_km.valueChanged.connect(self._on_km_z_slider)
        self._z_auto_btn_km.clicked.connect(self._on_km_z_auto)
        ctrl.addWidget(self._z_slider_km)
        ctrl.addSpacing(4)
        ctrl.addWidget(self._z_label_km)
        ctrl.addWidget(self._z_auto_btn_km)
        lay.addLayout(ctrl)

        # Elbow plot
        self._elbow_pw = pg.PlotWidget()
        self._elbow_pw.setTitle("Elbow — click Elbow to populate")
        self._elbow_pw.setLabel("bottom", "k")
        self._elbow_pw.setLabel("left",   "Inertia")
        self._elbow_pw.setFixedHeight(180)
        lay.addWidget(self._elbow_pw)

        # Cluster scatter (initially grey, redrawn after Run K-means)
        sx = self._scores[:, 0].tolist()
        sy = (self._scores[:, 1].tolist()
              if self._scores.shape[1] > 1 else [0.0] * n)
        # Same popout as the PCA tab's score plot: it is the same score space,
        # so it gets the same PC-pair selector rather than a second one.
        self._km_scatter_pw = _ClickablePlot(
            on_double_click=self._open_score_popout)
        self._km_scatter_pw.setToolTip(
            "Double-click to open larger view, and to compare PC pairs")
        self._km_scatter_pw.setLabel("bottom",
            f"PC1  ({self._var[0]*100:.1f}%)" if len(self._var) > 0 else "PC1")
        self._km_scatter_pw.setLabel("left",
            f"PC2  ({self._var[1]*100:.1f}%)" if len(self._var) > 1 else "PC2")
        self._km_scatter_pw.showGrid(x=True, y=True, alpha=0.15)
        init_scatter = pg.ScatterPlotItem(
            x=sx, y=sy,
            size=6, pen=None,
            brush=pg.mkBrush(*style.NON_HIT_RGBA),
        )
        self._km_scatter_pw.addItem(init_scatter)
        lay.addWidget(self._km_scatter_pw, stretch=1)

        # Cluster histogram row — populated by _run_kmeans
        self._cluster_row   = QWidget()
        self._cluster_lay   = QHBoxLayout(self._cluster_row)
        self._cluster_lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._cluster_row, stretch=1)

        return w

    # ── Analysis callbacks ────────────────────────────────────────────────────

    def _run_elbow(self) -> None:
        import warnings
        from sklearn.cluster import KMeans
        n         = len(self._paths)
        scores_k  = self._scores[:, :min(_KMEANS_N_PCS, self._scores.shape[1])]
        k_range   = range(1, min(n, 11))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak")
            inertias  = [
                KMeans(n_clusters=k, n_init=_KMEANS_N_INIT, random_state=_KMEANS_SEED)
                .fit(scores_k).inertia_
                for k in k_range
            ]
        # Kept for the export: the elbow is how k was CHOSEN, so it belongs
        # with the clustering it justifies, not just on screen.
        self._elbow = [(int(k), float(v)) for k, v in zip(k_range, inertias)]
        self._elbow_pw.clear()
        self._elbow_pw.setTitle("Elbow plot")
        self._elbow_pw.plot(
            list(k_range), inertias,
            pen=style.model_pen(style.SERIES_LINE[0]),
            symbol="o", symbolSize=7,
            symbolBrush=pg.mkBrush(style.SERIES_LINE[0]),
        )

    def _run_kmeans(self) -> None:
        import warnings
        from sklearn.cluster import KMeans
        k         = self._k_spin.value()
        n_pcs     = min(_KMEANS_N_PCS, self._scores.shape[1])
        scores_k  = self._scores[:, :n_pcs]
        km        = KMeans(n_clusters=k, n_init=_KMEANS_N_INIT,
                           random_state=_KMEANS_SEED)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak")
            labels    = km.fit_predict(scores_k)
        # Renumber left-to-right by PC1 centroid before anything sees the
        # labels. K-means numbers arbitrarily, so without this the same
        # physical group gets a different integer on every re-run and every
        # colour in every window jumps for no reason.  Done here, once, so the
        # panels below, the export and the session registry all agree.
        # Apply the same permutation to every cluster-indexed result.  The
        # labels, display matrices and sizes below use the PC1 ordering, so the
        # exported centre at index c must describe that same cluster c.
        cluster_order = _clustering.first_pc_order(labels, scores_k)
        labels = _clustering.order_by_first_pc(labels, scores_k)
        self._cluster_labels = labels
        self._kmeans_k       = k
        self._kmeans_n_pcs   = n_pcs
        self._kmeans_inertia = float(km.inertia_)
        self._cluster_centers = km.cluster_centers_[cluster_order].astype(float)
        self._publish_clustering()

        n   = len(self._paths)
        sx  = self._scores[:, 0].tolist()
        sy  = (self._scores[:, 1].tolist()
               if self._scores.shape[1] > 1 else [0.0] * n)

        # Build per-point spot dicts (supports per-point colour in pyqtgraph).
        #
        # Cluster labels use the shared labelled-series palette so every
        # downstream window renders a given cluster identity consistently.
        def _spots(xs, ys, lbls, alpha=180):
            return [
                {"pos": (float(xs[i]), float(ys[i])),
                 "brush": style.scatter_brush(
                     style.series_labeled(int(lbls[i])), alpha),
                 "size": 6, "pen": None}
                for i in range(len(xs))
            ]

        # K-means tab scatter
        self._km_scatter_pw.clear()
        self._km_scatter_pw.showGrid(x=True, y=True, alpha=0.15)
        self._km_scatter_pw.addItem(pg.ScatterPlotItem(spots=_spots(sx, sy, labels)))

        # PCA tab scatter — also update to show cluster membership
        self._pca_scatter_pw.removeItem(self._pca_scatter)
        self._pca_scatter = pg.ScatterPlotItem(
            spots=_spots(sx, sy, labels, alpha=160)
        )
        self._pca_scatter_pw.addItem(self._pca_scatter)

        # Cluster histograms
        while self._cluster_lay.count():
            item = self._cluster_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._cluster_displays = []
        self._cluster_hists    = []
        self._cluster_imgs     = []

        for c in range(k):
            mask    = labels == c
            hist    = (self._X_display[mask]
                       .mean(axis=0)
                       .reshape(self._display_x_bins, self._display_f_bins)
                       .astype(np.float32))
            display = np.sqrt(hist)
            self._cluster_displays.append(display)
            self._cluster_hists.append(hist)

            pw = FixedDomainPlot(
                self._display_x_range, self._display_f_range,
                title=f"Cluster {c}  (n={int(mask.sum())})",
                on_double_click=lambda c=c: self._open_cluster_popout(c),
            )
            self._label_grid_axes(pw)
            pw.setToolTip("Double-click to open larger view")
            img = pg.ImageItem()
            pw.addItem(img)
            self._cluster_imgs.append(img)
            self._cluster_lay.addWidget(pw)

        all_max = max((float(d.max()) for d in self._cluster_displays), default=1.0)
        self._cluster_auto_max = max(all_max, 1e-10)
        self._z_slider_km.setValue(100)
        self._draw_clusters()

    # ── Cluster z-scale ───────────────────────────────────────────────────────

    def _publish_clustering(self) -> None:
        """Hand the labels to the session so other windows can colour by them.

        Ephemeral by design — see clustering.py.  This window keeps its own
        copy for its own panels and its export; the registry exists so Explore
        Events, the any-vs-any scatter and the variable timeseries can ask for
        a label without knowing anything about PCA.
        """
        if self._cluster_labels is None:
            return
        source = {
            "window":           self._provenance.get("window"),
            "population":       self._provenance.get("population"),
            "segment":          self._provenance.get("segment"),
            "align_mode":       self._provenance.get("align_mode"),
            "selection_window": bool(self._provenance.get("selection_window")),
        }
        _clustering.set_current(_clustering.Clustering(
            labels          = {p: int(c) for p, c in
                               zip(self._paths, self._cluster_labels)},
            k               = self._kmeans_k,
            seed            = _KMEANS_SEED,
            n_pcs           = self._kmeans_n_pcs,
            sklearn_version = _sklearn_version(),
            source          = source,
            created_at      = _clustering.now_stamp(),
        ))

    def _build_lut(self) -> np.ndarray:
        return style.intensity_lut()

    def _apply_z_scale(self, display: np.ndarray) -> np.ndarray:
        clip_level = max(self._cluster_auto_max * self._z_pct / 100.0, 1e-10)
        scaled = np.clip(display / clip_level * 254.0, 0, 254).astype(np.uint8)
        scaled[display > clip_level] = 255
        return scaled

    def _draw_clusters(self) -> None:
        rect = QRectF(
            self._display_x_range[0], self._display_f_range[0],
            self._display_x_range[1] - self._display_x_range[0],
            self._display_f_range[1] - self._display_f_range[0],
        )
        for img, display in zip(self._cluster_imgs, self._cluster_displays):
            scaled = self._apply_z_scale(display)
            img.setImage(scaled, autoLevels=False, lut=self._lut, levels=(0, 255))
            img.setRect(rect)

    def _on_km_z_slider(self, val: int) -> None:
        self._z_pct = val
        self._z_label_km.setText(f"{val}%")
        self._draw_clusters()

    def _on_km_z_auto(self) -> None:
        self._z_slider_km.setValue(100)

    def _label_grid_axes(self, pw) -> None:
        """Label a plot drawn in the parent 2DH's coordinates.

        Cluster tiles and PC loading maps are the same grid as the 2DH they
        came from, so they share its axis text and unit. si=False for the same reason as the 2DH's
        own axes: the grid ranges are set in plain-unit spin boxes.
        """
        set_si_label(pw, "bottom", self._x_label, self._x_unit, si=False)
        set_si_label(pw, "left",   self._f_label, self._f_unit, si=False)

    def _open_cluster_popout(self, c: int) -> None:
        if c >= len(self._cluster_displays):
            return
        cluster_paths = [p for p, lbl in zip(self._paths, self._cluster_labels) if lbl == c]
        win = MeanCurveWindow(
            title    = f"Cluster {c}  (n={len(cluster_paths)})",
            display  = self._cluster_displays[c],
            counts   = self._cluster_hists[c],
            auto_max = self._cluster_auto_max,
            z_pct    = self._z_pct,
            lut      = self._lut,
            x_range  = self._display_x_range,
            f_range  = self._display_f_range,
            x_label  = self._x_label,
            f_label  = self._f_label,
            x_unit   = self._x_unit,
            f_unit   = self._f_unit,
            caption  = self._caption,
            physical = self._physical,
            paths      = cluster_paths,
            overlay_fn = self._overlay_fn,
            db_path    = self._db_path,
            provenance = {**self._provenance, "source": "pca_cluster",
                          "cluster": c, "kmeans_k": self._kmeans_k,
                          "kmeans_random_state": _KMEANS_SEED},
        )
        self._popout_wins.append(win)
        win.show()

    def _open_total_popout(self) -> None:
        """Gaussian-ridge view of the total counts/trace 2DH."""
        hist = (self._X_display
                .mean(axis=0)
                .reshape(self._display_x_bins, self._display_f_bins)
                .astype(np.float32))
        display = np.sqrt(hist)
        win = MeanCurveWindow(
            title    = f"Total  (n={len(self._paths)})",
            display  = display,
            counts   = hist,
            auto_max = max(float(display.max()), 1e-10),
            z_pct    = 100,
            lut      = self._lut,
            x_range  = self._display_x_range,
            f_range  = self._display_f_range,
            x_label  = self._x_label,
            f_label  = self._f_label,
            x_unit   = self._x_unit,
            f_unit   = self._f_unit,
            caption  = self._caption,
            physical = self._physical,
            paths      = self._paths,
            overlay_fn = self._overlay_fn,
            db_path    = self._db_path,
            provenance = {**self._provenance, "source": "pca_total"},
        )
        self._popout_wins.append(win)
        win.show()

    # ── Scree popout ─────────────────────────────────────────────────────────

    def _build_scree_plot(self, *, embedded: bool = True) -> "_ClickablePlot":
        scree = _ClickablePlot(
            on_double_click=self._open_scree_popout if embedded else None,
        )
        # The window-level provenance line owns the long caption. Repeating it
        # in every PlotItem title makes pyqtgraph's internal canvas thousands
        # of pixels wide and the outer widget then clips its right-hand side.
        set_plot_title(scree, "Scree")
        set_si_label(scree, "bottom", "PC", "")
        scree.setLabel("left",   "Variance (%)")
        scree.showGrid(x=False, y=True, alpha=0.2)
        if embedded:
            scree.setToolTip("Double-click to open larger view")
        nc    = len(self._var)
        x_pos = np.arange(1, nc + 1, dtype=float)
        bars  = pg.BarGraphItem(
            x=x_pos, height=self._var * 100, width=0.6,
            brush=pg.mkBrush(*style.rgba(style.SERIES_LINE[0], 200)),
        )
        scree.addItem(bars)
        scree.plot(
            x_pos, np.cumsum(self._var) * 100,
            pen=pg.mkPen(style.REFERENCE, width=style.W_GUIDE),
            symbol="o", symbolSize=5, symbolBrush="r",
        )
        return scree

    def _open_scree_popout(self) -> None:
        win = QMainWindow()
        win.setWindowTitle("SMFS — PCA scree")
        win.setWindowFlag(Qt.WindowType.Window)
        fit_on_screen(win, 520, 520)
        win.setCentralWidget(self._popout_plot_host(
            self._build_scree_plot(embedded=False)))
        self._popout_wins.append(win)
        win.show()

    # ── Loading heatmap helper ────────────────────────────────────────────────

    def _make_loading_plot(self, pc_idx: int, *, embedded: bool = True) -> "_ClickablePlot":
        """PlotWidget with PC loading reshaped to 2D, RdBu_r diverging colormap."""
        loading  = self._loadings[pc_idx]           # (n_live_features,)
        img_flat = np.zeros(self._x_bins * self._f_bins, dtype=np.float32)
        img_flat[self._live] = loading
        img_2d   = img_flat.reshape(self._x_bins, self._f_bins)

        vmax = float(np.abs(img_2d).max()) or 1.0

        feature_rect = QRectF(
            self._x_range[0], self._f_range[0],
            self._x_range[1] - self._x_range[0],
            self._f_range[1] - self._f_range[0],
        )
        pw = FixedDomainPlot(
            self._x_range, self._f_range,
            on_double_click=(
                (lambda pc=pc_idx: self._open_loading_popout(pc)) if embedded else None
            ),
        )
        set_plot_title(pw, f"PC{pc_idx + 1}  ({self._var[pc_idx]*100:.1f}%)")
        self._label_grid_axes(pw)
        if embedded:
            pw.setToolTip("Double-click to open larger view")

        img = pg.ImageItem()
        img.setColorMap(style.pca_loading_colormap())
        img.setImage(img_2d, levels=(-vmax, vmax))
        img.setRect(feature_rect)
        pw.addItem(img)
        pw.fit_domain()
        return pw

    # ── Score scatter ─────────────────────────────────────────────────────────

    def _pc_label(self, i: int) -> str:
        return (f"PC{i + 1}  ({self._var[i] * 100:.1f}%)"
                if len(self._var) > i else f"PC{i + 1}")

    def _score_spots(self, xi: int, yi: int, alpha: int = 160) -> list[dict]:
        """Per-point spots for a PC-pair scatter, cluster-coloured if a
        clustering exists.  Same colours and alpha as the embedded plot, so a
        popout never disagrees with the panel it came from."""
        n = len(self._paths)

        def col(i: int) -> list[float]:
            return (self._scores[:, i].tolist()
                    if self._scores.shape[1] > i else [0.0] * n)

        xs, ys = col(xi), col(yi)
        labels = self._cluster_labels
        clustered = labels is not None and bool(self._kmeans_k)

        def brush(i: int):
            if clustered:
                return style.scatter_brush(
                    style.series_labeled(int(labels[i])), alpha)
            return style.scatter_brush(style.DATA)

        return [{"pos": (float(xs[i]), float(ys[i])),
                 "size": 6, "pen": None, "brush": brush(i)}
                for i in range(n)]

    def _make_score_plot(self, xi: int = 0, yi: int = 1) -> "_ClickablePlot":
        pw = _ClickablePlot()
        pw.setLabel("bottom", self._pc_label(xi))
        pw.setLabel("left", self._pc_label(yi))
        pw.showGrid(x=True, y=True, alpha=0.15)
        pw.addItem(pg.ScatterPlotItem(spots=self._score_spots(xi, yi)))
        return pw

    def _open_score_popout(self) -> None:
        """Larger score view, with a PC-pair selector.

        K-means uses up to the first three PCs, so PC1 vs PC2 alone may show
        only one projection of the clustering space. The selector exposes all
        available pairs without adding a separate 3D interaction model.
        """
        win = QMainWindow()
        win.setWindowTitle("SMFS — PCA scores")
        win.setWindowFlag(Qt.WindowType.Window)
        fit_on_screen(win, 640, 600)

        host = QWidget()
        outer = QVBoxLayout(host)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Axes:"))
        combo = QComboBox()
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            if self._scores.shape[1] > max(i, j):
                combo.addItem(f"PC{i + 1} vs PC{j + 1}", (i, j))
        bar.addWidget(combo)
        bar.addStretch()
        outer.addLayout(bar)

        holder = QVBoxLayout()
        outer.addLayout(holder, 1)

        def show_pair() -> None:
            while holder.count():
                item = holder.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            pair = combo.currentData()
            if pair is not None:
                holder.addWidget(self._make_score_plot(*pair))

        combo.currentIndexChanged.connect(show_pair)
        show_pair()

        win.setCentralWidget(host)
        self._popout_wins.append(win)
        win.show()

    def _open_loading_popout(self, pc_idx: int) -> None:
        win = QMainWindow()
        win.setWindowTitle(f"SMFS — PCA loading  PC{pc_idx + 1}")
        win.setWindowFlag(Qt.WindowType.Window)
        fit_on_screen(win, 560, 520)
        win.setCentralWidget(self._popout_plot_host(
            self._make_loading_plot(pc_idx, embedded=False)))
        self._popout_wins.append(win)
        win.show()

    def _popout_plot_host(self, plot: QWidget) -> QWidget:
        """Plot plus provenance that wraps without defining canvas width."""
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setContentsMargins(6, 6, 6, 6)
        if self._caption:
            cap = QLabel(self._caption)
            cap.setWordWrap(True)
            cap.setMinimumWidth(0)
            cap.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            lay.addWidget(cap)
        lay.addWidget(plot, stretch=1)
        return host

    # ── Export ────────────────────────────────────────────────────────────────

    def export_provenance(self) -> dict:
        """This analysis's settings, for an export manifest — the parent 2DH
        window's provenance (population/segment/grid/align mode, passed in at
        construction) plus what this window itself decided. Same protocol
        method as the other exporting windows."""
        d = dict(self._provenance)
        d.update({
            "window":                "pca",
            "physical":              self._physical,
            "x_bins":                self._x_bins,
            "f_bins":                self._f_bins,
            "x_range":               list(self._x_range),
            "f_range":               list(self._f_range),
            "n_events":              len(self._paths),
            "n_components_computed": len(self._var),
            "pca_svd_solver":        _PCA_SVD_SOLVER,
            "pca_random_state":      _PCA_SEED,
            "pca_input":             "per_trace_relative_bin_frequency",
            "pca_feature_scaling":   "zero_mean_unit_variance",
            "display_z_quantity":    "counts_per_trace",
            "caption":               self._caption,
        })
        return d

    def _on_export_pca(self) -> None:
        """Everything this window computed, in files that can rebuild its
        figures and reproduce its clustering:

          _scores.csv        ONE ROW PER CURVE: path, PC1..PCn, cluster.
                             The most important file here — it is both the
                             per-curve result (which cluster a curve landed
                             in) and the join key back to every other export.
          _scree.csv         variance explained per PC (+ cumulative).
          _pcN_matrix.csv    the PC1-3 loading heatmaps as bare matrices
                             (row=x bin, col=F bin — same orientation as the
                             2DH matrix they were computed from).
          _clusterN_matrix.csv  each cluster's counts/trace 2DH, i.e. the cluster
                             figure panels themselves, in the display grid.
          _elbow.csv         k vs inertia, if the elbow was run — this is HOW
                             k was chosen, so it travels with the choice.
          _manifest.json     population/segment/grid/align mode, the file
                             list, AND the clustering parameters (k, seed,
                             how many PCs it ran on, sklearn version).

        Cluster membership is required to reproduce downstream figures and
        therefore belongs in the export.

        A loading has no natural "count" to marginalize the way a histogram
        does, so unlike the 2DH export there are no paired 1D projection
        files; the manifest carries the axis definition instead."""
        n_load    = min(_N_LOADINGS, len(self._loadings))
        clustered = self._cluster_labels is not None
        stem = f"pca_{'physical' if self._physical else 'normalized'}"

        parts = ["_scores.csv", "_scree.csv"]
        parts += [f"_pc{i + 1}_matrix.csv" for i in range(n_load)]
        if clustered:
            parts += [f"_cluster{c}_matrix.csv" for c in range(len(self._cluster_hists))]
        if self._elbow:
            parts.append("_elbow.csv")

        with _export.export_group(
            self._db_path, stem, parts, kind="pca_kmeans",
        ) as g:
            g.contributing_files(self._paths)
            g.note_dict(self.export_provenance())
            g.note(n_loadings_exported=n_load)

            # Per-curve scores (+ cluster label where one exists).
            labels = (self._cluster_labels if clustered
                      else [None] * len(self._paths))
            header = ["path"] + [f"pc{i + 1}" for i in range(self._scores.shape[1])]
            if clustered:
                header.append("cluster")
            rows = []
            for i, p in enumerate(self._paths):
                row = [p] + [float(v) for v in self._scores[i]]
                if clustered:
                    row.append(int(labels[i]))
                rows.append(row)
            g.table("_scores.csv", header, rows)

            cum = np.cumsum(self._var)
            g.table(
                "_scree.csv", ["pc", "variance_pct", "cumulative_pct"],
                [(i + 1, float(self._var[i]) * 100, float(cum[i]) * 100)
                 for i in range(len(self._var))],
            )

            for i in range(n_load):
                img_flat = np.zeros(self._x_bins * self._f_bins, dtype=np.float32)
                img_flat[self._live] = self._loadings[i]
                g.matrix(f"_pc{i + 1}_matrix.csv",
                         img_flat.reshape(self._x_bins, self._f_bins))

            if clustered:
                counts = [int((self._cluster_labels == c).sum())
                          for c in range(self._kmeans_k)]
                for c, hist in enumerate(self._cluster_hists):
                    g.matrix(f"_cluster{c}_matrix.csv", hist)
                g.note(
                    kmeans_k=self._kmeans_k,
                    kmeans_random_state=_KMEANS_SEED,
                    kmeans_n_init=_KMEANS_N_INIT,
                    kmeans_n_pcs=self._kmeans_n_pcs,
                    kmeans_inertia=self._kmeans_inertia,
                    kmeans_cluster_sizes=counts,
                    kmeans_cluster_centers=(
                        self._cluster_centers.tolist()
                        if self._cluster_centers is not None else None),
                    cluster_matrix_x_bins=self._display_x_bins,
                    cluster_matrix_f_bins=self._display_f_bins,
                    cluster_matrix_x_range=list(self._display_x_range),
                    cluster_matrix_f_range=list(self._display_f_range),
                    cluster_matrix_z_quantity="counts_per_trace",
                    sklearn_version=_sklearn_version(),
                )
            else:
                g.note(kmeans_k=None,
                       kmeans_note="K-means had not been run when this was exported.")

            if self._elbow:
                g.table("_elbow.csv", ["k", "inertia"], self._elbow)

        QMessageBox.information(self, "Export PCA", g.message())
