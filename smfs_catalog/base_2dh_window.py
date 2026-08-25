# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/base_2dh_window.py
#
# _TwoDHWindowBase — everything shared by the two 2DH windows (normalized_2dh_
# window.py, physical_2dh_window.py): each computes a per-curve 2D histogram
# ("a 2DH") and averages them into one cohort counts/trace 2DH. They differ in
# the coordinate transform applied before binning:
#
#   normalized — x̃ = x/l_c, F̃ = F·l_p/kT  (SCALE by the curve's own fit;
#                divides its own singularity to x̃=1 for every curve, which is
#                what makes the universal-collapse reference curve valid)
#   physical   — Δx = x − anchor(F)         (SHIFT by an anchor chosen from a
#                menu: onset / F* / snap-off / l_c / rupture)
#
# These two operations don't compose: subtracting a per-curve anchor before
# dividing by l_c would move the WLC singularity to a different x̃ for every
# curve and break the collapse the normalized view exists to show. So
# "align mode" (the anchor menu) is physical-only, permanently — not just
# unimplemented for normalized. Segment selection (first, penultimate, last,
# primary, or secondary) chooses which stored ROI segment feeds the transform;
# it is orthogonal to alignment mode and shared by both windows.

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
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

from . import db as _db
from . import quantities as _quant
from . import export_utils as _export
from . import ledger as _ledger
from .db import write_event_histograms_bulk
from .curve_loader import LoadError, load_force_curve
from . import style
from .qt_utils import (
    CancelableProgress, FixedDomainPlot, _make_session_header, set_plot_title,
    set_si_label, fit_on_screen,
)
from .trace_overlay_panel import TraceOverlayPanel


def _counts_per_trace(histograms) -> np.ndarray:
    """Mean raw bin count over contributing traces."""
    values = list(histograms)
    if not values:
        raise ValueError("counts/trace requires at least one histogram")
    return np.stack(values).mean(axis=0, dtype=np.float64)

_BINS_CHOICES = ["32", "64", "128", "256", "512"]

# Which stored ROI segment (onset→r1, r1→r2, …, →terminal) feeds the
# transform. Shared by both windows' grid dialogs. Primary and Secondary
# read the same manual per-curve override the queue table's "primary segment"
# uses — two more individually-selectable segments, same as First/Penultimate/
# Last; nothing about being a pair partner (Secondary's other role, in
# dF/isoforce) stops either one being viewed alone here.
_ALIGN_SEG_CHOICES = [
    ("First",       "first"),
    ("Penultimate", "penult"),
    ("Last",        "last"),
    ("Primary",     "primary"),
    ("Secondary",   "secondary"),
]


class _GridDialog(QDialog):
    """Modal dialog for a 2DH's grid: bins, axis ranges, and which stored
    segment the histogram is built from. Subclassed by
    the physical window (_PhysicalGridDialog) to add F*/align-mode fields
    that only make sense for a shift-based, physical-units transform."""

    _TITLE = "Grid settings"
    _WARN_COLOR = style.TEXT_WARNING

    def __init__(self, parent, x_bins, f_bins, x_min, x_max, f_min, f_max,
                 align_segment, defaults):
        super().__init__(parent)
        self.setWindowTitle(self._TITLE)
        self.setModal(True)
        self._defaults = defaults

        root = QVBoxLayout(self)

        box = QGroupBox("Grid parameters  —  requires Rebuild")
        box.setStyleSheet(style.qss_emphasis(
            self._WARN_COLOR, selector="QGroupBox"))
        self._form = QFormLayout(box)

        self._xb = QComboBox()
        self._fb = QComboBox()
        for lbl in _BINS_CHOICES:
            self._xb.addItem(lbl)
            self._fb.addItem(lbl)
        self._xb.setCurrentText(str(x_bins))
        self._fb.setCurrentText(str(f_bins))

        self._x_min, self._x_max, self._f_min, self._f_max = self._make_range_spins(
            x_min, x_max, f_min, f_max)

        x_row = QHBoxLayout()
        x_row.addWidget(QLabel("min")); x_row.addWidget(self._x_min)
        x_row.addSpacing(8)
        x_row.addWidget(QLabel("max")); x_row.addWidget(self._x_max)

        f_row = QHBoxLayout()
        f_row.addWidget(QLabel("min")); f_row.addWidget(self._f_min)
        f_row.addSpacing(8)
        f_row.addWidget(QLabel("max")); f_row.addWidget(self._f_max)

        self._form.addRow(self._x_bins_label(), self._xb)
        self._form.addRow(self._f_bins_label(), self._fb)
        self._form.addRow(self._x_range_label(), x_row)
        self._form.addRow(self._f_range_label(), f_row)

        self._seg = QComboBox()
        for lbl, key in _ALIGN_SEG_CHOICES:
            self._seg.addItem(lbl, key)
        j = self._seg.findData(align_segment)
        self._seg.setCurrentIndex(j if j >= 0 else 0)
        self._form.addRow("Segment:", self._seg)

        self._add_extra_rows()

        warn = QLabel("⚠  Changing these settings invalidates all cached 2DHs.")
        warn.setStyleSheet(f"color: {self._WARN_COLOR};")
        self._form.addRow(warn)
        root.addWidget(box)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Reset |
            QDialogButtonBox.StandardButton.Cancel |
            QDialogButtonBox.StandardButton.Apply,
        )
        btns.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.accept)
        btns.button(QDialogButtonBox.StandardButton.Cancel).clicked.connect(self.reject)
        btns.button(QDialogButtonBox.StandardButton.Reset).clicked.connect(self._reset)
        root.addWidget(btns)

    # ── Hooks for _PhysicalGridDialog ───────────────────────────────────────

    def _x_bins_label(self) -> str:
        return "X bins:"

    def _f_bins_label(self) -> str:
        return "F bins:"

    def _range_spec(self) -> tuple[tuple[float, float, str], ...]:
        """(lo, hi, quantities_key) for the x-min/x-max/F-min/F-max spins, in
        that order.  A subclass whose axes carry different units overrides ONLY
        this, leaving widget construction shared."""
        return ((-10.0, 10.0, "wlc_x_min"), (-10.0, 10.0, "wlc_x_max"),
                (-50.0, 200.0, "wlc_f_min"), (-50.0, 200.0, "wlc_f_max"))

    def _make_range_spins(self, x_min, x_max, f_min, f_max):
        """The only place a 2DH range spin box is constructed."""
        spins = []
        for (lo, hi, key), val in zip(self._range_spec(),
                                      (x_min, x_max, f_min, f_max)):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            _quant.configure_spinbox(s, key)   # step, and the " nm"/" pN" suffix
            s.setValue(val)
            spins.append(s)
        return tuple(spins)

    # Row labels for the range spins.  The normalized window's axes are BOTH
    # dimensionless (x/l_c and F·l_p/kT); the physical window's identically
    # laid-out dialog says " pN", so a bare "F range:" here invites reading
    # 45.0 as piconewtons.  The plot axes have always said otherwise — this
    # makes the dialog agree with them.
    #
    # Use plain text here; style.X_TILDE/L_C contain HTML intended for plot
    # renderers rather than Qt control labels.
    def _x_range_label(self) -> str:
        return "X range (x/l_c, dimensionless):"

    def _f_range_label(self) -> str:
        return "F range (F·l_p/kT, dimensionless):"

    def _add_extra_rows(self) -> None:
        pass

    def _reset(self) -> None:
        d = self._defaults
        self._xb.setCurrentText(str(d["x_bins"]))
        self._fb.setCurrentText(str(d["f_bins"]))
        self._x_min.setValue(d["x_min"]); self._x_max.setValue(d["x_max"])
        self._f_min.setValue(d["f_min"]); self._f_max.setValue(d["f_max"])
        self._seg.setCurrentIndex(max(0, self._seg.findData(d["align_segment"])))

    @property
    def values(self) -> dict:
        return {
            "x_bins": int(self._xb.currentText()),
            "f_bins": int(self._fb.currentText()),
            "x_min":  self._x_min.value(),
            "x_max":  self._x_max.value(),
            "f_min":  self._f_min.value(),
            "f_max":  self._f_max.value(),
            "align_segment": self._seg.currentData(),
        }


class _TwoDHWindowBase(QMainWindow):
    """
    Cohort 2D histogram over the selected Event Summary population. Per-curve
    histograms are raw integer counts; the displayed/exported matrix is their sum
    divided by the number of contributing curves (counts/trace).

    Two update paths:
      add_event()               — incremental, per new event during live analysis
      sync_from_event_summary() — full rebuild on first open / threshold change

    Subclasses (Normalized2DHWindow, Physical2DHWindow) supply only the
    coordinate transform and its UI trimmings — see the module docstring for
    which hooks to implement.
    """

    _physical = False   # Physical2DHWindow sets this True (PCAWindow labeling)

    def __init__(
        self,
        prepass_results: list[dict],
        db_path:         str,
        session_info:    dict | None = None,
        experimentalist:    str | None  = None,
        *,
        window_title:    str,
        population:      str = "hit",
    ) -> None:
        super().__init__()
        # Which EventSummaryWindow population ("hit"/"non_hit") this window
        # was opened for — fixed at open time, NOT reactive to that window's
        # Population selector afterward, so a Hits-2DH and a Non-Hits-2DH can
        # be open side by side without fighting over which one "wins" on the
        # next live refresh (sync_from_event_summary reads population_paths
        # for exactly this population, every time).
        self._population = population
        pop_label = "Hits" if population == "hit" else "Non-Hits"
        self.setWindowTitle(f"{window_title}  ({pop_label})")
        self.setWindowFlag(Qt.WindowType.Window)
        fit_on_screen(self, 750, 600)
        self._results          = prepass_results
        self._db_path          = db_path
        self._experimentalist     = experimentalist
        self._event_histograms: dict[str, np.ndarray] = {}
        self._event_summary_win  = None
        # Describes the inputs and exclusions for the current histogram build.
        self._ledger = _ledger.Ledger("2DH build", [])

        self._display:    np.ndarray | None = None
        self._cumulative: np.ndarray | None = None
        self._auto_max: float = 1.0
        self._z_pct:    int   = 100

        self._load_grid_params()

        style.apply_plot_defaults()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(4)
        root.setContentsMargins(8, 6, 8, 6)

        if session_info:
            hdr = _make_session_header(session_info)
            if hdr is not None:
                root.addWidget(hdr)

        # ── Controls row ──────────────────────────────────────────────────────
        ctrl = QHBoxLayout()
        self._stats_label = QLabel("0 events")
        font = self._stats_label.font()
        font.setPointSize(style.FONT_SMALL_PT)
        self._stats_label.setFont(font)

        self._grid_btn    = QPushButton("Grid settings…")
        self._rebuild_btn = QPushButton("Rebuild")
        self._rebuild_btn.setEnabled(False)
        self._sel_btn     = QPushButton("Select 2DH area")
        self._sel_btn.setCheckable(True)
        self._sel_btn.setToolTip(
            "Draw a rectangle on the 2DH to select the histogram bins used as "
            "features by PCA and k-means. Cluster histograms still show the "
            "full 2DH.\n\n"
            "This is a crop of THIS PICTURE. It is unrelated to a curve's own "
            "ROI, the region the event search found."
        )
        self._pca_btn     = QPushButton("Run PCA")
        self._pca_btn.setEnabled(False)
        self._export_btn  = QPushButton("Export 2DH…")
        self._export_btn.setToolTip(
            "Write the total 2DH (bare matrix + paired X/F 1D projections) "
            "and a provenance manifest (params + file list) to the export folder."
        )
        self._export_btn.setEnabled(False)

        self._grid_btn.clicked.connect(self._on_grid_settings)
        self._rebuild_btn.clicked.connect(self._on_rebuild)
        self._sel_btn.toggled.connect(self._on_selection_toggled)
        self._pca_btn.clicked.connect(self._run_pca)
        self._export_btn.clicked.connect(self._on_export_2dh)

        self._selection: pg.RectROI | None = None

        self._z_slider = QSlider(Qt.Orientation.Horizontal)
        self._z_slider.setRange(1, 200)
        self._z_slider.setValue(100)
        self._z_slider.setFixedWidth(110)
        self._z_slider.setToolTip("Intensity clip: drag left to highlight dense regions (clipped values shown in red)")
        self._z_label  = QLabel("100%")
        self._z_label.setMinimumWidth(36)
        self._z_auto_btn = QPushButton("Auto")
        self._z_auto_btn.setFixedWidth(42)
        self._z_slider.valueChanged.connect(self._on_z_slider)
        self._z_auto_btn.clicked.connect(self._on_z_auto)

        ctrl.addWidget(self._stats_label)
        ctrl.addSpacing(8)
        ctrl.addWidget(QLabel("Clip:"))
        ctrl.addWidget(self._z_slider)
        ctrl.addSpacing(4)
        ctrl.addWidget(self._z_label)
        ctrl.addWidget(self._z_auto_btn)
        ctrl.addStretch()
        ctrl.addWidget(self._grid_btn)
        ctrl.addSpacing(6)
        ctrl.addWidget(self._rebuild_btn)
        ctrl.addSpacing(6)
        ctrl.addWidget(self._sel_btn)
        ctrl.addSpacing(12)
        self._build_extra_controls(ctrl)   # hook: physical's Gaussian-ridge button
        ctrl.addWidget(self._pca_btn)
        ctrl.addWidget(self._export_btn)
        root.addLayout(ctrl)

        # ── 2D histogram plot ─────────────────────────────────────────────────
        self._plot = FixedDomainPlot(
            (self._x_min, self._x_max), (self._f_min, self._f_max))
        self._apply_axis_labels()   # sets units and prefixing together
        self._plot.showGrid(x=True, y=True, alpha=0.15)

        # ── Trace overlay panel ──────────────────────────────────────────────
        # Checkbox per curve currently in the total 2DH; ticking one plots
        # that single curve's own transformed (x, F) trace on top of the
        # cumulative histogram ("hard to see how individual traces build up
        # the 2DH"). MeanCurveWindow's PCA popouts reuse the same panel.
        self._overlay_panel = TraceOverlayPanel(self._plot, self._overlay_xF)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._plot)
        splitter.addWidget(self._overlay_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        root.addWidget(splitter, stretch=1)

        self._image = pg.ImageItem()
        self._lut = self._build_lut()
        self._plot.addItem(self._image)

        self._after_plot_setup()   # hook: normalized's master curve, physical's registration lines
        self._refresh_provenance_caption()

    # ── Grid params ───────────────────────────────────────────────────────────

    @property
    def _profile_key(self) -> str:
        # Fall back to a shared default so grid settings persist even when the
        # experimentalist is unknown — never silently no-op (see db.DEFAULT_EXPERIMENTALIST).
        return self._experimentalist or _db.DEFAULT_EXPERIMENTALIST

    def _load_grid_params(self) -> None:
        """This person's grid settings, falling back to the lab's.

        Precedence is deliberately the SAME as db.load_analysis_params and
        criteria_gate.get_criteria: this person's own value where they have one,
        the DEFAULT_EXPERIMENTALIST row's otherwise, the declared constant last.

        Constants are seeded into the Default row at initialise(); the final
        fallback also covers keys added after a database was created.
        """
        p = _db.get_experimentalist_profile(self._profile_key, self._db_path) or {}
        shared = ({} if self._profile_key == _db.DEFAULT_EXPERIMENTALIST
                  else _db.get_experimentalist_profile(
                      _db.DEFAULT_EXPERIMENTALIST, self._db_path) or {})
        for attr, key, caster, default in self._profile_spec():
            for src in (p, shared):
                if key in src:
                    setattr(self, attr, caster(src[key]))
                    break
            else:
                setattr(self, attr, default)

    def _save_grid_params(self) -> None:
        # Atomic merge (db.merge_experimentalist_profile), not read-then-write —
        # a stale read here would silently discard whatever ROIWindow/
        # DecompositionWindow/the other 2DH window just saved to this same row.
        updates = {key: getattr(self, attr) for attr, key, _caster, _default in self._profile_spec()}
        _db.merge_experimentalist_profile(self._profile_key, updates, self._db_path)

    def _profile_spec(self) -> list[tuple[str, str, type, object]]:
        """[(attr_name, profile_key, caster, default), …] — declares every
        grid-settings attribute a subclass wants persisted. Declaring the
        mapping once keeps the load and save paths synchronized."""
        raise NotImplementedError

    # ── Z-axis / intensity clip ───────────────────────────────────────────────

    def _build_lut(self) -> np.ndarray:
        return style.intensity_lut()                       # (256, 3) uint8

    def _apply_z_scale(self, display: np.ndarray) -> np.ndarray:
        clip_level = max(self._auto_max * self._z_pct / 100.0, 1e-10)
        scaled = np.clip(display / clip_level * 254.0, 0, 254).astype(np.uint8)
        scaled[display > clip_level] = 255
        return scaled

    def _draw(self) -> None:
        if self._display is None:
            return
        scaled = self._apply_z_scale(self._display)
        self._image.setImage(scaled, autoLevels=False, lut=self._lut, levels=(0, 255))
        self._image.setRect(QRectF(
            self._x_min, self._f_min,
            self._x_max - self._x_min,
            self._f_max - self._f_min,
        ))

    def _on_z_slider(self, val: int) -> None:
        self._z_pct = val
        self._z_label.setText(f"{val}%")
        self._refresh_provenance_caption()
        self._draw()

    def _on_z_auto(self) -> None:
        self._z_slider.setValue(100)

    # ── Controls ──────────────────────────────────────────────────────────────

    def _build_extra_controls(self, ctrl_layout: QHBoxLayout) -> None:
        pass   # physical adds its Gaussian-ridge button here

    def _on_grid_settings(self) -> None:
        dlg = self._make_grid_dialog()
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        v = dlg.values
        self._x_bins = v["x_bins"]; self._f_bins = v["f_bins"]
        self._x_min  = v["x_min"];  self._x_max  = v["x_max"]
        self._f_min  = v["f_min"];  self._f_max  = v["f_max"]
        self._align_segment = v["align_segment"]
        self._apply_extra_dialog_values(v)
        self._save_grid_params()
        self._plot.set_domain(
            (self._x_min, self._x_max), (self._f_min, self._f_max))
        self._apply_axis_labels()
        self._refresh_provenance_caption()
        self._after_grid_settings_applied()
        self._on_rebuild()

    def _make_grid_dialog(self) -> _GridDialog:
        raise NotImplementedError

    def _apply_extra_dialog_values(self, v: dict) -> None:
        pass   # physical pulls f_star/align_mode out of v

    def _apply_axis_labels(self) -> None:
        """(Re-)read _axis_labels() and push it onto the plot. Called at
        construction AND every time Grid settings are applied — physical's
        _axis_labels() reads self._align_mode, so this is what keeps the axis
        title from freezing at whatever mode was active when the window
        opened."""
        (x_text, x_unit), (f_text, f_unit) = self._axis_labels()
        # si=False on both: the grid dialog sets this window's x/F ranges in
        # plain nm and pN, and physical's F* line is positioned from a spin
        # box on this very axis.  An axis free to relabel itself µm would
        # disagree with the boxes that set it.
        set_si_label(self._plot, "bottom", x_text, x_unit, si=False)
        set_si_label(self._plot, "left",   f_text, f_unit, si=False)

    # ── Settings-provenance caption ──────────────────────────────────────────
    # What produced the CURRENT plot — population, event count, segment,
    # grid resolution, z-clip — written into the PlotItem's own title, not a
    # sibling QLabel. On-canvas deliberately: pyqtgraph's built-in right-click
    # "Export…" only captures what's inside the PlotItem (axis labels, title,
    # addItem'd elements), not the window title bar or _stats_label — so a
    # QLabel caption would look fine on screen and vanish from every exported
    # image. Rebuilt on every event that can change one of these fields, so
    # it cannot drift from what is currently plotted.

    _SEG_LABELS = {
        "first": "First", "penult": "Penultimate", "last": "Last",
        "primary": "Primary", "secondary": "Secondary",
    }

    def _provenance_caption(self) -> str:
        pop_label = "Hits" if self._population == "hit" else "Non-Hits"
        seg_label = self._SEG_LABELS.get(self._align_segment, self._align_segment)
        n = len(self._event_histograms)
        parts = [
            pop_label,
            f"{n} events",
            f"seg: {seg_label}",
            f"grid: {self._x_bins}×{self._f_bins}",
            f"z-clip: {self._z_pct}%",
        ]
        extra = self._provenance_extra()
        if extra:
            parts.append(extra)
        return "   ·   ".join(parts)

    def _provenance_extra(self) -> str:
        return ""   # physical adds align mode (+ F* when in F*-mode)

    def _refresh_provenance_caption(self) -> None:
        set_plot_title(self._plot, caption=self._provenance_caption())

    def export_provenance(self) -> dict:
        """Same facts as _provenance_caption(), structured for a machine
        reader (export manifests) instead of flattened into display text —
        the caption string is for a human looking at the plot; this is for
        a script re-checking what produced an exported file later."""
        d = {
            "window":      "physical" if self._physical else "normalized",
            "population":  self._population,
            "segment":     self._align_segment,
            "n_events":    len(self._event_histograms),
            "x_bins":      self._x_bins,
            "f_bins":      self._f_bins,
            "x_range":     [self._x_min, self._x_max],
            "f_range":     [self._f_min, self._f_max],
            "z_clip_pct":  self._z_pct,
            "z_quantity":  "counts_per_trace",
            "z_normalization": "sum_of_raw_bin_counts_divided_by_n_traces",
            "per_trace_histogram_storage": "uint32_raw_bin_counts",
            # This stage's reconciled tally. Absorbed upstream records remain
            # available to the in-memory journey but do not change these
            # stage-local manifest counts.
            "drops":       self._ledger.manifest(),
        }
        d.update(self._export_provenance_extra())
        return d

    def _export_provenance_extra(self) -> dict:
        return {}   # physical adds align_mode (+ f_star_pN when in F*-mode)

    def _after_grid_settings_applied(self) -> None:
        pass   # physical updates its vline/hline registration visuals here

    def _on_rebuild(self) -> None:
        self._event_histograms.clear()
        if self._event_summary_win is not None:
            self.sync_from_event_summary(self._event_summary_win)
        else:
            self._refresh()

    # ── Incremental update (one new event) ──────────────────────────────────────

    def add_event(self, path: str) -> None:
        fit = self._stored_segment_fit(path)
        if fit is None:
            return
        l_p, l_c, _right_idx = fit
        if self._requires_wlc_fit() and (l_p is None or l_c is None):
            return
        H = self._load_or_compute(path)
        if H is not None:
            self._event_histograms[path] = H
            self._refresh()

    # ── Full rebuild ──────────────────────────────────────────────────────────

    def sync_from_event_summary(self, event_summary_win) -> None:
        self._event_summary_win = event_summary_win
        self._rebuild_btn.setEnabled(False)
        self._stats_label.setText("Building…")
        QApplication.processEvents()

        # Absorb the upstream ledger so exclusions made earlier in the
        # pipeline stay visible in this build's provenance.
        upstream = event_summary_win.population_ledger(self._population)
        valid_paths = upstream.kept()
        led = _ledger.Ledger(
            f"{'Physical' if self._physical else 'Normalized'} 2DH build",
            valid_paths)

        if not valid_paths:
            self._event_histograms.clear()
            led.absorb(upstream)
            self._ledger = led
            self._rebuild_btn.setEnabled(True)
            self._refresh()
            return

        prog_dlg = CancelableProgress(self, "Building 2D histogram…", len(valid_paths))

        # Landmark bulk query — eliminates per-curve DB round-trips for offset/
        # invols/snap-off. The WLC fit itself is looked up per-curve below via
        # _stored_segment_fit, which reads the chosen segment from event_map.
        landmark_cached = _db.get_derived_results_bulk_latest(
            valid_paths, ["snapoff_piezo_nm", "offset_retr", "invols_slope"], self._db_path
        )

        new_histograms: dict[str, np.ndarray] = {}
        pending_writes: list = []   # (file_id, H, grid_key) — flushed in one transaction
        n = len(valid_paths)

        # Share one connection across the rebuild's repeated database reads.
        conn = _db.get_connection(self._db_path)
        try:
            needs_fit = self._requires_wlc_fit()
            for i, path in enumerate(valid_paths):
                key = _db.normalize_path(path)
                fit = self._stored_segment_fit(path, conn=conn)
                if fit is None:
                    led.drop(path, "no_segment_chosen", f"segment: {self._align_segment}")
                    continue
                l_p, l_c, _right_idx = fit
                # Only require the WLC fit when the transform actually USES
                # it. Onset, snap-off, and rupture anchors are observed data
                # points and therefore do not require a fit.
                if needs_fit and (l_p is None or l_c is None):
                    led.drop(path, "no_fit",
                             f"segment: {self._align_segment}, "
                             f"l_p={'ok' if l_p is not None else 'None'}, "
                             f"l_c={'ok' if l_c is not None else 'None'}")
                    continue

                file_id = _db.get_file_id(path, self._db_path, conn=conn)
                if file_id is None:
                    led.drop(path, "not_in_catalog")
                    continue

                H = _db.get_event_histogram(file_id, self._grid_key, self._db_path, conn=conn)
                if H is None:
                    ld = landmark_cached.get(key, {})
                    H = self._compute_from_curve(path, pre_fetched=ld, conn=conn)
                    if H is not None:
                        pending_writes.append((file_id, H, self._grid_key))

                if H is not None:
                    new_histograms[path] = H
                else:
                    led.drop(path, "no_histogram")

                if i % 10 == 0 or i == n - 1:
                    self._stats_label.setText(f"Building… {i + 1}/{n}")
                    if prog_dlg.tick(i + 1, n):
                        # Partial results below are still real and are kept —
                        # but the curves never reached are recorded as such,
                        # so a cancelled build reports a smaller cohort with a
                        # reason rather than looking like a complete one.
                        led.drop_all(valid_paths[i + 1:], "cancelled")
                        break
        finally:
            conn.close()
            prog_dlg.close()

        if pending_writes:
            write_event_histograms_bulk(pending_writes, self._db_path)

        led.absorb(upstream)
        self._ledger = led
        self._event_histograms = new_histograms
        self._rebuild_btn.setEnabled(True)
        self._refresh()

    # ── Per-event histogram: DB-first, curve fallback ───────────────────────────

    def _load_or_compute(self, file_path: str) -> np.ndarray | None:
        file_id = _db.get_file_id(file_path, self._db_path)
        if file_id is None:
            return None
        H = _db.get_event_histogram(file_id, self._grid_key, self._db_path)
        if H is not None:
            return H
        H = self._compute_from_curve(file_path)
        if H is not None:
            _db.write_event_histogram(file_id, H, self._grid_key, self._db_path)
        return H

    # ── Shared read path: curve → full-retract (x, F) → chosen-segment fit ────

    def _full_xF(
        self, file_path: str, pre_fetched: dict | None = None,
    ) -> "tuple[np.ndarray, np.ndarray] | None":
        """Full retract (x, F) in physical nm/pN units — x measured from
        snap-off (tip–surface contact), F = k·δ. Neither window's own
        transform (physical's anchor shift, normalized's l_c/l_p scaling) has
        happened yet; this is the common prefix both build on."""
        if pre_fetched is not None:
            d = pre_fetched
        else:
            cached = _db.get_derived_results_bulk_latest(
                [file_path],
                ["snapoff_piezo_nm", "offset_retr", "invols_slope"],
                self._db_path,
            )
            d = cached.get(_db.normalize_path(file_path), {})
        if any(k not in d for k in ["snapoff_piezo_nm", "offset_retr", "invols_slope"]):
            return None
        try:
            curve = load_force_curve(file_path)
        except LoadError:
            return None
        defl_corr = (curve.defl_retr - d["offset_retr"][0]) / d["invols_slope"][0]
        x = (curve.piezo_retr - d["snapoff_piezo_nm"][0]) - defl_corr
        F = curve.spring_constant * defl_corr
        return x, F

    def _resolve_fit(self, file_path: str, n: int, conn=None):
        """(lo, hi, l_p, l_c, right_idx) for the last-outer-ROI's chosen
        segment (self._align_segment), or None if there's no stored ROI, the
        span is degenerate, or the chosen segment doesn't exist. l_p/l_c may
        still be None within a non-None result (fit failed) — callers that
        need a real fit (both windows' histogram/overlay builders) check for
        that themselves; add_event's fit-quality gate does too."""
        span = self._stored_roi_span(file_path, n, conn=conn)
        if span is None:
            return None
        fit = self._stored_segment_fit(file_path, conn=conn)
        if fit is None:
            return None
        lo, hi = span
        l_p, l_c, right_idx = fit
        return lo, hi, l_p, l_c, right_idx

    def _stored_roi_span(self, file_path: str, n: int, conn=None):
        """[onset_idx, return_idx] of the LAST (most baseline-ward) outer ROI,
        READ from the roi finder's MOST RECENTLY registered event_map document
        for this file — the ROI's own bounds, verbatim, regardless of which
        params it was registered under. Pure read, never runs the finder.
        Returns None only when the file has no registered ROI carrying
        ruptures, or the span is degenerate."""
        fid = _db.get_file_id(file_path, self._db_path, conn=conn)
        if fid is None:
            return None
        from .roi_events import payload_to_events
        import json as _json
        doc = _db.get_latest_event_map(fid, self._db_path, conn=conn)
        if doc is None:
            return None
        events = payload_to_events(_json.loads(doc) if isinstance(doc, str) else doc)
        if events is None:
            return None
        roi = next((r for r in reversed(events.rois) if r.ruptures), None)
        if roi is None:
            return None
        lo, hi = sorted((int(roi.onset_idx), int(roi.return_idx)))
        lo, hi = max(0, lo), min(n - 1, hi)
        return (lo, hi) if hi - lo >= 1 else None

    def _stored_segment_fit(self, file_path: str, conn=None):
        """(l_p, l_c, right_idx) of the CHOSEN inner segment (self.
        _align_segment: first/penultimate/last/primary/secondary) of the
        last-outer ROI, READ from the finder's stored event_map document.
        l_p/l_c are None if that segment's WLC fit failed or is missing;
        right_idx (the segment's terminating rupture, an index into the
        curve's arrays — used by physical's "rupture" align mode) doesn't
        depend on the fit and is set whenever the segment itself exists.
        Pure read. Returns None when there is no stored ROI, or the chosen
        segment doesn't exist, or (primary/secondary only) that manual
        pick is unset or stale relative to the current segmentation — see
        roi_pipeline.resolve_segment_override."""
        fid = _db.get_file_id(file_path, self._db_path, conn=conn)
        if fid is None:
            return None
        from .roi_events import payload_to_events
        import json as _json
        doc = _db.get_latest_event_map(fid, self._db_path, conn=conn)
        if doc is None:
            return None
        events = payload_to_events(_json.loads(doc) if isinstance(doc, str) else doc)
        if events is None:
            return None
        roi = next((r for r in reversed(events.rois) if r.ruptures), None)
        if roi is None or not roi.segments:
            return None
        segs = roi.segments                       # onset→r1, r1→r2, …, →terminal
        if self._align_segment in ("primary", "secondary"):
            from .roi_pipeline import event_geometry_identity, resolve_segment_override
            override = _db.get_segment_override(fid, self._db_path, conn=conn)
            current_params = _db.get_latest_event_map_params(fid, self._db_path, conn=conn)
            primary_idx, secondary_idx = resolve_segment_override(
                override, current_params, len(segs), event_geometry_identity(events))
            idx = primary_idx if self._align_segment == "primary" else secondary_idx
            if idx is None:
                return None
            seg = segs[idx]
        elif self._align_segment == "last":
            seg = segs[-1]
        elif self._align_segment == "penult":
            seg = segs[-2] if len(segs) >= 2 else segs[-1]   # "when present", else last
        else:                                     # "first"
            seg = segs[0]
        l_p = float(seg.l_p_nm) if seg.l_p_nm is not None else None
        l_c = float(seg.l_c_nm) if seg.l_c_nm is not None else None
        return l_p, l_c, seg.right_idx

    def _compute_from_curve(
        self, file_path: str, pre_fetched: dict | None = None, conn=None,
    ) -> np.ndarray | None:
        """This curve's own 2DH for the LAST (most baseline-ward) outer ROI's
        chosen segment. The ROI *and* its WLC fit are both CONSUMED from the
        roi finder's stored output (event_map) — never re-derived, never
        re-fit here."""
        full = self._full_xF(file_path, pre_fetched)
        if full is None:
            return None
        x, F = full
        resolved = self._resolve_fit(file_path, len(x), conn=conn)
        if resolved is None:
            return None
        lo, hi, l_p, l_c, right_idx = resolved
        return self._build_histogram(x, F, lo, hi, l_p, l_c, right_idx)

    def _build_histogram(self, x, F, lo, hi, l_p, l_c, right_idx) -> np.ndarray | None:
        """Subclass hook: this curve's transform + binning."""
        raise NotImplementedError

    def _requires_wlc_fit(self) -> bool:
        """Does THIS window's transform actually need l_p/l_c to place a curve?

        True on the base class because the normalized 2DH divides x by l_c —
        without a fit there is no x̃ and the curve genuinely cannot be placed.
        Physical overrides it: only its fstar and lc anchors consume the fit,
        while onset, snap-off and rupture are all real data points.

        This exists as a hook rather than a flag because the answer changes
        with the align mode at runtime — a curve excluded from an F*-aligned
        2DH must reappear the moment the user switches to Rupture, which is
        required by the current transformation.
        """
        return True

    # ── Display ───────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        n = len(self._event_histograms)
        # Say what this build was given, not only what it kept. The
        # tooltip carries the per-reason breakdown; the line itself stays one
        # line because it sits directly under the image it describes.
        if self._ledger.n_dropped:
            self._stats_label.setText(f"{n:,} events   ·   {self._ledger.summary('built')}")
            self._stats_label.setToolTip(self._ledger.report())
        else:
            self._stats_label.setText(f"{n} events")
            self._stats_label.setToolTip("")
        self._refresh_provenance_caption()
        self._pca_btn.setEnabled(n >= 2)
        self._export_btn.setEnabled(n > 0)
        self._on_refresh_extra(n)
        self._overlay_panel.set_paths(self._event_histograms.keys())
        if n == 0:
            self._display = None
            self._cumulative = None
            self._image.clear()
            return
        self._cumulative = _counts_per_trace(self._event_histograms.values())
        self._display  = np.sqrt(self._cumulative)
        self._auto_max = max(float(self._display.max()), 1e-10)
        self._draw()

    def _on_refresh_extra(self, n: int) -> None:
        pass   # physical enables its Gaussian-ridge button here

    # ── Export ────────────────────────────────────────────────────────────────

    def _on_export_2dh(self) -> None:
        """Write the total 2DH as four files sharing one basename:
          _matrix.csv    bare x_bins × f_bins grid (self._cumulative,
                         row=x bin, col=F bin — numpy's own histogram2d
                         orientation, unrotated) — nothing else, so it pastes
                         straight into Excel/Origin as a matrix.
          _x.csv/_f.csv  each axis's own marginal projection (bin_left,
                         bin_right, count) — real 1D histograms in their own
                         right, and their edges ARE the matrix file's axis
                         definition, which is why the matrix can stay bare.
          _manifest.json population/segment/grid/z-clip/align mode + the
                         file list that went into the sum — the "parameters
                         that made it" the matrix/edge files can't carry
                         themselves.
        Values are counts/trace: raw integer bin counts summed over the cohort
        and divided by the number of contributing traces (see _refresh()).
        """
        if self._cumulative is None or len(self._event_histograms) == 0:
            QMessageBox.information(
                self, "Export 2DH", "Nothing to export — build the 2D histogram first.")
            return

        stem = f"2dh_{'physical' if self._physical else 'normalized'}_{self._population}"
        with _export.export_group(
            self._db_path, stem, ["_matrix.csv", "_x.csv", "_f.csv"],
            kind="2dh",
        ) as g:
            g.contributing_files(self._event_histograms.keys())
            g.note_dict(self.export_provenance())
            x_edges = np.linspace(self._x_min, self._x_max, self._x_bins + 1)
            f_edges = np.linspace(self._f_min, self._f_max, self._f_bins + 1)
            g.matrix("_matrix.csv", self._cumulative)
            g.histogram("_x.csv", x_edges, self._cumulative.sum(axis=1))
            g.histogram("_f.csv", f_edges, self._cumulative.sum(axis=0))

        QMessageBox.information(self, "Export 2DH", g.message())

    def _after_plot_setup(self) -> None:
        pass   # normalized draws the master WLC curve; physical adds registration lines

    # ── Trace overlay ────────────────────────────────────────────────────────

    def _overlay_xF(self, file_path: str, pre_fetched: dict | None = None, conn=None):
        """(x, F) for one curve, transformed into the same coordinates as the
        2DH plot — used only by the overlay (lazy, on checkbox toggle)."""
        full = self._full_xF(file_path, pre_fetched)
        if full is None:
            return None
        x, F = full
        resolved = self._resolve_fit(file_path, len(x), conn=conn)
        if resolved is None:
            return None
        lo, hi, l_p, l_c, right_idx = resolved
        return self._build_overlay_xF(x, F, lo, hi, l_p, l_c, right_idx)

    def _build_overlay_xF(self, x, F, lo, hi, l_p, l_c, right_idx):
        """Subclass hook: same transform as _build_histogram, but returning
        plot-ready (x, F) arrays instead of a binned histogram."""
        raise NotImplementedError

    # ── Selection window ──────────────────────────────────────────────────────
    # A view-level crop of THIS PICTURE, restricting what PCA/k-means are fed.
    # Deliberately not called an ROI: that word names the per-curve
    # detection region the event search produces — the scientific object
    # carrying ruptures and segments, which reaches event_map, roi_pipeline,
    # the seg_* columns and the criteria gate.
    # (pg.RectROI below is pyqtgraph's own class name, not ours.)

    def _on_selection_toggled(self, checked: bool) -> None:
        if checked:
            cx = (self._x_min + self._x_max) / 2
            cy = (self._f_min + self._f_max) / 2
            rw = (self._x_max - self._x_min) * 0.4
            rh = (self._f_max - self._f_min) * 0.4
            self._selection = pg.RectROI(
                [cx - rw / 2, cy - rh / 2], [rw, rh],
                pen=pg.mkPen(style.REFERENCE, width=style.W_GUIDE),
            )
            self._plot.addItem(self._selection)
        else:
            if self._selection is not None:
                self._plot.removeItem(self._selection)
                self._selection = None

    # ── PCA ──────────────────────────────────────────────────────────────────

    def _run_pca(self) -> None:
        try:
            self._run_pca_impl()
        except Exception:
            import traceback
            QMessageBox.critical(
                self, "Run PCA failed",
                "PCA could not be computed:\n\n" + traceback.format_exc(),
            )

    def _run_pca_impl(self) -> None:
        from .pca_window import PCAWindow

        if not self._event_histograms:
            raise RuntimeError(
                "No event histograms are loaded — build the 2D histogram first."
            )

        pca_histograms  = dict(self._event_histograms)
        x_bins          = self._x_bins
        f_bins          = self._f_bins
        x_range         = (self._x_min, self._x_max)
        f_range         = (self._f_min, self._f_max)
        display_hists   = None
        display_x_range = None
        display_f_range = None
        feature_space_selected = False

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
                feature_space_selected = True
                x_bins  = xi_hi - xi_lo
                f_bins  = fi_hi - fi_lo
                x_range = (self._x_min + xi_lo * dx, self._x_min + xi_hi * dx)
                f_range = (self._f_min + fi_lo * df, self._f_min + fi_hi * df)
                pca_histograms  = {
                    p: H[xi_lo:xi_hi, fi_lo:fi_hi]
                    for p, H in self._event_histograms.items()
                }
                display_hists   = dict(self._event_histograms)
                display_x_range = (self._x_min, self._x_max)
                display_f_range = (self._f_min, self._f_max)

        (x_label, x_unit), (f_label, f_unit) = self._axis_labels()
        provenance = dict(self.export_provenance())
        if feature_space_selected:
            # PCA ran on the selected 2DH area, not the full grid — the
            # manifest must describe what was ACTUALLY fed to PCA, not the
            # window's full grid settings. The key says "feature space" rather
            # than "ROI", which elsewhere means a curve's detection region.
            provenance["x_bins"] = x_bins
            provenance["f_bins"] = f_bins
            provenance["x_range"] = list(x_range)
            provenance["f_range"] = list(f_range)
            # Keep the flat selection_window flag for downstream consumers;
            # feature_space_selection carries the exact crop.
            provenance["selection_window"] = True
            provenance["feature_space_selection"] = {
                "x_bin_start": xi_lo,
                "x_bin_stop": xi_hi,
                "f_bin_start": fi_lo,
                "f_bin_stop": fi_hi,
                "x_range": list(x_range),
                "f_range": list(f_range),
            }
        caption = self._provenance_caption()
        if feature_space_selected:
            caption += (
                f"   ·   PCA feature space: {x_bins}×{f_bins} bins"
                f"   ·   {x_label}: {x_range[0]:g}–{x_range[1]:g} {x_unit}"
                f"   ·   {f_label}: {f_range[0]:g}–{f_range[1]:g} {f_unit}"
            )
        self._pca_win = PCAWindow(
            histograms         = pca_histograms,
            x_bins              = x_bins,
            f_bins              = f_bins,
            x_range             = x_range,
            f_range             = f_range,
            x_label             = x_label,
            f_label             = f_label,
            x_unit              = x_unit,
            f_unit              = f_unit,
            caption             = caption,
            provenance          = provenance,
            db_path             = self._db_path,
            display_histograms  = display_hists,
            display_x_range     = display_x_range,
            display_f_range     = display_f_range,
            physical            = self._physical,
            overlay_fn          = self._overlay_xF,
        )
        self._pca_win.show()

    def _axis_labels(self) -> tuple[tuple[str, str], tuple[str, str]]:
        """((x text, x unit), (F text, F unit)) for this window's coordinates.

        Text and unit are separate, not one string with "[nm]" spliced in.
        The unit belongs to the display layer (set_si_label decides how to
        render it); the text is what reaches the export manifest, where it
        must stay plain and machine-readable.
        """
        raise NotImplementedError
