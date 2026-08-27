# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/display_roi.py
#
# ROI detection diagnostic window.
#
# Four stacked panels; rows 0-2 are x-linked in natural (forward) retract
# orientation, row 3 is force-extension (its own x-axis):
#   row 0 — low-channel retract (nm, baseline-subtracted)
#   row 1 — sliding mean deviation from anchor mean (nm)
#   row 2 — first derivative via Savitzky-Golay (nm deflection per nm piezo)
#   row 3 — force vs extension (pN vs nm): per-segment WLC fits + rupture forces
#
# All signals are plotted in their physical units — not normalised — so
# magnitudes can be compared against physical expectations (e.g. WLC slope).
#
# Overlays on every panel:
#   - translucent band over the far-baseline anchor region (excluded from search)
#   - translucent band over the post-snap-off mask region (excluded from search)
#   - per ROI: onset (orange dash) / return (orange dot) boundary lines
#   - per rupture: a vertical line, GREEN for the outer/terminal rupture that
#     opened the ROI, MAGENTA for any inner sub-event rupture inside it — see
#     _draw_multi's docstring for why these must never share a colour.
#
# Row 3 is the exception to that colour pairing: there the fits and markers are
# keyed to ROI IDENTITY (style.roi_segment_qcolor, right-most ROI always blue),
# and terminal-vs-inner is carried by marker SYMBOL (circle vs diamond) rather
# than by hue — otherwise the green/magenta pair would have to dodge the ROI
# hues drawn in the same panel.  See style.py § E.
# Its force/extension data, rupture markers, and WLC overlays all use the same
# decomposed low-frequency retract that roi_events.fit_segments fits. The raw
# retract is context for overview plots, not fit data.
#
# Draggable threshold markers:
#   - horizontal lines on the d1 panel       → _threshold_nm_per_nm (outer, green),
#                                                _inner_threshold_nm_per_nm (inner, magenta)
#   - horizontal line on the mean_dev panel   → _onset_threshold_nm
#
# The window displays the multi-event detector used by the analysis pipeline.
# find_rupture/find_onset remain the stage-one event classifier but are not a
# separate display mode here.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .curve_loader import ForceCurve, LoadError, load_force_curve
from . import db as _db
from . import quantities as _quant
from .models import wlc
from .roi_pipeline import (
    DETECTOR_BY_IDX, DETECTOR_MODE_LABELS, MODE_TO_STORED_IDX,
    compute_curve_events_coords, event_geometry_identity, event_params_from,
    resolve_segment_override_state,
)
from . import sample_marks
from . import style
from .qt_utils import _make_session_header, set_si_label, fit_on_screen
from .widgets import FlowLayout, LabeledControl, SampleMarksToggle
from .navigator_bar import WorkerNavBar
from .provenance import cache_version


# ── Colours ──────────────────────────────────────────────────────────────────
# All colours come from style.py. Red is reserved for status, while the d1
# signal uses violet to remain distinguishable from rupture markers. The F-x
# panel uses the same per-ROI hue and per-segment shade as the WLC fit window.
# The detection panels stack signals, thresholds, masks and rupture markers on
# one plot, so everything here is drawn slightly transparent to read through
# what is over it.  For the three signal traces that is a declared exception to
# rule 1's "opaque" — style.data_pen takes the alpha rather than each trace
# building its own pen.
_ALPHA_SIGNAL   = 220
_ALPHA_FX       = 200
_COLOR_ZERO     = style.rgba(style.GRID, 220)
_COLOR_MASK     = style.rgba(style.INK_FAINT, 70)
_COLOR_RUPTURE  = style.rgba(style.LM_RUPTURE, 220)        # OUTER/TERMINAL rupture
_COLOR_RUPTURE_INNER = style.rgba(style.LM_RUPTURE_I, 220)  # INNER (sub-event)
_COLOR_ONSET    = style.rgba(style.LM_ONSET, 220)
_COLOR_THRESH   = style.rgba(style.LM_THRESHOLD, 220)


# ── Keys this window owns in the experimentalist_profiles JSON blob ─────────────────────
class ROIWindow(QWidget):
    """
    Diagnostic window for the multi-event ROI/rupture/segment detector: the
    same build_curve_events + fit_segments the batch worker runs, shown live on
    one curve so its parameters can be tuned against real data.

    X axis: piezo position (nm), natural/forward orientation.
    """

    # (label, mode) — mode persisted via roi_pipeline.MODE_TO_STORED_IDX, so the
    # on-disk integer means the same thing here as it does to the worker. This
    # is the single UI-facing options list; roi_pipeline.py owns the encoding.
    _DETECTOR_MODES = DETECTOR_MODE_LABELS

    def __init__(
        self,
        db_path:      str,
        experimentalist: str | None  = None,
        session_info: dict | None = None,
        worker=None,
    ) -> None:
        super().__init__()
        self._db_path       = db_path
        # Profile owner — INITIAL value only.  _sync_profile_owner re-resolves
        # it from db.active_param_owner (the file at position one of the
        # analysis queue), which is the ONE answer to "whose parameter set are
        # we using".  It does NOT follow the curve on screen.
        self._experimentalist  = experimentalist
        self._owner_synced  = False   # force a profile load on the first draw
        self._worker        = worker
        self._nav           = None   # WorkerNavBar, built below in worker mode
        self._results:  list[dict] = []
        self._current_index: int = -1
        self._last_curve: ForceCurve | None = None
        self._last_path:  str | None = None
        self._last_events = None

        # Title names whose parameter set these controls hold — the one answer
        # (db.active_param_owner), refreshed by _sync_profile_owner whenever
        # the queue changes it.
        # The verb, then the established noun: this window is where
        # events are SEARCHED FOR, and what it finds is that curve's ROI —
        # the name the schema, the seg_* columns and every export use.
        self._title_stem = "SMFS — event search — ROI detection signals"
        self.setWindowTitle(self._title_stem)
        fit_on_screen(self, 900, 700)
        # ── Load persistent settings ──────────────────────────────────────────
        # Load one complete profile so every control uses the same parameter
        # owner. _sync_profile_owner reloads it once a curve is on screen.
        _ps = _db.load_analysis_params(db_path)
        self._window_pts                = int(  _ps['roi_window_pts'])
        self._threshold_nm_per_nm       = float(_ps['roi_threshold_nm_per_nm'])
        # Inner (sub-event) d1 threshold — smaller than the outer/terminal one.
        # Defaults to the outer value (single-tier) until the user lowers it;
        # AnalysisParams applies that fallback, so it is not repeated here.
        self._inner_threshold_nm_per_nm = float(_ps['roi_inner_threshold_nm_per_nm'])
        self._post_snapoff_mask_nm      = float(_ps['roi_post_snapoff_mask_nm'])
        self._onset_threshold_nm        = float(_ps['roi_onset_threshold_nm'])

        # Detector: build_curve_events with the d1-threshold vs find_peaks inner
        # detector — the exact same function the batch worker calls (see
        # roi_pipeline.compute_curve_events). The mode is persisted as an
        # INTEGER INDEX, decoded via roi_pipeline.DETECTOR_BY_IDX so this
        # window and the worker can never decode a stored value two different
        # ways.
        self._detector_mode  = DETECTOR_BY_IDX.get(
            int(_ps['roi_detector_mode_idx']), "threshold")
        self._prominence     = float(_ps['roi_prominence'])
        self._distance_pts   = int(  _ps['roi_min_distance_pts'])
        # Must mirror roi_pipeline.event_params_from's default: this window is
        # supposed to SHOW what the worker STORES, so a drop_frac that differs
        # from the worker's means the markers on screen are not the ones going
        # into event_map. Both now come from one AnalysisParams snapshot, so they cannot
        # differ.
        # Dynamically created (variable-count) markers for multi mode.
        self._multi_items: list = []

        # ── Layout ────────────────────────────────────────────────────────────
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        if session_info:
            root.addWidget(_make_session_header(session_info))

        # Curve identity — directory + filename of the curve on screen, selectable
        # so it can be copied/read straight off the window (updated in update_curve).
        self._file_label = QLabel("(no curve)")
        self._file_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self._file_label.setStyleSheet(
            style.qss_text(style.UI_TEXT, size_px=11, mono=True))
        root.addWidget(self._file_label)

        self._params_label = QLabel("Parameters: not resolved")
        self._params_label.setStyleSheet(style.qss_text(style.UI_MUTED))
        root.addWidget(self._params_label)

        # ── Worker navigation strip (Prev/Next/scrubber) ──────────────────────
        # Present only in worker mode: drives the shared worker so this window can
        # scrub the queue itself and stays synced with every other worker view.
        if self._worker is not None:
            self._nav = WorkerNavBar(self._worker, db_path)
            self._nav.curve_selected.connect(self._on_nav_curve_selected)
            self._nav.curve_cleared.connect(self._on_nav_curve_cleared)
            root.addWidget(self._nav)

        # Controls row.  FlowLayout so the strip wraps to another line instead
        # of setting the window's minimum width -- the spin boxes size
        # themselves from the font now, so their width is not known in advance.
        ctrl = FlowLayout(margin=0, h_spacing=14, v_spacing=4)

        self._spin_window = QSpinBox()
        self._spin_window.setRange(11, 501)
        _quant.configure_spinbox(self._spin_window, "roi_window_pts", suffix=False)
        self._spin_window.setValue(self._window_pts)
        self._spin_window.editingFinished.connect(self._commit_window)
        ctrl.addWidget(LabeledControl("Window (pts):", self._spin_window))

        self._spin_threshold = QDoubleSpinBox()
        # No policy ceiling on a detection threshold: mathematical necessity is
        # enforced, taste is not.  On a noisy cohort a d1 threshold above 10 is
        # the right answer, and a clamp would snap the dragged line back with
        # nothing on screen saying why.  +-1e6 is a spin-box necessity (a
        # QDoubleSpinBox must have SOME range), not a claim about the science.
        self._spin_threshold.setRange(-1e6, 1e6)
        _quant.configure_spinbox(self._spin_threshold, "roi_threshold_nm_per_nm", suffix=False)
        self._spin_threshold.setValue(self._threshold_nm_per_nm)
        self._spin_threshold.setMinimumWidth(88)
        self._spin_threshold.editingFinished.connect(self._commit_threshold)
        ctrl.addWidget(LabeledControl("d¹ outer thr (nm/nm):", self._spin_threshold))

        self._spin_mask = QDoubleSpinBox()
        self._spin_mask.setRange(0.0, 2000.0)
        _quant.configure_spinbox(self._spin_mask, "roi_post_snapoff_mask_nm", suffix=False)
        self._spin_mask.setValue(self._post_snapoff_mask_nm)
        self._spin_mask.editingFinished.connect(self._commit_mask)
        ctrl.addWidget(LabeledControl("Post-snap mask (nm):", self._spin_mask))

        self._spin_onset = QDoubleSpinBox()
        self._spin_onset.setRange(-1e6, 1e6)
        _quant.configure_spinbox(self._spin_onset, "roi_onset_threshold_nm", suffix=False)
        self._spin_onset.setValue(self._onset_threshold_nm)
        self._spin_onset.setMinimumWidth(88)
        self._spin_onset.editingFinished.connect(self._commit_onset)
        ctrl.addWidget(LabeledControl("Onset threshold (nm):", self._spin_onset))

        _ctrl_bar = QWidget()
        _ctrl_bar.setLayout(ctrl)
        _ctrl_bar.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        root.addWidget(_ctrl_bar)

        # ── Second controls row: multi-event detector + status ────────────────
        # Kept on its own row so the window's minimum width is not forced wide by
        # a single long control strip.
        ctrl2 = FlowLayout(margin=0, h_spacing=14, v_spacing=4)

        self._spin_inner = QDoubleSpinBox()
        self._spin_inner.setRange(-1e6, 1e6)
        _quant.configure_spinbox(self._spin_inner, "roi_inner_threshold_nm_per_nm", suffix=False)
        self._spin_inner.setValue(self._inner_threshold_nm_per_nm)
        self._spin_inner.setMinimumWidth(88)
        self._spin_inner.setToolTip("Smaller d¹ threshold for inner sub-events; "
                                    "an ROI still needs one rupture above the outer "
                                    "threshold to be kept.")
        self._spin_inner.editingFinished.connect(self._commit_inner_threshold)
        ctrl2.addWidget(LabeledControl("d¹ inner thr:", self._spin_inner))

        self._combo_detector = QComboBox()
        for label, _mode in self._DETECTOR_MODES:
            self._combo_detector.addItem(label)
        cur = next((i for i, (_, m) in enumerate(self._DETECTOR_MODES)
                    if m == self._detector_mode), 0)
        self._combo_detector.setCurrentIndex(cur)
        self._combo_detector.currentIndexChanged.connect(self._on_detector_changed)
        ctrl2.addWidget(LabeledControl("Detector:", self._combo_detector))

        self._spin_prom = QDoubleSpinBox()
        self._spin_prom.setRange(0.0, 10.0)
        _quant.configure_spinbox(self._spin_prom, "roi_prominence", suffix=False)
        self._spin_prom.setValue(self._prominence)
        self._spin_prom.editingFinished.connect(self._commit_prominence)
        ctrl2.addWidget(LabeledControl("Prominence:", self._spin_prom))

        self._spin_dist = QSpinBox()
        self._spin_dist.setRange(1, 2000)
        _quant.configure_spinbox(self._spin_dist, "roi_min_distance_pts", suffix=False)
        self._spin_dist.setValue(self._distance_pts)
        self._spin_dist.editingFinished.connect(self._commit_distance)
        ctrl2.addWidget(LabeledControl("Min dist (pts):", self._spin_dist))

        # Word-wrapped: this carries runtime text ("detector: threshold | no
        # ROI found"), and an unwrapped label reports a minimum width equal to
        # its whole string.
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet(style.qss_text(size_px=11))
        ctrl2.addWidget(self._status_label)

        _ctrl2_bar = QWidget()
        _ctrl2_bar.setLayout(ctrl2)
        _ctrl2_bar.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        root.addWidget(_ctrl2_bar)

        # ── Manual segment override ──────────────────────────────────────────
        # Same mechanism as wlc_view_window.py's View Fits window — off by
        # default so an ordinary click on the F-x panel never accidentally
        # sets an override; only active while armed.
        ctrl3 = FlowLayout(margin=0, h_spacing=6, v_spacing=4)
        self._manual_mode_btn = QPushButton("Manually Select Segment(s)")
        self._manual_mode_btn.setCheckable(True)
        self._manual_mode_btn.toggled.connect(self._on_manual_mode_toggled)
        ctrl3.addWidget(self._manual_mode_btn)

        self._select_primary_btn = QPushButton("Select Primary")
        self._select_primary_btn.setCheckable(True)
        self._select_primary_btn.setEnabled(False)
        self._select_secondary_btn = QPushButton("Select Secondary")
        self._select_secondary_btn.setCheckable(True)
        self._select_secondary_btn.setEnabled(False)
        self._select_group = QButtonGroup(self)
        self._select_group.setExclusive(True)
        self._select_group.addButton(self._select_primary_btn)
        self._select_group.addButton(self._select_secondary_btn)
        ctrl3.addWidget(self._select_primary_btn)
        ctrl3.addWidget(self._select_secondary_btn)

        self._manual_status_label = QLabel("")
        self._manual_status_label.setWordWrap(True)
        ctrl3.addWidget(self._manual_status_label)

        ctrl3.addWidget(SampleMarksToggle())

        _ctrl3_bar = QWidget()
        _ctrl3_bar.setLayout(ctrl3)
        _ctrl3_bar.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        root.addWidget(_ctrl3_bar)

        self._clickable_segments: list = []   # [(x_min, x_max, seg_idx), ...]

        # Graphics layout
        glw = pg.GraphicsLayoutWidget()
        glw.setBackground(style.SURFACE)
        root.addWidget(glw, stretch=1)

        _zero_pen = pg.mkPen(_COLOR_ZERO, width=1, style=Qt.PenStyle.DashLine)

        # ── Row 0: raw low channel (nm, baseline-subtracted) ─────────────────
        self._raw_plot = glw.addPlot(row=0, col=0)
        set_si_label(self._raw_plot, "left", "deflection", _quant.NM)
        self._raw_plot.showGrid(x=True, y=True, alpha=0.15)
        self._raw_curve = sample_marks.trace(
            self._raw_plot, color=style.DATA, alpha=_ALPHA_SIGNAL
        )
        self._raw_plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=_zero_pen))
        # Rows 0-2 are x-linked onto one piezo axis, which row 2 labels.  Only
        # a labelled axis SI-prefixes, so leaving these two drawing their own
        # values would show the same shared axis at two scales at once.  Same
        # treatment fft_window and decomposition_window already give their
        # stacked panels.
        self._raw_plot.getAxis("bottom").setStyle(showValues=False)

        # ── Row 1: mean_dev (nm) ─────────────────────────────────────────────
        self._mean_plot = glw.addPlot(row=1, col=0)
        # si=False: the onset-threshold line below is dragged on this axis
        # and typed into _spin_onset, which shows plain nm and cannot carry
        # an SI prefix.  A free axis could relabel itself pm while the box
        # beside it said nm, for the same number.
        set_si_label(self._mean_plot, "left", "mean dev", _quant.NM, si=False)
        self._mean_plot.showGrid(x=True, y=True, alpha=0.15)
        self._mean_plot.setXLink(self._raw_plot)
        self._mean_curve = sample_marks.trace(
            self._mean_plot, color=style.SIG_MEAN_DEV,
            width=style.W_SIGNAL, alpha=_ALPHA_SIGNAL,
        )
        self._mean_plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=_zero_pen))
        self._mean_plot.getAxis("bottom").setStyle(showValues=False)

        # Draggable horizontal threshold on mean_dev panel — onset threshold.
        self._thresh_line_onset = pg.InfiniteLine(
            pos=self._onset_threshold_nm, angle=0, movable=True,
            pen=pg.mkPen(_COLOR_ONSET, width=2, style=Qt.PenStyle.DashLine),
            hoverPen=pg.mkPen(_COLOR_ONSET, width=3),
            label='onset thresh',
            labelOpts={'position': 0.05, 'color': _COLOR_ONSET},
        )
        self._thresh_line_onset.sigPositionChangeFinished.connect(
            self._on_onset_line_moved
        )
        self._mean_plot.addItem(self._thresh_line_onset)

        # ── Row 2: d1 (nm deflection per nm piezo) ───────────────────────────
        self._d1_plot = glw.addPlot(row=2, col=0)
        # d¹ is a ratio (nm/nm): not an SI unit, never prefixed — and the two
        # threshold lines on it are typed into spin boxes as well.
        set_si_label(self._d1_plot, "left",   "d¹",    _quant.NM_PER_NM)
        set_si_label(self._d1_plot, "bottom", "piezo", _quant.NM)
        self._d1_plot.showGrid(x=True, y=True, alpha=0.15)
        self._d1_plot.setXLink(self._raw_plot)
        self._d1_curve = sample_marks.trace(
            self._d1_plot, color=style.SIG_D1,
            width=style.W_SIGNAL, alpha=_ALPHA_SIGNAL,
        )
        self._d1_plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=_zero_pen))

        # Draggable horizontal thresholds on d1 panel — outer (solid) validates
        # the ROI; inner (dotted) detects the smaller sub-events.
        self._thresh_line_d1 = pg.InfiniteLine(
            pos=self._threshold_nm_per_nm, angle=0, movable=True,
            pen=pg.mkPen(_COLOR_RUPTURE, width=2, style=Qt.PenStyle.DashLine),
            hoverPen=pg.mkPen(_COLOR_RUPTURE, width=3),
            label='d¹ outer',
            labelOpts={'position': 0.05, 'color': _COLOR_RUPTURE},
        )
        self._thresh_line_d1.sigPositionChangeFinished.connect(
            self._on_threshold_line_moved
        )
        self._d1_plot.addItem(self._thresh_line_d1)

        self._thresh_line_inner = pg.InfiniteLine(
            pos=self._inner_threshold_nm_per_nm, angle=0, movable=True,
            # Was the d1 signal's own colour (then red) — the same colour as the
            # curve it sits on top of, distinguished only by a thin dotted style.
            # Invisible in practice. Now matches _COLOR_RUPTURE_INNER, the
            # colour of the inner rupture markers it corresponds to.
            pen=pg.mkPen(_COLOR_RUPTURE_INNER, width=2, style=Qt.PenStyle.DotLine),
            hoverPen=pg.mkPen(_COLOR_RUPTURE_INNER, width=3),
            label='d¹ inner',
            labelOpts={'position': 0.12, 'color': _COLOR_RUPTURE_INNER},
        )
        self._thresh_line_inner.sigPositionChangeFinished.connect(
            self._on_inner_line_moved
        )
        self._d1_plot.addItem(self._thresh_line_inner)

        # ── Row 3: Force vs Extension — per-segment WLC fits ──────────────────
        # Different x-axis (extension, not piezo) so it is NOT x-linked to the
        # panels above.  In a multi mode it shows the retract in force-extension
        # space with each detected segment's WLC fit overlaid and each rupture's
        # force marked — the view for eyeballing the fits on real curves.
        self._fx_plot = glw.addPlot(row=3, col=0)
        set_si_label(self._fx_plot, "left",   "force",     _quant.PN)
        set_si_label(self._fx_plot, "bottom", "extension", _quant.NM)
        self._fx_plot.showGrid(x=True, y=True, alpha=0.15)
        self._fx_data = sample_marks.trace(
            self._fx_plot, color=style.DATA, alpha=_ALPHA_FX)
        self._fx_plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=_zero_pen))
        # Dynamically created (variable-count) fit lines + force markers.
        self._fx_items: list = []
        # _fx_plot shares glw's ONE scene with _raw_plot/_mean_plot/_d1_plot
        # (all pg.PlotItems from the same GraphicsLayoutWidget), so the click
        # handler must itself check the click landed in _fx_plot, not assume it.
        glw.scene().sigMouseClicked.connect(self._on_plot_clicked)

        # Per-panel overlay items (masks only — ROI/rupture markers are drawn
        # dynamically per-curve by _draw_multi, since their count varies).
        self._overlays: dict[str, list] = {"anchor": [], "postsnap": []}
        for plot in (self._raw_plot, self._mean_plot, self._d1_plot):
            anchor_region = pg.LinearRegionItem(
                values=(0.0, 0.0),
                brush=pg.mkBrush(_COLOR_MASK),
                pen=pg.mkPen(None),
                movable=False,
            )
            postsnap_region = pg.LinearRegionItem(
                values=(0.0, 0.0),
                brush=pg.mkBrush(_COLOR_MASK),
                pen=pg.mkPen(None),
                movable=False,
            )
            plot.addItem(anchor_region)
            plot.addItem(postsnap_region)
            self._overlays["anchor"].append(anchor_region)
            self._overlays["postsnap"].append(postsnap_region)

    # Opening the window does not write a profile. _sync_profile_owner seeds a
    # profile when a curve is first displayed.

    # ── Public API ────────────────────────────────────────────────────────────

    def set_results(self, results: list[dict]) -> None:
        """Called by analysis_runner to hand over the file list."""
        self._results = results

    def update_curve(self, index: int) -> None:
        """
        Load curve at results[index], compute all detection signals, redraw.
        Called by RawCurveWindow whenever it navigates to a new curve.
        """
        if not self._results or index < 0 or index >= len(self._results):
            return

        self._current_index = index
        path = self._results[index].get("path", "")
        self._set_file_label(path)
        self._sync_profile_owner(path)
        try:
            curve: ForceCurve = load_force_curve(path)
        except (LoadError, Exception):
            self._clear()
            return

        self._last_curve = curve
        self._last_path  = path
        self._recompute_and_draw(curve)

    # ── Worker navigation ─────────────────────────────────────────────────────

    def showEvent(self, event) -> None:
        # Catch up to the worker's current playhead when re-shown after being
        # hidden (the nav bar skips per-curve work while hidden).
        super().showEvent(event)
        if self._nav is not None:
            self._nav.sync_now()

    def _on_nav_curve_selected(self, path: str, file_id: int) -> None:
        """Worker moved the playhead — load and redraw the new curve."""
        self.set_results([{"path": path}])
        self.update_curve(0)

    def _on_nav_curve_cleared(self) -> None:
        """The displayed curve left the queue — show nothing rather than it."""
        self.set_results([])
        self._current_index = -1
        self._last_curve = None
        self._last_path  = None
        self._set_file_label(None)
        self._clear()

    def _recompute_and_draw(self, curve: ForceCurve) -> None:
        """
        Recompute detection signals, run the multi-event finder, refresh panels.

        Calls roi_pipeline.compute_curve_events_coords, the same function used
        by the batch worker and backfill, so all callers share one
        decompose→baseline→detect→snap-off→rupture→segment→fit implementation.

        The ROI-specific fields below (window/thresholds/detector) come from
        this window's committed state. Spin boxes call this only after
        editingFinished and draggable lines only after release, because this
        computation persists its event_map result and transient edit values
        must never replace the catalog's current document.
        Only anchor_nm/cutoff_hz/trim_pts/var_window_ms/thresh_retr/invOLS
        window are read from DB settings, since this window has no controls for
        them; file_id/db_path/code_ver let compute_curve_events_coords reuse
        this file's already-cached offset/snap-off-index/invOLS for those
        instead of recomputing them on every completed edit.
        """
        try:
            # These seven have no control in this window, so they come from a
            # stored profile — and it MUST be the profile of the curve on
            # screen (self._experimentalist, kept in step by
            # _sync_profile_owner), not db.get_param's queue owner.  Reading
            # these from the queue owner builds one EventParams out of two
            # people — the knobs above from the curve's owner, these from
            # whoever is queued — and that combination silently decides where
            # the search band goes.
            # ONE read of THE parameter set in force — the file at position
            # one of the analysis queue decides it (db.active_param_owner),
            # exactly as it does for the worker and for every other window.
            ps = _db.load_analysis_params(self._db_path)
            from dataclasses import replace
            committed = replace(
                ps,
                roi_window_pts=self._window_pts,
                roi_threshold_nm_per_nm=self._threshold_nm_per_nm,
                roi_inner_threshold_nm_per_nm=self._inner_threshold_nm_per_nm,
                roi_post_snapoff_mask_nm=self._post_snapoff_mask_nm,
                roi_onset_threshold_nm=self._onset_threshold_nm,
                roi_detector_mode_idx=MODE_TO_STORED_IDX[self._detector_mode],
                roi_prominence=self._prominence,
                roi_min_distance_pts=self._distance_pts,
            )
            ep = event_params_from(committed, detector=self._detector_mode)

            file_id = _db.get_file_id(self._last_path, self._db_path) if self._last_path else None
            res = compute_curve_events_coords(
                curve, ep,
                db_path=self._db_path, code_ver=cache_version(), file_id=file_id,
                param_set=committed,
            )
        except Exception as exc:
            import sys
            print(
                f"[roi] computation failed — {type(exc).__name__}: {str(exc)[:160]}",
                file=sys.stderr,
            )
            self._clear()
            self._status_label.setText(
                f"ROI calculation failed: {type(exc).__name__}: {str(exc)[:120]}"
            )
            return

        sigs, events, snapoff_idx = res.sigs, res.events, res.snapoff_idx

        px = sigs.piezo
        self._raw_curve.setData(x=px.tolist(),  y=sigs.low.tolist())
        self._mean_curve.setData(x=px.tolist(), y=sigs.mean_dev.tolist())
        self._d1_curve.setData(x=px.tolist(),   y=sigs.d1.tolist())

        # Keep draggable threshold lines in sync with stored values.
        self._thresh_line_d1.blockSignals(True)
        self._thresh_line_d1.setValue(self._threshold_nm_per_nm)
        self._thresh_line_d1.blockSignals(False)
        self._thresh_line_inner.blockSignals(True)
        self._thresh_line_inner.setValue(self._inner_threshold_nm_per_nm)
        self._thresh_line_inner.blockSignals(False)
        self._thresh_line_onset.blockSignals(True)
        self._thresh_line_onset.setValue(self._onset_threshold_nm)
        self._thresh_line_onset.blockSignals(False)

        # Masks (in piezo-space).  Anchor covers piezo[mask_anchor_idx : ];
        # post-snap covers piezo[snapoff_idx : mask_postsnap_idx].
        n = len(px)
        anchor_lo = float(px[res.mask_anchor_idx])   if 0 <= res.mask_anchor_idx   < n else float(px[-1])
        anchor_hi = float(px[-1])
        snap_lo   = float(px[snapoff_idx])           if 0 <= snapoff_idx          < n else float(px[0])
        snap_hi   = float(px[res.mask_postsnap_idx - 1]) if 0 < res.mask_postsnap_idx <= n else snap_lo

        for region in self._overlays["anchor"]:
            region.setRegion((min(anchor_lo, anchor_hi), max(anchor_lo, anchor_hi)))
        for region in self._overlays["postsnap"]:
            region.setRegion((min(snap_lo, snap_hi), max(snap_lo, snap_hi)))

        # events already carries every segment's
        # WLC fit — compute_curve_events_coords applied both internally — so
        # there's nothing left to compute here, only to draw.
        self._draw_multi(events)
        self._draw_fx(
            events, curve, res.dc.low_retr, res.offset, res.invols, res.snap_piezo,
        )
        self._status_label.setText(self._multi_status(events))

    def _clear(self) -> None:
        """Clear all curves on load error."""
        for c in (self._raw_curve, self._mean_curve, self._d1_curve):
            c.setData(x=[], y=[])
        self._clear_multi()
        self._clear_fx()
        self._fx_data.setData(x=[], y=[])
        self._status_label.setText("")

    # ── Multi-event marker drawing ─────────────────────────────────────────────

    def _clear_multi(self) -> None:
        """Remove all dynamically-created multi-event marker lines."""
        for plot, item in self._multi_items:
            plot.removeItem(item)
        self._multi_items = []

    def _draw_multi(self, events) -> None:
        """
        Draw every ROI boundary (onset solid / return dotted, orange) and every
        rupture across all three panels.  Variable count, so the lines are
        created fresh each redraw and tracked for removal.

        roi.ruptures is ascending surface -> baseline (see the ROI docstring in
        roi_events.py), so ruptures[-1] is the one that crossed the OUTER/
        TERMINAL threshold — the one that made this an ROI at all — and every
        earlier entry is an INNER sub-event found only because it sits inside
        an already-open junction. These are drawn in two different colours so
        a single ROI with an inner+outer rupture pair can never be mistaken for
        two separate outer ROIs sitting side by side.
        """
        self._clear_multi()
        plots = (self._raw_plot, self._mean_plot, self._d1_plot)
        for roi in events.rois:
            n = len(roi.ruptures)
            for plot in plots:
                onset_line = pg.InfiniteLine(
                    pos=roi.onset_piezo_nm, angle=90,
                    pen=pg.mkPen(_COLOR_ONSET, width=style.W_GUIDE, style=Qt.PenStyle.DashLine),
                )
                return_line = pg.InfiniteLine(
                    pos=roi.return_piezo_nm, angle=90,
                    pen=pg.mkPen(_COLOR_ONSET, width=style.W_GUIDE, style=Qt.PenStyle.DotLine),
                )
                plot.addItem(onset_line);  self._multi_items.append((plot, onset_line))
                plot.addItem(return_line); self._multi_items.append((plot, return_line))
                for i, rup in enumerate(roi.ruptures):
                    is_terminal = (i == n - 1)
                    color = _COLOR_RUPTURE if is_terminal else _COLOR_RUPTURE_INNER
                    rup_line = pg.InfiniteLine(
                        pos=rup.piezo_nm, angle=90,
                        pen=pg.mkPen(color, width=style.W_GUIDE, style=Qt.PenStyle.DashLine),
                    )
                    plot.addItem(rup_line); self._multi_items.append((plot, rup_line))

    # ── Force-extension fit panel ──────────────────────────────────────────────

    def _clear_fx(self) -> None:
        """Remove all dynamically-created F-x fit lines and force markers."""
        for item in self._fx_items:
            self._fx_plot.removeItem(item)
        self._fx_items = []
        self._clickable_segments = []
        self._manual_status_label.setText("")

    def _draw_fx(self, events, curve, low_retr, offset: float, inv: float,
                 snap_piezo: float) -> None:
        """
        Draw the low-frequency retract in force-extension space over the event
        region, overlay each fitted segment's WLC curve, and mark each rupture's
        force. Extension is zeroed at snap-off. Data, model, markers and residual
        coordinates therefore match fit_segments exactly.
        """
        self._clear_fx()
        self._last_events = events
        if not events.rois:
            self._fx_data.setData(x=[], y=[])
            return
        k         = curve.spring_constant
        # These are the exact coordinates supplied to fit_segments.  Using the
        # raw channel here would compare the low-pass fit with different data.
        defl_corr = (np.asarray(low_retr, dtype=float) - offset) / inv
        force     = k * defl_corr
        ext       = (curve.piezo_retr - snap_piezo) - defl_corr

        lo = min(r.onset_idx  for r in events.rois)
        hi = max(r.return_idx for r in events.rois)
        self._fx_data.setData(x=ext[lo:hi + 1].tolist(), y=force[lo:hi + 1].tolist())

        # Primary/Secondary only ever address the right-most outer ROI
        # with ruptures — same scope Ultimate/Penultimate and the 2DH windows
        # already use (see roi_pipeline.segment_summary_bulk).
        target_roi = next((r for r in reversed(events.rois) if r.ruptures), None)
        file_id = _db.get_file_id(self._last_path, self._db_path) if self._last_path else None
        primary_idx = secondary_idx = None
        if target_roi is not None and file_id is not None:
            override = _db.get_segment_override(file_id, self._db_path)
            current_params = _db.get_latest_event_map_params(file_id, self._db_path)
            override_state = resolve_segment_override_state(
                override, current_params, len(target_roi.segments),
                event_geometry_identity(events))
            primary_idx = override_state.primary_idx
            secondary_idx = override_state.secondary_idx
            review = "   (stored choice needs review)" if override_state.status == "needs_review" else ""
        else:
            review = ""
        self._manual_status_label.setText(
            f"Primary: {'seg ' + str(primary_idx) if primary_idx is not None else 'none'}   "
            f"Secondary: {'seg ' + str(secondary_idx) if secondary_idx is not None else 'none'}"
            f"{review}"
        )

        n_rois = len(events.rois)
        for ri, roi in enumerate(events.rois):
            n_segs = len(roi.segments)
            for si, seg in enumerate(roi.segments):
                if seg.l_p_nm is None or seg.l_c_nm is None:
                    continue
                # Draw the WLC model over exactly the fitted window (reload/onset
                # bottom → force peak), not the whole d1-bounded segment.
                a = seg.fit_lo_idx if seg.fit_lo_idx is not None else seg.left_idx
                b = seg.fit_hi_idx if seg.fit_hi_idx is not None else seg.right_idx
                xs = ext[a:b + 1]
                xs = xs[xs > 0]
                if xs.size < 2:
                    continue
                xs = np.sort(xs)
                ys = np.asarray(wlc(xs, seg.l_p_nm, seg.l_c_nm))
                col = style.roi_segment_qcolor(ri, n_rois, si, n_segs,
                                               alpha=style.A_MODEL)
                item = self._fx_plot.plot(
                    x=xs.tolist(), y=ys.tolist(),
                    pen=pg.mkPen(col, width=style.W_MODEL),
                )
                self._fx_items.append(item)
                if roi is target_roi:
                    self._clickable_segments.append(
                        (float(xs.min()), float(xs.max()), si))
                    tag = "P" if si == primary_idx else ("S" if si == secondary_idx else None)
                    if tag is not None:
                        mid = len(xs) // 2
                        label = pg.TextItem(tag, color=style.INK, anchor=(0.5, 1.2))
                        label.setPos(float(xs[mid]), float(ys[mid]))
                        self._fx_plot.addItem(label)
                        self._fx_items.append(label)
            n_rup = len(roi.ruptures)
            # On the F-x panel the markers carry the ROI's own hue (matching the
            # WLC window), and terminal-vs-inner is carried by SYMBOL instead of
            # by colour: green/magenta sits only 3.2 dE from the orange ROI hue
            # under protanopia.  Shape is a cleaner channel for a two-way
            # distinction than a hue that has to dodge the fits.
            mark_brush = pg.mkBrush(
                style.roi_segment_qcolor(ri, n_rois, 0, 1, alpha=235))
            for i, rup in enumerate(roi.ruptures):
                if rup.force_pN is None:
                    continue
                # Marker on the force PEAK (physical rupture point), not the d1 idx.
                mi = rup.force_idx if rup.force_idx is not None else rup.idx
                marker = pg.ScatterPlotItem(
                    x=[float(ext[mi])], y=[float(rup.force_pN)],
                    size=style.MARKER_SIZE,
                    symbol="o" if i == n_rup - 1 else "d",
                    brush=mark_brush, pen=style.MARKER_PEN,
                )
                self._fx_plot.addItem(marker)
                self._fx_items.append(marker)

    # ── Manual segment override ──────────────────────────────────────────────

    def _on_manual_mode_toggled(self, checked: bool) -> None:
        self._select_primary_btn.setEnabled(checked)
        self._select_secondary_btn.setEnabled(checked)
        if not checked:
            self._select_primary_btn.setChecked(False)
            self._select_secondary_btn.setChecked(False)

    def _on_plot_clicked(self, ev) -> None:
        """Commit a click on the F-x panel as the armed role's segment pick
        A no-op unless "Manually Select Segment(s)" is on and one of
        Select Primary/Select Secondary is armed, and the click actually
        landed in the F-x panel (it shares glw's one scene with the other
        three panels above it)."""
        if not self._manual_mode_btn.isChecked():
            return
        armed = "primary" if self._select_primary_btn.isChecked() else (
            "secondary" if self._select_secondary_btn.isChecked() else None)
        if armed is None or not self._clickable_segments or not self._last_path:
            return
        if not self._fx_plot.sceneBoundingRect().contains(ev.scenePos()):
            return
        view_pos = self._fx_plot.vb.mapSceneToView(ev.scenePos())
        x = view_pos.x()
        seg_idx = next(
            (si for (x_lo, x_hi, si) in self._clickable_segments if x_lo <= x <= x_hi),
            None,
        )
        if seg_idx is None:
            return
        file_id = _db.get_file_id(self._last_path, self._db_path)
        if file_id is None:
            return
        if self._last_events is None:
            return
        params_json = event_geometry_identity(self._last_events)
        if armed == "primary":
            _db.set_primary_segment_idx(file_id, seg_idx, params_json, self._db_path)
        else:
            _db.set_secondary_segment_idx(file_id, seg_idx, params_json, self._db_path)
        self._select_primary_btn.setChecked(False)
        self._select_secondary_btn.setChecked(False)
        self.update_curve(self._current_index)

    def _multi_status(self, events) -> str:
        """One-line summary of the multi-event result for the status label."""
        if not events.rois:
            return f"detector: {self._detector_mode}   |   no ROI found"
        parts = []
        for i, roi in enumerate(events.rois):
            fs = "/".join(_quant.format_value('seg_force_pN', r.force_pN)
                          for r in roi.ruptures if r.force_pN is not None) or "—"
            dx = "/".join(_quant.format_value('seg_dX_ext_nm', v)
                          for v in roi.dX_ext_pairs if v is not None)
            dxs = f"  ΔXext={dx} nm" if dx else ""
            parts.append(f"ROI{i}[{roi.ordering}] F={fs} pN{dxs}")
        return (f"detector: {self._detector_mode}   |   "
                f"{events.n_rois} ROI(s)   |   " + "   ".join(parts))

    # ── User-profile persistence ──────────────────────────────────────────────
    #
    # The profile key FOLLOWS THE CURVE ON SCREEN: update_curve resolves the
    # owner (experimentalist of the file's watched directory) and, when it
    # changes, loads that user's stored knobs into the controls + settings
    # table.  A key frozen at construction breaks in worker mode, where one
    # queue interleaves several users' files, so one person's edits land under
    # another's key.

    def _sync_profile_owner(self, path: str | None) -> None:
        """Re-key the profile to THE parameter set in force, and load its
        knobs if that has changed.

        Whose set that is has ONE answer: the experimentalist on the file at
        position one of the analysis queue (db.active_param_owner).  Clear the
        queue, add to it, delete from it, restart the app — recheck.  Between
        those, the answer is fixed, and the curve currently on screen does not
        enter into it.

        The active queue owner governs both the displayed controls and the
        parameters without controls, preventing mixed-profile computations.
        """
        try:
            owner = _db.active_param_owner(self._db_path)
        except Exception:
            return
        if self._owner_synced and owner == self._experimentalist:
            return
        self._owner_synced = True
        self._experimentalist = owner
        self._params_label.setText(f"Parameters: {owner}")
        # ONE fetch of ONE profile, always complete — unset keys come back as
        # the single code default, so there is no "did this person have a
        # profile" branch and no borrowing another bucket's values.
        profile = _db.load_analysis_params(self._db_path)
        if profile:
            self._apply_profile(profile)
        # Seed/backfill: merge-write this window's knobs under the owner's key
        # so the profile is complete from first sighting — values just loaded
        # from their profile, or carried over for keys it did not have yet
        # (new user, or a profile written before a knob existed).  The merge
        # never touches other windows' keys, and applying-then-saving cannot
        # put another user's values under this key because the key IS the
        # on-screen curve's owner.
        self._save_user_profile()

    def _apply_profile(self, p: dict) -> None:
        """
        Load a user's stored ROI knobs into fields, widgets, and the settings
        table.  Widget signals are blocked — this is a load, not an edit, so
        it must not re-save the profile; the caller redraws once afterwards.
        """
        def _f(key: str, cur: float) -> float:
            try:
                return float(p[key])
            except (KeyError, TypeError, ValueError):
                return float(cur)

        self._window_pts                = int(_f("roi_window_pts", self._window_pts))
        self._threshold_nm_per_nm       = _f("roi_threshold_nm_per_nm", self._threshold_nm_per_nm)
        self._inner_threshold_nm_per_nm = _f("roi_inner_threshold_nm_per_nm",
                                             self._inner_threshold_nm_per_nm)
        self._post_snapoff_mask_nm      = _f("roi_post_snapoff_mask_nm", self._post_snapoff_mask_nm)
        self._onset_threshold_nm        = _f("roi_onset_threshold_nm", self._onset_threshold_nm)
        # Decode via DETECTOR_BY_IDX (the on-disk scheme shared with the
        # worker), not by _DETECTOR_MODES list position — those are not the
        # same numbering; "single (legacy)" is not in that list.
        cur_stored = MODE_TO_STORED_IDX.get(self._detector_mode, 1)
        raw_idx = int(_f("roi_detector_mode_idx", cur_stored))
        self._detector_mode = DETECTOR_BY_IDX.get(raw_idx, "threshold")
        idx = next((i for i, (_, m) in enumerate(self._DETECTOR_MODES)
                    if m == self._detector_mode), 0)
        self._prominence    = _f("roi_prominence", self._prominence)
        self._distance_pts  = int(_f("roi_min_distance_pts", self._distance_pts))

        for w, val in (
            (self._spin_window,    self._window_pts),
            (self._spin_threshold, self._threshold_nm_per_nm),
            (self._spin_inner,     self._inner_threshold_nm_per_nm),
            (self._spin_mask,      self._post_snapoff_mask_nm),
            (self._spin_onset,     self._onset_threshold_nm),
            (self._spin_prom,      self._prominence),
            (self._spin_dist,      self._distance_pts),
        ):
            w.blockSignals(True)
            w.setValue(val)
            w.blockSignals(False)
        self._combo_detector.blockSignals(True)
        self._combo_detector.setCurrentIndex(idx)
        self._combo_detector.blockSignals(False)

        # No mirror into the `settings` table. The parameter set lives in
        # exactly one place - the queue owner's profile - and the pipeline
        # reads it there (db.get_param). Copying it into a catalog-wide table
        # makes that table mean "whoever was displayed last", and every other
        # experimentalist inherits it.

    def _save_user_profile(self) -> None:
        """
        Merge the current ROI params into experimentalist_profiles[experimentalist].
        Uses db.merge_experimentalist_profile (a single atomic SQL statement)
        rather than read-then-write, so a concurrent save from another window
        or the analysis worker's QThread cannot be silently discarded.
        """
        # Save to THE set in force — the same one answer the read side uses
        # (db.active_param_owner), asked again here rather than trusting
        # self._experimentalist, which is None until the first curve triggers
        # _sync_profile_owner.
        key = _db.active_param_owner(self._db_path)
        _db.merge_experimentalist_profile(key, {
            "roi_window_pts":                float(self._window_pts),
            "roi_threshold_nm_per_nm":       float(self._threshold_nm_per_nm),
            "roi_inner_threshold_nm_per_nm": float(self._inner_threshold_nm_per_nm),
            "roi_post_snapoff_mask_nm":      float(self._post_snapoff_mask_nm),
            "roi_onset_threshold_nm":        float(self._onset_threshold_nm),
            "roi_detector_mode_idx":         float(
                MODE_TO_STORED_IDX.get(self._detector_mode, 1)),
            "roi_prominence":                float(self._prominence),
            "roi_min_distance_pts":          float(self._distance_pts),
        }, self._db_path)

    # ── Controls ──────────────────────────────────────────────────────────────
    #
    # Every spinbox-driven parameter here is a GLOBAL setting — it changes what
    # every curve in the catalog gets analysed under, not just the one on
    # screen. Spin-box changes commit and recalculate once on editingFinished;
    # they deliberately do no live preview while typing.
    #
    # A draggable line's release (sigPositionChangeFinished) already fires only
    # once, so it previews and commits together — the release IS "done".

    def _preview_window(self, value: int) -> None:
        self._window_pts = value
        self._spin_window.setValue(value)

    def _commit_window(self) -> None:
        self._window_pts = int(self._spin_window.value())
        _db.update_analysis_param('roi_window_pts', self._window_pts, self._db_path)
        self._save_user_profile()
        self._recompute_current()

    def _preview_threshold(self, value: float) -> None:
        self._threshold_nm_per_nm = float(value)
        self._spin_threshold.setValue(value)

    def _commit_threshold(self) -> None:
        self._threshold_nm_per_nm = float(self._spin_threshold.value())
        if self._inner_threshold_nm_per_nm > self._threshold_nm_per_nm:
            self._inner_threshold_nm_per_nm = self._threshold_nm_per_nm
            self._spin_inner.setValue(self._inner_threshold_nm_per_nm)
        _db.update_analysis_param('roi_threshold_nm_per_nm', float(self._threshold_nm_per_nm), self._db_path)
        _db.update_analysis_param('roi_inner_threshold_nm_per_nm',
                                  float(self._inner_threshold_nm_per_nm), self._db_path)
        self._save_user_profile()
        self._recompute_current()

    def _preview_inner_threshold(self, value: float) -> None:
        self._inner_threshold_nm_per_nm = min(float(value), self._threshold_nm_per_nm)
        self._spin_inner.setValue(self._inner_threshold_nm_per_nm)

    def _commit_inner_threshold(self) -> None:
        self._inner_threshold_nm_per_nm = min(
            float(self._spin_inner.value()), self._threshold_nm_per_nm,
        )
        self._spin_inner.setValue(self._inner_threshold_nm_per_nm)
        _db.update_analysis_param('roi_inner_threshold_nm_per_nm',
                        float(self._inner_threshold_nm_per_nm), self._db_path)
        self._save_user_profile()
        self._recompute_current()

    def _on_inner_line_moved(self) -> None:
        """Commit the quantized inner threshold once after its drag ends."""
        new_val = float(self._thresh_line_inner.value())
        new_val = max(self._spin_inner.minimum(), min(self._spin_inner.maximum(), new_val))
        # Quantize the raw mouse position before display or persistence so the
        # spin box and database hold the same value.
        new_val = _quant.quantize("roi_inner_threshold_nm_per_nm", new_val)
        self._spin_inner.blockSignals(True)
        self._spin_inner.setValue(new_val)
        self._spin_inner.blockSignals(False)
        self._preview_inner_threshold(new_val)
        self._commit_inner_threshold()

    def _preview_mask(self, value: float) -> None:
        self._post_snapoff_mask_nm = float(value)
        self._spin_mask.setValue(value)

    def _commit_mask(self) -> None:
        self._post_snapoff_mask_nm = float(self._spin_mask.value())
        _db.update_analysis_param('roi_post_snapoff_mask_nm', float(self._post_snapoff_mask_nm), self._db_path)
        self._save_user_profile()
        self._recompute_current()

    def _preview_onset(self, value: float) -> None:
        self._onset_threshold_nm = float(value)
        self._spin_onset.setValue(value)

    def _commit_onset(self) -> None:
        self._onset_threshold_nm = float(self._spin_onset.value())
        _db.update_analysis_param('roi_onset_threshold_nm', float(self._onset_threshold_nm), self._db_path)
        self._save_user_profile()
        self._recompute_current()

    def _on_detector_changed(self, index: int) -> None:
        # A combo selection is already one deliberate, complete action (unlike
        # a spinbox mid-typing) — no separate preview/commit split needed.
        if 0 <= index < len(self._DETECTOR_MODES):
            mode = self._DETECTOR_MODES[index][1]
            self._detector_mode = mode
            # Persist via the canonical index (roi_pipeline.MODE_TO_STORED_IDX),
            # NOT the combo's raw position — the on-disk numbering has to match
            # what the worker's DETECTOR_BY_IDX decodes, and the two differ:
            # "single (legacy)" is not in this list. set_setting stores REALs,
            # not strings.
            _db.update_analysis_param('roi_detector_mode_idx',
                            float(MODE_TO_STORED_IDX[mode]), self._db_path)
            self._save_user_profile()
            if self._last_curve is not None:
                self._recompute_and_draw(self._last_curve)

    def _preview_prominence(self, value: float) -> None:
        self._prominence = float(value)
        self._spin_prom.setValue(value)

    def _commit_prominence(self) -> None:
        self._prominence = float(self._spin_prom.value())
        _db.update_analysis_param('roi_prominence', float(self._prominence), self._db_path)
        self._save_user_profile()
        if self._detector_mode == "find_peaks":
            self._recompute_current()

    def _preview_distance(self, value: int) -> None:
        self._distance_pts = int(value)
        self._spin_dist.setValue(value)

    def _commit_distance(self) -> None:
        self._distance_pts = int(self._spin_dist.value())
        _db.update_analysis_param('roi_min_distance_pts', int(self._distance_pts), self._db_path)
        self._save_user_profile()
        if self._detector_mode == "find_peaks":
            self._recompute_current()

    def _recompute_current(self) -> None:
        """Recalculate and persist once after a completed parameter edit."""
        if self._last_curve is not None:
            self._recompute_and_draw(self._last_curve)

    def _on_threshold_line_moved(self) -> None:
        """Commit the quantized outer threshold once after its drag ends."""
        new_val = float(self._thresh_line_d1.value())
        new_val = max(self._spin_threshold.minimum(), min(self._spin_threshold.maximum(), new_val))
        # Quantize the raw mouse position before display or persistence so the
        # spin box and database hold the same value.
        new_val = _quant.quantize("roi_threshold_nm_per_nm", new_val)
        self._spin_threshold.blockSignals(True)
        self._spin_threshold.setValue(new_val)
        self._spin_threshold.blockSignals(False)
        self._preview_threshold(new_val)
        self._commit_threshold()

    def _on_onset_line_moved(self) -> None:
        """Commit the quantized onset threshold once after its drag ends."""
        new_val = float(self._thresh_line_onset.value())
        new_val = max(self._spin_onset.minimum(), min(self._spin_onset.maximum(), new_val))
        # Quantize the raw mouse position before display or persistence so the
        # spin box and database hold the same value.
        new_val = _quant.quantize("roi_onset_threshold_nm", new_val)
        self._spin_onset.blockSignals(True)
        self._spin_onset.setValue(new_val)
        self._spin_onset.blockSignals(False)
        self._preview_onset(new_val)
        self._commit_onset()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_file_label(self, path: str | None) -> None:
        """Show the current curve's directory + filename in the header."""
        if not path:
            self._file_label.setText("(no curve)")
            return
        p = Path(path)
        self._file_label.setText(f"{p.parent}/  {p.name}")
