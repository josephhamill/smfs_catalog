# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/wlc_view_window.py
#
# WlcViewWindow — WLC fit navigator for confirmed events.
#   left   : raw retract, retained as acquisition context
#   top    : low-frequency force vs extension (fit data + WLC model overlay)
#   bottom : low-frequency residuals (fit data − model) vs extension
#
# Accepts an ordered list of event paths and navigates through them with
# prev/next buttons, autoplay, and a scrollable track list.
# The caller supplies the population; this window does not query one
# independently.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from . import db as _db
from . import export_utils as _export
from .curve_loader import LoadError, load_force_curve
from .models import wlc
from . import style
from .qt_utils import _make_session_header, set_si_label, fit_on_screen
from . import quantities as _quant
# This window drives its OWN QTimer over its own curve list — it is not the
# analysis worker's playhead, so it shares navigator_bar's slider maths but
# deliberately not its NavigatorBar.
from .navigator_bar import (
    slider_to_interval_ms, rate_to_slider, rate_label,
    SLIDER_MIN, SLIDER_MAX, DEFAULT_RATE_HZ,
)

# Every colour and weight here comes from style.py — see its header for the
# three rules (data neutral / model bold + translucent / guide bold-dashed).
_PEN_DATA  = style.data_pen()
_PEN_RESID = style.data_pen(style.SERIES_LINE[2])
_PEN_ZERO  = style.hair_pen()
_PEN_RAW   = style.data_pen(style.SERIES_LINE[0], width=1.2)
# Selection span over the raw retract.
_ROI_BRUSH = pg.mkBrush(*style._COLOR_ROI_FILL_RGBA)


class WlcViewWindow(QMainWindow):
    """
    Two-panel WLC fit navigator.

    event_paths : ordered list of confirmed event file paths (from the user's selection)
    index     : which path to show first
    """

    def __init__(
        self,
        event_paths:    list[str],
        index:        int,
        db_path:      str,
        session_info: dict | None = None,
        two_dh_win:   object | None = None,
        raw_window:   object | None = None,
    ) -> None:
        super().__init__()
        self.setWindowFlag(Qt.WindowType.Window)
        self.setWindowTitle("SMFS — WLC fit")
        fit_on_screen(self, 1300, 700)
        self._event_paths    = event_paths
        self._index        = index
        self._db_path      = db_path
        self._session_info = session_info
        self._2dh_win = two_dh_win
        self._raw_win      = raw_window
        self._auto_dir     = 1   # +1 = forward, -1 = backward

        self._nav_timer = QTimer(self)
        self._nav_timer.timeout.connect(self._auto_step)

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

        # ── Navigation row ────────────────────────────────────────────────────
        nav = QHBoxLayout()

        self._btn_auto_rev = QPushButton("◀◀")
        self._btn_auto_rev.setCheckable(True)
        self._btn_auto_rev.setFixedWidth(44)
        self._btn_auto_rev.clicked.connect(
            lambda checked: self._toggle_auto(-1, checked)
        )

        self._prev_btn = QPushButton("◀ Prev")
        self._next_btn = QPushButton("Next ▶")
        self._prev_btn.clicked.connect(self._go_prev)
        self._next_btn.clicked.connect(self._go_next)

        self._btn_auto_fwd = QPushButton("▶▶")
        self._btn_auto_fwd.setCheckable(True)
        self._btn_auto_fwd.setFixedWidth(44)
        self._btn_auto_fwd.clicked.connect(
            lambda checked: self._toggle_auto(1, checked)
        )

        font = self._prev_btn.font()
        font.setPointSize(style.FONT_SMALL_PT)

        lbl_slow = QLabel("Slow")
        lbl_slow.setFont(font)
        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(SLIDER_MIN, SLIDER_MAX)
        self._speed_slider.setValue(rate_to_slider(DEFAULT_RATE_HZ))
        self._speed_slider.setFixedWidth(100)
        self._speed_slider.valueChanged.connect(self._on_speed_change)
        lbl_fast = QLabel("Fast")
        lbl_fast.setFont(font)
        self._speed_label = QLabel(rate_label(self._speed_slider.value()))
        self._speed_label.setFixedWidth(64)
        self._speed_label.setFont(font)

        self._counter = QLabel()
        self._counter.setFont(font)
        self._fname_label = QLabel()
        self._fname_label.setFont(font)

        nav.addWidget(self._btn_auto_rev)
        nav.addWidget(self._prev_btn)
        nav.addWidget(self._next_btn)
        nav.addWidget(self._btn_auto_fwd)
        nav.addSpacing(12)
        nav.addWidget(lbl_slow)
        nav.addWidget(self._speed_slider)
        nav.addWidget(lbl_fast)
        nav.addWidget(self._speed_label)
        nav.addSpacing(16)
        nav.addWidget(self._counter)
        nav.addSpacing(16)
        nav.addWidget(self._fname_label)
        nav.addStretch()
        # Cross-window jumps for this curve: Raw (the scan window), ROI (the
        # detection window), Dashboard.  Raw/ROI need a linked scan window; the
        # dashboard is a singleton found among the top-level windows.
        nav.addWidget(QLabel("Go to:"))
        # Raw/ROI route through the dashboard, which opens or reuses the
        # singleton viewer. The directly linked raw window remains a fallback.
        self._go_raw_btn = QPushButton("Raw")
        self._go_raw_btn.setToolTip("Pause analysis and open this curve in the scan (raw) window")
        self._go_raw_btn.clicked.connect(self._on_go_to_scan)
        nav.addWidget(self._go_raw_btn)

        self._go_roi_btn = QPushButton("ROI")
        self._go_roi_btn.setToolTip("Open the ROI detection window on this curve")
        self._go_roi_btn.clicked.connect(self._on_go_to_roi)
        nav.addWidget(self._go_roi_btn)

        self._go_dash_btn = QPushButton("Dashboard")
        self._go_dash_btn.setToolTip("Bring the dashboard window to the front")
        self._go_dash_btn.clicked.connect(self._on_go_to_dashboard)
        nav.addWidget(self._go_dash_btn)

        nav.addStretch()
        self._export_btn = QPushButton("Export fits…")
        self._export_btn.setToolTip(
            "Write the per-segment WLC fits for EVERY curve in this window's "
            "population (not just the one on screen) to the export folder, "
            "with a manifest."
        )
        self._export_btn.clicked.connect(self._on_export)
        nav.addWidget(self._export_btn)
        root.addLayout(nav)

        # ── Manual segment override ──────────────────────────────────────────
        # Off by default — click-to-pick must be deliberately armed so an
        # ordinary click while browsing fits never accidentally sets an
        # override. "Manually Select Segment(s)" gates whether the two arm
        # buttons do anything at all; each arm button, once checked, makes
        # the NEXT click on a segment's fit line commit that segment as this
        # curve's Primary/Secondary (an absolute per-curve override — see
        # roi_pipeline.segment_summary_bulk) and then disarms itself.
        manual_row = QHBoxLayout()
        self._manual_mode_btn = QPushButton("Manually Select Segment(s)")
        self._manual_mode_btn.setCheckable(True)
        self._manual_mode_btn.toggled.connect(self._on_manual_mode_toggled)
        manual_row.addWidget(self._manual_mode_btn)

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
        manual_row.addWidget(self._select_primary_btn)
        manual_row.addWidget(self._select_secondary_btn)

        self._manual_status_label = QLabel("")
        manual_row.addWidget(self._manual_status_label)
        manual_row.addStretch()

        # Parameter-variation envelope from the stored marginal standard
        # errors. It is descriptive because the stored result has no covariance.
        self._ci_chk = QCheckBox("Show fit uncertainty envelope")
        self._ci_chk.setChecked(True)
        self._ci_chk.setToolTip(
            f"Evaluate the WLC at the ±1σ corners of the stored {style.L_P} and "
            f"{style.L_C} marginal uncertainties. This descriptive envelope "
            "does not include parameter covariance."
        )
        self._ci_chk.toggled.connect(lambda _checked: self._show_current())
        manual_row.addWidget(self._ci_chk)
        root.addLayout(manual_row)

        self._armed_role: str | None = None          # 'primary' | 'secondary' | None
        self._clickable_segments: list = []           # [(x_min, x_max, seg_idx), ...]
        self._current_file_id: int | None = None

        # ── Total-2DH inclusion status ────────────────────────────────────────
        self._2dh_label = QLabel("")
        self._2dh_label.setFont(font)
        root.addWidget(self._2dh_label)

        # ── Main area: raw retract | ROI plots | track list ──────────────────
        main_hsplit = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(main_hsplit, stretch=1)

        # ── Left: full retract in raw units ───────────────────────────────────
        self._raw_plot = pg.PlotWidget()
        set_si_label(self._raw_plot, "left",   "Deflection", _quant.NM)
        set_si_label(self._raw_plot, "bottom", "Piezo",      _quant.NM)
        # (axis labels are HTML — see style.py § K)
        self._raw_plot.showGrid(x=True, y=True, alpha=0.2)

        self._raw_curve  = self._raw_plot.plot(pen=_PEN_RAW)
        self._raw_region = pg.LinearRegionItem(
            values=[0, 1], movable=False,
            brush=_ROI_BRUSH, pen=pg.mkPen(None),
        )
        self._raw_region.hide()
        self._raw_plot.addItem(self._raw_region)

        main_hsplit.addWidget(self._raw_plot)

        vsplit = QSplitter(Qt.Orientation.Vertical)

        # ── Top: force vs extension ───────────────────────────────────────────
        self._top = pg.PlotWidget()
        set_si_label(self._top, "left",   f"Force {style.FORCE}",         _quant.PN)
        set_si_label(self._top, "bottom", f"Extension {style.EXTENSION}", _quant.NM)
        self._top.showGrid(x=True, y=True, alpha=0.2)
        legend = self._top.addLegend(offset=(10, 10))
        legend.setParentItem(self._top.getPlotItem())

        self._data_line = self._top.plot(pen=_PEN_DATA, name="data")
        self._top.scene().sigMouseClicked.connect(self._on_plot_clicked)
        vsplit.addWidget(self._top)

        # ── Bottom: residuals (of the per-segment fits) ───────────────────────
        self._bot = pg.PlotWidget()
        set_si_label(self._bot, "left",   "Residual",                     _quant.PN)
        set_si_label(self._bot, "bottom", f"Extension {style.EXTENSION}", _quant.NM)
        self._bot.showGrid(x=True, y=True, alpha=0.2)
        self._bot.getViewBox().setXLink(self._top.getViewBox())

        self._resid_line = self._bot.plot(pen=_PEN_RESID)
        self._bot.addItem(pg.InfiniteLine(angle=0, movable=False, pen=_PEN_ZERO))
        vsplit.addWidget(self._bot)

        vsplit.setSizes([400, 200])
        main_hsplit.addWidget(vsplit)

        # ── Track list ────────────────────────────────────────────────────────
        self._track_list = QListWidget()
        self._track_list.setMinimumWidth(140)
        self._track_list.setMaximumWidth(220)
        # Solid-blue selection highlight, matching the side lists/tables in the
        # other windows (EventSummaryWindow, dashboard) so the chosen curve reads
        # the same everywhere.
        self._track_list.setStyleSheet(style.LIST_QSS)
        for p in event_paths:
            self._track_list.addItem(Path(p).name)
        self._track_list.currentRowChanged.connect(self._on_track_selected)
        main_hsplit.addWidget(self._track_list)
        main_hsplit.setStretchFactor(0, 1)   # raw retract
        main_hsplit.setStretchFactor(1, 2)   # ROI plots
        main_hsplit.setStretchFactor(2, 0)   # track list

        # Dynamically created per-segment multi-event fit lines + rupture markers.
        self._multi_items: list = []

        self._show_current()

    def update_event_list(self, event_paths: list[str]) -> None:
        """Called by EventSummaryWindow when a new event arrives during analysis."""
        current_path = self._event_paths[self._index] if self._event_paths else None
        self._event_paths = event_paths
        if current_path in event_paths:
            self._index = event_paths.index(current_path)
        else:
            self._index = min(self._index, len(event_paths) - 1)

        self._track_list.blockSignals(True)
        self._track_list.clear()
        for p in event_paths:
            self._track_list.addItem(Path(p).name)
        self._track_list.setCurrentRow(self._index)
        self._track_list.blockSignals(False)

        n = len(event_paths)
        self._counter.setText(f"{self._index + 1} / {n}")
        self._prev_btn.setEnabled(self._index > 0)
        self._next_btn.setEnabled(self._index < n - 1)

    # ── Export ───────────────────────────────────────────────────────────────

    def export_provenance(self) -> dict:
        """This window's settings, for an export manifest — same protocol
        method as the other exporting windows."""
        return {
            "window":   "wlc_view",
            "n_curves": len(self._event_paths),
        }

    def _on_export(self) -> None:
        """Export every stored WLC fit for this window's population — one row
        per (curve, segment), not just the curve on screen.

        Reads the persisted event_map documents via roi_pipeline.assemble_rows
        (the same projection the ROI Explorer uses), so no curve files are
        loaded and nothing is refitted: this reports exactly the fits the app
        already holds. Curves with no stored document are skipped by
        assemble_rows and simply do not appear — populate them first."""
        if not self._event_paths:
            QMessageBox.information(self, "Export fits", "No curves in this population.")
            return
        from .roi_pipeline import assemble_rows
        rows = assemble_rows(self._event_paths, self._db_path, mode="all")
        if not rows:
            QMessageBox.information(
                self, "Export fits",
                "No stored fits for these curves under the current settings.")
            return
        from . import clustering as _clustering
        rows = _clustering.labels_for_rows(rows)
        headers = ["path", "roi_index", "n_ruptures", "ordering", "position",
                   "seg_index", "l_p_nm", "l_c_nm", "l_p_err", "l_c_err",
                   "rupture_force_pN", "dX_from_prev_nm", "dF_from_prev_pN"]
        if _clustering.current() is not None:
            headers.append("cluster")
        with _export.export_group(
            self._db_path, "wlc_fits", [".csv"], kind="wlc_fits",
        ) as g:
            g.contributing_files(r.get("path") for r in rows)
            g.note_dict(self.export_provenance())
            g.note_dict(_clustering.provenance(self._event_paths, shown=False))
            g.note(columns=headers, n_rows=len(rows),
                   n_curves_with_fits=len({r.get("path") for r in rows}))
            g.dict_table(".csv", headers, headers, rows)
        QMessageBox.information(self, "Export fits", g.message())

    # ── Scan window link ──────────────────────────────────────────────────────

    def _find_dashboard(self):
        """Locate the singleton dashboard among the app's top-level windows.
        (The fit window is always reached from the dashboard, so it is alive.)"""
        from PyQt6.QtWidgets import QApplication
        for w in QApplication.topLevelWidgets():
            if type(w).__name__ == "DashboardWindow":
                return w
        return None

    def _on_go_to_scan(self) -> None:
        if not self._event_paths:
            return
        path = self._event_paths[self._index]
        # Prefer the dashboard (opens/reuses the singleton viewer on demand); fall
        # back to a directly-linked raw window if one was wired in.
        dash = self._find_dashboard()
        if dash is not None and hasattr(dash, "reveal_raw_at"):
            dash.reveal_raw_at(path)
            return
        if self._raw_win is not None and self._raw_win.go_to_path(path):
            return
        self._warn_no_target("scan (raw)")

    def _on_go_to_roi(self) -> None:
        if not self._event_paths:
            return
        path = self._event_paths[self._index]
        dash = self._find_dashboard()
        if dash is not None and hasattr(dash, "reveal_roi_at"):
            dash.reveal_roi_at(path)
            return
        if self._raw_win is not None:
            opener = getattr(self._raw_win, "open_roi_window", None)
            if callable(opener):
                opener()
            self._raw_win.go_to_path(path)
            return
        self._warn_no_target("ROI")

    def _on_go_to_dashboard(self) -> None:
        dash = self._find_dashboard()
        if dash is not None:
            dash.show()
            dash.raise_()
            dash.activateWindow()

    def _warn_no_target(self, what: str) -> None:
        QMessageBox.information(
            self, f"Go to {what}", f"No {what} window is available.")

    # ── Navigation ────────────────────────────────────────────────────────────

    def _go_prev(self) -> None:
        self._stop_auto()
        if self._index > 0:
            self._index -= 1
            self._show_current()

    def _go_next(self) -> None:
        self._stop_auto()
        if self._index < len(self._event_paths) - 1:
            self._index += 1
            self._show_current()

    def _on_track_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._event_paths):
            return
        self._stop_auto()
        self._index = row
        self._show_current()

    # ── Autoplay ──────────────────────────────────────────────────────────────

    def _toggle_auto(self, direction: int, checked: bool) -> None:
        if checked:
            self._auto_dir = direction
            if direction > 0:
                self._btn_auto_rev.setChecked(False)
            else:
                self._btn_auto_fwd.setChecked(False)
            self._nav_timer.start(slider_to_interval_ms(self._speed_slider.value()))
        else:
            self._nav_timer.stop()

    def _stop_auto(self) -> None:
        self._nav_timer.stop()
        self._btn_auto_fwd.setChecked(False)
        self._btn_auto_rev.setChecked(False)

    def _auto_step(self) -> None:
        if self._auto_dir > 0:
            if self._index < len(self._event_paths) - 1:
                self._index += 1
                self._show_current()
            else:
                self._nav_timer.stop()
                self._btn_auto_fwd.setChecked(False)
        else:
            if self._index > 0:
                self._index -= 1
                self._show_current()
            else:
                self._nav_timer.stop()
                self._btn_auto_rev.setChecked(False)

    def _on_speed_change(self, value: int) -> None:
        self._speed_label.setText(rate_label(value))
        if self._nav_timer.isActive():
            self._nav_timer.setInterval(slider_to_interval_ms(value))

    def _show_current(self) -> None:
        n = len(self._event_paths)
        self._counter.setText(f"{self._index + 1} / {n}")
        self._prev_btn.setEnabled(self._index > 0)
        self._next_btn.setEnabled(self._index < n - 1)

        self._track_list.blockSignals(True)
        self._track_list.setCurrentRow(self._index)
        self._track_list.blockSignals(False)
        self._track_list.scrollTo(self._track_list.currentIndex())

        path = self._event_paths[self._index]
        self._fname_label.setText(Path(path).name)

        self._clear_plots()
        self._load_and_fit(path)

    def _clear_plots(self) -> None:
        self._raw_curve.setData([], [])
        self._raw_region.hide()
        self._data_line.setData([], [])
        self._resid_line.setData([], [])
        for it in self._multi_items:
            self._top.removeItem(it)
        self._multi_items = []
        self._top.setTitle("")
        self._2dh_label.setText("")
        self._2dh_label.setStyleSheet("")
        self._clickable_segments = []
        self._manual_status_label.setText("")

    # ── Manual segment override ──────────────────────────────────────────────

    def _on_manual_mode_toggled(self, checked: bool) -> None:
        self._select_primary_btn.setEnabled(checked)
        self._select_secondary_btn.setEnabled(checked)
        if not checked:
            self._select_primary_btn.setChecked(False)
            self._select_secondary_btn.setChecked(False)

    def _on_plot_clicked(self, ev) -> None:
        """Commit a click as the armed role's segment pick. A no-op
        unless "Manually Select Segment(s)" is on AND one of Select Primary/
        Select Secondary is armed — an ordinary click while browsing fits
        never sets anything."""
        if not self._manual_mode_btn.isChecked():
            return
        armed = "primary" if self._select_primary_btn.isChecked() else (
            "secondary" if self._select_secondary_btn.isChecked() else None)
        if armed is None or not self._clickable_segments or self._current_file_id is None:
            return
        plot_item = self._top.getPlotItem()
        if not plot_item.sceneBoundingRect().contains(ev.scenePos()):
            return
        view_pos = plot_item.vb.mapSceneToView(ev.scenePos())
        x = view_pos.x()
        seg_idx = next(
            (si for (x_lo, x_hi, si) in self._clickable_segments if x_lo <= x <= x_hi),
            None,
        )
        if seg_idx is None:
            return
        from .roi_pipeline import event_geometry_identity
        if not hasattr(self, "_current_events"):
            return
        params_json = event_geometry_identity(self._current_events)
        if armed == "primary":
            _db.set_primary_segment_idx(self._current_file_id, seg_idx, params_json, self._db_path)
        else:
            _db.set_secondary_segment_idx(self._current_file_id, seg_idx, params_json, self._db_path)
        self._select_primary_btn.setChecked(False)
        self._select_secondary_btn.setChecked(False)
        self._show_current()

    # ── Load + fit ────────────────────────────────────────────────────────────

    def _load_and_fit(self, file_path: str) -> None:
        """
        Multi-event only: draw the low-frequency event-region data, each stored
        segment fit, rupture markers, and its piecewise residual. The raw retract
        remains in the overview only. No fit is repeated here on a cache hit and
        no single full-width fit is computed, plotted, or written.
        """
        try:
            curve = load_force_curve(file_path)
        except LoadError:
            self._top.setTitle("Could not load curve file")
            return

        # Whole retract in raw units (left panel).
        self._raw_curve.setData(curve.piezo_retr.tolist(), curve.defl_retr.tolist())

        from .roi_pipeline import (
            compute_curve_events_coords, event_geometry_identity, event_params_from,
            resolve_segment_override_state,
        )
        from .provenance import cache_version
        try:
            # ONE rule: the file at position one of the analysis queue decides
            # the parameter set (db.active_param_owner). No per-curve
            # resolution — a second way of deciding is how one computation
            # ended up built from two people's numbers.
            param_set = _db.load_analysis_params(self._db_path)
            ep = event_params_from(param_set)
            file_id = _db.get_file_id(file_path, self._db_path)
            res = compute_curve_events_coords(
                curve, ep, db_path=self._db_path, code_ver=cache_version(),
                file_id=file_id, param_set=param_set,
            )
            events, offset, inv, snap = res.events, res.offset, res.invols, res.snap_piezo
        except Exception:
            self._top.setTitle("Multi-event computation failed")
            return
        if not events.rois:
            self._top.setTitle("No ROI found — analyse / adjust thresholds")
            return

        self._current_file_id = file_id
        self._current_events = events

        # Primary/Secondary only ever address the right-most outer ROI
        # with ruptures — same scope Ultimate/Penultimate and the 2DH windows
        # already use. Resolved once per curve so the click hit-test and the
        # visual "P"/"S" markers below agree with segment_summary_bulk/the
        # 2DH windows about which segment those labels mean.
        target_roi = next((r for r in reversed(events.rois) if r.ruptures), None)
        if target_roi is not None:
            override = _db.get_segment_override(file_id, self._db_path)
            current_params = _db.get_latest_event_map_params(file_id, self._db_path)
            override_state = resolve_segment_override_state(
                override, current_params, len(target_roi.segments),
                event_geometry_identity(events))
            primary_idx = override_state.primary_idx
            secondary_idx = override_state.secondary_idx
            review = "   (stored choice needs review)" if override_state.status == "needs_review" else ""
        else:
            primary_idx = secondary_idx = None
            review = ""
        self._manual_status_label.setText(
            f"Primary: {'seg ' + str(primary_idx) if primary_idx is not None else 'none'}   "
            f"Secondary: {'seg ' + str(secondary_idx) if secondary_idx is not None else 'none'}"
            f"{review}"
        )

        k = curve.spring_constant
        # Match fit_segments exactly: points, markers, and residuals all use
        # the same low-frequency force envelope and extension coordinate.
        defl_corr = (res.dc.low_retr - offset) / inv
        force = k * defl_corr
        ext   = (curve.piezo_retr - snap) - defl_corr

        # Event region: first onset → terminal (last rupture).  Highlight in piezo.
        lo = min(r.onset_idx for r in events.rois)
        term_idx = max((r.ruptures[-1].idx for r in events.rois if r.ruptures),
                       default=lo)
        self._data_line.setData(ext[lo:term_idx + 1].tolist(),
                                force[lo:term_idx + 1].tolist())
        self._raw_region.setRegion([float(curve.piezo_retr[lo]),
                                    float(curve.piezo_retr[term_idx])])
        self._raw_region.show()

        # Per-segment fits (coloured) + rupture markers + piecewise residual.
        resid_x: list[float] = []
        resid_y: list[float] = []
        summary: list[str] = []
        has_terminal_fit = False   # any segment fit at all → included in total 2DH
        n_rois = len(events.rois)
        for ri, roi in enumerate(events.rois):
            n_segs = len(roi.segments)
            for si, seg in enumerate(roi.segments):
                if seg.l_p_nm is None or seg.l_c_nm is None:
                    continue
                a = seg.fit_lo_idx if seg.fit_lo_idx is not None else seg.left_idx
                b = seg.fit_hi_idx if seg.fit_hi_idx is not None else seg.right_idx
                xs = ext[a:b + 1]
                Fs = force[a:b + 1]
                m = xs > 0
                xs, Fs = xs[m], Fs[m]
                if xs.size < 2:
                    continue
                order = np.argsort(xs)
                xs, Fs = xs[order], Fs[order]
                # ROI hue (ranked FROM THE RIGHT — style.roi_hue) + per-segment
                # shade; bold and semi-transparent so the neutral data shows
                # through and same-ROI segments read as one group.
                col = style.roi_segment_qcolor(ri, n_rois, si, n_segs,
                                               alpha=style.A_MODEL)
                xm = np.linspace(float(xs.min()), float(xs.max()), 300)
                ym = np.asarray(wlc(xm, seg.l_p_nm, seg.l_c_nm))
                line = self._top.plot(
                    xm.tolist(), ym.tolist(),
                    pen=pg.mkPen(col, width=style.W_MODEL),
                )
                self._multi_items.append(line)
                self._multi_items += self._draw_fit_ci(xm, seg, col)
                resid_x += xs.tolist()
                resid_y += (Fs - wlc(xs, seg.l_p_nm, seg.l_c_nm)).tolist()
                summary.append(f"{_quant.format_value('seg_l_p_nm', seg.l_p_nm)}"
                               f"/{_quant.format_value('seg_l_c_nm', seg.l_c_nm)}")
                has_terminal_fit = True
                if roi is target_roi:
                    self._clickable_segments.append(
                        (float(xs.min()), float(xs.max()), si))
                    tag = "P" if si == primary_idx else ("S" if si == secondary_idx else None)
                    if tag is not None:
                        mid = len(xm) // 2
                        label = pg.TextItem(tag, color=style.INK, anchor=(0.5, 1.2))
                        label.setPos(float(xm[mid]), float(ym[mid]))
                        self._top.addItem(label)
                        self._multi_items.append(label)
            # Rupture markers carry the ROI's base hue (which ROI they belong to),
            # ringed in black so they stay separable where they overlap a fit line.
            mark_brush = pg.mkBrush(style.roi_segment_qcolor(ri, n_rois, 0, 1,
                                                             alpha=235))
            for rup in roi.ruptures:
                if rup.force_pN is None:
                    continue
                mi = rup.force_idx if rup.force_idx is not None else rup.idx
                mk = pg.ScatterPlotItem(
                    x=[float(ext[mi])], y=[float(rup.force_pN)],
                    size=style.MARKER_SIZE, brush=mark_brush,
                    pen=style.MARKER_PEN,
                )
                self._top.addItem(mk)
                self._multi_items.append(mk)

        if resid_x:
            ordr = np.argsort(resid_x)
            self._resid_line.setData(np.asarray(resid_x)[ordr].tolist(),
                                     np.asarray(resid_y)[ordr].tolist())
        else:
            self._resid_line.setData([], [])

        self._top.setTitle(
            f"segments ({style.L_P}/{style.L_C}): {',  '.join(summary)}" if summary
            else "No fittable segments"
        )
        if has_terminal_fit:
            self._update_2dh_status()

    # ── Per-segment parameter-variation envelope ──────────────────────────────

    def _draw_fit_ci(self, xm, seg, col) -> list:
        """Parameter-variation envelope around one segment's WLC fit.

        Uses the stored marginal standard errors (`l_p_err`/`l_c_err`, computed
        by roi_events.fit_segments and persisted in event_map).

        The same stored uncertainties are available as dashboard columns,
        criteria-gate variables, and export fields.

        The envelope spans the model evaluated at the four
        (`l_p ± σ`, `l_c ± σ`) corners. It is not a joint confidence band and
        makes no covariance claim. Purely informative; nothing gates on it.
        """
        if not self._ci_chk.isChecked():
            return []
        lp_e = seg.l_p_err if seg.l_p_err is not None else 0.0
        lc_e = seg.l_c_err if seg.l_c_err is not None else 0.0
        if not (np.isfinite(lp_e) and np.isfinite(lc_e)) or (lp_e == 0.0 and lc_e == 0.0):
            return []

        corners = []
        for dp in (-lp_e, lp_e):
            for dc in (-lc_e, lc_e):
                lp = max(seg.l_p_nm + dp, 1e-6)
                lc = seg.l_c_nm + dc
                if lc <= float(xm.max()):
                    continue                       # WLC pole — undefined there
                with np.errstate(all="ignore"):
                    corners.append(np.asarray(wlc(xm, lp, lc), dtype=float))
        if not corners:
            return []
        stack = np.vstack(corners)
        if not np.isfinite(stack).all():
            return []

        up = pg.PlotDataItem(xm.tolist(), stack.max(axis=0).tolist())
        lo = pg.PlotDataItem(xm.tolist(), stack.min(axis=0).tolist())
        band = pg.FillBetweenItem(up, lo, brush=style.band_brush(col, alpha=55))
        self._top.addItem(band)
        return [band]

    # ── Total-2DH inclusion indicator ─────────────────────────────────────────
    # Every curve with a real segment fit is included in the total 2DH. This
    # indicator reports that inclusion without applying another quality gate.

    def _update_2dh_status(self) -> None:
        if self._2dh_win is None:
            return
        self._2dh_label.setText("Included in total 2DH")
        self._2dh_label.setStyleSheet(style.qss_text(style.TEXT_GOOD, bold=True))
