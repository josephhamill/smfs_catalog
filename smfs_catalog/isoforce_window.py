# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/isoforce_window.py
#
# IsoforceWindow — single-curve navigator for the lab's manual "isoforce
# distance" measurement, made from the same fit-independent geometry
# roi_pipeline.segment_summary_bulk already surfaces as seg_dX_iso_nm/
# seg_dF_pN. The resolved adjacent pair is the manual Primary/Secondary pair
# when both are set, otherwise the final adjacent pair.
#
# Uses the same prev/next, autoplay, track-list, and cross-window navigation as
# WlcViewWindow. Deliberately does not draw WLC fit lines: the
# isoforce distance is a direct read of force/extension (roi_events.py:
# "Fit-independent — a direct read of force/extension, not derived from
# l_p/l_c"). Those direct points are measured on the decomposed low-frequency
# force/extension trace, the same coordinate fit_segments stores them from;
# "fit-independent" does not mean raw-channel geometry.
#
# Population: curves whose right-most outer ROI has a usable adjacent isoforce
# pair: the current manual Primary/Secondary pair when both are set, otherwise
# the last two ruptures. Supplied by the caller
# (EventSummaryWindow._isoforce_paths), never queried independently here.

from __future__ import annotations

from pathlib import Path

import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
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

from . import sample_marks
from . import style
from .widgets import SampleMarksToggle
from . import db as _db
from . import export_utils as _export
from .curve_loader import LoadError, load_force_curve
from .qt_utils import _make_session_header, set_si_label, fit_on_screen
from .quantities import format_value as _q   # ONE formatter: unit and
# meaningful digits come from quantities.py, so the same measurement
# cannot print as 166, 166.2 and 166.20 in three different windows.
from . import quantities as _quant
# This window drives its OWN QTimer over its own curve list — it is not the
# analysis worker's playhead, so it shares navigator_bar's slider maths but
# deliberately not its NavigatorBar.
from .navigator_bar import (
    slider_to_interval_ms, rate_to_slider, rate_label,
    SLIDER_MIN, SLIDER_MAX, DEFAULT_RATE_HZ,
)

_PEN_GUIDE    = style.hair_pen(style.INK_MUTED)
_PEN_DX       = style.guide_pen(style.LM_RUPTURE,   width=style.W_MODEL)
_PEN_DF       = style.guide_pen(style.SERIES_LINE[1], width=style.W_MODEL)

# The three marked points are the subject of this window, so they get three
# measured-distinct hues; everything else in the ROI is neutral.
_BRUSH_RUP_A  = style.marker_brush(style.LM_ONSET)        # earlier rupture (amber)
_BRUSH_CROSS  = style.marker_brush(style.LM_RUPTURE)      # isoforce crossing (green)
_BRUSH_RUP_B  = style.marker_brush(style.SERIES_LINE[2])  # terminal rupture (violet)
_BRUSH_OTHER  = pg.mkBrush(*style.NON_HIT_RGBA)           # other ruptures in the ROI


class IsoforceWindow(QMainWindow):
    """
    Single-panel isoforce navigator.

    event_paths : ordered list of paths with a usable adjacent isoforce pair
                  on the right-most outer ROI, from the caller's population —
                  never queried from the DB independently.
    index       : which path to show first.
    population  : "hit" / "non_hit" — which EventSummaryWindow population this
                  was opened for, so a live refresh asks for the right list.
    """

    def __init__(
        self,
        event_paths:  list[str],
        index:        int,
        db_path:      str,
        session_info: dict | None = None,
        raw_window:   object | None = None,
        population:   str = "hit",
    ) -> None:
        super().__init__()
        self.setWindowFlag(Qt.WindowType.Window)
        self.setWindowTitle("SMFS — isoforce")
        fit_on_screen(self, 1100, 700)
        self._event_paths  = event_paths
        self._index         = index
        self._db_path       = db_path
        self._session_info  = session_info
        self._raw_win       = raw_window
        self.population      = population
        self._auto_dir      = 1

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

        # ── Navigation row ───────────────────────────────────────────────────
        nav = QHBoxLayout()

        self._btn_auto_rev = QPushButton("◀◀")
        self._btn_auto_rev.setCheckable(True)
        self._btn_auto_rev.setFixedWidth(44)
        self._btn_auto_rev.clicked.connect(lambda checked: self._toggle_auto(-1, checked))

        self._prev_btn = QPushButton("◀ Prev")
        self._next_btn = QPushButton("Next ▶")
        self._prev_btn.clicked.connect(self._go_prev)
        self._next_btn.clicked.connect(self._go_next)

        self._btn_auto_fwd = QPushButton("▶▶")
        self._btn_auto_fwd.setCheckable(True)
        self._btn_auto_fwd.setFixedWidth(44)
        self._btn_auto_fwd.clicked.connect(lambda checked: self._toggle_auto(1, checked))

        font = self._prev_btn.font()
        font.setPointSize(style.FONT_SMALL_PT)

        lbl_slow = QLabel("Slow"); lbl_slow.setFont(font)
        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(SLIDER_MIN, SLIDER_MAX)
        self._speed_slider.setValue(rate_to_slider(DEFAULT_RATE_HZ))
        self._speed_slider.setFixedWidth(100)
        self._speed_slider.valueChanged.connect(self._on_speed_change)
        lbl_fast = QLabel("Fast"); lbl_fast.setFont(font)
        self._speed_label = QLabel(rate_label(self._speed_slider.value()))
        self._speed_label.setFixedWidth(64)
        self._speed_label.setFont(font)

        self._counter = QLabel(); self._counter.setFont(font)
        self._fname_label = QLabel(); self._fname_label.setFont(font)

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

        nav.addWidget(QLabel("Go to:"))
        self._go_raw_btn = QPushButton("Raw")
        self._go_raw_btn.setToolTip("Pause analysis and open this curve in the scan (raw) window")
        self._go_raw_btn.clicked.connect(self._on_go_to_scan)
        nav.addWidget(self._go_raw_btn)

        self._go_roi_btn = QPushButton("ROI")
        self._go_roi_btn.setToolTip(
            "Open the event search — the detection signals and thresholds "
            "that decide where this curve's ROIs are, and therefore which "
            "ruptures and segments it has."
        )
        self._go_roi_btn.setToolTip("Open the ROI detection window on this curve")
        self._go_roi_btn.clicked.connect(self._on_go_to_roi)
        nav.addWidget(self._go_roi_btn)

        self._go_dash_btn = QPushButton("Dashboard")
        self._go_dash_btn.setToolTip("Bring the dashboard window to the front")
        self._go_dash_btn.clicked.connect(self._on_go_to_dashboard)
        nav.addWidget(self._go_dash_btn)

        nav.addStretch()
        nav.addWidget(SampleMarksToggle())
        self._export_btn = QPushButton("Export…")
        self._export_btn.setToolTip(
            "Write the isoforce measurements for EVERY curve in this window's "
            "population (not just the one on screen) to the export folder, "
            "with a manifest."
        )
        self._export_btn.clicked.connect(self._on_export)
        nav.addWidget(self._export_btn)
        root.addLayout(nav)

        # ── Readout row ──────────────────────────────────────────────────────
        self._readout = QLabel("")
        self._readout.setFont(font)
        root.addWidget(self._readout)

        # ── Main area: plot | track list ────────────────────────────────────
        main_hsplit = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(main_hsplit, stretch=1)

        self._plot = pg.PlotWidget()
        set_si_label(self._plot, "left",   f"Force {style.FORCE}",         _quant.PN)
        set_si_label(self._plot, "bottom", f"Extension {style.EXTENSION}", _quant.NM)
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._data_line = sample_marks.trace(self._plot, name="data")
        main_hsplit.addWidget(self._plot)

        self._track_list = QListWidget()
        self._track_list.setMinimumWidth(140)
        self._track_list.setMaximumWidth(220)
        self._track_list.setStyleSheet(style.LIST_QSS)
        for p in event_paths:
            self._track_list.addItem(Path(p).name)
        self._track_list.currentRowChanged.connect(self._on_track_selected)
        main_hsplit.addWidget(self._track_list)
        main_hsplit.setStretchFactor(0, 1)
        main_hsplit.setStretchFactor(1, 0)

        # Dynamically created markers/guides, cleared and rebuilt per curve.
        self._plot_items: list = []

        self._show_current()

    def update_event_list(self, event_paths: list[str]) -> None:
        """Called by EventSummaryWindow when the isoforce population changes
        during live analysis (a curve gains/loses a usable adjacent pair)."""
        current_path = self._event_paths[self._index] if self._event_paths else None
        self._event_paths = event_paths
        if current_path in event_paths:
            self._index = event_paths.index(current_path)
        else:
            self._index = min(self._index, len(event_paths) - 1) if event_paths else 0

        self._track_list.blockSignals(True)
        self._track_list.clear()
        for p in event_paths:
            self._track_list.addItem(Path(p).name)
        if event_paths:
            self._track_list.setCurrentRow(self._index)
        self._track_list.blockSignals(False)

        if not event_paths:
            self._clear_plots()
            self._counter.setText("0 / 0")
            self._readout.setText("No curves currently have a usable adjacent isoforce pair.")
            self._prev_btn.setEnabled(False)
            self._next_btn.setEnabled(False)
            return

        n = len(event_paths)
        self._counter.setText(f"{self._index + 1} / {n}")
        self._prev_btn.setEnabled(self._index > 0)
        self._next_btn.setEnabled(self._index < n - 1)
        self._show_current()

    # ── Export ───────────────────────────────────────────────────────────────

    def export_provenance(self) -> dict:
        """This window's settings, for an export manifest — same protocol
        method as the other exporting windows."""
        return {
            "window":     "isoforce",
            "population": "curves with a usable adjacent isoforce pair",
            "n_curves":   len(self._event_paths),
        }

    def _on_export(self) -> None:
        """Export the whole population's isoforce measurements, one row per
        curve — not just the curve currently displayed.

        Values come from roi_pipeline.segment_summary_bulk, the same source
        the dashboard queue, Explore Events and the criteria gate read, so
        this file cannot disagree with them. It reads stored event_map
        documents, so no curve files are loaded and this stays fast over a
        large population.

        dX_iso_nm is legitimately blank wherever the later rupture is the
        weaker one: that curve never reloaded back up to the earlier force, so
        blank is the scientifically meaningful result."""
        if not self._event_paths:
            QMessageBox.information(self, "Export", "No curves in this population.")
            return
        from .roi_pipeline import read_segment_select, segment_summary_bulk
        select = read_segment_select(self._db_path)
        summ = segment_summary_bulk(self._event_paths, select, self._db_path)
        rows = []
        for p in self._event_paths:
            d = summ.get(_db.normalize_path(p)) or {}
            rows.append((
                p,
                d.get("dX_iso_nm"), d.get("dF_pN"), d.get("dX_ext_nm"),
                d.get("force_pN"),
                d.get("l_p_nm"), d.get("l_p_err"),
                d.get("l_c_nm"), d.get("l_c_err"),
                d.get("n_segments"),
            ))
        with _export.export_group(
            self._db_path, "isoforce", [".csv"], kind="isoforce",
        ) as g:
            g.contributing_files(self._event_paths)
            g.note_dict(self.export_provenance())
            g.note(segment_select=select,
                   n_with_dx_iso=sum(1 for r in rows if r[1] is not None))
            g.table(
                ".csv",
                ["path", "dX_iso_nm", "dF_pN", "dX_ext_nm", "force_pN",
                 "l_p_nm", "l_p_err_nm", "l_c_nm", "l_c_err_nm",
                 "n_segments"],
                [["" if v is None else v for v in r] for r in rows],
            )
        QMessageBox.information(self, "Export", g.message())

    # ── Cross-window navigation ─────────────────────────────────────────────

    def _find_dashboard(self):
        from PyQt6.QtWidgets import QApplication
        for w in QApplication.topLevelWidgets():
            if type(w).__name__ == "DashboardWindow":
                return w
        return None

    def _on_go_to_scan(self) -> None:
        if not self._event_paths:
            return
        path = self._event_paths[self._index]
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
            dash.show(); dash.raise_(); dash.activateWindow()

    def _warn_no_target(self, what: str) -> None:
        QMessageBox.information(self, f"Go to {what}", f"No {what} window is available.")

    # ── Navigation ───────────────────────────────────────────────────────────

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

    # ── Autoplay ─────────────────────────────────────────────────────────────

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
        self._load_and_mark(path)

    def _clear_plots(self) -> None:
        self._data_line.setData([], [])
        for it in self._plot_items:
            self._plot.removeItem(it)
        self._plot_items = []
        self._plot.setTitle("")
        self._readout.setText("")

    # ── Load + mark ──────────────────────────────────────────────────────────

    def _load_and_mark(self, file_path: str) -> None:
        """Draw the event-region force/extension data and mark the resolved
        isoforce pair — no WLC fit lines (see module docstring)."""
        try:
            curve = load_force_curve(file_path)
        except LoadError:
            self._plot.setTitle("Could not load curve file")
            return

        from .roi_pipeline import (
            compute_curve_events_coords, event_geometry_identity,
            event_params_from, resolve_isoforce_pair,
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
            self._plot.setTitle("Multi-event computation failed")
            return
        if not events.rois:
            self._plot.setTitle("No ROI found — analyse / adjust thresholds")
            return

        roi = next((r for r in reversed(events.rois) if r.ruptures), None)
        if roi is None or len(roi.ruptures) < 2:
            self._plot.setTitle("This curve no longer has a second rupture on its last ROI")
            self._readout.setText("Not a qualifying curve (< 2 ruptures) — dropped on next refresh.")
            return

        override = _db.get_segment_override(file_id, self._db_path)
        current_params = _db.get_latest_event_map_params(file_id, self._db_path)
        override_state = resolve_segment_override_state(
            override, current_params, len(roi.segments),
            event_geometry_identity(events),
        )
        pair = resolve_isoforce_pair(
            len(roi.segments), override_state.primary_idx,
            override_state.secondary_idx,
        )
        if pair is None:
            self._plot.setTitle("Selected segments have no isoforce geometry")
            self._readout.setText(
                "Primary and Secondary must be adjacent for an isoforce crossing."
            )
            return

        k = curve.spring_constant
        defl_corr = (res.dc.low_retr - offset) / inv
        force = k * defl_corr
        ext   = (curve.piezo_retr - snap) - defl_corr

        lo = min(r.onset_idx for r in events.rois)
        term_idx = max((r.ruptures[-1].idx for r in events.rois if r.ruptures), default=lo)
        self._data_line.setData(ext[lo:term_idx + 1].tolist(), force[lo:term_idx + 1].tolist())

        lo_idx, hi_idx = pair
        rup_a, rup_b = roi.ruptures[lo_idx], roi.ruptures[hi_idx]
        seg_next = roi.segments[hi_idx]
        if (rup_a.extension_nm is None or rup_a.force_pN is None
                or rup_b.extension_nm is None or rup_b.force_pN is None
                or seg_next.isoforce_x_nm is None):
            self._plot.setTitle("Isoforce geometry incomplete for this curve")
            self._readout.setText("Missing extension/force/crossing value — re-run analysis to fill it in.")
            return

        cross_x, cross_y = seg_next.isoforce_x_nm, rup_a.force_pN
        dx_iso = cross_x - rup_a.extension_nm
        dF     = rup_b.force_pN - rup_a.force_pN

        # Other ruptures in this ROI, dim, for context.
        for rup in roi.ruptures:
            if rup in (rup_a, rup_b) or rup.extension_nm is None or rup.force_pN is None:
                continue
            mk = pg.ScatterPlotItem(
                x=[rup.extension_nm], y=[rup.force_pN],
                size=9, brush=_BRUSH_OTHER, pen=style.MARKER_PEN,
            )
            self._plot.addItem(mk)
            self._plot_items.append(mk)

        # L-shaped guide: across at rup_a's force, then up to rup_b's force.
        guide_h = self._plot.plot(
            [rup_a.extension_nm, rup_b.extension_nm], [rup_a.force_pN, rup_a.force_pN],
            pen=_PEN_GUIDE,
        )
        guide_v = self._plot.plot(
            [rup_b.extension_nm, rup_b.extension_nm], [rup_a.force_pN, rup_b.force_pN],
            pen=_PEN_GUIDE,
        )
        self._plot_items += [guide_h, guide_v]

        # ΔX_iso: rup_a's extension -> the crossing point, drawn bold over the guide.
        dx_line = self._plot.plot(
            [rup_a.extension_nm, cross_x], [rup_a.force_pN, rup_a.force_pN], pen=_PEN_DX,
        )
        self._plot_items.append(dx_line)
        dx_label = pg.TextItem(f"ΔX_iso = {_q('seg_dX_iso_nm', dx_iso, with_unit=True)}",
                               color=(0, 110, 50), anchor=(0.5, 1.1))
        dx_label.setPos((rup_a.extension_nm + cross_x) / 2, rup_a.force_pN)
        self._plot.addItem(dx_label)
        self._plot_items.append(dx_label)

        # ΔF: rup_a's force -> rup_b's force, drawn bold over the vertical guide.
        df_line = self._plot.plot(
            [rup_b.extension_nm, rup_b.extension_nm], [rup_a.force_pN, rup_b.force_pN], pen=_PEN_DF,
        )
        self._plot_items.append(df_line)
        df_label = pg.TextItem(f"ΔF = {_q('seg_dF_pN', dF, with_unit=True)}",
                               color=(160, 50, 0), anchor=(-0.1, 0.5))
        df_label.setPos(rup_b.extension_nm, (rup_a.force_pN + rup_b.force_pN) / 2)
        self._plot.addItem(df_label)
        self._plot_items.append(df_label)

        # The three markers, drawn last so they sit on top of the guides/lines.
        for x, y, brush, sym in (
            (rup_a.extension_nm, rup_a.force_pN, _BRUSH_RUP_A, "s"),
            (cross_x,            cross_y,        _BRUSH_CROSS, "d"),
            (rup_b.extension_nm, rup_b.force_pN, _BRUSH_RUP_B, "o"),
        ):
            mk = pg.ScatterPlotItem(
                x=[x], y=[y], size=14, symbol=sym,
                brush=brush, pen=pg.mkPen(style.INK, width=1.3),
            )
            self._plot.addItem(mk)
            self._plot_items.append(mk)

        self._plot.setTitle(
            f"rupture 1 (■) {_q('seg_force_pN', rup_a.force_pN, with_unit=True)}  →  "
            f"crossing (◆)  →  rupture 2 (●) {_q('seg_force_pN', rup_b.force_pN, with_unit=True)}"
        )
        self._readout.setText(
            f"ΔX_iso (isoforce distance) = {_q('seg_dX_iso_nm', dx_iso, with_unit=True)}      "
            f"ΔF (rupture 2 − rupture 1) = {_q('seg_dF_pN', dF, with_unit=True)}"
        )
