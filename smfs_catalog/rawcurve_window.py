# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

import json
import numpy as np
from pathlib import Path

import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .curve_loader import (
    ForceCurve, LoadError, RawTrace, UnusableCurveError, load_force_curve,
    load_raw_trace,
)
from . import db as _db
from .provenance import cache_version
from .decomposition_window import DecompositionWindow
from .fft_window import FftWindow
from .navigator_bar import NavigatorBar
from . import style
from .qt_utils import (
    fit_on_screen,
    _make_session_header,
    set_si_label,
)
from . import quantities as _quant

_SMALL_FONT_PT = style.FONT_CAPTION_PT
_COLOR_CONTACT = style.LM_CONTACT
_COLOR_SNAPOFF = style.LM_SNAPOFF


# Playback (step/play/speed/position) is NOT implemented here.  It belongs to
# the worker, and NavigatorBar is the one control surface for it — the same
# widget the dashboard shows, so the two can never disagree about where the
# playhead is or how fast it is allowed to run.  See navigator_bar.py.


# ── Metadata panel row definitions ────────────────────────────────────────────
# Each entry: (field_key, display_label, unit_suffix)
# "_sep" inserts a horizontal rule — no value label.
_META_ROWS: list[tuple[str, str, str]] = [
    ("filename",      "File",        ""),
    ("directory",     "Directory",   ""),
    ("date",          "Date",        ""),
    ("spring_k",      "k",           "pN/nm"),
    ("velocity",      "Velocity",    "nm/s"),
    ("trigger",       "Trigger",     "nN"),  # TriggerPoint in wave note is Newtons (SI); ×1e9 → nN, not nm
    ("force_dist",    "Force dist",  "nm"),
    ("inv_ols",       "InvOLS",      "nm/V"),
    ("xy",            "XY",          "µm"),
    ("sample_rate",   "Sample rate", "Hz"),
]



# ── Axis choices ──────────────────────────────────────────────────────────────
# (label, key, x-quantity, y-quantity).  Deflection against piezo is the force
# curve; the two time axes are the only way to see a segment held at constant
# position, where the piezo axis collapses the whole hold onto one x value.
_AXES: list[tuple[str, str, str, str]] = [
    ("Defl vs Piezo", "defl_piezo", "piezo", "defl"),
    ("Defl vs Time",  "defl_time",  "time",  "defl"),
    ("Piezo vs Time", "piezo_time", "time",  "piezo"),
]
_AXIS_LABEL = {"piezo": ("Piezo", _quant.NM),
               "defl":  ("Deflection", _quant.NM),
               "time":  ("Time", _quant.S)}


# ── Raw curve window ───────────────────────────────────────────────────────────

class RawCurveWindow(QWidget):
    """
    Standalone SMFS curve viewer, and one of the worker's two control surfaces.

    Left pane  : approach (red) + retract (blue) plot.
    Right pane : fixed-position metadata panel — rows never reorder,
                 values update in-place on every navigation step.
    Top        : a NavigatorBar — the SAME widget the dashboard carries, driving
                 the same worker.  This window has no navigation of its own.

    Emits curve_changed(index) on every navigation step, including the
    initial display.
    """

    curve_changed        = pyqtSignal(int)
    derived_result_ready = pyqtSignal(int, float, float, float, float, float, float, float, float, str, str, str, str)   # index, offset, flatness, contact_z, snapoff_z, rupture_z, onset_z, invols_slope, rupture_force, cd_params_json, bl_params_json, roi_params_json, invols_params_json

    def __init__(
        self,
        paths: list[str],
        db_path: str | None = None,
        session_info: dict | None = None,
        experimentalist: str | None = None,
        worker=None,
    ) -> None:
        """
        The window follows a running AnalysisWorker: it shows whichever file the
        worker's playhead is on (via playhead_changed), and its NavigatorBar
        drives that playhead.  `paths` is ignored — the worker owns the timeline;
        pass [].

        `worker` is required in practice.  It stayed keyword-optional only so the
        signature doesn't change under existing callers; without one the window
        draws nothing, having no timeline to follow.
        """
        super().__init__()

        self._worker               = worker
        # A 1-element list, rebuilt on each playhead change — the rest of the
        # window's drawing code doesn't care how many curves are notionally
        # in the list.
        self._paths: list[str]     = []
        self._index                = 0
        self._n_total              = 0
        self._n_errors             = 0
        self._last_displayed_index = -1   # index last drawn; -1 = nothing yet
        self._current_file_id: int | None = None
        self._db_path              = db_path or _db.DEFAULT_DB_PATH
        self._experimentalist         = experimentalist
        self._decomp_win:  DecompositionWindow  | None = None   # created lazily
        self._fft_win:       FftWindow       | None = None   # created lazily
        self._roi_win:       "ROIWindow"        | None = None
        # Last thing drawn, so changing axes re-plots without re-reading the
        # file.  Either a ForceCurve (a ramp, drawn as approach + retract) or a
        # RawTrace (anything else, drawn as one series).
        self._drawn: ForceCurve | RawTrace | None = None
        self._axes                 = _AXES[0][1]

        # ── Window ────────────────────────────────────────────────────────────
        self.setWindowTitle("SMFS — raw curves")
        self.setWindowFlag(Qt.WindowType.Window)
        fit_on_screen(self, 1100, 640)
        # ── Root layout ───────────────────────────────────────────────────────
        root = QVBoxLayout(self)
        root.setSpacing(4)
        root.setContentsMargins(6, 6, 6, 6)

        # ── Session header — read-only one-liner showing analysis context ────────
        self._session_info = session_info
        _hdr = _make_session_header(session_info)
        if _hdr is not None:
            root.addWidget(_hdr)

        # ── Navigator — the shared worker transport ────────────────────────────
        # Placed above the plot so it stays clear of the taskbar on small screens.
        # This is the same widget class the dashboard shows; neither is the
        # master.  See navigator_bar.py.
        self._nav: NavigatorBar | None = None
        if self._worker is not None:
            self._nav = NavigatorBar(self._worker, self)
            root.addWidget(self._nav)

        # ── Tool bar (per-curve inspection windows) ───────────────────────────
        ctrl = QWidget()
        ctrl_layout = QHBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(0, 2, 0, 2)
        ctrl_layout.setSpacing(8)

        self._status_label = QLabel("")
        ctrl_layout.addWidget(self._status_label)
        ctrl_layout.addSpacing(16)

        self._axis_box = QComboBox()
        for label, key, _x, _y in _AXES:
            self._axis_box.addItem(label, key)
        self._axis_box.setToolTip(
            "Which quantities to plot.  A held segment is flat in piezo, so it "
            "is only visible against time."
        )
        self._axis_box.currentIndexChanged.connect(self._on_axes_changed)
        ctrl_layout.addWidget(self._axis_box)
        ctrl_layout.addSpacing(8)

        self._btn_decomp = QPushButton("Decomp")
        self._btn_decomp.setFixedWidth(72)
        self._btn_decomp.setCheckable(True)
        self._btn_decomp.clicked.connect(self._toggle_decomp)
        ctrl_layout.addWidget(self._btn_decomp)

        self._btn_fft = QPushButton("FFT")
        self._btn_fft.setFixedWidth(52)
        self._btn_fft.setCheckable(True)
        self._btn_fft.clicked.connect(self._toggle_fft)
        ctrl_layout.addWidget(self._btn_fft)

        self._btn_roi = QPushButton("ROI")
        self._btn_roi.setToolTip(
            "Open the event search — the detection signals and thresholds "
            "that decide where this curve's ROIs are, and therefore which "
            "ruptures and segments it has."
        )
        self._btn_roi.setFixedWidth(52)
        self._btn_roi.setEnabled(False)
        self._btn_roi.clicked.connect(self._show_roi_window)
        ctrl_layout.addWidget(self._btn_roi)

        # Per-ROI and per-segment cohort exploration belongs in Explore Events,
        # where population selection and export manifests preserve scope.
        ctrl_layout.addStretch()

        root.addWidget(ctrl)

        # ── Horizontal splitter: plot | metadata ──────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        # ── Plot widget ───────────────────────────────────────────────────────
        style.apply_plot_defaults()

        self._plot = pg.PlotWidget()
        set_si_label(self._plot, "bottom", "Piezo",      _quant.NM)
        set_si_label(self._plot, "left",   "Deflection", _quant.NM)
        _legend = self._plot.addLegend()
        _legend.anchor(itemPos=(0.5, 0.5), parentPos=(0.5, 0.5))
        self._plot.showGrid(x=True, y=True, alpha=0.2)

        self._curve_appr = self._plot.plot(
            [], [], pen=style.data_pen(style.SIG_APPROACH), name="approach"
        )
        self._curve_retr = self._plot.plot(
            [], [], pen=style.data_pen(style.SIG_RETRACT), name="retract"
        )
        # One series for a curve with no approach/retract split to make.
        self._curve_raw = self._plot.plot(
            [], [], pen=style.data_pen(style.SIG_RETRACT), name="trace"
        )
        # Contact marker lines — added/removed each draw, None when not shown
        self._contact_appr_line = None   # vertical dashed line: contact onset (approach)
        self._contact_retr_line = None   # vertical dashed line: snap-off (retract)
        # Rupture/onset are drawn one pair per outer ROI (a curve can hold more
        # than one), not a single scalar pair — see _draw_derived.
        self._rupture_lines: list = []   # vertical dashed lines: one per ROI's rupture
        self._onset_lines:   list = []   # vertical dashed lines: one per ROI's onset
        splitter.addWidget(self._plot)

        # ── Metadata panel ────────────────────────────────────────────────────
        self._meta_vals: dict[str, QLabel] = {}

        meta_inner = QWidget()
        meta_inner.setMinimumWidth(180)
        meta_vbox = QVBoxLayout(meta_inner)
        meta_vbox.setContentsMargins(8, 6, 6, 6)
        meta_vbox.setSpacing(0)

        # Build two QFormLayouts separated by a horizontal rule
        form_top = QFormLayout()
        form_top.setSpacing(3)
        form_top.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form_bot = QFormLayout()
        form_bot.setSpacing(3)
        form_bot.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        current_form = form_top
        # Give each form a visible owner before adding rows.  With PyQt6,
        # populating these parentless layouts first left their QLabel widgets
        # orphaned (parentWidget() was None), reserving a blank panel beside
        # the plot instead of painting the metadata.
        meta_vbox.addLayout(form_top)
        for key, label_text, _unit in _META_ROWS:
            if key == "_sep":
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.HLine)
                sep.setFrameShadow(QFrame.Shadow.Sunken)
                meta_vbox.addSpacing(4)
                meta_vbox.addWidget(sep)
                meta_vbox.addSpacing(4)
                current_form = form_bot
                meta_vbox.addLayout(form_bot)
                continue
            val_lbl = QLabel("—")
            val_lbl.setWordWrap(True)
            font = val_lbl.font()
            font.setPointSize(_SMALL_FONT_PT)
            val_lbl.setFont(font)
            row_lbl = QLabel(label_text)
            row_lbl.setFont(font)
            self._meta_vals[key] = val_lbl
            current_form.addRow(row_lbl, val_lbl)

        meta_vbox.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(meta_inner)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMinimumWidth(180)
        scroll.setMaximumWidth(320)

        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)   # plot gets all extra space
        splitter.setStretchFactor(1, 0)   # panel stays at its natural width

        # ── Follow the worker ─────────────────────────────────────────────────
        if self._worker is not None:
            self._worker.playhead_changed.connect(self._on_worker_playhead)
            self._worker.queue_empty.connect(self._on_worker_queue_empty)
            self._worker.file_done.connect(self._on_worker_file_done)
            self._worker.file_error.connect(self._on_worker_file_error)
            self._worker.data_unavailable.connect(self._on_worker_data_unavailable)

            # Recorded so closeEvent can detach them again. Qt's close() only
            # hides a window; without this, the hidden viewer would keep
            # loading and drawing each analysed curve.
            self._worker_signal_links = [
                (self._worker.playhead_changed,  self._on_worker_playhead),
                (self._worker.queue_empty,       self._on_worker_queue_empty),
                (self._worker.file_done,         self._on_worker_file_done),
                (self._worker.file_error,        self._on_worker_file_error),
                (self._worker.data_unavailable, self._on_worker_data_unavailable),
            ]
            self._worker_linked = True

            # Create the ROI window eagerly — it is the tuning surface for
            # ROI detection params and belongs alongside the raw view.
            try:
                from .display_roi import ROIWindow
                self._roi_win = ROIWindow(
                    self._db_path, experimentalist=experimentalist,
                    worker=self._worker,
                )
                self._btn_roi.setEnabled(True)
            except Exception as exc:
                # If ROIWindow construction fails (missing deps in some env),
                # don't crash the raw viewer — just leave the button disabled.
                self._roi_win = None
                self._btn_roi.setToolTip(
                    "ROI detection window unavailable: "
                    f"{type(exc).__name__}: {str(exc)[:160]}"
                )

            # If the worker already has a playhead (re-opening), draw it;
            # otherwise tell the user where to start.
            cur = self._worker.playhead()
            if cur is not None:
                self._on_worker_playhead(cur)
            else:
                self._show_worker_hint()

    def go_to_path(self, path: str) -> bool:
        """
        Navigate to the curve at `path`, pausing playback.  Returns False only if
        the curve cannot be resolved at all.

        The worker owns the timeline, so this resolves path→file_id and drives
        the worker's playhead with the same transient-enqueue + step_to pattern
        the dashboard uses for double-click navigation.
        """
        if self._worker is None:
            return False
        fid = _db.get_file_id(path, self._db_path)
        if fid is None:
            return False
        self._worker.set_paused(True)
        _db.enqueue_files([fid], self._db_path)
        self._worker.invalidate_queue_cache()   # navigator bars re-range on its signal
        self._worker.step_to(fid)
        self.raise_()
        self.activateWindow()
        return True

    def _toggle_decomp(self, checked: bool) -> None:
        if checked:
            if self._decomp_win is None:
                self._decomp_win = DecompositionWindow(
                    self._db_path, self._experimentalist, self._session_info,
                    worker=self._worker,
                )
                self._decomp_win.analysis_params_changed.connect(self._on_analysis_params_changed)
            self._decomp_win.show()
            self._decomp_win.raise_()
            # Populate immediately with the currently displayed curve
            if self._last_displayed_index >= 0:
                self._do_draw(self._last_displayed_index)
        else:
            if self._decomp_win is not None:
                self._decomp_win.hide()

    def _toggle_fft(self, checked: bool) -> None:
        if checked:
            if self._fft_win is None:
                self._fft_win = FftWindow(self._session_info)
            self._fft_win.show()
            self._fft_win.raise_()
            if self._last_displayed_index >= 0:
                self._do_draw(self._last_displayed_index)
        else:
            if self._fft_win is not None:
                self._fft_win.hide()

    def set_roi_window(self, win: "ROIWindow") -> None:
        """Called by analysis_runner once the ROIWindow is created."""
        self._roi_win = win
        self._btn_roi.setEnabled(True)

    def open_roi_window(self) -> bool:
        """Public entry point (used by the WLC fit window's 'ROI' button) to
        reveal the ROI detection window on the current curve.  Returns False when
        no ROI window exists (e.g. construction failed in a minimal env)."""
        if self._roi_win is None:
            return False
        self._show_roi_window()
        return True

    def _show_roi_window(self) -> None:
        if self._roi_win is not None:
            self._roi_win.show()
            self._roi_win.raise_()
            self._roi_win.activateWindow()
            # Worker mode: the ROI window's showEvent syncs it to the worker's
            # current playhead, so don't push a RawCurve index it doesn't share.
            if self._worker is None:
                idx = self._last_displayed_index if self._last_displayed_index >= 0 else self._index
                if idx >= 0:
                    self._roi_win.update_curve(idx)

    def _on_analysis_params_changed(self) -> None:
        """Redraw current curve when any spectral analysis parameter changes."""
        if self._last_displayed_index >= 0:
            self._do_draw(self._last_displayed_index)

    # -- live-work attach/detach ---------------------------------------------
    # A closed window must cost nothing.  Qt's close() only hides, so the
    # worker's per-file signals keep arriving otherwise.

    def _detach_live_work(self) -> None:
        """Stop everything that costs time while the window is not visible."""
        if self._nav is not None:
            self._nav.detach()
        if getattr(self, "_worker_linked", False):
            for signal, slot in getattr(self, "_worker_signal_links", []):
                try:
                    signal.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass          # already disconnected; nothing to undo
            self._worker_linked = False

    def _attach_live_work(self) -> None:
        """Re-subscribe on reopen.  Guarded so a re-show cannot double-connect."""
        # Playback state is NOT restored here — whether the worker is running is
        # the user's choice, held by the worker, not something a window reopening
        # gets to change.  The bar just re-reads it.
        if self._nav is not None:
            self._nav.attach()
        links = getattr(self, "_worker_signal_links", [])
        if not links or getattr(self, "_worker_linked", False):
            return
        for signal, slot in links:
            signal.connect(slot)
        self._worker_linked = True

    def closeEvent(self, event) -> None:
        self._detach_live_work()
        super().closeEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._attach_live_work()

    # ── Worker mode wiring ────────────────────────────────────────────────────

    def _on_worker_playhead(self, file_id: int) -> None:
        """Worker tells us a new file is current — load and draw it."""
        self._current_file_id = int(file_id)
        conn = _db.get_connection(self._db_path)
        row = conn.execute(
            "SELECT path FROM files WHERE id = ?", (file_id,)
        ).fetchone()
        conn.close()
        if row is None:
            return
        path = row["path"]
        # Treat the current file as a 1-element timeline.  _do_draw will use
        # _paths[_index] like normal; the rest of the window code doesn't care
        # how many curves are notionally in the list.
        self._paths = [path]
        self._index = 0
        self._n_total = 1
        # Position readout belongs to the navigator bar, which is subscribed to
        # the same signal — nothing to push from here.
        self._status_label.setText("")
        self.curve_changed.emit(0)
        self._do_draw(0)
        self._last_displayed_index = 0
        # ROI + Decomp windows follow the worker themselves (their own nav bars
        # subscribe to playhead_changed), so there is no push from here in worker
        # mode — that would double-load the same curve.

    def _on_worker_file_done(
        self, file_id: int, event: str, _was_cached: bool,
    ) -> None:
        """Draw only the current file's results after the worker persists them."""
        if int(file_id) != self._current_file_id:
            return
        try:
            available = self._draw_persisted_overlays(int(file_id), event=event)
        except Exception as exc:
            self._show_overlay_error(exc)
            return
        if available:
            self._status_label.setText("")
        else:
            self._status_label.setText("analysis finished — overlays unavailable")

    def _on_worker_file_error(self, file_id: int, message: str) -> None:
        if int(file_id) == self._current_file_id:
            self._status_label.setText(
                f"analysis failed — overlays unavailable: {str(message)[:160]}"
            )

    def _on_worker_data_unavailable(
        self, file_id: int, _path: str, detail: str,
    ) -> None:
        if int(file_id) == self._current_file_id:
            self._status_label.setText(
                f"raw data unavailable: {str(detail)[:160]}"
            )

    def _on_worker_queue_empty(self) -> None:
        """Worker auto-paused because the queue ran out in the current direction."""
        self._show_worker_hint("queue empty in this direction — send files from the dashboard")

    def _show_worker_hint(self, msg: str = "") -> None:
        if not msg:
            n = len(self._worker.queue_ids()) if self._worker is not None else 0
            msg = (f"{n} files in queue — press ▶▶ to start"
                   if n else "queue empty — send files from the dashboard")
        self._status_label.setText(msg)

    # ── Curve loading and drawing ──────────────────────────────────────────────

    def _do_draw(self, index: int) -> None:
        """Load one curve and update the plot + metadata panel."""
        path = self._paths[index]

        self._clear_markers()

        try:
            curve = load_force_curve(path)
        except UnusableCurveError:
            # Not a ramp — held, indented, or something not yet named.  It is
            # still a recording, so it is still viewable; only the analysis
            # overlays below have nothing to say about it.
            try:
                curve = load_raw_trace(path)
            except LoadError:
                self._show_load_failure(path)
                return
        except LoadError:
            self._show_load_failure(path)
            return

        self._draw(curve)
        if not isinstance(curve, ForceCurve):
            self._status_label.setText(
                f"{curve.curve_type.replace('_', ' ')} — viewing only"
            )
            return

        # Update spectral window if it is open.  Worker mode: the Decomp window
        # follows the worker itself (its nav bar subscribes to playhead_changed),
        # so pushing here too would double-load the curve — legacy mode only.
        if (self._worker is None
                and self._decomp_win is not None and self._decomp_win.isVisible()):
            self._decomp_win.update_curve(curve)

        # Update FFT window if it is open
        if self._fft_win is not None and self._fft_win.isVisible():
            self._fft_win.update_curve(curve)

        # Update ROI detection window if it is open (legacy mode only — in worker
        # mode the ROI window follows the worker via its own nav bar).
        if (self._worker is None
                and self._roi_win is not None and self._roi_win.isVisible()):
            self._roi_win.update_curve(index)

        if self._worker is None:
            # Compatibility for the retired standalone caller. Production
            # worker mode never computes scientific results on the GUI thread.
            self._draw_derived(index, path, curve)
        elif self._current_file_id is not None:
            try:
                available = self._draw_persisted_overlays(self._current_file_id)
            except Exception as exc:
                self._show_overlay_error(exc)
                return
            if available:
                self._status_label.setText("")
            else:
                self._status_label.setText("analysis in progress…")

    def _show_load_failure(self, path: str) -> None:
        self._n_errors += 1
        for item in (self._curve_appr, self._curve_retr, self._curve_raw):
            item.setData([], [])
        self._meta_vals["filename"].setText(f"✗ {Path(path).name}")
        self._meta_vals["directory"].setText(str(Path(path).parent))
        for key in (
            "date", "spring_k", "velocity", "trigger", "force_dist",
            "inv_ols", "xy", "sample_rate",
        ):
            self._meta_vals[key].setText("—")

    def _show_overlay_error(self, exc: Exception) -> None:
        self._status_label.setText(
            f"analysis overlays unavailable: {type(exc).__name__}: "
            f"{str(exc)[:120]}"
        )

    def _draw_persisted_overlays(
        self, file_id: int, *, event: str | None = None,
    ) -> bool:
        """Read and draw worker-produced landmarks without running analysis."""
        if self._axes != _AXES[0][1]:
            # Every landmark here is a piezo position.  On a time axis it would
            # land at a moment it was never measured at, so none are drawn and
            # there is nothing missing to report.
            return True

        from .curve_analysis import pipeline_params_from
        from .roi_pipeline import (
            event_map_params_json,
            event_params_from,
        )
        from .roi_events import payload_to_events

        param_set = _db.load_analysis_params(self._db_path)
        p = pipeline_params_from(param_set)
        code_ver = cache_version()
        if code_ver is None:
            return False

        verdict = _db.get_analysis_result(
            file_id, "event", p.all_params, code_ver, self._db_path,
        )
        if verdict is None:
            return False
        is_event = event == "event" if event is not None else verdict >= 0.5

        self._clear_markers()
        contact = _db.get_analysis_results_multi(
            file_id,
            ["contact_piezo_nm", "snapoff_piezo_nm"],
            p.params_cd,
            code_ver,
            self._db_path,
        )
        self._draw_contact_markers(
            contact.get("contact_piezo_nm"),
            contact.get("snapoff_piezo_nm"),
        )

        if is_event:
            ep = event_params_from(param_set)
            payload = _db.get_event_map(
                file_id,
                event_map_params_json(ep),
                code_ver,
                self._db_path,
            )
            if payload is None:
                return False
            events = payload_to_events(json.loads(payload))
            if events is None:
                return False
            self._draw_event_markers(events.rois)
        return True

    def _clear_markers(self) -> None:
        """Remove every analytical overlay belonging to the previous curve."""
        if self._contact_appr_line is not None:
            self._plot.removeItem(self._contact_appr_line)
            self._contact_appr_line = None
        if self._contact_retr_line is not None:
            self._plot.removeItem(self._contact_retr_line)
            self._contact_retr_line = None
        for line in self._rupture_lines:
            self._plot.removeItem(line)
        self._rupture_lines = []
        for line in self._onset_lines:
            self._plot.removeItem(line)
        self._onset_lines = []

    def _draw_contact_markers(
        self, contact_z: float | None, snapoff_z: float | None,
    ) -> None:
        if contact_z is not None and not np.isnan(contact_z):
            self._contact_appr_line = pg.InfiniteLine(
                pos=contact_z, angle=90, movable=False,
                pen=style.guide_pen(_COLOR_CONTACT),
                label="contact",
                labelOpts={"position": 0.95, "color": _COLOR_CONTACT},
            )
            self._plot.addItem(self._contact_appr_line)
        if snapoff_z is not None and not np.isnan(snapoff_z):
            self._contact_retr_line = pg.InfiniteLine(
                pos=snapoff_z, angle=90, movable=False,
                pen=style.guide_pen(_COLOR_SNAPOFF),
                label="snap-off",
                labelOpts={"position": 0.82, "color": _COLOR_SNAPOFF},
            )
            self._plot.addItem(self._contact_retr_line)

    def _draw_event_markers(self, rois) -> None:
        self._draw_event_marker_coords(
            (roi.ruptures[-1].piezo_nm, roi.onset_piezo_nm)
            for roi in rois
        )

    def _draw_event_marker_coords(self, coords) -> None:
        rupture_rgb = (40, 160, 40)
        onset_rgb = (220, 130, 0)
        for rupture_z, onset_z in coords:
            rup_line = pg.InfiniteLine(
                pos=rupture_z, angle=90, movable=False,
                pen=style.guide_pen(rupture_rgb),
                label="rupture",
                labelOpts={"position": 0.70, "color": rupture_rgb},
            )
            self._plot.addItem(rup_line)
            self._rupture_lines.append(rup_line)
            ons_line = pg.InfiniteLine(
                pos=onset_z, angle=90, movable=False,
                pen=style.guide_pen(onset_rgb),
                label="onset",
                labelOpts={"position": 0.58, "color": onset_rgb},
            )
            self._plot.addItem(ons_line)
            self._onset_lines.append(ons_line)

    def _draw_derived(self, index: int, path: str, curve: ForceCurve) -> None:
        """
        Run the shared analyse_curve routine, draw the contact/snap-off marker
        lines from its scalar result plus one rupture/onset marker pair per
        outer ROI (from the same event_map document the ROI/View Fits windows
        read), and (in legacy mode) emit derived_result_ready for
        AnalysisWindow / EventSummaryWindow.

        contact_z/snapoff_z are NaN for a non_event, so those two lines simply
        don't appear; a non_event likewise has no event_map document, so no
        rupture/onset lines are drawn — exactly what the DB would store.
        """
        from .curve_analysis import analyse_curve, pipeline_params_from

        try:
            # ONE rule for whose parameters apply: the file at position one of
            # the analysis queue (db.active_param_owner). Same answer here as
            # in the ROI window and the worker — there is no second way to
            # decide it, and no per-curve resolution.
            param_set = _db.load_analysis_params(self._db_path)
            p        = pipeline_params_from(param_set)
            code_ver = cache_version()
            file_id  = _db.get_file_id(path, self._db_path)
            result, _stage1 = analyse_curve(
                curve, p,
                db_path  = self._db_path,
                code_ver = code_ver,
                file_id  = file_id,
            )
        except Exception as exc:
            # Keep this terse and on stderr.  Some numpy/DB errors embed an entire
            # array in their message; traceback.print_exc() then floods the
            # terminal with raw bytes.  This path is usually benign — it fires
            # when a final draw races window teardown.
            import sys
            print(
                f"[rawcurve] derived computation skipped — "
                f"{type(exc).__name__}: {str(exc)[:160]}",
                file=sys.stderr,
            )
            self._status_label.setText(
                f"analysis overlays unavailable: {type(exc).__name__}: "
                f"{str(exc)[:120]}"
            )
            return

        # ── Marker lines ──────────────────────────────────────────────────────
        # Match ROIWindow colours: green rupture, orange onset.
        _RUPTURE_RGB = (40, 160, 40)
        _ONSET_RGB   = (220, 130, 0)

        if not np.isnan(result.contact_z):
            self._contact_appr_line = pg.InfiniteLine(
                pos=result.contact_z, angle=90, movable=False,
                pen=style.guide_pen(_COLOR_CONTACT),
                label='contact', labelOpts={'position': 0.95, 'color': _COLOR_CONTACT},
            )
            self._plot.addItem(self._contact_appr_line)

        if not np.isnan(result.snapoff_z):
            self._contact_retr_line = pg.InfiniteLine(
                pos=result.snapoff_z, angle=90, movable=False,
                pen=style.guide_pen(_COLOR_SNAPOFF),
                label='snap-off', labelOpts={'position': 0.82, 'color': _COLOR_SNAPOFF},
            )
            self._plot.addItem(self._contact_retr_line)

        # Rupture/onset: one pair per outer ROI, not result.rupture_z/onset_z's
        # single scalar — a curve can hold more than one outer ROI, and event_map carries the
        # full list. Reuses `stage1` from the analyse_curve call above, so this
        # is the same zero-recompute reuse `_persist_multi_event_roi` does — on
        # a hit curve the worker has typically already written this exact
        # document, so this is a cache hit, not fresh work. Only meaningful for
        # a validated event; a non_event has no event_map row (see
        # curve_analysis._persist_multi_event_roi / delete_event_map).
        if result.event:
            try:
                from .roi_pipeline import compute_curve_events_coords, event_params_from
                ep  = event_params_from(param_set)
                res = compute_curve_events_coords(
                    curve, ep, db_path=self._db_path, code_ver=code_ver,
                    file_id=file_id, stage1=_stage1, param_set=param_set,
                )
                for roi in res.events.rois:
                    rup_line = pg.InfiniteLine(
                        pos=roi.ruptures[-1].piezo_nm, angle=90, movable=False,
                        pen=style.guide_pen(_RUPTURE_RGB),
                        label='rupture', labelOpts={'position': 0.70, 'color': _RUPTURE_RGB},
                    )
                    self._plot.addItem(rup_line)
                    self._rupture_lines.append(rup_line)

                    ons_line = pg.InfiniteLine(
                        pos=roi.onset_piezo_nm, angle=90, movable=False,
                        pen=style.guide_pen(_ONSET_RGB),
                        label='onset', labelOpts={'position': 0.58, 'color': _ONSET_RGB},
                    )
                    self._plot.addItem(ons_line)
                    self._onset_lines.append(ons_line)
            except Exception as exc:
                import sys
                print(
                    f"[rawcurve] multi-ROI markers skipped — "
                    f"{type(exc).__name__}: {str(exc)[:160]}",
                    file=sys.stderr,
                )

        # ── Emit to AnalysisWindow / EventSummaryWindow (legacy mode only) ────
        if self._worker is None:
            self.derived_result_ready.emit(
                index, result.offset, result.flatness,
                result.contact_z, result.snapoff_z,
                result.rupture_z, result.onset_z,
                result.invols_slope, result.rupture_force,
                p.params_cd, p.params_bl, p.params_roi, p.params_invols,
            )

    # ── Axes ──────────────────────────────────────────────────────────────────

    def _on_axes_changed(self) -> None:
        self._axes = self._axis_box.currentData()
        if self._drawn is not None:
            self._plot_axes(self._drawn)
        # Markers and overlays are positioned in piezo, so they mean nothing on
        # a time axis.  Redrawing them there would put a landmark at a spot it
        # was never measured at.
        self._clear_markers()
        if (self._axes == _AXES[0][1] and self._worker is not None
                and self._current_file_id is not None
                and isinstance(self._drawn, ForceCurve)):
            try:
                self._draw_persisted_overlays(self._current_file_id)
            except Exception as exc:
                self._show_overlay_error(exc)

    @staticmethod
    def _ramp_series(curve: ForceCurve, kind: str):
        """(approach, retract) arrays of one quantity; (None, None) if absent."""
        if kind == "piezo":
            return curve.piezo_appr, curve.piezo_retr
        if kind == "defl":
            return curve.defl_appr, curve.defl_retr
        if not curve.sample_rate_hz:
            return None, None
        n_a, n_r = len(curve.defl_appr), len(curve.defl_retr)
        # The turnaround sample sits between the two halves and belongs to
        # neither, which is why the retract clock starts at n_a + 1.
        t = np.arange(n_a + n_r + 1, dtype=float) / curve.sample_rate_hz
        return t[:n_a], t[n_a + 1:]

    @staticmethod
    def _trace_series(trace: RawTrace, kind: str):
        return {"piezo": trace.piezo_nm,
                "defl":  trace.defl_nm,
                "time":  trace.time_s}.get(kind)

    def _plot_axes(self, obj) -> None:
        label, _key, xk, yk = next(a for a in _AXES if a[1] == self._axes)
        if isinstance(obj, ForceCurve):
            self._curve_raw.setData([], [])
            xa, xr = self._ramp_series(obj, xk)
            ya, yr = self._ramp_series(obj, yk)
            if xa is None or ya is None:
                self._curve_appr.setData([], [])
                self._curve_retr.setData([], [])
                self._status_label.setText(
                    f"{label}: this curve states no sample rate")
                return
            self._curve_appr.setData(xa, ya)
            self._curve_retr.setData(xr, yr)
        else:
            self._curve_appr.setData([], [])
            self._curve_retr.setData([], [])
            x, y = self._trace_series(obj, xk), self._trace_series(obj, yk)
            if x is None or y is None:
                self._curve_raw.setData([], [])
                missing = xk if x is None else yk
                self._status_label.setText(
                    f"{label}: this file has no {missing} channel")
                return
            self._curve_raw.setData(x, y)

        xn, xu = _AXIS_LABEL[xk]
        yn, yu = _AXIS_LABEL[yk]
        set_si_label(self._plot, "bottom", xn, xu)
        set_si_label(self._plot, "left",   yn, yu)

    def _draw(self, curve) -> None:
        self._drawn = curve
        self._plot_axes(curve)

        p = Path(curve.path)
        self._meta_vals["filename"].setText(p.name)
        self._meta_vals["directory"].setText(str(p.parent))
        self._meta_vals["date"].setText(curve.measured_date or "—")
        self._meta_vals["spring_k"].setText(
            _quant.format_value("spring_constant_pn_nm", curve.spring_constant,
                                with_unit=True) or "\N{EM DASH}"
        )
        self._meta_vals["velocity"].setText(
            _quant.format_value("velocity_nm_s", curve.velocity_nm_s,
                                with_unit=True) or "\N{EM DASH}"
        )
        self._meta_vals["trigger"].setText(
            _quant.format_value("trigger_point_nn", curve.trigger_point_nn,
                                with_unit=True) or "\N{EM DASH}"
        )
        self._meta_vals["force_dist"].setText(
            _quant.format_value("force_dist_nm", curve.force_dist_nm,
                                with_unit=True) or "\N{EM DASH}"
        )
        self._meta_vals["inv_ols"].setText(
            _quant.format_value("inv_ols_nm_v", curve.inv_ols_nm_v,
                                with_unit=True) or "\N{EM DASH}"
        )
        if curve.xpos is not None and curve.ypos is not None:
            self._meta_vals["xy"].setText(
                f"({_quant.format_value('xpos_um', curve.xpos)}, "
                f"{_quant.format_value('ypos_um', curve.ypos)}) µm"
            )
        else:
            self._meta_vals["xy"].setText("—")
        self._meta_vals["sample_rate"].setText(
            _quant.format_value("sample_rate_hz", curve.sample_rate_hz,
                                with_unit=True) or "\N{EM DASH}"
        )
