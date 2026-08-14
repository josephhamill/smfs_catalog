# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/navigator_bar.py
#
# Reusable control surfaces for the one global AnalysisWorker playhead.
#
# The worker is the playback engine (the video-player model): it owns the
# playhead, the play/pause state, the direction and the inter-file throttle.
# This widget owns NONE of that.  Every control writes to the worker; every
# readout is repainted from a worker signal.  Two NavigatorBars therefore never
# talk to each other and cannot drift — they are two windows onto one state, not
# two copies of it.
#
# Consequences worth stating, because they are the reason for this shape:
#
#   • Closing a window cannot reset anything.  The throttle a user chose lives on
#     the worker, which the dashboard creates once at launch and stops at quit.
#     A bar reads that value when it is built and NEVER writes a default into it
#     (#126: "the leak is intended — surface it, do not reset it").
#
#   • The scrubber's meaning changes as the queue changes — position 400 is a
#     different curve after a repopulate.  Handled in one place, refresh_queue(),
#     rather than once per window.
#
# NavigatorBar is the full transport used by the dashboard and raw viewer.
# WorkerNavBar is the compact Prev/Next/scrubber adapter used by decomposition
# and ROI diagnostics. Both navigate the ANALYSIS QUEUE. They are distinct from
# the window-local browsers in Events/WLC/Isoforce/class-list views, whose
# selection changes only their own cohort index.
#
# A worker navigation step processes the selected curve with the current
# parameters before its diagnostic views update. This is intentional WYSIWYG:
# analysis displayed in front of the user is current and persisted, never an
# unrecorded fit performed only for display.
#
# Queue order comes from worker.queue_ids(), which is cached and invalidated on
# membership change, so a bar costs no DB read per analysed file.  This matters:
# the dashboard's bar is open for the entire duration of every batch.

from __future__ import annotations

import math

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from . import db as _db
from . import style


# ── Speed slider math ─────────────────────────────────────────────────────────
# Positions 1–19: logarithmic spacing from RATE_MIN_HZ to RATE_MAX_HZ.
# Position 20   : no limit at all (throttle 0 / timer interval 0).
#
# Log spacing means every slider step multiplies the rate by the same factor
# (~×1.48 per step), so a click always feels like the same proportional change.
#
# Lives here rather than in rawcurve_window because three other windows
# (wlc_view_window, isoforce_window, class_lineplot_window) drive their OWN
# QTimer over their own curve list with the same slider and used to reach into
# rawcurve_window for these private helpers.  Those windows are not the worker's
# playhead and deliberately do not use NavigatorBar — they just share the maths.
SLIDER_MIN  = 1
SLIDER_MAX  = 20
RATE_MIN_HZ = 0.1    #  one curve every 10 s — slow inspection
RATE_MAX_HZ = 20.0   #  20 curves/s — above this the display timer throttles anyway

# Default starting playback rate for the timer-driven windows.  The worker-driven
# NavigatorBar does NOT use this: it reads the worker's live throttle instead.
DEFAULT_RATE_HZ = 1.0

_LOG_MIN = math.log10(RATE_MIN_HZ)
_LOG_MAX = math.log10(RATE_MAX_HZ)


def slider_to_rate_hz(value: int) -> float:
    """Slider position → curves per second.  Position 20 returns inf (no limit)."""
    if value >= SLIDER_MAX:
        return math.inf
    t = (value - SLIDER_MIN) / (SLIDER_MAX - 1 - SLIDER_MIN)   # 0.0 – 1.0
    return 10.0 ** (_LOG_MIN + t * (_LOG_MAX - _LOG_MIN))


def slider_to_interval_ms(value: int) -> int:
    """Slider position → QTimer interval in ms.  Position 20 returns 0."""
    if value >= SLIDER_MAX:
        return 0
    return int(1000.0 / slider_to_rate_hz(value))


def rate_to_slider(rate_hz: float) -> int:
    """Nearest slider position for a given rate in Hz."""
    rate_hz = max(RATE_MIN_HZ, min(RATE_MAX_HZ, rate_hz))
    t = (math.log10(rate_hz) - _LOG_MIN) / (_LOG_MAX - _LOG_MIN)
    raw = int(round(SLIDER_MIN + t * (SLIDER_MAX - 1 - SLIDER_MIN)))
    return max(SLIDER_MIN, min(SLIDER_MAX - 1, raw))


def rate_label(value: int) -> str:
    """Human-readable rate string for the speed display label."""
    if value >= SLIDER_MAX:
        return "max"
    hz = slider_to_rate_hz(value)
    if hz >= 1.0:
        return f"{hz:.1f} /s"
    return f"{1.0/hz:.1f} s/curve"


# ── Throttle ↔ slider ─────────────────────────────────────────────────────────
# The worker's knob is an inter-file delay in ms, not a rate, so the two ends of
# the slider invert: slider max = throttle 0 = churn at full speed.  Both
# directions live here so the dashboard and the viewer cannot map them
# differently (they did: the dashboard read throttle 0 back as "20 /s", one notch
# short of the "max" the worker was actually running at).

def slider_to_throttle_ms(value: int) -> int:
    """Slider position → worker inter-file delay in ms.  Position 20 → 0."""
    rate = slider_to_rate_hz(value)
    if not math.isfinite(rate) or rate <= 0:
        return 0
    return int(round(1000.0 / rate))


def throttle_to_slider(ms: int) -> int:
    """Worker inter-file delay in ms → slider position.  0 (no delay) → 20."""
    if ms <= 0:
        return SLIDER_MAX
    return rate_to_slider(1000.0 / ms)


def throttle_label(ms: int) -> str:
    """What the current throttle means, in the units a user asked for it in.

    Phrased as a ceiling ('≤ 5.0 /s') because the delay caps the rate rather
    than setting it — the real rate is 1 / (compute time + delay).
    """
    if ms <= 0:
        return "no limit"
    return f"≤ {rate_label(throttle_to_slider(ms))}"


# ── The bar ───────────────────────────────────────────────────────────────────

class NavigatorBar(QWidget):
    """
    Transport controls for a running AnalysisWorker: step/play in either
    direction, the rate limit, and a position scrubber over the queue.

    Holds no state.  Construct one per window that needs to drive or watch the
    worker; they stay in sync through the worker's own signals.

    Owners must call detach() when their window closes and attach() when it
    reopens — Qt's close() only HIDES a window, and a hidden bar left subscribed
    keeps doing per-file work nobody can see.
    """

    def __init__(self, worker, parent: QWidget | None = None,
                 compact: bool = False) -> None:
        super().__init__(parent)
        self._worker = worker
        self._queue_ids: list[int] = []
        self._linked = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self._btn_rev = QPushButton("◀◀")
        self._btn_rev.setFixedWidth(44)
        self._btn_rev.setCheckable(True)
        self._btn_rev.setToolTip("Play backwards through the queue.")
        self._btn_rev.clicked.connect(lambda on: self._on_auto(-1, on))

        self._btn_prev = QPushButton("◀ Prev")
        self._btn_prev.setToolTip("Step back one file.  Works while paused.")
        self._btn_prev.clicked.connect(lambda: self._worker.step_relative(-1))

        self._btn_pause = QPushButton("⏸ Pause")
        self._btn_pause.setToolTip("Stop playback in either direction.")
        self._btn_pause.clicked.connect(lambda: self._worker.set_paused(True))

        self._btn_next = QPushButton("Next ▶")
        self._btn_next.setToolTip("Step forward one file.  Works while paused.")
        self._btn_next.clicked.connect(lambda: self._worker.step_relative(+1))

        self._btn_fwd = QPushButton("▶▶")
        self._btn_fwd.setFixedWidth(44)
        self._btn_fwd.setCheckable(True)
        self._btn_fwd.setToolTip("Play forwards through the queue — runs the analysis.")
        self._btn_fwd.clicked.connect(lambda on: self._on_auto(+1, on))

        for w in (self._btn_rev, self._btn_prev, self._btn_pause,
                  self._btn_next, self._btn_fwd):
            row.addWidget(w)

        row.addSpacing(10)

        # Rate limit.  Shared with the batch worker BY DESIGN (#126): this is one
        # playback engine, so a speed the user chose here holds until they change
        # it — closing a window never resets it.  It is shown wherever its effect
        # is felt so it can never throttle a batch invisibly again.
        _lbl_slow = QLabel("Slow")
        _lbl_slow.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(_lbl_slow)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(SLIDER_MIN, SLIDER_MAX)
        self._slider.setFixedWidth(120 if compact else 140)
        self._slider.setToolTip(
            "Rate limit for playback AND batch analysis — the same worker runs "
            "both.\nSlide to the far right for no limit (full speed).\n"
            "This setting is deliberately kept when a window closes; it resets "
            "only when the app restarts."
        )
        self._slider.valueChanged.connect(self._on_slider_moved)
        row.addWidget(self._slider)

        row.addWidget(QLabel("Fast"))

        self._rate_label = QLabel("")
        self._rate_label.setMinimumWidth(88)
        self._rate_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self._rate_label)

        self._btn_clear = QPushButton("Clear limit")
        self._btn_clear.setToolTip(
            "Remove the rate limit and let the worker run at full speed.")
        self._btn_clear.clicked.connect(lambda: self._worker.set_throttle_ms(0))
        row.addWidget(self._btn_clear)

        row.addStretch(1)

        self._pos_label = QLabel("")
        self._pos_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self._pos_label)

        root.addLayout(row)

        # Position scrubber — drag to jump anywhere in the queue, to the end, or
        # back to replay a section.  step_to runs even while paused, so scrubbing
        # works whether or not playback is on.  Tracking off: a drag jumps only on
        # release, so dragging across thousands of files doesn't enqueue thousands
        # of per-curve loads.
        self._scrubber = QSlider(Qt.Orientation.Horizontal)
        self._scrubber.setTracking(False)
        self._scrubber.setToolTip(
            "Where the worker is in the queue.  Drag to jump; release to go.")
        self._scrubber.valueChanged.connect(self._on_scrubber_moved)
        root.addWidget(self._scrubber)

        # Paint from the worker's CURRENT state — never push a default into it.
        self.refresh_queue()
        self._sync_throttle(self._worker.throttle_ms())
        self._sync_transport(self._worker.is_paused())
        self.attach()

    # ── Subscription ─────────────────────────────────────────────────────────

    def attach(self) -> None:
        """Subscribe to the worker.  Guarded so a re-show can't double-connect."""
        if self._linked:
            return
        for signal, slot in self._links():
            signal.connect(slot)
        self._linked = True
        # The queue and playhead may have moved while we were detached.
        self.refresh_queue()
        self._sync_transport(self._worker.is_paused())

    def detach(self) -> None:
        """Unsubscribe.  A closed window must cost nothing per analysed file."""
        if not self._linked:
            return
        for signal, slot in self._links():
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass          # already disconnected; nothing to undo
        self._linked = False

    def _links(self) -> list[tuple]:
        """Every subscription, in one list, so attach and detach cannot drift."""
        return [
            (self._worker.playhead_changed,  self._on_playhead),
            (self._worker.paused_changed,    self._on_paused),
            (self._worker.direction_changed, self._on_direction),
            (self._worker.throttle_changed,  self._sync_throttle),
            (self._worker.queue_changed,     self.refresh_queue),
        ]

    # ── Queue order ──────────────────────────────────────────────────────────

    def refresh_queue(self) -> None:
        """
        Re-read queue order and re-range the scrubber.

        Call after any change to queue membership — the scrubber's positions mean
        different curves afterwards.  Served from the worker's cached id-list, so
        this is not a DB read in the common case.
        """
        self._queue_ids = list(self._worker.queue_ids())
        self._scrubber.blockSignals(True)
        self._scrubber.setRange(0, max(0, len(self._queue_ids) - 1))
        self._scrubber.blockSignals(False)
        self._scrubber.setEnabled(bool(self._queue_ids))
        self._sync_position(self._worker.playhead())

    # ── Worker → widgets ─────────────────────────────────────────────────────

    def _sync_position(self, file_id: int | None) -> None:
        total = len(self._queue_ids)
        if not total:
            self._pos_label.setText("queue empty")
            return
        idx = None
        if file_id is not None:
            try:
                idx = self._queue_ids.index(int(file_id))
            except ValueError:
                idx = None
        if idx is None:
            self._pos_label.setText(f"— / {total:,}")
            return
        self._scrubber.blockSignals(True)
        self._scrubber.setValue(idx)
        self._scrubber.blockSignals(False)
        self._pos_label.setText(f"{idx + 1:,} / {total:,}")

    def _sync_transport(self, paused: bool) -> None:
        forward = self._worker.direction() >= 0
        for btn, on in ((self._btn_fwd, not paused and forward),
                        (self._btn_rev, not paused and not forward)):
            btn.blockSignals(True)
            btn.setChecked(on)
            btn.blockSignals(False)

    def _sync_throttle(self, ms: int) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(throttle_to_slider(int(ms)))
        self._slider.blockSignals(False)
        self._rate_label.setText(throttle_label(int(ms)))
        self._btn_clear.setEnabled(int(ms) > 0)

    def _on_playhead(self, file_id: int) -> None:
        # A playhead outside the known id-list means membership changed under us
        # (e.g. a transient enqueue for "go to this curve") — re-read once.
        if int(file_id) not in self._queue_ids:
            self.refresh_queue()
        else:
            self._sync_position(file_id)

    def _on_paused(self, paused: bool) -> None:
        self._sync_transport(paused)

    def _on_direction(self, d: int) -> None:
        self._sync_transport(self._worker.is_paused())

    # ── Widgets → worker ─────────────────────────────────────────────────────

    def _on_auto(self, direction: int, checked: bool) -> None:
        if not checked:
            self._worker.set_paused(True)
            return
        self._worker.set_direction(direction)
        self._worker.set_paused(False)
        self._worker.notify_work_available()

    def _on_slider_moved(self, value: int) -> None:
        self._worker.set_throttle_ms(slider_to_throttle_ms(value))

    def _on_scrubber_moved(self, value: int) -> None:
        if 0 <= value < len(self._queue_ids):
            self._worker.step_to(self._queue_ids[value])


class WorkerNavBar(QWidget):
    """Compact analysis-queue navigator for worker-driven diagnostics.

    Like :class:`NavigatorBar`, this drives and follows the one global worker
    playhead. It is not a navigator for a window-local event, hit, non-event, or
    fit list; those local browsers own their own indices.

    A worker step intentionally analyzes the selected curve using the current
    parameters before the diagnostic displays update, preserving the WYSIWYG
    guarantee that displayed analysis is current and persisted.

    This compact variant omits playback and throttle controls. It also resolves
    the worker's file ID and emits ``curve_selected(path, file_id)`` for its
    owning diagnostic window. Hidden windows defer that lookup and redraw until
    :meth:`sync_now` is called from their ``showEvent``.
    """

    curve_selected = pyqtSignal(str, int)   # (path, file_id)

    def __init__(self, worker, db_path: str, parent=None) -> None:
        super().__init__(parent)
        self._worker      = worker
        self._db_path     = db_path
        self._queue_ids: list[int] = []
        self._file_id: int | None = None
        self._path:    str | None = None

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self._btn_prev = QPushButton("◀ Prev")
        self._btn_prev.setFixedWidth(72)
        self._btn_prev.clicked.connect(self._go_prev)
        row.addWidget(self._btn_prev)

        self._btn_next = QPushButton("Next ▶")
        self._btn_next.setFixedWidth(72)
        self._btn_next.clicked.connect(self._go_next)
        row.addWidget(self._btn_next)

        self._scrubber = QSlider(Qt.Orientation.Horizontal)
        self._scrubber.setRange(0, 0)
        self._scrubber.setTracking(False)   # jump on release, not on every drag step
        self._scrubber.valueChanged.connect(self._on_scrubber_moved)
        row.addWidget(self._scrubber, stretch=1)

        self._label = QLabel("— / —")
        self._label.setStyleSheet(style.qss_text(size_px=11))
        row.addWidget(self._label)

        worker.playhead_changed.connect(self._on_playhead)
        worker.queue_changed.connect(self._on_queue_changed)
        self._sync_scrubber(worker.playhead())

    # ── Accessors ─────────────────────────────────────────────────────────────

    def current_path(self) -> str | None:
        return self._path

    # ── Driving the worker ────────────────────────────────────────────────────

    def _go_prev(self) -> None:
        self._worker.step_relative(-1)

    def _go_next(self) -> None:
        self._worker.step_relative(+1)

    def _on_scrubber_moved(self, value: int) -> None:
        # step_to runs even while the worker is paused, so scrubbing works
        # whether or not autoplay is running.
        if 0 <= value < len(self._queue_ids):
            self._worker.step_to(self._queue_ids[value])

    # ── Following the worker ──────────────────────────────────────────────────

    def sync_now(self) -> None:
        """Redraw the current playhead — called by the owning window on show so a
        window hidden while the playhead moved catches up immediately."""
        fid = self._worker.playhead()
        if fid is not None:
            self._apply_playhead(int(fid))

    def _on_playhead(self, file_id: int) -> None:
        # Skip all per-curve work while the owning window is hidden; sync_now()
        # replays the current playhead when it is shown again.
        if not self.isVisible():
            self._file_id = int(file_id)
            self._path = None
            return
        self._apply_playhead(int(file_id))

    def _on_queue_changed(self) -> None:
        """Re-range against the worker's invalidated queue cache."""
        self._sync_scrubber(self._worker.playhead())

    def _apply_playhead(self, file_id: int) -> None:
        self._file_id = file_id
        self._path = _db.get_path(file_id, self._db_path)
        self._sync_scrubber(file_id)
        if self._path:
            self.curve_selected.emit(self._path, file_id)

    def _sync_scrubber(self, file_id: int | None) -> None:
        # Share the worker's generation-aware cache. Querying list_queue() here
        # made every visible diagnostic perform a full queue JOIN per curve.
        self._queue_ids = list(self._worker.queue_ids())
        total = len(self._queue_ids)
        self._scrubber.blockSignals(True)
        self._scrubber.setRange(0, max(0, total - 1))
        self._scrubber.setEnabled(bool(total))
        pos: int | None = None
        if file_id is not None:
            try:
                idx = self._queue_ids.index(int(file_id))
                self._scrubber.setValue(idx)
                pos = idx + 1
            except ValueError:
                pass
        self._scrubber.blockSignals(False)
        self._label.setText(f"{pos} / {total}" if pos is not None else f"— / {total}")
