# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from scipy.signal import sosfiltfilt, tf2sos

from . import style
from . import quantities as _quant
from .qt_utils import (
    fit_on_screen,
    _make_session_header,
    set_si_label,
)
from .curve_loader import ForceCurve

_SMALL_FONT_PT = style.FONT_CAPTION_PT
_COLOR_DEFL = style.SIG_DEFL
_COLOR_PIEZO = style.SIG_PIEZO
_COLOR_DIVIDER = style._COLOR_DIVIDER
_COLOR_ROI_FILL_RGBA = style._COLOR_ROI_FILL_RGBA

# ── FFT inspector window ───────────────────────────────────────────────────────

class FftWindow(QWidget):
    """
    Popup FFT inspector for a single SMFS force curve.

    Four vertically stacked plots:
      1. Raw deflection (nm)      — time domain, sample index x-axis
      2. Read piezo (nm)          — time domain, same x-axis as panel 1
      3. FFT magnitude — defl     — frequency domain (Hz)
      4. FFT magnitude — piezo    — frequency domain, same x-axis as panel 3

    A LinearRegionItem (ROI) spans panels 1 and 2 (linked x-axes).  Drag either
    edge or the whole shaded region to select a segment; panels 3 and 4 update
    when the drag is released.  A Hann window is applied before the FFT to
    reduce spectral leakage.  DC (0 Hz) is excluded from the FFT display.

    A grey vertical line marks the approach/retract turnaround in panels 1 and 2.

    The frequency and magnitude axes default to logarithmic and can be toggled
    linear.
    """

    def __init__(self, session_info: dict | None = None) -> None:
        super().__init__()
        self._curve  = None
        self._log_axes = True
        self._updating_roi = False   # re-entrancy guard for ROI sync

        self.setWindowTitle("SMFS — FFT inspector")
        self.setWindowFlag(Qt.WindowType.Window)
        fit_on_screen(self, 1000, 820)
        root = QVBoxLayout(self)
        root.setSpacing(2)
        root.setContentsMargins(6, 6, 6, 6)

        _hdr = _make_session_header(session_info)
        if _hdr is not None:
            root.addWidget(_hdr)

        # ── Control bar ───────────────────────────────────────────────────────
        ctrl = QWidget()
        ctrl_layout = QHBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(0, 2, 0, 2)
        ctrl_layout.setSpacing(8)

        self._btn_log = QPushButton("Log axes")
        self._btn_log.setCheckable(True)
        self._btn_log.setChecked(True)
        self._btn_log.setFixedWidth(72)
        self._btn_log.clicked.connect(self._on_log_toggle)
        ctrl_layout.addWidget(self._btn_log)

        # ── Notch controls — preview-only bandstop on defl (not persisted) ───
        ctrl_layout.addSpacing(12)
        self._chk_notch = QCheckBox("Notch")
        self._chk_notch.setChecked(False)
        self._chk_notch.toggled.connect(self._on_notch_toggled)
        ctrl_layout.addWidget(self._chk_notch)

        ctrl_layout.addWidget(QLabel("f₀ (Hz):"))
        self._spin_f0 = QDoubleSpinBox()
        self._spin_f0.setRange(1.0, 100000.0)
        _quant.configure_spinbox(self._spin_f0, "notch_f0_hz", suffix=False)
        self._spin_f0.setValue(130.0)
        self._spin_f0.valueChanged.connect(self._on_notch_param_changed)
        ctrl_layout.addWidget(self._spin_f0)

        ctrl_layout.addWidget(QLabel("BW (Hz):"))
        self._spin_bw = QDoubleSpinBox()
        self._spin_bw.setRange(0.5, 10000.0)
        _quant.configure_spinbox(self._spin_bw, "notch_bw_hz", suffix=False)
        self._spin_bw.setValue(10.0)
        self._spin_bw.valueChanged.connect(self._on_notch_param_changed)
        ctrl_layout.addWidget(self._spin_bw)

        ctrl_layout.addWidget(QLabel("depth (dB):"))
        self._spin_depth = QDoubleSpinBox()
        self._spin_depth.setRange(0.0, 80.0)
        _quant.configure_spinbox(self._spin_depth, "notch_depth_db", suffix=False)
        self._spin_depth.setValue(20.0)
        self._spin_depth.valueChanged.connect(self._on_notch_param_changed)
        ctrl_layout.addWidget(self._spin_depth)

        self._roi_label = QLabel("")
        self._roi_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        font = self._roi_label.font()
        font.setPointSize(_SMALL_FONT_PT)
        self._roi_label.setFont(font)
        ctrl_layout.addWidget(self._roi_label)

        ctrl_layout.addStretch()
        root.addWidget(ctrl)

        # ── GraphicsLayoutWidget — 4 stacked panels ───────────────────────────
        _glw = pg.GraphicsLayoutWidget()
        _glw.setBackground(style.SURFACE)

        self._plt_defl  = _glw.addPlot(row=0, col=0)
        self._plt_piezo = _glw.addPlot(row=1, col=0)
        self._plt_fft_d = _glw.addPlot(row=2, col=0)
        self._plt_fft_p = _glw.addPlot(row=3, col=0)

        for pi, ylabel, yunits in (
            (self._plt_defl,  "Deflection",   _quant.NM),
            (self._plt_piezo, "Piezo (read)", _quant.NM),
            (self._plt_fft_d, "FFT defl",     _quant.NM),
            (self._plt_fft_p, "FFT piezo",    _quant.NM),
        ):
            # The two FFT panels default to logarithmic frequency and magnitude
            # axes (controlled together by the button above).
            # pyqtgraph 0.14 applies the axis scale inside logTickStrings
            # too, so the prefix and the tick values stay consistent there.
            set_si_label(pi, "left", ylabel, yunits)
            pi.showGrid(x=True, y=True, alpha=0.2)

        # Hide x-tick labels on top three panels; only the bottom needs them
        self._plt_defl.getAxis('bottom').setStyle(showValues=False)
        self._plt_piezo.getAxis('bottom').setStyle(showValues=False)
        self._plt_fft_d.getAxis('bottom').setStyle(showValues=False)

        # Shared x-axes: time panels together, FFT panels together
        self._plt_piezo.setXLink(self._plt_defl)
        self._plt_fft_p.setXLink(self._plt_fft_d)
        self._plt_fft_d.setLogMode(x=True, y=True)
        self._plt_fft_p.setLogMode(x=True, y=True)

        self._plt_piezo.setLabel("bottom", "Sample index")
        set_si_label(self._plt_fft_p, "bottom", "Frequency", _quant.HZ)

        root.addWidget(_glw, stretch=1)

        # ── Data curves ───────────────────────────────────────────────────────
        _pen_d = style.data_pen(_COLOR_DEFL, width=1.0)   # blue — deflection
        _pen_p = style.data_pen(_COLOR_PIEZO, width=1.0)  # orange — piezo

        self._trace_defl  = self._plt_defl.plot([], [], pen=_pen_d)
        self._trace_piezo = self._plt_piezo.plot([], [], pen=_pen_p)
        self._fft_defl    = self._plt_fft_d.plot([], [], pen=_pen_d)
        self._fft_piezo   = self._plt_fft_p.plot([], [], pen=_pen_p)

        # Filtered overlays — deflection only (analysis uses written piezo, so
        # filtering the read-piezo trace is decorative).  Initially hidden.
        _pen_df = style.data_pen(style.SIG_FILTERED, width=1.0)   # violet — filtered defl
        self._trace_defl_f = self._plt_defl.plot([],  [], pen=_pen_df)
        self._fft_defl_f   = self._plt_fft_d.plot([], [], pen=_pen_df)
        for item in (self._trace_defl_f, self._fft_defl_f):
            item.setVisible(False)

        # ── Approach/retract turnaround marker (time-domain panels) ───────────
        _div_pen = pg.mkPen(_COLOR_DIVIDER, width=1)
        self._div_defl  = pg.InfiniteLine(pos=0, angle=90, movable=False, pen=_div_pen)
        self._div_piezo = pg.InfiniteLine(pos=0, angle=90, movable=False, pen=_div_pen)
        self._plt_defl.addItem(self._div_defl)
        self._plt_piezo.addItem(self._div_piezo)

        # ── ROI — LinearRegionItem on both time-domain panels (kept in sync) ──
        _roi_brush = pg.mkBrush(*_COLOR_ROI_FILL_RGBA)
        self._roi_defl  = pg.LinearRegionItem(brush=_roi_brush, movable=True)
        self._roi_piezo = pg.LinearRegionItem(brush=_roi_brush, movable=True)
        self._plt_defl.addItem(self._roi_defl)
        self._plt_piezo.addItem(self._roi_piezo)

        self._roi_defl.sigRegionChanged.connect(self._sync_roi_from_defl)
        self._roi_piezo.sigRegionChanged.connect(self._sync_roi_from_piezo)
        self._roi_defl.sigRegionChangeFinished.connect(self._update_fft)
        self._roi_piezo.sigRegionChangeFinished.connect(self._update_fft)

    # ── ROI sync ──────────────────────────────────────────────────────────────

    def _sync_roi_from_defl(self) -> None:
        if self._updating_roi:
            return
        self._updating_roi = True
        self._roi_piezo.setRegion(self._roi_defl.getRegion())
        self._updating_roi = False

    def _sync_roi_from_piezo(self) -> None:
        if self._updating_roi:
            return
        self._updating_roi = True
        self._roi_defl.setRegion(self._roi_piezo.getRegion())
        self._updating_roi = False

    # ── Log-axis toggle ───────────────────────────────────────────────────────

    def _on_log_toggle(self, checked: bool) -> None:
        self._log_axes = checked
        self._plt_fft_d.setLogMode(x=checked, y=checked)
        self._plt_fft_p.setLogMode(x=checked, y=checked)

    # ── Notch controls ────────────────────────────────────────────────────────

    def _on_notch_toggled(self, checked: bool) -> None:
        for item in (self._trace_defl_f, self._fft_defl_f):
            item.setVisible(checked)
        self._update_fft()

    def _on_notch_param_changed(self, _value: float) -> None:
        if self._chk_notch.isChecked():
            self._update_fft()

    def _notch_sos(self, fs: float) -> np.ndarray | None:
        """
        Build a peaking-EQ biquad (Audio-EQ Cookbook) configured as a finite-depth
        notch.  Returns SOS coefficients, or None if the params are not realisable
        at the current sample rate.

        Parameters map as:
          f0 (Hz)    — centre frequency
          BW (Hz)    — nominal bandwidth parameter (Q = f0 / BW)
          depth (dB) — attenuation at f0 (positive value; internally negated)

        Note: sosfiltfilt applies the filter twice (forward + reverse) for zero
        phase, which squares the magnitude response.  We halve the configured
        depth inside the biquad so the post-filtfilt response matches the label.
        """
        f0    = float(self._spin_f0.value())
        bw    = max(0.5, float(self._spin_bw.value()))
        depth = float(self._spin_depth.value())
        if fs <= 0 or f0 <= 0 or f0 >= fs / 2:
            return None
        Q    = f0 / bw
        # Halve depth because filtfilt doubles it (two passes).
        # Standard A = 10^(gain_db/40); halving gain → divide exponent by 2 → /80.
        A    = 10.0 ** (-depth / 80.0)           # negative gain → dip
        w0   = 2.0 * np.pi * f0 / fs
        alpha = np.sin(w0) / (2.0 * Q)
        cosw0 = np.cos(w0)

        b0 = 1 + alpha * A
        b1 = -2 * cosw0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cosw0
        a2 = 1 - alpha / A

        b = np.array([b0, b1, b2]) / a0
        a = np.array([1.0, a1 / a0, a2 / a0])
        return tf2sos(b, a)

    # ── Public API ────────────────────────────────────────────────────────────

    def update_curve(self, curve: "ForceCurve") -> None:
        """Load a new curve: populate time-domain panels, preserve ROI if possible.

        On the first curve the ROI is initialised to the full trace.
        On subsequent curves the existing ROI bounds are kept and clamped to the
        new curve's length — so two consecutive curves can be compared with the
        same frequency window.  If clamping produces a degenerate region (e.g.
        the new curve is shorter than the ROI start) it falls back to full trace.
        """
        is_first = self._curve is None
        if not is_first:
            old_r0, old_r1 = self._roi_defl.getRegion()

        self._curve = curve
        if (
            curve is None
            or curve.raw_defl is None
            or curve.raw_piezo_read is None
            or curve.sample_rate_hz <= 0
        ):
            self._clear()
            return

        n_total = len(curve.raw_defl)
        idx     = np.arange(n_total)

        self._trace_defl.setData(idx, curve.raw_defl)
        self._trace_piezo.setData(idx, curve.raw_piezo_read)

        # Filtered full-length defl populated in _update_fft (depends on fs).
        self._trace_defl_f.setData([], [])

        self._div_defl.setValue(curve.idx_turn - 0.5)
        self._div_piezo.setValue(curve.idx_turn - 0.5)

        # Determine ROI bounds for the new curve
        if is_first:
            new_r0, new_r1 = 0, n_total
        else:
            new_r0 = max(0, min(int(old_r0), n_total - 1))
            new_r1 = max(new_r0 + 1, min(int(old_r1), n_total))
            if new_r0 >= new_r1:           # degenerate — fall back to full
                new_r0, new_r1 = 0, n_total

        self._roi_defl.blockSignals(True)
        self._roi_piezo.blockSignals(True)
        self._roi_defl.setRegion((new_r0, new_r1))
        self._roi_piezo.setRegion((new_r0, new_r1))
        self._roi_defl.blockSignals(False)
        self._roi_piezo.blockSignals(False)

        self._update_fft()

    # ── FFT computation ───────────────────────────────────────────────────────

    def _update_fft(self) -> None:
        """Recompute FFT of the selected ROI and refresh panels 3 and 4."""
        curve = self._curve
        if curve is None or curve.raw_defl is None:
            return

        r0, r1 = self._roi_defl.getRegion()
        i0 = max(0, int(round(r0)))
        i1 = min(len(curve.raw_defl), int(round(r1)))
        n  = i1 - i0

        sr = curve.sample_rate_hz
        dur_ms = n / sr * 1000
        self._roi_label.setText(
            f"ROI: samples {i0}–{i1}  ({n} pts, {dur_ms:.1f} ms)"
        )

        if n < 8:
            self._fft_defl.setData([], [])
            self._fft_piezo.setData([], [])
            return

        seg_d = curve.raw_defl[i0:i1]
        seg_p = curve.raw_piezo_read[i0:i1]

        window = np.hanning(n)
        freqs  = np.fft.rfftfreq(n, d=1.0 / sr)

        mag_d = np.abs(np.fft.rfft(seg_d * window))
        mag_p = np.abs(np.fft.rfft(seg_p * window))

        # Skip DC: zero cannot be represented on the default logarithmic x-axis.
        self._fft_defl.setData(freqs[1:], mag_d[1:])
        self._fft_piezo.setData(freqs[1:], mag_p[1:])

        # ── Notch preview ────────────────────────────────────────────────────
        # Deflection only — analysis uses the ideal written piezo, not the
        # sensor read-piezo shown here, so filtering the piezo panel is
        # decorative.  The hardware is the right fix for the piezo side.
        if self._chk_notch.isChecked():
            sos = self._notch_sos(sr)
            if sos is None:
                self._trace_defl_f.setData([], [])
                self._fft_defl_f.setData([], [])
                return
            try:
                defl_f = sosfiltfilt(sos, curve.raw_defl)
            except (ValueError, FloatingPointError):
                self._trace_defl_f.setData([], [])
                self._fft_defl_f.setData([], [])
                return
            idx = np.arange(len(curve.raw_defl))
            self._trace_defl_f.setData(idx, defl_f)

            seg_df = defl_f[i0:i1]
            mag_df = np.abs(np.fft.rfft(seg_df * window))
            self._fft_defl_f.setData(freqs[1:], mag_df[1:])

    def _clear(self) -> None:
        for item in (self._trace_defl, self._trace_piezo,
                     self._fft_defl, self._fft_piezo,
                     self._trace_defl_f, self._fft_defl_f):
            item.setData([], [])
        self._roi_label.setText("")

