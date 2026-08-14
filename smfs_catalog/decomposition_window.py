# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

import os
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QSpinBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .curve_loader import ForceCurve, load_force_curve
from . import db as _db
from . import quantities as _quant
# Top-level, unlike this module's other signal_processing uses: the verdict is a
# pure comparison with no Qt or db dependency, and it is needed while building
# the UI, before any curve has been loaded.
from . import signal_processing as _sp
from .bandwidth_warning import filter_bandwidth_warning
from .quantities import format_value as _q   # ONE formatter: unit and
# meaningful digits come from quantities.py, so the same measurement
# cannot print as 166, 166.2 and 166.20 in three different windows.
from .widgets import FlowLayout, LabeledControl
from .navigator_bar import WorkerNavBar
from .provenance import cache_version
from . import style
from .qt_utils import (
    _make_session_header,
    fit_on_screen,
    set_si_label,
    shrinkable,
)

_COLOR_CONTACT = style.LM_CONTACT
_COLOR_SNAPOFF = style.LM_SNAPOFF
_COLOR_DIVIDER = style._COLOR_DIVIDER
_COLOR_MUTED = style._COLOR_MUTED
_SMALL_FONT_PT = style.FONT_CAPTION_PT

# ── Spectral cutoff slider values ─────────────────────────────────────────────
# Discrete cutoff frequencies (Hz).  Slider position = index into this list, so
# it MUST stay sorted ascending — a slider that runs backwards is not a slider.
#
# Low cutoffs remain selectable because over-smoothing is a scientific choice
# to report, not an invalid computation. Nyquist is a mathematical constraint
# and is enforced by _refresh_cutoff_limits.
_CUTOFF_VALUES: list[int] = [100, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000]

# Detection-threshold display and storage conversions are defined by
# quantities.py so controls, plots, profiles, and exports use the same units.
_THRESH_KEYS = ("detection_threshold_appr", "detection_threshold_retr")


def _thresh_unit() -> str:
    """The unit these thresholds are SHOWN in — the boxes and the axis agree."""
    return _quant.get(_THRESH_KEYS[0]).shown_unit


def _to_shown(stored: float) -> float:
    return _quant.get(_THRESH_KEYS[0]).to_display(stored)


def _to_stored(shown: float) -> float:
    return _quant.get(_THRESH_KEYS[0]).to_stored(shown)


def _seed_threshold_box(spin, key: str, stored_nm2: float) -> None:
    """Put a stored threshold into its box WITHOUT rounding it away.

    QDoubleSpinBox.value() returns its value rounded to the box's decimals,
    so seeding a box declared at 6 decimals with a stored 0.001623588188...
    displays 0.001624 — and the next arrow press writes that back over a real
    analysis parameter. decimals_for widens the box just enough to show
    what is actually stored; audit_stored_precision is how such values get
    tidied, deliberately, rather than by a display quietly rounding them off.
    without changing it through display rounding.
    """
    shown = _to_shown(stored_nm2)
    _quant.configure_spinbox(spin, key, decimals=_quant.decimals_for(key, shown),
                             suffix=False)
    spin.blockSignals(True)
    spin.setValue(shown)
    spin.blockSignals(False)


# ── Decomposition window ──────────────────────────────────────────────────────

class DecompositionWindow(QWidget):
    """
    Popup showing spectral decomposition of the current SMFS curve.

    Three vertically stacked plots, all sharing the same X axis (sample index):
      1. Low-frequency channel  (nm)  — approach red, retract blue
      2. High-frequency channel (nm)  — approach red, retract blue
      3. Moving variance of high channel (nm²) — approach red, retract blue

    A grey vertical line marks the approach / retract boundary in each plot.
    Sample index runs continuously: 0…N_appr-1 then N_appr…N_appr+N_retr-1,
    so approach and retract lay out side-by-side like an open book.

    A cutoff-frequency slider at the top selects from _CUTOFF_VALUES.  The chosen
    value is persisted in the DB ('spectral_cutoff_hz') and emitted via
    cutoff_changed so RawCurveWindow can redraw its contact markers.
    """

    analysis_params_changed = pyqtSignal()   # emitted when any analysis parameter changes

    def __init__(
        self,
        db_path:      str,
        experimentalist: str | None  = None,
        session_info: dict | None = None,
        worker=None,
    ) -> None:
        super().__init__()
        self._db_path      = db_path
        # Profile owner — INITIAL value only.  The key follows the curve on
        # screen: _sync_profile_owner re-resolves it from every displayed
        # file's watched directory (see the profile section further down).
        self._experimentalist = experimentalist
        self._owner_synced = False   # force a profile load on the first curve
        self._owner_dir_cache: dict[str, str | None] = {}
        self._worker       = worker
        self._nav          = None   # WorkerNavBar, built below in worker mode

        # ONE fetch of THE parameter set in force — not eight separate lookups
        # that each re-ask whose it is.  Same call display_roi and the worker
        # make.  (These are all nm² / nm / pts as stored; nothing is converted.)
        _ps = _db.load_analysis_params(db_path)
        # The value IN FORCE, verbatim — never snapped to a slider position.
        # Snapping here made the window filter, plot and label at one number
        # while the database and the batch worker used another, with nothing on
        # screen saying so.  The slider POSITION is derived from this value
        # (_cutoff_index); this value is never derived from the slider.
        self._cutoff_hz      = float(_ps['spectral_cutoff_hz'])
        self._current_curve  = None   # set by update_curve; used by slider redraws
        self._n_appr         = 0      # set by update_curve; used by drag handlers
        self._trim_pts          = int(_ps['turnaround_trim_pts'])
        self._var_window_ms     = _ps['var_window_ms']
        self._thresh_appr_val   = _ps['detection_threshold_appr']
        self._thresh_retr_val   = _ps['detection_threshold_retr']
        self._anchor_nm         = _ps['baseline_anchor_nm']
        self._invols_offset_pts = int(_ps['invols_offset_pts'])
        self._invols_window_pts = int(_ps['invols_window_pts'])

        self.setWindowTitle("SMFS — decomposition")
        self.setWindowFlag(Qt.WindowType.Window)
        fit_on_screen(self, 900, 700)

        root = QVBoxLayout(self)
        root.setSpacing(2)
        root.setContentsMargins(6, 6, 6, 6)

        _hdr = _make_session_header(session_info)
        if _hdr is not None:
            root.addWidget(_hdr)

        self._context_label = QLabel("No curve")
        self._context_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self._context_label.setStyleSheet(style.qss_text(style.UI_TEXT))
        root.addWidget(self._context_label)

        # ── Worker navigation strip (Prev/Next/scrubber) ──────────────────────
        # Present only in worker mode: drives the shared worker so this window can
        # scrub the queue itself and stays synced with every other worker view.
        if self._worker is not None:
            self._nav = WorkerNavBar(self._worker, db_path)
            self._nav.curve_selected.connect(self._on_nav_curve_selected)
            root.addWidget(self._nav)

        # ── Cutoff slider row ──────────────────────────────────────────────────
        ctrl = QWidget()
        # FlowLayout, not QHBoxLayout.  This strip is eight labelled controls
        # wide; in one non-wrapping row its minimum width came to roughly
        # 2100 px, which silently over-ruled resize(900, 700) and opened the
        # window wider than a 1920 px screen.  A control strip is bound BY its
        # window, not the reverse — this one now takes extra rows instead of
        # extra width.  Each label travels with its own control (LabeledControl)
        # so a wrap can never separate the two.
        ctrl_layout = FlowLayout(ctrl, margin=0, h_spacing=14, v_spacing=4)

        self._cutoff_slider = QSlider(Qt.Orientation.Horizontal)
        self._cutoff_slider.setRange(0, len(_CUTOFF_VALUES) - 1)
        self._cutoff_slider.setValue(self._cutoff_index(self._cutoff_hz))
        self._cutoff_slider.setFixedWidth(160)
        self._cutoff_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._cutoff_slider.setTickInterval(1)
        self._cutoff_slider.setToolTip(
            "Low-pass cutoff of the Bessel filter. The smooth channel is what "
            "the WLC fitter sees; the noise channel is what contact and "
            "snap-off detection measure.\n\n"
            "Filtering harder does not add information — it lowers the scatter "
            "and raises the correlation time \u03c4 by the same factor, so the "
            "error bars come out the same either way.\n\n"
            "The slider stops below this curve's Nyquist frequency, where the "
            "filter is undefined.")
        self._cutoff_slider.valueChanged.connect(self._on_cutoff_slider)

        self._cutoff_label = QLabel(self._cutoff_text())
        self._cutoff_label.setMinimumWidth(90)
        ctrl_layout.addWidget(
            LabeledControl("Cutoff:", self._cutoff_slider, self._cutoff_label))

        # What this cutoff costs the error bars, for the curve on screen.  Next
        # to the control that decides it, because a cutoff chosen without knowing
        # its sqrt(tau) is chosen blind — the same reasoning that put seg_tau
        # beside seg_l_p_err in the queue rather than in a diagnostics block.
        self._tau_label = QLabel("")
        self._tau_label.setStyleSheet(style.qss_text(style.UI_MUTED))
        self._tau_label.setVisible(False)

        # Nyquist: the one hard limit here.  Silent unless the slider is
        # actually being held back by it.
        self._cutoff_limit_label = QLabel("")
        self._cutoff_limit_label.setStyleSheet(style.qss_text(style.TEXT_WARNING))
        self._cutoff_limit_label.setVisible(False)

        # Sits with the slider, not in a status bar: this is a fact about the
        # value being chosen right here, and it is silent unless it applies.
        self._acq_filter_label = QLabel("")
        self._acq_filter_label.setStyleSheet(style.qss_text(style.TEXT_WARNING))
        self._acq_filter_label.setWordWrap(False)
        self._acq_filter_label.setVisible(False)

        # The three readouts travel as ONE flow item, so they wrap onto their
        # own line together rather than each widening the strip on its own.
        ctrl_layout.addWidget(LabeledControl(
            "", self._tau_label, self._cutoff_limit_label,
            self._acq_filter_label))

        self._trim_spinbox = QSpinBox()
        self._trim_spinbox.setRange(0, 9999)   # upper bound set dynamically per curve
        _quant.configure_spinbox(self._trim_spinbox, "turnaround_trim_pts", suffix=False)
        self._trim_spinbox.setValue(self._trim_pts)
        self._trim_spinbox.setToolTip(
            "Samples skipped either side of the piezo turnaround before contact "
            "and snap-off are searched for. The reversal leaves a ringing "
            "artefact that can cross the variance threshold and be read as a "
            "real snap-off. Raise this if snap-off is being found right at the "
            "turnaround.")
        self._trim_spinbox.valueChanged.connect(self._on_trim_spinbox)
        ctrl_layout.addWidget(LabeledControl("Trim pts:", self._trim_spinbox))

        self._var_win_spinbox = QDoubleSpinBox()
        self._var_win_spinbox.setRange(0.1, 50.0)
        _quant.configure_spinbox(self._var_win_spinbox, "var_window_ms", suffix=False)
        self._var_win_spinbox.setValue(self._var_window_ms)
        self._var_win_spinbox.setToolTip(
            "Length of the moving-variance window that finds contact and "
            "snap-off, in milliseconds — so it becomes a different number of "
            "samples for each cohort's sample rate. Contact is the first place "
            "the variance steps up.\n\n"
            "How much it changes the landmarks is cohort-dependent: worth "
            "checking on your own curves rather than assuming.")
        self._var_win_spinbox.valueChanged.connect(self._on_var_win_spinbox)
        ctrl_layout.addWidget(
            LabeledControl("Var. win (ms):", self._var_win_spinbox))

        self._thresh_appr_spinbox = QDoubleSpinBox()
        self._thresh_appr_spinbox.setRange(_to_shown(1e-4), _to_shown(1e3))
        _seed_threshold_box(self._thresh_appr_spinbox, "detection_threshold_appr",
                            self._thresh_appr_val)
        # Wide enough for a widened box: a threshold written by a drag
        # before quantize() existed carries mouse-position noise, and
        # _seed_threshold_box shows it rather than rounding it away.
        # quantities.audit_stored_precision lists these for tidying.
        self._thresh_appr_spinbox.setToolTip(
            "Moving-variance threshold that marks contact on the approach, in "
            "nm\u00b2. Scanning outward from deepest contact, the first sample "
            "above this is taken as the tip leaving the surface.\n\n"
            "You can drag the line on the variance panel instead of typing — "
            "either way the box and the stored value are the same number.")
        self._thresh_appr_spinbox.valueChanged.connect(self._on_thresh_appr_spinbox)
        ctrl_layout.addWidget(LabeledControl(
            f"Thr. appr ({_thresh_unit()}):", self._thresh_appr_spinbox))

        self._thresh_retr_spinbox = QDoubleSpinBox()
        self._thresh_retr_spinbox.setRange(_to_shown(1e-4), _to_shown(1e3))
        _seed_threshold_box(self._thresh_retr_spinbox, "detection_threshold_retr",
                            self._thresh_retr_val)
        self._thresh_retr_spinbox.setToolTip(
            "Moving-variance threshold that marks snap-off on the retract, in "
            "nm\u00b2. Scanning forward from contact, the first sample above "
            "this is snap-off.\n\n"
            "Snap-off is the zero that every piezo landmark and every extension "
            "in the app is measured from, so this control moves more downstream "
            "numbers than anything else in this window.")
        self._thresh_retr_spinbox.valueChanged.connect(self._on_thresh_retr_spinbox)
        ctrl_layout.addWidget(LabeledControl(
            f"Thr. retr ({_thresh_unit()}):", self._thresh_retr_spinbox))

        self._anchor_spinbox = QSpinBox()
        self._anchor_spinbox.setRange(10, 2000)
        _quant.configure_spinbox(self._anchor_spinbox, "baseline_anchor_nm", suffix=False)
        self._anchor_spinbox.setValue(int(round(self._anchor_nm)))
        self._anchor_spinbox.setToolTip(
            "Width of the far-retract region used to characterize the "
            "zero-force baseline, in nm of piezo travel. It is a statistically "
            "useful subset, not necessarily the entire baseline. ROI searching "
            "begins at its inner boundary.\n\n"
            "It must sit entirely after the molecule has let go. Too short and "
            "the baseline statistics are noisy; too long and the region starts "
            "including real signal. The fitted offset and RMS are shown beside it.")
        self._anchor_spinbox.valueChanged.connect(self._on_anchor_spinbox)

        self._baseline_label = QLabel("")
        _bl_font = self._baseline_label.font()
        _bl_font.setPointSize(_SMALL_FONT_PT)
        self._baseline_label.setFont(_bl_font)
        self._baseline_label.setStyleSheet(f"color: {_COLOR_MUTED};")
        self._baseline_label.setMinimumWidth(170)
        ctrl_layout.addWidget(LabeledControl(
            "Baseline width (nm):", self._anchor_spinbox, self._baseline_label))

        self._invols_offset_spinbox = QSpinBox()
        self._invols_offset_spinbox.setRange(0, 9999)
        _quant.configure_spinbox(self._invols_offset_spinbox, "invols_offset_pts", suffix=False)
        self._invols_offset_spinbox.setValue(self._invols_offset_pts)
        self._invols_offset_spinbox.setToolTip(
            "Samples skipped back from the turnaround before the invOLS fit "
            "window begins. The deepest-contact samples carry piezo ringing and "
            "any sample deformation at peak load, both of which bend the line "
            "that converts deflection into force.")
        self._invols_offset_spinbox.valueChanged.connect(self._on_invols_offset_spinbox)
        ctrl_layout.addWidget(LabeledControl(
            "invOLS off (pts):", self._invols_offset_spinbox))

        self._invols_window_spinbox = QSpinBox()
        self._invols_window_spinbox.setRange(10, 9999)
        _quant.configure_spinbox(self._invols_window_spinbox, "invols_window_pts", suffix=False)
        self._invols_window_spinbox.setValue(self._invols_window_pts)
        self._invols_window_spinbox.setToolTip(
            "Number of samples in the invOLS straight-line fit, counted back "
            "from the offset above toward shallower contact.\n\n"
            "This slope is what turns raw deflection into force, so it scales "
            "every force in the app. The fitted slope and RMS residual are shown "
            "beside it.")
        self._invols_window_spinbox.valueChanged.connect(self._on_invols_window_spinbox)

        self._invols_label = QLabel("")
        _iol_font = self._invols_label.font()
        _iol_font.setPointSize(_SMALL_FONT_PT)
        self._invols_label.setFont(_iol_font)
        self._invols_label.setStyleSheet(f"color: {_COLOR_MUTED};")
        self._invols_label.setMinimumWidth(170)
        ctrl_layout.addWidget(LabeledControl(
            "invOLS win (pts):", self._invols_window_spinbox,
            self._invols_label))

        # No addStretch(): a FlowLayout packs from the left already, and QLayout
        # has no stretch to add.  The strip keeps its natural height for the
        # window's current width.
        ctrl.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Minimum)
        root.addWidget(ctrl)

        # ── Three stacked plots in a shared GraphicsLayoutWidget ──────────────
        # Using GraphicsLayoutWidget (rather than three separate PlotWidgets)
        # ensures all Y-axes are the same pixel width, so the plot areas
        # are perfectly aligned and share a true common X extent.
        _glw = pg.GraphicsLayoutWidget()
        _glw.setBackground(style.SURFACE)

        self._plt_low  = _glw.addPlot(row=0, col=0)
        self._plt_high = _glw.addPlot(row=1, col=0)
        self._plt_var  = _glw.addPlot(row=2, col=0)

        # The variance panel takes its unit from the quantity whose threshold
        # lines are drawn on it, so the axis, the two spin boxes and the
        # database can only ever be in one unit — declared in ONE place.
        # si=False for the same reason: a spin box cannot carry an SI prefix,
        # so the axis the box's own value is dragged on must not float either.
        _var_unit = _quant.get("detection_threshold_appr").shown_unit
        for pi, ylabel, yunits, si in (
            (self._plt_low,  "Low",      _quant.NM, True),
            (self._plt_high, "High",     _quant.NM, True),
            (self._plt_var,  "Variance", _var_unit, False),
        ):
            set_si_label(pi, "left", ylabel, yunits, si=si)
            pi.showGrid(x=True, y=True, alpha=0.2)

        # Hide X tick values on the upper two plots — only the bottom needs them
        self._plt_low.getAxis('bottom').setStyle(showValues=False)
        self._plt_high.getAxis('bottom').setStyle(showValues=False)

        self._plt_high.setXLink(self._plt_low)
        self._plt_var.setXLink(self._plt_low)
        self._plt_var.setLabel("bottom", "Sample index")

        # shrinkable(): three stacked plots accumulate a minimum height, which
        # is the other half of a window that will not fit on the screen.  The
        # plots redraw smaller instead.
        root.addWidget(shrinkable(_glw, min_w=320, min_h=240), stretch=3)

        # ── Curve items — approach (red) and retract (blue) ───────────────────
        _appr_pen = style.data_pen(style.SIG_APPROACH, width=1.0)
        _retr_pen = style.data_pen(style.SIG_RETRACT, width=1.0)

        self._low_appr  = self._plt_low.plot([], [], pen=_appr_pen)
        self._low_retr  = self._plt_low.plot([], [], pen=_retr_pen)
        self._high_appr = self._plt_high.plot([], [], pen=_appr_pen)
        self._high_retr = self._plt_high.plot([], [], pen=_retr_pen)
        self._var_appr  = self._plt_var.plot([], [], pen=_appr_pen)
        self._var_retr  = self._plt_var.plot([], [], pen=_retr_pen)

        # ── Approach/retract divider lines (one per plot) ─────────────────────
        _div_pen = pg.mkPen(_COLOR_DIVIDER, width=1)
        self._div_low  = pg.InfiniteLine(pos=0, angle=90, movable=False, pen=_div_pen)
        self._div_high = pg.InfiniteLine(pos=0, angle=90, movable=False, pen=_div_pen)
        self._div_var  = pg.InfiniteLine(pos=0, angle=90, movable=False, pen=_div_pen)
        self._plt_low.addItem(self._div_low)
        self._plt_high.addItem(self._div_high)
        self._plt_var.addItem(self._div_var)

        # ── Turnaround trim markers — variance plot only ───────────────────────
        # Black vertical lines showing the _TRIM_PTS exclusion zone on each side
        # of the approach/retract boundary.  The search for the variance threshold
        # crossing never looks inside this region.
        _trim_pen = style.hair_pen(style.INK_MUTED, style=Qt.PenStyle.SolidLine)
        self._trim_left  = pg.InfiniteLine(pos=0, angle=90, movable=True, pen=_trim_pen)
        self._trim_right = pg.InfiniteLine(pos=0, angle=90, movable=True, pen=_trim_pen)
        self._trim_left.sigPositionChangeFinished.connect(
            lambda: self._on_trim_line_moved(self._trim_left, left=True)
        )
        self._trim_right.sigPositionChangeFinished.connect(
            lambda: self._on_trim_line_moved(self._trim_right, left=False)
        )
        self._plt_var.addItem(self._trim_left)
        self._plt_var.addItem(self._trim_right)

        # ── Contact markers — low plot only ───────────────────────────────────
        # Vertical dashed lines matching the colours used in RawCurveWindow.
        _contact_pen  = pg.mkPen(_COLOR_CONTACT, width=2, style=Qt.PenStyle.DashLine)
        _snapoff_pen  = pg.mkPen(_COLOR_SNAPOFF, width=2, style=Qt.PenStyle.DashLine)
        self._low_contact = pg.InfiniteLine(
            pos=0, angle=90, movable=False, pen=_contact_pen,
            label='contact', labelOpts={'position': 0.95, 'color': _COLOR_CONTACT},
        )
        self._low_snapoff = pg.InfiniteLine(
            pos=0, angle=90, movable=False, pen=_snapoff_pen,
            label='snap-off', labelOpts={'position': 0.82, 'color': _COLOR_SNAPOFF},
        )
        self._low_contact.setVisible(False)
        self._low_snapoff.setVisible(False)
        self._plt_low.addItem(self._low_contact)
        self._plt_low.addItem(self._low_snapoff)

        # ── Baseline anchor marker — low plot, retract side ───────────────────
        # Dotted blue line at the inner edge of the characterization region.
        # Position (sample-index space): n_appr + (n_retr - n_anchor)
        self._anchor_line = pg.InfiniteLine(
            pos=0, angle=90, movable=True,
            pen=style.guide_pen(style.SIG_RETRACT, style=Qt.PenStyle.DotLine),
            label='baseline', labelOpts={'position': 0.10, 'color': 'b'},
        )
        self._anchor_line.sigPositionChangeFinished.connect(self._on_anchor_line_moved)
        self._anchor_line.setVisible(False)
        self._plt_low.addItem(self._anchor_line)

        # ── invOLS fit overlay — low plot, approach side ─────────────────────
        # Bold black line segment drawn through the fitted window.
        self._invols_fit_line = self._plt_low.plot(
            [], [], pen=style.model_pen(style.INK, alpha=255)
        )

        # ── Baseline fit overlay — low plot, retract anchor region ──────────
        # Same idea as the invOLS line above: bold black segment through the
        # characterization-region fit (the drift/flatness diagnostic, not the
        # constant offset itself — see BaselineFit in signal_processing.py).
        self._baseline_fit_line = self._plt_low.plot(
            [], [], pen=style.model_pen(style.INK, alpha=255)
        )

        # ── Threshold lines — variance plot only ──────────────────────────────
        # Horizontal lines showing the computed variance threshold for each half.
        self._thresh_appr = pg.InfiniteLine(
            pos=0, angle=0, movable=True,
            pen=style.guide_pen(_COLOR_CONTACT, style=Qt.PenStyle.DotLine),
        )
        self._thresh_retr = pg.InfiniteLine(
            pos=0, angle=0, movable=True,
            pen=style.guide_pen(_COLOR_SNAPOFF, style=Qt.PenStyle.DotLine),
        )
        self._thresh_appr.sigPositionChangeFinished.connect(
            lambda: self._on_thresh_line_moved(appr=True)
        )
        self._thresh_retr.sigPositionChangeFinished.connect(
            lambda: self._on_thresh_line_moved(appr=False)
        )
        self._thresh_appr.setVisible(False)
        self._thresh_retr.setVisible(False)
        self._plt_var.addItem(self._thresh_appr)
        self._plt_var.addItem(self._thresh_retr)

        # Opening the window does not write a profile. _sync_profile_owner
        # seeds one when a curve is first displayed.

    @staticmethod
    def _cutoff_index(hz: float) -> int:
        """Slider position for a cutoff — the nearest offered value's index.

        Position only.  It does NOT change self._cutoff_hz: a stored value that
        is not on the list (an older profile, a hand-edited row) keeps its exact
        number and is labelled as off-list, rather than being silently rounded to
        whatever the slider can express.  Rounding it here would then get written
        back to the database by the next unrelated profile save.
        """
        return min(range(len(_CUTOFF_VALUES)),
                   key=lambda i: abs(_CUTOFF_VALUES[i] - hz))

    def _cutoff_text(self) -> str:
        """The cutoff as shown: the real value, marked when it is off-list."""
        hz = self._cutoff_hz
        suffix = "" if int(round(hz)) in _CUTOFF_VALUES else "*"
        return f"{hz:,g} Hz{suffix}"

    def _refresh_cutoff_limits(self) -> None:
        """Offer only cutoffs this curve can actually be filtered at.

        signal_processing.bessel_decompose raises at or above Nyquist, so a
        position above it is not a choice the user could take and have work — it
        is a crash waiting for a slider drag.  This is the one thing here that IS
        gated, and for the same reason the WLC fitter's l_c floor is: the
        computation is undefined, not merely inadvisable.  Everything else about
        the cutoff informs and lets the user decide.

        Nothing stored moves.  The slider's reach shrinks; self._cutoff_hz is
        untouched, so scrubbing across a cohort with mixed sample rates cannot
        silently rewrite the parameter set.

        THAT LAST GUARANTEE NEEDS blockSignals AND IT IS NOT OPTIONAL.  QSlider
        CLAMPS its value when the maximum drops below it, and emits valueChanged
        while doing so — which lands in _on_cutoff_slider, which writes to the
        database and tells the whole app the parameters changed.  Unblocked, then,
        merely scrubbing onto a slow curve would rewrite the cohort's cutoff, with
        no user action at all: the displayed-vs-stored defect class arriving
        through a queue scrub instead of a keystroke.  Verified against a real
        QSlider, not assumed — the test does the same.
        """
        def _set_reach(max_index: int, enabled: bool) -> None:
            self._cutoff_slider.blockSignals(True)
            self._cutoff_slider.setMaximum(max_index)
            # Re-seat the handle under the value in force.  Qt has already
            # clamped it if it had to; this makes the position derive from
            # self._cutoff_hz as it does everywhere else, rather than from
            # whatever the clamp happened to leave behind.
            self._cutoff_slider.setValue(
                min(self._cutoff_index(self._cutoff_hz), max_index))
            self._cutoff_slider.blockSignals(False)
            self._cutoff_slider.setEnabled(enabled)

        curve = self._current_curve
        rate = float(getattr(curve, "sample_rate_hz", 0.0) or 0.0) if curve is not None else 0.0
        if rate <= 0:
            # No curve, or a file whose header gave no usable rate: offer the
            # whole list rather than inventing a limit from a number we don't have.
            _set_reach(len(_CUTOFF_VALUES) - 1, True)
            self._cutoff_limit_label.setVisible(False)
            return
        nyquist = rate / 2.0
        usable = [i for i, v in enumerate(_CUTOFF_VALUES) if v < nyquist]
        if not usable:
            # Sample rate too low for any offered cutoff — real in this catalog,
            # where a handful of files carry a header rate of 1 Hz.  Say so
            # instead of letting the drag raise.
            _set_reach(0, False)
            self._cutoff_limit_label.setText(
                f"⚠ sample rate {rate:,.0f} Hz — no cutoff below Nyquist")
            self._cutoff_limit_label.setToolTip(
                "A low-pass cutoff must sit below the Nyquist frequency "
                f"({nyquist:,.1f} Hz) or the filter is undefined. This file's "
                "header reports a sample rate too low for any offered value.")
            self._cutoff_limit_label.setVisible(True)
            return
        _set_reach(usable[-1], True)
        clipped = usable[-1] < len(_CUTOFF_VALUES) - 1
        if clipped:
            self._cutoff_limit_label.setText(f"⚠ Nyquist {nyquist:,.0f} Hz")
            self._cutoff_limit_label.setToolTip(
                f"This curve was sampled at {rate:,.0f} Hz, so cutoffs at or "
                f"above {nyquist:,.1f} Hz are undefined and are not offered.")
        self._cutoff_limit_label.setVisible(clipped)

    def _refresh_tau_hint(self) -> None:
        """What this cutoff costs the error bars, for the curve on screen.

        tau ~ sample_rate / cutoff estimates the part of the integrated
        autocorrelation time contributed by filtering, and error bars carry
        sqrt(tau). Both numbers are already in hand, so
        this turns an abstract choice into the thing it actually decides.

        Stated as "at least", because it is a floor: the acquisition filter and
        WLC model error both add to the measured tau and neither is separable
        from a residual (§3).  The real tau is measured per fit, never this.
        """
        curve = self._current_curve
        rate = float(getattr(curve, "sample_rate_hz", 0.0) or 0.0) if curve is not None else 0.0
        if rate <= 0 or self._cutoff_hz <= 0 or self._cutoff_hz >= rate / 2.0:
            self._tau_label.setVisible(False)
            return
        tau = rate / self._cutoff_hz
        self._tau_label.setText(f"τ ≳ {tau:,.1f}  (error bars ×{tau ** 0.5:,.1f})")
        self._tau_label.setToolTip(
            f"Filtering {rate:,.0f} Hz data at {self._cutoff_hz:,.0f} Hz leaves "
            f"neighbouring samples correlated over about {tau:,.1f} of them, so a "
            f"fit's error bars are widened by √τ ≈ {tau ** 0.5:,.1f}.\n\n"
            "A floor, not the answer: the acquisition filter and any model "
            "misfit add to the τ each fit actually measures from its own "
            "residual. Filtering harder does not shrink a reported error bar — "
            "σ falls and τ rises by the same factor."
        )
        self._tau_label.setVisible(True)

    def _refresh_acq_filter_warning(self) -> None:
        """
        Say so when our cutoff stops being the narrower filter.

        Driven off the displayed curve's own wave-note bandwidth, so it costs no
        DB query during playback, and re-checked on BOTH inputs that can change
        the answer — a new curve and a new cutoff.  Checking only the slider
        would leave a stale verdict on screen the moment the playhead moved to
        another experimentalist's cohort, which is the failure mode this whole
        warning exists to catch.
        """
        curve = self._current_curve
        why = filter_bandwidth_warning(
            float(self._cutoff_hz),
            getattr(curve, "force_filter_bw_hz", None) if curve is not None else None,
        )
        conflict = bool(why)
        if conflict:
            acq = curve.force_filter_bw_hz
            self._acq_filter_label.setText(f"⚠ acquisition filter {acq:,.0f} Hz")
            self._acq_filter_label.setToolTip(why)
        else:
            self._acq_filter_label.setText("")
            self._acq_filter_label.setToolTip("")
        self._acq_filter_label.setVisible(conflict)

    def _on_cutoff_slider(self, pos: int) -> None:
        """Slider moved — update cutoff, persist to DB, re-plot, notify main window."""
        self._cutoff_hz = float(_CUTOFF_VALUES[pos])
        self._cutoff_label.setText(self._cutoff_text())
        self._refresh_tau_hint()
        self._refresh_acq_filter_warning()
        _db.update_analysis_param('spectral_cutoff_hz', float(self._cutoff_hz), self._db_path)
        self._save_user_profile()
        if self._current_curve is not None:
            self.update_curve(self._current_curve)
        self.analysis_params_changed.emit()

    def _on_trim_spinbox(self, value: int) -> None:
        """Spinbox changed — update trim, persist to DB, redraw, notify main window."""
        self._trim_pts = value
        _db.update_analysis_param('turnaround_trim_pts', float(value), self._db_path)
        self._save_user_profile()
        if self._current_curve is not None:
            self.update_curve(self._current_curve)
        self.analysis_params_changed.emit()

    def _on_trim_line_moved(self, line: pg.InfiniteLine, *, left: bool) -> None:
        """Drag finished — derive new trim from line position, update spinbox."""
        if self._n_appr == 0:
            return
        if left:
            new_val = int(round(self._n_appr - line.value()))
        else:
            new_val = int(round(line.value() - self._n_appr))
        new_val = max(0, min(self._trim_spinbox.maximum(), new_val))
        # Already integral by construction above, so this is a no-op today —
        # kept so that EVERY drag handler quantises, with no exceptions to
        # remember.  An exception here is how the other four came to write
        # raw mouse positions to the database.
        new_val = int(_quant.quantize('turnaround_trim_pts', new_val))
        # Block spinbox signal to avoid double-emit; set lines directly then emit once
        self._trim_spinbox.blockSignals(True)
        self._trim_spinbox.setValue(new_val)
        self._trim_spinbox.blockSignals(False)
        self._trim_pts = new_val
        _db.update_analysis_param('turnaround_trim_pts', float(new_val), self._db_path)
        # Mirror the other line
        self._trim_left.setValue(self._n_appr - self._trim_pts)
        self._trim_right.setValue(self._n_appr + self._trim_pts)
        self._save_user_profile()
        # A line drag bypasses the spinbox's valueChanged handler, so request
        # the same recomputation explicitly. Otherwise the guides move while
        # contact and snap-off still reflect the previous trim value.
        if self._current_curve is not None:
            self.update_curve(self._current_curve)
        self.analysis_params_changed.emit()

    def _refresh_threshold_guides(self) -> None:
        """Show live thresholds whenever the variance curves are available.

        These are editable inputs, not successful-detection outputs. Keeping
        them visible when a landmark is absent gives the user the drag target
        needed to recover a failed detection.
        """
        self._thresh_appr.setValue(_to_shown(self._thresh_appr_val))
        self._thresh_retr.setValue(_to_shown(self._thresh_retr_val))
        self._thresh_appr.setVisible(True)
        self._thresh_retr.setVisible(True)

    def _on_thresh_appr_spinbox(self, value: float) -> None:
        self._thresh_appr_val = _to_stored(value)   # detection + DB work in nm²
        _db.update_analysis_param('detection_threshold_appr', self._thresh_appr_val, self._db_path)
        self._save_user_profile()
        if self._current_curve is not None:
            self.update_curve(self._current_curve)
        self.analysis_params_changed.emit()

    def _on_thresh_retr_spinbox(self, value: float) -> None:
        self._thresh_retr_val = _to_stored(value)   # detection + DB work in nm²
        _db.update_analysis_param('detection_threshold_retr', self._thresh_retr_val, self._db_path)
        self._save_user_profile()
        if self._current_curve is not None:
            self.update_curve(self._current_curve)
        self.analysis_params_changed.emit()

    def _on_var_win_spinbox(self, value: float) -> None:
        self._var_window_ms = value
        _db.update_analysis_param('var_window_ms', value, self._db_path)
        self._save_user_profile()
        if self._current_curve is not None:
            self.update_curve(self._current_curve)
        self.analysis_params_changed.emit()

    def _on_thresh_line_moved(self, *, appr: bool) -> None:
        """Drag finished — convert the line position to the stored unit.

        The stored value is QUANTISED to what the spin box can display, so the
        screen and the database cannot hold different numbers for the same
        threshold. A drag is a gesture rather than a choice of digits, so
        quantizing it to display precision loses no user-selected precision.
        """
        key = 'detection_threshold_appr' if appr else 'detection_threshold_retr'
        line    = self._thresh_appr if appr else self._thresh_retr
        spinbox = self._thresh_appr_spinbox if appr else self._thresh_retr_spinbox

        floor = spinbox.minimum()
        new_thresh_nm2 = _quant.quantize(key, _to_stored(max(floor, line.value())))
        new_thresh_shown = _to_shown(new_thresh_nm2)
        if appr:
            self._thresh_appr_val = new_thresh_nm2
        else:
            self._thresh_retr_val = new_thresh_nm2
        _db.update_analysis_param(key, new_thresh_nm2, self._db_path)

        spinbox.blockSignals(True)
        spinbox.setValue(new_thresh_shown)
        spinbox.blockSignals(False)

        self._save_user_profile()
        if self._current_curve is not None:
            self.update_curve(self._current_curve)
        self.analysis_params_changed.emit()

    def _on_invols_offset_spinbox(self, value: int) -> None:
        self._invols_offset_pts = int(value)
        _db.update_analysis_param('invols_offset_pts', float(value), self._db_path)
        self._save_user_profile()
        if self._current_curve is not None:
            self.update_curve(self._current_curve)
        self.analysis_params_changed.emit()

    def _on_invols_window_spinbox(self, value: int) -> None:
        self._invols_window_pts = int(value)
        _db.update_analysis_param('invols_window_pts', float(value), self._db_path)
        self._save_user_profile()
        if self._current_curve is not None:
            self.update_curve(self._current_curve)
        self.analysis_params_changed.emit()

    def _on_anchor_spinbox(self, value: int) -> None:
        self._anchor_nm = float(value)
        _db.update_analysis_param('baseline_anchor_nm', float(value), self._db_path)
        self._save_user_profile()
        if self._current_curve is not None:
            self.update_curve(self._current_curve)
        self.analysis_params_changed.emit()

    def _on_anchor_line_moved(self) -> None:
        """Drag finished — derive baseline width from the line and update the control."""
        if self._current_curve is None or self._n_appr == 0:
            return
        n_retr     = len(self._current_curve.piezo_retr)
        pos        = self._anchor_line.value()
        # Clamp to valid retract range (leave at least 10 pts on each side)
        pos        = max(self._n_appr + 1, min(self._n_appr + n_retr - 10, pos))
        n_anchor   = (self._n_appr + n_retr) - int(round(pos))
        n_anchor   = max(10, min(n_retr - 1, n_anchor))

        piezo_retr  = self._current_curve.piezo_retr
        retr_range  = abs(float(piezo_retr[-1]) - float(piezo_retr[0]))
        anchor_nm   = (n_anchor / n_retr) * retr_range if n_retr > 0 else 150.0
        anchor_nm   = max(10.0, min(2000.0, anchor_nm))

        self._anchor_nm = anchor_nm
        self._anchor_spinbox.blockSignals(True)
        self._anchor_spinbox.setValue(int(round(anchor_nm)))
        self._anchor_spinbox.blockSignals(False)

        # Snap line to the integer-rounded spinbox value
        frac     = min(self._anchor_nm / retr_range, 1.0) if retr_range > 0 else 0.15
        n_snapped = max(10, int(round(frac * n_retr)))
        self._anchor_line.setValue(self._n_appr + (n_retr - n_snapped))

        # baseline_anchor_nm is declared an INTEGER quantity, and the spin
        # box above shows int(round(anchor_nm)) — but the raw float went to
        # the database, so the box and the DB disagreed by up to half a
        # nanometre for every dragged anchor.  quantize() rounds it the same
        # way the box does.
        anchor_nm = _quant.quantize('baseline_anchor_nm', anchor_nm)
        self._anchor_nm = anchor_nm
        _db.update_analysis_param('baseline_anchor_nm', anchor_nm, self._db_path)
        self._save_user_profile()
        if self._current_curve is not None:
            self.update_curve(self._current_curve)
        self.analysis_params_changed.emit()

    # ── User-profile persistence ──────────────────────────────────────────────
    #
    # The profile key FOLLOWS THE CURVE ON SCREEN: worker navigation resolves
    # the owner (experimentalist of the file's watched directory) and, when it
    # changes, loads that user's stored knobs into the controls + settings
    # table.  A key frozen at construction breaks in worker mode, where one
    # queue interleaves several users' files — that was the recurring
    # "changing her settings changes his" bug.

    def _sync_profile_owner(self, path: str | None) -> None:
        """Re-key the profile to THE parameter set in force, and load its
        knobs if that has changed.  Identical rule and identical code shape to
        display_roi._sync_profile_owner — both ask db.active_param_owner, the
        one place that answers "whose parameter set are we using".  The curve
        on screen does not enter into it.
        """
        try:
            owner = _db.active_param_owner(self._db_path)
        except Exception:
            return
        if self._owner_synced and owner == self._experimentalist:
            return
        self._owner_synced = True
        self._experimentalist = owner
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
        Load a user's stored decomposition knobs into fields, widgets, and the
        settings table.  Widget signals are blocked — this is a load, not an
        edit, so it must not re-save the profile; the caller redraws once
        afterwards via update_curve.
        """
        def _f(key: str, cur: float) -> float:
            try:
                return float(p[key])
            except (KeyError, TypeError, ValueError):
                return float(cur)

        self._cutoff_hz         = _f("spectral_cutoff_hz", self._cutoff_hz)
        self._trim_pts          = int(_f("turnaround_trim_pts", self._trim_pts))
        self._var_window_ms     = _f("var_window_ms", self._var_window_ms)
        self._thresh_appr_val   = _f("detection_threshold_appr", self._thresh_appr_val)
        self._thresh_retr_val   = _f("detection_threshold_retr", self._thresh_retr_val)
        self._anchor_nm         = _f("baseline_anchor_nm", self._anchor_nm)
        self._invols_offset_pts = int(_f("invols_offset_pts", self._invols_offset_pts))
        self._invols_window_pts = int(_f("invols_window_pts", self._invols_window_pts))

        self._cutoff_slider.blockSignals(True)
        self._cutoff_slider.setValue(self._cutoff_index(self._cutoff_hz))
        self._cutoff_slider.blockSignals(False)
        self._cutoff_label.setText(self._cutoff_text())
        self._refresh_tau_hint()
        for w, val in (
            (self._trim_spinbox,          self._trim_pts),
            (self._var_win_spinbox,       self._var_window_ms),
            (self._anchor_spinbox,        int(round(self._anchor_nm))),
            (self._invols_offset_spinbox, self._invols_offset_pts),
            (self._invols_window_spinbox, self._invols_window_pts),
        ):
            w.blockSignals(True)
            w.setValue(val)
            w.blockSignals(False)

        # The two thresholds are seeded separately: they need their box
        # widened to whatever the reloaded profile actually holds before the
        # value goes in, or the box would round it and then write the rounded
        # number back.  See _seed_threshold_box.
        _seed_threshold_box(self._thresh_appr_spinbox,
                            "detection_threshold_appr", self._thresh_appr_val)
        _seed_threshold_box(self._thresh_retr_spinbox,
                            "detection_threshold_retr", self._thresh_retr_val)

        # No mirror into the `settings` table. The parameter set lives in
        # exactly one place - the queue owner's profile - and the pipeline
        # reads it there (db.get_param). Copying it into a catalog-wide
        # table is what made that table mean "whoever was displayed last",
        # so every other experimentalist silently inherited it.

    def _save_user_profile(self) -> None:
        """
        Merge current DecompositionWindow param values into
        experimentalist_profiles[experimentalist] via db.merge_experimentalist_profile
        — a single atomic SQL statement, not read-then-write, so a concurrent
        save from another window or the analysis worker's QThread can never
        be silently discarded (see that function's docstring).
        """
        # Save to THE set in force — same one answer as the read side, asked
        # here rather than trusting self._experimentalist (None until the
        # first curve syncs).  See display_roi._save_user_profile.
        key = _db.active_param_owner(self._db_path)
        _db.merge_experimentalist_profile(key, {
            "spectral_cutoff_hz":       float(self._cutoff_hz),
            "turnaround_trim_pts":      float(self._trim_pts),
            "var_window_ms":            self._var_window_ms,
            "detection_threshold_appr": self._thresh_appr_val,
            "detection_threshold_retr": self._thresh_retr_val,
            "baseline_anchor_nm":       self._anchor_nm,
            "invols_offset_pts":        float(self._invols_offset_pts),
            "invols_window_pts":        float(self._invols_window_pts),
        }, self._db_path)

    def update_curve(self, curve: "ForceCurve") -> None:
        """Decompose curve and refresh all three plots.  Silent on any failure."""
        self._current_curve = curve   # remember for slider redraws
        # All three read the curve's own sample rate / bandwidth, so all three
        # are re-checked on a new curve as well as on a new cutoff.  Refreshing
        # only on the slider would leave a stale verdict the moment the playhead
        # crossed into another cohort — the failure these exist to catch.
        self._refresh_cutoff_limits()
        self._refresh_tau_hint()
        self._refresh_acq_filter_warning()
        if curve is None or curve.sample_rate_hz <= 0:
            self._clear()
            return
        # Keep changing analysis context visible without turning the taskbar
        # entry into a rapidly changing status line.
        self._context_label.setText(
            f"{os.path.basename(curve.path)}   |   "
            f"parameters: {_db.active_param_owner(self._db_path)}")
        try:
            from .signal_processing import decompose_curve, _moving_variance, _ms_to_pts
            dc = decompose_curve(curve, cutoff_hz=float(self._cutoff_hz))
        except Exception:
            self._clear()
            return

        n_appr        = len(dc.low_appr)
        n_retr        = len(dc.low_retr)
        self._n_appr  = n_appr
        idx_appr      = np.arange(n_appr)
        idx_retr      = np.arange(n_appr, n_appr + n_retr)
        _trim_pts     = self._trim_pts

        self._trim_spinbox.setMaximum(min(n_appr, n_retr))

        win_pts  = _ms_to_pts(self._var_window_ms, dc.sample_rate_hz)
        var_appr = _moving_variance(dc.high_appr, win_pts)
        var_retr = _moving_variance(dc.high_retr, win_pts)

        self._low_appr.setData(idx_appr, dc.low_appr)
        self._low_retr.setData(idx_retr, dc.low_retr)
        self._high_appr.setData(idx_appr, dc.high_appr)
        self._high_retr.setData(idx_retr, dc.high_retr)
        # Computed in nm²; drawn in whatever quantities.py says to show.
        self._var_appr.setData(idx_appr, _to_shown(var_appr))
        self._var_retr.setData(idx_retr, _to_shown(var_retr))
        self._refresh_threshold_guides()

        boundary = n_appr - 0.5
        self._div_low.setValue(boundary)
        self._div_high.setValue(boundary)
        self._div_var.setValue(boundary)

        self._trim_left.setValue(n_appr - _trim_pts)
        self._trim_right.setValue(n_appr + _trim_pts)

        # ── Landmarks + invOLS — via the ONE owning routine ────────────────────
        # Call curve_analysis.analyse_curve, the same owning routine used by
        # the worker, with this window's live (possibly not-yet-committed)
        # parameter values; analyse_curve persists whatever it computes on its
        # own, unconditionally, whenever it can identify the file.
        try:
            from dataclasses import replace
            from .curve_analysis import analyse_curve, pipeline_params_from
            snapshot = _db.load_analysis_params(self._db_path)
            preview = replace(
                snapshot,
                baseline_anchor_nm=self._anchor_nm,
                spectral_cutoff_hz=self._cutoff_hz,
                turnaround_trim_pts=self._trim_pts,
                var_window_ms=self._var_window_ms,
                detection_threshold_appr=self._thresh_appr_val,
                detection_threshold_retr=self._thresh_retr_val,
                invols_offset_pts=self._invols_offset_pts,
                invols_window_pts=self._invols_window_pts,
            )
            live_params = pipeline_params_from(preview)
            file_id = _db.get_file_id(curve.path, self._db_path)
            result, stage1 = analyse_curve(
                curve, live_params,
                db_path=self._db_path, code_ver=cache_version(), file_id=file_id,
            )
        except Exception:
            result, stage1 = None, None

        # ── Contact detection — marker positions ───────────────────────────────
        if result is not None and not np.isnan(result.contact_z) and not np.isnan(result.snapoff_z):
            begin_idx = int(np.argmin(np.abs(curve.piezo_appr - result.contact_z)))
            end_idx   = int(np.argmin(np.abs(curve.piezo_retr - result.snapoff_z)))

            self._low_contact.setValue(begin_idx)
            self._low_snapoff.setValue(n_appr + end_idx)
            self._low_contact.setVisible(True)
            self._low_snapoff.setVisible(True)

        else:
            self._low_contact.setVisible(False)
            self._low_snapoff.setVisible(False)

        # ── Baseline anchor marker ─────────────────────────────────────────────
        # Mark the start of the anchor region on the retract side of the low plot.
        # Convert anchor_nm (piezo distance) to a sample-index position.
        piezo_retr  = curve.piezo_retr
        retr_range  = abs(float(piezo_retr[-1]) - float(piezo_retr[0]))
        frac_anchor = min(self._anchor_nm / retr_range, 1.0) if retr_range > 0 else 0.15
        n_anchor    = max(10, int(round(frac_anchor * n_retr)))
        anchor_pos  = n_appr + (n_retr - n_anchor)
        self._anchor_line.setValue(anchor_pos)
        self._anchor_line.setVisible(True)

        # ── invOLS fit — approach, deep-contact band ──────────────────────────
        # slope comes from `result` (fresh compute or cache hit, either way).
        # The fit-window/intercept/R² diagnostics are persisted alongside the
        # slope (curve_analysis.py, params_invols) and so are ALSO available on
        # a cache hit via `stage1` — the elif fallback below now only fires for
        # a curve whose invols_slope was cached before diagnostics started
        # being persisted; it clears itself the next time that curve is
        # re-analysed under the current params.
        if result is not None and not np.isnan(result.invols_slope) \
                and stage1 is not None and stage1.invols_fit_lo_idx is not None:
            lo, hi = stage1.invols_fit_lo_idx, stage1.invols_fit_hi_idx
            x_lo_piezo = float(curve.piezo_appr[lo])
            x_hi_piezo = float(curve.piezo_appr[hi - 1])
            y_lo = result.invols_slope * x_lo_piezo + stage1.invols_intercept
            y_hi = result.invols_slope * x_hi_piezo + stage1.invols_intercept
            self._invols_fit_line.setData([lo, hi - 1], [y_lo, y_hi])
            self._invols_label.setText(
                f"invOLS slope={_q('invols_slope', result.invols_slope)}  "
                f"RMS={_q('invols_rms', stage1.invols_rms, with_unit=True)}"
            )
        elif result is not None and not np.isnan(result.invols_slope):
            self._invols_fit_line.setData([], [])
            self._invols_label.setText(
                f"invOLS slope={_q('invols_slope', result.invols_slope)}  RMS=(cached)")
        else:
            self._invols_fit_line.setData([], [])
            self._invols_label.setText("invOLS: —")

        # ── Baseline fit — retract, anchor region ──────────────────────────────
        # Same pattern as invOLS just above: offset/flatness come from `result`
        # (always present); the fit-window/intercept/RMS diagnostics come from
        # `stage1`, persisted alongside them (params_bl) so available on a
        # cache hit too — the elif fallback only fires for a curve analysed
        # before diagnostics started being persisted. RMS, not R², is the
        # goodness-of-fit number here — see BaselineFit.r2's docstring
        # (signal_processing.py): a flat, well-fit anchor region has little
        # variance for a line to explain, so r2 reads near 0 even on a good
        # fit and isn't useful for spotting a bad one.
        if result is not None and not np.isnan(result.flatness) \
                and stage1 is not None and stage1.baseline_fit_lo_idx is not None:
            lo, hi = stage1.baseline_fit_lo_idx, stage1.baseline_fit_hi_idx
            x_lo_piezo = float(curve.piezo_retr[lo])
            x_hi_piezo = float(curve.piezo_retr[hi - 1])
            y_lo = result.flatness * x_lo_piezo + stage1.baseline_intercept
            y_hi = result.flatness * x_hi_piezo + stage1.baseline_intercept
            self._baseline_fit_line.setData([n_appr + lo, n_appr + hi - 1], [y_lo, y_hi])
            self._baseline_label.setText(
                f"baseline slope={_q('flatness_slope', result.flatness)}  "
                f"RMS={_q('baseline_rms', stage1.baseline_rms, with_unit=True)}"
            )
        elif result is not None and not np.isnan(result.flatness):
            self._baseline_fit_line.setData([], [])
            self._baseline_label.setText(
                f"baseline slope={_q('flatness_slope', result.flatness)}  RMS=(cached)")
        else:
            self._baseline_fit_line.setData([], [])
            self._baseline_label.setText("baseline: —")

    # ── Worker navigation ─────────────────────────────────────────────────────

    def showEvent(self, event) -> None:
        # Catch up to the worker's current playhead when re-shown after being
        # hidden (the nav bar skips per-curve work while hidden).
        super().showEvent(event)
        if self._nav is not None:
            self._nav.sync_now()

    def _on_nav_curve_selected(self, path: str, file_id: int) -> None:
        """Worker moved the playhead — load and decompose the new curve."""
        self._sync_profile_owner(path)
        try:
            curve = load_force_curve(path)
        except Exception:
            self._clear()
            return
        self.update_curve(curve)

    def _clear(self) -> None:
        self._context_label.setText("No curve")
        for item in (
            self._low_appr,  self._low_retr,
            self._high_appr, self._high_retr,
            self._var_appr,  self._var_retr,
            self._invols_fit_line, self._baseline_fit_line,
        ):
            item.setData([], [])
        self._invols_label.setText("")
        self._baseline_label.setText("")
        for line in (
            self._low_contact, self._low_snapoff,
            self._thresh_appr, self._thresh_retr,
            self._anchor_line,
        ):
            line.setVisible(False)
