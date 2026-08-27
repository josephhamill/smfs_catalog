# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/qt_utils.py
#
# Shared Qt and pyqtgraph display helpers.  No window classes live here.
#
# Tints for rugs, params, categories and 2DH trace overlays all come from
# style.series_labeled(i) — a fixed order that stops rather than wrapping into a
# new hue, which is what puts two barely separable colours on one plot.

from __future__ import annotations

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtWidgets import (
    QApplication, QLabel, QProgressDialog, QScrollArea, QSizePolicy, QWidget,
)

import datetime
import pyqtgraph as pg

from .style import (
    _COLOR_MUTED,
    FONT_CAPTION_PT,
)

# ── Small UI constants ────────────────────────────────────────────────────────

_SMALL_FONT_PT = FONT_CAPTION_PT


class FixedDomainPlot(pg.PlotWidget):
    """Plot a matrix over one authoritative display-coordinate rectangle.

    `x_range` and `y_range` are axis coordinates (for example nm/pN or
    normalized WLC coordinates), never matrix-index ranges. Matrix dimensions
    remain independent and belong only to ImageItem.setImage/reshape.
    """

    def __init__(self, x_range, y_range, *args, on_double_click=None, **kwargs):
        object.__setattr__(self, "_domain_rect", self._rect(x_range, y_range))
        object.__setattr__(self, "_domain_ready", False)
        object.__setattr__(self, "_on_double_click", on_double_click)
        super().__init__(*args, **kwargs)
        self.centralWidget.setMinimumSize(0, 0)
        self.centralWidget.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        view_box = self.getViewBox()
        view_box.setAspectLocked(False)
        view_box.sigResized.connect(self.fit_domain)
        self._domain_ready = True
        self.fit_domain()

    @staticmethod
    def _rect(x_range, y_range) -> QRectF:
        x0, x1 = map(float, x_range)
        y0, y1 = map(float, y_range)
        if not x1 > x0 or not y1 > y0:
            raise ValueError("fixed plot domain ranges must increase")
        return QRectF(x0, y0, x1 - x0, y1 - y0)

    @property
    def domain_rect(self) -> QRectF:
        return QRectF(self._domain_rect)

    def set_domain(self, x_range, y_range) -> None:
        self._domain_rect = self._rect(x_range, y_range)
        self.fit_domain()

    def fit_domain(self) -> None:
        self.getViewBox().setRange(
            rect=self._domain_rect, padding=0, disableAutoRange=True)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._domain_ready:
            self.fit_domain()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._domain_ready:
            # QGraphicsWidget size hints otherwise keep the PlotItem wider
            # than a narrow viewport and QGraphicsView clips its right side.
            self.centralWidget.setGeometry(self.range)
            self.fit_domain()

    def mouseDoubleClickEvent(self, event) -> None:
        if self._on_double_click:
            self._on_double_click()
        super().mouseDoubleClickEvent(event)


# ── Window geometry ───────────────────────────────────────────────────────────
#
# A window's opening size is a REQUEST, and it is only half of the problem.
# Qt will not shrink a window below its layout's minimumSizeHint(), so a
# control column that cannot shrink defines the width no matter what resize()
# asks for -- which is how a plot ends up "forcing the window wide" instead of
# being bound by it.  The three helpers below are the two halves of that fix:
# fit_on_screen() bounds the request, shrinkable()/scrollable_column() stop the
# contents from over-ruling it.

_SCREEN_MARGIN = 96      # px kept clear of the work area, for chrome + panel
_MIN_WINDOW    = (480, 360)


def fit_on_screen(win, width: int, height: int) -> None:
    """Open `win` at `width`x`height`, shrunk to fit the screen it lands on.

    Every window hard-coded its opening size (1400x900 for the dashboard,
    1300x700 for View Fits), all chosen on a large monitor.  On a 1920x1080
    screen several of those opened with their buttons under the taskbar or off
    the right-hand edge.  The available geometry -- not the full screen -- is
    the bound, because it already excludes the panel/taskbar.

    Use this instead of calling resize() directly; tests/test_window_sizing.py
    checks that no window module calls resize() with a literal size.
    """
    screen = win.screen() or QApplication.primaryScreen()
    if screen is None:
        win.resize(max(width, _MIN_WINDOW[0]), max(height, _MIN_WINDOW[1]))
        return

    avail  = screen.availableGeometry()
    max_width = max(1, avail.width() - _SCREEN_MARGIN)
    max_height = max(1, avail.height() - _SCREEN_MARGIN)
    width = min(max(width, _MIN_WINDOW[0]), max_width)
    height = min(max(height, _MIN_WINDOW[1]), max_height)
    win.resize(width, height)

    # Pull the frame back inside the work area if it would hang off the top or
    # left.  A window whose title bar is above the desktop cannot be dragged
    # back by hand -- there is nothing left to grab -- so it has to be placed
    # correctly or not at all.  Only moved when it is actually outside; a
    # window the window manager has already placed sensibly is left alone.
    frame = win.frameGeometry()
    max_x = avail.x() + avail.width() - frame.width()
    max_y = avail.y() + avail.height() - frame.height()
    x = min(max(frame.x(), avail.x()), max(avail.x(), max_x))
    y = min(max(frame.y(), avail.y()), max(avail.y(), max_y))
    if (x, y) != (frame.x(), frame.y()):
        win.move(x, y)


def shrinkable(widget, min_w: int = 240, min_h: int = 160):
    """Let a plot pane follow the window instead of defining it.

    pyqtgraph's PlotWidget reports an Expanding policy but keeps whatever
    minimum it has been given, and a GraphicsLayoutWidget holding a stack of
    plots accumulates them.  Pinning a small explicit minimum is what makes the
    window genuinely resizable downwards; the plot simply redraws smaller.

    Returns `widget`, so it can wrap an addWidget() argument in place.
    """
    widget.setMinimumSize(min_w, min_h)
    widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    return widget


def scrollable_column(inner: QWidget, *, min_w: int | None = None) -> QScrollArea:
    """Wrap a tall column of controls so it scrolls instead of setting the
    window's minimum height.

    A side panel of spin boxes and check boxes has a minimumSizeHint as tall as
    all of them stacked; on a laptop that is what pushes the window past the
    bottom of the screen.  Inside a QScrollArea the panel keeps its natural
    width (so the numbers stay readable -- it is the HEIGHT that is negotiable)
    and the window is free to be shorter than its contents.
    """
    area = QScrollArea()
    area.setWidget(inner)
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    width = min_w if min_w is not None else inner.sizeHint().width()
    area.setMinimumWidth(width + 4)          # + a little, for the scrollbar
    area.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
    return area


# ── Session header helper ─────────────────────────────────────────────────────

def _make_session_header(session_info: dict | None) -> "QLabel | None":
    """
    Build a read-only one-liner QLabel summarising the analysis context.
    Returns None if session_info is None (so callers can skip addWidget).
    """
    if session_info is None:
        return None
    def _text(key: str, fallback: str = "—") -> str:
        value = session_info.get(key)
        return fallback if value is None or value == "" else str(value)

    parts = [
        _text("experimentalist"),
        _text("directory"),
        _text("analyte"),
        _text("technique"),
        f"{_text('n_curves', '?')} curves",
    ]
    lbl = QLabel("  ·  ".join(parts))
    font = lbl.font()
    font.setPointSize(_SMALL_FONT_PT)
    lbl.setFont(font)
    lbl.setStyleSheet(f"color: {_COLOR_MUTED};")
    return lbl


def set_plot_title(plot, title: str = "", caption: str = "") -> None:
    """Set a PlotItem's/PlotWidget's ON-CANVAS title, optionally with a
    smaller, muted settings-provenance caption underneath (what population,
    segment, grid, align mode, etc. produced this plot). Handles all three
    shapes a caller needs:
        title only    → plain bold title
        caption only  → the caption itself, small/muted, as the whole title
        both          → bold title, caption as a second line underneath

    On-canvas deliberately: this is what pyqtgraph's built-in right-click
    Export… captures (axis labels, title, addItem'd elements) — a sibling
    QLabel would look right on screen and simply vanish from any exported
    image. Shared by the 2DH windows, their PCA/Mean-curve popouts, and the
    DistFit/GMM fit windows to keep title and provenance rendering consistent.

    `plot` may be a bare PlotItem (from GraphicsLayoutWidget.addPlot()) or a
    PlotWidget (whose attribute access only delegates *callables* to its
    PlotItem, so plain attributes like titleLabel/layout need this).
    """
    pi = plot.getPlotItem() if hasattr(plot, "getPlotItem") else plot
    if title and caption:
        pi.setTitle(
            f"{title}<br>"
            f"<span style='font-size:{_SMALL_FONT_PT}pt;color:{_COLOR_MUTED}'>{caption}</span>"
        )
        pi.titleLabel.setMaximumHeight(46)
        pi.layout.setRowFixedHeight(0, 46)
    elif caption:
        pi.setTitle(caption, size=f"{_SMALL_FONT_PT}pt", color=_COLOR_MUTED)
    else:
        pi.setTitle(title or None)


def set_si_label(plot, axis: str, text: str, unit: str = "", *,
                 key: str | None = None, si: bool = True) -> None:
    """Label one axis, and settle its SI prefixing in the same call.

    Both `units=` and `enableAutoSIPrefix` must be decided together.
    `setLabel(units="nm")` alone tells pyqtgraph that "nm" is a base unit, so
    it may prefix an already-prefixed value and render 1800 nm as "1.8 knm".
    This helper keeps that policy consistent for every axis.

    So the unit is declared once, in quantities.SI_UNITS, and this converts:

        set_si_label(plot, "bottom", "Piezo", quantities.NM)
        →  setLabel(units="m"); axis.setScale(1e-9)
        →  1800 nm reads "1.8 µm", 44 nm reads "44 nm", 0.26 nm reads "260 pm"

    `setScale` scales tick DRAWING only.  No value moves: ROI positions,
    InfiniteLine positions, mouse coordinates, the fitter and every stored
    bound are in nm before and after.

    `unit` is a quantities unit constant; pass `key=` instead to take the unit
    from a quantity's own registration.  A unit absent from SI_UNITS (Å², a
    ratio, a count) is shown verbatim with prefixing off — see the table there.

    `si=False` PINS an axis to its stated unit.  Use it wherever the same
    number is also shown in a spin box: a spin box cannot carry an SI prefix,
    so an axis free to relabel itself µm while the box beside it says nm would
    make one value read two ways.  That is the case for every axis carrying a
    threshold line the user types into — decomposition's variance panel and
    display_roi's mean-deviation panel — and for nothing else.
    """
    from . import quantities as _quant

    if not unit and key is not None:
        unit = _quant.get(key).shown_unit

    base = _quant.si_for(unit) if si else None
    ax = plot.getAxis(axis)

    if base is not None:
        plot.setLabel(axis, text, units=base.base, unitPower=base.power)
        ax.setScale(base.factor)
        ax.enableAutoSIPrefix(True)      # recomputes against the live range
    else:
        # enableAutoSIPrefix(False) stops FUTURE rescaling but does not undo a
        # prefix already applied — pyqtgraph's updateAutoSIPrefix() never reads
        # the flag, and enableAutoSIPrefix calls it unconditionally on the way
        # past.  On an axis relabelled while its range is already live (the
        # 2DH windows re-apply their labels every time Grid settings change,
        # over a ±2000 nm range) that leaves the prefix frozen at whatever it
        # was: "knm" again, by a different route.  So clear it explicitly.
        ax.enableAutoSIPrefix(False)
        ax.autoSIPrefixScale = 1.0
        ax.setScale(1.0)
        plot.setLabel(axis, text, units=unit or None)   # resets labelUnitPrefix


class CancelableProgress:
    """
    Label + count + Cancel progress dialog for long-running builds and
    rebuilds.

    Deterministic close: callers must call close() (typically in a
    `finally`) rather than relying on the final tick landing exactly on
    `total` — that's what left QProgressDialog stuck open near 100%.
    """

    def __init__(self, parent, label: str, total: int) -> None:
        self._dlg = QProgressDialog(label, "Cancel", 0, total, parent)
        self._dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._dlg.setMinimumDuration(0)
        self._dlg.setValue(0)
        self.cancelled = False

    def tick(self, i: int, total: int, label: str | None = None) -> bool:
        """Advance to i/total (and optionally relabel); returns cancelled."""
        if label is not None:
            self._dlg.setLabelText(label)
        self._dlg.setValue(min(i, total))
        QApplication.processEvents()
        if self._dlg.wasCanceled():
            self.cancelled = True
        return self.cancelled

    def close(self) -> None:
        self._dlg.reset()


class _DateAxis(pg.AxisItem):
    """
    Bottom axis that formats Unix timestamps (seconds since epoch) as
    human-readable date/time strings.

    Inherits from plain AxisItem — NOT DateAxisItem — to avoid PyQtGraph's
    SI-prefix rescaling machinery, which sees values ~1.7e9 and applies a
    scale factor (e.g. 1/3600) before calling tickStrings, corrupting the
    timestamps beyond recovery.

    tickStrings() receives raw tick values and ignores the `scale` parameter
    entirely, calling datetime.fromtimestamp() directly.  Format adapts to
    the tick spacing selected for the visible range:
        < 2 min   → MM-DD HH:MM:SS
        < 2 days  → MM-DD HH:MM
        else      → YYYY-MM-DD
    """

    def __init__(self, **kwargs):
        super().__init__(orientation="bottom", **kwargs)
        self.enableAutoSIPrefix(False)   # keep scale = 1.0 always

    def updateAutoSIPrefix(self):
        pass  # never rescale — timestamps are not physical quantities

    def tickStrings(self, values, scale, spacing):
        # spacing is in the same units as values (seconds), regardless of scale
        if not values:
            return []
        if spacing < 120:
            fmt = "%m-%d %H:%M:%S"
        elif spacing < 172800:
            fmt = "%m-%d %H:%M"
        else:
            fmt = "%Y-%m-%d"
        out = []
        for v in values:
            try:
                out.append(datetime.datetime.fromtimestamp(float(v)).strftime(fmt))
            except (OSError, OverflowError, ValueError):
                out.append("")
        return out
