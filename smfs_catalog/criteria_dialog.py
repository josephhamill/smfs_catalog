# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/criteria_dialog.py
#
# CriteriaDialog — what defines a hit, drawn on the distributions it cuts.
#
# One card per variable: the histogram, a draggable interval ON that histogram,
# and the bounds as numbers.  Dragging the interval IS setting the criterion —
# there is no apply step and no participation switch, because a bound that
# exists gates (criteria_gate's module docstring has the why).  That is how
# flow-cytometry gating works and how faceted search works, for the same
# reason: a cut you cannot see yourself making is a cut you cannot check.
#
# The interval is a LinearRegionItem INSIDE the plot rather than a slider
# under it, so it shares the histogram's axis by construction. A separate
# track would have to be kept aligned, and the first resize that broke the
# alignment would make the card lie about where the bound is.
#
# THE NUMBER IS THE TRUTH, THE HANDLE FOLLOWS.  A dragged handle reports a
# mouse position; a reported bound has to be a number someone can put in a
# methods section.  Both are snapped through quantities.quantize, so the
# handle, the box and the stored bound are one value.
#
# Each card says how many curves have NO value for its variable, because that
# is the moment the user is deciding whether to impose the criterion —
# criteria_cards' docstring has that argument in full.
#
# Two sections: in force, then available.  "What is cutting my data" is the
# question this window is opened to answer, so it is the top of the window,
# not something to find by scrolling.
#
# Membership is derived by criteria_gate.classify — never stored.

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QMainWindow, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from . import criteria_cards as _cards
from . import criteria_gate as _gate
from . import db as _db
from . import quantities as _quant
from . import style
from . import variables as _vars
from .qt_utils import fit_on_screen, set_si_label

# Three columns.  Wide enough that a histogram reads as a shape rather than a
# smear, narrow enough that three fit inside test_window_sizing's laptop width.
_CARD_W = 400
_PLOT_H = 110
_COLUMNS = 3

# Every mark here asks style.py for a ROLE rather than restating a colour.
# The histogram is DATA, so it is neutral grey and split the way the variable
# window splits its own: what the bounds keep drawn over what they cut, so the
# cut is visible in the bars and not only in the handles.
_PEN_NONE     = pg.mkPen(None)
_BRUSH_WITHIN = pg.mkBrush(*style.rgba(style.INK_STRONG, 190))
_BRUSH_CUT    = pg.mkBrush(*style.rgba(style.INK_FAINT, 190))
# The interval is a GUIDE: bold-dashed, with a light fill to read through.
_PEN_GATING   = style.guide_pen(style.LM_THRESHOLD)
_PEN_INACTIVE = style.hair_pen(style.GRID)
_BRUSH_GATING = style.band_brush(style.INK_STRONG, style.A_FILL)
_BRUSH_INERT  = style.band_brush(style.INK_FAINT, style.A_FILL // 3)

_BOUNDS_MEAN = (
    "Setting either bound makes this variable gate the hit. Clearing both "
    "stops it gating; there is nothing else to switch.\n\n"
    "A gating variable is REQUIRED: a curve with no finite value for it "
    "becomes a non-hit even if everything else passes. That matters most for "
    "variables only some curves have — the reload and rupture-separation "
    "distances exist only where an ROI has two or more ruptures, which is why "
    "each card says how many curves have no value."
)


def _criterion_tooltip(key: str) -> str:
    """What the variable means, then what bounding it does.

    The first half is variables.describe() — the same sentence the queue
    header and the scatter axes show, so there is one place to edit it.  The
    second half is about the GATE rather than the variable, which is why it
    is written here and not in the register.
    """
    desc = _vars.describe(key)
    return f"{desc}\n\n———\n\n{_BOUNDS_MEAN}" if desc else _BOUNDS_MEAN


class _CardWidget(QWidget):
    """One variable: its distribution, the interval cutting it, its numbers."""

    # (key, lower, upper) — a bound settled on, ready to store.
    committed = pyqtSignal(str, object, object)
    # A drag in progress: the window refreshes this card's own count only.
    moved = pyqtSignal(str)
    open_detail = pyqtSignal(str, str)

    def __init__(self, card: _cards.Card, parent=None) -> None:
        super().__init__(parent)
        self._card = card
        self._updating = False
        # Before any PlotWidget is built, as every other plot window in the
        # app does it: this is what puts the plot on the house surface
        # instead of pyqtgraph's own black.
        style.apply_plot_defaults()
        self.setFixedWidth(_CARD_W)
        self.setFrameless()

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(3)

        self._title = QLabel(card.label)
        f = self._title.font(); f.setBold(True); self._title.setFont(f)
        self._title.setToolTip(_criterion_tooltip(card.key))
        root.addWidget(self._title)

        self._cost = QLabel("")
        self._cost.setStyleSheet(style.qss_text(style.UI_MUTED))
        self._cost.setFont(style.font(self._cost.font(),
                                      size_pt=style.FONT_SMALL_PT))
        root.addWidget(self._cost)

        self._plot = pg.PlotWidget()
        self._plot.setFixedHeight(_PLOT_H)
        self._plot.setMenuEnabled(False)
        self._plot.hideButtons()
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.getPlotItem().hideAxis("left")
        self._plot.getPlotItem().showAxis("bottom")
        # Pinned: the bound spin boxes beside it cannot carry an SI prefix.
        set_si_label(self._plot, "bottom", "", key=card.key, si=False)
        root.addWidget(self._plot)
        self._draw_histogram()

        # Always present, never hidden. A card with no interval to grab gives
        # the user nothing to drag and no way to START a criterion, which is
        # the opposite of what a window built around dragging should do.
        # Whether it GATES is said by its style, by the two end ticks, and in
        # words on the cost line — three tells, so a faint interval can never
        # be mistaken for a bound at the full range.
        self._region = pg.LinearRegionItem()
        self._region.setZValue(10)
        self._plot.addItem(self._region)
        self._region.sigRegionChanged.connect(self._on_region_moved)
        self._region.sigRegionChangeFinished.connect(self._on_region_settled)

        root.addLayout(self._build_controls())
        self._sync_from_card()

    def _style_region(self, in_force: bool) -> None:
        """Solid and filled when it gates; a faint outline when it does not."""
        brush = _BRUSH_GATING if in_force else _BRUSH_INERT
        pen = _PEN_GATING if in_force else _PEN_INACTIVE
        self._region.setBrush(brush)
        # The region has no pen of its own — its edges are InfiniteLines, and
        # each carries the pen that gets drawn.
        for line in self._region.lines:
            line.setPen(pen)
            line.setHoverPen(_PEN_GATING)
        self._region.update()

    def setFrameless(self) -> None:
        """A card reads as one object, so it gets one border."""
        self.setAutoFillBackground(True)
        self.setStyleSheet(
            f"_CardWidget {{ border: 1px solid {style.GRID}; "
            f"border-radius: 4px; background: {style.SURFACE}; }}")

    # ── Construction ─────────────────────────────────────────────────────────

    def _draw_histogram(self) -> None:
        """The distribution this criterion cuts, with the cut drawn into it.

        A histogram rather than a box or violin: the cut usually goes BETWEEN
        modes, and that is the one thing a box plot cannot show and a violin's
        kernel width can invent.  Binning is histogram_binning's, the same
        geometry the variable window draws.

        Two bar items, kept from that window: what the bounds cut underneath
        in faint ink, what they keep drawn over it in strong ink.  The bars
        themselves then say where the cut falls, so the card still reads at a
        glance when the handles are off the edge of the visible range.
        """
        c = self._card
        self._bars_cut = pg.BarGraphItem(x0=[], x1=[], y0=[], y1=[],
                                         pen=_PEN_NONE, brush=_BRUSH_CUT)
        self._bars_within = pg.BarGraphItem(x0=[], x1=[], y0=[], y1=[],
                                            pen=_PEN_NONE, brush=_BRUSH_WITHIN)
        self._plot.addItem(self._bars_cut)
        self._plot.addItem(self._bars_within)
        span = c.span()
        if span:
            self._plot.setXRange(*span, padding=0.02)
        self.redraw_bars()

    def redraw_bars(self, lower=None, upper=None) -> None:
        """Re-split the bars for the bounds being asked about.

        Counting is one numpy comparison per bin, so this keeps up with a
        drag — the expensive question (is this curve a hit) is a conjunction
        over every criterion and waits for the handle to be let go.
        """
        c = self._card
        if c.bins is None:
            return
        counts = c.bins.count(c.values)
        edges = c.bins.edges
        lo = c.lower if lower is None else lower
        hi = c.upper if upper is None else upper
        centres = 0.5 * (edges[:-1] + edges[1:])
        keep = np.ones(centres.shape, dtype=bool)
        if lo is not None:
            keep &= centres >= lo
        if hi is not None:
            keep &= centres <= hi
        self._bars_cut.setOpts(x0=edges[:-1], x1=edges[1:], y0=0, y1=counts)
        self._bars_within.setOpts(x0=edges[:-1][keep], x1=edges[1:][keep],
                                  y0=0, y1=counts[keep])

    def _build_controls(self) -> QHBoxLayout:
        """Per-end tick + number, then Clear.

        The ticks are how a ONE-SIDED bound is expressed, which dragging a
        handle to the edge cannot say: a handle parked at today's maximum is
        a real stored number and will cut next session's data that lands above
        it, while an unticked end is NULL and cannot cut anything.
        """
        row = QHBoxLayout()
        row.setSpacing(4)
        self._chk_lo = QCheckBox("≥")
        self._chk_hi = QCheckBox("≤")
        self._spin_lo = QDoubleSpinBox()
        self._spin_hi = QDoubleSpinBox()
        for chk, spin in ((self._chk_lo, self._spin_lo),
                          (self._chk_hi, self._spin_hi)):
            spin.setRange(-1e12, 1e12)
            _quant.configure_spinbox(spin, self._card.key, suffix=False)
            spin.setMinimumWidth(96)
            chk.setToolTip("Whether this end is bounded at all. Unticked is "
                           "no bound, not a bound at the edge of the data.")
            row.addWidget(chk)
            row.addWidget(spin, 1)
        self._chk_lo.toggled.connect(lambda on: self._on_end_toggled(on, True))
        self._chk_hi.toggled.connect(lambda on: self._on_end_toggled(on, False))
        self._spin_lo.valueChanged.connect(lambda v: self._on_spin(v, True))
        self._spin_hi.valueChanged.connect(lambda v: self._on_spin(v, False))

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setToolTip(
            "Remove both bounds, so this variable stops gating the hit.\n\n"
            "The numbers are not kept: a bound not in force is not stored, "
            "which is what makes this window the whole answer to what "
            "defines a hit.")
        self._clear_btn.clicked.connect(self._on_clear)
        row.addWidget(self._clear_btn)
        return row

    # ── State ────────────────────────────────────────────────────────────────

    def set_card(self, card: _cards.Card) -> None:
        """Adopt a freshly computed card (the cohort or the bounds changed)."""
        self._card = card
        self._sync_from_card()

    def _sync_from_card(self) -> None:
        """Put every control on the card's bounds, without re-emitting."""
        c = self._card
        span = c.span() or (0.0, 1.0)
        self._updating = True
        try:
            lo = c.lower if c.lower is not None else span[0]
            hi = c.upper if c.upper is not None else span[1]
            self._chk_lo.setChecked(c.lower is not None)
            self._chk_hi.setChecked(c.upper is not None)
            self._spin_lo.setEnabled(c.lower is not None)
            self._spin_hi.setEnabled(c.upper is not None)
            self._spin_lo.setValue(lo)
            self._spin_hi.setValue(hi)
            self._region.setRegion((lo, hi))
            self._style_region(c.in_force)
            self._clear_btn.setEnabled(c.in_force)
        finally:
            self._updating = False
        self.refresh_cost()
        self.redraw_bars()

    def refresh_cost(self, lower=None, upper=None) -> None:
        """The line under the title, from the bounds being asked about."""
        c = self._card
        if lower is None and upper is None:
            self._cost.setText(c.cost_text())
            return
        kept = c.within(lower, upper, proposed=True)
        self._cost.setText(f"{kept:,} of {c.n_curves:,} within"
                           + (f"  ·  {c.n_missing:,} have no value"
                              if c.n_missing else ""))

    def _current(self) -> tuple:
        """The bounds the controls are currently expressing."""
        return (self._spin_lo.value() if self._chk_lo.isChecked() else None,
                self._spin_hi.value() if self._chk_hi.isChecked() else None)

    # ── Interaction ──────────────────────────────────────────────────────────

    def _on_region_moved(self) -> None:
        """Live during a drag: this card's own count only.

        The joint hit count is a conjunction over every criterion and costs a
        full gate pass, so it waits for the handle to be let go. This number
        is one comparison over one array and can keep up with a mouse.
        """
        if self._updating:
            return
        lo, hi = (_quant.quantize(self._card.key, v)
                  for v in self._region.getRegion())
        self._updating = True
        try:
            if not (self._chk_lo.isChecked() or self._chk_hi.isChecked()):
                # Grabbing the interval on a card that does not gate is the
                # gesture for starting a criterion — the alternative is
                # making the user find the ticks first, which is the
                # two-step this window exists to remove.
                for chk, spin in ((self._chk_lo, self._spin_lo),
                                  (self._chk_hi, self._spin_hi)):
                    chk.setChecked(True)
                    spin.setEnabled(True)
                self._style_region(True)
            if self._chk_lo.isChecked():
                self._spin_lo.setValue(lo)
            if self._chk_hi.isChecked():
                self._spin_hi.setValue(hi)
        finally:
            self._updating = False
        self.refresh_cost(*self._current())
        self.redraw_bars(*self._current())
        self.moved.emit(self._card.key)

    def _on_region_settled(self) -> None:
        """Handle released — snap it to the stored value and commit."""
        if self._updating:
            return
        self._on_region_moved()
        lo, hi = self._current()
        self._updating = True
        try:
            self._region.setRegion((
                lo if lo is not None else self._region.getRegion()[0],
                hi if hi is not None else self._region.getRegion()[1]))
        finally:
            self._updating = False
        self.committed.emit(self._card.key, lo, hi)

    def _on_spin(self, _value: float, is_lower: bool) -> None:
        """Typed or stepped: the box is the truth, the handle follows it."""
        if self._updating:
            return
        lo, hi = self._current()
        self._updating = True
        try:
            r = list(self._region.getRegion())
            if is_lower and lo is not None:
                r[0] = lo
            elif not is_lower and hi is not None:
                r[1] = hi
            self._region.setRegion(tuple(sorted(r)))
        finally:
            self._updating = False
        self.refresh_cost(lo, hi)
        self.redraw_bars(lo, hi)
        self.committed.emit(self._card.key, lo, hi)

    def _on_end_toggled(self, on: bool, is_lower: bool) -> None:
        """Tick an end to bound it; untick to leave that side unconstrained."""
        if self._updating:
            return
        (self._spin_lo if is_lower else self._spin_hi).setEnabled(on)
        lo, hi = self._current()
        self._updating = True
        try:
            self._style_region(lo is not None or hi is not None)
        finally:
            self._updating = False
        self.refresh_cost(lo, hi)
        self.redraw_bars(lo, hi)
        self.committed.emit(self._card.key, lo, hi)

    def _on_clear(self) -> None:
        self.committed.emit(self._card.key, None, None)

    def mouseDoubleClickEvent(self, event) -> None:
        """The full variable window: timeseries, fits, exports, exact numbers.

        The card is for seeing the shape and moving the cut quickly. Anything
        that needs the real detail opens the window that already does it well.
        """
        self.open_detail.emit(self._card.key, self._card.label)
        super().mouseDoubleClickEvent(event)


class CriteriaDialog(QMainWindow):
    """The criteria in force, on the distributions they cut."""

    # Re-emitted from a spawned variable window so the dashboard can route a
    # double-clicked file to its singleton worker viewer.
    view_file_requested = pyqtSignal(str)
    # Emitted whenever the gate changes, so the dashboard updates the Hit
    # column, its criteria line and any open population window.
    criteria_changed = pyqtSignal()

    def __init__(
        self,
        variables: list[tuple[str, str]],
        event_paths: list[str],
        db_path:   str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.Window)
        fit_on_screen(self, 1320, 800)
        self._db_path = db_path
        self._variables = variables
        self._event_paths = list(event_paths)
        self._var_wins: list = []      # spawned variable windows, kept from GC
        self._widgets: dict[str, _CardWidget] = {}
        # The queue has one active profile owner, selected by its first row.
        # Use the gate's resolver so the cards and the live verdict can never
        # describe different profiles when the queue holds several owners.
        self._experimentalist = _gate.active_owner(db_path)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        intro = QLabel(
            "Drag an interval to set what counts as a hit — it applies as you "
            "let go. A variable with no bounds doesn’t constrain. Double-click "
            "a card for its full window."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(style.qss_text())
        root.addWidget(intro)

        self._context_label = QLabel("")
        self._context_label.setStyleSheet(style.qss_text(style.UI_TEXT))
        root.addWidget(self._context_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        self._body = QVBoxLayout(inner)
        self._body.setContentsMargins(0, 0, 0, 0)
        self._body.setSpacing(8)
        scroll.setWidget(inner)
        root.addWidget(scroll, 1)

        self._count_lbl = QLabel("")
        f = self._count_lbl.font(); f.setBold(True); self._count_lbl.setFont(f)
        root.addWidget(self._count_lbl)

        self._build_sections()
        self._update_title()
        self.rebuild()

    # ── Building ─────────────────────────────────────────────────────────────

    def _build_sections(self) -> None:
        """The two section headers and their grids, made ONCE.

        Persistent so that moving a card between sections is a reparent
        rather than a teardown — see _relayout.
        """
        self._in_force_hint = QLabel("")
        self._available_hint = QLabel("")
        self._in_force_host = QWidget()
        self._available_host = QWidget()
        for host in (self._in_force_host, self._available_host):
            g = QGridLayout(host)
            g.setContentsMargins(0, 0, 0, 0)
            g.setSpacing(8)
            g.setColumnStretch(_COLUMNS, 1)
        for title, hint, host in (
            ("In force", self._in_force_hint, self._in_force_host),
            ("Available", self._available_hint, self._available_host),
        ):
            lbl = QLabel(title)
            f = lbl.font(); f.setBold(True); lbl.setFont(f)
            self._body.addWidget(lbl)
            hint.setStyleSheet(style.qss_text(style.UI_FAINT))
            self._body.addWidget(hint)
            self._body.addWidget(host)
        self._body.addStretch(1)

    def rebuild(self) -> None:
        """Make a card per variable, from scratch.

        Only for a changed cohort or owner. A bound moving must NOT come
        through here: it would tear down the card under the cursor that set
        it, and a plot destroyed inside an event it is still delivering takes
        the process with it.
        """
        for w in self._widgets.values():
            w.setParent(None)
            w.deleteLater()
        self._widgets.clear()

        for card in _cards.cohort(self._variables, self._event_paths,
                                  self._experimentalist, self._db_path):
            w = _CardWidget(card)
            w.committed.connect(self._on_committed)
            w.moved.connect(self._on_moved)
            w.open_detail.connect(self._open_detail)
            self._widgets[card.key] = w
        self._relayout()
        self.refresh_total()

    def _relayout(self) -> None:
        """Put each card in the section its bounds put it in.

        Cards are MOVED, never rebuilt: removeWidget detaches without
        destroying, so a card can cross sections in the turn after the drag
        that activated it without the plot being deleted under its own
        mouse-release.
        """
        in_force, available = [], []
        for key, _lbl in self._variables:
            w = self._widgets.get(key)
            if w is None:
                continue
            (in_force if w._card.in_force else available).append(w)

        for host, group in ((self._in_force_host, in_force),
                            (self._available_host, available)):
            grid = host.layout()
            while grid.count():
                grid.takeAt(0)
            for i, w in enumerate(group):
                grid.addWidget(w, i // _COLUMNS, i % _COLUMNS)
            host.setVisible(bool(group))

        self._in_force_hint.setText(
            "These define the hit." if in_force else
            "Nothing yet — every event is a hit. Set bounds on a variable "
            "below.")
        self._available_hint.setText(
            "No bounds set, so these do not constrain. Drag an interval or "
            "tick an end to start bounding one." if available else
            "Every variable is bounded.")
        self._sectioned = {w._card.key for w in in_force}

    # ── Live state ───────────────────────────────────────────────────────────

    def _update_title(self) -> None:
        """Keep ownership explicit in the body, where it cannot truncate."""
        who = self._experimentalist or _db.DEFAULT_EXPERIMENTALIST
        self.setWindowTitle("SMFS — hit criteria")
        self._context_label.setText(f"Criteria owner: {who}")

    def set_event_paths(self, event_paths: list[str]) -> None:
        """Update the input cohort (dashboard calls this on queue changes).

        REBUILDS ONLY IF THE COHORT ACTUALLY CHANGED, and this is load-bearing
        rather than an optimisation.  Committing a bound emits
        criteria_changed, the dashboard answers it by fanning out over its
        children, and that fan-out reaches this window and calls this method.
        Rebuilding here would destroy the PlotWidget whose mouse-release is
        still on the stack — a deleted GraphicsScene receiving its own release
        event, which is a segfault, not an exception.
        """
        paths = list(event_paths)
        owner = _gate.active_owner(self._db_path)
        if paths == self._event_paths and owner == self._experimentalist:
            self.refresh()
            return
        self._event_paths = paths
        self._experimentalist = owner
        self._build_sections()
        self._update_title()
        self.rebuild()

    def refresh(self) -> None:
        """Resync the cards to the stored bounds without relaying them out."""
        cards = _cards.cohort(self._variables, self._event_paths,
                              self._experimentalist, self._db_path)
        by_key = {c.key: c for c in cards}
        for key, w in self._widgets.items():
            if key in by_key:
                w.set_card(by_key[key])
        self.refresh_total()

    def refresh_total(self) -> None:
        """The footer: the whole gate's verdict, asked of the gate itself."""
        self._count_lbl.setText(
            _cards.joint([], self._event_paths, self._db_path))

    # ── Interaction ──────────────────────────────────────────────────────────

    def _on_moved(self, _key: str) -> None:
        """A handle is moving. The card updated its own count; nothing else
        may run here, because this fires on every mouse-move event."""

    def _on_committed(self, key: str, lower, upper) -> None:
        """A bound settled. Store it, then tell everyone downstream.

        Stored on release rather than behind an Apply button: the bound IS
        the criterion, so an unsaved bound would be a second state to explain.
        """
        _db.set_threshold(key, lower, upper, "", self._experimentalist,
                          self._db_path)
        w = self._widgets.get(key)
        if w is not None:
            w.set_card(_cards.cohort([(key, w._card.label)], self._event_paths,
                                     self._experimentalist, self._db_path)[0])
            if (key in self._sectioned) != w._card.in_force:
                # It crossed between the sections. Deferred by a turn of the
                # event loop, because this runs inside the mouse-release that
                # set the bound and reparenting the plot under it is not
                # something to do mid-event.
                QTimer.singleShot(0, self._relayout)
        self.refresh_total()
        self.criteria_changed.emit()

    def _open_detail(self, key: str, label: str) -> None:
        from .variable_window import VariableStatsWindow
        win = VariableStatsWindow(
            key, self._event_paths, self._db_path, session_info=None,
            experimentalist=self._experimentalist)
        win.thresholds_changed.connect(self.refresh)
        win.thresholds_changed.connect(self.criteria_changed)
        win.view_file_requested.connect(self.view_file_requested)
        self._var_wins.append(win)
        win.show()
        win.raise_()
        win.activateWindow()
