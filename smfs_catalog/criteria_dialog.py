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
from PyQt6.QtCore import Qt, pyqtSignal
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
from .qt_utils import fit_on_screen

# Two columns of cards.  Wide enough that a histogram reads as a shape rather
# than a smear, narrow enough that two fit beside each other on a laptop.
_CARD_W = 420
_PLOT_H = 110
_COLUMNS = 2

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
        self._plot.setLabel("bottom", "", units=_quant.unit_of(card.key))
        root.addWidget(self._plot)
        self._draw_histogram()

        self._region = pg.LinearRegionItem(
            brush=pg.mkBrush(*style.rgba(style.INK_STRONG, 38)),
            pen=pg.mkPen(style.LM_THRESHOLD, width=2))
        self._region.setZValue(10)
        self._plot.addItem(self._region)
        self._region.sigRegionChanged.connect(self._on_region_moved)
        self._region.sigRegionChangeFinished.connect(self._on_region_settled)

        root.addLayout(self._build_controls())
        self._sync_from_card()

    def setFrameless(self) -> None:
        """A card reads as one object, so it gets one border."""
        self.setAutoFillBackground(True)
        self.setStyleSheet(
            f"_CardWidget {{ border: 1px solid {style.GRID}; "
            f"border-radius: 4px; background: {style.SURFACE}; }}")

    # ── Construction ─────────────────────────────────────────────────────────

    def _draw_histogram(self) -> None:
        """The distribution this criterion cuts, as bars.

        A histogram rather than a box or violin: the cut usually goes BETWEEN
        modes, and that is the one thing a box plot cannot show and a violin's
        kernel width can invent. Binning is histogram_binning's, the same
        geometry the variable window draws.
        """
        c = self._card
        if c.bins is None:
            empty = QLabel("no values in this cohort")
            empty.setStyleSheet(style.qss_text(style.UI_FAINT))
            return
        counts = c.bins.count(c.values)
        self._plot.plot(
            c.bins.edges, counts, stepMode="center", fillLevel=0,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(*style.rgba(style.INK_FAINT, 190)))
        span = c.span()
        if span:
            self._plot.setXRange(*span, padding=0.02)

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
            self._region.setVisible(c.in_force)
            self._clear_btn.setEnabled(c.in_force)
        finally:
            self._updating = False
        self.refresh_cost()

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
            if self._chk_lo.isChecked():
                self._spin_lo.setValue(lo)
            if self._chk_hi.isChecked():
                self._spin_hi.setValue(hi)
        finally:
            self._updating = False
        self.refresh_cost(*self._current())
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
        self.committed.emit(self._card.key, lo, hi)

    def _on_end_toggled(self, on: bool, is_lower: bool) -> None:
        """Tick an end to bound it; untick to leave that side unconstrained."""
        if self._updating:
            return
        (self._spin_lo if is_lower else self._spin_hi).setEnabled(on)
        lo, hi = self._current()
        self._updating = True
        try:
            self._region.setVisible(lo is not None or hi is not None)
        finally:
            self._updating = False
        self.refresh_cost(lo, hi)
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
        fit_on_screen(self, 980, 760)
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

        self._update_title()
        self.rebuild()

    # ── Building ─────────────────────────────────────────────────────────────

    def _section(self, title: str, hint: str) -> None:
        lbl = QLabel(title)
        f = lbl.font(); f.setBold(True); lbl.setFont(f)
        self._body.addWidget(lbl)
        sub = QLabel(hint)
        sub.setStyleSheet(style.qss_text(style.UI_FAINT))
        self._body.addWidget(sub)

    def rebuild(self) -> None:
        """Recompute every card and lay the window out again.

        Called when the cohort or the owner changes. A bound moving does NOT
        come through here — that would tear down the card under the cursor
        that set it.
        """
        while self._body.count():
            item = self._body.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        self._widgets.clear()

        cards = _cards.cohort(self._variables, self._event_paths,
                              self._experimentalist, self._db_path)
        in_force = [c for c in cards if c.in_force]
        available = [c for c in cards if not c.in_force]

        if in_force:
            self._section("In force", "These define the hit.")
            self._body.addWidget(self._grid(in_force))
        else:
            self._section("In force",
                          "Nothing yet — every event is a hit. "
                          "Set bounds on a variable below.")
        self._section(
            "Available",
            "No bounds set, so these do not constrain. "
            "Tick an end to start bounding one.")
        self._body.addWidget(self._grid(available))
        self._body.addStretch(1)
        self.refresh_total()

    def _grid(self, cards: list[_cards.Card]) -> QWidget:
        """One row of cards per _COLUMNS, in the order given."""
        host = QWidget()
        grid = QGridLayout(host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(8)
        for i, card in enumerate(cards):
            w = _CardWidget(card)
            w.committed.connect(self._on_committed)
            w.moved.connect(self._on_moved)
            w.open_detail.connect(self._open_detail)
            grid.addWidget(w, i // _COLUMNS, i % _COLUMNS)
            self._widgets[card.key] = w
        grid.setColumnStretch(_COLUMNS, 1)
        return host

    # ── Live state ───────────────────────────────────────────────────────────

    def _update_title(self) -> None:
        """Keep ownership explicit in the body, where it cannot truncate."""
        who = self._experimentalist or _db.DEFAULT_EXPERIMENTALIST
        self.setWindowTitle("SMFS — hit criteria")
        self._context_label.setText(f"Criteria owner: {who}")

    def set_event_paths(self, event_paths: list[str]) -> None:
        """Update the input cohort (dashboard calls this on queue changes)."""
        self._event_paths = list(event_paths)
        self._experimentalist = _gate.active_owner(self._db_path)
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
        self.refresh_total()
        self.criteria_changed.emit()

    def _open_detail(self, key: str, label: str) -> None:
        from .variable_window import VariableStatsWindow
        win = VariableStatsWindow(
            key, label, self._event_paths, self._db_path, session_info=None,
            experimentalist=self._experimentalist)
        win.thresholds_changed.connect(self.refresh)
        win.thresholds_changed.connect(self.criteria_changed)
        win.view_file_requested.connect(self.view_file_requested)
        self._var_wins.append(win)
        win.show()
        win.raise_()
        win.activateWindow()
