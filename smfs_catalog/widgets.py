# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/widgets.py
#
# Reusable Qt layout and composite-widget helpers shared by multiple windows.
# Analysis-queue transport controls live in navigator_bar.py.

from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect, QSignalBlocker, QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QToolButton, QFrame, QSizePolicy,
    QLabel, QPushButton, QCheckBox, QLayout,
)

from . import sample_marks, style


class FlowLayout(QLayout):
    """A horizontal layout that WRAPS to the next line when it runs out of width.

    This is the fix for control strips defining their window's width.  A
    QHBoxLayout has no wrap: its minimum width is the sum of everything in it,
    and Qt will not shrink a window below that, so a row of eight labelled spin
    boxes silently over-ruled `resize(900, 700)` and opened the decomposition
    window about 2100 px wide -- off the side of a 1920 px screen.  A control
    strip should be bound BY the window, not the other way round.

    FlowLayout reports a minimum width of its widest single item and supplies a
    height-for-width, so a narrow window gets a taller strip instead of a
    horizontal scrollbar or an unreachable edge.  Keep a label and the control
    it names together in one `LabeledControl` so a wrap never separates them.

    Standard Qt "Flow Layout" example, ported to PyQt6.
    """

    def __init__(self, parent=None, margin: int = 0,
                 h_spacing: int = 12, v_spacing: int = 4) -> None:
        super().__init__(parent)
        self._items: list = []
        self._h_space = h_spacing
        self._v_space = v_spacing
        self.setContentsMargins(margin, margin, margin, margin)

    # ── QLayout plumbing ─────────────────────────────────────────────────────

    def addItem(self, item) -> None:          # noqa: N802  (Qt override)
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index):                  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):                  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):            # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:      # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:   # noqa: N802
        return self._layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect) -> None:      # noqa: N802
        super().setGeometry(rect)
        self._layout(rect, test_only=False)

    def sizeHint(self) -> QSize:              # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:           # noqa: N802
        # The WIDEST SINGLE ITEM, not the sum -- this is the whole point.  Any
        # narrower and an item would be clipped; anything wider is negotiable
        # and gets taken out in extra rows.
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    # ── the wrap itself ──────────────────────────────────────────────────────

    def _layout(self, rect: QRect, test_only: bool) -> int:
        """Place items left-to-right, wrapping; returns the total height."""
        m = self.contentsMargins()
        eff = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, row_height = eff.x(), eff.y(), 0

        for item in self._items:
            hint = item.sizeHint()
            # Respect a maximum width.  A status label given long runtime text
            # reports a sizeHint as wide as the text; without this clamp the
            # flow would place it at that width and the strip would grow with
            # its contents -- the exact defect this layout exists to stop.
            cap = item.maximumSize().width()
            if cap and hint.width() > cap:
                hint = QSize(cap, item.heightForWidth(cap)
                             if item.hasHeightForWidth() else hint.height())
            next_x = x + hint.width() + self._h_space
            if row_height and next_x - self._h_space > eff.right():
                # Wrap: this item does not fit on the current row.
                x = eff.x()
                y = y + row_height + self._v_space
                next_x = x + hint.width() + self._h_space
                row_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            row_height = max(row_height, hint.height())

        return y + row_height - rect.y() + m.bottom()


class LabeledControl(QWidget):
    """A caption and the control(s) it names, kept together as one flow item.

    Exists so a FlowLayout wrap can never put "Thr. appr (nm²):" at the end of
    one row and its spin box at the start of the next.
    """

    def __init__(self, label: str, *controls, parent=None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        if label:
            lay.addWidget(QLabel(label))
        for c in controls:
            if c is not None:
                lay.addWidget(c)


class SampleMarksToggle(QCheckBox):
    """The dots/lines switch, for the control strip of any window that draws
    samples.

    The mode is one app-wide setting, so every one of these shows the same
    state: each instance follows sample_marks.changed rather than its own
    window's copy of the answer.  PyQt drops the connection when the checkbox
    is destroyed, so a closed window leaves nothing behind.
    """

    def __init__(self, parent=None) -> None:
        super().__init__("Dots", parent)
        self.setToolTip(
            "Draw each sample as a dot instead of joining them with a line.\n"
            "A line through a held segment draws motion that never happened; "
            "dots show where the samples actually are.\n"
            "Lines pan and zoom faster on a large curve."
        )
        self.setChecked(sample_marks.dots())
        self.toggled.connect(sample_marks.set_dots)
        sample_marks.changed.connect(self._follow)

    def _follow(self, on: bool) -> None:
        if self.isChecked() != on:
            with QSignalBlocker(self):
                self.setChecked(on)


class CollapsibleSection(QWidget):
    # Emitted on collapse/expand (True = expanded) so a parent splitter can
    # redistribute pane sizes — hiding the body alone doesn't free splitter space.
    toggled = pyqtSignal(bool)
    """
    A titled section whose body can be hidden/shown by clicking the header.
    Click the title arrow ▸ to collapse; click ▾ to expand.

    Usage:
        sec = CollapsibleSection("Scope filters")
        sec.body_layout.addWidget(some_widget)
        parent_layout.addWidget(sec)
    """

    def __init__(self, title: str, expanded: bool = True, parent=None) -> None:
        super().__init__(parent)

        self._toggle = QToolButton(self)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setStyleSheet(style.QSS_COLLAPSIBLE_HEADER)
        self._toggle.toggled.connect(self._on_toggled)

        self._body = QFrame(self)
        self.body_layout = QVBoxLayout(self._body)
        self.body_layout.setContentsMargins(8, 4, 0, 4)
        self._body.setVisible(expanded)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._toggle)
        outer.addWidget(self._body)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._apply_collapse_height(expanded)

    # Qt's "max widget size" sentinel — restores unconstrained height.
    _UNCONSTRAINED = 16777215

    def _apply_collapse_height(self, expanded: bool) -> None:
        """
        Collapse must actually free space (inside a splitter, hiding the body
        alone leaves the section's old size reserved).  When collapsed we cap
        the whole section to the header height so siblings reclaim the room;
        when expanded we lift the cap so it can grow / be resized again.
        """
        if expanded:
            self.setMaximumHeight(self._UNCONSTRAINED)
        else:
            self.setMaximumHeight(self._toggle.sizeHint().height() + 4)

    def is_expanded(self) -> bool:
        return self._toggle.isChecked()

    def header_height(self) -> int:
        return self._toggle.sizeHint().height() + 4

    def _on_toggled(self, expanded: bool) -> None:
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self._body.setVisible(expanded)
        self._apply_collapse_height(expanded)
        self.toggled.emit(expanded)


class ClusterColourBar(QWidget):
    """The 'Colour by cluster' control, shared by every window that offers it.

    One implementation because there are three consumers (#63: Explore Events,
    the any-vs-any scatter, the variable timeseries) and three copies of a
    checkbox plus a caption plus a subscribe/unsubscribe pair is three
    copies to keep in step.

    It owns no cluster data.  It reads clustering.current(), reports coverage
    for whatever cohort its host passes in, and emits `changed` when the host
    should repaint — whether that is because the user ticked the box or
    because a k-means run in another window published new labels.

    Disabled with an explanatory tooltip when no clustering exists, so there
    is never a state where the user is wondering why the option is missing.
    """

    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        from . import clustering as _clustering
        self._clustering = _clustering

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self._chk = QCheckBox("Colour by cluster")
        self._chk.setChecked(False)
        self._chk.toggled.connect(lambda _c: self.changed.emit())
        lay.addWidget(self._chk)

        self._caption = QLabel("")
        self._caption.setFont(style.font(
            self._caption.font(), size_pt=style.FONT_SMALL_PT))
        self._caption.setStyleSheet(style.qss_text())
        lay.addWidget(self._caption, 1)

        self._clear_btn = QPushButton("Clear clusters")
        self._clear_btn.setToolTip(
            "Forget the current cluster assignment.\n\n"
            "Useful mid-session: change the criteria or repopulate the queue "
            "and a clustering computed before that no longer describes the "
            "cohort. Nothing clears it automatically — the coverage line "
            "reports the mismatch and you decide."
        )
        self._clear_btn.clicked.connect(self._on_clear)
        lay.addWidget(self._clear_btn)

        self._clustering.subscribe(self._on_registry_changed)
        self.refresh([])

    # ── state ────────────────────────────────────────────────────────────────

    def is_active(self) -> bool:
        """Ticked AND there is actually something to colour by."""
        return self._chk.isChecked() and self._clustering.current() is not None

    def refresh(self, paths: list[str]) -> None:
        """Re-read the registry and restate coverage for this cohort.

        `paths` is the host's CURRENT cohort, which is rarely the one that was
        clustered — the 2DH holds one population, Explore Events shows both —
        so coverage is computed per host rather than stored on the run.
        """
        c = self._clustering.current()
        self._chk.setEnabled(c is not None)
        self._clear_btn.setEnabled(c is not None)
        if c is None:
            self._chk.setToolTip(
                "Run k-means first: open a 2DH from Explore Events, Run PCA, "
                "then the K-means tab.\n\n"
                "The clustering lasts for this session only. Export anything "
                "you want to keep — every export gains a cluster column."
            )
            self._caption.setText("no clustering in this session")
            return
        self._chk.setToolTip(
            "Colour every mark by the cluster its curve was assigned.\n\n"
            "The clustering ran on the 2DH ENSEMBLE, not on the values plotted "
            "here — so these colours are that result projected back, not "
            "clustering in this space."
        )
        cover = self._clustering.coverage_text(paths) if paths else ""
        self._caption.setText(
            c.describe() + (f" · {cover}" if cover else ""))

    def legend_text(self, paths: list[str]) -> str:
        """The same facts for an on-canvas caption, which survives an image
        export where a sibling QLabel does not."""
        c = self._clustering.current()
        if c is None or not self.is_active():
            return ""
        cover = self._clustering.coverage_text(paths)
        return f"clustered on the 2DH ensemble — {c.describe()} · {cover}"

    def _on_clear(self) -> None:
        self._clustering.clear()

    def _on_registry_changed(self) -> None:
        # A k-means run in another window landed. Repaint, but never tick the
        # box on the user's behalf — recolouring a plot they are reading, with
        # no action from them, is the surprise this avoids.
        self.changed.emit()

    def closeEvent(self, event):
        self._clustering.unsubscribe(self._on_registry_changed)
        super().closeEvent(event)

    def detach(self) -> None:
        """Hosts call this from their own closeEvent: a subscriber list holding
        a deleted widget raises on the next publish, and the traceback points
        at whichever window happened to be first rather than at this one."""
        self._clustering.unsubscribe(self._on_registry_changed)
