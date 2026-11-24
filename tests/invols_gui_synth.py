# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Shared helpers for the invOLS window and dashboard tests: the plan's four-file
fixture as a temporary catalog, a pin for the code identity, and readers that
look at a window the way a user does (button texts, label texts, the plot's
spots) without naming the window's private attributes.

Every catalog is a temporary file passed explicitly; nothing here touches the
app's own catalog.  Expected numbers are hand calculations (199.14 / 0.95 and
so on), never read back from the code under test.
"""
from __future__ import annotations

import datetime
import html
import importlib
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication as _QApplication  # noqa: E402

# The app must exist before any smfs_catalog module is imported: some of them
# build QObjects at import time, which fails when no application exists yet.
_APP = _QApplication.instance() or _QApplication([])

import invols_synth as synth  # noqa: E402
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import (
    QAbstractButton, QApplication, QComboBox, QGroupBox, QLabel, QPlainTextEdit,
    QPushButton, QTextEdit, QWidget,
)

import pyqtgraph as pg

from smfs_catalog import db

CODE = synth.CODE
INSTRUMENT = 199.14
SLOPES = (-0.95, -1.0, -1.05, None)
OWN = {1: 209.62105, 2: 199.14, 3: 189.65714}      # 199.14 / |slope|
MEAN_3 = 199.47273
TRIANGLES = {"t", "t1", "t2", "t3"}
BANNED = ("accurate", "calibrated", "true invols", "corrected")
PINNED_MODULES = ("smfs_catalog.provenance", "smfs_catalog.invols_store",
                  "smfs_catalog.curve_analysis", "smfs_catalog.roi_pipeline",
                  "smfs_catalog.rawcurve_window", "smfs_catalog.invols_window",
                  "smfs_catalog.dashboard_window")


def pin_code_version(monkeypatch):
    """The code identity every module stamps and checks, fixed so the tests
    need no git checkout (invols_synth writes its rows under the same text)."""
    for name in PINNED_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        monkeypatch.setattr(module, "cache_version", lambda: CODE,
                            raising=False)


# ── the catalog ──────────────────────────────────────────────────────────────

def hour(n: int) -> float:
    """Unix seconds of 2026-03-18 `n`:00 local time."""
    return datetime.datetime(2026, 3, 18, n).timestamp()


def stamp(n: int) -> str:
    return f"2026-03-18 {n:02d}:00:00"


@dataclass
class Catalog:
    db: str
    ids: list
    paths: list


def _instrument_list(instruments, n):
    """One instrument value per file: a number is repeated, None means the
    default."""
    if instruments is None:
        instruments = INSTRUMENT
    if isinstance(instruments, (list, tuple)):
        return list(instruments)
    return [instruments] * n


def _file_specs(slopes, instruments, hours):
    n = len(slopes)
    if hours == "auto":
        hours = [10 + i for i in range(n)]
    return [{"instrument": instruments[i], "slope": slopes[i],
             "measured_at": None if hours[i] is None else stamp(hours[i]),
             "queued": False} for i in range(n)]


def _write_users(db_path, ids, users):
    con = sqlite3.connect(db_path)
    try:
        with con:
            con.execute("UPDATE files SET parse_ok = 1")
            for file_id, user in zip(ids, users or []):
                con.execute("UPDATE files SET experimentalist = ? "
                            "WHERE id = ?", (user, file_id))
    finally:
        con.close()


def build(tmp_path, slopes=SLOPES, instruments=None, hours="auto",
          **extra) -> Catalog:
    """The plan's fixture, generalised.  `instruments` is one number or a list
    (default 199.14); `hours` is a list of acquisition hours, None meaning no
    acquisition time (default 10, 11, 12, ...).  `extra` may give `users`
    (files.experimentalist) and `rms` (a list of stored invols_rms values).  No
    file is queued, every file is eligible for a dashboard scope, and the
    stored slope rows are stamped as made under the fit settings in force."""
    assert set(extra) <= {"users", "rms"}, f"unknown options {extra}"
    n = len(slopes)
    specs = _file_specs(slopes, _instrument_list(instruments, n), hours)
    db_path, ids = synth.make_catalog(tmp_path, specs)
    _write_users(db_path, ids, extra.get("users"))
    synth.stamp_fit_key(db_path, ids)
    for file_id, value in zip(ids, extra.get("rms") or []):
        if value is not None:
            synth.add_result(db_path, file_id, "invols_rms", value)
    return Catalog(db_path, ids, [synth.path_of(i + 1) for i in range(n)])


def dump(db_path) -> dict:
    """Every row of every table, for a before/after comparison."""
    con = sqlite3.connect(db_path)
    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        return {t: sorted(map(repr, con.execute(f"SELECT * FROM {t}")))
                for t in tables}
    finally:
        con.close()


# ── reading a window ─────────────────────────────────────────────────────────

def qapp():
    return QApplication.instance() or QApplication([])


def pump(times: int = 3) -> None:
    for _ in range(times):
        QApplication.processEvents()


def _plain(text: str) -> str:
    text = re.sub(r"<br\s*/?>|</p>|</li>|</div>|</tr>", "\n", text or "")
    return html.unescape(re.sub(r"<[^>]+>", "", text))


def _combo_texts(combo, tooltips: bool) -> list[str]:
    out = []
    for i in range(combo.count()):
        out.append(combo.itemText(i))
        if tooltips:
            out.append(str(combo.itemData(i, Qt.ItemDataRole.ToolTipRole) or ""))
    return out


def _widget_texts(w, tooltips: bool) -> list[str]:
    if isinstance(w, (QLabel, QAbstractButton)):
        out = [w.text()]
    elif isinstance(w, QGroupBox):
        out = [w.title()]
    elif isinstance(w, (QTextEdit, QPlainTextEdit)):
        out = [w.toPlainText()]
    elif isinstance(w, QComboBox):
        out = _combo_texts(w, tooltips)
    else:
        out = []
    return out + [w.toolTip()] if tooltips else out


def texts(win, tooltips: bool = False) -> list[str]:
    """Every piece of text a user can read in the window: titles, labels,
    buttons, group titles, text areas and combo items; with `tooltips`, the
    hover texts too."""
    out = [win.windowTitle()]
    for w in [win] + win.findChildren(QWidget):
        out.extend(_widget_texts(w, tooltips))
    return [_plain(t) for t in out if t]


def screen(win) -> str:
    """The visible text of the window, one block."""
    return "\n".join(texts(win))


def everything(win) -> str:
    """Visible text plus every tooltip."""
    return "\n".join(texts(win, tooltips=True))


def lines_with(text: str, needle: str) -> list[str]:
    return [ln for ln in text.splitlines() if needle in ln]


def button(win, prefix: str) -> QPushButton:
    found = [b for b in win.findChildren(QPushButton)
             if b.text().replace("&", "").startswith(prefix)]
    assert len(found) == 1, f"expected one button starting {prefix!r}, got {len(found)}"
    return found[0]


def apply_button(win) -> QPushButton:
    return button(win, "Apply")


def open_button(win) -> QPushButton:
    return button(win, "Open this trace")


def plot_of(win):
    from smfs_catalog.invols_plot import InvolsPlot
    found = win.findChildren(InvolsPlot)
    assert len(found) == 1, "the window must hold exactly one InvolsPlot"
    return found[0]


def controls_of(win):
    from smfs_catalog.invols_fit_controls import InvolsFitControls
    found = win.findChildren(InvolsFitControls)
    assert len(found) == 1, "the window must hold exactly one InvolsFitControls"
    return found[0]


# ── reading the plot (the way pyqtgraph exposes it) ──────────────────────────

def _items(plot):
    return list(plot.getViewBox().childGroup.childItems())


def spots(plot):
    return [pt for it in _items(plot) if isinstance(it, pg.ScatterPlotItem)
            for pt in it.points()]


def points(plot):
    return [p for p in spots(plot) if p.symbol() not in TRIANGLES]


def is_filled(spot) -> bool:
    brush = spot.brush()
    return (brush.style() != Qt.BrushStyle.NoBrush
            and brush.color().alpha() > 0)


def pen_hex(spot) -> str:
    return spot.pen().color().name().lower()


def screen_pos(plot, t: float, y: float) -> QPoint:
    scene = plot.getViewBox().mapViewToScene(QPointF(t, y))
    return plot.mapFromScene(scene)


def send(plot, kind, pos, button_, buttons):
    vp = plot.viewport()
    ev = QMouseEvent(kind, QPointF(pos), QPointF(vp.mapToGlobal(pos)), button_,
                     buttons, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(vp, ev)
    QApplication.processEvents()


def click_point(plot, spot) -> None:
    """A real left click on a drawn spot."""
    pos = screen_pos(plot, spot.pos().x(), spot.pos().y())
    left = Qt.MouseButton.LeftButton
    send(plot, QEvent.Type.MouseButtonPress, pos, left, left)
    send(plot, QEvent.Type.MouseButtonRelease, pos, left,
         Qt.MouseButton.NoButton)


def spot_for_path(plot, path: str):
    name = Path(path).name
    found = [p for p in points(plot) if p.data()["name"] == name]
    assert len(found) == 1, f"no single drawn spot for {name}"
    return found[0]
