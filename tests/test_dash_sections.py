# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Regression test: collapsing a dashboard section and expanding it again leaves
the three sections where they were.

Run with the smfs-catalog env, from the repo root:
    python -m pytest tests/test_dash_sections.py -q
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
app = QApplication.instance() or QApplication(sys.argv)

from smfs_catalog import db as _db
from smfs_catalog.dashboard_window import DashboardWindow, _MIN_SECTION_H

import checkstyle                                          # noqa: E402

check = checkstyle.CheckRunner()

tmp = tempfile.mkdtemp(prefix="dashsec_")
DB = os.path.join(tmp, "test.sqlite")
_db.initialise(DB)

win = DashboardWindow(DB)
win.resize(1200, 900)
win.show()

SPLIT = win._splitter
SECS = win._sections
HEADS = [s.header_height() for s in SECS]


def settle():
    # Four turns: the toggle's own layout, the deferred redistribution, and
    # the layout that follows it.
    for _ in range(4):
        app.processEvents()


def toggle(i: int, on: bool) -> list[int]:
    SECS[i]._toggle.setChecked(on)
    settle()
    return SPLIT.sizes()


def fits(sizes) -> bool:
    """No section is off the bottom: the panes fill the splitter, no more."""
    return sum(sizes) == sum(SPLIT.sizes()) and all(h > 0 for h in sizes)


def usable(sizes) -> bool:
    return all(h >= _MIN_SECTION_H for h in sizes)


settle()
start = SPLIT.sizes()
check("the dashboard opens with three usable sections",
      len(start) == 3 and usable(start), str(start))

# ── one section at a time ────────────────────────────────────────────────────

for i, name in enumerate(("scope", "database", "queue")):
    collapsed = toggle(i, False)
    check(f"collapsing {name} frees all but its header",
          collapsed[i] <= HEADS[i], str(collapsed))
    check(f"collapsing {name} leaves the others usable and on screen",
          fits(collapsed) and all(h >= _MIN_SECTION_H
                                  for j, h in enumerate(collapsed) if j != i),
          str(collapsed))
    back = toggle(i, True)
    check(f"expanding {name} restores the layout it left",
          all(abs(a - b) <= 4 for a, b in zip(back, start)),
          f"{start} -> {back}")

# ── all three, which is where the splitter loses its own height ─────────────

for i in range(3):
    toggle(i, False)
allc = SPLIT.sizes()
check("all three collapse to their headers",
      all(h <= HEADS[i] for i, h in enumerate(allc)), str(allc))

for i in range(3):
    expanded = toggle(i, True)
    check(f"expanding section {i} of three gives it room to be read",
          expanded[i] >= _MIN_SECTION_H, str(expanded))
check("expanding all three restores the opening layout",
      all(abs(a - b) <= 8 for a, b in zip(SPLIT.sizes(), start)),
      f"{start} -> {SPLIT.sizes()}")

# ── a height the user chose survives a collapse ─────────────────────────────

dragged = [500, 150, sum(start) - 650]
SPLIT.setSizes(dragged)
SPLIT.splitterMoved.emit(0, 1)
settle()
dragged = SPLIT.sizes()
toggle(1, False)
after = toggle(1, True)
check("a collapse and expand keeps the heights the user dragged to",
      all(abs(a - b) <= 4 for a, b in zip(after, dragged)),
      f"{dragged} -> {after}")

# ── and in a window too short to give every section its wish ────────────────

win.resize(1200, 420)
settle()
toggle(0, False)
small = toggle(0, True)
check("a short window still expands into three usable sections",
      usable(small) and fits(small), str(small))

win.close()
print()


test_check = checkstyle.pytest_cases(check)
