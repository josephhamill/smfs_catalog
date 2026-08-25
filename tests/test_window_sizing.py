# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Regression test: a window is bound BY the screen, it does not define its own
size.

WHAT WENT WRONG.  Every window opened with a hard-coded `self.resize(W, H)`,
chosen on a large monitor.  That is only half a size request, and it was the
half that does not win: Qt will not shrink a window below its layout's
`minimumSizeHint()`, which is computed bottom-up from the children.  So a
control strip built as one non-wrapping QHBoxLayout silently over-ruled the
resize:

    decomposition_window   asked 900 px, minimum was 2054 px
    event_summary_window   asked 1100 px, minimum was 1822 px
    dashboard_window       asked 1400 px, minimum was 1503 px

On a 1920x1080 screen those open with their right-hand buttons off the edge and
their bottom row under the taskbar.  `setFixedWidth` made it worse — it is an
absolute veto that removes Qt's ability to negotiate at all.

THE CONTRACT, in three parts:

(a) No window module calls `resize()` with a literal size.  `fit_on_screen()`
    clamps to the available geometry (which already excludes the panel), so the
    opening size can never exceed the screen it lands on.

(b) A FlowLayout is never sent a QBoxLayout-only method.  `QLayout` has no
    `addStretch`, `addSpacing` or `addLayout`; sending one raises AttributeError
    at window-construction time, which means the window simply does not open.
    This check exists because that mistake was made twice during the conversion
    itself — `addStretch()` at the end of a converted row is the easy one to
    miss, because it sits far below the line that changed.

(c) The windows that can be built headlessly stay under a laptop's width.
    1440 px is the bound: it leaves a 1920 px screen comfortable and still fits
    the 1600 px-wide machines in the lab.  This is the check that would have
    caught the original defect; (a) and (b) only protect the mechanism.

Run with the smfs-catalog env, from the repo root:
    python -m pytest tests/test_window_sizing.py -q
"""
import os
import re
import sys
import pathlib
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import checkstyle

check = checkstyle.CheckRunner()

PKG = pathlib.Path(__file__).resolve().parent.parent / "smfs_catalog"

# Modules that own a top-level window or dialog.  qt_utils is where the helper
# itself lives, so it is the one place allowed to call resize().
WINDOW_FILES = sorted(
    p for p in PKG.glob("*.py")
    if p.name.endswith(("_window.py", "_dialog.py")) or p.name == "display_roi.py"
)

check("there are window modules to inspect", len(WINDOW_FILES) > 15,
      f"found only {len(WINDOW_FILES)}")


# ── (a) nobody calls resize() with a literal size ────────────────────────────

_LITERAL_RESIZE = re.compile(r"\.resize\(\s*\d")

offenders = []
for path in WINDOW_FILES:
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if _LITERAL_RESIZE.search(line):
            offenders.append(f"{path.name}:{n}  {line.strip()}")

check(
    "no window module calls resize() with a literal size",
    not offenders,
    "these must go through qt_utils.fit_on_screen(), which clamps to the "
    "screen's available geometry:\n  " + "\n  ".join(offenders),
)


# ── (b) no QBoxLayout-only method is sent to a FlowLayout ────────────────────

_BOX_ONLY = ("addStretch", "addSpacing", "addLayout")

misuse = []
for path in PKG.glob("*.py"):
    src = path.read_text()
    if "FlowLayout(" not in src:
        continue
    # Every local name bound to a FlowLayout in this file.
    flow_vars = set(re.findall(r"(\w+)\s*=\s*FlowLayout\(", src))
    for var in flow_vars:
        for meth in _BOX_ONLY:
            for m in re.finditer(rf"\b{re.escape(var)}\.{meth}\(", src):
                line_no = src[:m.start()].count("\n") + 1
                misuse.append(f"{path.name}:{line_no}  {var}.{meth}()")

check(
    "no FlowLayout is sent addStretch/addSpacing/addLayout",
    not misuse,
    "QLayout has none of these; the window raises AttributeError and never "
    "opens. Wrap a nested layout in a QWidget, and drop the stretch — a flow "
    "packs left already:\n  " + "\n  ".join(misuse),
)


# ── (b2) a numeric input is never pinned to a pixel width ────────────────────
#
# A QSpinBox/QDoubleSpinBox computes its own width from the font, its range,
# its decimals and its suffix, so it is correct at any DPI.  setFixedWidth()
# replaces that with a constant measured on one machine's font -- which is why
# "Trim pts: 100" rendered as "1·" and "Prominence: 0.100" as "0.1(" on a
# Windows box at a different scaling.  setMinimumWidth is fine: Qt takes the
# larger of it and the widget's own hint, so it can still grow.

spin_names = set()
for path in PKG.glob("*.py"):
    src = path.read_text()
    spin_names |= set(re.findall(r"self\.(\w+)\s*=\s*Q(?:Double)?SpinBox\(", src))

pinned = []
for path in PKG.glob("*.py"):
    src = path.read_text()
    for name in sorted(spin_names):
        for m in re.finditer(rf"self\.{re.escape(name)}\.setFixedWidth\((\d+)\)", src):
            line_no = src[:m.start()].count("\n") + 1
            pinned.append(f"{path.name}:{line_no}  {name}.setFixedWidth({m.group(1)})")

check(
    "no spin box is pinned to a fixed pixel width",
    not pinned,
    "these clip their own digits at another font size or DPI; delete the call "
    "and let the widget size itself, or use setMinimumWidth:\n  "
    + "\n  ".join(pinned),
)


# ── (c) the buildable windows fit a laptop ───────────────────────────────────

MAX_WIDTH = 1440
# Height matters as much as width and is worse when it fails: a window taller
# than the screen puts its title bar above the desktop, and there is then
# nothing left to grab to drag it back.  900 leaves room for a taskbar on a
# 1080 screen.  The dashboard's minimum was 1200 px until its splitter
# sections were given an explicit floor.
MAX_HEIGHT = 900

from PyQt6.QtWidgets import QApplication          # noqa: E402
app = QApplication.instance() or QApplication(sys.argv)

from smfs_catalog import db as _db                # noqa: E402

_tmp = tempfile.mkdtemp(prefix="smfs_sizing_")
_DB = os.path.join(_tmp, "sizing.db")
_db.initialise(_DB)

from smfs_catalog import dashboard_window         # noqa: E402
from smfs_catalog import decomposition_window     # noqa: E402
from smfs_catalog import display_roi              # noqa: E402
from smfs_catalog import event_summary_window     # noqa: E402

CASES = [
    ("DashboardWindow",     lambda: dashboard_window.DashboardWindow(_DB)),
    ("DecompositionWindow", lambda: decomposition_window.DecompositionWindow(_DB)),
    ("ROIWindow",           lambda: display_roi.ROIWindow(_DB)),
    ("EventSummaryWindow",  lambda: event_summary_window.EventSummaryWindow([], _DB)),
]

for name, build in CASES:
    try:
        win = build()
        win.show()
        app.processEvents()
        hint = win.minimumSizeHint()
        width, height = hint.width(), hint.height()
        win.close()
    except Exception as exc:                       # pragma: no cover
        check(f"{name} can be constructed headlessly", False, repr(exc))
        continue
    check(
        f"{name} minimum width fits a laptop ({width} px)",
        width <= MAX_WIDTH,
        f"{name} cannot be made narrower than {width} px, so it opens with "
        f"its right-hand edge off a {MAX_WIDTH} px screen. Look for a control "
        f"strip built as one QHBoxLayout — widgets.FlowLayout wraps instead.",
    )
    check(
        f"{name} minimum height fits a laptop ({height} px)",
        height <= MAX_HEIGHT,
        f"{name} cannot be made shorter than {height} px, so its title bar "
        f"lands above the top of a {MAX_HEIGHT}-ish work area and the window "
        f"cannot be dragged back. A vertical QSplitter's minimum is the SUM "
        f"of its children's — give each section an explicit minimum height.",
    )


test_check = checkstyle.pytest_cases(check)
