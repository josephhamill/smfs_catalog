# -*- mode: python ; coding: utf-8 -*-
# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.
#
# PyInstaller build recipe — the standalone app colleagues are handed (#111).
#
#     conda activate smfs-catalog
#     python scripts/write_build_stamp.py    # identity, see below
#     pyinstaller pyinstaller.spec --noconfirm
#     rm smfs_catalog/_build_stamp.py        # never commit it
#
# ONE SPEC, THREE MACHINES.  PyInstaller cannot cross-compile: a build only
# ever produces an executable for the OS it ran on.  So this file is run by
# hand on Lemaitre (Debian), Englert (Windows 11) and Damien's Mac, and every
# per-OS difference lives here as a `sys.platform` branch — not as three spec
# files that drift, for the same reason environment.yml is one dependency list.
#
# THE BUILD STAMP IS NOT OPTIONAL IN A RELEASE.  A frozen build never falls
# back to git (tests/test_frozen_build.py explains why), so without
# _build_stamp.py the app runs with code_version() is None — which is honest,
# but disables the analysis cache entirely, and hands the user results no
# figure can be traced back to.  Building without it is fine for a smoke test
# and wrong for anything shipped.  All three machines must stamp the SAME
# commit, or the three builds are three different applications.

import sys
from pathlib import Path

APP_NAME = "SMFS-Catalog"
ROOT = Path(SPECPATH)                        # also what a dev run puts on sys.path
ICONS = ROOT / "smfs_catalog" / "assets" / "icons"

# resource_path() in run_dashboard.py joins onto sys._MEIPASS, so the layout
# inside the bundle has to mirror the checkout exactly: smfs_catalog/assets/…
datas = [(str(ROOT / "smfs_catalog" / "assets"), "smfs_catalog/assets")]
binaries = []
hiddenimports = []

# NO collect_all() HERE, for pyqtgraph least of all.  pyqtgraph loads its
# Qt-version-specific UI templates by name, which static analysis cannot see —
# but PyInstaller's own hook already collects exactly those, and skips
# `pyqtgraph.examples`, which constructs a QApplication on import and so kills
# the build outright on a headless machine (that is how this was found:
# collect_all here, exit code -6, "could not connect to display").  scipy,
# numpy, scikit-learn and PyQt6 likewise have hooks that need nothing from us.

# Kept OUT of the bundle deliberately — see environment.yml's absent-list.
# matplotlib and pandas belong to legacy/ and standalone_hist_fit/, which are
# not the app; the rest are things a GUI drags in by accident and then has to
# ship.  Excluding them is checked, not assumed: the app imports cleanly with
# all of them blocked at the import hook.
excludes = [
    "matplotlib",
    "pandas",
    "anthropic",
    "rich",
    "questionary",
    "tkinter",
    "PyQt5",
    "PySide2",
    "PySide6",
    "IPython",
    "pytest",
    "sqlalchemy",
]

a = Analysis(
    [str(ROOT / "run_dashboard.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

# Windows and macOS want their own icon format; PyInstaller ignores the
# parameter on Linux, where the desktop reads the icon from a .desktop file
# rather than from the executable.
if sys.platform == "win32":
    icon = str(ICONS / "icon.ico")
elif sys.platform == "darwin":
    icon = str(ICONS / "icon.icns")
else:
    icon = None

# --onefile: one file to send a colleague, which is the whole point of the
# issue.  The cost is a few seconds of unpacking at every launch; --onedir
# starts faster but is a folder that has to be zipped, and arrives as a
# directory of DLLs with an .exe somewhere inside it.
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX-packed binaries are a reliable antivirus false positive
    runtime_tmpdir=None,
    console=False,      # --windowed: no terminal behind the GUI
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon,
)

if sys.platform == "darwin":
    app = BUNDLE(
        exe,
        name=f"{APP_NAME}.app",
        icon=icon,
        # macOS's internal name for the app: it files preferences and
        # permissions under this, so two apps sharing one are treated as the
        # same app.  Reverse-DNS of the project's own home rather than an
        # institution's — the copyright is Joseph Hamill's, not a university's.
        bundle_identifier="io.github.josephhamill.smfs-catalog",
    )
