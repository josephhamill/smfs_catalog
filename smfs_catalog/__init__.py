# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/__init__.py

# The app's version, recorded in every export manifest (provenance.
# app_version) so an exported figure can be traced back to the code that
# produced it.
#
# Declared here rather than read from pyproject.toml at runtime because it
# has to work in all three cases: a git checkout (no package metadata), an
# installed package, and a PyInstaller bundle (no pyproject.toml shipped).
# It MUST match pyproject.toml's `version`; tests/test_export_convention.py
# fails if the two drift apart.
__version__ = "1.3.0"
