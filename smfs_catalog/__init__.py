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
# The only declaration of the version. A literal because the PyInstaller
# bundle ships no pyproject.toml and a checkout has no package metadata.
__version__ = "1.4.0"
