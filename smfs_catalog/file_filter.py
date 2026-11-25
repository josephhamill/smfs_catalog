# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""The one object that says which catalogued files a query selects."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional


@dataclass(frozen=True)
class FileFilter:
    """Every field defaults to None, meaning "no constraint"."""

    users: Optional[list] = None
    analytes: Optional[list] = None
    solvents: Optional[list] = None
    techniques: Optional[list] = None
    substrates: Optional[list] = None
    sample_preps: Optional[list] = None
    afm_units: Optional[list] = None
    curve_types: Optional[list] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    search: Optional[str] = None
    parse_ok: Optional[bool] = None
    usable: Optional[bool] = None
    unique: Optional[bool] = None
    directory: Optional[str] = None
    directories: Optional[list] = None
    curve_type: Optional[str] = None
    quality: Optional[str] = None
    k_min: Optional[float] = None
    k_max: Optional[float] = None

    def set_fields(self) -> dict:
        """The fields that carry a constraint; empty when nothing is set."""
        return {f.name: getattr(self, f.name) for f in fields(self)
                if getattr(self, f.name) is not None}
