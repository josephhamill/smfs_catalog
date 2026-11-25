# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

# smfs_catalog/plot_context.py
#
# The settings a plot window is handed by whoever opened it, grouped so each
# window constructor takes a few objects instead of a long list of loose
# arguments. Windows copy the lists and dicts they keep; none mutates a context.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class AxisLabels:
    """Axis text and units of a plot. The unit stays separate from the text:
    the display layer decides how to render it, while the text alone reaches
    the export manifest."""
    x_label: str
    y_label: str
    x_unit: str = ""
    y_unit: str = ""


@dataclass(frozen=True)
class HistView:
    """What a 2D-histogram view draws: the counts, the intensity clip
    (auto_max scaled by z_pct percent), the colour table and the axis ranges."""
    counts: np.ndarray
    auto_max: float
    z_pct: int
    lut: np.ndarray
    x_range: tuple[float, float]
    f_range: tuple[float, float]


@dataclass(frozen=True)
class ExportContext:
    """Where a window's view came from, for its export: the on-screen caption,
    the parent window's provenance dict, the contributing file paths, the
    database to export into (None hides the export button), whether the axes
    are physical (nm/pN), and how to fetch one curve's overlay trace."""
    caption: str = ""
    provenance: dict = field(default_factory=dict)
    paths: list[str] = field(default_factory=list)
    db_path: str | None = None
    physical: bool = False
    overlay_fn: Callable[[str], tuple | None] | None = None


@dataclass(frozen=True)
class FitContext:
    """Where a fit pane's data came from: the database its fits are saved to,
    the file paths behind each data point, and the on-screen caption."""
    db_path: str
    paths: list[str] = field(default_factory=list)
    caption: str = ""
