"""
rupture_algorithms.py — the algorithms a profile can choose between for
locating a rupture on its loading ramp.

Each entry names one algorithm, the profile settings that belong to it, and
which signal its force and extension are read from.  None of them is "the"
rupture point; the profile says which one is in force.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuptureAlgorithm:
    """
    key        : stored in EventParams and passed to fit_segments.
    name       : what the user sees.
    settings   : the AnalysisParams names shown beside this algorithm.
    raw_signal : True when the rupture force and extension, the chain fit
                 and the loading slopes all use the saved signal; False when
                 they use the low-pass one.
    """
    key:        str
    name:       str
    settings:   tuple[str, ...]
    raw_signal: bool


# A profile's rupture_algorithm_idx is a position in this registry. Append
# only: reordering would change what every stored profile means.
RUPTURE_ALGORITHMS: dict[str, RuptureAlgorithm] = {a.key: a for a in (
    RuptureAlgorithm("low_pass_max", "Low-pass maximum", (), raw_signal=False),
    RuptureAlgorithm("drawdown_walk_back", "Drawdown walk-back",
                     ("walk_back_fit_widths", "walk_back_noise_multiple"),
                     raw_signal=True),
)}
RUPTURE_ALGORITHM_BY_IDX: tuple[str, ...] = tuple(RUPTURE_ALGORITHMS)


def fit_deflection(algorithm: str, raw_defl, low_defl):
    """The deflection signal `algorithm`'s fits are made on: the raw retract
    for an algorithm that fits the saved signal, the low-pass one otherwise.
    A plot of fits draws this signal, so the data under a fit is the data it
    was fitted to."""
    return raw_defl if RUPTURE_ALGORITHMS[algorithm].raw_signal else low_defl
