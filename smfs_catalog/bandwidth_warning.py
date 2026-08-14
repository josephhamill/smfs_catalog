"""Presentation text for acquisition/application filter conflicts."""

from __future__ import annotations

from .signal_processing import filter_bandwidth_conflict


FILTER_BANDWIDTH_CONSEQUENCE = (
    "The samples are already correlated before we filter, so τ is not "
    "explained by f_s/cutoff and error bars derived from that ratio would be "
    "too small. Fits are unaffected — τ is measured per fit, not "
    "predicted. Lower the cutoff to analyse below the acquisition bandwidth."
)


def filter_bandwidth_warning(cutoff_hz: float | None, acq_bw_hz: float | None) -> str:
    """Return explanatory UI text for a filter conflict, or an empty string."""
    if not filter_bandwidth_conflict(cutoff_hz, acq_bw_hz):
        return ""

    cutoff_hz = float(cutoff_hz)
    acq_bw_hz = float(acq_bw_hz)
    if cutoff_hz > acq_bw_hz:
        lead = (f"Our {cutoff_hz:,.0f} Hz cutoff is ABOVE the "
                f"{acq_bw_hz:,.0f} Hz filter used during acquisition")
        effect = "so our filter is doing nothing — the data was already narrower"
    else:
        lead = (f"Our {cutoff_hz:,.0f} Hz cutoff EQUALS the "
                f"{acq_bw_hz:,.0f} Hz filter used during acquisition")
        effect = "so our filter adds almost nothing — the data arrived at this bandwidth"
    return f"{lead}, {effect}. {FILTER_BANDWIDTH_CONSEQUENCE}"
