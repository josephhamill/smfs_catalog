import numpy as np
import pytest

from smfs_catalog.roi_detection import compute_detection_signals


def test_detection_window_is_bounded_to_short_odd_trace():
    result = compute_detection_signals(
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 1.0, 2.0]),
        window_pts=31,
    )
    assert result.d1.shape == (3,)
    assert result.mean_dev.shape == (3,)


def test_detection_rejects_trace_too_short_for_derivative():
    with pytest.raises(ValueError, match="at least 3 samples"):
        compute_detection_signals(
            np.array([0.0, 1.0]), np.array([0.0, 1.0]), window_pts=31,
        )


def test_detection_rejects_mismatched_signal_lengths():
    with pytest.raises(ValueError, match="same length"):
        compute_detection_signals(
            np.zeros(5), np.arange(4, dtype=float), window_pts=5,
        )
