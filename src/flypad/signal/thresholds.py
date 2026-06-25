"""Derivative-threshold strategies for sip-edge detection (design §5.6 / §6.4).

Three strategies produce per-channel positive/negative thresholds (broadcastable
to the derivative's ``(time, channel)`` shape):

* ``ibis_noise``   — MATLAB v2.2: max/min of ``diff`` of the de-trended signal during
                     non-active periods (a single constant per channel).
* ``adaptive_mad`` — robust, time-varying: ``mad_scale * rolling_median`` of the
                     positive/negative derivative, floored at ``offset``.
* ``fixed_height`` — a constant ``±height`` (old Python port).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from flypad.config.models import SipThreshold, ThresholdMethod

FloatArray = npt.NDArray[np.float64]


def rolling_nanmedian(arr: npt.ArrayLike, window: int) -> FloatArray:
    """Centred, NaN-skipping rolling median over a ``(time, channel)`` array."""
    frame = pd.DataFrame(np.asarray(arr, dtype=np.float64))
    rolled = frame.rolling(window, min_periods=1, center=True).median()
    return np.asarray(rolled.to_numpy(), dtype=np.float64)


def ibis_noise_thresholds(
    delta: npt.ArrayLike, activity_mask: npt.ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Per-channel constant thresholds from derivative extremes during non-activity."""
    d = np.asarray(delta, dtype=np.float64)
    mask = np.asarray(activity_mask).astype(bool)
    ibis = np.where(mask, 0.0, d)  # de-trended signal during inter-bout intervals
    diff = np.diff(ibis, axis=0)
    t_pos = np.asarray(np.nanmax(diff, axis=0, keepdims=True), dtype=np.float64)  # (1, C)
    t_neg = np.asarray(np.nanmin(diff, axis=0, keepdims=True), dtype=np.float64)
    return t_pos, t_neg


def adaptive_mad_thresholds(
    derivative: npt.ArrayLike, window: int, mad_scale: float, offset: float
) -> tuple[FloatArray, FloatArray]:
    """Time-varying robust thresholds (Quiroga-style), floored at ``±offset``."""
    deriv = np.asarray(derivative, dtype=np.float64)
    pos = np.where(deriv < 0, np.nan, deriv)
    t_pos = mad_scale * rolling_nanmedian(pos, window)
    t_pos = np.where(np.isnan(t_pos) | (t_pos < offset), offset, t_pos)
    neg = np.where(deriv > 0, np.nan, deriv)
    t_neg = mad_scale * rolling_nanmedian(neg, window)
    t_neg = np.where(np.isnan(t_neg) | (t_neg > -offset), -offset, t_neg)
    return np.asarray(t_pos, dtype=np.float64), np.asarray(t_neg, dtype=np.float64)


def fixed_height_thresholds(height: float) -> tuple[FloatArray, FloatArray]:
    """Constant ``±height`` thresholds."""
    return np.array([[height]], dtype=np.float64), np.array([[-height]], dtype=np.float64)


def compute_thresholds(
    delta: npt.ArrayLike,
    derivative: npt.ArrayLike,
    activity_mask: npt.ArrayLike,
    threshold: SipThreshold,
) -> tuple[FloatArray, FloatArray]:
    """Dispatch to the configured threshold strategy; returns ``(T_pos, T_neg)``."""
    if threshold.method is ThresholdMethod.ibis_noise:
        return ibis_noise_thresholds(delta, activity_mask)
    if threshold.method is ThresholdMethod.adaptive_mad:
        return adaptive_mad_thresholds(
            derivative, threshold.window, threshold.mad_scale, threshold.offset
        )
    if threshold.method is ThresholdMethod.fixed_height:
        return fixed_height_thresholds(threshold.fixed_height)
    raise ValueError(f"unknown threshold method: {threshold.method!r}")
