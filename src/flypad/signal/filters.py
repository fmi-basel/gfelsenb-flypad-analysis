"""Median filtering, baseline de-trending and the derivative (design §5.3 / §6.1).

All operations are vectorised across channels. The median filter supports MATLAB's
even kernel (e.g. ``medfilt1(x, 6)``), which scipy's ``medfilt`` cannot express:
an even-length median is the mean of the two central order statistics, computed
here from two C-level ``rank_filter`` passes.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.ndimage import median_filter, rank_filter, uniform_filter1d

from flypad.config.models import EdgeHandling

FloatArray = npt.NDArray[np.float64]


def _axis_size(ndim: int, axis: int, kernel: int) -> tuple[int, ...]:
    size = [1] * ndim
    size[axis] = kernel
    return tuple(size)


def median_filter_1d(
    x: npt.ArrayLike,
    kernel: int,
    axis: int = 0,
    mode: str = "nearest",
) -> FloatArray:
    """Median filter along ``axis`` supporting odd *and* even kernels.

    For an even ``kernel`` the result is the mean of the two central order
    statistics (MATLAB ``medfilt1`` semantics), not a single rank.
    """
    arr = np.asarray(x, dtype=np.float64)
    size = _axis_size(arr.ndim, axis, kernel)
    if kernel % 2 == 1:
        return np.asarray(median_filter(arr, size=size, mode=mode), dtype=np.float64)
    lo = rank_filter(arr, rank=kernel // 2 - 1, size=size, mode=mode)
    hi = rank_filter(arr, rank=kernel // 2, size=size, mode=mode)
    mid = (np.asarray(lo, dtype=np.float64) + np.asarray(hi, dtype=np.float64)) / 2.0
    return np.asarray(mid, dtype=np.float64)


def moving_average(
    x: npt.ArrayLike, span: int, axis: int = 0, mode: str = "constant"
) -> FloatArray:
    """Sliding mean along ``axis`` (zero-padded by default, like MATLAB conv 'same')."""
    arr = np.asarray(x, dtype=np.float64)
    return np.asarray(
        uniform_filter1d(arr, size=span, axis=axis, mode=mode, cval=0.0),
        dtype=np.float64,
    )


def detrend(
    x: npt.ArrayLike,
    *,
    kernel: int,
    span: int,
    edge_handling: EdgeHandling = EdgeHandling.crop,
    axis: int = 0,
) -> tuple[FloatArray, FloatArray]:
    """Median-filter then subtract a moving-average baseline.

    Returns ``(delta, baseline)`` where ``delta = medfilt(x) - moving_average(...)``,
    with edges handled per ``edge_handling``:

    * ``crop``    — drop ``span`` samples from each end (MATLAB; shortens the array)
    * ``zero``    — keep length, zero the first ``span + 1`` / last ``span`` samples
    * ``reflect`` — keep length, reflect the baseline at the edges (no edge dip)
    """
    med = median_filter_1d(x, kernel, axis=axis)
    base_mode = "reflect" if edge_handling is EdgeHandling.reflect else "constant"
    baseline = moving_average(med, span, axis=axis, mode=base_mode)
    delta = med - baseline

    if edge_handling is EdgeHandling.crop:
        keep = [slice(None)] * delta.ndim
        keep[axis] = slice(span, -span)
        return delta[tuple(keep)], baseline[tuple(keep)]
    if edge_handling is EdgeHandling.zero:
        head = [slice(None)] * delta.ndim
        head[axis] = slice(0, span + 1)
        tail = [slice(None)] * delta.ndim
        tail[axis] = slice(-span, None)
        delta[tuple(head)] = 0.0
        delta[tuple(tail)] = 0.0
    return delta, baseline


def derivative(x: npt.ArrayLike, axis: int = 0) -> FloatArray:
    """First difference along ``axis``, length-preserving (a zero is prepended)."""
    arr = np.asarray(x, dtype=np.float64)
    return np.asarray(
        np.diff(arr, axis=axis, prepend=np.take(arr, [0], axis=axis) * 0.0), dtype=np.float64
    )
