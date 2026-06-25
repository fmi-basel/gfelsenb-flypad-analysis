"""Instantaneous RMS power via a sliding window (design §5.5 / §6.2).

Ported from Scott McKinney's MATLAB ``fastrms``: the RMS over a length-``window``
rectangular window equals ``sqrt(moving_mean(x**2))``. We compute the moving mean
with :func:`scipy.ndimage.uniform_filter1d` (zero-padded, like MATLAB ``conv(...,'same')``),
so all channels are processed in one vectorised C call instead of a Python loop.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.ndimage import uniform_filter1d


def fastrms(
    x: npt.ArrayLike,
    window: int = 5,
    axis: int = 0,
    ampl: bool = False,
) -> npt.NDArray[np.float64]:
    """Sliding-window RMS power of ``x`` along ``axis``.

    Parameters
    ----------
    x : input signal (vector or matrix).
    window : rectangular window length in samples.
    axis : axis to run along (default 0 = time).
    ampl : if true, multiply by sqrt(2) (equivalent sinusoid amplitude).
    """
    power = np.square(np.asarray(x, dtype=np.float64))
    mean_power = np.asarray(
        uniform_filter1d(power, size=window, axis=axis, mode="constant", cval=0.0),
        dtype=np.float64,
    )
    rms = np.sqrt(mean_power)
    if ampl:
        rms = rms * np.sqrt(2.0)
    return rms
