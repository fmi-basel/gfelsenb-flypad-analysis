"""Signal processing: filtering, RMS, derivative thresholds (design §5.3-§5.6)."""

from flypad.signal.filters import (
    derivative,
    detrend,
    median_filter_1d,
    moving_average,
)
from flypad.signal.rms import fastrms
from flypad.signal.thresholds import (
    adaptive_mad_thresholds,
    compute_thresholds,
    fixed_height_thresholds,
    ibis_noise_thresholds,
    rolling_nanmedian,
)

__all__ = [
    "adaptive_mad_thresholds",
    "compute_thresholds",
    "derivative",
    "detrend",
    "fastrms",
    "fixed_height_thresholds",
    "ibis_noise_thresholds",
    "median_filter_1d",
    "moving_average",
    "rolling_nanmedian",
]
