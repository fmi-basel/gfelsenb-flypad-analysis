"""Human-readable axis labels and time units for figures (design §10).

Figures should read in the language of the experiment, not in column names: ``n_sips``
becomes "Sips per fly", and a 71-minute recording gets a *minutes* axis rather than
0-4250 seconds.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

#: Column name -> axis label (with unit where one applies).
METRIC_LABELS: dict[str, str] = {
    "n_sips": "Sips per fly",
    "n_feeding_bursts": "Feeding bursts per fly",
    "n_sips_in_bursts": "Sips in bursts per fly",
    "n_isolated_sips": "Isolated sips per fly",
    "n_activity_bouts": "Activity bouts per fly",
    "mean_burst_size": "Mean burst size (sips)",
    "mean_sip_duration_ms": "Mean sip duration (ms)",
    "total_sip_duration_ms": "Total sip duration (ms)",
    "mean_ifi_ms": "Mean inter-feeding interval (ms)",
    "latency_to_first_sip_ms": "Latency to first sip (ms)",
    "spill_fraction": "Saturated samples (fraction)",
    "zero_fraction": "Zero samples (fraction)",
    "preference": "Preference index",
}


def metric_label(metric: str) -> str:
    """Axis label for a metric column, falling back to a prettified column name."""
    known = METRIC_LABELS.get(metric)
    if known is not None:
        return known
    pretty = metric.replace("_", " ").strip()
    return pretty[:1].upper() + pretty[1:] if pretty else metric


def time_axis(seconds: npt.ArrayLike) -> tuple[FloatArray, str]:
    """Rescale a seconds axis to the most readable unit.

    Returns ``(values, label)`` in seconds, minutes or hours depending on the span, so a
    long recording is not reported as thousands of seconds.
    """
    arr = np.asarray(seconds, dtype=np.float64)
    span = float(np.nanmax(arr)) if arr.size else 0.0
    if span >= 7200:
        return arr / 3600.0, "Time (h)"
    if span >= 180:
        return arr / 60.0, "Time (min)"
    return arr, "Time (s)"
