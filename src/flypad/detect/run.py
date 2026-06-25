"""Per-recording detection orchestration (design §5).

Ties the signal stages together: de-trend -> RMS activity bouts -> derivative ->
thresholds -> sips. Pure and Qt-free; the CLI/GUI call this.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from flypad.config.models import Config, EdgeHandling
from flypad.detect.bouts import extract_bouts
from flypad.detect.results import DetectResult
from flypad.detect.sips import detect_sips
from flypad.signal.filters import derivative, detrend
from flypad.signal.rms import fastrms
from flypad.signal.thresholds import compute_thresholds


def detect_recording(capacitance: xr.DataArray, config: Config) -> DetectResult:
    """Run the full per-recording detection pipeline."""
    data = np.asarray(capacitance.values, dtype=np.float64)

    pp = config.preprocessing
    delta, _ = detrend(
        data,
        kernel=pp.median_kernel,
        span=pp.baseline_span,
        edge_handling=pp.edge_handling,
    )
    crop_offset = pp.baseline_span if pp.edge_handling is EdgeHandling.crop else 0

    ab = config.activity_bouts
    rms = fastrms(delta, ab.rms_window, axis=0)
    mask = rms > ab.rms_threshold
    bouts = extract_bouts(mask)

    deriv = derivative(delta)
    t_pos, t_neg = compute_thresholds(delta, deriv, mask, config.sip_detection.threshold)
    sips = detect_sips(deriv, t_pos, t_neg, config.sip_detection)

    return DetectResult(
        sips=sips, bouts=bouts, crop_offset=crop_offset, n_samples=int(delta.shape[0])
    )
