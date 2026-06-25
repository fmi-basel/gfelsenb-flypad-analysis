"""Synthetic signal generator for tests (a clean trace with known sips)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def make_trace(
    n: int,
    sip_onsets: list[int],
    amplitude: float = 200.0,
    width: int = 12,
    seed: int = 0,
) -> npt.NDArray[np.float64]:
    """Build a de-trended-like 1D trace: flat baseline with rectangular 'sip' bumps.

    Each sip is an up-step (attachment) held for ``width`` samples then a down-step
    (detachment) back to baseline -- a clean version of the real capacitance shape.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n) * 1.0  # low noise
    for on in sip_onsets:
        x[on : on + width] += amplitude
    return np.asarray(x, dtype=np.float64)
