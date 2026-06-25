"""Read raw flyPAD capacitance recordings into labelled arrays (design §5.1 / §7).

The on-disk format is channel-interleaved ``uint16`` (all channels for t=0, then
all channels for t=1, …). A Fortran-order reshape recovers the per-channel traces;
we transpose to the canonical ``(time, channel)`` layout.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import xarray as xr

logger = logging.getLogger(__name__)


def load_raw(
    path: str | Path,
    n_channels: int = 64,
    duration: int | None = 360_000,
    dtype: npt.DTypeLike = np.uint16,
    sampling_rate_hz: int = 100,
) -> xr.DataArray:
    """Load a raw capacitance file into a ``(time, channel)`` :class:`xr.DataArray`.

    Parameters
    ----------
    path : binary file of channel-interleaved ``uint16`` samples.
    n_channels : number of electrode channels.
    duration : keep at most this many time samples (``None`` keeps all).
    dtype : on-disk sample dtype.
    sampling_rate_hz : recorded in ``attrs`` (samples stay the index unit).

    Raises
    ------
    ValueError : empty file, or sample count not divisible by ``n_channels``.
    """
    path = Path(path)
    raw = np.fromfile(path, dtype=dtype)
    if raw.size == 0:
        raise ValueError(f"empty capacitance file: {path}")
    if raw.size % n_channels:
        raise ValueError(f"{path.name}: {raw.size} samples not divisible by {n_channels} channels")
    data = np.ascontiguousarray(raw.reshape(n_channels, -1, order="F").T)  # (time, channel)
    n_time = int(data.shape[0])
    if duration is not None:
        if duration < n_time:
            data = data[:duration]
        elif duration > n_time:
            logger.warning(
                "%s: %d samples, shorter than requested duration %d",
                path.name,
                n_time,
                duration,
            )
    return xr.DataArray(
        data,
        dims=("time", "channel"),
        coords={
            "time": np.arange(data.shape[0]),
            "channel": np.arange(n_channels),
        },
        name="capacitance",
        attrs={
            "source": str(path),
            "sampling_rate_hz": sampling_rate_hz,
            "dtype": np.dtype(dtype).name,
        },
    )
