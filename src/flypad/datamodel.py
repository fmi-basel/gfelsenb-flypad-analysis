"""Core in-memory data types (design §7)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import xarray as xr

from flypad.config.models import Config
from flypad.io.discovery import FileMeta, parse_filename
from flypad.io.raw import load_raw


@dataclass
class Recording:
    """A single capacitance recording: the signal plus its filename metadata."""

    capacitance: xr.DataArray  # dims (time, channel)
    meta: FileMeta

    @property
    def n_channels(self) -> int:
        return int(self.capacitance.sizes["channel"])

    @property
    def n_samples(self) -> int:
        return int(self.capacitance.sizes["time"])


def load_recording(path: str | Path, config: Config) -> Recording:
    """Load one recording, using channel count / duration / rate from ``config``."""
    da = load_raw(
        path,
        n_channels=config.hardware.n_channels,
        duration=config.acquisition.duration_samples,
        sampling_rate_hz=config.hardware.sampling_rate_hz,
    )
    return Recording(capacitance=da, meta=parse_filename(path))
