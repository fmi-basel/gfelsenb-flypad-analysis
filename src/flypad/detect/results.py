"""Result containers for detection (design §7)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int64]


@dataclass
class ChannelSips:
    """Detected sips for one channel (sample indices into the de-trended trace)."""

    onsets: IntArray
    offsets: IntArray

    @property
    def durations(self) -> IntArray:
        return np.asarray(self.offsets - self.onsets, dtype=np.int64)

    @property
    def ifi(self) -> IntArray:
        if self.onsets.size < 2:
            return np.asarray([], dtype=np.int64)
        return np.asarray(self.onsets[1:] - self.offsets[:-1], dtype=np.int64)

    def __len__(self) -> int:
        return int(self.onsets.size)


@dataclass
class ChannelBouts:
    """Activity bouts for one channel."""

    onsets: IntArray
    offsets: IntArray

    @property
    def durations(self) -> IntArray:
        return np.asarray(self.offsets - self.onsets, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.onsets.size)


@dataclass
class DetectResult:
    """Per-recording detection output."""

    sips: list[ChannelSips]
    bouts: list[ChannelBouts]
    crop_offset: int  # samples trimmed at the front (for index alignment, design §5.4)
    n_samples: int

    @property
    def n_channels(self) -> int:
        return len(self.sips)
