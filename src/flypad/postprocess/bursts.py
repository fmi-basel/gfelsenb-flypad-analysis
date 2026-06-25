"""Feeding bursts — ``GET_FEEDING_BURSTS`` equivalent (design §5 stage 7, M4).

A *feeding burst* is a maximal run of consecutive sips on one channel whose
inter-feeding intervals (IFI = onset[i+1] - offset[i]) all stay at or below a
threshold, and which contains at least ``min_sips`` sips. Following the MATLAB
v2.2 convention the IFI threshold is ``2 x mode(IFI)`` (``ifi_criterion="mode_x2"``).

The threshold can be derived per channel or, more robustly for sparse channels,
from the pooled IFIs of every channel (``scope="pooled"`` — the default), which is
how the MATLAB suite computes it across a recording.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from flypad.config.models import FeedingBursts as FeedingBurstsConfig
from flypad.detect.results import ChannelSips

IntArray = npt.NDArray[np.int64]

Scope = Literal["pooled", "per_channel"]


@dataclass
class ChannelBursts:
    """Feeding bursts for one channel (indices into that channel's sip arrays)."""

    onsets: IntArray  # onset sample of the first sip in each burst
    offsets: IntArray  # offset sample of the last sip in each burst
    n_sips: IntArray  # number of sips in each burst
    sip_start: IntArray  # index of the first sip of each burst (inclusive)
    sip_end: IntArray  # index of the last sip of each burst (inclusive)

    @property
    def durations(self) -> IntArray:
        return np.asarray(self.offsets - self.onsets, dtype=np.int64)

    @property
    def total_sips(self) -> int:
        return int(self.n_sips.sum())

    def __len__(self) -> int:
        return int(self.onsets.size)


def _empty_bursts() -> ChannelBursts:
    e = np.asarray([], dtype=np.int64)
    return ChannelBursts(
        onsets=e, offsets=e.copy(), n_sips=e.copy(), sip_start=e.copy(), sip_end=e.copy()
    )


def mode_int(values: npt.ArrayLike) -> int:
    """Most frequent integer value (smallest on ties), matching MATLAB ``mode``."""
    arr = np.asarray(values, dtype=np.int64).ravel()
    if arr.size == 0:
        return 0
    uniq, counts = np.unique(arr, return_counts=True)
    return int(uniq[int(np.argmax(counts))])  # np.unique is sorted -> smallest on ties


def feeding_ifi_threshold(
    ifis: npt.ArrayLike,
    criterion: str = "mode_x2",
) -> float:
    """IFI threshold for burst grouping. ``mode_x2`` -> ``2 x mode(IFI)``."""
    if criterion != "mode_x2":
        raise ValueError(f"unknown ifi_criterion: {criterion!r}")
    return 2.0 * mode_int(ifis)


def group_feeding_bursts(
    sips: ChannelSips,
    ifi_threshold: float,
    min_sips: int = 3,
) -> ChannelBursts:
    """Group one channel's sips into feeding bursts.

    Consecutive sips belong to the same burst while their IFI is ``<= ifi_threshold``;
    a strictly larger gap starts a new burst. Bursts with fewer than ``min_sips`` sips
    are dropped.
    """
    n = len(sips)
    if n < min_sips:
        return _empty_bursts()

    ifi = sips.ifi  # length n - 1
    breaks = np.flatnonzero(ifi > ifi_threshold)  # gap after sip i starts a new burst
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [n - 1]))  # inclusive last sip of each run

    counts = ends - starts + 1
    keep = counts >= min_sips
    starts, ends = starts[keep], ends[keep]

    return ChannelBursts(
        onsets=np.asarray(sips.onsets[starts], dtype=np.int64),
        offsets=np.asarray(sips.offsets[ends], dtype=np.int64),
        n_sips=np.asarray(ends - starts + 1, dtype=np.int64),
        sip_start=np.asarray(starts, dtype=np.int64),
        sip_end=np.asarray(ends, dtype=np.int64),
    )


def pooled_ifi_threshold(
    sips_list: list[ChannelSips],
    criterion: str = "mode_x2",
) -> float:
    """IFI threshold from the IFIs pooled across all channels."""
    pieces = [s.ifi for s in sips_list if len(s) >= 2]
    pooled = np.concatenate(pieces) if pieces else np.asarray([], dtype=np.int64)
    return feeding_ifi_threshold(pooled, criterion)


def detect_feeding_bursts(
    sips_list: list[ChannelSips],
    config: FeedingBurstsConfig,
    *,
    scope: Scope = "pooled",
) -> tuple[float, list[ChannelBursts]]:
    """Group every channel's sips into feeding bursts.

    Returns the IFI threshold used and the per-channel bursts. With ``scope="pooled"``
    a single threshold (from all channels' IFIs) is applied; ``per_channel`` derives a
    threshold from each channel independently.
    """
    if scope == "pooled":
        threshold = pooled_ifi_threshold(sips_list, config.ifi_criterion)
        bursts = [group_feeding_bursts(s, threshold, config.min_sips) for s in sips_list]
        return threshold, bursts

    out: list[ChannelBursts] = []
    last = 0.0
    for s in sips_list:
        last = feeding_ifi_threshold(s.ifi, config.ifi_criterion)
        out.append(group_feeding_bursts(s, last, config.min_sips))
    return last, out
