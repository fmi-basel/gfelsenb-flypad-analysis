"""Per-arena alignment to the manual fill timestamps (design §5.1).

A plate is loaded one arena at a time, so a fly in board position 1 is exposed to food
for many minutes longer than one in position 12. Comparing raw sip counts across arenas
therefore compares exposure time as much as appetite.

This module re-bases each channel on *its own* arena start and crops every channel to a
common window, so all flies contribute an equal-length recording measured from the
moment their food arrived:

* the first arena starts at sample ``0``; arena *p* (``p >= 2``) starts at the
  ``(p - 2)``-th manual timestamp — each key press marks the start of the **next** eight
  channels, so the final press is a spare end-of-loading marker;
* every channel keeps the events in ``[start, start + duration)``, re-referenced so
  ``0`` is that arena's start.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from flypad.detect.results import ChannelBouts, ChannelSips

IntArray = npt.NDArray[np.int64]


def arena_start_samples(fills: pd.DataFrame, n_positions: int) -> dict[int, int]:
    """Map each 1-based board position to the sample at which its arena was filled.

    Position 1 starts at ``0`` (the recording is already running when loading begins);
    position ``p >= 2`` starts at timestamp ``p - 2``. Positions beyond the available
    timestamps are omitted.
    """
    stamps = [int(s) for s in fills["sample"]]
    starts: dict[int, int] = {1: 0}
    for position in range(2, n_positions + 1):
        index = position - 2
        if index >= len(stamps):
            break
        starts[position] = stamps[index]
    return starts


def channel_start_samples(
    fills: pd.DataFrame,
    n_channels: int,
    channels_per_board_position: int,
) -> IntArray:
    """Per-channel start sample, expanding each board position over its channels.

    Channels whose board position has no timestamp fall back to ``0`` (unaligned).
    """
    n_positions = -(-n_channels // channels_per_board_position)  # ceil
    starts = arena_start_samples(fills, n_positions)
    out = np.zeros(n_channels, dtype=np.int64)
    for channel in range(n_channels):
        position = channel // channels_per_board_position + 1
        out[channel] = starts.get(position, 0)
    return out


def available_duration(starts: npt.ArrayLike, n_samples: int) -> int:
    """Longest window every channel can supply: ``n_samples - latest start``."""
    arr = np.asarray(starts, dtype=np.int64)
    latest = int(arr.max()) if arr.size else 0
    return max(0, int(n_samples) - latest)


def window_sips(sips: ChannelSips, start: int, duration: int) -> ChannelSips:
    """Keep sips whose onset lies in ``[start, start + duration)``, re-based to ``start``."""
    keep = (sips.onsets >= start) & (sips.onsets < start + duration)
    return ChannelSips(
        onsets=np.asarray(sips.onsets[keep] - start, dtype=np.int64),
        offsets=np.asarray(sips.offsets[keep] - start, dtype=np.int64),
    )


def window_bouts(bouts: ChannelBouts, start: int, duration: int) -> ChannelBouts:
    """Keep bouts whose onset lies in ``[start, start + duration)``, re-based to ``start``."""
    keep = (bouts.onsets >= start) & (bouts.onsets < start + duration)
    return ChannelBouts(
        onsets=np.asarray(bouts.onsets[keep] - start, dtype=np.int64),
        offsets=np.asarray(bouts.offsets[keep] - start, dtype=np.int64),
    )


def align_channels(
    sips: list[ChannelSips],
    bouts: list[ChannelBouts],
    starts: npt.ArrayLike,
    duration: int,
) -> tuple[list[ChannelSips], list[ChannelBouts]]:
    """Apply :func:`window_sips` / :func:`window_bouts` to every channel."""
    arr = np.asarray(starts, dtype=np.int64)
    aligned_sips = [window_sips(s, int(arr[i]), duration) for i, s in enumerate(sips)]
    aligned_bouts = [window_bouts(b, int(arr[i]), duration) for i, b in enumerate(bouts)]
    return aligned_sips, aligned_bouts
