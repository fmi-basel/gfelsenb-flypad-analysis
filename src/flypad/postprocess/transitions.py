"""Channel-to-channel transitions within an arena (design §5, M4).

A flyPAD arena gives one fly two food channels (left = even, right = odd). Merging
both channels' sips into a single time-ordered sequence reveals how the fly *switches*
between food sources:

* a **transition** is two consecutive sips on different channels; its **TransitionIBI**
  is the gap between the end of the last sip on one channel and the start of the first
  sip on the other,
* each sip is **in-burst** (part of a feeding burst) or **isolated**,
* **latency** is the onset of the first sip on each channel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from flypad.detect.results import ChannelSips
from flypad.postprocess.bursts import ChannelBursts

IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


def classify_in_burst(n_sips: int, bursts: ChannelBursts) -> BoolArray:
    """Boolean mask (length ``n_sips``): ``True`` where the sip belongs to a burst."""
    mask = np.zeros(n_sips, dtype=bool)
    for start, end in zip(bursts.sip_start, bursts.sip_end, strict=True):
        mask[int(start) : int(end) + 1] = True
    return mask


@dataclass
class ArenaTransitions:
    """Channel-switching summary for one arena (left/right substrate pair)."""

    times: IntArray  # onset sample of the sip that starts on the new channel
    ibi: IntArray  # gap into the new channel (new.onset - prev.offset)
    from_side: IntArray  # 0 = left, 1 = right (channel sipped before the switch)
    to_side: IntArray  # 0 = left, 1 = right (channel sipped after the switch)
    latency_left: int  # first left-channel sip onset (-1 if none)
    latency_right: int  # first right-channel sip onset (-1 if none)

    @property
    def n_transitions(self) -> int:
        return int(self.times.size)


def _latency(sips: ChannelSips) -> int:
    return int(sips.onsets[0]) if len(sips) else -1


def channel_transitions(left: ChannelSips, right: ChannelSips) -> ArenaTransitions:
    """Compute transitions between the two food channels of one arena."""
    onsets = np.concatenate([left.onsets, right.onsets]).astype(np.int64)
    offsets = np.concatenate([left.offsets, right.offsets]).astype(np.int64)
    side = np.concatenate([np.zeros(len(left), np.int64), np.ones(len(right), np.int64)])

    empty = np.asarray([], dtype=np.int64)
    if onsets.size < 2:
        return ArenaTransitions(
            times=empty,
            ibi=empty.copy(),
            from_side=empty.copy(),
            to_side=empty.copy(),
            latency_left=_latency(left),
            latency_right=_latency(right),
        )

    # Stable sort by onset keeps left-before-right for simultaneous onsets.
    order = np.argsort(onsets, kind="stable")
    onsets, offsets, side = onsets[order], offsets[order], side[order]

    switch = np.flatnonzero(side[1:] != side[:-1])  # transition before sip switch+1
    return ArenaTransitions(
        times=np.asarray(onsets[switch + 1], dtype=np.int64),
        ibi=np.asarray(onsets[switch + 1] - offsets[switch], dtype=np.int64),
        from_side=np.asarray(side[switch], dtype=np.int64),
        to_side=np.asarray(side[switch + 1], dtype=np.int64),
        latency_left=_latency(left),
        latency_right=_latency(right),
    )
