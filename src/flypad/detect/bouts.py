"""Activity-bout extraction from the RMS mask (design §5.5)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from flypad.detect.results import ChannelBouts


def extract_bouts(mask: npt.ArrayLike) -> list[ChannelBouts]:
    """Extract per-channel activity bouts from a boolean ``(time, channel)`` mask.

    Bouts are the runs of ``True``; onset/offset are the rising/falling edges.
    Leading offsets and trailing onsets (partial bouts at the ends) are dropped,
    mirroring MATLAB's ``ProcessDataDamPAD`` bout extraction.
    """
    m = np.asarray(mask).astype(bool)
    out: list[ChannelBouts] = []
    for ch in range(m.shape[1]):
        diff = np.diff(m[:, ch].astype(np.int8))
        onsets = np.flatnonzero(diff > 0) + 1
        offsets = np.flatnonzero(diff < 0) + 1
        if onsets.size and offsets.size and onsets[0] > offsets[0]:
            offsets = offsets[1:]
        if onsets.size and offsets.size and onsets[-1] > offsets[-1]:
            onsets = onsets[:-1]
        n = min(onsets.size, offsets.size)
        out.append(
            ChannelBouts(
                onsets=onsets[:n].astype(np.int64),
                offsets=offsets[:n].astype(np.int64),
            )
        )
    return out
