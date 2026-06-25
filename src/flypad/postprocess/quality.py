"""Channel quality control: spill / unconnected / non-eater removal (design §5, M4).

Three independent removal criteria, each producing a boolean per-channel mask
(``True`` = drop the channel):

* **spill quality** — fraction of *saturated* samples (``== spill_saturation_value``,
  4095 for a 12-bit ADC). A leaking/short-circuited channel pins high.
* **unconnected** — fraction of *zero* samples. An unplugged electrode reads zero.
* **non-eater** — a fly that produced too few feeding events. With per-substrate
  removal a single channel is dropped; with global removal both channels of an arena
  (a fly's two food sources, ``2k`` / ``2k+1``) are dropped when the fly as a whole
  ate too little.

The spill/unconnected *fractions* reproduce ``Events.SpillQuality`` / ``Events.Unconnected``
from the MATLAB ground truth to < 1e-5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from flypad.config.models import NonEaters as NonEatersConfig
from flypad.config.models import QualityControl as QualityControlConfig

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]


def saturation_fraction(raw: npt.ArrayLike, saturation_value: int = 4095) -> FloatArray:
    """Per-channel fraction of saturated samples (``Events.SpillQuality``).

    ``raw`` is a ``(time, channel)`` array of the *unfiltered* capacitance.
    """
    data = np.asarray(raw)
    return np.asarray((data >= saturation_value).mean(axis=0), dtype=np.float64)


def zero_fraction(raw: npt.ArrayLike) -> FloatArray:
    """Per-channel fraction of zero samples (``Events.Unconnected``)."""
    data = np.asarray(raw)
    return np.asarray((data == 0).mean(axis=0), dtype=np.float64)


def flag_spill_channels(spill_fraction: npt.ArrayLike, threshold: float = 0.5) -> BoolArray:
    """Channels whose saturated fraction exceeds ``threshold`` (``True`` = remove)."""
    return np.asarray(np.asarray(spill_fraction, dtype=np.float64) > threshold)


def flag_unconnected_channels(zero_frac: npt.ArrayLike, threshold: float = 0.5) -> BoolArray:
    """Channels whose zero fraction exceeds ``threshold`` (``True`` = remove)."""
    return np.asarray(np.asarray(zero_frac, dtype=np.float64) > threshold)


def flag_non_eaters(
    counts: npt.ArrayLike,
    *,
    remove_global: bool = True,
    remove_substrate: bool = False,
    threshold: int = 2,
) -> BoolArray:
    """Flag non-eater channels from per-channel feeding-event counts.

    Parameters
    ----------
    counts : per-channel number of feeding events (feeding bursts by default).
    remove_substrate : drop an individual channel with ``count < threshold``.
    remove_global : drop *both* channels of an arena (``2k`` / ``2k+1``) when the
        arena's combined count is ``< threshold`` — i.e. the fly barely fed on either
        food source.
    threshold : minimum feeding-event count to be kept (a channel/arena with strictly
        fewer events is a non-eater).

    Returns a boolean mask (``True`` = remove).
    """
    n = np.asarray(counts, dtype=np.int64)
    remove = np.zeros(n.shape, dtype=bool)
    if remove_substrate:
        remove |= n < threshold
    if remove_global:
        pair = n.size // 2
        arena = n[: pair * 2].reshape(pair, 2).sum(axis=1) < threshold
        remove[: pair * 2] |= np.repeat(arena, 2)
        if n.size % 2:  # odd trailing channel has no partner -> judge it alone
            remove[-1] |= n[-1] < threshold
    return remove


@dataclass
class QualityResult:
    """Per-channel QC fractions and the combined removal mask."""

    spill_fraction: FloatArray
    zero_fraction: FloatArray
    spill_mask: BoolArray
    unconnected_mask: BoolArray
    non_eater_mask: BoolArray

    @property
    def remove_mask(self) -> BoolArray:
        return np.asarray(self.spill_mask | self.unconnected_mask | self.non_eater_mask)

    @property
    def kept_channels(self) -> IntArray:
        return np.asarray(np.flatnonzero(~self.remove_mask), dtype=np.int64)


def assess_quality(
    raw: npt.ArrayLike,
    feeding_counts: npt.ArrayLike,
    qc: QualityControlConfig,
    non_eaters: NonEatersConfig,
) -> QualityResult:
    """Compute spill/unconnected fractions and the full channel-removal mask."""
    spill = saturation_fraction(raw, qc.spill_saturation_value)
    zeros = zero_fraction(raw)
    spill_mask = (
        flag_spill_channels(spill, qc.spill_quality_threshold)
        if qc.remove_spill_quality
        else np.zeros(spill.shape, dtype=bool)
    )
    unconnected_mask = flag_unconnected_channels(zeros, qc.unconnected_zero_fraction)
    non_eater_mask = flag_non_eaters(
        feeding_counts,
        remove_global=non_eaters.remove_global,
        remove_substrate=non_eaters.remove_substrate,
        threshold=non_eaters.threshold_bouts,
    )
    return QualityResult(
        spill_fraction=spill,
        zero_fraction=zeros,
        spill_mask=spill_mask,
        unconnected_mask=unconnected_mask,
        non_eater_mask=non_eater_mask,
    )
