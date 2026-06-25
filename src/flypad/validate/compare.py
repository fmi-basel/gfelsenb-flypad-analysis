"""Compare Python detections against MATLAB ground truth (design §9).

MATLAB onset indices live in a cropped, 1-based frame; Python's are in the
de-trended frame offset by ``crop_offset``. We estimate a single integer offset
(robustly, then refined to maximise matches), then greedily match onsets within a
small sample tolerance and report per-channel / per-file precision and recall.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int64]


@dataclass
class ChannelComparison:
    n_python: int
    n_matlab: int
    matched: int

    @property
    def precision(self) -> float:
        return self.matched / self.n_python if self.n_python else float("nan")

    @property
    def recall(self) -> float:
        return self.matched / self.n_matlab if self.n_matlab else float("nan")


@dataclass
class FileComparison:
    channels: list[ChannelComparison]
    offset: int
    tolerance: int

    @property
    def n_python(self) -> int:
        return sum(c.n_python for c in self.channels)

    @property
    def n_matlab(self) -> int:
        return sum(c.n_matlab for c in self.channels)

    @property
    def matched(self) -> int:
        return sum(c.matched for c in self.channels)

    @property
    def precision(self) -> float:
        return self.matched / self.n_python if self.n_python else float("nan")

    @property
    def recall(self) -> float:
        return self.matched / self.n_matlab if self.n_matlab else float("nan")


def _match(py: IntArray, mat: IntArray, tolerance: int) -> int:
    """Greedy nearest match count between two sorted onset arrays."""
    i = j = matched = 0
    while i < py.size and j < mat.size:
        diff = int(py[i]) - int(mat[j])
        if abs(diff) <= tolerance:
            matched += 1
            i += 1
            j += 1
        elif py[i] < mat[j]:
            i += 1
        else:
            j += 1
    return matched


def estimate_offset(
    py_by_ch: list[IntArray], mat_by_ch: list[IntArray], max_offset: int = 300
) -> int:
    """Estimate the constant frame offset (to add to ``py``) as the mode of pairwise
    ``mat - py`` differences within ``±max_offset``.

    Robust to dense events: the true constant offset recurs on every matched pair and
    forms a sharp histogram peak, while spurious differences spread out thinly.
    """
    rel: list[IntArray] = []
    for py, mat in zip(py_by_ch, mat_by_ch, strict=True):
        if not py.size or not mat.size:
            continue
        ms = np.sort(mat)
        for p in py:
            lo = int(np.searchsorted(ms, p - max_offset, "left"))
            hi = int(np.searchsorted(ms, p + max_offset, "right"))
            rel.append(ms[lo:hi] - p)
    if not rel:
        return 0
    diffs = np.concatenate(rel) + max_offset  # shift into [0, 2*max_offset]
    counts = np.bincount(diffs.astype(np.int64), minlength=2 * max_offset + 1)
    return int(np.argmax(counts)) - max_offset


def compare_file(
    py_by_ch: list[IntArray],
    mat_by_ch: list[IntArray],
    tolerance: int = 2,
    offset: int | None = None,
) -> FileComparison:
    """Compare one file's per-channel Python vs MATLAB onsets."""
    if offset is None:
        offset = estimate_offset(py_by_ch, mat_by_ch)
    channels = [
        ChannelComparison(
            n_python=int(py.size),
            n_matlab=int(mat.size),
            matched=_match(np.sort(py + offset), np.sort(mat), tolerance),
        )
        for py, mat in zip(py_by_ch, mat_by_ch, strict=True)
    ]
    return FileComparison(channels=channels, offset=offset, tolerance=tolerance)
