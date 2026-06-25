"""Read a legacy MATLAB ``Events`` .mat (v7.3 / HDF5) for validation (design §9).

The struct stores cell arrays of per-(channel, file) sip indices as HDF5 object
references; we dereference them into plain integer arrays. Empty MATLAB cells are
flagged with a ``MATLAB_empty`` attribute and returned as empty arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int64]


@dataclass
class MatEvents:
    """Per-(file, channel) sip events read from a MATLAB Events.mat."""

    ons: list[list[IntArray]]  # ons[file][channel]
    offs: list[list[IntArray]]
    durations: list[list[IntArray]]
    n_files: int
    n_channels: int
    configuration: str | None
    duration_samples: int | None


def _deref(f: h5py.File, ref: object) -> IntArray:
    try:
        obj = f[ref]
    except (ValueError, KeyError):
        return np.asarray([], dtype=np.int64)
    if obj.attrs.get("MATLAB_empty", 0):
        return np.asarray([], dtype=np.int64)
    arr = np.asarray(obj).ravel()
    if arr.size == 0:
        return np.asarray([], dtype=np.int64)
    return arr.astype(np.int64)


def _matstr(f: h5py.File, key: str) -> str | None:
    if key not in f:
        return None
    arr = np.asarray(f[key]).ravel()
    try:
        return "".join(chr(int(c)) for c in arr if c != 0)
    except (ValueError, TypeError):
        return None


def _scalar(f: h5py.File, key: str) -> int | None:
    if key not in f:
        return None
    arr = np.asarray(f[key]).ravel()
    return int(arr[0]) if arr.size else None


def read_events_mat(path: str | Path) -> MatEvents:
    """Load sip events (Ons/Offs/Durations) and key settings from an Events.mat."""
    with h5py.File(Path(path), "r") as f:
        ev = f["Events"]
        ons_refs = np.asarray(ev["Ons"])  # shape (channels, files)
        offs_refs = np.asarray(ev["Offs"])
        dur_refs = np.asarray(ev["Durations"])
        n_ch, n_files = ons_refs.shape

        def grid(refs: np.ndarray) -> list[list[IntArray]]:
            return [[_deref(f, refs[c, fi]) for c in range(n_ch)] for fi in range(n_files)]

        return MatEvents(
            ons=grid(ons_refs),
            offs=grid(offs_refs),
            durations=grid(dur_refs),
            n_files=int(n_files),
            n_channels=int(n_ch),
            configuration=_matstr(f, "Configuration"),
            duration_samples=_scalar(f, "Dur"),
        )
