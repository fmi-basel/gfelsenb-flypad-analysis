"""Core in-memory data types (skeleton — expanded in M1; design §7)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Recording:
    """A single capacitance recording and its essential metadata (placeholder).

    Expanded in M1 to carry the labelled signal arrays (xarray) and per-channel
    results; for now it just anchors the type so the package structure is real.
    """

    path: str
    n_channels: int
