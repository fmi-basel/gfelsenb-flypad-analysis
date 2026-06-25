"""Discover recordings and parse their filename metadata (design §5.1).

A capacitance filename encodes the condition→channel spans and a timestamp, e.g.::

    CapacitanceData_C01_01_96_2024-02-15T11_28_04.8078208+01_00

``C01_01_96`` => condition 1 spans channels 1-96 (1-based, as written by the rig).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

CAP_PREFIX = "CapacitanceData"
_COND_RE = re.compile(r"C(\d{2})_(\d+)_(\d+)")
_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2})_(\d{2})_(\d{2})")


@dataclass(frozen=True)
class ConditionSpan:
    """A condition's inclusive 1-based channel range, parsed from the filename."""

    condition: int
    channel_start: int
    channel_end: int


@dataclass(frozen=True)
class FileMeta:
    """Metadata parsed from a capacitance filename."""

    path: Path
    filename: str
    condition_spans: tuple[ConditionSpan, ...]
    date: str | None
    time: str | None


def find_capacitance_files(directory: str | Path) -> list[Path]:
    """Recursively find ``CapacitanceData*`` files, sorted by name (≈ chronological)."""
    root = Path(directory)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.name.startswith(CAP_PREFIX))


def parse_filename(name: str | Path) -> FileMeta:
    """Parse condition spans and timestamp out of a capacitance filename."""
    path = Path(name)
    filename = path.name
    spans = tuple(
        ConditionSpan(int(cond), int(lo), int(hi)) for cond, lo, hi in _COND_RE.findall(filename)
    )
    ts = _TS_RE.search(filename)
    date = ts.group(1) if ts else None
    time = f"{ts.group(2)}:{ts.group(3)}:{ts.group(4)}" if ts else None
    return FileMeta(path=path, filename=filename, condition_spans=spans, date=date, time=time)
