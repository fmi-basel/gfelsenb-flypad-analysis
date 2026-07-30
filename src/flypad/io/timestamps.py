"""Manual arena-fill timestamps recorded alongside a recording (design §5.1).

While a plate is loaded the experimenter presses a key as each flyPAD arena is filled,
producing a ``timestamps_manual_<stamp>.csv`` sidecar next to its ``CapacitanceData_*``
file. The format is headerless ``index, sample, key``::

    0,9612,Right
    1,16261,Right
    ...
    11,79204,Space

Row *i* is the *(i+1)*-th board position, so it covers channels
``i * channels_per_board_position … + (S - 1)``. The key column records which key was
pressed to advance and carries no analysis meaning. Sample indices are raw file
positions (the same clock as the recording).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from flypad.io.discovery import parse_filename

MANUAL_PREFIX = "timestamps_manual"

#: Columns of the parsed table.
FILL_COLUMNS = ("arena", "board_position", "sample", "key")


def read_arena_fills(path: str | Path) -> pd.DataFrame:
    """Parse a ``timestamps_manual_*.csv`` into ``arena / board_position / sample / key``.

    ``arena`` is the 0-based row index as written in the file and ``board_position`` its
    1-based equivalent (matching the ``board_position`` column of the channel map). A
    header row, if present, is ignored.
    """
    frame = pd.read_csv(Path(path).expanduser(), header=None, names=["arena", "sample", "key"])
    # tolerate an optional header row
    numeric = pd.to_numeric(frame["sample"], errors="coerce")
    frame = frame[numeric.notna()].copy()
    frame["arena"] = pd.to_numeric(frame["arena"], errors="coerce").astype("int64")
    frame["sample"] = numeric[numeric.notna()].astype("int64")
    frame["key"] = frame["key"].astype(str).str.strip()
    frame["board_position"] = frame["arena"] + 1
    return frame.reset_index(drop=True)[list(FILL_COLUMNS)]


def _stamp(name: str) -> str | None:
    """The ``YYYY-MM-DDTHH_MM_SS`` token shared by a recording and its sidecars."""
    meta = parse_filename(name)
    if not meta.date or not meta.time:
        return None
    return f"{meta.date}T{meta.time.replace(':', '_')}"


def find_arena_fills(
    data_dir: str | Path,
    capacitance_name: str,
) -> pd.DataFrame | None:
    """Locate and parse the manual-fill sidecar belonging to one recording.

    The sidecar is matched on the timestamp token embedded in both filenames; returns
    ``None`` when the recording has no timestamp or no matching sidecar exists.
    """
    stamp = _stamp(capacitance_name)
    if stamp is None:
        return None
    directory = Path(data_dir).expanduser()
    for candidate in sorted(directory.glob(f"{MANUAL_PREFIX}*.csv")):
        if stamp in candidate.name:
            return read_arena_fills(candidate)
    return None
