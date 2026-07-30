"""I/O: raw readers, file discovery, and writers (design §4)."""

from flypad.io.discovery import (
    CAP_PREFIX,
    ConditionSpan,
    FileMeta,
    find_capacitance_files,
    parse_filename,
)
from flypad.io.matlab import MatEvents, read_events_mat
from flypad.io.raw import load_raw
from flypad.io.timestamps import (
    FILL_COLUMNS,
    MANUAL_PREFIX,
    find_arena_fills,
    read_arena_fills,
)

__all__ = [
    "CAP_PREFIX",
    "FILL_COLUMNS",
    "MANUAL_PREFIX",
    "ConditionSpan",
    "FileMeta",
    "MatEvents",
    "find_arena_fills",
    "find_capacitance_files",
    "load_raw",
    "parse_filename",
    "read_arena_fills",
    "read_events_mat",
]
