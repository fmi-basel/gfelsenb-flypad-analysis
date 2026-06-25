"""I/O: raw readers, file discovery, and writers (design §4)."""

from flypad.io.discovery import (
    CAP_PREFIX,
    ConditionSpan,
    FileMeta,
    find_capacitance_files,
    parse_filename,
)
from flypad.io.raw import load_raw

__all__ = [
    "CAP_PREFIX",
    "ConditionSpan",
    "FileMeta",
    "find_capacitance_files",
    "load_raw",
    "parse_filename",
]
