"""Validation against MATLAB ground truth (design §9)."""

from flypad.validate.compare import (
    ChannelComparison,
    FileComparison,
    compare_file,
    estimate_offset,
)
from flypad.validate.run import validate_dataset

__all__ = [
    "ChannelComparison",
    "FileComparison",
    "compare_file",
    "estimate_offset",
    "validate_dataset",
]
