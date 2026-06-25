"""The typed object threaded through pipeline stages (design §5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Context:
    """Mutable bag of intermediate results passed stage-to-stage.

    Concrete typed fields (signals, events, thresholds, …) are promoted as
    milestones land; ``extra`` holds anything not yet first-class.
    """

    extra: dict[str, Any] = field(default_factory=dict)
