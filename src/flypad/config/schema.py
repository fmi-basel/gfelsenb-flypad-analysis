"""JSON-Schema export for the configuration (editor autocomplete; design §6)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flypad.config.models import Config


def config_json_schema() -> dict[str, Any]:
    """Return the JSON Schema for the top-level :class:`Config`."""
    return Config.model_json_schema()


def write_schema(path: str | Path) -> Path:
    """Write the config JSON Schema to ``path`` and return it."""
    out = Path(path)
    out.write_text(json.dumps(config_json_schema(), indent=2) + "\n", encoding="utf-8")
    return out
