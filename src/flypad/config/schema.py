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


def _deref(schema: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    """Resolve a ``$ref`` / single-``allOf`` node into its ``$defs`` target.

    Sibling keys on the node (e.g. ``default``) are merged over the target. Scalar
    nodes (no ref) are returned unchanged.
    """
    ref = node.get("$ref")
    if ref is None:
        all_of = node.get("allOf")
        if isinstance(all_of, list) and len(all_of) == 1 and "$ref" in all_of[0]:
            ref = all_of[0]["$ref"]
    if ref is None:
        return node
    name = ref.rsplit("/", 1)[-1]
    target = dict(schema.get("$defs", {}).get(name, {}))
    target.update({k: v for k, v in node.items() if k not in ("$ref", "allOf")})
    return target


def iter_config_fields(schema: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Flatten the schema into ``(dotted_path, leaf_field_schema)`` pairs.

    Recurses through submodels (following ``$ref``) so nested fields like
    ``sip_detection.threshold.window`` are reachable — the basis for a fully
    schema-driven config form.
    """
    out: list[tuple[str, dict[str, Any]]] = []

    def walk(props: dict[str, Any], prefix: str) -> None:
        for name, node in props.items():
            resolved = _deref(schema, node)
            if "properties" in resolved:
                walk(resolved["properties"], f"{prefix}{name}.")
            else:
                out.append((f"{prefix}{name}", resolved))

    walk(schema.get("properties", {}), "")
    return out


def resolve_field(schema: dict[str, Any], dotted_path: str) -> dict[str, Any] | None:
    """Resolve a dotted config path (e.g. ``sip_detection.equality_factor``) to its
    leaf field schema (``type`` / ``default`` / bounds / ``enum``), or ``None``.

    Follows ``$ref`` links into ``$defs`` so the GUI can build editors straight from
    :func:`config_json_schema` — a schema-driven config panel.
    """
    parts = dotted_path.split(".")
    props = schema.get("properties", {})
    for i, part in enumerate(parts):
        field = props.get(part)
        if field is None:
            return None
        resolved = _deref(schema, field)
        if i == len(parts) - 1:
            return resolved
        props = resolved.get("properties", {})
    return None
