"""Layered configuration loading (design §6).

Resolution order (later wins):
  1. preset  — ``matlab_compat`` or ``corrected`` (config/presets/*.yaml),
               chosen by the ``preset`` arg, else the experiment's ``mode``, else "corrected"
  2. experiment YAML  — the user's file
  3. CLI overrides    — dotted ``key=value`` strings (values parsed as YAML scalars)

The merged mapping is validated into a :class:`~flypad.config.models.Config`.
"""

from __future__ import annotations

from copy import deepcopy
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from flypad.config.models import Config

KNOWN_PRESETS = ("matlab_compat", "corrected")


def _read_yaml(text: str) -> dict[str, Any]:
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"expected a YAML mapping, got {type(data).__name__}")
    return data


def load_preset(name: str) -> dict[str, Any]:
    """Load a built-in preset YAML as a plain mapping."""
    if name not in KNOWN_PRESETS:
        raise ValueError(f"unknown preset {name!r}; expected one of {list(KNOWN_PRESETS)}")
    resource = resources.files("flypad.config") / "presets" / f"{name}.yaml"
    return _read_yaml(resource.read_text(encoding="utf-8"))


def read_yaml_file(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a plain mapping."""
    return _read_yaml(Path(path).read_text(encoding="utf-8"))


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` (override wins on leaves)."""
    out = deepcopy(base)
    for key, value in override.items():
        existing = out.get(key)
        if isinstance(value, dict) and isinstance(existing, dict):
            out[key] = deep_merge(existing, value)
        else:
            out[key] = deepcopy(value)
    return out


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply dotted ``key.path=value`` overrides; values parsed as YAML scalars."""
    out = deepcopy(data)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} must be of the form key.path=value")
        node: dict[str, Any] = out
        parts = key.split(".")
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def load_config(
    experiment: str | Path | None = None,
    *,
    preset: str | None = None,
    overrides: list[str] | None = None,
) -> Config:
    """Resolve preset -> experiment YAML -> CLI overrides into a validated Config."""
    exp = read_yaml_file(experiment) if experiment is not None else {}
    base_name = preset or exp.get("mode") or "corrected"
    if base_name not in KNOWN_PRESETS:
        raise ValueError(
            f"unknown mode/preset {base_name!r}; expected one of {list(KNOWN_PRESETS)}"
        )
    merged = deep_merge(load_preset(base_name), exp)
    if overrides:
        merged = apply_overrides(merged, overrides)
    return Config.model_validate(merged)
