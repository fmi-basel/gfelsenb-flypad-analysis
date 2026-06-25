"""Configuration models and layered loading (design §6)."""

from flypad.config.loader import (
    KNOWN_PRESETS,
    apply_overrides,
    deep_merge,
    load_config,
    load_preset,
    read_yaml_file,
)
from flypad.config.models import (
    Acquisition,
    Analysis,
    Config,
    EdgeHandling,
    Hardware,
    Mode,
    Pairing,
    Stats,
    ThresholdMethod,
)
from flypad.config.schema import (
    config_json_schema,
    iter_config_fields,
    resolve_field,
    write_schema,
)

__all__ = [
    "KNOWN_PRESETS",
    "Acquisition",
    "Analysis",
    "Config",
    "EdgeHandling",
    "Hardware",
    "Mode",
    "Pairing",
    "Stats",
    "ThresholdMethod",
    "apply_overrides",
    "config_json_schema",
    "deep_merge",
    "iter_config_fields",
    "load_config",
    "load_preset",
    "read_yaml_file",
    "resolve_field",
    "write_schema",
]
