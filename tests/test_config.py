"""M1: configuration models, presets, layering, and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from flypad.config import (
    Config,
    EdgeHandling,
    Mode,
    Pairing,
    ThresholdMethod,
    apply_overrides,
    config_json_schema,
    deep_merge,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_matlab_compat_preset() -> None:
    cfg = load_config(preset="matlab_compat")
    assert cfg.mode is Mode.matlab_compat
    assert cfg.preprocessing.median_kernel == 6
    assert cfg.preprocessing.edge_handling is EdgeHandling.crop
    assert cfg.sip_detection.pairing is Pairing.greedy
    assert cfg.sip_detection.threshold.method is ThresholdMethod.ibis_noise
    assert cfg.sip_detection.max_duration_samples == 100


def test_corrected_preset() -> None:
    cfg = load_config(preset="corrected")
    assert cfg.mode is Mode.corrected
    assert cfg.sip_detection.threshold.method is ThresholdMethod.adaptive_mad
    assert cfg.preprocessing.edge_handling is EdgeHandling.reflect


def test_layering_experiment_over_preset(tmp_path: Path) -> None:
    exp = tmp_path / "exp.yaml"
    exp.write_text("mode: matlab_compat\nhardware:\n  n_channels: 96\n")
    cfg = load_config(exp)
    assert cfg.mode is Mode.matlab_compat  # preset base
    assert cfg.hardware.n_channels == 96  # experiment override
    assert cfg.preprocessing.median_kernel == 6  # untouched preset value


def test_cli_overrides_win() -> None:
    cfg = load_config(preset="matlab_compat", overrides=["sip_detection.equality_factor=0.4"])
    assert cfg.sip_detection.equality_factor == 0.4


def test_deep_merge_is_recursive() -> None:
    merged = deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 9}})
    assert merged == {"a": {"x": 1, "y": 9}}


def test_apply_overrides_yaml_typing() -> None:
    out = apply_overrides({}, ["hardware.n_channels=96", "plotting.enabled=false"])
    assert out["hardware"]["n_channels"] == 96  # int, not "96"
    assert out["plotting"]["enabled"] is False


def test_extra_key_forbidden() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"nonsense": 1})


def test_unknown_preset_raises() -> None:
    with pytest.raises(ValueError):
        load_config(preset="banana")


def test_roundtrip_dump_reload() -> None:
    cfg = load_config(preset="corrected")
    again = Config.model_validate(cfg.model_dump(mode="json"))
    assert again == cfg


def test_json_schema_has_properties() -> None:
    schema = config_json_schema()
    assert "properties" in schema
    assert "sip_detection" in schema["properties"]


def test_example_experiment_validates() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "example_experiment.yaml")
    assert cfg.hardware.n_channels == 96
    assert cfg.acquisition.duration_samples == 425391
    assert len(cfg.metadata.conditions) == 5
