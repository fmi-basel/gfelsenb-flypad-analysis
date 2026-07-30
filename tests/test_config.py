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


def test_spill_unconnected_removal_toggles_per_preset() -> None:
    # corrected auto-removes bad channels; matlab_compat reproduces v2.2 (no auto-removal).
    corrected = load_config(preset="corrected")
    assert corrected.quality_control.remove_spill_quality is True
    assert corrected.quality_control.remove_unconnected is True
    compat = load_config(preset="matlab_compat")
    assert compat.quality_control.remove_spill_quality is False
    assert compat.quality_control.remove_unconnected is False


def test_plotting_condition_labels_override(tmp_path: Path) -> None:
    exp = tmp_path / "exp.yaml"
    exp.write_text(
        "mode: corrected\nplotting:\n  condition_labels:\n"
        '    "fully fed": "Fed control"\n    "24h wet starved": "Starved 24h"\n'
    )
    cfg = load_config(exp)
    assert cfg.plotting.condition_labels == {
        "fully fed": "Fed control",
        "24h wet starved": "Starved 24h",
    }


def test_plotting_condition_labels_default_empty() -> None:
    assert load_config(preset="corrected").plotting.condition_labels == {}


def test_plotting_facet_by_default_and_override(tmp_path: Path) -> None:
    assert load_config(preset="corrected").plotting.facet_by == "substrate"  # default
    exp = tmp_path / "exp.yaml"
    exp.write_text("mode: corrected\nplotting:\n  facet_by: file\n")
    assert load_config(exp).plotting.facet_by == "file"


def test_plotting_facet_by_rejects_unknown(tmp_path: Path) -> None:
    exp = tmp_path / "exp.yaml"
    exp.write_text("mode: corrected\nplotting:\n  facet_by: banana\n")
    with pytest.raises(ValidationError):
        load_config(exp)


def test_stats_statistic_default_and_override() -> None:
    assert load_config(preset="corrected").stats.statistic == "mean"
    cfg = load_config(preset="corrected", overrides=["stats.statistic=median"])
    assert cfg.stats.statistic == "median"


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


def test_resolve_field_follows_refs() -> None:
    from flypad.config import resolve_field

    schema = config_json_schema()
    assert resolve_field(schema, "feeding_bursts.min_sips")["type"] == "integer"
    assert resolve_field(schema, "sip_detection.equality_factor")["default"] == 0.5
    assert resolve_field(schema, "plotting.vector_format")["enum"] == ["pdf", "eps", "svg", "none"]
    assert resolve_field(schema, "nonexistent.field") is None
