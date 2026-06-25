"""M9: packaging + provenance — version wiring, py.typed, run metadata."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import yaml

from flypad import __version__
from flypad.config import Config, load_config
from flypad.pipeline import config_hash, write_provenance

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_version_matches_pyproject() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert __version__ == pyproject["project"]["version"]
    assert __version__ != "0.0.0"  # installed, not a bare source tree


def test_py_typed_marker_present() -> None:
    assert (REPO_ROOT / "src" / "flypad" / "py.typed").exists()


def test_distribution_metadata() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    project = pyproject["project"]
    assert "gui" in project["optional-dependencies"]
    assert "docs" in project["optional-dependencies"]
    assert any("Typing :: Typed" in c for c in project["classifiers"])


def test_config_hash_is_stable_and_sensitive() -> None:
    a = load_config(preset="matlab_compat")
    b = load_config(preset="matlab_compat")
    c = load_config(preset="matlab_compat", overrides=["feeding_bursts.min_sips=9"])
    assert config_hash(a) == config_hash(b)  # deterministic
    assert config_hash(a) != config_hash(c)  # sensitive to changes
    assert len(config_hash(a)) == 64  # sha-256 hex


def test_write_provenance_roundtrip(tmp_path: Path) -> None:
    cfg = load_config(preset="matlab_compat")
    paths = write_provenance(
        tmp_path,
        cfg,
        files=[Path("CapacitanceData_A"), Path("CapacitanceData_B")],
        command="run",
        extra={"n_sips": 42},
        timestamp="2026-06-26T00:00:00+00:00",
    )
    assert {p.name for p in paths} == {"run_info.json", "config.used.yaml"}

    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["flypad_version"] == __version__
    assert info["config_hash"] == config_hash(cfg)
    assert info["n_files"] == 2
    assert info["files"] == ["CapacitanceData_A", "CapacitanceData_B"]
    assert info["n_sips"] == 42

    # config.used.yaml must re-validate into an identical Config
    reloaded = Config.model_validate(yaml.safe_load((tmp_path / "config.used.yaml").read_text()))
    assert reloaded == cfg
