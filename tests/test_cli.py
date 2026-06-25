"""M7: CLI — run / detect / stats / plot wired to the experiment runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from flypad.cli.app import app

runner = CliRunner()

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "sample" / "20240215"
SAMPLE_MAT = SAMPLE_DIR / "20240215.mat"
EXAMPLE_CFG = REPO_ROOT / "configs" / "example_experiment.yaml"
RAW_FILES = sorted(SAMPLE_DIR.glob("CapacitanceData_*"))


def _make_dataset(tmp_path: Path, n_channels: int = 8, n_time: int = 3000) -> tuple[Path, Path]:
    """A tiny synthetic recording (no sidecars) + a matching config."""
    arr = np.full((n_time, n_channels), 1000, dtype=np.uint16)
    for t0 in range(200, n_time - 200, 300):  # a few spike pairs on channel 0
        arr[t0 : t0 + 5, 0] = 1600
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    name = "CapacitanceData_C01_01_08_2024-01-01T00_00_00.0000000+00_00"
    arr.tofile(data_dir / name)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "mode: matlab_compat\n"
        "hardware:\n  n_channels: 8\n  sampling_rate_hz: 100\n"
        "acquisition:\n  duration_samples: 3000\n"
        "output:\n  formats: [csv]\n"
    )
    return data_dir, cfg


def _invoke(args: list[str]) -> object:
    result = runner.invoke(app, args)
    if result.exit_code != 0:  # surface the failure
        raise AssertionError(f"`{' '.join(args)}` failed:\n{result.output}\n{result.exception}")
    return result


# --------------------------------------------------------------------------- #
# basics
# --------------------------------------------------------------------------- #
def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "flypad" in result.output


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    for cmd in ("run", "detect", "stats", "plot", "validate", "config"):
        assert cmd in result.output


# --------------------------------------------------------------------------- #
# run / detect / stats / plot on a synthetic dataset
# --------------------------------------------------------------------------- #
def test_run_writes_tables_and_figures(tmp_path: Path) -> None:
    data_dir, cfg = _make_dataset(tmp_path)
    out = tmp_path / "results"
    _invoke(["run", str(data_dir), "-c", str(cfg), "-o", str(out)])
    for name in ("events", "per_fly", "per_condition"):
        assert (out / f"{name}.csv").exists()
    assert (out / "figures").is_dir()
    assert list((out / "figures").glob("*.png"))
    # per_fly has one row per channel
    assert len(pd.read_csv(out / "per_fly.csv")) == 8


def test_run_no_plots(tmp_path: Path) -> None:
    data_dir, cfg = _make_dataset(tmp_path)
    out = tmp_path / "results"
    _invoke(["run", str(data_dir), "-c", str(cfg), "-o", str(out), "--no-plots"])
    assert (out / "per_condition.csv").exists()
    assert not (out / "figures").exists()


def test_run_set_override(tmp_path: Path) -> None:
    data_dir, cfg = _make_dataset(tmp_path)
    out = tmp_path / "results"
    # override is accepted and the run still succeeds
    _invoke(
        [
            "run",
            str(data_dir),
            "-c",
            str(cfg),
            "-o",
            str(out),
            "--set",
            "feeding_bursts.min_sips=4",
            "--no-plots",
        ]
    )
    assert (out / "events.csv").exists()


def test_detect_then_stats_then_plot(tmp_path: Path) -> None:
    data_dir, cfg = _make_dataset(tmp_path)
    out = tmp_path / "results"
    _invoke(["detect", str(data_dir), "-c", str(cfg), "-o", str(out)])
    assert (out / "events.csv").exists() and (out / "per_fly.csv").exists()
    assert not (out / "per_condition.csv").exists()  # detect doesn't aggregate

    res = _invoke(["stats", str(out)])
    assert (out / "per_condition.csv").exists()
    assert "per-condition" in res.output  # the printed table title

    _invoke(["plot", str(out), "--kind", "boxplot,cdf", "-c", str(cfg)])
    assert list((out / "figures").glob("*.png"))


def test_detect_unknown_dir_fails() -> None:
    result = runner.invoke(app, ["detect", "/no/such/dir"])
    assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# full integration on the real sample (needs the git-ignored raw binaries)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not RAW_FILES, reason="raw sample binaries not present (git-ignored)")
def test_run_on_real_sample(tmp_path: Path) -> None:
    out = tmp_path / "results"
    _invoke(["run", str(SAMPLE_DIR), "-c", str(EXAMPLE_CFG), "-o", str(out)])
    per_condition = pd.read_csv(out / "per_condition.csv")
    # five conditions resolved from the exp_*.txt sidecars
    assert per_condition["condition_label"].nunique() == 5
    # starvation response: starved flies sip more than fully-fed ones
    nsips = per_condition[per_condition["metric"] == "n_sips"].set_index("condition_label")["mean"]
    assert nsips["44h wet starved"] > nsips["fully fed"]
    assert list((out / "figures").glob("dashboard*.png"))
