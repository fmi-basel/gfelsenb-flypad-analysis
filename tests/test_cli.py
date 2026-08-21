"""M7: CLI — run / detect / stats / plot wired to the experiment runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
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
    for name in ("events", "per_fly", "per_condition", "comparisons"):
        assert (out / f"{name}.csv").exists()
    assert (out / "figures").is_dir()
    assert list((out / "figures").glob("*.png"))
    assert list((out / "figures").glob("*.pdf"))  # PDF is the default vector format
    assert not list((out / "figures").glob("*.eps"))
    # per_fly has one row per channel
    assert len(pd.read_csv(out / "per_fly.csv")) == 8


def test_run_no_plots(tmp_path: Path) -> None:
    data_dir, cfg = _make_dataset(tmp_path)
    out = tmp_path / "results"
    _invoke(["run", str(data_dir), "-c", str(cfg), "-o", str(out), "--no-plots"])
    assert (out / "per_condition.csv").exists()
    assert not (out / "figures").exists()


# --------------------------------------------------------------------------- #
# CLI / GUI consistency: both drive pipeline.run_experiment, so a results directory
# is the same whichever produced it (#1 — the GUI's own copy of the sequence had
# silently omitted the provenance sidecars).
# --------------------------------------------------------------------------- #
def _run_via_cli(data_dir: Path, cfg: Path, out: Path) -> None:
    _invoke(["run", str(data_dir), "-c", str(cfg), "-o", str(out), "--no-plots"])


def _run_via_shared_orchestration(data_dir: Path, cfg: Path, out: Path) -> object:
    """What the GUI calls, minus Qt — `run_pipeline_job` is a delegation to this."""
    from flypad.config import load_config
    from flypad.pipeline import run_experiment

    return run_experiment(
        data_dir, load_config(cfg, preset="matlab_compat"), out, make_plots=False, command="gui"
    )


def test_run_writes_provenance(tmp_path: Path) -> None:
    data_dir, cfg = _make_dataset(tmp_path)
    out = tmp_path / "results"
    _run_via_cli(data_dir, cfg, out)
    info = json.loads((out / "run_info.json").read_text())
    assert info["command"] == "run"
    assert (out / "config.used.yaml").exists()


def test_cli_and_gui_produce_the_same_files(tmp_path: Path) -> None:
    data_dir, cfg = _make_dataset(tmp_path)
    cli_out, gui_out = tmp_path / "cli", tmp_path / "gui"
    _run_via_cli(data_dir, cfg, cli_out)
    _run_via_shared_orchestration(data_dir, cfg, gui_out)
    assert {p.name for p in cli_out.iterdir()} == {p.name for p in gui_out.iterdir()}
    # both write the provenance sidecars, which is what the GUI used to skip
    assert {"run_info.json", "config.used.yaml"} <= {p.name for p in gui_out.iterdir()}


def test_cli_and_gui_produce_the_same_tables(tmp_path: Path) -> None:
    data_dir, cfg = _make_dataset(tmp_path)
    cli_out, gui_out = tmp_path / "cli", tmp_path / "gui"
    _run_via_cli(data_dir, cfg, cli_out)
    _run_via_shared_orchestration(data_dir, cfg, gui_out)
    for name in ("events", "per_fly", "per_condition", "comparisons"):
        pd.testing.assert_frame_equal(
            pd.read_csv(cli_out / f"{name}.csv"), pd.read_csv(gui_out / f"{name}.csv")
        )


def test_cli_and_gui_run_info_differs_only_by_command_and_timestamp(tmp_path: Path) -> None:
    data_dir, cfg = _make_dataset(tmp_path)
    cli_out, gui_out = tmp_path / "cli", tmp_path / "gui"
    _run_via_cli(data_dir, cfg, cli_out)
    _run_via_shared_orchestration(data_dir, cfg, gui_out)
    cli_info = json.loads((cli_out / "run_info.json").read_text())
    gui_info = json.loads((gui_out / "run_info.json").read_text())
    assert (cli_info["command"], gui_info["command"]) == ("run", "gui")
    volatile = {"command", "timestamp"}
    assert {k: v for k, v in cli_info.items() if k not in volatile} == {
        k: v for k, v in gui_info.items() if k not in volatile
    }
    # same config in, same config hash out — the directories are interchangeable
    assert cli_info["config_hash"] == gui_info["config_hash"]
    assert (cli_out / "config.used.yaml").read_text() == (gui_out / "config.used.yaml").read_text()


def test_stats_reuses_the_gui_runs_config(tmp_path: Path) -> None:
    """`flypad stats` fell back to a default Config on GUI directories (#1)."""
    from flypad.config.models import Config

    data_dir, cfg_path = _make_dataset(tmp_path)
    out = tmp_path / "gui"
    _run_via_shared_orchestration(data_dir, cfg_path, out)
    used = Config.model_validate(yaml.safe_load((out / "config.used.yaml").read_text()))
    assert used.mode.value == "matlab_compat"  # not the default-constructed Config()
    _invoke(["stats", str(out), "--metric", "n_sips"])


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
# config label overrides (metadata.conditions -> tables; plotting -> plots only)
# --------------------------------------------------------------------------- #
def _cfg_with(tmp_path: Path, extra: str) -> Path:
    cfg = tmp_path / "cfg_override.yaml"
    cfg.write_text(
        "mode: matlab_compat\n"
        "hardware:\n  n_channels: 8\n  sampling_rate_hz: 100\n"
        "acquisition:\n  duration_samples: 3000\n"
        "output:\n  formats: [csv]\n" + extra
    )
    return cfg


def test_metadata_conditions_override_renames_table_labels(tmp_path: Path) -> None:
    data_dir, _ = _make_dataset(tmp_path)
    cfg = _cfg_with(tmp_path, 'metadata:\n  conditions: ["Renamed Condition"]\n')
    out = tmp_path / "results"
    _invoke(["run", str(data_dir), "-c", str(cfg), "-o", str(out), "--no-plots"])
    per_fly = pd.read_csv(out / "per_fly.csv")
    assert set(per_fly["condition_label"]) == {"Renamed Condition"}


def test_metadata_substrates_override_renames_table_labels(tmp_path: Path) -> None:
    data_dir, _ = _make_dataset(tmp_path)
    cfg = _cfg_with(tmp_path, 'metadata:\n  substrates: ["10% yeast", "20mM sucrose"]\n')
    out = tmp_path / "results"
    _invoke(["run", str(data_dir), "-c", str(cfg), "-o", str(out), "--no-plots"])
    per_fly = pd.read_csv(out / "per_fly.csv")
    left = per_fly[per_fly["substrate_side"] == "left"]["substrate_label"]
    right = per_fly[per_fly["substrate_side"] == "right"]["substrate_label"]
    assert set(left) == {"10% yeast"} and set(right) == {"20mM sucrose"}


def test_plotting_condition_labels_do_not_touch_tables(tmp_path: Path) -> None:
    data_dir, _ = _make_dataset(tmp_path)
    # plot-only rename: the exported table keeps the canonical "condition 1" label.
    cfg = _cfg_with(tmp_path, 'plotting:\n  condition_labels:\n    "condition 1": "Displayed"\n')
    out = tmp_path / "results"
    _invoke(["run", str(data_dir), "-c", str(cfg), "-o", str(out)])
    per_fly = pd.read_csv(out / "per_fly.csv")
    assert set(per_fly["condition_label"]) == {"condition 1"}  # table unchanged
    assert list((out / "figures").glob("*.png"))  # figures still rendered


def test_write_tables_expands_tilde(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from flypad.pipeline import write_tables

    monkeypatch.setenv("HOME", str(tmp_path))  # ~ -> tmp_path
    written = write_tables({"events": pd.DataFrame({"a": [1, 2]})}, "~/d/results", formats=("csv",))
    assert (tmp_path / "d" / "results" / "events.csv").exists()  # written to the real home
    assert all(str(p).startswith(str(tmp_path)) for p in written)
    assert not (Path.cwd() / "~" / "d").exists()  # no literal "~/d" dir left in the cwd


def test_detection_reports_measured_length(tmp_path: Path) -> None:
    from flypad.config import load_config
    from flypad.pipeline import detect_experiment

    data_dir, cfg = _make_dataset(tmp_path, n_time=3000)
    # config claims a longer recording than the file actually holds
    conf = load_config(cfg, overrides=["acquisition.duration_samples=99999"])
    detection = detect_experiment(data_dir, conf)
    assert detection.n_samples == 3000  # the measured length, not the configured one


def test_raster_written_per_recording(tmp_path: Path) -> None:
    from flypad.config import load_config
    from flypad.pipeline import render_figures
    from flypad.pipeline.runner import _slug

    names = {
        0: "CapacitanceData_C01_01_02_2026-07-09T10_01_52.0+02_00",
        1: "CapacitanceData_C01_01_02_2026-07-09T15_43_50.0+02_00",
    }
    events = pd.DataFrame(
        {
            "file_index": [0, 0, 1, 1],
            "file_name": [names[0], names[0], names[1], names[1]],
            "channel": [0, 1, 0, 1],
            "condition": [1, 1, 1, 1],
            "condition_label": ["a", "a", "a", "a"],
            "onset": [10, 20, 30, 40],
        }
    )
    per_fly = pd.DataFrame(
        {
            "file_index": [0, 1],
            "channel": [0, 0],
            "condition": [1, 1],
            "condition_label": ["a", "a"],
            "n_sips": [2.0, 2.0],
        }
    )
    cfg = load_config(
        preset="corrected", overrides=["plotting.vector_format=none", "hardware.n_channels=2"]
    )
    written = render_figures(per_fly, events, tmp_path, cfg, kinds=["raster"])
    stems = sorted(p.stem for p in written)
    assert stems == ["raster_2026-07-09_10-01-52", "raster_2026-07-09_15-43-50"]
    assert _slug("2026-07-09 10:01:52") == "2026-07-09_10-01-52"


def test_arena_fill_markers_reach_the_raster(tmp_path: Path) -> None:
    from flypad.config import load_config
    from flypad.pipeline.runner import _arena_fill_markers

    name = "CapacitanceData_C01_01_16_2026-07-09T10_01_52.0+02_00"
    (tmp_path / "timestamps_manual_2026-07-09T10_01_52.csv").write_text(
        "0,9612,Right\n1,16261,Right\n"
    )
    events = pd.DataFrame({"file_name": [name], "onset": [100]})
    cfg = load_config(
        preset="corrected",
        overrides=[
            "hardware.n_channels=16",
            "metadata.channels_per_board_position=8",
            "alignment.enabled=false",  # absolute time -> markers are meaningful
        ],
    )
    # absolute time: arena 1's window starts at 0, arena 2's at the first key press.
    # Unaligned every channel spans the whole file, so only the starts are drawn.
    starts_only = [(0.0, 0.0, 7.0), (9612.0, 8.0, 15.0)]
    assert _arena_fill_markers(tmp_path, events, cfg) == starts_only
    assert _arena_fill_markers(tmp_path, events, cfg, 5000) == starts_only
    assert _arena_fill_markers(None, events, cfg) is None  # no data dir -> no markers
    assert _arena_fill_markers(tmp_path / "nope", events, cfg) is None  # no sidecar

    # arena-aligned: every window is [0, duration), so the pair bounds the whole plate
    aligned = load_config(
        preset="corrected",
        overrides=["hardware.n_channels=16", "alignment.enabled=true"],
    )
    assert _arena_fill_markers(tmp_path, events, aligned, 5000) == [
        (0.0, 0.0, 15.0),
        (5000.0, 0.0, 15.0),
    ]
    assert _arena_fill_markers(tmp_path, events, aligned) is None  # no window -> nothing to bound


def test_substrate_figure_skipped_for_single_substrate(tmp_path: Path) -> None:
    from flypad.config import load_config
    from flypad.pipeline import render_figures

    per_fly = pd.DataFrame(
        {
            "condition": [1, 1, 2, 2],
            "condition_label": ["a", "a", "b", "b"],
            "substrate_label": ["sucrose"] * 4,  # same food both sides -> not a choice assay
            "substrate_side": ["left", "right", "left", "right"],
            "n_sips": [10.0, 4.0, 8.0, 6.0],
        }
    )
    cfg = load_config(preset="corrected", overrides=["plotting.vector_format=none"])
    written = render_figures(per_fly, None, tmp_path, cfg, kinds=["substrate", "boxplot"])
    names = [p.name for p in written]
    assert not any(n.startswith("substrate") for n in names)  # skipped
    assert any(n.startswith("boxplot") for n in names)  # other figures unaffected


def test_relabel_conditions_helper() -> None:
    from flypad.pipeline.runner import _relabel_conditions

    df = pd.DataFrame({"condition_label": ["a", "b", "a"], "n_sips": [1, 2, 3]})
    out = _relabel_conditions(df, "condition_label", {"a": "A"})
    assert list(out["condition_label"]) == ["A", "b", "A"]  # unmapped "b" kept
    assert list(df["condition_label"]) == ["a", "b", "a"]  # input not mutated
    assert _relabel_conditions(df, "condition_label", {}) is df  # empty map is a no-op


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
