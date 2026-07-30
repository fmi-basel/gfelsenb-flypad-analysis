"""M1: raw reader (xarray) and filename/metadata parsing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flypad.config import Config
from flypad.datamodel import load_recording
from flypad.io import find_capacitance_files, load_raw, parse_filename

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "sample" / "20240215"
SAMPLE_FILE = SAMPLE_DIR / "CapacitanceData_C01_01_96_2024-02-15T11_28_04.8078208+01_00"


def _write_interleaved(path: Path, n_channels: int, n_time: int) -> np.ndarray:
    """Write a synthetic channel-interleaved file; value[c, t] = c * 1000 + t."""
    mat = np.empty((n_channels, n_time), dtype=np.uint16)
    for c in range(n_channels):
        for t in range(n_time):
            mat[c, t] = c * 1000 + t
    mat.flatten(order="F").astype(np.uint16).tofile(path)
    return mat


def test_load_raw_recovers_interleaving(tmp_path: Path) -> None:
    path = tmp_path / "CapacitanceData_C01_01_04_synthetic"
    mat = _write_interleaved(path, n_channels=4, n_time=10)
    da = load_raw(path, n_channels=4, duration=None)
    assert da.dims == ("time", "channel")
    assert da.shape == (10, 4)
    # da[t, c] should equal mat[c, t]
    np.testing.assert_array_equal(da.values, mat.T)


def test_load_raw_truncates_to_duration(tmp_path: Path) -> None:
    path = tmp_path / "CapacitanceData_C01_01_04_short"
    _write_interleaved(path, n_channels=4, n_time=10)
    da = load_raw(path, n_channels=4, duration=6)
    assert da.shape == (6, 4)


def test_load_raw_divisibility_error(tmp_path: Path) -> None:
    path = tmp_path / "CapacitanceData_bad"
    np.arange(10, dtype=np.uint16).tofile(path)  # 10 not divisible by 4
    with pytest.raises(ValueError, match="divisible"):
        load_raw(path, n_channels=4)


def test_parse_filename() -> None:
    meta = parse_filename(SAMPLE_FILE)
    assert meta.condition_spans == ((1, 1, 96),) or meta.condition_spans[0].channel_end == 96
    span = meta.condition_spans[0]
    assert (span.condition, span.channel_start, span.channel_end) == (1, 1, 96)
    assert meta.date == "2024-02-15"
    assert meta.time == "11:28:04"


# ---- tests that need the (git-ignored) raw sample file: skip if absent ----
_missing = not SAMPLE_FILE.exists()


@pytest.mark.skipif(_missing, reason="raw sample binary not present (git-ignored)")
def test_find_capacitance_files() -> None:
    files = find_capacitance_files(SAMPLE_DIR)
    assert len(files) == 2
    assert all(p.name.startswith("CapacitanceData") for p in files)


@pytest.mark.skipif(_missing, reason="raw sample binary not present (git-ignored)")
def test_load_real_sample_recording() -> None:
    cfg = Config.model_validate(
        {"hardware": {"n_channels": 96}, "acquisition": {"duration_samples": 425391}}
    )
    rec = load_recording(SAMPLE_FILE, cfg)
    assert rec.n_channels == 96
    assert rec.n_samples == 425391
    assert rec.meta.date == "2024-02-15"


# --------------------------------------------------------------------------- #
# manual arena-fill timestamps (timestamps_manual_*.csv)
# --------------------------------------------------------------------------- #
_FILLS_CSV = "0,9612,Right\n1,16261,Right\n2,22344,Right\n3,79204,Space\n"
_CAP_NAME = "CapacitanceData_C01_01_96_2026-07-09T10_01_52.3332352+02_00"


def test_read_arena_fills_parses_rows(tmp_path: Path) -> None:
    from flypad.io import read_arena_fills

    path = tmp_path / "timestamps_manual_2026-07-09T10_01_52.csv"
    path.write_text(_FILLS_CSV)
    fills = read_arena_fills(path)
    assert list(fills.columns) == ["arena", "board_position", "sample", "key"]
    assert list(fills["arena"]) == [0, 1, 2, 3]
    assert list(fills["board_position"]) == [1, 2, 3, 4]  # 1-based, matches the channel map
    assert list(fills["sample"]) == [9612, 16261, 22344, 79204]
    assert fills["key"].iloc[-1] == "Space"  # key column preserved but not interpreted


def test_read_arena_fills_tolerates_header(tmp_path: Path) -> None:
    from flypad.io import read_arena_fills

    path = tmp_path / "ts.csv"
    path.write_text("arena,sample,key\n" + _FILLS_CSV)
    assert len(read_arena_fills(path)) == 4  # header row dropped


def test_find_arena_fills_matches_on_timestamp(tmp_path: Path) -> None:
    from flypad.io import find_arena_fills

    (tmp_path / "timestamps_manual_2026-07-09T10_01_52.csv").write_text(_FILLS_CSV)
    (tmp_path / "timestamps_manual_2026-07-09T15_43_50.csv").write_text("0,111,Right\n")
    fills = find_arena_fills(tmp_path, _CAP_NAME)
    assert fills is not None
    assert list(fills["sample"]) == [9612, 16261, 22344, 79204]  # picked the 10:01:52 sidecar


def test_find_arena_fills_missing_returns_none(tmp_path: Path) -> None:
    from flypad.io import find_arena_fills

    assert find_arena_fills(tmp_path, _CAP_NAME) is None  # no sidecar at all
    assert find_arena_fills(tmp_path, "CapacitanceData_no_timestamp") is None
