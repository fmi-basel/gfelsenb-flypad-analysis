"""M3: validation harness — matching, offset estimation, and .mat reading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flypad.validate.compare import compare_file, estimate_offset

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "sample" / "20240215"
SAMPLE_MAT = SAMPLE_DIR / "20240215.mat"
SAMPLE_RAW = SAMPLE_DIR / "CapacitanceData_C01_01_96_2024-02-15T11_28_04.8078208+01_00"


def test_estimate_offset_recovers_constant_shift() -> None:
    mat = [np.array([100, 250, 900], dtype=np.int64), np.array([400, 1200], dtype=np.int64)]
    py = [m - 53 for m in mat]  # python frame shifted back by 53
    assert estimate_offset(py, mat) == 53


def test_compare_file_perfect_match() -> None:
    mat = [np.array([100, 250, 900], dtype=np.int64)]
    py = [np.array([99, 251, 901], dtype=np.int64)]  # within tol after offset 0
    fc = compare_file(py, mat, tolerance=2, offset=0)
    assert fc.matched == 3
    assert fc.precision == 1.0 and fc.recall == 1.0


def test_compare_file_partial() -> None:
    mat = [np.array([100, 250, 900], dtype=np.int64)]
    py = [np.array([100, 250], dtype=np.int64)]  # missed one
    fc = compare_file(py, mat, tolerance=2, offset=0)
    assert fc.matched == 2
    assert fc.recall == pytest.approx(2 / 3)
    assert fc.precision == 1.0


# ---- needs the (tracked) ground-truth .mat ----
@pytest.mark.skipif(not SAMPLE_MAT.exists(), reason="sample .mat not present")
def test_read_events_mat_shapes() -> None:
    from flypad.io import read_events_mat

    ev = read_events_mat(SAMPLE_MAT)
    assert ev.n_files == 2
    assert ev.n_channels == 96
    # channel 0 of file 0 had ~463 sips in the inspection
    assert ev.ons[0][0].size > 100
    assert ev.duration_samples == 425391


@pytest.mark.skipif(not SAMPLE_RAW.exists(), reason="raw sample binaries not present (git-ignored)")
def test_compat_parity_against_matlab_source() -> None:
    """matlab_compat must reproduce the MATLAB source events at the event level."""
    from flypad.config import load_config
    from flypad.validate import validate_dataset

    cfg = load_config(REPO_ROOT / "configs" / "example_experiment.yaml", preset="matlab_compat")
    result = validate_dataset(SAMPLE_DIR, SAMPLE_MAT, cfg, tolerance=2)
    # counts within ~2%; onsets >85% matched within +-2 samples; consistent frame offset
    assert abs(result.n_python - result.n_matlab) / result.n_matlab < 0.02
    assert result.recall > 0.85
    assert result.precision > 0.85
    assert len({f.offset for f in result.files}) == 1
