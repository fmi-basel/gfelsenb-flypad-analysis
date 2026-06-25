"""Run the end-to-end validation: detect a dataset and compare to a MATLAB .mat."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flypad.config.models import Config
from flypad.datamodel import load_recording
from flypad.detect.run import detect_recording
from flypad.io.discovery import find_capacitance_files
from flypad.io.matlab import read_events_mat
from flypad.validate.compare import FileComparison, compare_file


@dataclass
class DatasetValidation:
    files: list[FileComparison]
    tolerance: int

    @property
    def matched(self) -> int:
        return sum(f.matched for f in self.files)

    @property
    def n_python(self) -> int:
        return sum(f.n_python for f in self.files)

    @property
    def n_matlab(self) -> int:
        return sum(f.n_matlab for f in self.files)

    @property
    def precision(self) -> float:
        return self.matched / self.n_python if self.n_python else float("nan")

    @property
    def recall(self) -> float:
        return self.matched / self.n_matlab if self.n_matlab else float("nan")


def validate_dataset(
    data_dir: str | Path,
    mat_path: str | Path,
    config: Config,
    tolerance: int = 2,
) -> DatasetValidation:
    """Detect every recording in ``data_dir`` and compare onsets to ``mat_path``."""
    mat = read_events_mat(mat_path)
    files = find_capacitance_files(data_dir)
    comparisons: list[FileComparison] = []
    for fi, path in enumerate(files):
        if fi >= mat.n_files:
            break
        rec = load_recording(path, config)
        result = detect_recording(rec.capacitance, config)
        py_by_ch = [np.asarray(s.onsets + result.crop_offset, dtype=np.int64) for s in result.sips]
        mat_by_ch = mat.ons[fi][: len(py_by_ch)]
        comparisons.append(compare_file(py_by_ch, mat_by_ch, tolerance=tolerance))
    return DatasetValidation(files=comparisons, tolerance=tolerance)
