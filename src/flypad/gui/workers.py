"""Background pipeline execution for the GUI (design §4, M8).

The heavy lifting is a plain, Qt-free function (:func:`run_pipeline_job`) so it can be
unit-tested directly; :class:`PipelineWorker` is the thin ``QObject`` that runs it on a
worker thread and re-emits progress/result/error as Qt signals (keeping the UI
responsive — the coupling bug the old port had).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

from flypad.config.models import Config
from flypad.pipeline import build_tables, detect_experiment, render_figures, write_tables

Progress = Callable[[str], None]


@dataclass
class JobResult:
    """Outcome of a full GUI pipeline run."""

    out_dir: Path
    n_files: int
    n_sips: int
    n_flies_kept: int
    written: list[Path]
    per_fly: pd.DataFrame
    per_condition: pd.DataFrame


def run_pipeline_job(
    data_dir: str | Path,
    config: Config,
    out_dir: str | Path,
    *,
    make_plots: bool = True,
    progress: Progress | None = None,
) -> JobResult:
    """Run detect → tables → (figures) → write and summarise the result.

    Pure orchestration over :mod:`flypad.pipeline`; no Qt here.
    """
    detection = detect_experiment(data_dir, config, progress=progress)
    if progress is not None:
        progress("building tables")
    tables = build_tables(detection, config)
    written = write_tables(tables, out_dir, formats=config.output.formats)
    if make_plots and config.plotting.enabled:
        written += render_figures(
            tables["per_fly"], tables["events"], out_dir, config, progress=progress
        )
    kept = len(tables["per_fly"]) - int(tables["per_fly"]["non_eater"].sum())
    return JobResult(
        out_dir=Path(out_dir),
        n_files=len(detection.files),
        n_sips=len(tables["events"]),
        n_flies_kept=kept,
        written=written,
        per_fly=tables["per_fly"],
        per_condition=tables["per_condition"],
    )


class PipelineWorker(QObject):
    """Runs :func:`run_pipeline_job` and re-emits its progress/result as signals."""

    progressed = Signal(str)
    finished = Signal(object)  # JobResult
    failed = Signal(str)

    def __init__(
        self,
        data_dir: str | Path,
        config: Config,
        out_dir: str | Path,
        *,
        make_plots: bool = True,
    ) -> None:
        super().__init__()
        self._data_dir = data_dir
        self._config = config
        self._out_dir = out_dir
        self._make_plots = make_plots

    def run(self) -> None:
        """Slot invoked on the worker thread."""
        try:
            result = run_pipeline_job(
                self._data_dir,
                self._config,
                self._out_dir,
                make_plots=self._make_plots,
                progress=self.progressed.emit,
            )
        except Exception as exc:  # surface any failure to the UI thread
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(result)
