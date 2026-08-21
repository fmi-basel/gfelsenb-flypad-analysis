"""Background pipeline execution for the GUI (design §4, M8).

The heavy lifting is :func:`flypad.pipeline.run_experiment`, shared with ``flypad run``
so the two entry points cannot drift apart; :func:`run_pipeline_job` is the Qt-free
delegation to it and :class:`PipelineWorker` the thin ``QObject`` that runs it on a
worker thread, re-emitting progress/result/error as Qt signals (keeping the UI
responsive — the coupling bug the old port had).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

from flypad.config.models import Config
from flypad.pipeline import ExperimentResult, run_experiment

Progress = Callable[[str], None]

#: Outcome of a full GUI pipeline run — the same result the CLI gets.
JobResult = ExperimentResult


def run_pipeline_job(
    data_dir: str | Path,
    config: Config,
    out_dir: str | Path,
    *,
    make_plots: bool = True,
    progress: Progress | None = None,
) -> JobResult:
    """Run the pipeline for the GUI and summarise the result.

    A thin delegation to :func:`flypad.pipeline.run_experiment` — the orchestration
    itself is shared with ``flypad run`` so the two cannot drift apart. It is tagged
    ``command="gui"`` in ``run_info.json``; everything else about the results directory
    is identical.
    """
    return run_experiment(
        data_dir,
        config,
        out_dir,
        make_plots=make_plots,
        command="gui",
        progress=progress,
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
