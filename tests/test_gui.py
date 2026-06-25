"""M8: GUI — headless tests for the worker, widgets, and main window.

Runs with the offscreen Qt platform; skipped entirely if the ``gui`` extra
(PySide6 + qtpy) is not installed.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("qtpy")
pytest.importorskip("PySide6")

from qtpy.QtWidgets import QApplication

from flypad.config.models import Config
from flypad.gui import (
    ConfigPanel,
    DropFolderWidget,
    JobResult,
    MainWindow,
    PipelineWorker,
    run_pipeline_job,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _cleanup(qapp: QApplication):
    """Close windows / figures after each test so interpreter teardown is clean."""
    import matplotlib.pyplot as plt

    yield
    plt.close("all")
    for widget in qapp.topLevelWidgets():
        widget.close()
    qapp.processEvents()


def _make_dataset(tmp_path: Path, n_channels: int = 8, n_time: int = 3000) -> tuple[Path, Path]:
    arr = np.full((n_time, n_channels), 1000, dtype=np.uint16)
    for t0 in range(200, n_time - 200, 300):
        arr[t0 : t0 + 5, 0] = 1600
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    arr.tofile(data_dir / "CapacitanceData_C01_01_08_2024-01-01T00_00_00.0000000+00_00")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "mode: matlab_compat\n"
        "hardware:\n  n_channels: 8\n  sampling_rate_hz: 100\n"
        "acquisition:\n  duration_samples: 3000\n"
        "output:\n  formats: [csv]\n"
    )
    return data_dir, cfg


# --------------------------------------------------------------------------- #
# pure job + worker
# --------------------------------------------------------------------------- #
def test_run_pipeline_job_synthetic(tmp_path: Path) -> None:
    data_dir, cfg_path = _make_dataset(tmp_path)
    from flypad.config import load_config

    result = run_pipeline_job(
        data_dir, load_config(cfg_path, preset="matlab_compat"), tmp_path / "out", make_plots=False
    )
    assert isinstance(result, JobResult)
    assert result.n_files == 1
    assert len(result.per_fly) == 8
    assert (tmp_path / "out" / "per_fly.csv").exists()


def test_pipeline_worker_emits_finished(qapp: QApplication, tmp_path: Path) -> None:
    data_dir, cfg_path = _make_dataset(tmp_path)
    from flypad.config import load_config

    worker = PipelineWorker(
        data_dir, load_config(cfg_path, preset="matlab_compat"), tmp_path / "out", make_plots=False
    )
    captured: list[JobResult] = []
    worker.finished.connect(captured.append)
    worker.run()  # synchronous (no thread) — deterministic
    assert len(captured) == 1
    assert captured[0].n_files == 1


def test_pipeline_worker_emits_failed_on_bad_dir(qapp: QApplication, tmp_path: Path) -> None:
    worker = PipelineWorker(tmp_path / "missing", Config(), tmp_path / "out", make_plots=False)
    errors: list[str] = []
    worker.failed.connect(errors.append)
    worker.run()
    assert errors  # a failure was reported
    assert "missing" in errors[0]  # names the offending directory


# --------------------------------------------------------------------------- #
# widgets
# --------------------------------------------------------------------------- #
def test_drop_folder_widget(qapp: QApplication) -> None:
    widget = DropFolderWidget()
    seen: list[str] = []
    widget.pathChanged.connect(seen.append)
    widget.set_path("/some/data")
    assert widget.path() == "/some/data"
    assert seen == ["/some/data"]


def test_config_panel_builds_config_and_overrides(qapp: QApplication) -> None:
    panel = ConfigPanel()
    cfg = panel.build_config()
    assert isinstance(cfg, Config)
    assert cfg.mode.value == "matlab_compat"  # default selection

    # change a schema-driven editor -> reflected in the built config
    editor, _default = panel._editors["feeding_bursts.min_sips"]
    editor.setValue(7)
    assert "feeding_bursts.min_sips=7" in panel.overrides()
    assert panel.build_config().feeding_bursts.min_sips == 7


def test_config_panel_unchanged_fields_do_not_override(qapp: QApplication) -> None:
    panel = ConfigPanel()
    assert panel._changed_overrides() == []  # nothing touched


# --------------------------------------------------------------------------- #
# main window
# --------------------------------------------------------------------------- #
def test_main_window_constructs(qapp: QApplication) -> None:
    window = MainWindow()
    assert window.drop.path() == ""
    assert window.run_button.isEnabled()
    assert window.windowTitle() == "flypad"


def test_main_window_renders_results(qapp: QApplication, tmp_path: Path) -> None:
    data_dir, cfg_path = _make_dataset(tmp_path)
    from flypad.config import load_config

    result = run_pipeline_job(
        data_dir, load_config(cfg_path, preset="matlab_compat"), tmp_path / "out", make_plots=False
    )
    window = MainWindow()
    window._on_finished(result)
    assert "Done" in window.status.text()
    assert window.table.rowCount() >= 1  # per-condition table filled
    assert window.run_button.isEnabled()
    assert window._canvas_layout.count() == 1  # embedded dashboard canvas


def test_main_window_threaded_run_writes_outputs(qapp: QApplication, tmp_path: Path) -> None:
    data_dir, cfg_path = _make_dataset(tmp_path)
    out = tmp_path / "gui_out"
    window = MainWindow()
    window.drop.set_path(str(data_dir))
    window.config_panel.set_config_path(str(cfg_path))
    window.config_panel._out_edit.setText(str(out))
    window.config_panel._plots.setChecked(False)
    window._start()
    thread = window._thread
    assert thread is not None
    # Pump the event loop until the worker finishes (app.exec() would do this in the
    # real GUI). A blocking wait() would deadlock the queued finished->quit signal.
    for _ in range(600):
        QApplication.processEvents()
        if not thread.isRunning():
            break
        thread.wait(50)
    QApplication.processEvents()
    assert not thread.isRunning()
    assert (out / "per_fly.csv").exists()
    assert window.run_button.isEnabled()  # re-enabled after finish
