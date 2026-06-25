"""flypad desktop GUI — a thin Qt view over the pipeline runner (design §4, M8).

Drag-drop a recordings folder, tweak the schema-driven config panel, and run the full
pipeline on a worker thread; progress streams to a log, and the per-condition table and
dashboard appear when it finishes. All science lives in :mod:`flypad.pipeline` /
:mod:`flypad.stats` / :mod:`flypad.plotting` — this module only wires widgets to it.
"""

from __future__ import annotations

import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")  # worker-thread figure saving stays headless/thread-safe
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from qtpy.QtCore import Qt, QThread
from qtpy.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from flypad.gui.widgets import ConfigPanel, DropFolderWidget
from flypad.gui.workers import JobResult, PipelineWorker

_DASHBOARD_METRIC = "n_sips"


class MainWindow(QMainWindow):
    """Top-level window: inputs on the left, log + results on the right."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("flypad")
        self.resize(1100, 720)
        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None

        self.drop = DropFolderWidget()
        self.config_panel = ConfigPanel()
        self.run_button = QPushButton("Run analysis")
        self.run_button.clicked.connect(self._start)
        self.status = QLabel("Ready.")

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self.drop)
        left_layout.addWidget(self.config_panel, stretch=1)
        left_layout.addWidget(self.run_button)
        left_layout.addWidget(self.status)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        self.table = QTableWidget()
        self._canvas_holder = QWidget()
        self._canvas_layout = QVBoxLayout(self._canvas_holder)
        self._canvas_layout.setContentsMargins(0, 0, 0, 0)

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.log)
        right.addWidget(self.table)
        right.addWidget(self._canvas_holder)
        right.setSizes([160, 180, 380])

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([360, 740])
        self.setCentralWidget(splitter)

    # -- run lifecycle ----------------------------------------------------- #
    def _start(self) -> None:
        data_dir = self.drop.path()
        if not data_dir:
            self.status.setText("Choose a recordings folder first.")
            return
        try:
            config = self.config_panel.build_config()
        except Exception as exc:  # invalid config -> show, don't crash
            self.status.setText(f"Config error: {exc}")
            return

        self.run_button.setEnabled(False)
        self.log.clear()
        self.status.setText("Running…")

        self._thread = QThread(self)
        self._worker = PipelineWorker(
            data_dir,
            config,
            self.config_panel.output_dir(),
            make_plots=self.config_panel.make_plots(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progressed.connect(self._append_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.start()

    def _append_log(self, message: str) -> None:
        self.log.appendPlainText(message)

    def _on_finished(self, result: JobResult) -> None:
        self.run_button.setEnabled(True)
        self.status.setText(
            f"Done · {result.n_files} files · {result.n_sips:,} sips · "
            f"{result.n_flies_kept} flies kept · {len(result.written)} files → {result.out_dir}"
        )
        self._fill_table(result)
        self._show_dashboard(result)

    def _on_failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.status.setText("Failed.")
        self._append_log(f"ERROR: {message}")

    # -- results rendering ------------------------------------------------- #
    def _fill_table(self, result: JobResult) -> None:
        pc = result.per_condition
        rows = pc[pc["metric"] == _DASHBOARD_METRIC] if "metric" in pc.columns else pc
        cols = [
            c
            for c in ("condition_label", "n", "mean", "median", "ci_low", "ci_high")
            if c in rows.columns
        ]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(rows))
        for r, (_, row) in enumerate(rows.iterrows()):
            for c, col in enumerate(cols):
                value = row[col]
                text = f"{value:.1f}" if isinstance(value, float) else str(value)
                self.table.setItem(r, c, QTableWidgetItem(text))

    def _show_dashboard(self, result: JobResult) -> None:
        while self._canvas_layout.count():
            item = self._canvas_layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
        if result.per_fly.empty:
            return
        from flypad.plotting import set_theme, standalone_dashboard

        set_theme()
        figure = standalone_dashboard(result.per_fly, _DASHBOARD_METRIC, central="median")
        self._canvas_layout.addWidget(FigureCanvasQTAgg(figure))


def launch(argv: list[str] | None = None) -> int:
    """Create the application, show the main window, and run the event loop."""
    app: Any = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    window = MainWindow()
    window.show()
    return int(app.exec())
