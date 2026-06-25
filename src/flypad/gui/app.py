"""flypad desktop GUI — a thin Qt view over the pipeline runner (design §4, M8).

Drag-drop a recordings folder, tweak the schema-driven config panel, pick a metric, and
run the full pipeline on a worker thread; progress streams to a bar + log, and the
per-condition table and an interactive dashboard appear when it finishes. All science
lives in :mod:`flypad.pipeline` / :mod:`flypad.stats` / :mod:`flypad.plotting`.
"""

from __future__ import annotations

import re
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")  # worker-thread figure saving stays headless/thread-safe
from matplotlib.backends.backend_qtagg import (  # type: ignore[attr-defined]
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from qtpy.QtCore import QSettings, Qt, QThread
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from flypad.gui.widgets import ConfigPanel, DropFolderWidget
from flypad.gui.workers import JobResult, PipelineWorker
from flypad.stats import METRIC_COLUMNS

_STEP_RE = re.compile(r"\[(\d+)/(\d+)\]")


class MainWindow(QMainWindow):
    """Top-level window: inputs on the left, log + results on the right."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("flypad")
        self.resize(1180, 760)
        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None
        self._result: JobResult | None = None
        self._settings = QSettings("flypad", "flypad")

        self.drop = DropFolderWidget()
        self.config_panel = ConfigPanel()
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(list(METRIC_COLUMNS))
        self.metric_combo.currentTextChanged.connect(self._refresh_views)
        self.run_button = QPushButton("Run analysis")
        self.run_button.clicked.connect(self._start)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.status = QLabel("Ready.")

        metric_row = QHBoxLayout()
        metric_row.addWidget(QLabel("Metric"))
        metric_row.addWidget(self.metric_combo, stretch=1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self.drop)
        left_layout.addWidget(self.config_panel, stretch=1)
        left_layout.addLayout(metric_row)
        left_layout.addWidget(self.run_button)
        left_layout.addWidget(self.progress)
        left_layout.addWidget(self.status)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.table = QTableWidget()
        self._canvas_holder = QWidget()
        self._canvas_layout = QVBoxLayout(self._canvas_holder)
        self._canvas_layout.setContentsMargins(0, 0, 0, 0)

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.log)
        right.addWidget(self.table)
        right.addWidget(self._canvas_holder)
        right.setSizes([150, 180, 420])

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([400, 780])
        self.setCentralWidget(splitter)
        self._restore_settings()

    # -- settings ---------------------------------------------------------- #
    def _setting(self, key: str) -> str:
        return str(self._settings.value(key, "", type=str) or "")

    def _restore_settings(self) -> None:
        if data_dir := self._setting("data_dir"):
            self.drop.set_path(data_dir)
        if config_path := self._setting("config_path"):
            self.config_panel.set_config_path(config_path)
        if out := self._setting("output_dir"):
            self.config_panel._out_edit.setText(out)
        metric = self._setting("metric")
        if metric in METRIC_COLUMNS:
            self.metric_combo.setCurrentText(metric)

    def _save_settings(self) -> None:
        self._settings.setValue("data_dir", self.drop.path())
        self._settings.setValue("config_path", self.config_panel._config_edit.text())
        self._settings.setValue("output_dir", self.config_panel.output_dir())
        self._settings.setValue("metric", self.metric_combo.currentText())

    def closeEvent(self, event: Any) -> None:  # Qt camelCase override
        self._save_settings()
        super().closeEvent(event)

    # -- run lifecycle ----------------------------------------------------- #
    def _start(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return  # already running — no double-run
        data_dir = self.drop.path()
        if not data_dir:
            self.status.setText("Choose a recordings folder first.")
            return
        try:
            config = self.config_panel.build_config()
        except Exception as exc:  # invalid config -> dialog, don't crash
            QMessageBox.critical(self, "Config error", str(exc))
            return

        self._save_settings()
        self.run_button.setEnabled(False)
        self.log.clear()
        self.progress.setRange(0, 0)  # busy until the first [i/n]
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
        match = _STEP_RE.search(message)
        if match:
            done, total = int(match.group(1)), int(match.group(2))
            self.progress.setRange(0, total)
            self.progress.setValue(done)

    def _on_finished(self, result: JobResult) -> None:
        self._result = result
        self.run_button.setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.status.setText(
            f"Done · {result.n_files} files · {result.n_sips:,} sips · "
            f"{result.n_flies_kept} flies kept · {len(result.written)} files → {result.out_dir}"
        )
        self._refresh_views()

    def _on_failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.status.setText("Failed.")
        self._append_log(f"ERROR: {message}")
        QMessageBox.critical(self, "Run failed", message)

    # -- results rendering ------------------------------------------------- #
    def _refresh_views(self) -> None:
        if self._result is not None:
            self._fill_table(self._result)
            self._show_dashboard(self._result)

    def _fill_table(self, result: JobResult) -> None:
        metric = self.metric_combo.currentText()
        pc = result.per_condition
        rows = pc[pc["metric"] == metric] if "metric" in pc.columns else pc
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
        figure = standalone_dashboard(
            result.per_fly, self.metric_combo.currentText(), central="median"
        )
        canvas = FigureCanvasQTAgg(figure)
        self._canvas_layout.addWidget(NavigationToolbar2QT(canvas, self))
        self._canvas_layout.addWidget(canvas)


def launch(argv: list[str] | None = None) -> int:
    """Create the application, show the main window, and run the event loop."""
    app: Any = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    window = MainWindow()
    window.show()
    return int(app.exec())
