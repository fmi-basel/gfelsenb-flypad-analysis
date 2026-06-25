"""flypad desktop GUI (PySide6 + qtpy, design §4, M8).

Importing this package pulls in Qt (the optional ``gui`` extra). The CLI imports
:func:`launch` lazily so ``flypad`` works without Qt installed.
"""

from flypad.gui.app import MainWindow, launch
from flypad.gui.widgets import ConfigPanel, DropFolderWidget
from flypad.gui.workers import JobResult, PipelineWorker, run_pipeline_job

__all__ = [
    "ConfigPanel",
    "DropFolderWidget",
    "JobResult",
    "MainWindow",
    "PipelineWorker",
    "launch",
    "run_pipeline_job",
]
