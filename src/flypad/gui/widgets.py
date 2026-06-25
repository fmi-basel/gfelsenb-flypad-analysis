"""Reusable GUI widgets: folder drop target + schema-driven config panel (M8).

The config panel builds its editors straight from :func:`config_json_schema` via
:func:`resolve_field`, so it stays in sync with the Pydantic model — change a field's
type/bounds and the form follows.
"""

from __future__ import annotations

from typing import Any

from qtpy.QtCore import Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from flypad.config import config_json_schema, load_config, resolve_field
from flypad.config.models import Config

# Curated, frequently-tuned parameters surfaced as editors (dotted path -> label).
FORM_FIELDS: tuple[tuple[str, str], ...] = (
    ("sip_detection.equality_factor", "Sip equality factor"),
    ("activity_bouts.rms_threshold", "RMS bout threshold"),
    ("feeding_bursts.min_sips", "Min sips / burst"),
    ("non_eaters.threshold_bouts", "Non-eater bout threshold"),
)


class DropFolderWidget(QWidget):
    """A drop target (and Browse button) for choosing a data folder."""

    pathChanged = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._path = ""
        self._label = QLabel("Drop a recordings folder here, or Browse…")
        self._label.setStyleSheet("QLabel { border: 1px dashed #aaa; padding: 18px; }")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        layout = QHBoxLayout(self)
        layout.addWidget(self._label, stretch=1)
        layout.addWidget(browse)

    def path(self) -> str:
        return self._path

    def set_path(self, path: str) -> None:
        self._path = path
        self._label.setText(path or "Drop a recordings folder here, or Browse…")
        self.pathChanged.emit(path)

    def _browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Select recordings folder")
        if chosen:
            self.set_path(chosen)

    def dragEnterEvent(self, event: Any) -> None:  # Qt camelCase override
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: Any) -> None:  # Qt camelCase override
        urls = event.mimeData().urls()
        if urls:
            self.set_path(urls[0].toLocalFile())
            event.acceptProposedAction()


class ConfigPanel(QWidget):
    """Mode / config-file / output controls plus schema-driven parameter editors."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._schema = config_json_schema()
        self._editors: dict[str, tuple[QWidget, float | int]] = {}

        self._mode = QComboBox()
        self._mode.addItems(["matlab_compat", "corrected"])
        self._config_edit = QLineEdit()
        self._config_edit.setPlaceholderText("optional experiment YAML")
        config_browse = QPushButton("…")
        config_browse.setFixedWidth(32)
        config_browse.clicked.connect(self._browse_config)
        config_row = QHBoxLayout()
        config_row.addWidget(self._config_edit, stretch=1)
        config_row.addWidget(config_browse)

        self._out_edit = QLineEdit("results")
        self._overrides_edit = QLineEdit()
        self._overrides_edit.setPlaceholderText("extra overrides, e.g. hardware.n_channels=96")
        self._plots = QCheckBox("Render figures")
        self._plots.setChecked(True)

        form = QFormLayout()
        form.addRow("Mode", self._mode)
        form.addRow("Config YAML", config_row)
        form.addRow("Output dir", self._out_edit)
        form.addRow("Overrides", self._overrides_edit)
        form.addRow(self._plots)

        params = QFormLayout()
        for path, label in FORM_FIELDS:
            field = resolve_field(self._schema, path) or {}
            editor = self._make_editor(path, field)
            params.addRow(label, editor)

        params_box = QGroupBox("Parameters")
        params_box.setLayout(params)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(params_box)

    def _make_editor(self, path: str, field: dict[str, Any]) -> QWidget:
        default = field.get("default", 0)
        if field.get("type") == "integer":
            spin = QSpinBox()
            low = field.get("minimum", field.get("exclusiveMinimum", 0)) or 0
            spin.setRange(int(low), 1_000_000)
            spin.setValue(int(default))
            self._editors[path] = (spin, int(default))
            return spin
        dspin = QDoubleSpinBox()
        dspin.setDecimals(3)
        dspin.setRange(0.0, 1_000_000.0)
        dspin.setSingleStep(0.1)
        dspin.setValue(float(default))
        self._editors[path] = (dspin, float(default))
        return dspin

    def _changed_overrides(self) -> list[str]:
        out = []
        for path, (editor, default) in self._editors.items():
            value = editor.value()  # type: ignore[attr-defined]
            if value != default:
                out.append(f"{path}={value}")
        return out

    def overrides(self) -> list[str]:
        """Schema-editor overrides (only changed ones) plus the free-text overrides."""
        return self._changed_overrides() + self._overrides_edit.text().split()

    def output_dir(self) -> str:
        return self._out_edit.text().strip() or "results"

    def make_plots(self) -> bool:
        return self._plots.isChecked()

    def set_config_path(self, path: str) -> None:
        self._config_edit.setText(path)

    def build_config(self) -> Config:
        """Resolve the panel state into a validated :class:`Config`."""
        config_path = self._config_edit.text().strip() or None
        return load_config(config_path, preset=self._mode.currentText(), overrides=self.overrides())

    def _browse_config(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Select config YAML", "", "YAML (*.yaml *.yml)"
        )
        if chosen:
            self._config_edit.setText(chosen)
