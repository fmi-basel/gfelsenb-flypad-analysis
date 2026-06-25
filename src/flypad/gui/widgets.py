"""Reusable GUI widgets: folder drop target + schema-driven config panel (M8).

``ConfigPanel`` builds an editor for *every* scientific config field straight from
:func:`config_json_schema` (via :func:`iter_config_fields`), grouped by section, so it
stays in sync with the Pydantic model. It can also load an experiment YAML into the
form and save the current state back out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
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
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from flypad.config import config_json_schema, iter_config_fields, load_config
from flypad.config.models import Config

# Config sections surfaced as editor groups (scientific parameters; output/plotting/
# runtime/metadata are driven by the dedicated controls or the config YAML instead).
PARAM_SECTIONS = (
    "hardware",
    "acquisition",
    "quality_control",
    "preprocessing",
    "activity_bouts",
    "sip_detection",
    "feeding_bursts",
    "non_eaters",
    "analysis",
    "stats",
)
_EDITABLE_TYPES = {"integer", "number", "boolean", "string"}


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
        # path -> (editor, schema-default); used to emit only changed-value overrides.
        self._editors: dict[str, tuple[QWidget, Any]] = {}

        self._mode = QComboBox()
        self._mode.addItems(["matlab_compat", "corrected"])
        self._config_edit = QLineEdit()
        self._config_edit.setPlaceholderText("optional experiment YAML")
        self._out_edit = QLineEdit("results")
        self._overrides_edit = QLineEdit()
        self._overrides_edit.setPlaceholderText("extra overrides, e.g. hardware.n_channels=96")
        self._plots = QCheckBox("Render figures")
        self._plots.setChecked(True)

        load_btn = QPushButton("Load…")
        load_btn.clicked.connect(self._load_dialog)
        save_btn = QPushButton("Save…")
        save_btn.clicked.connect(self._save_dialog)
        config_row = QHBoxLayout()
        config_row.addWidget(self._config_edit, stretch=1)
        config_row.addWidget(load_btn)
        config_row.addWidget(save_btn)

        top = QFormLayout()
        top.addRow("Mode", self._mode)
        top.addRow("Config YAML", config_row)
        top.addRow("Output dir", self._out_edit)
        top.addRow("Overrides", self._overrides_edit)
        top.addRow(self._plots)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self._build_param_area(), stretch=1)

    # -- schema-driven form ------------------------------------------------ #
    def _build_param_area(self) -> QWidget:
        sections: dict[str, QFormLayout] = {}
        for path, field in iter_config_fields(self._schema):
            section = path.split(".", 1)[0]
            if section not in PARAM_SECTIONS or field.get("type") not in _EDITABLE_TYPES:
                continue
            editor = self._make_editor(path, field)
            if editor is None:
                continue
            form = sections.setdefault(section, QFormLayout())
            label = (
                path.split(".", 1)[1].replace(".", " ").replace("_", " ") if "." in path else path
            )
            form.addRow(label, editor)

        container = QWidget()
        vbox = QVBoxLayout(container)
        for section in PARAM_SECTIONS:
            if section in sections:
                box = QGroupBox(section.replace("_", " "))
                box.setLayout(sections[section])
                vbox.addWidget(box)
        vbox.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        return scroll

    def _make_editor(self, path: str, field: dict[str, Any]) -> QWidget | None:
        kind = field.get("type")
        default = field.get("default")
        if kind == "boolean":
            check = QCheckBox()
            check.setChecked(bool(default))
            self._editors[path] = (check, bool(default))
            return check
        if kind == "string":
            enum = field.get("enum")
            if enum:
                combo = QComboBox()
                combo.addItems([str(v) for v in enum])
                if default in enum:
                    combo.setCurrentText(str(default))
                self._editors[path] = (combo, default)
                return combo
            line = QLineEdit(str(default) if default is not None else "")
            self._editors[path] = (line, default)
            return line
        if kind == "integer":
            spin = QSpinBox()
            low = field.get("minimum", field.get("exclusiveMinimum", 0)) or 0
            spin.setRange(int(low), 10_000_000)
            spin.setValue(int(default if default is not None else 0))
            self._editors[path] = (spin, int(default if default is not None else 0))
            return spin
        dspin = QDoubleSpinBox()  # number
        dspin.setDecimals(3)
        dspin.setRange(0.0, 10_000_000.0)
        dspin.setSingleStep(0.1)
        dspin.setValue(float(default if default is not None else 0.0))
        self._editors[path] = (dspin, float(default if default is not None else 0.0))
        return dspin

    @staticmethod
    def _value(editor: QWidget) -> Any:
        if isinstance(editor, QCheckBox):
            return editor.isChecked()
        if isinstance(editor, QComboBox):
            return editor.currentText()
        if isinstance(editor, QLineEdit):
            return editor.text()
        return editor.value()  # type: ignore[attr-defined]

    @staticmethod
    def _set_value(editor: QWidget, value: Any) -> None:
        if isinstance(editor, QCheckBox):
            editor.setChecked(bool(value))
        elif isinstance(editor, QComboBox):
            editor.setCurrentText(str(value))
        elif isinstance(editor, QLineEdit):
            editor.setText(str(value))
        else:
            editor.setValue(value)  # type: ignore[attr-defined]

    def _changed_overrides(self) -> list[str]:
        out = []
        for path, (editor, default) in self._editors.items():
            value = self._value(editor)
            if value != default:
                out.append(f"{path}={value}")
        return out

    # -- public API -------------------------------------------------------- #
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

    def load_into_form(self, config: Config) -> None:
        """Populate the editors from a :class:`Config` (and treat them as the baseline)."""
        dumped = config.model_dump(mode="json")
        for path, (editor, _default) in self._editors.items():
            node: Any = dumped
            for part in path.split("."):
                if not isinstance(node, dict) or part not in node:
                    node = None
                    break
                node = node[part]
            if node is not None:
                self._set_value(editor, node)
                self._editors[path] = (editor, self._value(editor))
        self._mode.setCurrentText(config.mode.value)

    def _load_dialog(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(self, "Load config YAML", "", "YAML (*.yaml *.yml)")
        if chosen:
            self.set_config_path(chosen)
            self.load_into_form(load_config(chosen, preset=self._mode.currentText()))

    def _save_dialog(self) -> None:
        chosen, _ = QFileDialog.getSaveFileName(self, "Save config YAML", "", "YAML (*.yaml *.yml)")
        if chosen:
            data = self.build_config().model_dump(mode="json")
            Path(chosen).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
