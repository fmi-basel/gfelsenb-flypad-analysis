"""Pipeline orchestration: stage registry + experiment runner (design §5)."""

from flypad.pipeline.context import Context
from flypad.pipeline.runner import (
    DetectionResult,
    build_tables,
    detect_experiment,
    read_table,
    render_figures,
    write_tables,
)
from flypad.pipeline.stages import REGISTRY, register, registered_stages, run

__all__ = [
    "REGISTRY",
    "Context",
    "DetectionResult",
    "build_tables",
    "detect_experiment",
    "read_table",
    "register",
    "registered_stages",
    "render_figures",
    "run",
    "write_tables",
]
