"""Pipeline orchestration: stage registry + experiment runner (design §5)."""

from flypad.pipeline.context import Context
from flypad.pipeline.runner import (
    DetectionResult,
    ExperimentResult,
    absolute_onsets,
    build_tables,
    config_hash,
    detect_experiment,
    read_table,
    render_figures,
    run_experiment,
    write_provenance,
    write_tables,
)
from flypad.pipeline.stages import REGISTRY, register, registered_stages, run

__all__ = [
    "REGISTRY",
    "Context",
    "DetectionResult",
    "ExperimentResult",
    "absolute_onsets",
    "build_tables",
    "config_hash",
    "detect_experiment",
    "read_table",
    "register",
    "registered_stages",
    "render_figures",
    "run",
    "run_experiment",
    "write_provenance",
    "write_tables",
]
