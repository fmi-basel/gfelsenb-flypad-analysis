"""Pydantic configuration models (skeleton — full schema in M1; design §6).

Only enough is modelled here to drive the registry/runner and the M0 smoke
tests. Each milestone expands this with its parameters (all validated, layered
preset → YAML → CLI override).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Mode(StrEnum):
    """Analysis fidelity mode (design §0 / §8)."""

    matlab_compat = "matlab_compat"
    corrected = "corrected"


class Hardware(BaseModel):
    """Rig / board configuration."""

    n_channels: int = 96
    sampling_rate_hz: int = 100


class Acquisition(BaseModel):
    """Acquisition parameters."""

    duration_samples: int = 360_000


class Config(BaseModel):
    """Top-level flypad configuration (skeleton).

    See design §6 for the full intended schema (quality_control, preprocessing,
    activity_bouts, sip_detection, feeding_bursts, non_eaters, analysis,
    metadata, output, plotting, runtime).
    """

    model_config = {"extra": "forbid"}

    mode: Mode = Mode.corrected
    hardware: Hardware = Field(default_factory=Hardware)
    acquisition: Acquisition = Field(default_factory=Acquisition)
    pipeline_order: list[str] = Field(default_factory=list)
