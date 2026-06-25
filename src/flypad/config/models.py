"""Pydantic configuration models — the full schema (design §6).

Every analysis parameter is a typed, validated field here; nothing is hard-coded
elsewhere. Defaults are a sensible baseline — the ``matlab_compat`` / ``corrected``
presets (config/presets/*.yaml) set the mode-specific values, and an experiment
YAML + CLI overrides layer on top (see ``flypad.config.loader``).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_Strict = ConfigDict(extra="forbid")


class Mode(StrEnum):
    """Analysis fidelity mode (design §0 / §8)."""

    matlab_compat = "matlab_compat"
    corrected = "corrected"


class EdgeHandling(StrEnum):
    """How convolution edges are treated in baseline subtraction."""

    crop = "crop"  # MATLAB: shorten the trace
    zero = "zero"  # old Python port: keep length, zero the edges
    reflect = "reflect"  # corrected: keep full frame, reflect at edges


class ThresholdMethod(StrEnum):
    """Derivative-threshold strategy for sip-edge detection."""

    ibis_noise = "ibis_noise"  # MATLAB v2.2: max/min(diff(IBIS)) per channel
    adaptive_mad = "adaptive_mad"  # robust windowed MAD (4/0.675)
    fixed_height = "fixed_height"  # constant height (old Python port)


class Pairing(StrEnum):
    """Sip onset->offset pairing strategy."""

    greedy = "greedy"  # MATLAB while-loop forward level-crossing
    adjacent = "adjacent"  # old Python port peak-to-peak


class Hardware(BaseModel):
    """Rig / board configuration."""

    model_config = _Strict
    configuration: str | None = None  # informational, e.g. "48 flies"
    n_channels: int = Field(64, gt=0)
    sampling_rate_hz: int = Field(100, gt=0)


class Acquisition(BaseModel):
    """Acquisition parameters."""

    model_config = _Strict
    duration_samples: int = Field(360_000, gt=0)
    dtype: Literal["uint16"] = "uint16"


class QualityControl(BaseModel):
    """Channel-validity and spill/unconnected QC."""

    model_config = _Strict
    unconnected_zero_fraction: float = Field(0.5, ge=0.0, le=1.0)
    spill_saturation_value: int = 4095
    remove_spill_quality: bool = True
    spill_quality_threshold: float = Field(0.5, ge=0.0, le=1.0)


class Preprocessing(BaseModel):
    """Median filter + baseline subtraction."""

    model_config = _Strict
    median_kernel: int = Field(6, gt=0)
    baseline_span: int = Field(50, gt=0)
    edge_handling: EdgeHandling = EdgeHandling.crop


class ActivityBouts(BaseModel):
    """RMS activity-bout detection."""

    model_config = _Strict
    rms_window: int = Field(50, gt=0)
    rms_threshold: float = Field(10.0, ge=0.0)


class SipThreshold(BaseModel):
    """Derivative-threshold parameters."""

    model_config = _Strict
    method: ThresholdMethod = ThresholdMethod.ibis_noise
    fixed_height: float = 50.0
    mad_scale: float = 5.926  # 4 / 0.675
    window: int = Field(300, gt=0)
    offset: float = 50.0  # minimum |threshold| floor for adaptive_mad


class SipDetection(BaseModel):
    """Sip detection: thresholding + pairing + duration/amplitude gates."""

    model_config = _Strict
    threshold: SipThreshold = Field(default_factory=SipThreshold)
    pairing: Pairing = Pairing.greedy
    peak_min_distance: int = Field(7, gt=0)
    equality_factor: float = Field(0.5, gt=0.0)
    min_duration_samples: int = Field(4, gt=0)
    max_duration_samples: int = Field(100, gt=0)


class FeedingBursts(BaseModel):
    """Grouping sips into feeding bursts."""

    model_config = _Strict
    ifi_criterion: Literal["mode_x2"] = "mode_x2"
    min_sips: int = Field(3, gt=0)


class NonEaters(BaseModel):
    """Non-eater fly removal."""

    model_config = _Strict
    remove_substrate: bool = False
    remove_global: bool = True
    threshold_bouts: int = Field(2, ge=0)


class Analysis(BaseModel):
    """Time-course / cumulative analysis parameters."""

    model_config = _Strict
    time_window_samples: int = Field(60_000, gt=0)
    cumulative_bins: int = Field(1000, gt=0)


class Metadata(BaseModel):
    """Experiment metadata / sidecar parsing (design §17)."""

    model_config = _Strict
    log_file: str | None = None
    channels_per_board_position: int = Field(8, gt=0)
    substrates: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)


OutputFormat = Literal["parquet", "hdf5", "csv"]
_DEFAULT_FORMATS: list[OutputFormat] = ["parquet", "hdf5", "csv"]


class Output(BaseModel):
    """Output formats and destinations."""

    model_config = _Strict
    dir: str = "./results"
    formats: list[OutputFormat] = Field(default_factory=lambda: list(_DEFAULT_FORMATS))
    write_mat: bool = False
    save_intermediate_signals: bool = False


class Plotting(BaseModel):
    """Figure rendering options."""

    model_config = _Strict
    enabled: bool = True
    save_eps: bool = True
    dpi: int = Field(300, gt=0)


class Runtime(BaseModel):
    """Execution / logging options."""

    model_config = _Strict
    n_jobs: int = -1
    log_level: str = "INFO"


class Config(BaseModel):
    """Top-level flypad configuration (design §6)."""

    model_config = _Strict

    mode: Mode = Mode.corrected
    hardware: Hardware = Field(default_factory=Hardware)
    acquisition: Acquisition = Field(default_factory=Acquisition)
    quality_control: QualityControl = Field(default_factory=QualityControl)
    preprocessing: Preprocessing = Field(default_factory=Preprocessing)
    activity_bouts: ActivityBouts = Field(default_factory=ActivityBouts)
    sip_detection: SipDetection = Field(default_factory=SipDetection)
    feeding_bursts: FeedingBursts = Field(default_factory=FeedingBursts)
    non_eaters: NonEaters = Field(default_factory=NonEaters)
    analysis: Analysis = Field(default_factory=Analysis)
    metadata: Metadata = Field(default_factory=Metadata)
    output: Output = Field(default_factory=Output)
    plotting: Plotting = Field(default_factory=Plotting)
    runtime: Runtime = Field(default_factory=Runtime)
    pipeline_order: list[str] = Field(default_factory=list)
